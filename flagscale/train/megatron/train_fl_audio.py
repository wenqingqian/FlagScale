"""Pretrain FL Audio (TeleFLM + DepthGPT)."""

import logging
import os
import torch

from copy import deepcopy
from functools import partial
from typing import Optional
from megatron.training.utils import is_first_or_last_pipeline_stage
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_src_rank,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import StragglerDetector
from megatron.energon import (
    LimitDataset,
    RepeatDataset,
    WorkerConfig,
    get_loader,
    get_savable_loader,
    get_train_dataset,
    get_val_datasets,
)
from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.checkpointing import get_checkpoint_name
from megatron.training.training import pretrain

from flagscale.models.megatron.fl_audio import (
    FLAudioBackend,
    print_error_handler,
    TeleFLMForCausalLM,
    get_teleflm_config,
    get_depth_gpt_config,
    get_teleflm_layer_spec,
    get_depth_gpt_layer_spec,
    FLAudioTaskEncoder as TaskEncoder,
)

stimer = StragglerDetector()
SPIKY_LOSS_FACTOR = 10


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def model_provider(pre_process=True, post_process=True):
    """Build the TeleFLMForCausalLM model."""
    args = get_args()
    print_rank_0("building FL Audio model ...")

    base_config = core_transformer_config_from_args(args)
    teleflm_config = get_teleflm_config(args, base_config)
    depth_config = get_depth_gpt_config(args, base_config)

    backend = FLAudioBackend(args)
    teleflm_layer_spec = get_teleflm_layer_spec(backend)
    depth_layer_spec = get_depth_gpt_layer_spec(
        backend,
        use_cmlp=args.depth_use_channel_mlp
    )

    model = TeleFLMForCausalLM(
        teleflm_config=teleflm_config,
        teleflm_layer_spec=teleflm_layer_spec,
        depth_config=depth_config,
        depth_layer_spec=depth_layer_spec,
        pre_process=pre_process,
        post_process=post_process,
        backend=backend
    )

    print_rank_0(model)

    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def is_dataloader_rank(vp_stage=None, check_pp_stage_only=False):
    """Only TP rank 0 runs the dataloader."""
    _is_dataloader_rank = is_first_or_last_pipeline_stage(vp_stage)
    if check_pp_stage_only: 
        return _is_dataloader_rank
    _is_dataloader_rank = _is_dataloader_rank and get_tensor_model_parallel_rank() == 0
    return _is_dataloader_rank

def _broadcast_tensor(key, data, dtype):
    """Broadcast a single tensor from TP rank 0 to all TP ranks."""
    if get_tensor_model_parallel_world_size() == 1:
        if data is not None:
            return data[key].cuda(non_blocking=True)
        return None

    if get_tensor_model_parallel_rank() == 0:
        tensor = data[key].cuda(non_blocking=True)
        shape = torch.tensor(tensor.shape, device='cuda')
    else:
        shape = torch.empty(0, dtype=torch.long, device='cuda')
        tensor = None

    # Broadcast shape first
    if get_tensor_model_parallel_rank() == 0:
        shape_len = torch.tensor(len(tensor.shape), device='cuda')
    else:
        shape_len = torch.tensor(0, device='cuda')
    torch.distributed.broadcast(
        shape_len,
        src=get_tensor_model_parallel_src_rank(),
        group=parallel_state.get_tensor_model_parallel_group(),
    )
    if get_tensor_model_parallel_rank() != 0:
        shape = torch.empty(shape_len.item(), dtype=torch.long, device='cuda')
    torch.distributed.broadcast(
        shape,
        src=get_tensor_model_parallel_src_rank(),
        group=parallel_state.get_tensor_model_parallel_group(),
    )
    if get_tensor_model_parallel_rank() != 0:
        tensor = torch.empty(*shape.tolist(), dtype=dtype, device='cuda')
    torch.distributed.broadcast(
        tensor,
        src=get_tensor_model_parallel_src_rank(),
        group=parallel_state.get_tensor_model_parallel_group(),
    )
    return tensor


def get_batch(data_iterator):
    """Generate a batch.

    Dataset provides audio_ids: [b, s, c].
    position_ids and attention_mask are generated here for training.
    """
    args = get_args()

    if not is_dataloader_rank(check_pp_stage_only=True):
        return None, None, None

    if data_iterator is not None and get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
    else:
        data = None

    audio_ids = _broadcast_tensor("audio_ids", data, torch.int64)
    # Truncate to configured seq_length
    audio_ids = audio_ids[:, :args.seq_length, :]

    # For training: simple sequential position_ids [0, 1, ..., s-1]
    # attention_mask = None lets Megatron generate causal mask internally
    seq_len = audio_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=audio_ids.device)
    position_ids = position_ids.unsqueeze(0).expand(audio_ids.shape[0], -1)

    attention_mask = None

    return audio_ids, attention_mask, position_ids


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def loss_func(output_tensor, model=None):
    """Loss function.

    The model computes channel-wise loss internally in DepthGPT_Postprocessor.
    output_tensor is (loss, channel_losses).
    """
    args = get_args()
    loss, channel_losses = output_tensor

    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,
            fatal=True,
        )

    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,
            fatal=False,
        )

    reporting_loss = loss.clone().detach().view(1)
    report = {'lm loss': reporting_loss}
    for i, cl in enumerate(channel_losses):
        report[f'channel_{i}_loss'] = cl.clone().detach().view(1)

    num_tokens = torch.tensor(1, dtype=torch.int, device=loss.device)
    return (loss, num_tokens, report)


# ---------------------------------------------------------------------------
# Forward step
# ---------------------------------------------------------------------------

def forward_step(data_iterator, model):
    """Forward training step."""
    timers = get_timers()

    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        audio_ids, attention_mask, position_ids = get_batch(data_iterator)
    timers('batch-generator').stop()

    with stimer:
        output_tensor = model(
            audio_ids=audio_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    return output_tensor, partial(loss_func, model=model)


# ---------------------------------------------------------------------------
# Energon dataloaders
# ---------------------------------------------------------------------------

def datasets_provider(worker_config=None):
    """Create train, validation and test datasets."""
    args = get_args()
    dname = args.data_path[0] if isinstance(args.data_path, list) else args.data_path

    train_dataset = get_train_dataset(
        dname,
        handler       = print_error_handler,
        split_part    = "train",
        batch_size    = args.micro_batch_size,
        task_encoder  = TaskEncoder(),
        worker_config = worker_config,
        max_samples_per_sequence=getattr(args, 'max_samples_per_sequence', None),
        shuffle_buffer_size=getattr(args, 'shuffle_buffer_size', None),
    )

    val_datasets_without_source_datasets = None
    if args.eval_iters > 0:
        val_datasets = get_val_datasets(
            dname,
            handler       = print_error_handler,
            split_part    = "validation",
            batch_size    = args.micro_batch_size,
            task_encoder  = TaskEncoder(),
            worker_config = worker_config,
        )
        val_datasets_without_source_datasets = [
            LimitDataset(
                RepeatDataset(val_ds, worker_config=worker_config),
                length=args.eval_iters * get_num_microbatches(),
                worker_config=worker_config,
                reset_after_epoch=True,
            )
            for val_ds, _src_ds in val_datasets
        ]

    return train_dataset, val_datasets_without_source_datasets, None


def train_valid_test_dataloaders_provider(train_val_test_num_samples):
    """Build energon dataloaders."""
    args = get_args()

    if not is_dataloader_rank():
        return None, None, None

    rank = parallel_state.get_data_parallel_rank()
    world_size = parallel_state.get_data_parallel_world_size()
    data_parallel_group = parallel_state.get_data_parallel_group()

    worker_config = WorkerConfig(
        rank=rank,
        world_size=world_size,
        num_workers=args.num_workers,
        data_parallel_group=data_parallel_group,
        worker_debug_path=None,
        worker_log_level=0,
    )

    train_ds, valid_ds1, test_ds = datasets_provider(worker_config)

    train_dataloader = get_savable_loader(train_ds, worker_config=worker_config)

    # Restore dataloader state if resuming
    if args.load is not None:
        if getattr(args, "dataloader_save", None):
            dp_rank = parallel_state.get_data_parallel_rank()
            data_save_name = get_checkpoint_name(
                args.dataloader_save,
                args.iteration,
                pipeline_rank=0,
                basename=f"train_dataloader_dprank{dp_rank:03d}.pt",
            )
            if os.path.exists(data_save_name):
                try:
                    dataset_state_dict = torch.load(
                        data_save_name, map_location="cpu", weights_only=False
                    )
                    train_dataloader.restore_state_rank(dataset_state_dict["dataloader_state_dict"])
                    print_rank_0(f"restored dataset state from {data_save_name}")
                except Exception as e:
                    print_rank_0("loading dataloader checkpoint failed. Skipping. " + str(e))

    if valid_ds1 is not None:
        valid_dataloader = [
            EnergonDataloader(get_loader(valid_ds, worker_config=worker_config))
            for valid_ds in valid_ds1
        ]
    else:
        valid_dataloader = EnergonDataloader(None)

    test_dataloader = None

    return EnergonDataloader(train_dataloader), valid_dataloader, EnergonDataloader(test_dataloader)


class EnergonDataloader:
    """Wrapper to use Megatron Energon dataloader with Megatron-LM training loop."""

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self._iter = iter(cyclic_iter(dataloader))

    def __next__(self):
        return self._iter.__next__()

    def __iter__(self):
        return self._iter.__iter__()

    def save_state(self):
        return self._dataloader.save_state_rank()


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


# ---------------------------------------------------------------------------
# Extra args
# ---------------------------------------------------------------------------

def add_fl_audio_extra_args(parser):
    """FL Audio specific arguments."""
    group = parser.add_argument_group(title="fl_audio arguments")

    group.add_argument("--use-te", action="store_true", default=False)

    # TeleFLM use default Megatron arguments)

    # Audio
    group.add_argument("--num-channel", type=int, default=8)
    group.add_argument("--aud-emp-token-id", type=int, default=2049)
    group.add_argument("--loss-weights", type=float, nargs="+",
                       default=[1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    # muP
    group.add_argument("--use-mup", action="store_true", default=True)
    group.add_argument("--mup-scale-factor", type=float, default=28.0)
    group.add_argument("--input-mult", type=float, default=1.0)
    group.add_argument("--output-mult", type=float, default=28.0)

    # DepthGPT
    group.add_argument("--depth-use-channel-mlp", action="store_true", default=True)
    group.add_argument("--depth-n-layer", required=True, type=int)
    group.add_argument("--depth-n-head", required=True, type=int)
    group.add_argument("--depth-hidden-size", required=True, type=int)
    group.add_argument("--depth-dropout", type=float, default=0.0)
    group.add_argument("--depth-bias", action="store_true", default=False)
    group.add_argument("--vocab-parallel-size", type=int, default=1,
                       help="Vocab parallel size for embedding. cp = tp_size / vp.")

    # Energon dataloader
    group.add_argument("--dataloader-save", type=str, default=None,
                       help="Energon dataloader state save path")
    group.add_argument("--max-samples-per-sequence", type=int, default=2**31 - 1)
    group.add_argument("--shuffle-buffer-size", type=int, default=0)

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_valid_test_dataloaders_provider.is_distributed = True

    pretrain(
        train_valid_test_dataloaders_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_fl_audio_extra_args,
    )
