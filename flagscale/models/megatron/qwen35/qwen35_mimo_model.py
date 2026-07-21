# Copyright (c) 2025, BAAI. All rights reserved.

"""Qwen3.5 colocated MIMO model wrapper."""

from typing import Any, Dict, List

import torch
import torch.distributed as dist

from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from flagscale.models.mimo import switch_parallel_state
from flagscale.models.mimo.mimo_bridge import (
    get_source_vision_rank,
    broadcast_to_language_tp,
)
from flagscale.models.mimo.mimo_scheduler import (
    MIMOMicrobatchScheduler,
    compute_microbatch_token_counts,
    split_visual_embeds,
    concatenate_visual_grads,
)
from flagscale.models.megatron.qwen35.language_model import Qwen35LanguageModule
from flagscale.models.megatron.qwen35.rope import get_rope_index
from flagscale.models.megatron.qwen35.transformer_config import Qwen35TransformerConfig
from flagscale.models.megatron.qwen3_vl.vision_model import Qwen3VisionModel


class Qwen35MIMOModel(MegatronModule):
    """Qwen3.5 MIMO model with colocated vision and language modules.

    The vision encoder + projection and the language model are constructed as
    separate submodules, each running under its own ``parallel_state`` context.
    """

    def __init__(
        self,
        language_transformer_config: Qwen35TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        pg_collections: Dict[str, object],
        vision_parallelism,
        language_parallelism,
        vision_projection_type: str = "mlp",
        parallel_output: bool = True,
        language_position_embedding_type: str = "mrope",
        language_rotary_percent: float = 0.25,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        language_rotary_base: int = 10000000,
        fp16_lm_cross_entropy: bool = False,
        language_share_embeddings_and_output_weights: bool = False,
        mtp_block_spec=None,
        vit_batch_factor: int = 1,
        use_fp32_grad_cache: bool = False,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.pg_collections = pg_collections
        self.vision_pg = pg_collections["vision"]
        self.language_pg = pg_collections["language"]

        self.vision_model = None
        self.language_model = None

        self.vit_batch_factor = max(1, vit_batch_factor)
        self.use_scheduler = self.vit_batch_factor > 1
        self.scheduler = MIMOMicrobatchScheduler(
            vit_batch_factor=self.vit_batch_factor,
            vision_forward_fn=self._vision_forward_fn,
            vision_backward_fn=self._vision_backward_fn,
            use_fp32_grad_cache=use_fp32_grad_cache,
        )

        # Buffers populated during _vision_forward_fn and consumed by _vision_backward_fn.
        self._macro_vision_embeds: torch.Tensor | None = None
        self._macro_deepstack_features: list[torch.Tensor] | None = None

        # Build vision module under the vision parallel context.
        if self.pre_process and self.add_encoder:
            with switch_parallel_state(self.vision_pg):
                self.vision_model = Qwen3VisionModel(
                    vision_transformer_config,
                    vision_transformer_layer_spec,
                    vision_projection_config,
                    vision_projection_layer_spec,
                    projection_type=vision_projection_type,
                    pre_process=True,
                    post_process=True,
                )

        # Build language module under the language parallel context.
        with switch_parallel_state(self.language_pg):
            self.language_model = Qwen35LanguageModule(
                config=language_transformer_config,
                transformer_layer_spec=language_transformer_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                position_embedding_type=language_position_embedding_type,
                rotary_percent=language_rotary_percent,
                pre_process=self.pre_process,
                post_process=self.post_process,
                rotary_base=language_rotary_base,
                fp16_lm_cross_entropy=fp16_lm_cross_entropy,
                share_embeddings_and_output_weights=language_share_embeddings_and_output_weights,
                rope_scaling=False,
                mtp_block_spec=mtp_block_spec,
                pg_collection=self.language_pg,
            )

        self.share_embeddings_and_output_weights = (
            self.language_model.share_embeddings_and_output_weights
        )

    def shared_embedding_or_output_weight(self):
        """Surface the language model's word embeddings for gradient all-reduce."""
        if self.add_decoder:
            with switch_parallel_state(self.language_pg):
                return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        with switch_parallel_state(self.language_pg):
            self.language_model.set_input_tensor(input_tensor)

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules."""
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_model is not None and self.vision_model.projection is not None:
            modules.append(self.vision_model.projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def next_microbatch(self, data_iterator, get_batch_fn):
        """Return the next LLM microbatch and its vision output from the scheduler.

        Assembles a new ViT macro batch when the current one is exhausted.  All
        scheduler orchestration lives behind this method so that
        ``forward_step`` never touches the scheduler directly.  The batch dict
        must be returned to the caller (Megatron builds the model inputs and
        the loss closure from it), which is why the advance cannot happen
        inside ``forward``.
        """
        assert self.use_scheduler, "next_microbatch is only valid when the scheduler is on"
        if self.scheduler.need_new_macro_batch():
            self.scheduler.prepare_macro_batch(get_batch_fn, data_iterator, self)
        _, batch, vision_output = self.scheduler.advance()
        return batch, vision_output

    # ------------------------------------------------------------------
    # Scheduler callbacks: run ViT on a macro batch and back-propagate later.
    # ------------------------------------------------------------------
    def _my_microbatch_range(self, num_micro: int) -> tuple[int, int]:
        """Return this rank's ``[lo, hi)`` microbatch slice within the macro batch.

        The language TP group splits the macro batch into whole microbatches:
        TP rank j owns ``[j*num/tp, (j+1)*num/tp)``.  Each rank's ViT forward
        therefore covers exactly ``vision_micro_batch_size`` samples (vbs) and
        the TP group jointly covers the ``vbf * mbs`` samples of one entity's
        macro batch — no duplicated ViT compute within the group.
        """
        tp_size = max(1, dist.get_world_size(self.language_pg.tp))
        assert num_micro % tp_size == 0, (
            f"vit_batch_factor ({num_micro}) must be divisible by language TP size "
            f"({tp_size}) so each TP rank owns whole microbatches"
        )
        per_rank = num_micro // tp_size
        tp_rank = dist.get_rank(self.language_pg.tp) if tp_size > 1 else 0
        lo = tp_rank * per_rank
        return lo, lo + per_rank

    def _vision_forward_fn(self, batches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run one ViT forward over this rank's slice of the macro batch.

        ``batches`` is a list of dictionaries returned by the training loop's
        ``get_batch``.  This rank concatenates the visual inputs of its own
        microbatch slice only (see ``_my_microbatch_range``), runs the vision
        model under the vision parallel context, splits the output back into
        per-microbatch chunks, and exchanges chunks inside the language TP
        group so every rank holds the full macro batch's outputs (its own
        slice computed, the rest received from their owner ranks).
        """
        if not self.pre_process or not self.add_encoder or self.vision_model is None:
            return [{"vision_embeds": None, "deepstack_features": None} for _ in batches]

        vision_tp_size = dist.get_world_size(self.vision_pg.tp)
        assert vision_tp_size == 1, (
            f"MIMO vision backward currently supports vision TP=1 only; got {vision_tp_size}."
        )

        # ``get_batch`` returns separate image/video tensors; concatenate them
        # along the sample dimension to match the ViT input layout.
        def _cat_imgs_videos(b):
            imgs = b.get("imgs")
            videos = b.get("videos")
            if imgs is None:
                return videos
            if videos is None:
                return imgs
            return torch.cat([imgs, videos], dim=0)

        def _cat_grids(b):
            image_grids = b.get("image_thw_grids")
            video_grids = b.get("video_thw_grids")
            if image_grids is None:
                return video_grids
            if video_grids is None:
                return image_grids
            return torch.cat([image_grids, video_grids], dim=0)

        per_batch_imgs_videos = [_cat_imgs_videos(b) for b in batches]
        per_batch_grids = [_cat_grids(b) for b in batches]

        if not any(t is not None for t in per_batch_grids):
            return [{"vision_embeds": None, "deepstack_features": None} for _ in batches]

        # Each rank computes only its own slice of the macro batch; the TP
        # group jointly covers all microbatches (see _my_microbatch_range).
        lo, hi = self._my_microbatch_range(len(batches))
        my_imgs_videos = [t for t in per_batch_imgs_videos[lo:hi] if t is not None]
        my_grids = [t for t in per_batch_grids[lo:hi] if t is not None]

        if my_grids:
            vision_data = torch.cat(my_imgs_videos, dim=0)
            vision_grid_thw = torch.cat(my_grids, dim=0)
            with switch_parallel_state(self.vision_pg):
                if vision_grid_thw.shape[0] == 0:
                    macro_embeds = None
                    macro_deepstack = None
                else:
                    macro_embeds, macro_deepstack = self.vision_model(
                        vision_data=vision_data,
                        grid_thw=vision_grid_thw,
                    )
        else:
            macro_embeds = None
            macro_deepstack = None

        # This rank's macro output covers only its own slice of the macro batch.
        self._macro_vision_embeds = macro_embeds
        self._macro_deepstack_features = macro_deepstack

        # Account for Qwen3-VL spatial merge when splitting the macro output.
        spatial_merge_unit = getattr(self.vision_model, "spatial_merge_unit", 1)
        token_counts = compute_microbatch_token_counts(
            per_batch_grids, merge_unit=spatial_merge_unit
        )

        # Assemble the full per-microbatch output list: entries in this rank's
        # own slice hold locally computed tensors; other entries are empty
        # buffers that the broadcast below fills from their owner ranks.
        hidden_size = self.config.hidden_size
        dtype = macro_embeds.dtype if macro_embeds is not None else self.config.params_dtype
        device = torch.cuda.current_device()
        if macro_deepstack is not None:
            deepstack_levels = len(macro_deepstack)
        else:
            deepstack_levels = len(
                getattr(self.vision_model.config, "deepstack_visual_indexes", None) or []
            )

        def _empty_entry(n_tokens: int) -> Dict[str, Any]:
            entry = {
                "vision_embeds": torch.empty(n_tokens, hidden_size, dtype=dtype, device=device),
                "deepstack_features": None,
            }
            if deepstack_levels > 0:
                entry["deepstack_features"] = [
                    torch.empty(n_tokens, hidden_size, dtype=dtype, device=device)
                    for _ in range(deepstack_levels)
                ]
            return entry

        if macro_embeds is not None:
            my_outputs = split_visual_embeds(
                macro_embeds, macro_deepstack, token_counts[lo:hi], dim=0
            )
        else:
            # This rank's slice has no visual data; zero-token entries keep the
            # broadcast below collective-consistent with the other slices.
            my_outputs = [_empty_entry(0) for _ in range(lo, hi)]

        split_outputs = []
        my_idx = 0
        for i in range(len(batches)):
            if lo <= i < hi:
                split_outputs.append(my_outputs[my_idx])
                my_idx += 1
            else:
                split_outputs.append(_empty_entry(token_counts[i]))

        # Broadcast each microbatch's visual embeds inside its language TP
        # group.  The source rank is the owner that actually computed the
        # slice this microbatch belongs to.
        for forward_idx, output in enumerate(split_outputs):
            src_rank = get_source_vision_rank(self.language_pg, forward_idx, self.vit_batch_factor)
            if output["vision_embeds"] is not None:
                output["vision_embeds"] = broadcast_to_language_tp(
                    output["vision_embeds"], self.language_pg, src_rank
                )
            if output["deepstack_features"] is not None:
                output["deepstack_features"] = [
                    broadcast_to_language_tp(f, self.language_pg, src_rank)
                    for f in output["deepstack_features"]
                ]

        # Received buffers must require grad so that every served tensor
        # registers a grad hook — the scheduler's macro-batch completion
        # trigger relies on a hook firing for *every* microbatch.  Set after
        # the broadcast to avoid in-place writes into requires-grad leaves.
        # Grads captured for microbatches this rank does not own are unused.
        for output in split_outputs:
            if output["vision_embeds"] is not None:
                output["vision_embeds"].requires_grad_(True)
            if output["deepstack_features"] is not None:
                for f in output["deepstack_features"]:
                    f.requires_grad_(True)

        return split_outputs

    def _vision_backward_fn(self, gradients: List[Dict[str, Any]]) -> None:
        """Run ViT backward after all microbatch gradients are collected.

        Each rank's captured gradient for a microbatch is already the complete
        dL/d(vision_embeds): the LM's TP boundary collectives (all-reduce, or
        all-gather for sequence parallel) aggregate input gradients before they
        reach the embedding injection point, so every language TP peer holds an
        identical copy (verified by instrumentation: peers' captured grads are
        bitwise identical).  No reduce/broadcast is needed — each rank simply
        backwards the gradients of its own microbatch slice through its own
        ViT forward, and vision DDP then averages parameter gradients over
        disjoint slices (equivalent to averaging over the full global batch).
        """
        if not self.pre_process or not self.add_encoder or self.vision_model is None:
            return

        macro_embeds = self._macro_vision_embeds
        macro_deepstack = self._macro_deepstack_features
        if macro_embeds is None:
            return

        assert dist.get_world_size(self.vision_pg.tp) == 1, (
            "MIMO vision backward currently supports vision TP=1 only; got "
            f"{dist.get_world_size(self.vision_pg.tp)}."
        )

        lo, hi = self._my_microbatch_range(len(gradients))
        my_gradients = gradients[lo:hi]

        def _assemble_slice_grad(key: str) -> torch.Tensor:
            # Microbatches without visual data contribute ``None`` gradient
            # dicts; skip them so the concatenated gradient matches this
            # rank's slice of the macro output.
            if not any(g is not None and key in g for g in my_gradients):
                return None
            return concatenate_visual_grads(my_gradients, key=key, dim=0)

        embed_grad = _assemble_slice_grad("vision_embeds")
        if embed_grad is None:
            return

        # Cast gradients back to the ViT output dtype when fp32 cache was used.
        target_dtype = macro_embeds.dtype
        def _to_vit_dtype(g):
            return g.to(target_dtype) if g is not None else None

        grads = [_to_vit_dtype(embed_grad)]
        if macro_deepstack is not None:
            for i in range(len(macro_deepstack)):
                grads.append(_to_vit_dtype(_assemble_slice_grad(f"deepstack_{i}")))

        targets = [macro_embeds] + (macro_deepstack or [])
        with switch_parallel_state(self.vision_pg):
            torch.autograd.backward(targets, grads)

        self._macro_vision_embeds = None
        self._macro_deepstack_features = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        vision_data: torch.Tensor = None,
        vision_grid_thw: torch.Tensor = None,
        video_start_index: int = -1,
        image_input_mask: torch.Tensor = None,
        video_input_mask: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor | None = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        vision_output: dict = None,
    ) -> torch.Tensor:
        """Forward function of Qwen3.5 MIMO model."""
        use_inference_kv_cache = (
            inference_params is not None
            and "image_tokens_count" in inference_params.key_value_memory_dict
        )
        if use_inference_kv_cache:
            raise NotImplementedError()

        if self.pre_process and self.add_encoder:
            if vision_output is not None:
                # Scheduler path: caller already advanced the scheduler.
                vision_embeds = vision_output["vision_embeds"]
                deepstack_feature_lists = vision_output["deepstack_features"]
            elif self.use_scheduler:
                # Model-internal advance: a caller prepared the macro batch but
                # left advancing to the model (not used by forward_step, which
                # needs the batch up front and therefore advances externally).
                _, _, vision_output = self.scheduler.advance()
                vision_embeds = vision_output["vision_embeds"]
                deepstack_feature_lists = vision_output["deepstack_features"]
            else:
                # Direct path (vit_batch_factor == 1, e.g. vision_micro_batch_size
                # == language micro batch or vision TP == language TP): run ViT
                # for this single microbatch.  Not used when the scheduler is on.
                with switch_parallel_state(self.vision_pg):
                    vision_embeds = None
                    deepstack_feature_lists = None
                    if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                        vision_embeds, deepstack_feature_lists = self.vision_model(
                            vision_data=vision_data,
                            grid_thw=vision_grid_thw,
                        )
                    else:
                        vision_embeds = None
                        deepstack_feature_lists = None

                # Vision outputs are full [T, H] tensors on every ViT TP rank.
                # Broadcast them inside the language TP group so every language
                # rank sees the same visual embeddings, matching the baseline
                # behaviour where the embedding layer consumes full tensors.
                src_rank = get_source_vision_rank(
                    self.language_pg, 0, self.vit_batch_factor
                )
                if vision_embeds is not None:
                    vision_embeds = broadcast_to_language_tp(
                        vision_embeds, self.language_pg, src_rank
                    )
                    if deepstack_feature_lists is not None:
                        deepstack_feature_lists = [
                            broadcast_to_language_tp(f, self.language_pg, src_rank)
                            for f in deepstack_feature_lists
                        ]

            # Detach visual tensors from the ViT graph and capture gradients for
            # delayed ViT backward when the scheduler is active.
            if self.use_scheduler:
                vision_embeds = self.scheduler.register_visual_grad_hook(
                    vision_embeds, "vision_embeds"
                )
                if deepstack_feature_lists is not None:
                    deepstack_feature_lists = [
                        self.scheduler.register_visual_grad_hook(f, f"deepstack_{i}")
                        for i, f in enumerate(deepstack_feature_lists)
                    ]

            if inference_params is not None:
                raise NotImplementedError()

            with switch_parallel_state(self.language_pg):
                if use_inference_kv_cache:
                    language_embeddings = self.language_model.embedding(
                        input_ids=input_ids,
                        position_ids=None,
                    )
                    combined_embeddings = language_embeddings
                elif vision_embeds is not None:
                    if image_input_mask is not None:
                        image_input_mask = image_input_mask.T
                    if video_input_mask is not None:
                        video_input_mask = video_input_mask.T

                    if video_start_index == 0:
                        image_embeds = None
                        video_embeds = vision_embeds
                        visual_pos_masks = video_input_mask
                    elif video_start_index == vision_embeds.shape[0]:
                        image_embeds = vision_embeds
                        video_embeds = None
                        visual_pos_masks = image_input_mask
                    elif 0 < video_start_index < vision_embeds.shape[0]:
                        image_embeds = vision_embeds[:video_start_index]
                        video_embeds = vision_embeds[video_start_index:]
                        visual_pos_masks = torch.logical_or(image_input_mask, video_input_mask)
                    else:
                        raise ValueError(
                            f"Expect video token start index in range [0, {vision_embeds.shape[0]}], "
                            f"but got {video_start_index}"
                        )

                    combined_embeddings = self.language_model.embedding(
                        input_ids=input_ids,
                        position_ids=None,
                        image_input_mask=image_input_mask,
                        video_input_mask=video_input_mask,
                        image_embeds=image_embeds,
                        video_embeds=video_embeds,
                    )
                else:
                    combined_embeddings = self.language_model.embedding(
                        input_ids=input_ids,
                        position_ids=None,
                    )
                    visual_pos_masks = None
                    deepstack_feature_lists = None
        else:
            combined_embeddings = None
            visual_pos_masks = None
            deepstack_feature_lists = None

        with switch_parallel_state(self.language_pg):
            output = self.language_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=combined_embeddings,
                labels=labels,
                loss_mask=loss_mask,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_feature_lists,
                **(extra_block_kwargs or {}),
            )
        return output

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mRoPE position indices for Qwen3.5."""
        return get_rope_index(
            spatial_merge_size=self.config.spatial_merge_size,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
