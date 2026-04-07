from dataclasses import dataclass
import dataclasses
from datasets import load_from_disk
from typing import Dict, List
import torch
import sys
from megatron.energon import (
    DefaultTaskEncoder,
    Batch,
    WorkerConfig,
    get_train_dataset,
    get_val_dataset,
    get_loader,
    Cooker
)
import argparse
from megatron.energon.task_encoder.base import stateless
import os
import random
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn
from rich.panel import Panel

console = Console()

@dataclass
class FLAudioSample:
    __key__: str
    __restore_key__: str
    audio_ids:  torch.tensor

@dataclass
class FLAudioSampleBatch(Batch):
    __key__: str
    __restore_key__: str
    audio_ids:  torch.tensor
@stateless
def cook_text(sample: dict) -> FLAudioSample:
    key = int(sample["__key__"].strip('/').split('/')[-1])
    return FLAudioSample(
        __key__   = key,
        audio_ids = sample["audio_ids.pyd"],
        __restore_key__ = sample["__restore_key__"]
    )

class FLAudioTaskEncoder(DefaultTaskEncoder[FLAudioSample, FLAudioSample,FLAudioSampleBatch, dict]):
    cookers = [Cooker(cook_text)]

    @stateless
    def encode_sample(self, sample: dict) -> FLAudioSample:

        return FLAudioSample(
            __key__   = sample.__key__,
            __restore_key__ = sample.__restore_key__,
            audio_ids = sample.audio_ids,
        )

    def batch(self, samples: list[FLAudioSample]) -> FLAudioSampleBatch:

        input_batch = torch.stack([s.audio_ids for s in samples], dim=0)
        return FLAudioSampleBatch(
            __key__   = [s.__key__ for s in samples],
            __restore_key__ = [s.__restore_key__ for s in samples],
            audio_ids = input_batch,
        )

    def encode_batch(self, batch: FLAudioSampleBatch) -> dict:
        raw = dataclasses.asdict(batch)
        return raw

def build_train_loader(
    dataset_path: str,
    batch_size:   int = 4,
    num_workers:  int = 2,
    shuffle_buffer: int = 1000,
):
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=num_workers,
    )

    task_encoder = FLAudioTaskEncoder()

    train_ds = get_train_dataset(
        dataset_path,
        split_part      = "train",
        batch_size      = batch_size,
        task_encoder    = task_encoder,
        worker_config   = worker_config,
        shuffle_buffer_size = shuffle_buffer,
        max_samples_per_sequence = None,
    )

    return get_loader(train_ds, worker_config=worker_config)


def build_val_loader(
    dataset_path: str,
    batch_size:   int = 4,
    num_workers:  int = 2,
):
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=num_workers,
    )

    task_encoder = FLAudioTaskEncoder()

    val_ds = get_val_dataset(
        dataset_path,
        split_part    = "validation",
        batch_size    = batch_size,
        task_encoder  = task_encoder,
        worker_config = worker_config,
    )

    return get_loader(val_ds, worker_config=worker_config)


def build_test_loader(
    dataset_path: str,
    batch_size:   int = 4,
    num_workers:  int = 2,
):
    worker_config = WorkerConfig(
        rank=0,
        world_size=1,
        num_workers=num_workers,
    )

    task_encoder = FLAudioTaskEncoder()

    val_ds = get_val_dataset(
        dataset_path,
        split_part    = "test",
        batch_size    = batch_size,
        task_encoder  = task_encoder,
        worker_config = worker_config,
    )

    return get_loader(val_ds, worker_config=worker_config)


def _wds_sample_count(wds_path, split_name):
    """Read actual WDS sample count from .info.json for a given split."""
    import json
    info_path = os.path.join(wds_path, ".nv-meta", ".info.json")
    if not os.path.exists(info_path):
        return None
    with open(info_path) as f:
        info = json.load(f)
    total = 0
    for shard, count in info.get("shard_counts", {}).items():
        if shard.startswith(f"{split_name}/"):
            total += count
    return total if total > 0 else None


def run_tests(raw_data_path, wds_path, max_samples=300):
    original_dataset = load_from_disk(raw_data_path)

    fn = {
        "train": build_train_loader,
        "test" : build_test_loader,
        "validation": build_val_loader,
    }

    console.print(Panel.fit(
        "[bold]WDS Validation Test[/bold]",
        border_style="cyan",
    ))

    all_passed = True
    for b in [1, 2, 4]:
        for t in ["train", "validation", "test"]:
            wds_size = _wds_sample_count(wds_path, t)
            if wds_size is None:
                wds_size = len(original_dataset[t])
            # Cap by max_samples and actual WDS size
            max_batches = min(max_samples // b, wds_size // b)
            if max_batches <= 0:
                continue

            loader = fn[t](wds_path, batch_size=b)

            # Collect batches (loader may be infinite)
            all_batches = []
            for batch in loader:
                all_batches.append(batch)
                if len(all_batches) >= max_batches:
                    break

            # Random sample from collected batches
            check_count = min(max_batches, len(all_batches))
            sampled = random.sample(all_batches, check_count)

            correct = 0
            desc = f"{t:10s} batch={b}"
            with Progress(
                TextColumn("[cyan]{task.description}[/cyan]"),
                BarColumn(bar_width=30, complete_style="green", finished_style="green"),
                MofNCompleteColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(desc, total=check_count)
                for step, batch in enumerate(sampled):
                    wds_audio = batch["audio_ids"].squeeze(0)
                    ori_id_list = batch["__key__"]
                    actual_bs = len(ori_id_list)
                    orig_batch_list = []
                    for i in range(actual_bs):
                        orig_data = torch.tensor(original_dataset[t][ori_id_list[i]]["speak_ids"])
                        orig_batch_list.append(orig_data)
                    orig_batch = torch.stack(orig_batch_list)
                    wds_audio = wds_audio.reshape(orig_batch.shape)
                    if (
                        wds_audio.shape == orig_batch.shape
                        and wds_audio.dtype == orig_batch.dtype
                        and torch.equal(wds_audio, orig_batch)
                    ):
                        correct += 1
                    progress.update(task, completed=step + 1)

            status = "[green]✓ PASS[/green]" if correct == check_count else f"[red]✗ FAIL ({correct}/{check_count})[/red]"
            console.print(f"  {desc}  {status}")
            if correct != check_count:
                all_passed = False

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-data-path', type=str, required=True)
    parser.add_argument('--wds-path', type=str, required=True)
    args = parser.parse_args()
    passed = run_tests(args.raw_data_path, args.wds_path)
    if not passed:
        sys.exit(1)
