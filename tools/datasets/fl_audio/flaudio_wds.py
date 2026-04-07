import os
import sys
import time
import torch
import argparse
import webdataset as wds
from datasets import load_from_disk
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn

console = Console()

def print_header(data_input_path, wds_output_path):
    console.print(Panel.fit(
        f"[bold]WDS Dataset Builder[/bold]\n"
        f"[dim]source:[/dim]  {data_input_path}\n"
        f"[dim]output:[/dim]  {wds_output_path}",
        border_style="cyan",
    ))

def print_split_info(typex, dataset_len, wds_max_count, shard_count):
    is_partial = wds_max_count < dataset_len
    ratio_str = f"{wds_max_count / dataset_len * 100:.1f}%"
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("source size", str(dataset_len))
    table.add_row("target size", f"{wds_max_count}  ({ratio_str} of source)")
    table.add_row("shard count", str(shard_count))
    if is_partial:
        table.add_row("status", f"[yellow]⚠ partial build — only {ratio_str} converted[/yellow]")
    else:
        table.add_row("status", "[green]✓ full dataset[/green]")
    console.print(f"\n  [bold]\\[{typex}][/bold]")
    console.print(table)

def check_path_and_assert(path):
    assert os.path.exists(path), f"path ({path}) not exist"
def check_path_and_create(path):
    if not os.path.exists(path):
        os.mkdir(path)

def convert_hf_to_wds(args):
    data_input_path = args.data_path
    wds_output_path = args.output_path
    max_count = getattr(args, 'max_count', None)
    check_path_and_assert(data_input_path)
    check_path_and_create(wds_output_path)

    dataset = load_from_disk(data_input_path)

    print_header(data_input_path, wds_output_path)

    start_time = time.time()
    for typex in ["train", "test", "validation"]:
        path_typex = os.path.join(wds_output_path, typex)
        check_path_and_create(path_typex)
        dataset_len          = len(dataset[typex])
        this_wds_max_count   = min(dataset_len, max_count) if max_count else dataset_len
        this_shard_max_count = 10000
        this_wds_max_shard   = ((this_wds_max_count - 1) // this_shard_max_count) + 1

        print_split_info(typex, dataset_len, this_wds_max_count, this_wds_max_shard)

        offset = 0
        with Progress(
            TextColumn("[cyan]{task.description}[/cyan]"),
            BarColumn(bar_width=40, complete_style="green", finished_style="green"),
            MofNCompleteColumn(),
            TextColumn("[dim]ETA[/dim]"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(typex.ljust(10), total=this_wds_max_count)
            with wds.ShardWriter(
                os.path.join(path_typex,
                    f"{typex}-%03d.tar"), maxcount=this_shard_max_count
            ) as shard_writer:
                for idx in range(this_wds_max_count):
                    # NOTE[WQQ] The .pyd suffix may be meaningless, but it is required.
                    sample = {"__key__": str(idx+offset),
                              "audio_ids.pyd":
                              torch.tensor(dataset[typex][idx]["speak_ids"])}
                    shard_writer.write(sample)
                    progress.update(task, completed=idx + 1)
                offset += this_wds_max_count

    elapsed = time.time() - start_time
    console.print(Panel.fit(
        f"[green]✓ Done![/green]  elapsed: {elapsed:.1f}s",
        border_style="green",
    ))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--max-count', type=int, default=None, help="Max samples per split (default: all)")
    args = parser.parse_args()
    convert_hf_to_wds(args)

if __name__ == "__main__":
    main()
