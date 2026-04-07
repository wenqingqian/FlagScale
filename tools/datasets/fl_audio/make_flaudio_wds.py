import os
import re
import sys
import glob
import subprocess
import argparse
import shutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich.live import Live
from rich.spinner import Spinner
from rich.prompt import Confirm
from rich.text import Text

console = Console()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Step 1: detect tar files and build --split-parts args ──────────────────

def detect_split_parts(output_path):
    """Scan output_path for tar files, return dict {split_name: brace_pattern}."""
    splits = {}
    for split_name in ["train", "test", "validation"]:
        split_dir = os.path.join(output_path, split_name)
        if not os.path.isdir(split_dir):
            continue
        tars = sorted(glob.glob(os.path.join(split_dir, "*.tar")))
        if not tars:
            continue

        indices = []
        width = None
        for t in tars:
            m = re.search(r'-(\d+)\.tar$', os.path.basename(t))
            if m:
                indices.append(int(m.group(1)))
                if width is None:
                    width = len(m.group(1))

        if not indices or width is None:
            continue

        lo, hi = min(indices), max(indices)
        if len(indices) == hi - lo + 1:
            # Continuous range
            pattern = f"{split_name}:{split_name}/{split_name}-{{{lo:0{width}d}..{hi:0{width}d}}}.tar"
        else:
            # Non-continuous, enumerate
            parts = ",".join(f"{i:0{width}d}" for i in sorted(indices))
            pattern = f"{split_name}:{split_name}/{split_name}-{{{parts}}}.tar"

        splits[split_name] = (pattern, len(indices))
    return splits


def print_detection_result(splits):
    table = Table(title="Detected Splits", border_style="cyan", show_lines=False)
    table.add_column("Split", style="bold")
    table.add_column("Shards", justify="right")
    table.add_column("Pattern", style="dim")
    for name in ["train", "test", "validation"]:
        if name in splits:
            pattern, count = splits[name]
            table.add_row(name, str(count), pattern)
    console.print(table)
    console.print()


# ── Step 2: run energon prepare ────────────────────────────────────────────

def run_energon_prepare(output_path, splits):
    cmd = ["energon", "prepare", output_path, "--non-interactive"]
    for name in ["train", "test", "validation"]:
        if name in splits:
            pattern, _ = splits[name]
            cmd.extend(["--split-parts", pattern])
    cmd.extend(["--fix-duplicates", "--sample-type", "CrudeWebdataset", "--force-overwrite"])

    console.print(Rule("[bold]Energon Prepare[/bold]", style="cyan"))
    console.print(f"  [dim]cmd:[/dim] {' '.join(cmd)}\n")

    with Live(Spinner("dots", text="Running energon prepare..."), console=console, refresh_per_second=10) as live:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            live.update(Spinner("dots", text=Text(line[-80:] if len(line) > 80 else line)))
        proc.wait()

    if proc.returncode != 0:
        console.print(f"\n[red]✗ energon prepare failed (exit code {proc.returncode})[/red]")
        for l in lines[-20:]:
            console.print(f"  [dim]{l}[/dim]")
        return False

    console.print(f"[green]✓ energon prepare completed[/green]\n")
    return True


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FLAudio WDS Pipeline")
    parser.add_argument("--data-path", type=str, required=True, help="Source HF dataset path")
    parser.add_argument("--output-path", type=str, required=True, help="WDS output path")
    parser.add_argument("--skip-convert", action="store_true", help="Skip WDS conversion (reuse existing)")
    parser.add_argument("--skip-test", action="store_true", help="Skip validation test")
    parser.add_argument("--clean", action="store_true", help="Remove output dir before conversion")
    parser.add_argument("--max-count", type=int, default=None, help="Max samples per split (default: all)")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold]FLAudio WDS Pipeline[/bold]\n"
        f"[dim]source:[/dim]  {args.data_path}\n"
        f"[dim]output:[/dim]  {args.output_path}",
        border_style="cyan",
    ))

    # Check for existing WDS data
    existing_tars = glob.glob(os.path.join(args.output_path, "**/*.tar"), recursive=True)
    if existing_tars and not args.skip_convert:
        console.print(f"\n[yellow]⚠ Found {len(existing_tars)} existing tar files in {args.output_path}[/yellow]")
        if not Confirm.ask("  Existing WDS data will be [bold red]deleted[/bold red]. Continue?", console=console):
            console.print("[dim]Aborted.[/dim]")
            sys.exit(0)
        shutil.rmtree(args.output_path)
        console.print(f"  [dim]cleaned {args.output_path}[/dim]\n")
    elif args.clean and os.path.exists(args.output_path) and not args.skip_convert:
        console.print(f"\n[yellow]⚠ --clean specified, will remove {args.output_path}[/yellow]")
        if not Confirm.ask("  Confirm deletion?", console=console):
            console.print("[dim]Aborted.[/dim]")
            sys.exit(0)
        shutil.rmtree(args.output_path)
        console.print(f"  [dim]cleaned {args.output_path}[/dim]\n")

    # Step 1: WDS conversion
    if not args.skip_convert:
        console.print(Rule("[bold]Step 1 · WDS Conversion[/bold]", style="cyan"))

        from flaudio_wds import convert_hf_to_wds
        convert_args = argparse.Namespace(
            data_path=args.data_path,
            output_path=args.output_path,
            max_count=args.max_count,
        )
        convert_hf_to_wds(convert_args)
    else:
        console.print(Rule("[bold]Step 1 · WDS Conversion[/bold] [dim](skipped)[/dim]", style="cyan"))

    # Step 2: detect tars + energon prepare
    console.print(Rule("[bold]Step 2 · Detect & Prepare[/bold]", style="cyan"))
    splits = detect_split_parts(args.output_path)
    if not splits:
        console.print("[red]✗ No tar files found in output path[/red]")
        sys.exit(1)
    print_detection_result(splits)

    ok = run_energon_prepare(args.output_path, splits)
    if not ok:
        sys.exit(1)

    # Step 3: validation test
    if not args.skip_test:
        console.print(Rule("[bold]Step 3 · Validation Test[/bold]", style="cyan"))
        from flaudio_wds_test import run_tests
        passed = run_tests(args.data_path, args.output_path)
        if not passed:
            console.print("[red]✗ Some tests failed[/red]")
            sys.exit(1)
    else:
        console.print(Rule("[bold]Step 3 · Validation Test[/bold] [dim](skipped)[/dim]", style="cyan"))

    console.print(Panel.fit("[green]✓ All done![/green]", border_style="green"))


if __name__ == "__main__":
    main()
