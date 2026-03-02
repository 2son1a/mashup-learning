#!/usr/bin/env python3
"""
Generate LaTeX tables and plots from pipeline results.

Subcommands:
    run       Analyze a single pipeline run directory.
    model     Aggregate analysis across multiple seed runs for one model.
    combined  Cross-model combined tables for a group (lora, fullft, or all).

Examples:
    # Single run
    uv run python scripts/analyze.py run outputs/gemma3_1b_seed42_20260210/

    # Multi-seed model analysis
    uv run python scripts/analyze.py model "outputs/gemma3_1b_seed*" \\
        --output latex/gemma3_1b/

    # Combined tables for all LoRA models
    uv run python scripts/analyze.py combined lora --output latex/combined_lora/

    # Combined tables for all models
    uv run python scripts/analyze.py combined all --output latex/combined/
"""

import argparse
import glob as globmod
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> None:
    from mashup.analysis import analyze_run

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else run_dir / "analysis"
    analyze_run(run_dir, output_dir)


def cmd_model(args: argparse.Namespace) -> None:
    from mashup.analysis import analyze_model

    run_dirs = sorted(
        Path(p)
        for pattern in args.patterns
        for p in globmod.glob(pattern)
        if Path(p).is_dir()
    )
    if not run_dirs:
        print(f"Error: no directories match {args.patterns}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        first_cfg_name = run_dirs[0].name.rsplit("_seed", 1)[0]
        output_dir = Path("latex") / first_cfg_name
    analyze_model(run_dirs, output_dir)


def _discover_experiments(
    outputs_dir: Path,
    group: str,
) -> dict[str, list[Path]]:
    """Scan *outputs_dir* for run directories and group by experiment name.

    Each subdirectory must contain a ``pipeline.yaml`` with an analysis config.
    When *group* is not ``"all"``, only experiments whose config group matches
    are included.
    """
    from mashup.analysis.config import load_config_from_run

    by_experiment: dict[str, list[Path]] = {}
    if not outputs_dir.is_dir():
        return by_experiment

    for child in sorted(outputs_dir.iterdir()):
        if not child.is_dir():
            continue
        yaml_path = child / "pipeline.yaml"
        if not yaml_path.exists():
            continue
        try:
            cfg = load_config_from_run(child)
        except (ValueError, FileNotFoundError):
            continue
        if group != "all" and cfg.group != group:
            continue
        by_experiment.setdefault(cfg.experiment_name, []).append(child)

    return by_experiment


def cmd_combined(args: argparse.Namespace) -> None:
    from mashup.analysis import analyze_combined

    group = args.group
    outputs_dir = Path(args.outputs_dir)

    if args.output:
        output_dir = Path(args.output)
    elif group == "all":
        output_dir = Path("latex") / "combined"
    else:
        output_dir = Path("latex") / f"combined_{group}"

    run_dirs_by_experiment = _discover_experiments(outputs_dir, group)
    if not run_dirs_by_experiment:
        print(
            f"Error: no experiments found for group '{group}' in {outputs_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    analyze_combined(run_dirs_by_experiment, output_dir, group)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables and plots from pipeline results.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = sub.add_parser("run", help="Analyze a single pipeline run directory")
    p_run.add_argument("run_dir", help="Path to the pipeline run directory")
    p_run.add_argument(
        "-o", "--output", help="Output directory (default: <run_dir>/analysis/)"
    )
    p_run.set_defaults(func=cmd_run)

    # --- model ---
    p_model = sub.add_parser(
        "model",
        help="Multi-seed analysis for one model (glob patterns for seed dirs)",
    )
    p_model.add_argument(
        "patterns",
        nargs="+",
        help='Glob pattern(s) for seed directories, e.g. "outputs/gemma3_1b_seed*"',
    )
    p_model.add_argument(
        "-o", "--output", help="Output directory (default: latex/<model>/)"
    )
    p_model.set_defaults(func=cmd_model)

    # --- combined ---
    p_combined = sub.add_parser(
        "combined",
        help="Cross-model combined tables for a group",
    )
    p_combined.add_argument(
        "group",
        help="Group key: lora, fullft, or all",
    )
    p_combined.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Root outputs directory to scan (default: outputs/)",
    )
    p_combined.add_argument("-o", "--output", help="Output directory")
    p_combined.set_defaults(func=cmd_combined)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
