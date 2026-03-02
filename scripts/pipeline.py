#!/usr/bin/env python3
"""
Multi-stage pipeline with async GPU scheduling.

Usage:
    uv run python scripts/pipeline.py configs/pipeline/leave_one_out_lora.yaml

Resume a failed run (0-indexed stage):
    uv run python scripts/pipeline.py configs/pipeline/leave_one_out_lora.yaml \\
        --pipeline-checkpoint outputs/run_20260209_060441 --resume-stage 1

Run with a specific seed (for reproducibility across seed runs):
    uv run python scripts/pipeline.py configs/pipeline/leave_one_out_lora.yaml --seed 42

Override base model and experiment name (reuse one config for multiple models):
    uv run python scripts/pipeline.py configs/pipeline/leave_one_out_lora.yaml \\
        --base-model google/gemma-3-4b-it --experiment-name gemma3_4b --seed 42

Limit number of GPUs (e.g. to reduce disk pressure from parallel jobs):
    uv run python scripts/pipeline.py configs/pipeline/leave_one_out_fullft.yaml --max-gpus 5
"""

import argparse
import asyncio
from pathlib import Path

from omegaconf import OmegaConf

from mashup.pipeline import register_signal_handlers, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-stage pipeline with async GPU scheduling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config", type=Path, help="Path to the pipeline YAML config file"
    )
    parser.add_argument(
        "--pipeline-checkpoint",
        type=Path,
        default=None,
        help="Run directory to resume from (requires --resume-stage)",
    )
    parser.add_argument(
        "--resume-stage",
        type=int,
        default=None,
        help="0-indexed stage to resume from (requires --pipeline-checkpoint)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility across seed runs",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="HuggingFace model ID (overrides config)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (overrides config)",
    )
    parser.add_argument(
        "--max-gpus",
        type=int,
        default=None,
        help="Cap number of GPUs used (e.g. to reduce disk pressure)",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"Config not found: {args.config}")

    if (args.pipeline_checkpoint is None) != (args.resume_stage is None):
        parser.error(
            "--pipeline-checkpoint and --resume-stage must be provided together"
        )

    if args.pipeline_checkpoint is not None and not args.pipeline_checkpoint.exists():
        parser.error(f"Checkpoint directory not found: {args.pipeline_checkpoint}")

    register_signal_handlers()
    asyncio.run(
        run_pipeline(
            OmegaConf.load(args.config),
            pipeline_checkpoint=args.pipeline_checkpoint,
            resume_stage=args.resume_stage,
            seed=args.seed,
            base_model=args.base_model,
            experiment_name=args.experiment_name,
            max_gpus=args.max_gpus,
        )
    )


if __name__ == "__main__":
    main()
