#!/usr/bin/env python3
"""Preprocess datasets offline for training.

This script loads raw datasets from HuggingFace, applies template functions,
and saves processed datasets locally. This avoids runtime preprocessing issues
with distributed training and custom prompt strategies.

Usage:
    # Process all datasets
    uv run python scripts/preprocess_datasets.py --all

    # Process specific dataset
    uv run python scripts/preprocess_datasets.py --dataset piqa

    # Process multiple datasets
    uv run python scripts/preprocess_datasets.py --dataset piqa --dataset arc_easy
"""

import argparse
from pathlib import Path

from mashup.preprocessing import DATASETS, preprocess_all, preprocess_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all datasets
  python scripts/preprocess_datasets.py --all

  # Process specific dataset
  python scripts/preprocess_datasets.py --dataset piqa

  # Process multiple datasets
  python scripts/preprocess_datasets.py --dataset piqa --dataset arc_easy
        """,
    )

    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DATASETS.keys()),
        help="Dataset to process (can specify multiple times)",
    )

    parser.add_argument("--all", action="store_true", help="Process all datasets")

    parser.add_argument(
        "--split",
        action="append",
        help="Split(s) to process (can specify multiple times, default: all available)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed datasets (default: data/processed)",
    )

    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Must specify either --all or --dataset")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        preprocess_all(args.output_dir, splits=args.split)
    else:
        for dataset_name in args.dataset:
            try:
                preprocess_dataset(dataset_name, args.output_dir, splits=args.split)
            except Exception as e:
                print(f"✗ Failed to process {dataset_name}: {e}")
                continue


if __name__ == "__main__":
    main()
