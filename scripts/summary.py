#!/usr/bin/env python3
"""
Build a summary table from evaluation results.

Usage:
    uv run python scripts/summary.py <config.yaml>
"""

import argparse

from omegaconf import OmegaConf

from mashup.summary import build_summary


def main():
    parser = argparse.ArgumentParser(
        description="Build a summary table from evaluation results."
    )
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    build_summary(cfg)


if __name__ == "__main__":
    main()
