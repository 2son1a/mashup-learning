#!/usr/bin/env python3
"""
Best model selection from evaluation results.

Usage:
    uv run python scripts/select_best.py <config.yaml>
"""

import argparse

from omegaconf import OmegaConf

from mashup.select_best import select_best


def main():
    parser = argparse.ArgumentParser(
        description="Best model selection from evaluation results."
    )
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    select_best(cfg)


if __name__ == "__main__":
    main()
