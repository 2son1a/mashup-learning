#!/usr/bin/env python3
"""
Top-K model selection from evaluation results.

Usage:
    uv run python scripts/topk.py <config.yaml>
"""

import argparse

from omegaconf import OmegaConf

from mashup.topk import select_topk


def main():
    parser = argparse.ArgumentParser(
        description="Top-K model selection from evaluation results."
    )
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    select_topk(cfg)


if __name__ == "__main__":
    main()
