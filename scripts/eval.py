#!/usr/bin/env python3
"""
Model evaluation on preprocessed chat datasets.

Usage:
    uv run python scripts/eval.py <config.yaml>
"""

import argparse

from omegaconf import OmegaConf

from mashup.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Model evaluation on preprocessed chat datasets."
    )
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()
