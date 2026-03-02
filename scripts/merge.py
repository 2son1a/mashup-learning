#!/usr/bin/env python3
"""
Model merging via mergekit.

Usage:
    uv run python scripts/merge.py <config.yaml>
"""

import argparse

from omegaconf import OmegaConf

from mashup.merging import merge


def main():
    parser = argparse.ArgumentParser(description="Model merging via mergekit.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    merge(cfg)


if __name__ == "__main__":
    main()
