#!/usr/bin/env python3
"""
Delete directories matching glob patterns within the run directory.

Usage:
    uv run python scripts/cleanup.py <config.yaml>
"""

import argparse

from omegaconf import OmegaConf

from mashup.cleanup import cleanup


def main():
    parser = argparse.ArgumentParser(
        description="Delete directories matching glob patterns within the run directory."
    )
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cleanup(cfg)


if __name__ == "__main__":
    main()
