#!/usr/bin/env python3
"""
Wait until sufficient free disk space is available before proceeding.

Usage:
    uv run python scripts/wait_for_disk.py <config.yaml>
"""

import argparse

from omegaconf import OmegaConf

from mashup.wait_for_disk import wait_for_disk


def main():
    parser = argparse.ArgumentParser(
        description="Wait until sufficient free disk space is available."
    )
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    wait_for_disk(cfg)


if __name__ == "__main__":
    main()
