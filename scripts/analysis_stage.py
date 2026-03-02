#!/usr/bin/env python3
"""
Run analysis as a pipeline stage (single-run).

Usage:
    uv run python scripts/analysis_stage.py <config.yaml>
"""

import argparse

from omegaconf import OmegaConf

from mashup.analysis import run_stage


def main():
    parser = argparse.ArgumentParser(
        description="Run analysis as a pipeline stage (single-run)."
    )
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    run_stage(cfg)


if __name__ == "__main__":
    main()
