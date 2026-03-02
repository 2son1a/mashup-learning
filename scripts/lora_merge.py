#!/usr/bin/env python3
"""
LoRA adapter averaging.

Usage:
    uv run python scripts/lora_merge.py <config.yaml>
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from mashup.lora_merge import average_loras


def main():
    parser = argparse.ArgumentParser(description="LoRA adapter averaging.")
    parser.add_argument("config", type=str, help="Path to the YAML config file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    # Support both top-level models and merge.models (topk output format)
    models = (
        cfg.merge.models if "merge" in cfg and "models" in cfg.merge else cfg.models
    )
    adapter_dirs = [Path(m.path) for m in models]
    average_loras(adapter_dirs, Path(cfg.output_dir))


if __name__ == "__main__":
    main()
