#!/usr/bin/env python3
"""
Axolotl training script.

Usage:
    # Train with a config file
    uv run python scripts/train.py configs/llama-3-lora-1b.yml

    # Or use accelerate for multi-GPU
    uv run accelerate launch -m axolotl.cli.train configs/llama-3-lora-1b.yml
"""

import argparse
import os
from pathlib import Path


def load_dotenv():
    env_file = Path(".env")
    if env_file.exists():
        from dotenv import load_dotenv as _load_dotenv

        _load_dotenv(env_file)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Axolotl training script.")
    parser.add_argument("config", type=Path, help="Path to the YAML config file")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    print(f"Training with config: {config_path}")

    if os.environ.get("WANDB_API_KEY"):
        print("WandB logging enabled")
        if not os.environ.get("WANDB_NAME"):
            os.environ["WANDB_NAME"] = config_path.stem
    else:
        print("WandB logging disabled (no API key)")

    # Deferred import: axolotl is a heavy dependency with slow import time
    from axolotl.cli.main import train

    train([str(config_path)])


if __name__ == "__main__":
    main()
