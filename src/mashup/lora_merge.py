"""Simple LoRA adapter averaging: load safetensors, average matrices, save."""

import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def average_loras(adapter_dirs: list[Path], output_dir: Path) -> Path:
    """Average k LoRA adapters into one by averaging corresponding tensors."""
    if len(adapter_dirs) < 2:
        raise ValueError("Need at least 2 adapters")

    state_dicts = []
    for d in adapter_dirs:
        path = d / "adapter_model.safetensors"
        if not path.exists():
            raise FileNotFoundError(f"No adapter_model.safetensors in {d}")
        state_dicts.append(load_file(path))

    k = len(state_dicts)
    averaged = {}
    for key in state_dicts[0]:
        averaged[key] = sum(sd[key] for sd in state_dicts) / k

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(averaged, output_dir / "adapter_model.safetensors")
    shutil.copy2(
        adapter_dirs[0] / "adapter_config.json", output_dir / "adapter_config.json"
    )

    print(f"Averaged {k} adapters → {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Average LoRA adapters into one.")
    parser.add_argument(
        "output_dir", type=Path, help="Directory to write the averaged adapter"
    )
    parser.add_argument(
        "adapters",
        type=Path,
        nargs="+",
        help="Adapter directories to average (at least 2)",
    )
    args = parser.parse_args()
    average_loras(args.adapters, args.output_dir)
