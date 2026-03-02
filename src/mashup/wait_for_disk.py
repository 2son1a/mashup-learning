"""Wait until sufficient free disk space is available before proceeding.

Polls every 30 seconds and exits once the threshold is met.
Intended for use as a pipeline subjob step before train/eval to prevent
disk overcrowding when many jobs run in parallel.
"""

import shutil
import time
from pathlib import Path

from omegaconf import DictConfig


def wait_for_disk(cfg: DictConfig) -> None:
    """Block until free disk space meets the configured threshold.

    Config keys:
        output_dir      Used to derive the filesystem to check (via parent)
                        and for the pipeline .done marker.
        min_disk_gb     Minimum free disk space in GB (default: 500).
    """
    output_dir = Path(cfg.output_dir)
    check_path = output_dir.parent
    check_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_gb = float(cfg.get("min_disk_gb", 500))

    while True:
        free_gb = shutil.disk_usage(check_path).free / (1024**3)
        if free_gb >= min_gb:
            print(f"Disk OK: {free_gb:.1f} GB free (>= {min_gb:.0f} GB)")
            return
        print(
            f"Waiting for disk space: {free_gb:.1f} GB free, need {min_gb:.0f} GB. "
            f"Retrying in 30s..."
        )
        time.sleep(30)
