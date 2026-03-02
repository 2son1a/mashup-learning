"""
Summary table from evaluation results.

Collects eval_results.json files matching directory patterns and builds
a summary CSV.  Checkpoint-level results (from ``checkpoint-*``
subdirectories) are included by default (``include_checkpoints: false``
to disable per entry).
"""

import json
import re
import warnings
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig


def build_summary(cfg: DictConfig) -> None:
    """Collect evaluation results and write a summary table.

    Config keys::

        results        list of {name, pattern} dicts – each pattern is a
                       directory-name glob matched inside run_dir.
                       Optional flags per entry:
                         self_eval_only: true     keep only self-evaluations
                         include_checkpoints: false   skip checkpoint subdirs
        output_dir     where to write summary.csv (also used to derive run_dir)
    """
    output_dir = Path(cfg.output_dir)
    run_dir = output_dir.parent
    rows: list[dict] = []

    for entry in cfg.results:
        name = str(entry["name"])
        pattern = str(entry["pattern"])
        self_eval_only = bool(entry.get("self_eval_only", False))
        include_checkpoints = bool(entry.get("include_checkpoints", True))

        matched_dirs = sorted(run_dir.glob(pattern))
        if not matched_dirs:
            warnings.warn(
                f"No directories matching '{pattern}' in {run_dir}", stacklevel=2
            )
            continue

        for eval_dir in matched_dirs:
            # Collect all eval_results.json: top-level + checkpoint subdirs
            result_files = [eval_dir / "eval_results.json"]
            if include_checkpoints:
                result_files = (
                    sorted(eval_dir.glob("checkpoint-*/eval_results.json"))
                    + result_files
                )

            for results_file in result_files:
                if not results_file.exists():
                    continue
                data = json.loads(results_file.read_text())

                if self_eval_only:
                    if (
                        Path(data.get("dataset", "")).name
                        not in Path(data.get("model", "")).name
                    ):
                        continue

                # Parse learning rate from directory name (e.g. eval_..._lr0.0001)
                lr_match = re.search(r"_lr([\d.eE\-]+)", eval_dir.name)

                row = {
                    "name": name,
                    "dataset": data.get("dataset", ""),
                    "lr": float(lr_match.group(1)) if lr_match else None,
                    "accuracy": data.get("accuracy"),
                    "perplexity": data.get("perplexity"),
                }
                if "checkpoint" in data:
                    row["checkpoint"] = data["checkpoint"]
                    row["step"] = data.get("step")
                rows.append(row)

    if not rows:
        print("  No evaluation results found for any pattern.")
        return

    has_checkpoints = any("checkpoint" in r for r in rows)
    has_lr = any(r.get("lr") is not None for r in rows)
    base_cols = ["name", "dataset"]
    if has_lr:
        base_cols.append("lr")
    if has_checkpoints:
        base_cols += ["checkpoint", "step"]
    columns = base_cols + ["accuracy", "perplexity"]
    df = pd.DataFrame(rows, columns=columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 72}")
    print(f"  Summary  ({len(df)} rows)")
    print(f"{'=' * 72}")
    print(df.to_string(index=False))
    print(f"{'=' * 72}")
    print(f"\n  Saved to {csv_path}\n")
