"""Shared LaTeX utilities and aggregation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def aggregate_per_task(
    per_seed: dict[str, list[float]],
    tasks: list[str],
) -> tuple[list[float], list[float]]:
    """Aggregate per-seed per-task values into means and stds.

    Returns (means, stds) where each list has len(tasks) + 1 entries
    (per-task values followed by avg). The avg std is computed from
    seed-level averages (mean across tasks for each seed).
    """
    n_seeds = (
        min(len(per_seed[t]) for t in tasks) if all(per_seed[t] for t in tasks) else 0
    )

    means, stds = [], []
    for task in tasks:
        vals = per_seed[task]
        means.append(float(np.mean(vals)) if vals else 0.0)
        stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

    avg_mean = float(np.mean(means))
    if n_seeds > 1:
        seed_avgs = [
            float(np.mean([per_seed[t][i] for t in tasks])) for i in range(n_seeds)
        ]
        avg_std = float(np.std(seed_avgs))
    else:
        avg_std = 0.0

    means.append(avg_mean)
    stds.append(avg_std)
    return means, stds


def col_spec(n_data_cols: int, n_label_cols: int = 1) -> str:
    """Build a LaTeX tabular column spec: label columns + data columns."""
    return "l" * n_label_cols + "c" * n_data_cols


def fmt_val(mean: float, std: float, bold: bool = False, n_seeds: int = 1) -> str:
    """Format a value with optional ± std and bold."""
    if std > 0 and n_seeds > 1:
        s = f"{mean:.1f} $\\pm$ {std:.1f}"
    else:
        s = f"{mean:.1f}"
    return f"\\textbf{{{s}}}" if bold else s


def fmt_time(v: float | None) -> str:
    return "---" if v is None else f"{v:.0f}"


def fmt_pct(v: float | None) -> str:
    return "---" if v is None else f"{v * 100:.0f}\\%"


def fmt_speedup(mean: float | None, std: float = 0.0, n_seeds: int = 1) -> str:
    if mean is None:
        return "---"
    base = "0" if mean == 0.0 and (std == 0.0) else f"{mean:.1f}"
    if std > 0 and n_seeds > 1:
        return f"{base} $\\pm$ {std:.1f}"
    return base


def write_table(
    tabular: str,
    out_dir: Path,
    basename: str,
    caption: str,
    label: str,
    comment: str | None = None,
) -> None:
    """Write a ``table*`` environment wrapping *tabular* to a .tex file."""
    comment_line = f"% {comment}\n" if comment else ""
    tex = (
        f"{comment_line}"
        f"\\begin{{table*}}[t]\n"
        f"    \\centering\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{tab:{label}}}\n"
        f"    \\resizebox{{\\linewidth}}{{!}}{{\n"
        f"    {tabular}\n"
        f"    }}\n"
        f"\\end{{table*}}\n"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{basename}.tex"
    path.write_text(tex)
    print(f"Saved {path}")
