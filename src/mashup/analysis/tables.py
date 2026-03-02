"""Results table and LR-sweep table generation.

Generalized from latex/extract_data.py — auto-discovers tasks and LRs,
dynamically sizes LaTeX columns and headers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mashup.analysis.config import AnalysisConfig, lr_to_tex
from mashup.analysis.tex_utils import aggregate_per_task, col_spec, fmt_val, write_table

# ---------------------------------------------------------------------------
# LR sensitivity CSV extraction
# ---------------------------------------------------------------------------


def extract_lr_sensitivity(
    df: pd.DataFrame,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """Extract per-seed LR sweep data for both from-scratch and mashup sweeps.

    Returns a DataFrame with columns: method, dataset, lr, accuracy, seed.
    """
    records: list[dict] = []
    for role_key, sweep_name in cfg.sweeps.items():
        sweep = df[(df["name"] == sweep_name) & (df["checkpoint"] == "final")].copy()
        for seed_name in df["seed"].unique():
            seed_sweep = sweep[sweep["seed"] == seed_name]
            for task in cfg.tasks:
                for _, row in seed_sweep[seed_sweep["task"] == task].iterrows():
                    lr = row["lr"]
                    if pd.isna(lr):
                        continue
                    records.append(
                        {
                            "method": role_key,
                            "dataset": task,
                            "lr": float(lr),
                            "accuracy": row["accuracy"],
                            "seed": seed_name,
                        }
                    )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def _collect_method_data(
    df: pd.DataFrame,
    cfg: AnalysisConfig,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Collect per-task mean and std across seeds for each method.

    Returns (means_by_display, stds_by_display) where each value list has
    len(tasks) + 1 entries (tasks + avg).
    """
    means: dict[str, list[float]] = {}
    stds: dict[str, list[float]] = {}

    for minfo in cfg.methods.values():
        seed_accs: dict[str, list[float]] = {t: [] for t in cfg.tasks}
        for seed_name in df["seed"].unique():
            sdf = df[df["seed"] == seed_name]
            mask = sdf["name"] == minfo.name
            if minfo.checkpoint is not None:
                mask = mask & (sdf["checkpoint"] == minfo.checkpoint)
            mdf = sdf[mask]
            for task in cfg.tasks:
                tdf = mdf[mdf["task"] == task]
                if len(tdf) > 0:
                    seed_accs[task].append(tdf["accuracy"].values[0] * 100)

        means[minfo.display], stds[minfo.display] = aggregate_per_task(
            seed_accs, cfg.tasks
        )

    return means, stds


def _find_col_best(
    means_by_key: dict[str | int, list[float]],
) -> dict[str | int, list[bool]]:
    """Column-wise argmax: marks the best key per column with True."""
    keys = list(means_by_key.keys())
    n_cols = len(next(iter(means_by_key.values())))
    flags = {k: [False] * n_cols for k in keys}
    for col in range(n_cols):
        best_val, best_key = -1.0, None
        for k in keys:
            if means_by_key[k][col] > best_val:
                best_val = means_by_key[k][col]
                best_key = k
        if best_key is not None:
            flags[best_key][col] = True
    return flags


def generate_results_table(
    df: pd.DataFrame,
    cfg: AnalysisConfig,
    out_dir: Path,
    prefix: str,
) -> None:
    """Generate the main results .tex + .pdf table."""
    n_seeds = df["seed"].nunique()
    n_seeds_label = f"{n_seeds} seeds" if n_seeds > 1 else "1 seed"
    table_means, table_stds = _collect_method_data(df, cfg)
    method_displays = [m.display for m in cfg.methods.values()]
    best_flags = _find_col_best(table_means)

    header = cfg.task_header_cols
    n_data = len(cfg.tasks) + 1
    cspec = col_spec(n_data, 1)

    lines = []
    for display in method_displays:
        m = table_means[display]
        s = table_stds[display]
        f = best_flags[display]
        cells = [fmt_val(m[i], s[i], f[i], n_seeds) for i in range(len(m))]
        pad = " " * max(1, 22 - len(display))
        lines.append(f"    {display}{pad} & {' & '.join(cells)} \\\\")

    tabular = (
        f"\\begin{{tabular}}{{{cspec}}}\n"
        f"    \\toprule\n"
        f"    \\textbf{{Method}} & {header} & \\textbf{{Avg.}} \\\\\n"
        f"    \\midrule\n" + "\n".join(lines) + "\n"
        "    \\bottomrule\n"
        "\\end{tabular}"
    )

    model_name = cfg.experiment_name
    pm_note = " $\\pm$ std across seeds" if n_seeds > 1 else ""
    write_table(
        tabular,
        out_dir,
        f"{prefix}_results",
        caption=(
            f"\\textbf{{{model_name} results ({n_seeds_label}).}}"
            f" Accuracy (\\%) across benchmarks. Values are mean{pm_note}."
            f" Best per column in bold."
        ),
        label=f"{prefix}_results",
        comment=f"Main results table for {model_name} experiments ({n_seeds_label})",
    )


# ---------------------------------------------------------------------------
# LR sweep table
# ---------------------------------------------------------------------------


def generate_lr_sweep_table(
    lr_df: pd.DataFrame,
    df: pd.DataFrame,
    cfg: AnalysisConfig,
    out_dir: Path,
    prefix: str,
) -> None:
    """Generate the LR sweep .tex + .pdf table."""
    n_seeds = df["seed"].nunique()
    n_seeds_label = f"{n_seeds} seeds" if n_seeds > 1 else "1 seed"
    n_lrs = len(cfg.lrs)
    lr_tex_labels = [lr_to_tex(lr) for lr in cfg.lrs]
    header = cfg.task_header_cols
    n_data = len(cfg.tasks) + 1

    sweep_means: dict[str, dict[int, list[float]]] = {}
    sweep_stds: dict[str, dict[int, list[float]]] = {}

    for role_key in cfg.sweeps:
        mdf = lr_df[lr_df["method"] == role_key]
        sweep_means[role_key] = {}
        sweep_stds[role_key] = {}
        for lr_idx, lr in enumerate(cfg.lrs):
            lrdf = mdf[mdf["lr"] == lr]
            seed_accs: dict[str, list[float]] = {t: [] for t in cfg.tasks}
            for seed_name in df["seed"].unique():
                seed_lrdf = lrdf[lrdf["seed"] == seed_name]
                for task in cfg.tasks:
                    tdf = seed_lrdf[seed_lrdf["dataset"] == task]
                    if len(tdf) > 0:
                        seed_accs[task].append(tdf["accuracy"].values[0] * 100)
            sweep_means[role_key][lr_idx], sweep_stds[role_key][lr_idx] = (
                aggregate_per_task(seed_accs, cfg.tasks)
            )

    cspec = col_spec(n_data, 2)

    sweep_lines: list[str] = []
    for role_idx, role_key in enumerate(cfg.sweeps):
        best = _find_col_best(sweep_means[role_key])
        label = cfg.methods[role_key].display if role_key in cfg.methods else role_key
        for lr_idx in range(n_lrs):
            row_prefix = (
                f"    \\multirow{{{n_lrs}}}{{*}}{{{label}}}" if lr_idx == 0 else "   "
            )
            cells = [
                fmt_val(
                    sweep_means[role_key][lr_idx][i],
                    sweep_stds[role_key][lr_idx][i],
                    best[lr_idx][i],
                    n_seeds,
                )
                for i in range(n_data)
            ]
            sweep_lines.append(
                f"{row_prefix} & {lr_tex_labels[lr_idx]} & {' & '.join(cells)} \\\\"
            )
        if role_idx < len(cfg.sweeps) - 1:
            sweep_lines.append("    \\midrule")

    tabular = (
        f"\\begin{{tabular}}{{{cspec}}}\n"
        f"    \\toprule\n"
        f"    \\textbf{{Method}} & \\textbf{{LR}} & {header} & \\textbf{{Avg.}} \\\\\n"
        f"    \\midrule\n" + "\n".join(sweep_lines) + "\n"
        "    \\bottomrule\n"
        "\\end{tabular}"
    )

    model_name = cfg.experiment_name
    pm_note = " $\\pm$ std across seeds" if n_seeds > 1 else ""
    write_table(
        tabular,
        out_dir,
        f"{prefix}_lr_sweep",
        caption=(
            f"\\textbf{{Learning rate sweep on {model_name}"
            f" ({n_seeds_label}).}} Final accuracy (\\%) for each LR."
            f" Values are mean{pm_note}."
            f" Best LR per method per column in bold."
        ),
        label=f"{prefix}_lr_sweep",
        comment=f"Learning rate sweep table for {model_name} ({n_seeds_label})",
    )
