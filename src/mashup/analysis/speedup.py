"""Convergence speedup table generation.

Generalized from latex/compute_speedup.py — dynamically discovers tasks
and sizes LaTeX table columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mashup.analysis.config import AnalysisConfig
from mashup.analysis.tex_utils import (
    aggregate_per_task,
    col_spec,
    fmt_speedup,
    write_table,
)


def compute_speedup(
    baseline_acc: float,
    merged_acc: float,
    checkpoints_df: pd.DataFrame,
) -> float:
    """Percentage of training at which mashup matched baseline accuracy."""
    if merged_acc >= baseline_acc:
        return 0.0
    ckpts = checkpoints_df.sort_values("step")
    if len(ckpts) == 0:
        return 100.0
    final_step = ckpts["step"].max()
    steps = np.concatenate([[0], ckpts["step"].values])
    accs = np.concatenate([[merged_acc], ckpts["accuracy"].values])
    for i in range(len(accs)):
        if accs[i] >= baseline_acc:
            if i == 0:
                return 0.0
            s0, a0 = steps[i - 1], accs[i - 1]
            s1, a1 = steps[i], accs[i]
            crossing = (
                s0 if a1 == a0 else s0 + (baseline_acc - a0) / (a1 - a0) * (s1 - s0)
            )
            return crossing / final_step * 100
    return 100.0


def compute_all_speedups(
    seed_dfs: list[pd.DataFrame],
    cfg: AnalysisConfig,
) -> dict[str, list[float]]:
    """Compute per-task speedup across seeds. Returns {task: [values]}."""
    all_speedups: dict[str, list[float]] = {t: [] for t in cfg.tasks}

    for df in seed_dfs:
        seed_vals: dict[str, float] = {}
        skip = False
        for task in cfg.tasks:
            bl = df[
                (df["name"] == cfg.method_name("from_scratch"))
                & (df["task"] == task)
                & (df["checkpoint"] == "final")
            ]
            mr = df[(df["name"] == cfg.method_name("merged")) & (df["task"] == task)]
            if len(bl) == 0 or len(mr) == 0:
                skip = True
                break
            ckpts = df[
                (df["name"] == cfg.method_name("mashup"))
                & (df["task"] == task)
                & (df["checkpoint"] != "final")
            ]
            seed_vals[task] = compute_speedup(
                bl["accuracy"].values[0], mr["accuracy"].values[0], ckpts
            )
        if skip:
            continue
        for task in cfg.tasks:
            all_speedups[task].append(seed_vals[task])

    return all_speedups


def aggregate_speedups(
    all_speedups: dict[str, list[float]],
    tasks: list[str],
    strict: bool = False,
) -> tuple[dict[str, float | None], dict[str, float], float | None, float]:
    """Aggregate speedup values into means/stds.

    Returns (means_dict, stds_dict, avg_mean, avg_std).
    In strict mode, tasks where any seed >= 100% are set to None.
    """
    if not strict:
        mean_list, std_list = aggregate_per_task(all_speedups, tasks)
        means = {t: mean_list[i] for i, t in enumerate(tasks)}
        stds_d = {t: std_list[i] for i, t in enumerate(tasks)}
        avg_mean = (
            mean_list[-1] if mean_list[-1] != 0.0 or any(mean_list[:-1]) else None
        )
        return means, stds_d, avg_mean, std_list[-1]

    means: dict[str, float | None] = {}
    stds: dict[str, float] = {}
    for task in tasks:
        vals = all_speedups[task]
        if any(v >= 100.0 for v in vals):
            means[task] = None
            stds[task] = 0.0
        else:
            means[task] = float(np.mean(vals)) if vals else None
            stds[task] = float(np.std(vals)) if len(vals) > 1 else 0.0

    n_seeds = (
        min(len(all_speedups[t]) for t in tasks)
        if all(all_speedups[t] for t in tasks)
        else 0
    )
    seed_avgs = []
    for i in range(n_seeds):
        sv = [all_speedups[t][i] for t in tasks if means[t] is not None]
        if sv:
            seed_avgs.append(np.mean(sv))

    avg_mean = float(np.mean(seed_avgs)) if seed_avgs else None
    avg_std = float(np.std(seed_avgs)) if len(seed_avgs) > 1 else 0.0
    return means, stds, avg_mean, avg_std


def generate_speedup_table(
    seed_dfs: list[pd.DataFrame],
    cfg: AnalysisConfig,
    out_dir: Path,
    prefix: str,
) -> None:
    """Generate speedup .tex + .pdf tables (normal + strict)."""
    all_speedups = compute_all_speedups(seed_dfs, cfg)
    n_seeds = (
        min(len(all_speedups[t]) for t in cfg.tasks)
        if all(all_speedups[t] for t in cfg.tasks)
        else 0
    )
    n_data = len(cfg.tasks) + 1
    cspec = col_spec(n_data, 1)
    header = cfg.task_header_cols
    model_name = cfg.experiment_name

    for strict in (False, True):
        means, stds, avg_mean, avg_std = aggregate_speedups(
            all_speedups, cfg.tasks, strict=strict
        )
        suffix = "_strict" if strict else ""
        title_suffix = " (strict)" if strict else ""
        strict_note = (
            " ``---'' if any seed did not reach parity."
            if strict
            else " Never-reached counted as 100\\%."
        )
        pm_note = " Values are mean $\\pm$ std across seeds." if n_seeds > 1 else ""

        vals = [fmt_speedup(means[t], stds[t], n_seeds) for t in cfg.tasks]
        vals.append(fmt_speedup(avg_mean, avg_std, n_seeds))

        tabular = (
            f"\\begin{{tabular}}{{{cspec}}}\n"
            f"        \\toprule\n"
            f"        & {header} & \\textbf{{Avg.}} \\\\\n"
            f"        \\midrule\n"
            f"        Training \\% & {' & '.join(vals)} \\\\\n"
            f"        \\bottomrule\n"
            f"    \\end{{tabular}}"
        )

        basename = f"{prefix}_speedup{suffix}"
        write_table(
            tabular,
            out_dir,
            basename,
            caption=(
                f"\\textbf{{{model_name} convergence"
                f" speedup{title_suffix}.}} Percentage of training at which"
                f" Mashup Learning matched converged from-scratch accuracy."
                f" Lower is better.{strict_note}{pm_note}"
            ),
            label=f"{prefix}_speedup{suffix}",
            comment=f"Convergence speedup table for {model_name}{title_suffix}",
        )
