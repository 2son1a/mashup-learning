"""Wall-clock timing comparison table generation.

Generalized from latex/compute_timing.py — dynamically discovers tasks
and sizes LaTeX table columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mashup.analysis.config import AnalysisConfig
from mashup.analysis.speedup import compute_speedup
from mashup.analysis.tex_utils import col_spec, fmt_pct, fmt_time, write_table


def _compute_convergence_pct(
    summary_df: pd.DataFrame,
    task: str,
    cfg: AnalysisConfig,
) -> float:
    """Convergence percentage for a task, using raw summary data with
    ``data/processed/`` prefix on the dataset column.
    """
    ds = f"data/processed/{task}"
    bl = summary_df[
        (summary_df["name"] == cfg.method_name("from_scratch"))
        & (summary_df["dataset"] == ds)
        & (summary_df["checkpoint"] == "final")
    ]
    mr = summary_df[
        (summary_df["name"] == cfg.method_name("merged"))
        & (summary_df["dataset"] == ds)
    ]
    if len(bl) == 0 or len(mr) == 0:
        return 100.0
    ckpts = summary_df[
        (summary_df["name"] == cfg.method_name("mashup"))
        & (summary_df["dataset"] == ds)
        & (summary_df["checkpoint"] != "final")
    ]
    return compute_speedup(bl["accuracy"].values[0], mr["accuracy"].values[0], ckpts)


def _find_best_lr(
    summary_df: pd.DataFrame,
    sweep_name: str,
    task: str,
) -> str | None:
    rows = summary_df[
        (summary_df["name"] == sweep_name)
        & (summary_df["dataset"] == f"data/processed/{task}")
        & (summary_df["checkpoint"] == "final")
    ]
    if len(rows) == 0:
        return None
    return str(rows.loc[rows["accuracy"].idxmax(), "lr"])


def _get_train_time(
    timing_df: pd.DataFrame,
    stage: str,
    task: str,
    lr: str | None,
) -> float | None:
    if lr is None:
        return None
    job = f"{stage}_{task}_{lr}"
    subjob = f"{job}_train"
    row = timing_df[
        (timing_df["stage"] == stage)
        & (timing_df["job"] == job)
        & (timing_df["subjob"] == subjob)
        & (timing_df["job_type"] == "train")
    ]
    return row["elapsed_s"].values[0] if len(row) > 0 else None


def _get_relevance_time(
    timing_df: pd.DataFrame,
    stage: str,
    target_task: str,
) -> float | None:
    rows = timing_df[timing_df["stage"] == stage]
    matched = rows[rows["job"].str.endswith(f"_{target_task}")]
    return matched["elapsed_s"].min() if len(matched) > 0 else None


def _get_merge_time(
    timing_df: pd.DataFrame,
    stage: str,
    task: str,
) -> float | None:
    job = f"{stage}_{task}"
    row = timing_df[(timing_df["stage"] == stage) & (timing_df["job"] == job)]
    return row["elapsed_s"].values[0] if len(row) > 0 else None


def _has_cached(timing_df: pd.DataFrame, stages: list[str]) -> bool:
    rows = timing_df[timing_df["stage"].isin(stages)]
    return bool((rows["cached"] == True).any())  # noqa: E712


def compute_timing_data(
    seed_dirs: list[Path],
    cfg: AnalysisConfig,
) -> (
    tuple[dict[str, float | None], dict[str, float | None], dict[str, float | None]]
    | None
):
    """Compute per-task timing data across seeds.

    Returns (scratch_means, mashup_means, ratio_means) dicts mapping
    task (+ "avg") to mean values, or None if no valid seeds.
    """
    ts = cfg.timing_stages
    if not ts:
        return None

    train_stage = ts["train"]
    relevance_stage = ts["relevance"]
    merge_stage = ts["merge"]
    train_merged_stage = ts["train_merged"]
    required = [train_stage, relevance_stage, merge_stage, train_merged_stage]

    all_scratch: dict[str, list[float | None]] = {t: [] for t in cfg.tasks}
    all_mashup: dict[str, list[float | None]] = {t: [] for t in cfg.tasks}
    all_ratio: dict[str, list[float | None]] = {t: [] for t in cfg.tasks}
    valid_seeds = 0

    for seed_dir in seed_dirs:
        summary_path = seed_dir / "summary" / "summary.csv"
        timing_path = seed_dir / "timing.csv"
        if not summary_path.exists() or not timing_path.exists():
            continue

        summary = pd.read_csv(summary_path)
        timing = pd.read_csv(timing_path)

        if _has_cached(timing, required):
            continue

        valid_seeds += 1

        for task in cfg.tasks:
            best_scratch = _find_best_lr(summary, cfg.sweep_for("from_scratch"), task)
            best_mashup = _find_best_lr(summary, cfg.sweep_for("mashup"), task)

            scratch_t = _get_train_time(timing, train_stage, task, best_scratch)
            rel_t = _get_relevance_time(timing, relevance_stage, task)
            merge_t = _get_merge_time(timing, merge_stage, task)
            mashup_train_t = _get_train_time(
                timing, train_merged_stage, task, best_mashup
            )
            conv = _compute_convergence_pct(summary, task, cfg)

            if all(v is not None for v in [scratch_t, rel_t, merge_t, mashup_train_t]):
                eff = mashup_train_t * (conv / 100)
                mashup_t = rel_t + merge_t + eff
                ratio = mashup_t / scratch_t if scratch_t > 0 else None
            else:
                mashup_t = ratio = None

            all_scratch[task].append(scratch_t)
            all_mashup[task].append(mashup_t)
            all_ratio[task].append(ratio)

    if valid_seeds == 0:
        return None

    scratch_m: dict[str, float | None] = {}
    mashup_m: dict[str, float | None] = {}
    ratio_m: dict[str, float | None] = {}

    for task in cfg.tasks:
        sv = [v for v in all_scratch[task] if v is not None]
        mv = [v for v in all_mashup[task] if v is not None]
        rv = [v for v in all_ratio[task] if v is not None]
        scratch_m[task] = float(np.mean(sv)) if sv else None
        mashup_m[task] = float(np.mean(mv)) if mv else None
        ratio_m[task] = float(np.mean(rv)) if rv else None

    for d, all_d in [
        (scratch_m, all_scratch),
        (mashup_m, all_mashup),
        (ratio_m, all_ratio),
    ]:
        seed_avgs = []
        for i in range(valid_seeds):
            vals = [all_d[t][i] for t in cfg.tasks if all_d[t][i] is not None]
            if vals:
                seed_avgs.append(np.mean(vals))
        d["avg"] = float(np.mean(seed_avgs)) if seed_avgs else None

    return scratch_m, mashup_m, ratio_m


def generate_timing_table(
    seed_dirs: list[Path],
    cfg: AnalysisConfig,
    out_dir: Path,
    prefix: str,
) -> None:
    """Generate wall-clock timing .tex table."""
    result = compute_timing_data(seed_dirs, cfg)
    if result is None:
        print("  Skipping timing table (no timing_stages or no valid seeds)")
        return

    scratch_m, mashup_m, ratio_m = result
    tasks_avg = cfg.tasks + ["avg"]
    sv_str = [fmt_time(scratch_m[t]) for t in tasks_avg]
    mv_str = [fmt_time(mashup_m[t]) for t in tasks_avg]
    rv_str = [fmt_pct(ratio_m[t]) for t in tasks_avg]

    n_data = len(cfg.tasks) + 1
    cspec = col_spec(n_data, 1)
    header = cfg.task_header_cols

    tabular = (
        f"\\begin{{tabular}}{{{cspec}}}\n"
        f"    \\toprule\n"
        f"    & {header} & \\textbf{{Avg.}} \\\\\n"
        f"    \\midrule\n"
        f"    From scratch (s) & {' & '.join(sv_str)} \\\\\n"
        f"    Mashup (s) & {' & '.join(mv_str)} \\\\\n"
        f"    \\midrule\n"
        f"    Ratio (\\%) & {' & '.join(rv_str)} \\\\\n"
        f"    \\bottomrule\n"
        f"\\end{{tabular}}"
    )

    model_name = cfg.experiment_name
    n_label = f" ({len(seed_dirs)} seeds)" if len(seed_dirs) > 1 else ""
    write_table(
        tabular,
        out_dir,
        f"{prefix}_timing",
        caption=(
            f"\\textbf{{{model_name} wall-clock training"
            f" time{n_label}.}} Training time in seconds at the best learning"
            f" rate per task. Mashup includes relevance estimation and merge"
            f" overhead, with training time scaled by convergence percentage."
            f" Ratio $<100\\%$ means Mashup is faster."
        ),
        label=f"{prefix}_timing",
        comment=f"Wall-clock timing table for {model_name}{n_label}",
    )
