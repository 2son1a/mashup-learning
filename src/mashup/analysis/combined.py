"""Combined cross-model tables (results, timing, speedup).

Generalized from latex/combined_results.py, combined_timing.py,
combined_speedup.py — dynamically discovers tasks and sizes columns.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from mashup.analysis.config import AnalysisConfig
from mashup.analysis.speedup import aggregate_speedups, compute_all_speedups
from mashup.analysis.tex_utils import (
    aggregate_per_task,
    col_spec,
    fmt_pct,
    fmt_speedup,
    fmt_time,
    write_table,
)
from mashup.analysis.timing import compute_timing_data

SETUP_LABELS = {"lora": "LoRA", "fullft": "Full FT"}


@dataclass
class ModelEntry:
    """One model's data for combined tables."""

    display_name: str
    setup: str
    cfg: AnalysisConfig
    seed_dfs: list[pd.DataFrame]
    seed_dirs: list[Path]
    n_seeds: int


# ---------------------------------------------------------------------------
# Helpers shared across combined tables
# ---------------------------------------------------------------------------


def _get_method_accs(
    df: pd.DataFrame,
    name: str,
    tasks: list[str],
    checkpoint: str | None = None,
) -> tuple[list[float], list[float], int]:
    """Compute per-task mean/std across seeds for a method.

    Returns (means_with_avg, stds_with_avg, n_seeds).
    """
    seeds = df["seed"].unique()
    n_seeds = len(seeds)
    seed_accs: dict[str, list[float]] = {t: [] for t in tasks}

    for seed in seeds:
        sdf = df[df["seed"] == seed]
        mask = sdf["name"] == name
        if checkpoint:
            mask = mask & (sdf["checkpoint"] == checkpoint)
        mdf = sdf[mask]
        for task in tasks:
            tdf = mdf[mdf["task"] == task]
            if len(tdf) > 0:
                seed_accs[task].append(tdf["accuracy"].values[0] * 100)

    means, stds = aggregate_per_task(seed_accs, tasks)
    return means, stds, n_seeds


# ---------------------------------------------------------------------------
# Combined results table
# ---------------------------------------------------------------------------


def generate_combined_results(
    models: list[ModelEntry],
    out_dir: Path,
    group: str,
    ref_cfg: AnalysisConfig,
) -> None:
    """Generate combined results table for a group of models."""
    is_all = group == "all"
    tasks = ref_cfg.tasks
    n_data = len(ref_cfg.tasks) + 1

    all_data: list[
        tuple[str, str, list[float], list[float], list[float], list[float], int]
    ] = []
    for me in models:
        df = pd.concat(me.seed_dfs, ignore_index=True)
        scratch_m, scratch_s, ns = _get_method_accs(
            df, me.cfg.method_name("from_scratch"), tasks, "final"
        )
        mashup_m, mashup_s, _ = _get_method_accs(
            df, me.cfg.method_name("mashup"), tasks, "final"
        )
        all_data.append(
            (me.display_name, me.setup, scratch_m, scratch_s, mashup_m, mashup_s, ns)
        )

    def _fmt(mean, std, ns, bold=False):
        if is_all or std == 0 or ns <= 1:
            s = f"{mean:.1f}"
        else:
            s = f"{mean:.1f} $\\pm$ {std:.1f}"
        return f"\\textbf{{{s}}}" if bold else s

    if is_all:
        model_groups: OrderedDict[str, list] = OrderedDict()
        for entry in all_data:
            name = entry[0]
            model_groups.setdefault(name, []).append(entry)

        lines = []
        for model_idx, (name, entries) in enumerate(model_groups.items()):
            n_rows = len(entries) * 2
            first_model_row = True
            for setup_idx, (_, setup, sm, ss, mm, ms, ns) in enumerate(entries):
                scratch_vals = [
                    _fmt(sm[i], ss[i], ns, bold=sm[i] > mm[i]) for i in range(n_data)
                ]
                mashup_vals = [
                    _fmt(mm[i], ms[i], ns, bold=mm[i] > sm[i]) for i in range(n_data)
                ]

                model_col = (
                    f"\\multirow{{{n_rows}}}{{*}}{{{name}}}" if first_model_row else ""
                )
                setup_col = f"\\multirow{{2}}{{*}}{{{setup}}}"

                lines.append(
                    f"    {model_col} & {setup_col} & From scratch"
                    f" & {' & '.join(scratch_vals)} \\\\"
                )
                lines.append(
                    f"    & & Mashup Learning & {' & '.join(mashup_vals)} \\\\"
                )
                first_model_row = False

                if setup_idx < len(entries) - 1:
                    cmidrule_end = n_data + 3
                    lines.append(f"    \\cmidrule{{2-{cmidrule_end}}}")

            if model_idx < len(model_groups) - 1:
                lines.append("    \\midrule")

        cspec = col_spec(n_data, 3)
        tabular = (
            f"\\begin{{tabular}}{{{cspec}}}\n"
            f"    \\toprule\n"
            f"    \\textbf{{Model}} & \\textbf{{Setup}}"
            f" & \\textbf{{Method}} & {ref_cfg.task_header_cols}"
            f" & \\textbf{{Avg.}} \\\\\n"
            f"    \\midrule\n" + "\n".join(lines) + "\n"
            "    \\bottomrule\n"
            "\\end{tabular}"
        )
        caption = (
            "\\textbf{Results across all models.}"
            " Accuracy (\\%) across benchmarks comparing training from"
            " scratch to Mashup Learning. Best per model+setup in bold."
        )
        label = "combined_results_all"

    else:
        lines = []
        for idx, (name, _setup, sm, ss, mm, ms, ns) in enumerate(all_data):
            scratch_vals = [
                _fmt(sm[i], ss[i], ns, bold=sm[i] > mm[i]) for i in range(n_data)
            ]
            mashup_vals = [
                _fmt(mm[i], ms[i], ns, bold=mm[i] > sm[i]) for i in range(n_data)
            ]

            lines.append(
                f"    \\multirow{{2}}{{*}}{{{name}}}"
                f" & From scratch & {' & '.join(scratch_vals)} \\\\"
            )
            lines.append(f"    & Mashup Learning & {' & '.join(mashup_vals)} \\\\")
            if idx < len(all_data) - 1:
                lines.append("    \\midrule")

        group_label = SETUP_LABELS.get(group, group)
        cspec = col_spec(n_data, 2)
        tabular = (
            f"\\begin{{tabular}}{{{cspec}}}\n"
            f"    \\toprule\n"
            f"    \\textbf{{Model}} & \\textbf{{Method}}"
            f" & {ref_cfg.task_header_cols} & \\textbf{{Avg.}} \\\\\n"
            f"    \\midrule\n" + "\n".join(lines) + "\n"
            "    \\bottomrule\n"
            "\\end{tabular}"
        )
        caption = (
            f"\\textbf{{{group_label} results across models.}}"
            f" Accuracy (\\%) across benchmarks comparing training from"
            f" scratch to Mashup Learning."
            f" Values are mean $\\pm$ std across seeds."
            f" Best per model in bold."
        )
        label = "combined_results"

    write_table(
        tabular,
        out_dir,
        "combined_results",
        caption=caption,
        label=label,
        comment=f"Combined results table ({group})",
    )


# ---------------------------------------------------------------------------
# Combined timing table
# ---------------------------------------------------------------------------


def generate_combined_timing(
    models: list[ModelEntry],
    out_dir: Path,
    group: str,
    ref_cfg: AnalysisConfig,
) -> None:
    """Generate combined timing table."""
    is_all = group == "all"
    tasks = ref_cfg.tasks
    n_data = len(ref_cfg.tasks) + 1

    all_timing = []
    for me in models:
        result = compute_timing_data(me.seed_dirs, me.cfg)
        if result is None:
            continue
        sm, mm, rm = result
        if not sm:
            continue
        all_timing.append((me.display_name, me.setup, sm, mm, rm))

    if not all_timing:
        print("  No timing data available for combined table")
        return

    tasks_avg = tasks + ["avg"]

    if is_all:
        model_groups: OrderedDict[str, list] = OrderedDict()
        for entry in all_timing:
            model_groups.setdefault(entry[0], []).append(entry)

        lines = []
        for model_idx, (name, entries) in enumerate(model_groups.items()):
            n_rows = len(entries) * 3
            first_model_row = True
            for setup_idx, (_, setup, sm, mm, rm) in enumerate(entries):
                sv = [fmt_time(sm[t]) for t in tasks_avg]
                mv = [fmt_time(mm[t]) for t in tasks_avg]
                rv = [fmt_pct(rm[t]) for t in tasks_avg]

                model_col = (
                    f"\\multirow{{{n_rows}}}{{*}}{{{name}}}" if first_model_row else ""
                )
                setup_col = f"\\multirow{{3}}{{*}}{{{setup}}}"

                lines.append(
                    f"    {model_col} & {setup_col}"
                    f" & From scratch (s) & {' & '.join(sv)} \\\\"
                )
                lines.append(f"    & & Mashup (s) & {' & '.join(mv)} \\\\")
                lines.append(f"    & & Ratio (\\%) & {' & '.join(rv)} \\\\")
                first_model_row = False

                if setup_idx < len(entries) - 1:
                    cmidrule_end = n_data + 3
                    lines.append(f"    \\cmidrule{{2-{cmidrule_end}}}")

            if model_idx < len(model_groups) - 1:
                lines.append("    \\midrule")

        cspec = col_spec(n_data, 3)
        header = (
            f"    \\textbf{{Model}} & \\textbf{{Setup}} &"
            f" & {ref_cfg.task_header_cols} & \\textbf{{Avg.}} \\\\"
        )
    else:
        lines = []
        for idx, (name, _setup, sm, mm, rm) in enumerate(all_timing):
            sv = [fmt_time(sm[t]) for t in tasks_avg]
            mv = [fmt_time(mm[t]) for t in tasks_avg]
            rv = [fmt_pct(rm[t]) for t in tasks_avg]

            lines.append(
                f"    \\multirow{{3}}{{*}}{{{name}}}"
                f" & From scratch (s) & {' & '.join(sv)} \\\\"
            )
            lines.append(f"    & Mashup (s) & {' & '.join(mv)} \\\\")
            lines.append(f"    & Ratio (\\%) & {' & '.join(rv)} \\\\")
            if idx < len(all_timing) - 1:
                lines.append("    \\midrule")

        cspec = col_spec(n_data, 2)
        header = (
            f"    \\textbf{{Model}} &"
            f" & {ref_cfg.task_header_cols} & \\textbf{{Avg.}} \\\\"
        )

    tabular = (
        f"\\begin{{tabular}}{{{cspec}}}\n"
        f"    \\toprule\n"
        f"{header}\n"
        f"    \\midrule\n" + "\n".join(lines) + "\n"
        "    \\bottomrule\n"
        "\\end{tabular}"
    )

    group_label = SETUP_LABELS.get(group, group)
    write_table(
        tabular,
        out_dir,
        "combined_timing",
        caption=(
            f"\\textbf{{Wall-clock training time across"
            f" {'all ' if is_all else ''}{group_label} models.}}"
            f" Training time in seconds at the best learning rate."
            f" Mashup includes relevance estimation and merge overhead,"
            f" with training time scaled by convergence percentage."
            f" Ratio $<100\\%$ means Mashup is faster."
        ),
        label="combined_timing_all" if is_all else "combined_timing",
        comment=f"Combined timing table ({group})",
    )


# ---------------------------------------------------------------------------
# Combined speedup table
# ---------------------------------------------------------------------------


class SpeedupData(NamedTuple):
    """Aggregated speedup results for one model (normal + strict)."""

    means: dict[str, float | None]
    stds: dict[str, float]
    strict_means: dict[str, float | None]
    strict_stds: dict[str, float]
    avg_mean: float | None
    avg_std: float
    strict_avg_mean: float | None
    strict_avg_std: float
    n_seeds: int


def _compute_model_speedup(me: ModelEntry) -> SpeedupData:
    """Compute speedup data for one model across seeds (normal + strict)."""
    all_speedups = compute_all_speedups(me.seed_dfs, me.cfg)
    tasks = me.cfg.tasks
    actual_n_seeds = (
        min(len(all_speedups[t]) for t in tasks)
        if all(all_speedups[t] for t in tasks)
        else 0
    )

    means, stds, avg_mean, avg_std = aggregate_speedups(
        all_speedups, tasks, strict=False
    )
    strict_means, strict_stds, strict_avg_mean, strict_avg_std = aggregate_speedups(
        all_speedups, tasks, strict=True
    )

    return SpeedupData(
        means=means,
        stds=stds,
        strict_means=strict_means,
        strict_stds=strict_stds,
        avg_mean=avg_mean,
        avg_std=avg_std,
        strict_avg_mean=strict_avg_mean,
        strict_avg_std=strict_avg_std,
        n_seeds=actual_n_seeds,
    )


def generate_combined_speedup(
    models: list[ModelEntry],
    out_dir: Path,
    group: str,
    ref_cfg: AnalysisConfig,
) -> None:
    """Generate combined speedup tables (normal + strict)."""
    is_all = group == "all"
    tasks = ref_cfg.tasks
    n_data = len(ref_cfg.tasks) + 1

    all_data: list[tuple[str, str, SpeedupData]] = []
    for me in models:
        all_data.append((me.display_name, me.setup, _compute_model_speedup(me)))

    for strict in (False, True):
        suffix = "_strict" if strict else ""
        title_suffix = " (strict)" if strict else ""
        strict_note = (
            " ``---'' if any seed did not reach parity."
            if strict
            else " Never-reached counted as 100\\%."
        )

        if is_all:
            model_groups: OrderedDict[str, list[tuple[str, str, SpeedupData]]] = (
                OrderedDict()
            )
            for entry in all_data:
                model_groups.setdefault(entry[0], []).append(entry)

            lines = []
            for model_idx, (name, entries) in enumerate(model_groups.items()):
                n_rows = len(entries)
                first_model_row = True
                for setup_idx, (_, setup, sd) in enumerate(entries):
                    use_m = sd.strict_means if strict else sd.means
                    use_s = sd.strict_stds if strict else sd.stds
                    use_am = sd.strict_avg_mean if strict else sd.avg_mean
                    use_as = sd.strict_avg_std if strict else sd.avg_std

                    vals = [fmt_speedup(use_m[t], use_s[t], sd.n_seeds) for t in tasks]
                    vals.append(fmt_speedup(use_am, use_as, sd.n_seeds))

                    model_col = (
                        f"\\multirow{{{n_rows}}}{{*}}{{{name}}}"
                        if first_model_row
                        else ""
                    )
                    lines.append(f"    {model_col} & {setup} & {' & '.join(vals)} \\\\")
                    first_model_row = False

                    if setup_idx < len(entries) - 1:
                        cmidrule_end = n_data + 2
                        lines.append(f"    \\cmidrule{{2-{cmidrule_end}}}")

                if model_idx < len(model_groups) - 1:
                    lines.append("    \\midrule")

            cspec = col_spec(n_data, 2)
            header = (
                f"    \\textbf{{Model}} & \\textbf{{Setup}}"
                f" & {ref_cfg.task_header_cols} & \\textbf{{Avg.}} \\\\"
            )

        else:
            lines = []
            for idx, (name, _setup, sd) in enumerate(all_data):
                use_m = sd.strict_means if strict else sd.means
                use_s = sd.strict_stds if strict else sd.stds
                use_am = sd.strict_avg_mean if strict else sd.avg_mean
                use_as = sd.strict_avg_std if strict else sd.avg_std

                vals = [fmt_speedup(use_m[t], use_s[t], sd.n_seeds) for t in tasks]
                vals.append(fmt_speedup(use_am, use_as, sd.n_seeds))
                lines.append(f"    {name} & {' & '.join(vals)} \\\\")
                if idx < len(all_data) - 1:
                    lines.append("    \\midrule")

            cspec = col_spec(n_data, 1)
            header = (
                f"    \\textbf{{Model}}"
                f" & {ref_cfg.task_header_cols} & \\textbf{{Avg.}} \\\\"
            )

        tabular = (
            f"\\begin{{tabular}}{{{cspec}}}\n"
            f"    \\toprule\n"
            f"{header}\n"
            f"    \\midrule\n" + "\n".join(lines) + "\n"
            "    \\bottomrule\n"
            "\\end{tabular}"
        )

        group_label = SETUP_LABELS.get(group, group)
        basename = f"combined_speedup{suffix}"
        write_table(
            tabular,
            out_dir,
            basename,
            caption=(
                f"\\textbf{{{group_label} convergence speedup across"
                f" models{title_suffix}.}} Percentage of training at which"
                f" Mashup Learning matched converged from-scratch accuracy."
                f" Lower is better.{strict_note}"
                f" Values are mean $\\pm$ std across seeds."
            ),
            label=(
                f"combined_speedup_all{suffix}"
                if is_all
                else f"combined_speedup{suffix}"
            ),
            comment=f"Combined speedup table ({group}){title_suffix}",
        )
