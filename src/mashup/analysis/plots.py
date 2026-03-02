"""Training curve and LR sensitivity plot generation.

Generalized from latex/plot_training_curves.py and
latex/plot_lr_sensitivity.py — auto-sizes grids based on task count.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mashup.analysis.config import AnalysisConfig, lr_to_plot_label

matplotlib.use("Agg")


def _save_plot(out_dir: Path, name: str) -> None:
    """Save current figure as PDF + PNG then close."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [(".pdf", 300), (".png", 200)]:
        path = out_dir / f"{name}{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
        print(f"Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Shared style setup
# ---------------------------------------------------------------------------

_STYLE_APPLIED = False


def _apply_style() -> None:
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    sns.set_theme(style="whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelsize": 14,
            "font.size": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.3,
            "axes.linewidth": 0.4,
            "axes.edgecolor": (0, 0, 0, 0.3),
        }
    )
    _STYLE_APPLIED = True


def _get_colors() -> tuple:
    husl = sns.color_palette("husl", 8)
    return husl[5], husl[0]  # C_SCRATCH, C_MASHUP


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def _get_best_sweep_group(
    seed_df: pd.DataFrame,
    sweep_name: str,
    best_df: pd.DataFrame,
    tasks: list[str],
) -> pd.DataFrame:
    sweep = seed_df[seed_df["name"] == sweep_name].copy()
    records = []
    for task in tasks:
        task_sweep = sweep[sweep["task"] == task].reset_index(drop=True)
        best_final = best_df[best_df["task"] == task]
        if len(best_final) == 0:
            continue
        best_final_acc = best_final["accuracy"].values[0]

        finals = task_sweep[task_sweep["checkpoint"] == "final"]
        n_groups = len(finals)
        if n_groups == 0:
            continue
        group_size = len(task_sweep) // n_groups

        best_group, best_diff = None, float("inf")
        for g in range(n_groups):
            group = task_sweep.iloc[g * group_size : (g + 1) * group_size]
            final_row = group[group["checkpoint"] == "final"]
            if len(final_row) == 0:
                continue
            diff = abs(final_row["accuracy"].values[0] - best_final_acc)
            if diff < best_diff:
                best_diff = diff
                best_group = group

        if best_group is not None:
            checkpoints = best_group[best_group["checkpoint"] != "final"].sort_values(
                "step"
            )
            records.append(checkpoints)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def generate_training_curves(
    seed_dfs: list[pd.DataFrame],
    cfg: AnalysisConfig,
    out_dir: Path,
    prefix: str,
) -> None:
    """Generate a training curves plot (NxM grid)."""
    _apply_style()
    c_scratch, c_mashup = _get_colors()
    n_seeds = len(seed_dfs)
    rows, cols = cfg.plot_grid

    seed_data = []
    for sdf in seed_dfs:
        base = sdf[sdf["name"] == "base"].set_index("task")["accuracy"].to_dict()
        merged = (
            sdf[sdf["name"] == cfg.method_name("merged")]
            .set_index("task")["accuracy"]
            .to_dict()
        )

        trained_scratch = sdf[
            (sdf["name"] == cfg.method_name("from_scratch"))
            & (sdf["checkpoint"] == "final")
        ]
        trained_mashup = sdf[
            (sdf["name"] == cfg.method_name("mashup")) & (sdf["checkpoint"] == "final")
        ]

        from_scratch = _get_best_sweep_group(
            sdf, cfg.sweep_for("from_scratch"), trained_scratch, cfg.tasks
        )
        from_merged = _get_best_sweep_group(
            sdf, cfg.sweep_for("mashup"), trained_mashup, cfg.tasks
        )

        seed_data.append(
            {
                "base": base,
                "merged": merged,
                "trained_scratch": trained_scratch,
                "from_scratch": from_scratch,
                "from_merged": from_merged,
            }
        )

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if len(cfg.tasks) == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for i, task in enumerate(cfg.tasks):
        ax = axes_flat[i]
        label = cfg.get_task_label(task)

        scratch_curves, mashup_curves, converged_accs = [], [], []

        for sd in seed_data:
            base_acc = sd["base"].get(task, 0) * 100
            merged_acc = sd["merged"].get(task, 0) * 100

            ts = sd["from_scratch"]
            ts_task = (
                ts[ts["task"] == task].sort_values("step")
                if len(ts) > 0
                else pd.DataFrame()
            )
            if len(ts_task) > 0:
                steps = np.concatenate([[0], ts_task["step"].values])
                accs = np.concatenate([[base_acc], ts_task["accuracy"].values * 100])
                scratch_curves.append((steps, accs))

            tm = sd["from_merged"]
            tm_task = (
                tm[tm["task"] == task].sort_values("step")
                if len(tm) > 0
                else pd.DataFrame()
            )
            if len(tm_task) > 0:
                steps = np.concatenate([[0], tm_task["step"].values])
                accs = np.concatenate([[merged_acc], tm_task["accuracy"].values * 100])
                mashup_curves.append((steps, accs))

            tl_task = sd["trained_scratch"][sd["trained_scratch"]["task"] == task]
            if len(tl_task) > 0:
                converged_accs.append(tl_task["accuracy"].values[0] * 100)

        if scratch_curves:
            ref_steps = scratch_curves[0][0]
            all_accs = np.array([c[1] for c in scratch_curves])
            mean_accs = all_accs.mean(axis=0)
            if n_seeds > 1:
                std_accs = all_accs.std(axis=0)
                ax.fill_between(
                    ref_steps,
                    mean_accs - std_accs,
                    mean_accs + std_accs,
                    alpha=0.15,
                    color=c_scratch,
                    linewidth=0,
                )
            ax.plot(
                ref_steps,
                mean_accs,
                color=c_scratch,
                marker="o",
                markersize=4,
                linewidth=1.8,
                zorder=3,
            )

        if converged_accs:
            mean_converged = np.mean(converged_accs)
            ax.axhline(
                mean_converged,
                color=c_scratch,
                linestyle="--",
                linewidth=1.5,
                zorder=2,
            )
            if n_seeds > 1:
                std_converged = np.std(converged_accs)
                ax.axhspan(
                    mean_converged - std_converged,
                    mean_converged + std_converged,
                    alpha=0.08,
                    color=c_scratch,
                    linewidth=0,
                )

        if mashup_curves:
            ref_steps = mashup_curves[0][0]
            all_accs = np.array([c[1] for c in mashup_curves])
            mean_accs = all_accs.mean(axis=0)
            if n_seeds > 1:
                std_accs = all_accs.std(axis=0)
                ax.fill_between(
                    ref_steps,
                    mean_accs - std_accs,
                    mean_accs + std_accs,
                    alpha=0.15,
                    color=c_mashup,
                    linewidth=0,
                )
            ax.plot(
                ref_steps,
                mean_accs,
                color=c_mashup,
                marker="s",
                markersize=4,
                linewidth=1.8,
                zorder=3,
            )

        ax.set_title(label)
        ymin, ymax = ax.get_ylim()
        margin = (ymax - ymin) * 0.05
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy (%)" if i % cols == 0 else "")

    # Hide unused axes
    for j in range(len(cfg.tasks), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles = [
        mlines.Line2D(
            [0],
            [0],
            color=c_scratch,
            marker="o",
            markersize=5,
            linewidth=1.8,
            label="From scratch",
        ),
        mlines.Line2D(
            [0],
            [0],
            color=c_scratch,
            linestyle="--",
            linewidth=1.5,
            label="From scratch (converged)",
        ),
        mlines.Line2D(
            [0],
            [0],
            color=c_mashup,
            marker="s",
            markersize=5,
            linewidth=1.8,
            label="Mashup Learning",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=13,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    _save_plot(out_dir, f"{prefix}_training_curves")

    seeds_note = f" averaged over {n_seeds} seeds" if n_seeds > 1 else ""
    shading_note = (
        " Shaded bands show $\\pm 1$ standard deviation across seeds."
        if n_seeds > 1
        else ""
    )
    tex = (
        rf"% Training curves for {cfg.experiment_name}"
        f" (generated by analysis.plots)\n"
        rf"\begin{{figure*}}[t]" + "\n"
        r"    \centering" + "\n"
        rf"    \includegraphics[width=\linewidth]"
        rf"{{{prefix}_training_curves.pdf}}" + "\n"
        rf"    \caption{{\textbf{{Training curves on"
        rf" {cfg.experiment_name}{seeds_note}.}}"
        rf" Accuracy vs.\ training step for fine-tuning from scratch"
        rf" (teal) and from merged initialization (pink), using the best"
        rf" learning rate per task. The dashed line indicates converged"
        rf" from-scratch accuracy.{shading_note}}}" + "\n"
        rf"    \label{{fig:{prefix}_training_curves}}" + "\n"
        r"\end{figure*}" + "\n"
    )
    (out_dir / f"{prefix}_training_curves.tex").write_text(tex)
    print(f"Saved {out_dir / f'{prefix}_training_curves.tex'}")


# ---------------------------------------------------------------------------
# LR sensitivity plots
# ---------------------------------------------------------------------------


def _agg_by_lr(
    df: pd.DataFrame,
    lrs: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for lr in lrs:
        vals = df[np.isclose(df["lr"], lr)]["accuracy"].values
        means.append(np.mean(vals) if len(vals) > 0 else np.nan)
        stds.append(np.std(vals) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(stds)


def generate_lr_sensitivity(
    lr_df: pd.DataFrame,
    cfg: AnalysisConfig,
    out_dir: Path,
    prefix: str,
) -> None:
    """Generate LR sensitivity summary + per-task plots."""
    _apply_style()
    c_scratch, c_mashup = _get_colors()
    lr_df = lr_df.copy()
    lr_df["accuracy"] = lr_df["accuracy"] * 100
    n_seeds = lr_df["seed"].nunique()
    lr_labels = [lr_to_plot_label(lr) for lr in cfg.lrs]

    method_cfg = {
        "from_scratch": {
            "color": c_scratch,
            "marker": "o",
            "label": "From scratch",
        },
        "mashup": {
            "color": c_mashup,
            "marker": "s",
            "label": "Mashup Learning",
        },
    }

    # 1. Summary plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, mcfg in method_cfg.items():
        mdf = lr_df[lr_df["method"] == method]
        task_avg_per_lr = []
        for lr in cfg.lrs:
            lr_subset = mdf[np.isclose(mdf["lr"], lr)]
            seed_avgs = lr_subset.groupby("seed")["accuracy"].mean().values
            task_avg_per_lr.append(np.mean(seed_avgs) if len(seed_avgs) > 0 else np.nan)
        ax.plot(
            range(len(cfg.lrs)),
            task_avg_per_lr,
            color=mcfg["color"],
            marker=mcfg["marker"],
            markersize=7,
            linewidth=2.0,
            zorder=3,
            label=f"{mcfg['label']} (mean)",
        )

    ax.set_xticks(range(len(cfg.lrs)))
    ax.set_xticklabels(lr_labels)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.tight_layout()

    _save_plot(out_dir, f"{prefix}_lr_sensitivity_summary")

    # 2. Per-task plot
    rows, cols = cfg.plot_grid
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if len(cfg.tasks) == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for i, task in enumerate(cfg.tasks):
        ax = axes_flat[i]
        label = cfg.get_task_label(task)

        for method, mcfg in method_cfg.items():
            tdf = lr_df[(lr_df["method"] == method) & (lr_df["dataset"] == task)]
            means, stds = _agg_by_lr(tdf, cfg.lrs)
            x = range(len(cfg.lrs))

            if n_seeds > 1:
                ax.fill_between(
                    x,
                    means - stds,
                    means + stds,
                    alpha=0.15,
                    color=mcfg["color"],
                    linewidth=0,
                )
            ax.plot(
                x,
                means,
                color=mcfg["color"],
                marker=mcfg["marker"],
                markersize=4,
                linewidth=1.8,
                zorder=3,
            )

        ax.set_title(label)
        ax.set_xticks(range(len(cfg.lrs)))
        ax.set_xticklabels(lr_labels, fontsize=7)
        ax.set_xlabel("Learning Rate" if i >= (rows - 1) * cols else "")
        ax.set_ylabel("Accuracy (%)" if i % cols == 0 else "")

    for j in range(len(cfg.tasks), len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles = [
        mlines.Line2D(
            [0],
            [0],
            color=c_scratch,
            marker="o",
            markersize=5,
            linewidth=1.8,
            label="From scratch",
        ),
        mlines.Line2D(
            [0],
            [0],
            color=c_mashup,
            marker="s",
            markersize=5,
            linewidth=1.8,
            label="Mashup Learning",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    _save_plot(out_dir, f"{prefix}_lr_sensitivity_tasks")
