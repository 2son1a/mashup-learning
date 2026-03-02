"""Public API orchestrating all analysis steps."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mashup.analysis.combined import (
    SETUP_LABELS,
    ModelEntry,
    generate_combined_results,
    generate_combined_speedup,
    generate_combined_timing,
)
from mashup.analysis.config import AnalysisConfig, load_config_from_run
from mashup.analysis.plots import generate_lr_sensitivity, generate_training_curves
from mashup.analysis.speedup import generate_speedup_table
from mashup.analysis.tables import (
    extract_lr_sensitivity,
    generate_lr_sweep_table,
    generate_results_table,
)
from mashup.analysis.timing import generate_timing_table

# ---------------------------------------------------------------------------
# Data loading & config population (inlined from the former discovery module)
# ---------------------------------------------------------------------------


def _read_summary_csv(run_dir: Path) -> pd.DataFrame | None:
    """Read summary.csv from *run_dir*, returning None if it doesn't exist."""
    csv_path = run_dir / "summary" / "summary.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["task"] = df["dataset"].str.replace("data/processed/", "", regex=False)
    return df


def _load_summary(run_dir: Path) -> pd.DataFrame:
    """Load summary.csv from a run directory (raises if missing)."""
    df = _read_summary_csv(run_dir)
    if df is None:
        raise FileNotFoundError(f"No summary/summary.csv in {run_dir}")
    return df


def _load_seeds(run_dirs: list[Path]) -> tuple[list[pd.DataFrame], list[str]]:
    """Load summary data from multiple seed directories.

    Returns (list_of_dfs, list_of_seed_names) for directories that have data.
    """
    frames, names = [], []
    for d in run_dirs:
        df = _read_summary_csv(d)
        if df is None:
            continue
        df["seed"] = d.name
        frames.append(df)
        names.append(d.name)
    return frames, names


def _populate_config(cfg: AnalysisConfig, df: pd.DataFrame) -> None:
    """Fill in tasks and LRs on *cfg* from the summary data."""
    tasks = df["dataset"].str.replace("data/processed/", "", regex=False).unique()
    cfg.tasks = sorted(tasks)
    if cfg.sweeps:
        sweep_name = next(iter(cfg.sweeps.values()))
        sweep = df[(df["name"] == sweep_name) & (df["checkpoint"] == "final")]
        lrs = sweep["lr"].dropna().unique()
        cfg.lrs = sorted(float(lr) for lr in lrs)


def run_stage(stage_cfg) -> None:
    """Pipeline stage entry point — matches the ``function(cfg)`` convention."""
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(stage_cfg, resolve=True)
    output_dir = Path(raw["output_dir"])
    analyze_run(output_dir.parent, output_dir)


def analyze_run(
    run_dir: Path,
    output_dir: Path | None = None,
) -> None:
    """Single-run analysis (one seed)."""
    run_dir = Path(run_dir)
    cfg = load_config_from_run(run_dir)
    df = _load_summary(run_dir)
    _populate_config(cfg, df)

    if output_dir is None:
        output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = cfg.experiment_name
    df["seed"] = run_dir.name

    print(f"\n=== Analyzing run: {run_dir.name} ===")
    print(f"  Tasks: {cfg.tasks}")
    print(f"  LRs: {cfg.lrs}")

    generate_results_table(df, cfg, output_dir, prefix)

    if cfg.sweeps and cfg.lrs:
        lr_df = extract_lr_sensitivity(df, cfg)
        if len(lr_df) > 0:
            lr_csv = output_dir / "data" / "lr_sensitivity.csv"
            lr_csv.parent.mkdir(parents=True, exist_ok=True)
            lr_df.to_csv(lr_csv, index=False)
            print(f"Saved {lr_csv}")
            generate_lr_sweep_table(lr_df, df, cfg, output_dir, prefix)
            generate_lr_sensitivity(lr_df, cfg, output_dir, prefix)

    generate_speedup_table([df], cfg, output_dir, prefix)
    generate_timing_table([run_dir], cfg, output_dir, prefix)
    generate_training_curves([df], cfg, output_dir, prefix)

    print(f"\n=== Done: {output_dir} ===")


def analyze_model(
    run_dirs: list[Path],
    output_dir: Path,
) -> None:
    """Multi-seed analysis for a single model. Aggregates across seeds."""
    if not run_dirs:
        raise ValueError("No run directories provided")

    cfg = load_config_from_run(run_dirs[0])
    seed_frames, seed_names = _load_seeds(run_dirs)
    if not seed_frames:
        raise ValueError("No valid seed data found in provided directories")

    df = pd.concat(seed_frames, ignore_index=True)
    _populate_config(cfg, df)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = cfg.experiment_name
    n_seeds = len(seed_frames)

    print(f"\n=== Model analysis: {cfg.experiment_name} ({n_seeds} seeds) ===")
    print(f"  Tasks: {cfg.tasks}")
    print(f"  LRs: {cfg.lrs}")

    generate_results_table(df, cfg, output_dir, prefix)

    if cfg.sweeps and cfg.lrs:
        lr_df = extract_lr_sensitivity(df, cfg)
        if len(lr_df) > 0:
            lr_csv = output_dir / "data" / "lr_sensitivity.csv"
            lr_csv.parent.mkdir(parents=True, exist_ok=True)
            lr_df.to_csv(lr_csv, index=False)
            print(f"Saved {lr_csv}")
            generate_lr_sweep_table(lr_df, df, cfg, output_dir, prefix)
            generate_lr_sensitivity(lr_df, cfg, output_dir, prefix)

    generate_speedup_table(seed_frames, cfg, output_dir, prefix)
    generate_timing_table(run_dirs, cfg, output_dir, prefix)
    generate_training_curves(seed_frames, cfg, output_dir, prefix)

    print(f"\n=== Done: {output_dir} ===")


def analyze_combined(
    run_dirs_by_experiment: dict[str, list[Path]],
    output_dir: Path,
    group: str = "all",
) -> None:
    """Cross-model combined tables.

    *run_dirs_by_experiment* maps experiment names to lists of seed run dirs.
    *group* is used as a label in captions and filenames.
    """
    if not run_dirs_by_experiment:
        print(f"No runs found for group '{group}'")
        return

    print(f"\n=== Combined analysis: group={group} ===")

    model_entries: list[ModelEntry] = []
    ref_cfg: AnalysisConfig | None = None

    for _exp_name, dirs in sorted(run_dirs_by_experiment.items()):
        if not dirs:
            continue
        cfg = load_config_from_run(dirs[0])
        seed_frames, _ = _load_seeds(dirs)
        if not seed_frames:
            continue

        df_all = pd.concat(seed_frames, ignore_index=True)
        _populate_config(cfg, df_all)

        if ref_cfg is None:
            ref_cfg = cfg

        me = ModelEntry(
            display_name=cfg.experiment_name,
            setup=SETUP_LABELS.get(cfg.group, cfg.group),
            cfg=cfg,
            seed_dfs=seed_frames,
            seed_dirs=dirs,
            n_seeds=len(seed_frames),
        )
        model_entries.append(me)
        print(f"  {cfg.experiment_name} ({cfg.group}): {len(seed_frames)} seeds")

    if not model_entries or ref_cfg is None:
        print("No valid model data found")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    generate_combined_results(model_entries, output_dir, group, ref_cfg)
    generate_combined_timing(model_entries, output_dir, group, ref_cfg)
    generate_combined_speedup(model_entries, output_dir, group, ref_cfg)

    print(f"\n=== Done: {output_dir} ===")
