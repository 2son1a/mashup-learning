"""
Best model selection based on evaluation results.

Reads evaluation results matching a glob pattern, ranks models by a
configurable metric, creates a symlink to the best model directory
(and optionally to its eval results directory) so that downstream
pipeline stages can reference the canonical name unchanged.
"""

import json
import warnings
from pathlib import Path

from omegaconf import DictConfig

_LOWER_IS_BETTER = {"perplexity", "nll"}


def select_best(cfg: DictConfig) -> dict:
    """Select the best model by metric and create symlinks.

    Config keys::

        metric             eval_results.json field to rank by  (default "accuracy")
        eval_results_glob  glob pattern for eval output dirs
        link_name          symlink name for the best model directory
        eval_link_name     (optional) symlink name for the best eval results dir
        output_dir         where to write select_best_results.json

    The symlinks are created inside ``run_dir`` (derived from
    ``output_dir.parent``, same convention as the topk stage).

    Returns a dict with the best model and full ranking.
    """
    metric = cfg.get("metric", "accuracy")
    eval_results_glob = str(cfg.eval_results_glob)
    link_name = str(cfg.link_name)
    eval_link_name = cfg.get("eval_link_name")
    output_dir = Path(cfg.output_dir)

    # Derive run_dir from output_dir (output_dir is already prefixed by pipeline)
    run_dir = output_dir.parent

    # --- Collect eval results ---

    matched_dirs = sorted(run_dir.glob(eval_results_glob))
    if not matched_dirs:
        raise FileNotFoundError(
            f"No directories matching '{eval_results_glob}' found in {run_dir}"
        )

    results = []
    for eval_dir in matched_dirs:
        results_file = eval_dir / "eval_results.json"
        if not results_file.exists():
            warnings.warn(f"{results_file} not found, skipping", stacklevel=2)
            continue
        data = json.loads(results_file.read_text())
        if metric not in data:
            warnings.warn(
                f"Metric '{metric}' not in {results_file}, skipping", stacklevel=2
            )
            continue
        results.append(
            {
                "eval_dir": str(eval_dir),
                "model": data["model"],
                "dataset": data.get("dataset", ""),
                metric: data[metric],
                **{
                    k: data[k]
                    for k in ("accuracy", "exact_match", "perplexity", "nll")
                    if k in data and k != metric
                },
            }
        )

    if not results:
        raise ValueError(
            f"No valid eval results found for glob '{eval_results_glob}' in {run_dir}"
        )

    # --- Rank ---

    reverse = metric not in _LOWER_IS_BETTER  # descending for accuracy/exact_match
    results.sort(key=lambda r: r[metric], reverse=reverse)

    best = results[0]

    print(f"\n{'=' * 60}")
    print(f"  Best model by {metric} ({'higher' if reverse else 'lower'} is better)")
    print(f"  From {len(results)} candidates (glob: {eval_results_glob})")
    print(f"{'=' * 60}")
    for i, r in enumerate(results):
        marker = "  >>>" if i == 0 else "     "
        print(f"{marker} [{i + 1}] {metric}={r[metric]:.6f}  model={r['model']}")
    print(f"{'=' * 60}\n")

    # --- Create model symlink ---

    model_path = Path(best["model"])
    link_path = run_dir / link_name

    # Use a relative symlink (both live under run_dir)
    try:
        relative_target = model_path.relative_to(run_dir)
    except ValueError:
        # model_path is not under run_dir; fall back to absolute
        relative_target = model_path

    if link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(relative_target)
    print(f"  Model symlink: {link_path} -> {relative_target}")

    # --- Optionally create eval symlink ---

    if eval_link_name:
        eval_dir_path = Path(best["eval_dir"])
        eval_link_path = run_dir / str(eval_link_name)

        try:
            eval_relative = eval_dir_path.relative_to(run_dir)
        except ValueError:
            eval_relative = eval_dir_path

        if eval_link_path.is_symlink():
            eval_link_path.unlink()
        eval_link_path.symlink_to(eval_relative)
        print(f"  Eval symlink:  {eval_link_path} -> {eval_relative}")

    # --- Persist ---

    output_dir.mkdir(parents=True, exist_ok=True)

    details = {
        "metric": metric,
        "eval_results_glob": eval_results_glob,
        "num_candidates": len(results),
        "best": best,
        "link_name": link_name,
        "link_target": str(model_path),
        "eval_link_name": str(eval_link_name) if eval_link_name else None,
        "all_ranked": results,
    }
    details_path = output_dir / "select_best_results.json"
    details_path.write_text(json.dumps(details, indent=2))
    print(f"  Results saved to {details_path}")

    return details
