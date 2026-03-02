"""
Top-K model selection based on evaluation results.

Reads cross-evaluation results, ranks models by a configurable metric,
selects the top-k, and writes a merge config YAML consumable by the
merge stage.
"""

import json
import warnings
from fnmatch import fnmatch
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

# Metrics where lower values are better
_LOWER_IS_BETTER = {"perplexity", "nll"}


def select_topk(cfg: DictConfig) -> dict:
    """Select top-k models and write a merge config.

    Config keys::

        k                  number of models to select
        metric             eval_results.json field to rank by  (default "accuracy")
        weight_strategy    "equal" or "proportional"           (default "equal")
        eval_results_glob  glob pattern for eval output dirs   (e.g. "eval_*_on_arc_easy")
        exclude_glob       glob pattern for dirs to exclude    (optional)
        output_dir         where to write topk_config.yaml

    Returns a dict with the selected models and ranking details.
    """
    k = int(cfg.k)
    metric = cfg.get("metric", "accuracy")
    weight_strategy = cfg.get("weight_strategy", "equal")
    eval_results_glob = str(cfg.eval_results_glob)
    exclude_glob = cfg.get("exclude_glob")
    output_dir = Path(cfg.output_dir)

    # Derive run_dir from output_dir (output_dir is already prefixed by pipeline)
    run_dir = output_dir.parent

    # --- Collect eval results ---

    matched_dirs = sorted(run_dir.glob(eval_results_glob))
    if not matched_dirs:
        raise FileNotFoundError(
            f"No directories matching '{eval_results_glob}' found in {run_dir}"
        )

    if exclude_glob:
        exclude_pattern = str(exclude_glob)
        matched_dirs = [d for d in matched_dirs if not fnmatch(d.name, exclude_pattern)]

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
                    k_: data[k_]
                    for k_ in ("accuracy", "exact_match", "perplexity", "nll")
                    if k_ in data and k_ != metric
                },
            }
        )

    if not results:
        raise ValueError(
            f"No valid eval results found for glob '{eval_results_glob}' in {run_dir}"
        )

    # --- Rank and select ---

    reverse = metric not in _LOWER_IS_BETTER  # descending for accuracy/exact_match
    results.sort(key=lambda r: r[metric], reverse=reverse)

    selected = results[:k]

    print(f"\n{'=' * 60}")
    print(
        f"  Top-{k} models by {metric} ({'higher' if reverse else 'lower'} is better)"
    )
    print(f"  From {len(results)} candidates (glob: {eval_results_glob})")
    if exclude_glob:
        print(f"  Excluded: {exclude_glob}")
    print(f"{'=' * 60}")
    for i, r in enumerate(selected):
        print(f"  [{i + 1}] {metric}={r[metric]:.6f}  model={r['model']}")
    print(f"{'=' * 60}\n")

    # --- Compute weights ---

    if weight_strategy == "proportional":
        scores = [r[metric] for r in selected]
        if metric in _LOWER_IS_BETTER:
            # Invert so lower scores get higher weights
            max_score = max(scores)
            weights = [max_score / s if s > 0 else 0.0 for s in scores]
        else:
            weights = scores
    else:
        # Equal weights — mergekit normalize: true handles the rest
        weights = [1.0] * len(selected)

    # --- Build merge config ---

    merge_models = []
    for r, w in zip(selected, weights, strict=True):
        # Resolve to absolute path so _prefix_paths won't double-prefix
        model_path = str(Path(r["model"]).resolve())
        merge_models.append(
            {
                "path": model_path,
                "weight": round(w, 6),
            }
        )

    merge_config = {
        "merge": {
            "models": merge_models,
        },
    }

    # --- Persist ---

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write merge config YAML (consumed by merge stage via @output defaults)
    config_path = output_dir / "topk_config.yaml"
    OmegaConf.save(OmegaConf.create(merge_config), str(config_path))
    print(f"Merge config saved to {config_path}")

    # Write detailed results JSON for debugging / analysis
    topk_details = {
        "k": k,
        "metric": metric,
        "weight_strategy": weight_strategy,
        "eval_results_glob": eval_results_glob,
        "exclude_glob": str(exclude_glob) if exclude_glob else None,
        "num_candidates": len(results),
        "selected": [
            {**r, "weight": round(w, 6)} for r, w in zip(selected, weights, strict=True)
        ],
        "all_ranked": results,
    }
    details_path = output_dir / "topk_results.json"
    details_path.write_text(json.dumps(topk_details, indent=2))
    print(f"Detailed results saved to {details_path}")

    return topk_details
