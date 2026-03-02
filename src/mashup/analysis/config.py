"""Analysis configuration parsed from pipeline.yaml's ``analysis`` section."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


@dataclass
class MethodRole:
    """Maps a semantic role to the CSV name used in summary.csv."""

    name: str
    display: str
    checkpoint: str | None = None


@dataclass
class AnalysisConfig:
    """All metadata the analysis module needs, parsed from pipeline config."""

    group: str
    experiment_name: str
    methods: dict[str, MethodRole] = field(default_factory=dict)
    sweeps: dict[str, str] = field(default_factory=dict)
    timing_stages: dict[str, str] = field(default_factory=dict)
    task_labels: dict[str, str] | None = None

    # Populated at runtime from summary data
    tasks: list[str] = field(default_factory=list)
    lrs: list[float] = field(default_factory=list)

    def method_name(self, role: str) -> str:
        return self.methods[role].name

    def sweep_for(self, role: str) -> str:
        return self.sweeps[role]

    def get_task_label(self, task: str) -> str:
        if self.task_labels and task in self.task_labels:
            return self.task_labels[task]
        return _auto_task_label(task)

    @property
    def task_header_cols(self) -> str:
        return " & ".join(rf"\textbf{{{self.get_task_label(t)}}}" for t in self.tasks)

    @property
    def plot_grid(self) -> tuple[int, int]:
        cols = min(4, len(self.tasks))
        rows = math.ceil(len(self.tasks) / cols) if cols else 1
        return rows, cols


_KNOWN_TASK_LABELS: dict[str, str] = {
    "arc_easy": "ARC-e",
    "arc_challenge": "ARC-c",
    "commonsense_qa": "CSQA",
    "hellaswag": "Hella.",
    "math_qa": "MathQA",
    "openbookqa": "OBQA",
    "piqa": "PIQA",
    "social_iqa": "SIQA",
    "winogrande": "Wino.",
    "boolq": "BoolQ",
    "copa": "COPA",
    "rte": "RTE",
    "wic": "WiC",
    "multirc": "MultiRC",
}


def _auto_task_label(task: str) -> str:
    """Generate a short LaTeX-friendly label from a task slug.

    Rules: known abbreviations take priority, then title-case with
    truncation to keep column headers compact.
    """
    if task in _KNOWN_TASK_LABELS:
        return _KNOWN_TASK_LABELS[task]
    parts = task.replace("_", " ").title()
    return parts[:8] + "." if len(parts) > 8 else parts


def lr_to_tex(lr: float) -> str:
    r"""Format a learning rate as LaTeX scientific notation.

    5e-05  -> ``$5 \times 10^{-5}$``
    1.6e-4 -> ``$1.6 \times 10^{-4}$``
    """
    s = f"{lr:.2e}"
    m = re.match(r"^([\d.]+?)0*e([+-]?\d+)$", s)
    if not m:
        return str(lr)
    mantissa, exp_str = m.group(1), int(m.group(2))
    mantissa = mantissa.rstrip(".")
    return rf"${mantissa} \times 10^{{{exp_str}}}$"


def lr_to_plot_label(lr: float) -> str:
    r"""Compact LR label for matplotlib tick labels."""
    s = f"{lr:.2e}"
    m = re.match(r"^([\d.]+?)0*e([+-]?\d+)$", s)
    if not m:
        return str(lr)
    mantissa, exp_str = m.group(1), int(m.group(2))
    mantissa = mantissa.rstrip(".")
    return rf"${mantissa}\!\cdot\!10^{{{exp_str}}}$"


def parse_analysis_config(
    pipeline_cfg: DictConfig | dict[str, Any],
) -> AnalysisConfig:
    """Build an AnalysisConfig from a loaded pipeline YAML.

    Reads from ``stages.analysis.config`` (the only supported location).
    Raises ValueError if the section is missing.
    """
    if isinstance(pipeline_cfg, DictConfig):
        raw = OmegaConf.to_container(pipeline_cfg, resolve=True)
    else:
        raw = pipeline_cfg

    analysis = raw.get("stages", {}).get("analysis", {}).get("config")
    if not analysis or "group" not in analysis:
        raise ValueError(
            "Pipeline config must have a 'stages.analysis.config' section "
            "with at least a 'group' key. See CLAUDE.md for the expected format."
        )

    methods = {}
    for role_key, minfo in analysis["methods"].items():
        methods[role_key] = MethodRole(
            name=minfo["name"],
            display=minfo["display"],
            checkpoint=minfo.get("checkpoint"),
        )

    task_labels = analysis.get("task_labels")
    if isinstance(task_labels, dict) and not task_labels:
        task_labels = None

    return AnalysisConfig(
        group=analysis["group"],
        experiment_name=raw.get("experiment_name", "unknown"),
        methods=methods,
        sweeps=dict(analysis.get("sweeps", {})),
        timing_stages=dict(analysis.get("timing_stages", {})),
        task_labels=task_labels,
    )


def load_config_from_run(run_dir: Path) -> AnalysisConfig:
    """Load AnalysisConfig from a pipeline run directory."""
    yaml_path = run_dir / "pipeline.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No pipeline.yaml in {run_dir}")
    cfg = OmegaConf.load(yaml_path)
    return parse_analysis_config(cfg)
