"""Analysis module for generating LaTeX tables and plots from pipeline results.

Public API:
  - run_stage(cfg)                         pipeline stage entry point
  - analyze_run(run_dir, output_dir)       single-run analysis
  - analyze_model(run_dirs, output_dir)    multi-seed aggregation
  - analyze_combined(run_dirs_by_experiment, output_dir)  cross-model combined tables
"""

from mashup.analysis.core import analyze_combined, analyze_model, analyze_run, run_stage

__all__ = ["run_stage", "analyze_run", "analyze_model", "analyze_combined"]
