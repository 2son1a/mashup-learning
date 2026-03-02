"""Multi-stage pipeline with async GPU scheduling.

Core orchestration logic: config resolution, matrix expansion, GPU scheduling,
job execution, timing, and signal handling.  The CLI lives in scripts/pipeline.py.
"""

import asyncio
import csv
import itertools
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from rich.console import Console
from rich.text import Text

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_active_procs: list[asyncio.subprocess.Process] = []
_timing_records: list[dict] = []


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = logging.getLogger("mashup.pipeline")
log.setLevel(logging.INFO)


class _RichHandler(logging.Handler):
    """Renders Rich markup to the terminal via ``Console.print``."""

    def __init__(self):
        super().__init__()
        self._console = Console(highlight=False)

    def emit(self, record):
        try:
            self._console.print(record.getMessage())
        except Exception:
            self.handleError(record)


class _PlainFormatter(logging.Formatter):
    """Strips Rich markup so file logs are plain text."""

    def format(self, record):
        return Text.from_markup(record.getMessage()).plain


log.addHandler(_RichHandler())


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def _cleanup_on_signal(signum, _frame):
    """Kill all active child processes and re-raise the signal."""
    for proc in _active_procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def register_signal_handlers() -> None:
    """Install SIGINT/SIGTERM handlers that clean up child processes."""
    signal.signal(signal.SIGINT, _cleanup_on_signal)
    signal.signal(signal.SIGTERM, _cleanup_on_signal)


# ---------------------------------------------------------------------------
# Constants & data classes
# ---------------------------------------------------------------------------

JOB_TYPE_TO_SCRIPT = {
    "train": "scripts/train.py",
    "merge": "scripts/merge.py",
    "lora_merge": "scripts/lora_merge.py",
    "eval": "scripts/eval.py",
    "topk": "scripts/topk.py",
    "select_best": "scripts/select_best.py",
    "wait_for_disk": "scripts/wait_for_disk.py",
    "cleanup": "scripts/cleanup.py",
    "summary": "scripts/summary.py",
    "analysis": "scripts/analysis_stage.py",
}

PIPELINE_KEYS = {"type", "num_gpus", "matrix", "subjobs", "cleanup_checkpoints"}


@dataclass
class SubJob:
    name: str
    job_type: str
    config: OmegaConf


@dataclass
class Job:
    name: str
    job_type: str
    config: OmegaConf
    num_gpus: int = 1
    subjobs: list[SubJob] | None = None
    cleanup_checkpoints: str | None = None
    stage: str = ""


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _substitute(obj, subs: dict[str, str]):
    """Recursively substitute ``{var}`` placeholders in strings.

    When a value is exactly ``{var}`` (the entire string), the raw typed
    value from *subs* is returned so that numeric matrix variables (e.g.
    learning_rate, num_epochs) keep their original Python type.
    """
    if isinstance(obj, str):
        for k, v in subs.items():
            if obj == f"{{{k}}}":
                return v
        for k, v in subs.items():
            obj = obj.replace(f"{{{k}}}", str(v))
        return obj
    if isinstance(obj, dict):
        return {k: _substitute(v, subs) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(i, subs) for i in obj]
    return obj


def _resolve_defaults(
    config: dict, config_root: Path, run_dir: Path | None = None
) -> dict:
    """Merge ``defaults`` list into a flat config dict.

    Supports a special ``@output`` group prefix that resolves relative to
    ``run_dir`` instead of ``config_root``, allowing later stages to load
    configs produced by earlier stages (e.g. topk -> merge).
    """
    defaults = config.pop("defaults", [])
    base = OmegaConf.create({})
    self_applied = False
    for entry in defaults:
        if entry == "_self_":
            base = OmegaConf.merge(base, config)
            self_applied = True
        else:
            for group, name in entry.items():
                if group == "@output" and run_dir is not None:
                    path = run_dir / f"{name}.yaml"
                else:
                    path = config_root / group.lstrip("/") / f"{name}.yaml"
                base = OmegaConf.merge(base, OmegaConf.load(path))
    if not self_applied:
        base = OmegaConf.merge(base, config)
    return OmegaConf.to_container(base)


def _build_job_config(
    stage_cfg, pipeline_cfg, subs: dict, run_dir: Path | None = None
) -> OmegaConf:
    """Build a self-contained job config from a stage definition."""
    if "config" in stage_cfg:
        config = OmegaConf.to_container(stage_cfg.config, resolve=False)
    else:
        config = {
            k: OmegaConf.to_container(v, resolve=False) if OmegaConf.is_config(v) else v
            for k, v in stage_cfg.items()
            if k not in PIPELINE_KEYS
        }

    config = _substitute(config, subs)

    if "defaults" in config:
        config = _resolve_defaults(config, Path("configs"), run_dir)

    if "base_model" not in config and "base_model" in pipeline_cfg:
        config["base_model"] = str(pipeline_cfg.base_model)

    if "seed" not in config and "seed" in pipeline_cfg:
        config["seed"] = int(pipeline_cfg.seed)

    return OmegaConf.create(config)


# ---------------------------------------------------------------------------
# Matrix expansion
# ---------------------------------------------------------------------------


def _expand_stage(
    stage_name: str, stage_cfg, pipeline_cfg, run_dir: Path | None = None
) -> list[Job]:
    """Expand a stage (with optional matrix) into individual jobs.

    Stages may define a ``subjobs`` list instead of a single ``type`` +
    ``config``.  Each subjob is a step that runs sequentially on the same
    GPU(s), with per-subjob caching via ``.done`` markers.
    """
    variables = dict(pipeline_cfg.get("variables", {}))
    matrix = stage_cfg.get("matrix", {})
    num_gpus = stage_cfg.get("num_gpus", 1)
    has_subjobs = "subjobs" in stage_cfg

    if not matrix:
        all_subs: list[dict] = [{}]
    else:
        resolved = {}
        for k, v in matrix.items():
            if isinstance(v, str) and v in variables:
                resolved[k] = variables[v]
            elif isinstance(v, list):
                resolved[k] = v
            else:
                resolved[k] = [v]
        keys = list(resolved.keys())
        all_subs = [
            dict(zip(keys, combo, strict=True))
            for combo in itertools.product(*[resolved[k] for k in keys])
        ]

    jobs: list[Job] = []
    for subs in all_subs:
        suffix = "_".join(str(v) for v in subs.values()) if subs else ""
        job_name = f"{stage_name}_{suffix}" if suffix else stage_name

        if has_subjobs:
            subjobs = []
            for sj_cfg in stage_cfg.subjobs:
                sj_type = sj_cfg.get("type", "train")
                sj_config = _build_job_config(sj_cfg, pipeline_cfg, subs, run_dir)
                sj_name = f"{job_name}_{sj_type}"
                subjobs.append(SubJob(name=sj_name, job_type=sj_type, config=sj_config))

            cleanup = stage_cfg.get("cleanup_checkpoints", None)
            if cleanup is not None:
                cleanup = str(_substitute(str(cleanup), subs))

            jobs.append(
                Job(
                    name=job_name,
                    job_type="sequence",
                    config=OmegaConf.create({}),
                    num_gpus=num_gpus,
                    subjobs=subjobs,
                    cleanup_checkpoints=cleanup,
                )
            )
        else:
            job_type = stage_cfg.get("type", "train")
            jobs.append(
                Job(
                    name=job_name,
                    job_type=job_type,
                    config=_build_job_config(stage_cfg, pipeline_cfg, subs, run_dir),
                    num_gpus=num_gpus,
                )
            )

    return jobs


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _prefix_cfg_paths(cfg: dict, run_dir: Path) -> None:
    """Prefix relative paths in a config dict with the run directory (in-place)."""
    if "output_dir" in cfg:
        cfg["output_dir"] = str(run_dir / cfg["output_dir"])
    if "merge" in cfg and "models" in cfg["merge"]:
        for model in cfg["merge"]["models"]:
            if "path" in model and not Path(model["path"]).is_absolute():
                model["path"] = str(run_dir / model["path"])
    if "models" in cfg and "merge" not in cfg:
        for model in cfg["models"]:
            if "path" in model and not Path(model["path"]).is_absolute():
                model["path"] = str(run_dir / model["path"])
    if "model_path" in cfg and not Path(cfg["model_path"]).is_absolute():
        cfg["model_path"] = str(run_dir / cfg["model_path"])
    if (
        cfg.get("resume_from_checkpoint")
        and not Path(cfg["resume_from_checkpoint"]).is_absolute()
    ):
        cfg["resume_from_checkpoint"] = str(run_dir / cfg["resume_from_checkpoint"])
    if cfg.get("lora_model_dir") and not Path(cfg["lora_model_dir"]).is_absolute():
        cfg["lora_model_dir"] = str(run_dir / cfg["lora_model_dir"])
    if "base_model" in cfg and "/" not in cfg["base_model"]:
        cfg["base_model"] = str(run_dir / cfg["base_model"])


def _prefix_paths(job: Job, run_dir: Path) -> None:
    """Prefix relative output_dir and merge model paths with the run directory."""
    if job.subjobs:
        for sj in job.subjobs:
            cfg = OmegaConf.to_container(sj.config, resolve=True)
            _prefix_cfg_paths(cfg, run_dir)
            sj.config = OmegaConf.create(cfg)
        if job.cleanup_checkpoints and not Path(job.cleanup_checkpoints).is_absolute():
            job.cleanup_checkpoints = str(run_dir / job.cleanup_checkpoints)
        return

    cfg = OmegaConf.to_container(job.config, resolve=True)
    _prefix_cfg_paths(cfg, run_dir)
    # NOTE: topk fields (eval_results_glob, exclude_glob) are directory-name
    # patterns matched inside run_dir.  The topk module derives run_dir from
    # output_dir.parent, so no prefixing is needed here.
    job.config = OmegaConf.create(cfg)


def _set_wandb_name(job: Job, experiment_name: str) -> None:
    """Set a clean wandb run name and group if not already configured."""
    if job.subjobs:
        for sj in job.subjobs:
            if sj.job_type not in ("train", "eval", "lm_eval"):
                continue
            cfg = OmegaConf.to_container(sj.config, resolve=True)
            if not cfg.get("wandb_name"):
                cfg["wandb_name"] = sj.name
            if not cfg.get("wandb_group"):
                cfg["wandb_group"] = experiment_name
            sj.config = OmegaConf.create(cfg)
        return

    if job.job_type not in ("train", "eval", "lm_eval"):
        return

    cfg = OmegaConf.to_container(job.config, resolve=True)

    if not cfg.get("wandb_name"):
        cfg["wandb_name"] = job.name

    if not cfg.get("wandb_group"):
        cfg["wandb_group"] = experiment_name

    job.config = OmegaConf.create(cfg)


# ---------------------------------------------------------------------------
# GPU scheduling
# ---------------------------------------------------------------------------


def _detect_gpus() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True
        )
        return list(range(out.stdout.count("GPU ")))
    except Exception:
        return [0]


async def _run_job(job: "Job | SubJob", gpu_ids: list[int], log_dir: Path) -> int:
    """Execute a job on specified GPUs, logging output to a file."""
    script = JOB_TYPE_TO_SCRIPT[job.job_type]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids)),
        "PYTHONUNBUFFERED": "1",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(job.config, f.name)
        config_path = f.name

    try:
        with open(log_dir / f"{job.name}.log", "w") as log:
            proc = await asyncio.create_subprocess_exec(
                "uv",
                "run",
                "python",
                script,
                config_path,
                env=env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            _active_procs.append(proc)
            try:
                return await proc.wait()
            finally:
                _active_procs.remove(proc)
    finally:
        Path(config_path).unlink(missing_ok=True)


async def _run_subjobs_sequence(
    job: Job, gpus: list[int], log_dir: Path
) -> tuple[str, int]:
    """Run subjobs sequentially on the same GPUs, with per-subjob caching."""
    gpu_str = f"GPUs {gpus}" if gpus else "no GPU"

    for sj in job.subjobs:
        out_dir = OmegaConf.to_container(sj.config, resolve=True).get("output_dir", "")
        dir_suffix = f"  {out_dir}" if out_dir else ""

        if out_dir and (Path(out_dir) / ".done").exists():
            log.info(f"  [green]✓[/] {sj.name}  [dim](cached)[/]{dir_suffix}")
            _timing_records.append(
                {
                    "stage": job.stage,
                    "job": job.name,
                    "subjob": sj.name,
                    "job_type": sj.job_type,
                    "elapsed_s": 0,
                    "cached": True,
                }
            )
            continue

        log.info(f"  [dim]▸[/] {sj.name}  [dim]{gpu_str}[/]{dir_suffix}")
        t0 = asyncio.get_event_loop().time()
        rc = await _run_job(sj, gpus, log_dir)
        elapsed_s = asyncio.get_event_loop().time() - t0
        elapsed = _fmt_time(elapsed_s)
        _timing_records.append(
            {
                "stage": job.stage,
                "job": job.name,
                "subjob": sj.name,
                "job_type": sj.job_type,
                "elapsed_s": round(elapsed_s, 1),
                "cached": False,
            }
        )

        if rc == 0:
            log.info(f"  [green]✓[/] {sj.name}  [dim]{elapsed}[/]{dir_suffix}")
            if out_dir:
                Path(out_dir, ".done").touch()
        else:
            log.info(
                f"  [bright_red]✗ {sj.name}  FAILED[/]  [dim]{elapsed}[/]{dir_suffix}"
            )
            return job.name, rc

    if job.cleanup_checkpoints:
        ckpt_dir = Path(job.cleanup_checkpoints)
        if ckpt_dir.exists():
            ckpts = sorted(ckpt_dir.glob("checkpoint-*"))
            if ckpts:
                for ckpt in ckpts:
                    shutil.rmtree(ckpt)
                log.info(
                    f"  [dim]Cleaned {len(ckpts)} checkpoints from {ckpt_dir.name}[/]"
                )

    return job.name, 0


async def _run_on_gpus(
    job: Job, gpu_queue: asyncio.Queue, log_dir: Path
) -> tuple[str, int]:
    """Acquire GPUs from queue, run job, release them back.

    Jobs with ``num_gpus=0`` run immediately without acquiring any GPU slot
    (useful for I/O-only stages like topk).
    """
    name = job.name

    if job.num_gpus == 0:
        if job.subjobs:
            return await _run_subjobs_sequence(job, [], log_dir)

        out_dir = OmegaConf.to_container(job.config, resolve=True).get("output_dir", "")
        dir_suffix = f"  {out_dir}" if out_dir else ""
        log.info(f"  [dim]▸[/] {name}  [dim]no GPU[/]{dir_suffix}")
        t0 = asyncio.get_event_loop().time()
        rc = await _run_job(job, [], log_dir)
        elapsed_s = asyncio.get_event_loop().time() - t0
        elapsed = _fmt_time(elapsed_s)
        _timing_records.append(
            {
                "stage": job.stage,
                "job": job.name,
                "subjob": "",
                "job_type": job.job_type,
                "elapsed_s": round(elapsed_s, 1),
                "cached": False,
            }
        )
        if rc == 0:
            log.info(f"  [green]✓[/] {name}  [dim]{elapsed}[/]{dir_suffix}")
            if out_dir:
                Path(out_dir, ".done").touch()
        else:
            log.info(
                f"  [bright_red]✗ {name}  FAILED[/]  [dim]{elapsed}[/]{dir_suffix}"
            )
        return job.name, rc

    gpus = [await gpu_queue.get() for _ in range(job.num_gpus)]
    try:
        if job.subjobs:
            return await _run_subjobs_sequence(job, gpus, log_dir)

        out_dir = OmegaConf.to_container(job.config, resolve=True).get("output_dir", "")
        dir_suffix = f"  {out_dir}" if out_dir else ""
        log.info(f"  [dim]▸[/] {name}  [dim]GPUs {gpus}[/]{dir_suffix}")
        t0 = asyncio.get_event_loop().time()
        rc = await _run_job(job, gpus, log_dir)
        elapsed_s = asyncio.get_event_loop().time() - t0
        elapsed = _fmt_time(elapsed_s)
        _timing_records.append(
            {
                "stage": job.stage,
                "job": job.name,
                "subjob": "",
                "job_type": job.job_type,
                "elapsed_s": round(elapsed_s, 1),
                "cached": False,
            }
        )
        if rc == 0:
            log.info(f"  [green]✓[/] {name}  [dim]{elapsed}[/]{dir_suffix}")
            if out_dir:
                Path(out_dir, ".done").touch()
        else:
            log.info(
                f"  [bright_red]✗ {name}  FAILED[/]  [dim]{elapsed}[/]{dir_suffix}"
            )
        return job.name, rc
    finally:
        for g in gpus:
            gpu_queue.put_nowait(g)


# ---------------------------------------------------------------------------
# Shared tokenizer
# ---------------------------------------------------------------------------

_TOKENIZER_FILES = [
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "added_tokens.json",
]


def _save_shared_tokenizer(run_dir: Path) -> None:
    """Copy a tokenizer (with chat template) to run_dir/tokenizer/ once.

    Since there is one base model per pipeline, the tokenizer is shared
    across all stages.  Saving it at the run root lets eval stages find
    it even for merged models whose tokenizer lacks a chat template.
    """
    shared = run_dir / "tokenizer"
    if shared.exists():
        return

    for candidate in sorted(run_dir.iterdir()):
        tok_cfg = candidate / "tokenizer_config.json"
        if not candidate.is_dir() or not tok_cfg.exists():
            continue
        try:
            data = json.loads(tok_cfg.read_text())
            if data.get("chat_template"):
                shared.mkdir()
                for name in _TOKENIZER_FILES:
                    src = candidate / name
                    if src.exists():
                        shutil.copy2(src, shared / name)
                log.info(f"  [dim]Saved shared tokenizer from {candidate.name}[/]")
                return
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Job completion check
# ---------------------------------------------------------------------------


def _is_job_done(job: Job) -> bool:
    """Check if a job already completed successfully (.done marker exists).

    For jobs with subjobs, all subjobs must be complete.
    """
    if job.subjobs:
        return all(
            bool(d) and (Path(d) / ".done").exists()
            for sj in job.subjobs
            for d in [
                OmegaConf.to_container(sj.config, resolve=True).get("output_dir", "")
            ]
        )
    out_dir = OmegaConf.to_container(job.config, resolve=True).get("output_dir", "")
    return bool(out_dir) and (Path(out_dir) / ".done").exists()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _experiment_name(base_name: str, seed: int | None) -> str:
    """Build the experiment name, appending ``_seed{N}`` when a seed is set."""
    if seed is not None:
        return f"{base_name}_seed{seed}"
    return base_name


async def run_pipeline(
    pipeline_cfg: OmegaConf,
    *,
    pipeline_checkpoint: Path | None = None,
    resume_stage: int | None = None,
    seed: int | None = None,
    base_model: str | None = None,
    experiment_name: str | None = None,
    max_gpus: int | None = None,
) -> None:
    """Execute the full pipeline: stages sequentially, jobs in parallel.

    When *pipeline_checkpoint* and *resume_stage* are provided the pipeline
    resumes into the existing run directory, skipping stages before
    *resume_stage* (0-indexed) and skipping already-completed jobs within
    later stages.

    When *seed* is provided it is stored in ``pipeline_cfg`` so that
    ``_build_job_config`` injects it into every job config (Axolotl uses it
    for reproducible training).  The seed is also appended to the experiment
    name so that each seed run gets its own output directory.

    *base_model* and *experiment_name* override the corresponding keys in
    ``pipeline_cfg``, allowing the same config file to be reused across
    different models.
    """
    if seed is not None:
        pipeline_cfg.seed = seed
    if base_model is not None:
        pipeline_cfg.base_model = base_model
    if experiment_name is not None:
        pipeline_cfg.experiment_name = experiment_name

    gpu_ids = _detect_gpus()
    if max_gpus is not None:
        gpu_ids = gpu_ids[:max_gpus]
    gpu_queue: asyncio.Queue[int] = asyncio.Queue()
    for g in gpu_ids:
        gpu_queue.put_nowait(g)

    experiment_name = _experiment_name(pipeline_cfg.get("experiment_name", "run"), seed)
    resuming = pipeline_checkpoint is not None

    if resuming:
        run_dir = pipeline_checkpoint
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path("outputs") / f"{experiment_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    _file_handler = logging.FileHandler(run_dir / "pipeline.log")
    _file_handler.setFormatter(_PlainFormatter())
    log.addHandler(_file_handler)

    OmegaConf.save(pipeline_cfg, run_dir / "pipeline.yaml")

    stage_names = list(pipeline_cfg.stages.keys())
    stage_count = len(pipeline_cfg.stages)

    if resuming:
        skipped_names = ", ".join(stage_names[:resume_stage])
        log.info(f"\n  [yellow]Resuming[/]  {run_dir}")
        log.info(
            f"  [dim]Skipping[/]  {resume_stage} completed"
            f" {'stage' if resume_stage == 1 else 'stages'}"
            f" ({skipped_names})"
        )
    else:
        log.info(f"\n  [dim]Run dir[/]  {run_dir}")
    log.info(f"  [dim]GPUs[/]     {len(gpu_ids)} available {gpu_ids}")

    _timing_records.clear()
    pipeline_t0 = asyncio.get_event_loop().time()

    for stage_idx, (stage_name, stage_cfg) in enumerate(pipeline_cfg.stages.items(), 1):
        if resuming and (stage_idx - 1) < resume_stage:
            continue

        jobs = _expand_stage(stage_name, stage_cfg, pipeline_cfg, run_dir)

        for job in jobs:
            job.stage = stage_name
            _prefix_paths(job, run_dir)
            _set_wandb_name(job, experiment_name)

        cached_jobs = [j for j in jobs if _is_job_done(j)]
        pending_jobs = [j for j in jobs if not _is_job_done(j)]

        for job in pending_jobs:
            if job.subjobs:
                for sj in job.subjobs:
                    out_dir = OmegaConf.to_container(sj.config, resolve=True).get(
                        "output_dir"
                    )
                    if out_dir:
                        Path(out_dir).mkdir(parents=True, exist_ok=True)
                        OmegaConf.save(sj.config, Path(out_dir) / "job_config.yaml")
            else:
                out_dir = OmegaConf.to_container(job.config, resolve=True).get(
                    "output_dir"
                )
                if out_dir:
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    OmegaConf.save(job.config, Path(out_dir) / "job_config.yaml")
                else:
                    OmegaConf.save(job.config, run_dir / f"{job.name}.yaml")

        header = f" {stage_name} "
        step_label = f"[dim]\\[{stage_idx}/{stage_count}][/]"
        if cached_jobs and pending_jobs:
            n_jobs = (
                f"[dim]{len(jobs)} {'job' if len(jobs) == 1 else 'jobs'}"
                f" ({len(cached_jobs)} cached, {len(pending_jobs)} to run)[/]"
            )
        elif cached_jobs:
            n_jobs = (
                f"[dim]{len(jobs)} {'job' if len(jobs) == 1 else 'jobs'}"
                f" (all cached)[/]"
            )
        else:
            n_jobs = f"[dim]{len(jobs)} {'job' if len(jobs) == 1 else 'jobs'}[/]"
        log.info(f"\n[cyan]{'─' * 60}[/]")
        log.info(f"  [bold]{header}[/] {step_label}  {n_jobs}")
        log.info(f"[cyan]{'─' * 60}[/]")

        for job in cached_jobs:
            log.info(f"  [green]✓[/] {job.name}  [dim](cached)[/]")
            _timing_records.append(
                {
                    "stage": stage_name,
                    "job": job.name,
                    "subjob": "",
                    "job_type": job.job_type,
                    "elapsed_s": 0,
                    "cached": True,
                }
            )

        if not pending_jobs:
            log.info("  [dim]Stage done (all cached)[/]")
            _save_shared_tokenizer(run_dir)
            continue

        stage_t0 = asyncio.get_event_loop().time()
        results = await asyncio.gather(
            *[_run_on_gpus(j, gpu_queue, log_dir) for j in pending_jobs]
        )
        stage_elapsed = _fmt_time(asyncio.get_event_loop().time() - stage_t0)

        failed = [(n, rc) for n, rc in results if rc != 0]
        if failed:
            log.error(
                f"\n  [bold bright_red]Stage '{stage_name}' failed[/]  [dim]{stage_elapsed}[/]"
            )
            log.error("  [dim]Check logs:[/]")
            for n, rc in failed:
                log.error(
                    f"    [bright_red]✗[/] {n} [dim](exit {rc}): {log_dir / f'{n}.log'}[/]"
                )
            log.removeHandler(_file_handler)
            _file_handler.close()
            raise SystemExit(1)

        log.info(f"  [dim]Stage done in {stage_elapsed}[/]")

        _save_shared_tokenizer(run_dir)

    total_elapsed = _fmt_time(asyncio.get_event_loop().time() - pipeline_t0)
    log.info(f"\n[bold green]  ✓ Pipeline completed[/]  [dim]{total_elapsed}[/]\n")

    if _timing_records:
        timing_path = run_dir / "timing.csv"
        with open(timing_path, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "stage",
                    "job",
                    "subjob",
                    "job_type",
                    "elapsed_s",
                    "cached",
                ],
            )
            w.writeheader()
            w.writerows(_timing_records)

        log.info("  [bold]Timing breakdown:[/]")
        cur_stage = None
        for r in _timing_records:
            if r["stage"] != cur_stage:
                cur_stage = r["stage"]
                stage_total = sum(
                    x["elapsed_s"] for x in _timing_records if x["stage"] == cur_stage
                )
                log.info(f"    [cyan]{cur_stage}[/]  [dim]{_fmt_time(stage_total)}[/]")
            label = r["subjob"] if r["subjob"] else r["job"]
            if r["cached"]:
                log.info(f"      {label}  [dim](cached)[/]")
            else:
                log.info(f"      {label}  {_fmt_time(r['elapsed_s'])}")
        log.info("")

    log.removeHandler(_file_handler)
    _file_handler.close()
