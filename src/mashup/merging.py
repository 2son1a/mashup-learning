"""
Model merging using mergekit.

Integrates with Hydra config for composable merge configurations.
"""

import os
import shutil
import tempfile
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _strip_embedding_weights(adapter_dir: Path) -> Path | None:
    """Create a temporary copy of a LoRA adapter with embedding weights removed.

    When Axolotl resizes the embedding layer (e.g. to add a pad token),
    it saves the full ``embed_tokens`` and ``lm_head`` weights in the
    adapter checkpoint with ``save_embedding_layers=True``.  Mergekit
    cannot handle the resulting shape mismatch when loading the adapter
    onto the base model.

    This function detects the extra weights and returns a cleaned temp
    directory.  Returns ``None`` if no stripping was needed.
    """
    from safetensors.torch import load_file, save_file

    safetensor_path = adapter_dir / "adapter_model.safetensors"
    if not safetensor_path.exists():
        return None

    tensors = load_file(str(safetensor_path))
    embedding_keys = [k for k in tensors if "embed_tokens" in k or "lm_head" in k]
    # Only strip keys that are NOT LoRA A/B matrices (they are full weights)
    non_lora_embedding_keys = [
        k for k in embedding_keys if "lora_A" not in k and "lora_B" not in k
    ]

    if not non_lora_embedding_keys:
        return None

    print(
        f"  Stripping {len(non_lora_embedding_keys)} embedding weight(s) from "
        f"{adapter_dir.name} for merge compatibility"
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="merge_adapter_"))
    for f in adapter_dir.iterdir():
        if f.name == "adapter_model.safetensors":
            continue  # will be replaced
        if f.is_file():
            shutil.copy2(f, tmp_dir / f.name)

    cleaned = {k: v for k, v in tensors.items() if k not in non_lora_embedding_keys}
    save_file(cleaned, str(tmp_dir / "adapter_model.safetensors"))

    return tmp_dir


def merge(cfg: DictConfig) -> str:
    """
    Run model merge based on Hydra config.

    Args:
        cfg: Hydra config with 'merge' section containing:
            - method: linear, ties, or dare_ties
            - models: list of {path, weight, density}
            - base_model: base model path or HF ID
            - output_dir: where to save merged model
            - dtype: bfloat16, float16, or float32

    Returns:
        Path to merged model directory.
    """
    from mergekit.config import MergeConfiguration
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions

    use_wandb = os.environ.get("WANDB_API_KEY") is not None
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.get("wandb_project", "mashup"),
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=["merge", cfg.merge.method],
        )

    merge_cfg = cfg.merge

    # Build mergekit config dict, stripping embedding weights from LoRA
    # adapters that were trained with a resized vocabulary.
    models = []
    _tmp_dirs: list[Path] = []  # temp dirs to clean up after merge
    for m in merge_cfg.models:
        model_path = Path(m.path)
        is_lora = (model_path / "adapter_config.json").exists()

        if is_lora:
            cleaned = _strip_embedding_weights(model_path)
            if cleaned is not None:
                _tmp_dirs.append(cleaned)
                lora_path = str(cleaned)
            else:
                lora_path = str(model_path)

            # LoRA adapter: the lora field must be inside the ModelReference,
            # not at the InputModelDefinition level.
            model_entry = {
                "model": {
                    "model": cfg.base_model,
                    "lora": lora_path,
                },
                "parameters": {
                    "weight": float(m.get("weight", 1.0)),
                },
            }
        else:
            model_entry = {
                "model": str(model_path),
                "parameters": {
                    "weight": float(m.get("weight", 1.0)),
                },
            }
        if "density" in m:
            model_entry["parameters"]["density"] = float(m.density)
        models.append(model_entry)

    mergekit_config = {
        "models": models,
        "merge_method": merge_cfg.method,
        "base_model": cfg.base_model,
        "dtype": merge_cfg.get("dtype", "bfloat16"),
        "parameters": {
            "weight": 1.0,  # Default weight for any models without explicit weight
        },
    }

    if merge_cfg.method in ("ties", "dare_ties"):
        mergekit_config["parameters"]["normalize"] = merge_cfg.get("normalize", True)
        mergekit_config["parameters"]["int8_mask"] = merge_cfg.get("int8_mask", True)

    config = MergeConfiguration.model_validate(mergekit_config)

    output_dir = Path(cfg.output_dir)
    lora_cache = output_dir / ".lora_cache"
    lora_cache.mkdir(parents=True, exist_ok=True)

    options = MergeOptions(
        cuda=merge_cfg.get("cuda", True),
        copy_tokenizer=merge_cfg.get("copy_tokenizer", True),
        low_cpu_memory=merge_cfg.get("low_cpu_memory", False),
        lora_merge_cache=str(lora_cache),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merging {len(models)} models with method: {merge_cfg.method}")
    print(f"Output: {output_dir}")

    try:
        run_merge(
            merge_config=config,
            out_path=str(output_dir),
            options=options,
        )
    finally:
        for tmp_dir in _tmp_dirs:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    # Clean up lora cache directory (may contain files)
    if lora_cache.exists():
        shutil.rmtree(lora_cache)

    print(f"Merge complete: {output_dir}")

    if use_wandb:
        wandb.log(
            {
                "method": merge_cfg.method,
                "num_models": len(models),
                "output_dir": str(output_dir),
                "base_model": cfg.base_model,
            }
        )
        wandb.finish()

    return str(output_dir)
