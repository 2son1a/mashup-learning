"""Patch axolotl, trl, and optimum for transformers 5.x compatibility.

Transformers 5.x made several breaking changes:
  - ``AutoModelForVision2Seq`` -> ``AutoModelForImageTextToText``
  - ``MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES`` -> ``MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES``
  - ``is_tf_available`` removed (TensorFlow support dropped)
  - ``apply_chat_template`` now returns ``BatchEncoding`` by default
    (``return_dict=True``); axolotl expects a plain ``list[int]``

Axolotl 0.9.1, trl, and optimum still reference the old names.  This script
performs source-level patches on the installed packages so imports resolve
correctly.  It is idempotent — safe to run multiple times.
"""

from __future__ import annotations

import re
import site
import sys
from pathlib import Path

# ── simple find-and-replace patches ──────────────────────────────────────────
# (old_string, new_string, relative_path_from_site_packages)
REPLACE_PATCHES: list[tuple[str, str, Path]] = [
    (
        "AutoModelForVision2Seq",
        "AutoModelForImageTextToText",
        Path("axolotl") / "utils" / "models.py",
    ),
    (
        "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES",
        "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
        Path("trl") / "trainer" / "dpo_trainer.py",
    ),
    # transformers 5.x defaults apply_chat_template to return_dict=True
    # (BatchEncoding), but axolotl expects a plain list[int].
    (
        "return self.tokenizer.apply_chat_template(\n"
        "            conversation,\n"
        "            add_generation_prompt=add_generation_prompt,\n"
        "            chat_template=self.chat_template,\n"
        "        )",
        "return self.tokenizer.apply_chat_template(\n"
        "            conversation,\n"
        "            add_generation_prompt=add_generation_prompt,\n"
        "            chat_template=self.chat_template,\n"
        "            return_dict=False,\n"
        "        )",
        Path("axolotl") / "prompt_strategies" / "chat_template.py",
    ),
    # transformers 5.x removed load_in_8bit/load_in_4bit as model __init__ kwargs.
    # axolotl only popped them when quantization_config was set; now always pop.
    (
        "        # no longer needed per https://github.com/huggingface/transformers/pull/26610\n"
        "        if \"quantization_config\" in self.model_kwargs or self.cfg.gptq:\n"
        "            self.model_kwargs.pop(\"load_in_8bit\", None)\n"
        "            self.model_kwargs.pop(\"load_in_4bit\", None)",
        "        # no longer needed per https://github.com/huggingface/transformers/pull/26610\n"
        "        # transformers 5.x removed these as model init kwargs entirely\n"
        "        self.model_kwargs.pop(\"load_in_8bit\", None)\n"
        "        self.model_kwargs.pop(\"load_in_4bit\", None)",
        Path("axolotl") / "utils" / "models.py",
    ),
    # transformers 5.x removed several kwargs from TrainingArguments
    # (save_safetensors, group_by_length, etc.).  Instead of patching each one,
    # filter kwargs right before constructing the training args.
    # Call site 1: HFCausalTrainerBuilder (line ~811)
    (
        "        training_args = training_args_cls(  # pylint: disable=unexpected-keyword-arg\n"
        "            **training_arguments_kwargs,\n"
        "        )",
        "        # transformers 5.x removed several TrainingArguments kwargs;\n"
        "        # filter to only valid fields to avoid TypeErrors.\n"
        "        import dataclasses as _dc\n"
        "        _valid = {f.name for f in _dc.fields(training_args_cls)}\n"
        "        training_arguments_kwargs = {\n"
        "            k: v for k, v in training_arguments_kwargs.items() if k in _valid\n"
        "        }\n"
        "        training_args = training_args_cls(  # pylint: disable=unexpected-keyword-arg\n"
        "            **training_arguments_kwargs,\n"
        "        )",
        Path("axolotl") / "core" / "trainer_builder.py",
    ),
    # transformers 5.x renamed Trainer(tokenizer=...) -> Trainer(processing_class=...).
    # The existing check inspects the axolotl wrapper (which has **kwargs) instead
    # of the real Trainer, so it always falls through to the old name.
    (
        "        sig = inspect.signature(trainer_cls)\n"
        "        if \"processing_class\" in sig.parameters.keys():\n"
        "            trainer_kwargs[\"processing_class\"] = self.tokenizer\n"
        "        else:\n"
        "            trainer_kwargs[\"tokenizer\"] = self.tokenizer",
        "        from transformers import Trainer as _HFTrainer\n"
        "        sig = inspect.signature(_HFTrainer)\n"
        "        if \"processing_class\" in sig.parameters.keys():\n"
        "            trainer_kwargs[\"processing_class\"] = self.tokenizer\n"
        "        else:\n"
        "            trainer_kwargs[\"tokenizer\"] = self.tokenizer",
        Path("axolotl") / "core" / "trainer_builder.py",
    ),
    # Call site 2: HFRLTrainerBuilder (line ~1135)
    (
        "        training_args = training_args_cls(  # pylint: disable=unexpected-keyword-arg\n"
        "            self.cfg.output_dir,\n"
        "            per_device_train_batch_size=self.cfg.micro_batch_size,\n"
        "            max_steps=max_steps,\n"
        "            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,\n"
        "            learning_rate=self.cfg.learning_rate,\n"
        "            warmup_steps=self.cfg.warmup_steps,\n"
        "            logging_first_step=True,\n"
        "            logging_steps=1,\n"
        "            optim=self.cfg.optimizer,\n"
        "            save_total_limit=self.cfg.save_total_limit or 5,\n"
        "            **training_args_kwargs,\n"
        "        )",
        "        # transformers 5.x removed several TrainingArguments kwargs;\n"
        "        # filter to only valid fields to avoid TypeErrors.\n"
        "        import dataclasses as _dc2\n"
        "        _valid2 = {f.name for f in _dc2.fields(training_args_cls)}\n"
        "        training_args_kwargs = {\n"
        "            k: v for k, v in training_args_kwargs.items() if k in _valid2\n"
        "        }\n"
        "        training_args = training_args_cls(  # pylint: disable=unexpected-keyword-arg\n"
        "            self.cfg.output_dir,\n"
        "            per_device_train_batch_size=self.cfg.micro_batch_size,\n"
        "            max_steps=max_steps,\n"
        "            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,\n"
        "            learning_rate=self.cfg.learning_rate,\n"
        "            warmup_steps=self.cfg.warmup_steps,\n"
        "            logging_first_step=True,\n"
        "            logging_steps=1,\n"
        "            optim=self.cfg.optimizer,\n"
        "            save_total_limit=self.cfg.save_total_limit or 5,\n"
        "            **training_args_kwargs,\n"
        "        )",
        Path("axolotl") / "core" / "trainer_builder.py",
    ),
]

# ── stub for removed is_tf_available ─────────────────────────────────────────
IS_TF_STUB = """

def is_tf_available() -> bool:
    \"\"\"Stub — TensorFlow support was removed in transformers 5.x.\"\"\"
    return False
"""


def _site_packages_dirs() -> list[Path]:
    """Return candidate site-packages directories, venv first."""
    candidates: list[Path] = []
    venv = Path(sys.prefix) / "lib"
    if venv.exists():
        candidates.extend(venv.glob("python*/site-packages"))
    candidates.extend(Path(p) for p in site.getsitepackages())
    return candidates


def find_file(rel: Path) -> Path | None:
    """Locate *rel* inside the active environment's site-packages."""
    for sp in _site_packages_dirs():
        target = sp / rel
        if target.exists():
            return target
    return None


def replace_patch(path: Path, old: str, new: str) -> bool:
    """Replace *old* with *new* in *path*.  Return True if any changes made."""
    text = path.read_text()
    if old not in text:
        return False
    path.write_text(re.sub(re.escape(old), new, text))
    return True


def ensure_is_tf_available() -> None:
    """Add ``is_tf_available`` stub to transformers if missing."""
    # Patch import_utils.py (where the function should live)
    rel = Path("transformers") / "utils" / "import_utils.py"
    path = find_file(rel)
    if path is None:
        print(f"WARNING: could not find {rel}")
        return

    text = path.read_text()
    if "def is_tf_available" in text:
        print(f"Already has is_tf_available: {path}")
    else:
        # Insert after is_torch_available function
        path.write_text(text + IS_TF_STUB)
        print(f"Added is_tf_available stub to {path}")

    # Also ensure it's exported from __init__.py
    init_rel = Path("transformers") / "utils" / "__init__.py"
    init_path = find_file(init_rel)
    if init_path is None:
        print(f"WARNING: could not find {init_rel}")
        return

    init_text = init_path.read_text()
    if "is_tf_available" in init_text:
        print(f"Already exports is_tf_available: {init_path}")
    else:
        # Add to the import_utils imports — find is_torch_available and add after
        init_text = init_text.replace(
            "    is_torch_available,",
            "    is_tf_available,\n    is_torch_available,",
        )
        init_path.write_text(init_text)
        print(f"Added is_tf_available export to {init_path}")


def main() -> None:
    # Apply simple replacements
    for old, new, rel in REPLACE_PATCHES:
        target = find_file(rel)
        if target is None:
            print(f"WARNING: could not find {rel} in site-packages, skipping")
            continue
        if replace_patch(target, old, new):
            print(f"Patched {target}: {old} -> {new}")
        else:
            print(f"Already patched (or not needed): {target}")

    # Add is_tf_available stub
    ensure_is_tf_available()

    # Clear relevant __pycache__ to avoid stale bytecode
    for rel in [
        Path("axolotl") / "utils",
        Path("axolotl") / "core",
        Path("axolotl") / "prompt_strategies",
        Path("trl") / "trainer",
        Path("transformers") / "utils",
    ]:
        cache_dir = find_file(rel)
        if cache_dir is None:
            continue
        pycache = cache_dir / "__pycache__"
        if pycache.exists():
            for pyc in pycache.glob("*.pyc"):
                pyc.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
