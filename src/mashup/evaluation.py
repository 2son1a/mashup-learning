"""
Evaluation of language models on preprocessed chat datasets.

Measures how well a model predicts the final assistant turn in each
conversation.  Uses ``apply_chat_template`` to cleanly separate prompt
from answer tokens -- no per-tokenizer hacks needed.
"""

import json
import os
import re
from pathlib import Path

import torch
from cut_cross_entropy import linear_cross_entropy
from datasets import DatasetDict, load_from_disk
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_tokenizer(base_model: str, model_path: str | None) -> "AutoTokenizer":
    """Load a tokenizer that has a chat template.

    Tries several sources because merged models may lack a tokenizer
    and base models (e.g. Llama-3.2-1B) may lack a chat template:

      1. model_path  (LoRA adapters save the tokenizer with the template)
      2. base_model
      3. shared tokenizer at run_dir/tokenizer/ (saved by pipeline)
      4. chat_template.jinja file in model_path's parent directory
    """
    for source in filter(None, [model_path, base_model]):
        try:
            tokenizer = AutoTokenizer.from_pretrained(source)
            if tokenizer.chat_template:
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                return tokenizer
        except Exception:
            continue

    # No chat template found yet.  Load the base tokenizer and look for
    # a shared tokenizer or chat_template.jinja in the run directory.
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_path:
        parent = Path(model_path).parent

        shared = parent / "tokenizer"
        if shared.is_dir():
            try:
                shared_tok = AutoTokenizer.from_pretrained(str(shared))
                if shared_tok.chat_template:
                    tokenizer.chat_template = shared_tok.chat_template
                    print(f"Using shared tokenizer from {shared}")
                    return tokenizer
            except Exception:
                pass

        # Fall back to standalone chat_template.jinja in sibling dirs
        for jinja_file in sorted(parent.glob("*/chat_template.jinja")):
            tokenizer.chat_template = jinja_file.read_text()
            print(f"Using chat template from {jinja_file}")
            return tokenizer

    raise ValueError(
        f"No tokenizer with a chat template found. "
        f"Tried model_path={model_path}, base_model={base_model}, "
        f"and searched for shared tokenizer and chat_template.jinja "
        f"in the run directory."
    )


def _maybe_resize_embeddings(model, adapter_path: str) -> None:
    """Resize model embeddings if the LoRA adapter was trained with a resized vocab.

    Axolotl may resize embeddings to accommodate special tokens (e.g. pad_token).
    When that happens, the adapter checkpoint includes ``embed_tokens`` and
    ``lm_head`` weights with a larger vocab size than the base model.  We detect
    this and resize the base model *before* ``PeftModel.from_pretrained`` so that
    the state dict shapes match.
    """
    from safetensors import safe_open

    safetensor_path = Path(adapter_path) / "adapter_model.safetensors"
    if not safetensor_path.exists():
        return

    f = safe_open(str(safetensor_path), framework="pt")
    adapter_vocab_size = None
    for key in f.keys():
        if "embed_tokens" in key or "lm_head" in key:
            adapter_vocab_size = f.get_tensor(key).shape[0]
            break

    if adapter_vocab_size is None:
        return  # adapter does not contain embedding layers

    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    if adapter_vocab_size != current_vocab_size:
        print(
            f"Resizing embeddings: {current_vocab_size} → {adapter_vocab_size} "
            f"(adapter was trained with resized vocab)"
        )
        model.resize_token_embeddings(adapter_vocab_size)


def _resolve_base_model(base_model: str, model_path: str | None) -> str:
    """Resolve the actual base model path for loading.

    When ``base_model`` points to a LoRA adapter directory (e.g. because
    the pipeline config set ``base_model: trained_merged_{task}``), we
    read the adapter's ``base_model_name_or_path`` so that
    ``AutoModelForCausalLM.from_pretrained`` loads the true base weights
    instead of triggering auto-adapter-loading (which would fail on
    embedding-size mismatches).
    """
    import json

    adapter_cfg_path = Path(base_model) / "adapter_config.json"
    if adapter_cfg_path.exists():
        data = json.loads(adapter_cfg_path.read_text())
        resolved = data.get("base_model_name_or_path", base_model)
        if resolved != base_model:
            print(f"Resolved base_model: {base_model} → {resolved}")
        return resolved
    return base_model


def _load_model(
    base_model: str,
    model_path: str | None,
    dtype: torch.dtype,
):
    """Load a causal LM, auto-detecting LoRA adapters vs full models."""
    is_lora = model_path and (Path(model_path) / "adapter_config.json").exists()

    if is_lora:
        from peft import PeftModel

        resolved_base = _resolve_base_model(base_model, model_path)
        base = AutoModelForCausalLM.from_pretrained(
            resolved_base,
            torch_dtype=dtype,
            device_map="auto",
        )
        _maybe_resize_embeddings(base, model_path)
        model = PeftModel.from_pretrained(base, model_path)
        print(f"Loaded LoRA adapter: {model_path} on {resolved_base}")
    elif model_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )
        print(f"Loaded model: {model_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
        )
        print(f"Loaded base model: {base_model}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _discover_checkpoints(model_path: str) -> list[tuple[int, Path]]:
    """Find ``checkpoint-*`` directories and return sorted ``(step, path)`` tuples."""
    model_dir = Path(model_path)
    checkpoints = []
    for d in model_dir.iterdir():
        if d.is_dir():
            m = re.match(r"checkpoint-(\d+)$", d.name)
            if m:
                checkpoints.append((int(m.group(1)), d))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def _tokenize_chat(messages: list[dict], tokenizer) -> dict:
    """Tokenize a conversation, masking prompt tokens in labels.

    Uses the chat template to find where the answer starts:
    full conversation minus prompt-only gives us the answer boundary.
    Trailing template tokens (newlines, EOS, EOT) are masked out so
    only actual answer content is evaluated.
    """
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=False,
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
    )
    prompt_len = len(prompt_ids)
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    # Trim trailing template/formatting tokens from labels.
    # Chat templates append tokens like \n, <end_of_turn>, <eos> after the
    # assistant content.  These should not count as "answer tokens" because
    # a model correctly emits EOS/EOT to stop rather than reproducing the
    # template's formatting suffix.
    #
    # We use decode-based whitespace detection instead of matching token IDs
    # from tokenizer.encode("\n"), because BPE tokenization is context-
    # dependent: the token ID for "\n" in isolation may differ from the one
    # produced inside the actual chat-template output.
    _special = set(tokenizer.all_special_ids)
    if hasattr(tokenizer, "added_tokens_decoder"):
        for tid, tok in tokenizer.added_tokens_decoder.items():
            if tok.special:
                _special.add(tid)

    i = len(labels) - 1
    while i >= prompt_len:
        if labels[i] == -100 or labels[i] in _special:
            labels[i] = -100
            i -= 1
            continue
        if tokenizer.decode([labels[i]]).strip() == "":
            labels[i] = -100
            i -= 1
            continue
        break

    return {"input_ids": full_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _print_sanity_check(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    tokenizer,
    max_samples: int = 5,
) -> None:
    """Decode and print predicted vs expected tokens for the first batch."""
    n = min(preds.size(0), max_samples)
    print(f"\n{'─' * 60}")
    print("Sanity check (first batch, answer tokens only)")
    print(f"{'─' * 60}")
    for i in range(n):
        m = mask[i]
        if not m.any():
            continue
        pred_ids = preds[i][m].cpu().tolist()
        label_ids = labels[i][m].cpu().tolist()
        pred_text = tokenizer.decode(pred_ids)
        label_text = tokenizer.decode(label_ids)
        match = "ok" if pred_ids == label_ids else "MISMATCH"
        print(f"  [{i}] expected: {label_text!r}")
        print(f"       got:      {pred_text!r}  ({match})")
    print(f"{'─' * 60}\n")


@torch.inference_mode()
def _evaluate_single(
    model,
    tokenizer,
    tokenized: list[dict],
    num_samples: int,
    batch_size: int = 8,
    min_accuracy: float | None = None,
) -> dict:
    """Run the forward loop on pre-tokenized data and return metrics.

    Extracted so it can be called once per checkpoint without duplicating
    the forward loop.  Returns raw metrics (no dataset/model metadata).
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    is_first_batch = True

    total_correct = 0
    total_tokens = 0
    exact_matches = 0
    losses: list[float] = []

    # Special tokens (eot_id, eos, bos, …) should not count toward
    # accuracy / exact-match — the model may validly emit any stop token.
    # NOTE: tokenizer.all_special_ids may omit chat-template tokens like
    # <|eot_id|> that are marked special in added_tokens_decoder but not
    # listed in additional_special_tokens (common for base models).
    _special_set = set(tokenizer.all_special_ids)
    if hasattr(tokenizer, "added_tokens_decoder"):
        for tok_id, tok in tokenizer.added_tokens_decoder.items():
            if tok.special:
                _special_set.add(tok_id)
    _special_ids = torch.tensor(
        sorted(_special_set),
        dtype=torch.long,
        device=device,
    )

    _base = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Multimodal models (e.g. Gemma3ForConditionalGeneration) nest the
    # causal LM under .language_model
    if hasattr(_base, "language_model"):
        _base = _base.language_model
    _backbone, _lm_head = _base.model, _base.lm_head

    for start in tqdm(range(0, len(tokenized), batch_size), desc="Evaluating"):
        samples = tokenized[start : start + batch_size]
        max_len = max(len(s["input_ids"]) for s in samples)

        input_ids = torch.full((len(samples), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros(len(samples), max_len, dtype=torch.long)
        labels = torch.full((len(samples), max_len), -100, dtype=torch.long)
        for i, s in enumerate(samples):
            ids = torch.tensor(s["input_ids"], dtype=torch.long)
            lab = torch.tensor(s["labels"], dtype=torch.long)
            input_ids[i, : len(ids)] = ids
            attn_mask[i, : len(ids)] = 1
            labels[i, : len(lab)] = lab

        batch = {
            "input_ids": input_ids.to(device),
            "attention_mask": attn_mask.to(device),
            "labels": labels.to(device),
        }
        # Hidden states from backbone (skip lm_head → no (B,T,V) logits)
        hidden = _backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )[0]

        # Loss via CCE — fused linear + cross-entropy, never builds logits
        loss = linear_cross_entropy(
            hidden,
            _lm_head.weight,
            batch["labels"],
            shift=True,
        ).item()
        losses.append(loss)

        # Causal shift + chunked argmax for predictions
        shift_labels = batch["labels"][:, 1:]
        shift_hidden = hidden[:, :-1]
        del hidden
        preds = torch.cat(
            [
                _lm_head(shift_hidden[:, c : c + 128]).argmax(dim=-1)
                for c in range(0, shift_hidden.size(1), 128)
            ],
            dim=1,
        )
        del shift_hidden

        mask = (shift_labels != -100) & ~torch.isin(shift_labels, _special_ids)

        if is_first_batch:
            _print_sanity_check(preds, shift_labels, mask, tokenizer)
            is_first_batch = False

        if mask.any():
            total_correct += (preds[mask] == shift_labels[mask]).sum().cpu().item()
            total_tokens += mask.sum().cpu().item()

            for i in range(shift_labels.size(0)):
                m = mask[i]
                if m.any():
                    exact_matches += int((preds[i][m] == shift_labels[i][m]).all())

        if min_accuracy and total_tokens > 0:
            if total_correct / total_tokens < min_accuracy:
                print(
                    f"Early stop: accuracy {total_correct / total_tokens:.4f} < {min_accuracy}"
                )
                break

    # ── aggregate ─────────────────────────────────────────────────────
    accuracy = total_correct / total_tokens if total_tokens else 0.0
    exact_match = exact_matches / num_samples if num_samples else 0.0
    mean_nll = sum(losses) / len(losses) if losses else 0.0
    perplexity = torch.exp(torch.tensor(mean_nll)).item()

    return {
        "accuracy": round(accuracy, 6),
        "exact_match": round(exact_match, 6),
        "perplexity": round(perplexity, 4),
        "nll": round(mean_nll, 6),
        "num_samples": num_samples,
        "num_answer_tokens": total_tokens,
    }


@torch.inference_mode()
def evaluate(cfg: DictConfig) -> dict | list[dict]:
    """Evaluate a model on a preprocessed chat dataset.

    Config keys::

        base_model          HuggingFace model ID
        model_path          path to LoRA adapter or merged model  (optional)
        datasets            list with at least one ``{path: ...}`` entry
        eval.batch_size     default 8
        eval.max_samples    cap on dataset size          (optional)
        eval.min_accuracy   early-stop threshold          (optional)
        eval.dtype          "bfloat16" | "float16" | ...  (default bfloat16)
        eval.checkpoints    evaluate all training checkpoints  (default false)
        output_dir          where to write eval_results.json

    When ``eval.checkpoints`` is true and ``model_path`` contains
    ``checkpoint-*`` directories, each checkpoint is evaluated using the
    same base model (LoRA adapters are swapped efficiently).  Per-checkpoint
    results are saved in subdirectories, and an ``all_checkpoints.json``
    aggregates everything for learning-curve analysis.

    Returns a metrics dict (single eval) or list of dicts (checkpoint eval).
    """
    eval_cfg = cfg.get("eval", {})
    dtype = getattr(torch, eval_cfg.get("dtype", "bfloat16"))
    batch_size = eval_cfg.get("batch_size", 8)
    max_samples = eval_cfg.get("max_samples")
    min_accuracy = eval_cfg.get("min_accuracy")
    eval_checkpoints = eval_cfg.get("checkpoints", False)

    model_path = cfg.get("model_path")
    output_dir = Path(cfg.output_dir)

    # ── dataset ────────────────────────────────────────────────────────
    dataset_path = cfg.datasets[0].path
    split = eval_cfg.get("split", cfg.datasets[0].get("split"))

    ds = load_from_disk(dataset_path)
    if isinstance(ds, DatasetDict):
        available = list(ds.keys())
        if split and split in ds:
            ds = ds[split]
        elif split:
            raise ValueError(
                f"Split '{split}' not found in {dataset_path}. "
                f"Available splits: {available}"
            )
        else:
            split = available[0]
            ds = ds[split]
        print(f"Using split '{split}' from {dataset_path} (available: {available})")
    else:
        if split:
            raise ValueError(
                f"Split '{split}' requested but {dataset_path} is a flat dataset "
                f"with no splits. Re-run preprocessing to create proper splits."
            )
        print(f"Loaded flat dataset from {dataset_path}")

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    print(f"Evaluating on {len(ds)} samples")

    # ── tokenizer + tokenize ──────────────────────────────────────────
    tokenizer = _load_tokenizer(cfg.base_model, model_path)
    tokenized = [_tokenize_chat(row["messages"], tokenizer) for row in ds]

    # ── discover checkpoints ──────────────────────────────────────────
    checkpoints = []
    if eval_checkpoints and model_path:
        checkpoints = _discover_checkpoints(model_path)
        if not checkpoints:
            print("No checkpoints found, evaluating final model only")

    # ── checkpoint mode ──────────────────────────────────────────────
    if checkpoints:
        first_ckpt = checkpoints[0][1]
        is_lora_ckpt = (first_ckpt / "adapter_config.json").exists()

        targets = [(step, str(p), p.name) for step, p in checkpoints]
        targets.append((None, str(model_path), "final"))

        all_results = []

        if is_lora_ckpt:
            # LoRA adapters: load base once, swap adapters efficiently
            from peft import PeftModel

            base_model_name = _resolve_base_model(str(cfg.base_model), model_path)
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=dtype,
                device_map="auto",
            )
            _maybe_resize_embeddings(base, model_path)
            print(f"Loaded base model once: {base_model_name}")

            for step, adapter_path, ckpt_name in targets:
                print(f"\n{'━' * 60}")
                print(
                    f"  {'Final model' if step is None else f'Checkpoint: {ckpt_name} (step {step})'}"
                )
                print(f"{'━' * 60}")

                model = PeftModel.from_pretrained(base, adapter_path)
                model.eval()
                metrics = _evaluate_single(
                    model, tokenizer, tokenized, len(ds), batch_size, min_accuracy
                )
                results = {
                    **metrics,
                    "checkpoint": ckpt_name,
                    "step": step,
                    "dataset": str(dataset_path),
                    "split": split or "flat",
                    "model": str(model_path),
                    "base_model": base_model_name,
                }
                _print_results(results)

                dest = output_dir if step is None else output_dir / ckpt_name
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "eval_results.json").write_text(json.dumps(results, indent=2))
                print(f"Results saved to {dest / 'eval_results.json'}")

                all_results.append(results)
                base = model.unload()
        else:
            # Full model checkpoints: load each independently
            print("Full model checkpoints detected, loading each independently")

            for step, ckpt_path, ckpt_name in targets:
                print(f"\n{'━' * 60}")
                print(
                    f"  {'Final model' if step is None else f'Checkpoint: {ckpt_name} (step {step})'}"
                )
                print(f"{'━' * 60}")

                model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path,
                    torch_dtype=dtype,
                    device_map="auto",
                )
                model.eval()
                metrics = _evaluate_single(
                    model, tokenizer, tokenized, len(ds), batch_size, min_accuracy
                )
                results = {
                    **metrics,
                    "checkpoint": ckpt_name,
                    "step": step,
                    "dataset": str(dataset_path),
                    "split": split or "flat",
                    "model": str(model_path),
                    "base_model": str(cfg.base_model),
                }
                _print_results(results)

                dest = output_dir if step is None else output_dir / ckpt_name
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "eval_results.json").write_text(json.dumps(results, indent=2))
                print(f"Results saved to {dest / 'eval_results.json'}")

                all_results.append(results)
                del model

        all_ckpt_path = output_dir / "all_checkpoints.json"
        all_ckpt_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nAll checkpoint results saved to {all_ckpt_path}")

        if os.environ.get("WANDB_API_KEY"):
            _log_wandb(cfg, all_results, tags=["eval", "checkpoints"])

        return all_results

    # ── single model eval (default) ──────────────────────────────────
    model = _load_model(cfg.base_model, model_path, dtype)
    metrics = _evaluate_single(
        model, tokenizer, tokenized, len(ds), batch_size, min_accuracy
    )
    results = {
        **metrics,
        "dataset": str(dataset_path),
        "split": split or "flat",
        "model": str(cfg.get("model_path", cfg.base_model)),
        "base_model": str(cfg.base_model),
    }
    _print_results(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved to {output_dir / 'eval_results.json'}")

    if os.environ.get("WANDB_API_KEY"):
        _log_wandb(cfg, [results], tags=["eval"])

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_results(results: dict) -> None:
    ckpt = (
        f"  Checkpoint     : {results['checkpoint']}\n"
        if "checkpoint" in results
        else ""
    )
    print(
        f"\n{'=' * 50}\n"
        f"  Token accuracy : {results['accuracy']:.4f}\n"
        f"  Exact match    : {results['exact_match']:.4f}\n"
        f"  Perplexity     : {results['perplexity']:.2f}\n"
        f"  NLL            : {results['nll']:.4f}\n"
        f"  Samples        : {results['num_samples']}\n"
        f"  Answer tokens  : {results['num_answer_tokens']}\n"
        f"{ckpt}"
        f"{'=' * 50}"
    )


def _log_wandb(cfg: DictConfig, results: list[dict], tags: list[str]) -> None:
    import wandb

    wandb.init(
        project=cfg.get("wandb_project", "mashup"),
        name=cfg.get("wandb_name"),
        group=cfg.get("wandb_group"),
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=tags,
    )
    for r in results:
        wandb.log(r)
    wandb.finish()
