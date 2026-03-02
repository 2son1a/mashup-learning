"""Dataset preprocessing: registry, loading, and transformation.

Loads raw datasets from HuggingFace, applies chat-template transforms from
:mod:`mashup.dataset_transforms`, and saves processed datasets locally.
"""

import shutil
import warnings
from pathlib import Path

from datasets import DatasetDict, load_dataset

from mashup.dataset_transforms import (
    arc_easy_to_messages,
    commonsense_qa_to_messages,
    hellaswag_to_messages,
    math_qa_to_messages,
    openbookqa_to_messages,
    piqa_to_messages,
    social_iqa_to_messages,
    winogrande_to_messages,
)

DATASETS = {
    "piqa": {
        "hf_path": "piqa",
        "template_fn": piqa_to_messages,
        "kwargs": {},
    },
    "arc_easy": {
        "hf_path": "allenai/ai2_arc",
        "template_fn": arc_easy_to_messages,
        "kwargs": {"name": "ARC-Easy"},
    },
    "social_iqa": {
        "hf_path": "social_i_qa",
        "template_fn": social_iqa_to_messages,
        "kwargs": {},
    },
    "hellaswag": {
        "hf_path": "Rowan/hellaswag",
        "template_fn": hellaswag_to_messages,
        "kwargs": {},
    },
    "commonsense_qa": {
        "hf_path": "tau/commonsense_qa",
        "template_fn": commonsense_qa_to_messages,
        "kwargs": {},
    },
    "math_qa": {
        "hf_path": "allenai/math_qa",
        "template_fn": math_qa_to_messages,
        "kwargs": {},
    },
    "openbookqa": {
        "hf_path": "allenai/openbookqa",
        "template_fn": openbookqa_to_messages,
        "kwargs": {"name": "main"},
    },
    "winogrande": {
        "hf_path": "winogrande",
        "template_fn": winogrande_to_messages,
        "kwargs": {"name": "winogrande_xl"},
    },
}


def preprocess_dataset(
    dataset_name: str,
    output_dir: Path,
    splits: list[str] | None = None,
):
    """Load, preprocess, and save a single dataset (all requested splits).

    Args:
        dataset_name: Name of the dataset to process (must be in DATASETS dict)
        output_dir: Root directory for processed datasets
        splits: Splits to process (default: all available splits)
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}"
        )

    config = DATASETS[dataset_name]
    output_path = output_dir / dataset_name

    print(f"\n{'=' * 60}")
    print(f"Processing: {dataset_name}")
    print(f"HuggingFace path: {config['hf_path']}")
    print(f"Output path: {output_path}")
    print(f"{'=' * 60}\n")

    print("Loading dataset from HuggingFace...")
    ds_dict = load_dataset(config["hf_path"], **config["kwargs"])
    available_splits = list(ds_dict.keys())
    print(f"Available splits: {available_splits}")

    if splits:
        missing = set(splits) - set(available_splits)
        if missing:
            warnings.warn(f"Splits {missing} not found, skipping them", stacklevel=2)
        selected = [s for s in splits if s in available_splits]
    else:
        selected = available_splits

    if not selected:
        raise ValueError(f"No valid splits to process for {dataset_name}")

    processed = {}
    for split_name in selected:
        ds = ds_dict[split_name]
        print(f"\nProcessing split '{split_name}' ({len(ds)} examples)...")
        ds_processed = ds.map(
            config["template_fn"],
            desc=f"Processing {dataset_name}/{split_name}",
            num_proc=4,
        )

        if "messages" not in ds_processed.column_names:
            raise ValueError(
                f"Template function did not create 'messages' field "
                f"for {dataset_name}/{split_name}"
            )

        # Drop samples with empty assistant messages (e.g. test splits
        # with hidden labels) -- they are useless for training and eval.
        before = len(ds_processed)
        ds_processed = ds_processed.filter(lambda x: x["messages"][-1]["content"] != "")
        dropped = before - len(ds_processed)
        if dropped:
            print(f"  Filtered out {dropped} unlabeled samples")

        processed[split_name] = ds_processed

    first_split = next(iter(processed))
    print(f"\nExample processed message (from '{first_split}'):")
    print(f"Messages: {processed[first_split][0]['messages']}")

    # Save as DatasetDict (clean output dir first to avoid stale files
    # from earlier runs that could cause load_from_disk to pick up a flat
    # Dataset instead of the DatasetDict)
    print(f"\nSaving processed dataset to {output_path}")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    DatasetDict(processed).save_to_disk(str(output_path))

    print(f"✓ Successfully processed {dataset_name}")
    for split_name, ds in processed.items():
        print(f"  - {split_name}: {len(ds)} examples, columns: {ds.column_names}")
    print(f"  - Saved to: {output_path}\n")


def preprocess_all(output_dir: Path, splits: list[str] | None = None):
    """Preprocess all configured datasets.

    Args:
        output_dir: Root directory for processed datasets
        splits: Splits to process (default: all available splits)
    """
    print(f"Preprocessing all {len(DATASETS)} datasets...")

    for dataset_name in DATASETS.keys():
        try:
            preprocess_dataset(dataset_name, output_dir, splits=splits)
        except Exception as e:
            print(f"✗ Failed to process {dataset_name}: {e}")
            continue

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Processed datasets saved to: {output_dir}")
    print("=" * 60)
