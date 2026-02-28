#!/usr/bin/env python3
"""
Template script to train a finetunable classifier for the guardrail using Hugging Face Transformers.

Produces a model saved with model.save_pretrained() / tokenizer.save_pretrained()
that can be loaded with load_classifier_guardrail() and used in the chat pipeline
the same way as the LLM judge guardrail.

Usage:
  # From project/ directory (or repo root with PYTHONPATH=project):
  python -m src.guardrails.train_classifier_guardrail \\
    --data path/to/labeled_data.csv \\
    --output_dir models/my_guardrail \\
    [--base_model bert-base-uncased] [--text_column text --label_column label]

  Other Hugging Face models: roberta-base, distilbert-base-uncased, albert-base-v2, etc.

CSV format:
  - Default columns: "text" (content to classify), "label" (0 = safe, 1 = unsafe/harmful).

Output (Hugging Face format):
  - {output_dir}/config.json, model.safetensors, tokenizer files
  - {output_dir}/guardrail_config.json (threshold, fail_open)

Then in config or code:
  type: "finetunable"
  model_path: "models/my_guardrail"
  threshold: 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Project root = project/ (parent of src/)
_PROJECT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a finetunable classifier guardrail with Hugging Face Transformers."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV with text and label columns (0=safe, 1=unsafe)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save model and tokenizer (e.g. models/output_guardrail)",
    )
    parser.add_argument(
        "--base_model",
        default="bert-base-uncased",
        help="Hugging Face model id for sequence classification (default: bert-base-uncased). "
        "Any compatible model works, e.g. roberta-base, distilbert-base-uncased, albert-base-v2.",
    )
    parser.add_argument(
        "--text_column",
        default="text",
        help="CSV column name for content",
    )
    parser.add_argument(
        "--label_column",
        default="label",
        help="CSV column name for label (0=safe, 1=unsafe)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Default threshold for guardrail (score >= threshold -> FAIL)",
    )
    parser.add_argument(
        "--fail_open",
        action="store_true",
        help="Default fail_open for guardrail",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length for inputs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of data for validation (0 to disable)",
    )
    args = parser.parse_args()

    hackathon_path = _PROJECT.parent / "hackathon.json"
    if not hackathon_path.exists():
        print(f"Required config not found: {hackathon_path}", file=sys.stderr)
        return 1
    try:
        cfg = json.loads(hackathon_path.read_text())
    except Exception as exc:
        print(f"Invalid JSON in {hackathon_path}: {exc}", file=sys.stderr)
        return 1
    if not isinstance(cfg, dict):
        print(f"{hackathon_path} must contain a JSON object", file=sys.stderr)
        return 1
    if "needs_gpu" not in cfg:
        print(f"{hackathon_path} missing required field: needs_gpu", file=sys.stderr)
        return 1
    if not isinstance(cfg["needs_gpu"], bool):
        print(f"{hackathon_path} field 'needs_gpu' must be a boolean", file=sys.stderr)
        return 1
    needs_gpu = cfg["needs_gpu"]
    # Force CPU for local runs when project config says GPU is not required.
    if not needs_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        import torch
        import numpy as np
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
        )
        from transformers import EvalPrediction
    except ImportError:
        print("This script requires transformers and torch.", file=sys.stderr)
        print("Install with: pip install transformers torch", file=sys.stderr)
        return 1

    try:
        import pandas as pd
    except ImportError:
        print("This script requires pandas for CSV loading.", file=sys.stderr)
        print("Install with: pip install pandas", file=sys.stderr)
        return 1

    if needs_gpu and not torch.cuda.is_available():
        print("hackathon.json requires GPU (needs_gpu=true), but CUDA is not available.", file=sys.stderr)
        return 1

    path = Path(args.data)
    if not path.exists():
        print(f"Data file not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path)
    if args.text_column not in df.columns:
        print(f"Text column '{args.text_column}' not in CSV. Columns: {list(df.columns)}", file=sys.stderr)
        return 1
    if args.label_column not in df.columns:
        print(f"Label column '{args.label_column}' not in CSV. Columns: {list(df.columns)}", file=sys.stderr)
        return 1

    texts = df[args.text_column].astype(str).fillna("").tolist()
    labels = df[args.label_column].astype(int).tolist()
    if set(labels) - {0, 1}:
        print("Labels should be 0 (safe) and 1 (unsafe). Found:", sorted(set(labels)), file=sys.stderr)
        return 1

    print(f"Using base model: {args.base_model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label={0: "safe", 1: "unsafe"},
        label2id={"safe": 0, "unsafe": 1},
    )

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_length,
        padding="max_length",
        return_tensors="pt",
    )

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask, labels):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            return {
                "input_ids": self.input_ids[i],
                "attention_mask": self.attention_mask[i],
                "labels": self.labels[i],
            }

    full_dataset = SimpleDataset(enc["input_ids"], enc["attention_mask"], labels)

    if args.test_fraction > 0:
        n = len(full_dataset)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(42))
        n_test = int(n * args.test_fraction)
        train_idx, eval_idx = indices[n_test:], indices[:n_test]
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx.tolist())
        eval_dataset = torch.utils.data.Subset(full_dataset, eval_idx.tolist())
    else:
        train_dataset = full_dataset
        eval_dataset = None

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        load_best_model_at_end=bool(eval_dataset),
        metric_for_best_model="accuracy" if eval_dataset else None,
    )

    def compute_metrics(eval_pred: EvalPrediction):
        preds = np.argmax(eval_pred.predictions, axis=1)
        acc = (preds == eval_pred.label_ids).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics if eval_dataset else None,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    guardrail_config = {
        "threshold": args.threshold,
        "fail_open": args.fail_open,
    }
    config_path = Path(args.output_dir) / "guardrail_config.json"
    with open(config_path, "w") as f:
        json.dump(guardrail_config, f, indent=2)
    print(f"Saved guardrail config to {config_path}")

    print("\nTo use this guardrail:")
    print(f"  1. In YAML: type: \"finetunable\", model_path: \"{args.output_dir}\"")
    print("  2. In code: load_classifier_guardrail(model_path=..., name=..., description=..., threshold=...)")
    print("  3. In ChatPipeline: pass as input_guardrail or output_guardrail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
