#!/usr/bin/env python3
"""Run input guardrail on a labeled CSV and write per-sample predictions.

Loads guardrails via get_guardrails() from the submission module, runs the
input guardrail on the data, and writes predictions to CSV. Use
get_guardrail_metrics afterward to compute precision, recall, F1 and
latency from that file.

Usage:
    cd project && PYTHONPATH=. python -m src.guardrails.get_predictions \\
        --submission src/submission/example_submission_llm_judge.py \\
        --data ../datasets/distilbert_demo_data.csv \\
        --output-dir results/

Writes:
  - <output_dir>/predictions_input.csv

CSV format: must have text content (column "text", "content", or "prompt") and
label (column "label" or "is_harmful"). Label: 1/Yes/true = harmful, 0/No/false = safe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# Project root = project/ (parent of src/)
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails import (
    get_predictions,
    load_guardrails_from_module,
    load_evaluation_data,
    write_predictions_csv,
)
from src.guardrails.base import EvaluationType


def run_predictions(
    submission_path: Path,
    data_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Load guardrails from submission, run input guardrail on data CSV,
    write predictions_input.csv to output_dir.
    Returns dict with "input_predictions_path" and "total_samples".
    """
    evaluation_data = load_evaluation_data(data_path)
    if not evaluation_data:
        raise ValueError(f"No rows loaded from {data_path}")

    input_guardrail, _ = load_guardrails_from_module(submission_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "submission": str(submission_path),
        "data_path": str(data_path),
        "total_samples": len(evaluation_data),
        "output_dir": str(output_dir),
    }

    if input_guardrail is not None:
        input_predictions = get_predictions(
            input_guardrail,
            evaluation_data,
            evaluation_type=EvaluationType.USER_INPUT,
            include_latency=True,
            content_key="content",
            label_key="label",
        )
        out_path = output_dir / "predictions.csv"
        write_predictions_csv(input_predictions, out_path)
        result["input_predictions_path"] = str(out_path)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run guardrails on labeled CSV and write prediction CSVs."
    )
    parser.add_argument(
        "--submission",
        "-s",
        required=True,
        help="Path to submission module (defines get_guardrails()).",
    )
    parser.add_argument(
        "--data",
        "-d",
        required=True,
        help="Path to CSV with content and label (e.g. text,label or prompt,is_harmful).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to write predictions.csv.",
    )
    args = parser.parse_args()

    submission_path = Path(args.submission).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not submission_path.exists():
        print(f"Submission not found: {submission_path}", file=sys.stderr)
        return 1
    if not data_path.exists():
        print(f"Data CSV not found: {data_path}", file=sys.stderr)
        return 1

    try:
        result = run_predictions(
            submission_path=submission_path,
            data_path=data_path,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print("Predictions written")
    print("=" * 40)
    print(f"Samples: {result['total_samples']}")
    if result.get("input_predictions_path"):
        print(f"Predictions: {result['input_predictions_path']}")
    print("\nRun get_guardrail_metrics with --predictions-dir to compute metrics.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
