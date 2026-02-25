"""Shared helpers for loading submission guardrails and evaluation data."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_guardrails_from_module(module_path: Path) -> Tuple[Any, Any]:
    """Load module and return (input_guardrail, output_guardrail) from get_guardrails()."""
    submission_dir = str(module_path.resolve().parent)
    if submission_dir not in sys.path:
        sys.path.insert(0, submission_dir)

    spec = importlib.util.spec_from_file_location("participant_submission", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    get_guardrails = getattr(module, "get_guardrails", None)
    if get_guardrails is None:
        raise RuntimeError(
            "Module must define get_guardrails() -> (input_guardrail, output_guardrail)"
        )
    result = get_guardrails()
    if not isinstance(result, (list, tuple)) or len(result) != 2:
        raise TypeError(
            "get_guardrails() must return (input_guardrail, output_guardrail), tuple or list of length 2"
        )
    return result[0], result[1]


def load_evaluation_data(csv_path: Path) -> List[Dict[str, Any]]:
    """Load CSV with content and label columns. Normalize column names."""
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        for row in reader:
            content = (
                row.get("text") or row.get("content") or row.get("prompt") or ""
            ).strip()
            label = row.get("label") or row.get("is_harmful")
            rows.append({"content": content, "label": label})
    return rows


def write_predictions_csv(
    predictions: List[Dict[str, Any]],
    csv_path: Path,
) -> None:
    """Write per-sample predictions to CSV (each guardrail column + combined_pred)."""
    if not predictions:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(predictions[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            out = {}
            for k, v in row.items():
                if isinstance(v, bool):
                    out[k] = 1 if v else 0
                else:
                    out[k] = v
            writer.writerow(out)
