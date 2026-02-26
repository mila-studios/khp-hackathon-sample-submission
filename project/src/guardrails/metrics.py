"""Compute precision, recall, F1 and latency for guardrails (single or stacked).

Works with any guardrail that implements the guardrail protocol (e.g. LLM Judge,
ClassifierGuardrail). For stacked guardrails, predictions are combined: the
stack is considered to predict "harmful" if any guardrail in the stack returns
a harmful result; then metrics are computed on that combined prediction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from .base import GuardrailProtocol, GuardrailStatus, EvaluationType


# Single guardrail or a stack (sequence) of guardrails
GuardrailOrStack = Union[GuardrailProtocol, Sequence[GuardrailProtocol]]


def _normalize_guardrail_or_stack(
    guardrail: Optional[GuardrailOrStack],
) -> List[GuardrailProtocol]:
    """Normalize a single guardrail or sequence to a list (empty if None)."""
    if guardrail is None:
        return []
    if isinstance(guardrail, (list, tuple)):
        return list(guardrail)
    return [guardrail]


def _label_to_bool(label: Any) -> bool:
    """Convert various label formats to bool (True = harmful)."""
    if isinstance(label, bool):
        return label
    if isinstance(label, int):
        return label != 0
    if isinstance(label, str):
        v = label.strip().lower()
        if v in ("yes", "true", "1", "harmful", "unsafe"):
            return True
        if v in ("no", "false", "0", "safe"):
            return False
    return bool(label)


@dataclass
class GuardrailMetricsResult:
    """Result of guardrail metrics computation."""

    precision: float
    recall: float
    f1: float
    support_harmful: int  # number of true harmful samples
    support_safe: int     # number of true safe samples
    total_samples: int
    # Latency (ms): for the whole stack per sample
    latency_ms_mean: Optional[float] = None
    latency_ms_total: Optional[float] = None
    latency_ms_per_sample: Optional[List[float]] = None
    # Optional: per-guardrail names (for stacked)
    guardrail_names: List[str] = field(default_factory=list)


def _sanitize_csv_column(name: str) -> str:
    """Make a string safe for use as a CSV column header (no commas, newlines)."""
    return (name or "").replace(",", "_").replace("\n", " ").replace("\r", " ").strip() or "guardrail"


def get_predictions(
    guardrail: GuardrailOrStack,
    evaluation_data: List[Dict[str, Any]],
    evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    context: Optional[Dict[str, Any]] = None,
    *,
    content_key: str = "content",
    label_key: str = "label",
    include_latency: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run a guardrail (or stack) on labeled data and return per-sample predictions.

    For stacked guardrails: runs all guardrails on each sample (no short-circuit),
    then combines so that the stack predicts "harmful" if any guardrail returns
    a harmful result. Use compute_metrics_from_predictions() to get precision,
    recall, F1 and latency from the returned list.

    Args:
        guardrail: A single guardrail or a list/tuple of guardrails (stack).
        evaluation_data: List of dicts. Each must have content (or content_key)
            and label (or label_key). Optional: "evaluation_type" per row.
        evaluation_type: Default evaluation type (USER_INPUT or LLM_OUTPUT).
        context: Optional context passed to evaluate().
        content_key: Key in each row for the text to evaluate.
        label_key: Key in each row for the ground-truth label (True/1 = harmful).
        include_latency: Whether to include latency_ms per sample in each row.

    Returns:
        List of dicts per sample with keys: content, label, label_harmful,
        one key per guardrail (sanitized name) with bool, combined_pred (bool),
        and optionally latency_ms.
    """
    stack = _normalize_guardrail_or_stack(guardrail)
    if not stack:
        return []

    guardrail_names_sanitized = [
        _sanitize_csv_column(
            getattr(g, "config", None) and getattr(g.config, "name", "") or type(g).__name__
        )
        for g in stack
    ]
    seen: Dict[str, int] = {}
    unique_names: List[str] = []
    for n in guardrail_names_sanitized:
        count = seen.get(n, 0)
        seen[n] = count + 1
        unique_names.append(f"{n}_{count}" if count else n)

    predictions_list: List[Dict[str, Any]] = []

    for row in evaluation_data:
        content = row.get(content_key) or row.get("text") or row.get("prompt") or ""
        label = row.get(label_key) or row.get("is_harmful")
        label_harmful = _label_to_bool(label)

        et = row.get("evaluation_type")
        if isinstance(et, EvaluationType):
            ev_type = et
        elif isinstance(et, str):
            ev_type = (
                EvaluationType.LLM_OUTPUT
                if (et or "").strip().lower() in ("llm_output", "output")
                else EvaluationType.USER_INPUT
            )
        else:
            ev_type = evaluation_type

        t0 = time.perf_counter()
        any_harmful = False
        pred_row: Dict[str, Any] = {
            "content": str(content).strip(),
            "label": label,
            "label_harmful": label_harmful,
        }
        for gr, col_name in zip(stack, unique_names):
            result = gr.evaluate(
                content=str(content).strip(),
                context=context,
                evaluation_type=ev_type,
            )
            if isinstance(result, dict):
                status = result.get("status")
                if isinstance(status, GuardrailStatus):
                    is_harmful = status != GuardrailStatus.PASS
                else:
                    is_harmful = str(status).lower() == "fail"
            else:
                is_harmful = result.is_harmful
            any_harmful = any_harmful or is_harmful
            pred_row[col_name] = is_harmful
        pred_row["combined_pred"] = any_harmful
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if include_latency:
            pred_row["latency_ms"] = round(elapsed_ms, 4)
        predictions_list.append(pred_row)

    return predictions_list


def _pred_to_bool(v: Any) -> bool:
    """Convert a prediction value from CSV or dict to bool (True = harmful)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "harmful")
    return bool(v)


def compute_metrics_from_predictions(
    predictions: List[Dict[str, Any]],
    *,
    combined_pred_key: str = "combined_pred",
    label_key: str = "label_harmful",
    fallback_label_key: str = "label",
    latency_key: str = "latency_ms",
    guardrail_names: Optional[List[str]] = None,
) -> GuardrailMetricsResult:
    """
    Compute precision, recall, F1 and optional latency from a list of prediction rows.

    Use this when you have already run guardrails and have per-sample predictions
    (e.g. from get_predictions or a previous run). Each row must have the
    ground-truth label (harmful or not) and the combined prediction.

    Args:
        predictions: List of dicts. Each must have combined_pred (or combined_pred_key)
            and label (label_key or fallback_label_key). Optional: latency_key for
            per-sample latency in ms.
        combined_pred_key: Key for the guardrail(s) combined prediction (bool or 0/1).
        label_key: Key for ground-truth harmful flag (bool or 0/1). If missing,
            fallback_label_key is used and converted via _label_to_bool.
        fallback_label_key: Key for raw label when label_key is absent.
        latency_key: Key for per-sample latency in ms (optional).
        guardrail_names: Optional list of guardrail names for the result.

    Returns:
        GuardrailMetricsResult with precision, recall, F1, support, and latency
        stats if latency_key is present in the rows.
    """
    if not predictions:
        return GuardrailMetricsResult(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            support_harmful=0,
            support_safe=0,
            total_samples=0,
            guardrail_names=guardrail_names or [],
        )

    y_true: List[bool] = []
    y_pred: List[bool] = []
    latencies_ms: List[float] = []

    for row in predictions:
        label_val = row.get(label_key)
        if label_val is None:
            label_val = row.get(fallback_label_key)
            label_harmful = _label_to_bool(label_val)
        else:
            label_harmful = _pred_to_bool(label_val)
        pred_val = row.get(combined_pred_key)
        combined_pred = _pred_to_bool(pred_val) if pred_val is not None else False

        y_true.append(label_harmful)
        y_pred.append(combined_pred)

        if latency_key in row and row[latency_key] is not None:
            try:
                latencies_ms.append(float(row[latency_key]))
            except (TypeError, ValueError):
                pass

    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    support_harmful = sum(y_true)
    support_safe = len(y_true) - support_harmful

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    latency_ms_mean = (
        float(sum(latencies_ms)) / len(latencies_ms) if latencies_ms else None
    )
    latency_ms_total = sum(latencies_ms) if latencies_ms else None

    return GuardrailMetricsResult(
        precision=precision,
        recall=recall,
        f1=f1,
        support_harmful=support_harmful,
        support_safe=support_safe,
        total_samples=len(y_true),
        latency_ms_mean=latency_ms_mean,
        latency_ms_total=latency_ms_total,
        latency_ms_per_sample=latencies_ms if latencies_ms else None,
        guardrail_names=guardrail_names or [],
    )
