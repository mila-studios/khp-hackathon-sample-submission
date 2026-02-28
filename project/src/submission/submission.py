"""Canonical participant submission module.

Participants should edit this file and implement get_guardrails().
The shared predict/evaluator runners load this exact path.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from src.submission.example_submission import get_guardrails as _default_get_guardrails


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail)."""
    return _default_get_guardrails()
