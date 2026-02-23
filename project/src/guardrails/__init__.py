"""Guardrails: flexible, signature-based content safety evaluation.

Any object that implements the guardrail signature (evaluate(content, ...) -> GuardrailResult, plus config) can be used in the pipeline. Use
GuardrailProtocol for typing; subclass BaseGuardrail for shared helpers.

Supports single or stacked guardrails: pass one guardrail or a sequence
(list/tuple); stacks run in order and short-circuit on first failure.
Results are always list-based (input_guardrail_results, output_guardrail_results).

See README.md in this package for full documentation and examples.
"""

from .base import (
    BaseGuardrail,
    GuardrailProtocol,
    GuardrailResult,
    GuardrailConfig,
    GuardrailStatus,
    EvaluationType,
)
from .llm_judge import LLMJudgeGuardrail
from .classifier import ClassifierGuardrail, load_classifier_guardrail

# Base exports (always available)
__all__ = [
    # Base classes
    "BaseGuardrail",
    "GuardrailProtocol",
    "GuardrailResult",
    "GuardrailConfig",
    "GuardrailStatus",
    "EvaluationType",
    
    # LLM Judge guardrail
    "LLMJudgeGuardrail",
    # Finetunable classifier guardrail (load with load_classifier_guardrail)
    "ClassifierGuardrail",
    "load_classifier_guardrail",
]