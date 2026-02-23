"""Base classes for guardrails"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Protocol, runtime_checkable
from datetime import datetime
from enum import Enum


class GuardrailStatus(Enum):
    """Status of guardrail evaluation"""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class EvaluationType(Enum):
    """Type of content being evaluated"""
    USER_INPUT = "user_input"  # Guardrail for user queries/prompts
    LLM_OUTPUT = "llm_output"   # Guardrail for LLM responses


@dataclass
class GuardrailResult:
    """Result of a guardrail evaluation"""
    
    status: GuardrailStatus
    score: Optional[float] = None  # 0.0–1.0; convention is guardrail-specific (e.g. classifier: higher = more harmful/risk)
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: Optional[float] = None
    
    @property
    def is_harmful(self) -> bool:
        """Check if the result is harmful or unsafe"""
        return self.status != GuardrailStatus.PASS
    
    @property
    def is_violation(self) -> bool:
        """Check if there's a violation"""
        return self.status == GuardrailStatus.FAIL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "score": self.score,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
        }


@dataclass
class GuardrailConfig:
    """Configuration for a guardrail"""
    
    name: str
    description: str
    threshold: float = 0.5  # Score threshold for pass/fail
    enabled: bool = True
    fail_open: bool = False  # If True, allow on errors
    max_retries: int = 3
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "threshold": self.threshold,
            "enabled": self.enabled,
            "fail_open": self.fail_open,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
        }


@runtime_checkable
class GuardrailProtocol(Protocol):
    """
    Protocol for guardrails: any object with evaluate(content, ...) -> GuardrailResult
    and config can be used in the pipeline (structural subtyping).
    """

    config: GuardrailConfig

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        ...


class BaseGuardrail(ABC):
    """Base class for all guardrails. Subclasses implement evaluate()."""

    def __init__(self, config: GuardrailConfig):
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        """
        Evaluate content against the guardrail.

        Args:
            content: The content to evaluate
            context: Optional context (e.g. user input, conversation history)
            evaluation_type: USER_INPUT or LLM_OUTPUT

        Returns:
            GuardrailResult with status, score, reasoning, metadata.
        """
        pass
    
    def _create_result(
        self,
        status: GuardrailStatus,
        score: Optional[float] = None,
        reasoning: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GuardrailResult:
        """Helper method to create a GuardrailResult"""
        return GuardrailResult(
            status=status,
            score=score,
            reasoning=reasoning,
            metadata=metadata or {},
        )