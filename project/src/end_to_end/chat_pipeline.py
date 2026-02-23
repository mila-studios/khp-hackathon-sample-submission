"""End-to-end chat pipeline with guardrails.

Any guardrail type that meets the guardrail signature (evaluate(content, ...) -> GuardrailResult) is supported. Input/output can be a single guardrail or a
sequence (stack); stacks run in order and short-circuit on first failure.
"""

import time
from typing import Optional, Dict, Any, List, Union, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..guardrails.base import (
    BaseGuardrail,
    GuardrailProtocol,
    GuardrailStatus,
    EvaluationType,
)
from providers.base import BaseLLMProvider, LLMMessage, LLMResponse

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


class PipelineStage(Enum):
    """Stages in the chat pipeline"""
    INPUT_GUARDRAIL = "input_guardrail"
    LLM_GENERATION = "llm_generation"
    OUTPUT_GUARDRAIL = "output_guardrail"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineStatus(Enum):
    """Status of the pipeline execution"""
    SUCCESS = "success"
    BLOCKED = "blocked"
    ERROR = "error"


@dataclass
class PipelineResult:
    """Result from the chat pipeline.
    Guardrail results are always lists (one entry per guardrail in the stack).
    """

    # Final output
    response: Optional[str] = None
    status: PipelineStatus = PipelineStatus.SUCCESS

    # Stage information
    stage: PipelineStage = PipelineStage.COMPLETED
    blocked_at: Optional[PipelineStage] = None
    blocked_at_guardrail_index: Optional[int] = None  # 0-based index of guardrail that failed

    # Guardrail results: list of result dicts, one per guardrail (order matches pipeline stack)
    input_guardrail_results: Optional[List[Dict[str, Any]]] = None
    output_guardrail_results: Optional[List[Dict[str, Any]]] = None

    # LLM information
    llm_response: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    total_latency_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "response": self.response,
            "status": self.status.value,
            "stage": self.stage.value,
            "blocked_at": self.blocked_at.value if self.blocked_at else None,
            "blocked_at_guardrail_index": self.blocked_at_guardrail_index,
            "input_guardrail_results": self.input_guardrail_results,
            "output_guardrail_results": self.output_guardrail_results,
            "llm_response": self.llm_response,
            "timestamp": self.timestamp.isoformat(),
            "total_latency_ms": self.total_latency_ms,
            "error": self.error,
        }


class ChatPipeline:
    """
    End-to-end chat pipeline with guardrails.

    Flow: User Input → Input guardrail(s) → Main LLM → Output guardrail(s) → Response.
    Any guardrail type that matches the guardrail signature is supported; input/output
    can be a single guardrail or a sequence (stack). Stacks run in order and
    short-circuit on first failure.

    Results use list-based guardrail results only:
    - result.input_guardrail_results  (list of dicts, one per input guardrail)
    - result.output_guardrail_results (list of dicts, one per output guardrail)
    - result.blocked_at_guardrail_index when blocked

    Accessors: pipeline.input_guardrails, pipeline.output_guardrails (read-only lists).
    """
    
    def __init__(
        self,
        main_llm_provider: BaseLLMProvider,
        input_guardrail: Optional[GuardrailOrStack] = None,
        output_guardrail: Optional[GuardrailOrStack] = None,
        system_prompt: Optional[str] = None,
        block_on_input_failure: bool = True,
        block_on_output_failure: bool = True,
    ):
        """
        Initialize the chat pipeline
        
        Args:
            main_llm_provider: The main LLM provider (e.g., OpenAI)
            input_guardrail: Optional guardrail(s) for user input. A single guardrail
                (BaseGuardrail or any type implementing the guardrail signature) or a
                sequence of guardrails (stack). Stacks run in order; first failure blocks.
            output_guardrail: Optional guardrail(s) for LLM output. Same as input_guardrail.
            system_prompt: Optional system prompt for the main LLM. If None, no system message is
                added by the pipeline and the provider's system_prompt (if any) is used.
            block_on_input_failure: Whether to block if any input guardrail fails
            block_on_output_failure: Whether to block if any output guardrail fails
        """
        self.main_llm_provider = main_llm_provider
        self._input_guardrails: List[GuardrailProtocol] = _normalize_guardrail_or_stack(
            input_guardrail
        )
        self._output_guardrails: List[GuardrailProtocol] = _normalize_guardrail_or_stack(
            output_guardrail
        )
        self.system_prompt = system_prompt
        self.block_on_input_failure = block_on_input_failure
        self.block_on_output_failure = block_on_output_failure

    @property
    def input_guardrails(self) -> List[GuardrailProtocol]:
        """List of input guardrails (read-only). Pass a single guardrail as [g] or a stack as [g1, g2]."""
        return self._input_guardrails

    @property
    def output_guardrails(self) -> List[GuardrailProtocol]:
        """List of output guardrails (read-only)."""
        return self._output_guardrails

    def process(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> PipelineResult:
        """
        Process user input through the pipeline.

        When any guardrail blocks (input or output), the pipeline returns immediately
        with status BLOCKED and does not run subsequent stages (no further guardrails,
        no LLM call if blocked at input).

        Args:
            user_input: The user's input message
            context: Optional context information
            conversation_history: Optional conversation history [{"role": "user/assistant", "content": "..."}]
            temperature: Optional temperature override for main LLM
            max_tokens: Optional max tokens override for main LLM

        Returns:
            PipelineResult with complete pipeline information
        """
        start_time = time.time()
        result = PipelineResult(stage=PipelineStage.INPUT_GUARDRAIL)

        try:
            # Stage 1: Input Guardrail(s)
            input_results: List[Dict[str, Any]] = []
            for idx, gr in enumerate(self._input_guardrails):
                input_eval = gr.evaluate(
                    content=user_input,
                    context=context,
                    evaluation_type=EvaluationType.USER_INPUT,
                )
                input_results.append(input_eval.to_dict())
                result.input_guardrail_results = input_results
                if input_eval.is_harmful and self.block_on_input_failure:
                    result.status = PipelineStatus.BLOCKED
                    result.stage = PipelineStage.FAILED
                    result.blocked_at = PipelineStage.INPUT_GUARDRAIL
                    result.blocked_at_guardrail_index = idx
                    result.response = self._get_blocked_message("input", input_eval)
                    result.total_latency_ms = (time.time() - start_time) * 1000
                    return result  # Return immediately; do not run LLM or output guardrails

            # Stage 2: Main LLM Generation
            result.stage = PipelineStage.LLM_GENERATION
            messages = []
            if self.system_prompt is not None:
                messages.append(LLMMessage(role="system", content=self.system_prompt))
            if conversation_history:
                for msg in conversation_history:
                    messages.append(LLMMessage(role=msg["role"], content=msg["content"]))
            messages.append(LLMMessage(role="user", content=user_input))

            llm_response = self.main_llm_provider.generate_sync(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            result.llm_response = {
                "content": llm_response.content,
                "model": llm_response.model,
                "usage": llm_response.usage,
                "finish_reason": llm_response.finish_reason,
            }

            # Stage 3: Output Guardrail(s)
            result.stage = PipelineStage.OUTPUT_GUARDRAIL
            output_context = {**(context or {}), "user_input": user_input}
            output_results: List[Dict[str, Any]] = []
            for idx, gr in enumerate(self._output_guardrails):
                output_eval = gr.evaluate(
                    content=llm_response.content,
                    context=output_context,
                    evaluation_type=EvaluationType.LLM_OUTPUT,
                )
                output_results.append(output_eval.to_dict())
                result.output_guardrail_results = output_results
                if output_eval.is_harmful and self.block_on_output_failure:
                    result.status = PipelineStatus.BLOCKED
                    result.stage = PipelineStage.FAILED
                    result.blocked_at = PipelineStage.OUTPUT_GUARDRAIL
                    result.blocked_at_guardrail_index = idx
                    result.response = self._get_blocked_message("output", output_eval)
                    result.total_latency_ms = (time.time() - start_time) * 1000
                    return result  # Return immediately; do not complete pipeline

            result.stage = PipelineStage.COMPLETED
            result.status = PipelineStatus.SUCCESS
            result.response = llm_response.content
            result.total_latency_ms = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            result.status = PipelineStatus.ERROR
            result.stage = PipelineStage.FAILED
            result.error = str(e)
            result.total_latency_ms = (time.time() - start_time) * 1000
            return result

    def _get_blocked_message(self, stage: str, guardrail_result) -> str:
        """Generate a user-friendly blocked message aligned with triage: redirect to human support."""
        if stage == "input":
            message = (
                "I can't continue this conversation here due to safety concerns. "
                "I'm a navigation assistant, not a counsellor or crisis responder—I help you find and connect with the right support. "
                "Would you like me to help you connect with a real person who can support you?"
            )
        else:
            message = (
                "I can't show that response here due to safety concerns. "
                "I'm a navigation assistant—I'm here to connect you with human support, not to provide it myself. "
                "Would you like me to help you connect with a real person who can support you?"
            )
        if guardrail_result.reasoning:
            message += f"\n\nReason: {guardrail_result.reasoning}"
        return message
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration. Guardrail info is always list-based."""
        system_prompt = self.system_prompt
        if system_prompt is None:
            system_prompt = getattr(self.main_llm_provider, "system_prompt", None)
        info = {
            "main_llm_model": self.main_llm_provider.model,
            "input_guardrail_count": len(self._input_guardrails),
            "output_guardrail_count": len(self._output_guardrails),
            "block_on_input_failure": self.block_on_input_failure,
            "block_on_output_failure": self.block_on_output_failure,
            "system_prompt": system_prompt,
            "input_guardrail_configs": [g.config.to_dict() for g in self._input_guardrails],
            "input_guardrail_types": [type(g).__name__ for g in self._input_guardrails],
            "output_guardrail_configs": [g.config.to_dict() for g in self._output_guardrails],
            "output_guardrail_types": [type(g).__name__ for g in self._output_guardrails],
        }
        return info
