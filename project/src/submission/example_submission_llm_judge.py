"""Example submission module: LLM Judge guardrails via get_guardrails().

Hackathon participants can copy this file and tweak:
- provider/model selection in _get_guardrail_llm()
- thresholds/fail_open in get_guardrails()
- whether to return input, output, or both guardrails
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

# Ensure project root is importable when run directly
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails.base import GuardrailConfig
from src.guardrails.llm_judge import LLMJudgeGuardrail

OPENAI_GUARDRAIL_MODEL = "gpt-4o-mini"
COHERE_GUARDRAIL_MODEL = "command-r"


def _get_guardrail_llm(provider: str):
    """Choose an LLM provider for LLM Judge guardrails."""
    if provider == "demo":
        try:
            from providers.demo_provider import DemoProvider
            return DemoProvider(model="demo-model", temperature=0.0, max_tokens=1000)
        except Exception:
            pass
    elif provider == "openai":
        try:
            from providers.openai_provider import OpenAIProvider
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return OpenAIProvider(
                    api_key=api_key,
                    model=OPENAI_GUARDRAIL_MODEL,
                    temperature=1.0,
                    max_tokens=1000,
                )
        except Exception:
            pass
    elif provider == "cohere":
        try:
            from providers.cohere_provider import CohereProvider
            api_key = os.getenv("COHERE_API_KEY")
            if api_key:
                return CohereProvider(
                    api_key=api_key,
                    model=COHERE_GUARDRAIL_MODEL,
                    temperature=0.0,
                    max_tokens=1000,
                )
        except Exception:
            pass
    return None 


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail) for ChatPipeline."""
    judge_llm = _get_guardrail_llm(provider="openai")
    print(judge_llm)
    if judge_llm is None:
        return (None, None)

    input_guardrail = LLMJudgeGuardrail(
        config=GuardrailConfig(
            name="input_llm_judge",
            description="LLM Judge for input safety",
            threshold=0.6,
            fail_open=False,
            max_retries=2,
        ),
        llm_provider=judge_llm,
    )

    output_guardrail = None
    return (input_guardrail, output_guardrail)


if __name__ == "__main__":
    from providers.demo_provider import DemoProvider
    from src.end_to_end.chat_pipeline import ChatPipeline

    in_gr, out_gr = get_guardrails()
    pipeline = ChatPipeline(
        main_llm_provider=DemoProvider(model="demo-model", temperature=0.0, max_tokens=200),
        input_guardrail=in_gr,
        output_guardrail=out_gr,
        system_prompt="You are a helpful assistant.",
    )
    r = pipeline.process("Can you share healthy coping strategies for stress?")
    print("Status:", r.status.value)
    print("Response:", (r.response or "")[:200])
