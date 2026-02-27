"""Example submission: LLM Judge guardrail using Cohere as the judge LLM.

Requires COHERE_API_KEY (and optionally COHERE_BASE_URL, COHERE_MODEL) in environment.
Use this when you want the safety judge to run on Cohere instead of OpenAI or demo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails.base import GuardrailConfig
from src.guardrails.llm_judge import LLMJudgeGuardrail


def _get_cohere_judge_llm():
    """Build Cohere provider for LLM Judge. Uses COHERE_API_KEY, optional COHERE_BASE_URL and COHERE_MODEL."""
    try:
        from providers.cohere_provider import CohereProvider
    except ImportError:
        return None
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("COHERE_BASE_URL") or None
    model = os.getenv("COHERE_MODEL", "CohereLabs/c4ai-command-a-03-2025")
    return CohereProvider(
        base_url=base_url,
        model=model,
        temperature=0.0,
        max_tokens=1000,
        api_key=api_key,
    )


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail) using Cohere as the LLM judge."""
    judge_llm = _get_cohere_judge_llm()
    if judge_llm is None:
        return (None, None)

    input_guardrail = LLMJudgeGuardrail(
        config=GuardrailConfig(
            name="input_cohere_llm_judge",
            description="LLM Judge for input safety (Cohere)",
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
    if in_gr is None:
        print("Set COHERE_API_KEY to use Cohere LLM judge.")
    else:
        pipeline = ChatPipeline(
            main_llm_provider=DemoProvider(model="demo-model", temperature=0.0, max_tokens=200),
            input_guardrail=in_gr,
            output_guardrail=out_gr,
            system_prompt="You are a helpful assistant.",
        )
        result = pipeline.process("Can you share healthy coping strategies for stress?")
        print("Status:", result.status.value)
        print("Response:", (result.response or "")[:200])
