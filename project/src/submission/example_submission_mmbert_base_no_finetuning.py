"""Example submission: mmBERT-base from Hub with no finetuning.

Uses jhu-clsp/mmBERT-base directly from Hugging Face Hub. No training step;
the model is loaded as a sequence classifier (classification head is
initialized randomly if the Hub model is encoder-only). Useful for
quick demos or as a baseline before finetuning.

Contract: get_guardrails() -> (input_guardrail, output_guardrail).
"""

from __future__ import annotations

import sys
import torch
from pathlib import Path
from typing import Any, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails.classifier import load_classifier_guardrail


# Hugging Face Hub model id — no local finetuned checkpoint
BASE_MODEL_ID = "jhu-clsp/mmBERT-base"


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail). Uses base model from Hub, no finetuning."""
    input_guardrail = load_classifier_guardrail(
        model_path=BASE_MODEL_ID,
        name="input_mmbert_base_no_finetuning",
        description="mmBERT-base (jhu-clsp/mmBERT-base) from Hub, no finetuning",
        threshold=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fail_open=False,
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
    result = pipeline.process("Can you share healthy coping strategies for stress?")
    print("Status:", result.status.value)
    print("Response:", (result.response or "")[:200])
