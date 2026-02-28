"""Example submission: finetuned DistilBERT guardrail via get_guardrails().

Hackathon participants can copy this file and update:
- model_path to their own finetuned model directory
- threshold/device settings
- whether to guard input, output, or both
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

# Ensure project root is importable when loaded by runner or notebook
_THIS_DIR = Path(__file__).resolve().parent
# Project dir = parent of src/ (contains scripts/, models/, src/)
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.guardrails.classifier import load_classifier_guardrail
from src.submission._runtime_config import resolve_device_from_hackathon


def get_guardrails() -> Tuple[Optional[Any], Optional[Any]]:
    """Return (input_guardrail, output_guardrail) for ChatPipeline."""
    # Update this path to your own finetuned model directory (under project/).
    # Example training command (run from project/ or with full paths):
    # python -m src.guardrails.train_classifier_guardrail \
    #   --data ../datasets/distilbert_demo_data.csv \
    #   --output_dir models/distilbert_guardrail_demo \
    #   --base_model distilbert-base-uncased
    model_path = _PROJECT_ROOT / "models" / "distilbert_guardrail_demo"

    if not model_path.exists():
        # Keep notebook/runner usable even before training.
        return (None, None)

    device = resolve_device_from_hackathon(_PROJECT_ROOT)
    input_guardrail = load_classifier_guardrail(
        model_path=str(model_path),
        name="input_distilbert_guardrail",
        description="Finetuned DistilBERT input safety guardrail",
        threshold=0.3,  # block when P(unsafe) >= 0.3; use 0.5 for fewer false positives
        device=device,
        fail_open=False,
    )

    # Example: input-only setup. Change to another guardrail if needed.
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
