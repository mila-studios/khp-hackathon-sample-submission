"""Finetunable classifier guardrail using Hugging Face Transformers.

Same contract as LLMJudgeGuardrail: evaluate(content, context, evaluation_type) -> GuardrailResult.
Compatible with the chat pipeline and guardrails_loader (type: "finetunable" or "bert").
Uses only the Hugging Face transformers library (no sklearn).
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import (
    BaseGuardrail,
    GuardrailConfig,
    GuardrailResult,
    GuardrailStatus,
    EvaluationType,
)

# Artifact written by training script (transformers save_pretrained)
CONFIG_FILENAME = "guardrail_config.json"


def _load_transformers_pipeline(model_path: str, device: str = "cpu"):
    """
    Load a Hugging Face text-classification pipeline.

    Supports:
    - Local directory: path to a model saved with save_pretrained() (e.g. from
      scripts/train_classifier_guardrail.py). BERT and other HF models are saved
      in this format.
    - Hugging Face Hub model id: e.g. "bert-base-uncased", "roberta-base",
      "distilbert-base-uncased", or any model id that has a sequence
      classification head (num_labels=2 for binary safe/unsafe).
    """
    try:
        from transformers import pipeline
    except ImportError:
        raise ImportError(
            "transformers is required for the classifier guardrail. "
            "Install with: pip install transformers torch"
        )
    path = Path(model_path)
    load_from_local = path.exists() and path.is_dir()
    model_id = str(path) if load_from_local else model_path
    pipe = pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        device=0 if device == "cuda" else -1,  # -1 = CPU
        top_k=None,  # return all class scores
    )
    return pipe


def _load_config(model_path: str) -> Dict[str, Any]:
    """Load optional guardrail_config.json from model directory (local path only)."""
    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        return {}
    config_path = path / CONFIG_FILENAME
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _predict_risk(pipe: Any, text: str, id2label: Optional[Dict[int, str]] = None) -> float:
    """
    Return risk score in [0, 1] (higher = more harmful). Used for blocking: block when score >= threshold.
    Resolves P(unsafe) from the model's id2label (e.g. "unsafe" or LABEL_1).
    Training script uses 0=safe, 1=unsafe; we always return the score for the unsafe class.
    """
    if not text or not text.strip():
        return 0.0
    result = pipe(
        text.strip(),
        truncation=True,
        max_length=512,
        padding=True,
        return_all_scores=True,
    )
    if not result or not result[0]:
        return 0.0
    preds = result[0] if isinstance(result[0], list) else [result[0]]
    # Build set of names that mean "unsafe" (training uses 0=safe, 1=unsafe)
    unsafe_label_names = {"unsafe"}
    unsafe_index = 1  # default from training script
    if id2label:
        for idx, name in id2label.items():
            if name and str(name).lower() == "unsafe":
                unsafe_label_names.add(name)
                unsafe_index = int(idx)
                break
        if 1 in id2label and id2label[1]:
            unsafe_label_names.add(id2label[1])
    # Also match HF pipeline style "LABEL_0" / "LABEL_1"
    unsafe_label_names.add(f"LABEL_{unsafe_index}")
    # Find score for the unsafe class
    for p in preds:
        label = p.get("label", "")
        score = float(p.get("score", 0.0))
        if label in unsafe_label_names:
            return score
        if not id2label and ("1" in str(label) or str(label).lower() == "unsafe"):
            return score
    # Fallback: pipeline may return only winner or use different label names -> treat first as P(safe), risk = 1 - P(safe)
    return 1.0 - float(preds[0].get("score", 0.0))


class ClassifierGuardrail(BaseGuardrail):
    """
    Guardrail that uses a Hugging Face Transformers model to score content.

    Supports any HF model for text/sequence classification (e.g. BERT, RoBERTa,
    DistilBERT, ALBERT). Score convention matches LLM judge: higher score = more
    harmful/risky. FAIL when score >= config.threshold. Compatible with
    ChatPipeline and stacks.
    """

    def __init__(
        self,
        config: GuardrailConfig,
        pipeline: Any,
        device: str = "cpu",
        id2label: Optional[Dict[int, str]] = None,
    ):
        """
        Args:
            config: Guardrail configuration (threshold, fail_open, etc.).
            pipeline: Hugging Face text-classification pipeline (transformers).
            device: "cpu" or "cuda" for inference.
            id2label: Optional mapping from label index to name (for P(unsafe) lookup).
        """
        super().__init__(config)
        self.pipeline = pipeline
        self.device = device
        self.id2label = id2label or {}

    def evaluate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        evaluation_type: EvaluationType = EvaluationType.USER_INPUT,
    ) -> GuardrailResult:
        start_time = time.time()
        try:
            score = _predict_risk(self.pipeline, content, self.id2label)
            status = GuardrailStatus.FAIL if score >= self.config.threshold else GuardrailStatus.PASS
            latency_ms = (time.time() - start_time) * 1000
            return GuardrailResult(
                status=status,
                score=score,
                reasoning=None,
                metadata={
                    "evaluation_type": evaluation_type.value,
                    "model_type": "transformers",
                },
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            if self.config.fail_open:
                status = GuardrailStatus.PASS
            else:
                status = GuardrailStatus.ERROR
            return GuardrailResult(
                status=status,
                reasoning=f"Classifier error: {str(e)}",
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "evaluation_type": evaluation_type.value,
                },
                latency_ms=latency_ms,
            )


def load_classifier_guardrail(
    model_path: str,
    name: str,
    description: str,
    threshold: float = 0.5,
    device: str = "cpu",
    fail_open: bool = False,
) -> BaseGuardrail:
    """
    Load a finetunable classifier guardrail from a Hugging Face model.

    model_path can be:
    - A local directory: saved with the training script (save_pretrained).
      Example: "models/my_guardrail" (BERT or any HF model fine-tuned and saved).
    - A Hugging Face Hub model id: e.g. "bert-base-uncased", "roberta-base",
      "distilbert-base-uncased". The model must support sequence classification
      (binary: num_labels=2 for safe/unsafe). Fine-tune with the training
      script for best results.

    Compatible with ChatPipeline: use as input_guardrail or output_guardrail
    (single or in a stack with LLMJudgeGuardrail).

    Args:
        model_path: Local path to saved model dir, or HF model id (e.g. bert-base-uncased).
        name: Guardrail name (e.g. "input_guardrail").
        description: Short description for config.
        threshold: Score >= threshold -> FAIL (higher score = more harmful).
        device: "cpu" or "cuda" for inference.
        fail_open: If True, on error return PASS.

    Returns:
        ClassifierGuardrail instance usable in the pipeline and loader.
    """
    pipe = _load_transformers_pipeline(model_path, device)
    meta = _load_config(model_path)
    if meta:
        threshold = meta.get("threshold", threshold)
        fail_open = meta.get("fail_open", fail_open)
    config = GuardrailConfig(
        name=name,
        description=description,
        threshold=threshold,
        fail_open=fail_open,
    )
    id2label = getattr(pipe.model.config, "id2label", None)
    if id2label is not None and isinstance(id2label, dict):
        id2label = {int(k): v for k, v in id2label.items()}
    else:
        id2label = {}
    return ClassifierGuardrail(config=config, pipeline=pipe, device=device, id2label=id2label)
