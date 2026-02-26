# Guardrails

This module provides a **flexible guardrail framework** for content safety in the chat pipeline. Any implementation that matches the guardrail **signature** can be used. You can attach a **single** guardrail or **stack** multiple guardrails (they run in order and short-circuit on first failure).

---

## For hackathon participants

- Use **LLM judge** and/or **classifier** guardrails, or implement your **own** guardrail that follows the contract below.
- Plug guardrails into the pipeline via `input_guardrail` and `output_guardrail` (see [Using guardrails in the pipeline](#using-guardrails-in-the-pipeline)).
- For submission, you return your guardrails from `get_guardrails()` in a module in `src/submission/`; see the [repository root README](../../../README.md#submission-contract) for the contract and examples.

---

## Contract: guardrail signature

A guardrail is any object that has:

- **`config`** — a `GuardrailConfig` (name, description, threshold, etc.)
- **`evaluate(content, context=None, evaluation_type=...)`** — returns a `GuardrailResult` (sync)

No specific base class is required. For type hints you can use **`GuardrailProtocol`**; for shared helpers (e.g. `_create_result`) you can subclass **`BaseGuardrail`**.

### Main types

| Type | Purpose |
|------|--------|
| `GuardrailProtocol` | Protocol: any object with `config` and `evaluate` works in the pipeline. |
| `BaseGuardrail` | Abstract base with `config` and `_create_result()`; you implement `evaluate`. |
| `GuardrailResult` | `status` (PASS/FAIL/ERROR), optional `score`, `reasoning`, `metadata`; `.is_harmful` for pass/fail. |
| `GuardrailConfig` | `name`, `description`, `threshold`, `enabled`, `fail_open`, etc. |
| `EvaluationType` | `USER_INPUT` or `LLM_OUTPUT` for context. |

---

## Using guardrails in the pipeline

The chat pipeline accepts:

- **`input_guardrail`** / **`output_guardrail`**: `None`, a **single** guardrail, or a **sequence** (list/tuple) of guardrails.
- A single guardrail is treated as a one-element stack; a sequence runs **in order** and **short-circuits on first failure** when blocking is enabled.

Results are always list-based:

- **`PipelineResult.input_guardrail_results`** — one dict per input guardrail.
- **`PipelineResult.output_guardrail_results`** — one dict per output guardrail.
- **`PipelineResult.blocked_at_guardrail_index`** — when blocked, the 0-based index of the guardrail that failed.

Accessors on the pipeline:

- **`pipeline.input_guardrails`** / **`pipeline.output_guardrails`** — read-only lists of guardrail instances.

### Example: single guardrail

```python
from src.end_to_end.chat_pipeline import ChatPipeline
from src.guardrails import LLMJudgeGuardrail, GuardrailConfig

config = GuardrailConfig(name="safety", description="Safety check", threshold=0.6)
guardrail = LLMJudgeGuardrail(config=config, llm_provider=judge_llm)

pipeline = ChatPipeline(
    main_llm_provider=main_llm,
    input_guardrail=guardrail,
    output_guardrail=guardrail,
)
result = pipeline.process("Hello")
# result.input_guardrail_results is a list of one result dict
```

### Example: stacked guardrails

```python
pipeline = ChatPipeline(
    main_llm_provider=main_llm,
    input_guardrail=[classifier_guardrail, llm_judge_guardrail],
    output_guardrail=[classifier_guardrail, llm_judge_guardrail],
)
# Runs in order; first failure blocks. result.input_guardrail_results has two entries.
```

### Example: custom guardrail (signature only)

```python
from src.guardrails.base import GuardrailConfig, GuardrailResult, GuardrailStatus, EvaluationType

class MyGuardrail:
    def __init__(self):
        self.config = GuardrailConfig(name="my", description="Custom check")

    def evaluate(self, content: str, context=None, evaluation_type=EvaluationType.USER_INPUT):
        return GuardrailResult(status=GuardrailStatus.PASS, score=1.0)

pipeline = ChatPipeline(main_llm_provider=main_llm, input_guardrail=MyGuardrail())
```

---

## Built-in implementations

| Implementation | Description |
|----------------|-------------|
| **LLMJudgeGuardrail** | Uses an LLM as a judge for content evaluation. |
| **ClassifierGuardrail** | Hugging Face Transformers model (e.g. BERT). Load via **`load_classifier_guardrail()`** with a local path or Hub model id. Train with **`python -m src.guardrails.train_classifier_guardrail`** (default: BERT; override with `--base_model`). In YAML use `type: "finetunable"` (or `"bert"`) and `model_path`. |

Both implement `BaseGuardrail` and the protocol; they can be mixed in the same stack.

---

## Module layout

| File | Contents |
|------|----------|
| **`base.py`** | `GuardrailProtocol`, `BaseGuardrail`, `GuardrailResult`, `GuardrailConfig`, `GuardrailStatus`, `EvaluationType`. |
| **`llm_judge.py`** | `LLMJudgeGuardrail`. |
| **`classifier.py`** | `ClassifierGuardrail`, `load_classifier_guardrail`. |
| **`metrics.py`** | `get_predictions()`, `compute_metrics_from_predictions()`, `GuardrailMetricsResult` — run guardrails for per-sample predictions; compute precision, recall, F1 and latency from predictions. |
| **`get_predictions.py`** | CLI: `python -m src.guardrails.get_predictions` — run guardrails on a labeled CSV and write prediction CSV. |
| **`get_guardrail_metrics.py`** | CLI: `python -m src.guardrails.get_guardrail_metrics` — compute precision, recall, F1 and latency from a prediction CSV. |
| **`train_classifier_guardrail.py`** | CLI: `python -m src.guardrails.train_classifier_guardrail` — train a finetunable classifier guardrail (e.g. BERT, DistilBERT). |
| **`__init__.py`** | Re-exports for `from src.guardrails import ...`. |

---

## Guardrail metrics (precision, recall, F1, latency)

Use **`get_predictions()`** to run a guardrail (or stack) on labeled data and get per-sample predictions. For stacked guardrails, the combined prediction is “harmful” if *any* guardrail in the stack returns a harmful result. Then use **`compute_metrics_from_predictions()`** to get precision, recall, F1 and latency from that list. Works for both LLM and classifier guardrails.

```python
from src.guardrails import get_predictions, compute_metrics_from_predictions
from src.guardrails.base import EvaluationType

# evaluation_data: list of dicts with "content" and "label" (1/True = harmful)
data = [{"content": "How to hurt someone?", "label": 1}, {"content": "Tips for sleep.", "label": 0}]
predictions = get_predictions(
    input_guardrail, data,
    evaluation_type=EvaluationType.USER_INPUT,
    include_latency=True,
)
result = compute_metrics_from_predictions(predictions)
# result.precision, result.recall, result.f1, result.latency_ms_mean, result.latency_ms_total
```

From the command line, use a two-step workflow:

1. **`python -m src.guardrails.get_predictions`** — loads guardrails via `get_guardrails()` from a submission module, runs the input guardrail on a labeled CSV, and writes predictions CSV.
2. **`python -m src.guardrails.get_guardrail_metrics`** — reads that prediction CSV and computes precision, recall, F1 and latency, writing metrics JSON and `metrics.csv`.

You can also call **`compute_metrics_from_predictions()`** on an existing list of prediction dicts (e.g. loaded from a CSV from `get_predictions`); see `metrics.py` for the signature.
