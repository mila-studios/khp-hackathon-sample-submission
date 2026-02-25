# Hackathon submission (benchmarking)

This folder contains the **submission contract**, **example submissions**, and the **evaluation runner**. To submit, copy an example (e.g. `example_submission_llm_judge.py` or `example_submission_ditillbert_guardrail.py`), adapt it, and implement `get_guardrails()`.

---

## Quick start for participants

1. Copy one of the example submission files and rename it (e.g. `my_submission.py`).
2. Implement **`get_guardrails()`** so it returns your input and output guardrails (see [Contract](#contract) below).
3. Test locally (see [Run the example locally](#run-the-example-locally)).
4. Point organizers at your module for evaluation (see [Run evaluation](#run-evaluation-organizers)).

---

## Contract

Your submission must be a **single Python module** that defines:

```python
def get_guardrails() -> tuple[input_guardrail, output_guardrail]:
    """Return (input_guardrail, output_guardrail). Either may be None."""
    ...
```

Rules:

- **No arguments** are passed to `get_guardrails()`.
- The **main LLM and system prompt** are fixed by the evaluator; you only return guardrails.
- Each return value may be `None`, a **single** guardrail, or a **list/tuple** (stack). Stacks run in order and short-circuit on first failure.
- You may use your own LLM (env vars, config), your own classifier/BERT (model path), or any guardrail that implements the [guardrail protocol](../guardrails/README.md#contract-guardrail-signature).
- For stacked guardrails, return a list or tuple, e.g. `output_guardrail = [gr1, gr2]`.

---

## Files in this folder

| File | Purpose |
|------|---------|
| **`example_submission.py`** | Generic template; defines `get_guardrails()` and helpers. |
| **`example_submission_llm_judge.py`** | Example using an LLM judge guardrail. |
| **`example_submission_ditillbert_guardrail.py`** | Example using a DistilBERT-style classifier guardrail. |
| **`hackathon_runner.py`** | Evaluator: loads your module, builds the pipeline, runs benchmark and metrics. |

---

## Using your own models

- **LLM for guardrails:** In the examples, the guardrail LLM is configured via env (e.g. `OPENAI_API_KEY`, `OPENAI_MODEL`). Replace or extend this with your API, model name, or custom provider.
- **Classifier / BERT:** Use a path to your trained model (e.g. via `_load_own_bert_guardrail(role, model_path, device)`). Place model files in a folder (e.g. `models/`) next to your submission so the path works when the runner loads your module.
- **Custom guardrail:** Implement `evaluate(content, context=None, evaluation_type=...) -> GuardrailResult` and return it from `get_guardrails()`. See [Guardrails README](../guardrails/README.md).

---

## Run the example locally

From the **`project`** directory:

```bash
cd project
PYTHONPATH=. python src/benchmark/example_submission_llm_judge.py
```

Or with your own module:

```bash
PYTHONPATH=. python src/benchmark/my_submission.py
```

---

## Run evaluation (organizers)

```bash
cd project
PYTHONPATH=. python -m src.benchmark.hackathon_runner \
  --submission src/benchmark/example_submission.py \
  --benchmark-csv path/to/benchmark.csv \
  --config path/to/config.yaml \
  --output-dir results/hackathon
```

Participants can set `--submission` to their module (e.g. `submissions/team_42/submission.py`). The runner adds the submission’s directory to `sys.path`, so models can live next to the submission and be loaded by path.
