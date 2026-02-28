# Mental Health Safety Sandbox Hackathon

Welcome to the Mental Health Safety Sandbox Hackathon. This repository contains the starter code, evaluation framework, and documentation you need to build and submit content-safety guardrails for a chat pipeline.

---

## Background

The goal of this hackathon is to **build strong guardrails** for a safety-critical chatbot.

We provide a **KHP (Kids Help Phone) chat virtual assistant** that **triages users based on the conversation**. The virtual assistant is a **navigation and triage tool** for a crisis support service for children and teenagers: it is **not** a counsellor or therapist. It is designed to:

- Understand the user's intent and assess urgency/safety
- Route users to appropriate human or external support
- Explain available services and next steps
- Avoid providing emotional validation, therapy, or crisis intervention itself—and instead redirect to human care

The virtual assistant handles different situations (crisis/high-risk, emotional support requests, navigation/information, unclear messages) and must stay within strict boundaries (calm, clear, non-therapeutic, transparent about limitations).

**Participants can access this pipeline and do adversarial testing** to find failure modes. Use the **provided notebook** to run the chatbot and try prompts that might expose unsafe or off-role behavior:

- **[project/notebooks/chat_pipeline_demo.ipynb](project/notebooks/chat_pipeline_demo.ipynb)**. Use it to explore triage behavior and run adversarial tests.

**How to access the chat pipeline demo**

1. Install dependencies and set up the kernel (see [Installation](#installation) below).
2. Open `project/notebooks/chat_pipeline_demo.ipynb` and select the **"Python (aiss)"** kernel.
3. Run the setup cells, then run the pipeline with your own prompts.

---

## Hackathon task

The goal for participants is to **enhance the safety of the chatbot by building a guardrail system**. We are specifically interested in improving the **input guardrail**: it sees the **user query with context history** and **classifies** it (e.g. safe vs unsafe). Participants should build an input guardrail that **improves the failure modes** of the KHP chat pipeline—e.g. blocking or redirecting harmful, off-role, or policy-violating inputs before they reach the KHP virtual assistant.

**Test your guardrails** using the pipeline test notebook:

- **[project/notebooks/input_guardrail_test.ipynb](project/notebooks/input_guardrail_test.ipynb)** — Full pipeline with input guardrails. Use it to plug in your guardrail via `get_guardrails()` and verify that it blocks or allows the right prompts.

**How to use the chat pipeline test notebook to test guardrails**

1. Implement your guardrail and expose it via `get_guardrails()` in `project/src/submission/submission.py`. Use the example modules as references. See [Submission contract](#submission-contract) and [project/src/guardrails/README.md](project/src/guardrails/README.md) for the guardrail API.
2. Open `project/notebooks/input_guardrail_test.ipynb` and select the **"Python (aiss)"** kernel, with working directory `project/`.
3. In the notebook, set `USE_GUARDRAILS = True` and point the import to `submission` (or your test module). Run the cell that calls `get_guardrails()`.
4. Build the pipeline with the returned input guardrail (output is None) and run test prompts (including adversarial ones). Check `result.status`, `result.blocked_at`, and `result.response` to confirm your guardrail behaves as intended.

---

## Submission contract

Your submission is a **single Python module** in `project/src/submission/` that defines:

```python
def get_guardrails() -> tuple[input_guardrail, output_guardrail]:
    """Return (input_guardrail, output_guardrail). Either may be None."""
    ...
```

- **No arguments** are passed to `get_guardrails()`.
- The evaluator uses a fixed main LLM and system prompt; you only return guardrails.
- For this hackathon, return your **input** guardrail and **`None`** for output.
- Each return value may be `None`, a **single** guardrail, or a **list/tuple** (stack). Stacks run in order and short-circuit on first failure.
- You may use your own LLM (env vars), classifier/BERT (model path), or any guardrail that implements the guardrail protocol (see [project/src/guardrails/README.md](project/src/guardrails/README.md)).

---

## Submission folder

| File | Purpose |
|------|---------|
| **`example_submission.py`** | Generic template; defines `get_guardrails()` and helpers. |
| **`example_submission_llm_judge.py`** | Example using an LLM judge guardrail. |
| **`example_submission_ditillbert_guardrail.py`** | Example using a DistilBERT-style classifier guardrail. |
| **`hackathon_runner.py`** | Evaluator: loads your module, runs benchmark and metrics. |

Use `submission.py` as the canonical module loaded by shared scripts, and copy logic from examples into it.

---

## Using your own models

- **LLM for guardrails:** In the examples, environment variables are used only for API keys and provider base URLs (e.g. `OPENAI_API_KEY`, `COHERE_API_KEY`, `COHERE_BASE_URL`). Replace or extend with your API, model, or custom provider.
- **Classifier / BERT:** Use a path to your trained model. Place model files in a folder (e.g. `project/models/`) so the path works when the runner loads your module.
- **Custom guardrail:** Implement `evaluate(content, context=None, evaluation_type=...) -> GuardrailResult` and return it from `get_guardrails()`. See [project/src/guardrails/README.md](project/src/guardrails/README.md) for the protocol.

---

## Run locally / evaluation

**Test your submission locally** (from the `project/` directory):

```bash
cd project
PYTHONPATH=. python src/submission/submission.py
```

**Run evaluation** (organizers or self-serve):

```bash
cd project
PYTHONPATH=. python -m src.submission.hackathon_runner \
  --submission src/submission/submission.py \
  --benchmark-csv path/to/benchmark.csv \
  --config path/to/config.yaml \
  --output-dir results/hackathon
```

Use `src/submission/submission.py` as your canonical submission module path. The runner adds the submission's directory to `sys.path`, so models can live next to the submission and be loaded by path.

**Run the submission scripts (configure, predict, eval_metrics)**

Use these scripts to set up your environment, generate predictions with your guardrail, and compute evaluation metrics. Run them in order: **configure → predict → eval_metrics**. All commands below are from the **repository root**.

1. **configure** — Install dependencies and materialize any models or assets your guardrail needs.

   ```bash
   ./project/scripts/configure.sh
   ```

   This creates/uses a virtualenv, installs from `pyproject.toml` (or `requirements.txt`), and ensures `ipykernel` is available.

2. **predict** — Run your submission’s input guardrail on a labeled CSV and write a predictions CSV.

   ```bash
   ./project/scripts/predict.sh <path_to_input_file.csv> <path_to_predictions_output_file.csv>
   ```

   Example (using the sample guardrail data):

   ```bash
   ./project/scripts/predict.sh datasets/sample_guardrail_data.csv results/predictions.csv
   ```

3. **eval_metrics** — Compute precision, recall, F1, and latency from the predictions CSV and write metrics.

   ```bash
   ./project/scripts/eval.sh <path_to_predictions.csv> <path_to_eval_metrics.csv>
   ```

   Example:

   ```bash
   ./project/scripts/eval.sh results/predictions.csv results/eval_metrics.csv
   ```

   This produces `eval_metrics.csv` and `eval_metrics.json` in the output directory.

**Full workflow example** (from repo root, with venv activated):

```bash
source .venv/bin/activate
./project/scripts/configure.sh
./project/scripts/predict.sh datasets/sample_guardrail_data.csv results/predictions.csv
./project/scripts/eval.sh results/predictions.csv results/eval_metrics.csv
```

---

## Quick start

1. **Install dependencies** (see [Installation](#installation) below).
2. **Run the demo pipeline** from the `project/` directory to verify your setup.
3. **Build your guardrails** using the [Guardrails framework](project/src/guardrails/README.md) (LLM judge, classifier, or custom).
4. **Submit** by implementing `get_guardrails()` as described in [Submission contract](#submission-contract) below.

---

## Repository structure

| Path | Description |
|------|-------------|
| **`project/`** | Main code: chat pipeline, guardrails, providers, notebooks. **Do your development here.** |
| **`project/README.md`** | Setup and run instructions for the project. |
| **`project/src/guardrails/`** | Guardrail framework and built-in implementations (LLM judge, classifier). |
| **`project/src/submission/`** | Submission contract, example submissions, and evaluation runner. |
| **`project/notebooks/`** | **chat_pipeline_demo.ipynb** — main LLM only (adversarial testing). **input_guardrail_test.ipynb** — pipeline with input guardrail. |
| **`project/providers/`** | LLM providers (OpenAI, Cohere, demo). |
| **`datasets/`** | Datasets and data sources (if provided). |
| **`docs/`** | Additional docs (presentations, guides). |

---

## Installation

**Requirements:** Python 3.12 or later. We recommend [uv](https://docs.astral.sh/uv/) for fast, reproducible installs.


### 1. Clone the repository

```bash
git clone <repository-url>
cd Mental-Health-Safety-Sandbox-Hackathon-
```

### 2. Create environment and install

From the **repository root** (where `pyproject.toml` is):

```bash
uv venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
uv pip install -e .
```

Alternatively with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 4. Verify: run the chat pipeline

From the **project** directory (so `src` and `providers` are on the path):

```bash
cd project
PYTHONPATH=. python -c "
from src.end_to_end.chat_pipeline import ChatPipeline
from providers.demo_provider import DemoProvider
p = ChatPipeline(main_llm_provider=DemoProvider(model='demo'))
print(p.process('Hello').response)
"
```

You can also open `project/notebooks/input_guardrail_test.ipynb`, set the kernel working directory to `project/`, and run the cells.

### 5. Environment variables (optional)

- **OpenAI:** set `OPENAI_API_KEY`.
- **Cohere:** set `COHERE_BASE_URL` and `COHERE_API_KEY`.

Use a `.env` file in the repository root if you like; `python-dotenv` is already a dependency.

---

## Hackathon submission checklist

Use this checklist when preparing your submission. You are responsible for ensuring your setup works before submitting.

### 1. One-pager

Submit a **one-pager** that describes:

- How you tackle the problem
- How your solution works and how it improves the system
- **Results on the validation set provided** (report these in the one-pager)

### 2. Guardrail implementation: `get_guardrails()`

Implement **`get_guardrails()`** in **`project/src/submission/submission.py`**. Return your input guardrail and `None` for output (see [Submission contract](#submission-contract)).

- Use the example submissions as references and copy the logic you need into `submission.py`: [example_submission_llm_judge.py](project/src/submission/example_submission_llm_judge.py) (LLM judge), [example_submission_ditillbert_guardrail.py](project/src/submission/example_submission_ditillbert_guardrail.py) (finetuned model).
- Base classes are provided; see [project/src/guardrails/README.md](project/src/guardrails/README.md).

### 3. Scripts (configure, predict, eval)

Do not modify **`project/scripts/configure.sh`**, **`project/scripts/predict.sh`**, or **`project/scripts/eval.sh`**.
These scripts are shared runner entry points for local testing and evaluator execution.

- **configure.sh** — Installs dependencies, reads **`hackathon.json`**, checks GPU policy, and fetches any declared artifacts.
- **predict.sh** — Reads an input CSV and writes predictions using your guardrail (via `get_guardrails()`) from `src/submission/submission.py`.
- **eval.sh** — Evaluates your predictions CSV and writes metrics (e.g. precision, recall, F1).

Configure your runtime in **`hackathon.json`**:
- `needs_gpu` (**required**, boolean): set `true` only if your runtime requires CUDA/GPU.
- `artifacts` (**required**, list): external files to fetch during `configure.sh`.

Each item in `artifacts` supports:
- `uri` (**required**, string): `s3://...` or `https://...`
- `destination` (**required**, string): path under this repo where the file/extracted content should be written
- `sha256` (optional, string): checksum of the downloaded artifact; if provided, it is verified
- `required` (optional, boolean, default `true`): if `false`, configure continues when download/check fails

Example:

```json
{
  "needs_gpu": false,
  "artifacts": [
    {
      "uri": "s3://khp-tests/hackathon/tests/hackathon_s3_test.txt",
      "destination": "project/models/_s3_test/hackathon_s3_test.txt",
      "sha256": "cb2c64dec5e85a69592ee229c8a8ad10294ba1c4493457bd53e2486c18090564",
      "required": false
    }
  ]
}
```

`hackathon.json` is required for runtime execution and is validated strictly.
- Missing file or invalid JSON/field types causes `configure.sh` to fail.
- Classifier submission templates also fail fast if `hackathon.json` is missing/invalid.

Optional helper: **`project/scripts/publish_artifact.sh`** packages/uploads local artifacts to S3 and prints a ready-to-paste JSON snippet.

Run configure -> predict -> eval locally before submitting.

### 4. Problem statement

A short description of the mental-health or safety issue your solution addresses.

### 5. Objective

The challenge or need your solution contributes to solving.

### 6. Solution / data use case

A clear description of your data-based solution and how you use data (and guardrails) in the pipeline.

### 7. Pitch

A pitch of **maximum 5 minutes**.  
Recommended: add a link here (e.g. YouTube recording).

*Example: [Link to video]*

**GitHub limits:** 100MB per file, 10GB per repository.

### 8. Demo

A functional or conceptual demo showing how your solution works, e.g.:

- Short screen recording of the prototype or dashboard  
- Walkthrough of the pipeline or model results  
- Mock user journey (for early-stage concepts)

### 9. Datasets

**Location:** `/datasets`

Include:

- API access or download links for required data  
- Notes on data quality and any transformations  
- Any new data you generated (e.g. by merging datasets)

### 10. Project code

**Location:** `/project`

Your code should cover:

- Data transformations, merging, and quality checks  
- Model-related code (e.g. classifiers, guardrails)  
- Any user interface or demo scripts

### 11. Additional docs (optional)

**Location:** `/docs`

- Slides, flyers, extra videos, protocols, or guides

### 12. Terms and conditions

By submitting to the Mental Health Safety Sandbox Hackathon, you acknowledge and agree to the event's [Terms and Conditions](link to the T&C).

---

## Next steps

- **Set up and run:** [project/README.md](project/README.md)  
- **Implement guardrails:** [project/src/guardrails/README.md](project/src/guardrails/README.md)  
- **Guardrail API (base classes, LLM judge, classifier):** [project/src/guardrails/README.md](project/src/guardrails/README.md)
