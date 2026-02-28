# Project code

This directory contains the **main hackathon code**: chat pipeline, guardrails, LLM providers, and notebooks. New participants should set up and run from here.

For **background** (KHP chat pipeline, triage, adversarial testing), the **hackathon task**, and the **submission checklist** (one-pager, scripts, `get_guardrails()`), see the [repository root README](../README.md).

---

## What's in this directory

| Path | Description |
|------|-------------|
| **`src/`** | Core Python packages. |
| **`src/end_to_end/`** | Chat pipeline: user input → input guardrails → LLM → output guardrails → response. |
| **`src/guardrails/`** | Guardrail framework and implementations (LLM judge, classifier). See [src/guardrails/README.md](src/guardrails/README.md). |
| **`src/submission/`** | Submission module and examples; implement `get_guardrails()`. See [repository root README](../README.md#submission-contract). |
| **`src/prompt_templates/`** | System prompts and guardrail prompts. |
| **`providers/`** | LLM providers (OpenAI, Cohere, demo). |
| **`notebooks/`** | **chat_pipeline_demo.ipynb** — main LLM only (adversarial testing). **input_guardrail_test.ipynb** — full pipeline with guardrails. |
| **`scripts/`** | Training and utility scripts (e.g. classifier guardrail training). |

---

## Installation and run

Install from the **repository root** (where `pyproject.toml` lives), then run code from this **`project/`** directory so that `src` and `providers` are importable.

### Option A: Using uv (recommended)

From the **repository root**:

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e .
```

### Option B: Using pip

From the **repository root**:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### Run the pipeline from the command line

From this **`project/`** directory:

```bash
cd project
PYTHONPATH=. python -c "
from src.end_to_end.chat_pipeline import ChatPipeline
from providers.demo_provider import DemoProvider
p = ChatPipeline(main_llm_provider=DemoProvider(model='demo'))
result = p.process('Hello')
print(result.response)
"
```


### Run the notebooks

1. **Register the virtualenv as a Jupyter kernel** (from repo root with the venv activated):

   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name=aiss --display-name="Python (aiss)"
   ```

2. Open **`notebooks/chat_pipeline_demo.ipynb`** (main LLM only) or **`notebooks/input_guardrail_test.ipynb`** (with guardrails) and select the **"Python (aiss)"** kernel.

3. **Set the kernel's working directory to `project/`** (in VS Code/Cursor: use the kernel picker or "Select Another Kernel" and ensure cwd is `project/`).  
   This ensures `from src...` and `from providers...` work.

   Alternatively, at the top of the notebook you can run:

   ```python
   import sys
   sys.path.insert(0, '/path/to/repo/project')   # replace with your actual path
   ```

---

## Environment variables

- **OpenAI:** `OPENAI_API_KEY`
- **Cohere:** `COHERE_BASE_URL`, `COHERE_API_KEY`

You can put these in a `.env` file in the repository root.

---

## Where to go next

- **Implement or customize guardrails:** [src/guardrails/README.md](src/guardrails/README.md)  
- **Submission contract and examples:** [repository root README](../README.md#submission-contract)
