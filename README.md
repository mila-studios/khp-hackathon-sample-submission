# Mental Health Safety Sandbox Hackathon

Welcome to the Mental Health Safety Sandbox Hackathon. This repository contains the starter code, evaluation framework, and documentation you need to build and submit content-safety guardrails for a chat pipeline.

---

## Quick start

1. **Install dependencies** (see [Installation](#installation) below).
2. **Run the demo pipeline** from the `project/` directory to verify your setup.
3. **Build your guardrails** using the [Guardrails framework](project/src/guardrails/README.md) (LLM judge, classifier, or custom).
4. **Submit** by implementing `get_guardrails()` as described in [Benchmark & submission](project/src/benchmark/PARTICIPANT_README.md).

---

## Repository structure

| Path | Description |
|------|-------------|
| **`project/`** | Main code: chat pipeline, guardrails, providers, notebooks. **Do your development here.** |
| **`project/README.md`** | Setup and run instructions for the project. |
| **`project/src/guardrails/`** | Guardrail framework and built-in implementations (LLM judge, classifier). |
| **`project/src/benchmark/`** | Submission contract, example submissions, and evaluation runner. |
| **`project/notebooks/`** | Example notebook for testing the chat pipeline. |
| **`project/providers/`** | LLM providers (OpenAI, Cohere, demo). |
| **`datasets/`** | Datasets and data sources (if provided). |
| **`docs/`** | Additional docs (presentations, guides). |

---

## Installation

**Requirements:** Python 3.12 or later. We recommend [uv](https://docs.astral.sh/uv/) for fast, reproducible installs.

### 1. Install uv (optional but recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with pip: `pip install uv`.

### 2. Clone the repository

```bash
git clone <repository-url>
cd Mental-Health-Safety-Sandbox-Hackathon-
```

### 3. Create environment and install

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

You can also open `project/notebooks/chat_pipeline_test.ipynb`, set the kernel working directory to `project/`, and run the cells.

### 5. Environment variables (optional)

- **OpenAI:** set `OPENAI_API_KEY`.
- **Cohere:** set `COHERE_BASE_URL` (and optionally `COHERE_SYSTEM_PROMPT`).

Use a `.env` file in the repository root if you like; `python-dotenv` is already a dependency.

---

## Hackathon submission checklist

Use this structure when preparing your submission and pitch.

### 1. Problem statement

A short description of the mental-health or safety issue your solution addresses.

### 2. Objective

The challenge or need your solution contributes to solving.

### 3. Solution / data use case

A clear description of your data-based solution and how you use data (and guardrails) in the pipeline.

### 4. Pitch

A pitch of **maximum 5 minutes**.  
Recommended: add a link here (e.g. YouTube recording).

*Example: [Link to video]*

**GitHub limits:** 100MB per file, 10GB per repository.

### 5. Demo

A functional or conceptual demo showing how your solution works, e.g.:

- Short screen recording of the prototype or dashboard  
- Walkthrough of the pipeline or model results  
- Mock user journey (for early-stage concepts)

### 6. Datasets

**Location:** `/datasets`

Include:

- API access or download links for required data  
- Notes on data quality and any transformations  
- Any new data you generated (e.g. by merging datasets)

### 7. Project code

**Location:** `/project`

Your code should cover:

- Data transformations, merging, and quality checks  
- Model-related code (e.g. classifiers, guardrails)  
- Any user interface or demo scripts

### 8. Additional docs (optional)

**Location:** `/docs`

- Slides, flyers, extra videos, protocols, or guides

### 9. Terms and conditions

By submitting to the Mental Health Safety Sandbox Hackathon, you acknowledge and agree to the event’s [Terms and Conditions](link to the T&C).

---

## Next steps

- **Set up and run:** [project/README.md](project/README.md)  
- **Implement guardrails:** [project/src/guardrails/README.md](project/src/guardrails/README.md)  
- **Benchmark and submit:** [project/src/benchmark/PARTICIPANT_README.md](project/src/benchmark/PARTICIPANT_README.md)
