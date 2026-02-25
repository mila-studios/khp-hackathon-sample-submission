"""Hackathon evaluation runner: run a participant's guardrails on the benchmark and compute refusal metrics.

The main LLM and system prompt are fixed by the evaluator (from config). Participants
return only input and output guardrails; the runner builds the ChatPipeline. Contract:

    def get_guardrails() -> (input_guardrail, output_guardrail):
        ...  # each may be None, a single guardrail, or a list/tuple (stack)

The runner creates the pipeline, loads the benchmark CSV, runs each prompt through it,
writes a responses CSV compatible with the existing judge + metrics, then runs judge
and metrics to produce RRR/FRR and summary.

Usage:
    cd project && PYTHONPATH=. python -m src.benchmark.hackathon_runner \
        --submission src/benchmark/example_submission.py \
        --benchmark-csv path/to/benchmark.csv \
        --config path/to/config.yaml \
        --output-dir results/hackathon

Example submission lives in this folder (src/benchmark/); participants copy and adapt it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path so src.benchmark and src.end_to_end are importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2]  # benchmark -> src -> project
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.benchmark.config import load_config, resolve_paths
from src.benchmark.io_utils import load_benchmark_cases, write_results_atomic
from src.benchmark.metrics import summary_output_name
from src.benchmark.providers import create_llm_provider
from src.benchmark.stages import BASE_RESPONSES_TARGET, _build_fieldnames
from src.end_to_end.chat_pipeline import ChatPipeline, PipelineStatus, PipelineStage
from src.guardrails.submission_loader import load_guardrails_from_module


def _build_pipeline(
    main_llm_provider: Any,
    system_prompt: Optional[str],
    input_guardrail: Any,
    output_guardrail: Any,
) -> ChatPipeline:
    """Build the ChatPipeline with fixed main LLM, system prompt, and participant guardrails.

    input_guardrail and output_guardrail may be None, a single guardrail, or a sequence
    (stack) of guardrails; ChatPipeline accepts both (runs stacks in order, short-circuit on failure).
    """
    return ChatPipeline(
        main_llm_provider=main_llm_provider,
        input_guardrail=input_guardrail,
        output_guardrail=output_guardrail,
        system_prompt=system_prompt,
    )


def _is_harmful_to_data_type(is_harmful: str) -> str:
    """Map benchmark CSV is_harmful (Yes/No) to data_type for metrics."""
    v = (is_harmful or "").strip().lower()
    if v in ("yes", "true", "1"):
        return "harmful_query"
    if v in ("no", "false", "0", ""):
        return "safe_query"
    return ""


def run_pipeline_on_benchmark(
    pipeline: ChatPipeline,
    cases: List[Dict[str, Any]],
    run_dir: Path,
    *,
    max_rows: Optional[int] = None,
) -> Path:
    """Run the pipeline on each benchmark case and write responses CSV.

    Writes run_dir / responses_none_none.csv in the format expected by the judge stage.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    if max_rows:
        cases = cases[:max_rows]
    input_columns = list(cases[0].keys()) if cases else []
    # Ensure we have data_type for metrics (derive from is_harmful if missing)
    if "data_type" not in input_columns and "is_harmful" in input_columns:
        pass  # we'll add data_type per row

    responses_columns = [
        "guardrail_combo",
        "prompt_hash",
        "main_model",
        "main_provider",
        "pipeline_status",
        "blocked_at",
        "main_response_text",
        "main_response_usage",
        "main_response_finish_reason",
        "main_response_latency_ms",
        "response_status",
        "error_flag",
        "error_stage",
        "error_message",
        "final_response_text",
        "final_response_source",
    ]
    fieldnames = _build_fieldnames(
        input_columns, [], stage="responses"
    )
    if "data_type" not in fieldnames:
        # Insert after is_harmful or at end of input columns
        idx = len(input_columns)
        for i, c in enumerate(input_columns):
            if c == "is_harmful":
                idx = i + 1
                break
        fieldnames = fieldnames[:idx] + ["data_type"] + [c for c in fieldnames[idx:] if c != "data_type"]

    rows: List[Dict[str, str]] = []
    for idx, case in enumerate(cases, start=1):
        prompt = (case.get("prompt") or "").strip()
        result = pipeline.process(user_input=prompt)
        is_harmful = case.get("is_harmful", "")
        data_type = case.get("data_type") or _is_harmful_to_data_type(is_harmful)

        if result.status == PipelineStatus.ERROR:
            error_flag = "true"
            error_stage = "llm_generation"
            error_message = result.error or "Pipeline error"
            final_response = ""
            pipeline_status = "error"
            blocked_at = ""
        else:
            error_flag = "false"
            error_stage = ""
            error_message = ""
            final_response = result.response or ""
            pipeline_status = "success" if result.status == PipelineStatus.SUCCESS else "blocked"
            blocked_at = ""
            if result.blocked_at:
                blocked_at = result.blocked_at.value

        llm_response = result.llm_response or {}
        content = (result.response or llm_response.get("content") or "").strip()
        if not content and final_response:
            content = final_response
        main_response_text = content if result.status == PipelineStatus.SUCCESS else (result.response or "")

        row = {k: str(case.get(k, "")) for k in input_columns}
        row["data_type"] = data_type
        row["guardrail_combo"] = "none"
        row["prompt_hash"] = str(hash(prompt) % (2**32))
        row["main_model"] = getattr(pipeline.main_llm_provider, "model", "unknown")
        row["main_provider"] = type(pipeline.main_llm_provider).__name__
        row["pipeline_status"] = pipeline_status
        row["blocked_at"] = blocked_at
        row["main_response_text"] = main_response_text
        row["main_response_usage"] = json.dumps(llm_response.get("usage") or {})
        row["main_response_finish_reason"] = str(llm_response.get("finish_reason", ""))
        row["main_response_latency_ms"] = f"{result.total_latency_ms or 0:.4f}"
        row["response_status"] = "completed"
        row["error_flag"] = error_flag
        row["error_stage"] = error_stage
        row["error_message"] = error_message
        row["final_response_text"] = final_response
        row["final_response_source"] = "main_llm" if result.stage == PipelineStage.COMPLETED else "blocked"
        row["actual_behaviour"] = ""
        row["refusal_reason"] = ""
        row["is_correct"] = ""
        row["caught_stage"] = ""
        rows.append(row)

    out_path = run_dir / f"{BASE_RESPONSES_TARGET}.csv"
    write_results_atomic(out_path, rows, fieldnames)
    return out_path


def run_judge_and_metrics(
    config: Dict[str, Any],
    config_path: str,
    run_dir: Path,
    responses_path: Path,
) -> Dict[str, Any]:
    """Run judge stage then metrics on the responses CSV, using existing config for judge model."""
    from src.benchmark.stages import run_judge_stage, run_metrics_stage

    # Point config at this run's responses
    config = dict(config)
    config.setdefault("run", {})["responses_path"] = str(responses_path)
    config.setdefault("run", {})["use_responses_dir_as_run_dir"] = True
    config.setdefault("judge", {})["target"] = BASE_RESPONSES_TARGET

    run_judge_stage(config, config_path)
    run_metrics_stage(config, config_path)

    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _resolve_benchmark_csv_path(benchmark_csv: str, config_file: Path) -> Path:
    """Resolve benchmark CSV path; if relative, resolve relative to config file's directory."""
    p = Path(benchmark_csv).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (config_file.parent / p).resolve()


def main(
    submission_path: str,
    benchmark_csv: str,
    config_path: str,
    output_dir: str,
    max_rows: Optional[int] = None,
    skip_judge: bool = False,
) -> int:
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        print(f"Config not found: {config_file}", file=sys.stderr)
        return 1

    submission = Path(submission_path).expanduser().resolve()
    if not submission.exists():
        print(f"Submission not found: {submission}", file=sys.stderr)
        return 1

    csv_path = _resolve_benchmark_csv_path(benchmark_csv, config_file)
    if not csv_path.exists():
        print(f"Benchmark CSV not found: {csv_path}", file=sys.stderr)
        return 1

    out_dir = Path(output_dir).expanduser().resolve()

    config = load_config(str(config_file))
    config["benchmark"] = config.get("benchmark") or {}
    config["benchmark"]["csv_path"] = str(csv_path)
    if max_rows:
        config["benchmark"]["max_rows"] = max_rows
    config["run"] = config.get("run") or {}
    config["run"]["output_root"] = str(out_dir)
    # Use a fixed run_id so run_dir is deterministic per submission
    run_id = submission.stem + "_run"
    config["run"]["run_id"] = run_id

    paths = resolve_paths(config, str(config_file))
    run_dir = paths.run_dir

    main_llm_config = config.get("models", {}).get("main", {})
    if not main_llm_config:
        print("Config must define models.main for the fixed main LLM.", file=sys.stderr)
        return 1
    print("Creating fixed main LLM from config (models.main)...", flush=True)
    main_llm_provider = create_llm_provider(main_llm_config, "main")

    system_prompt = (config.get("pipeline") or {}).get("system_prompt")

    print("Loading guardrails from", submission, flush=True)
    input_guardrail, output_guardrail = load_guardrails_from_module(submission)
    print("Building ChatPipeline (fixed main LLM + system prompt + your guardrails)...", flush=True)
    pipeline = _build_pipeline(
        main_llm_provider, system_prompt, input_guardrail, output_guardrail
    )
    print("Loading benchmark cases from", csv_path, flush=True)
    cases = load_benchmark_cases(csv_path)
    print("Running pipeline on", len(cases), "cases...", flush=True)
    run_pipeline_on_benchmark(pipeline, cases, run_dir, max_rows=max_rows)
    print("Responses written to", run_dir, flush=True)

    if not skip_judge:
        responses_path = run_dir / f"{BASE_RESPONSES_TARGET}.csv"
        print("Running judge and metrics...", flush=True)
        metrics = run_judge_and_metrics(config, str(config_file), run_dir, responses_path)
        print("RRR (harmful):", metrics.get("rrr_harmful"), "FRR (safe):", metrics.get("frr_safe"), flush=True)
        print("Summary:", run_dir / summary_output_name(BASE_RESPONSES_TARGET), flush=True)
    else:
        print("Skipping judge (--skip-judge). Run judge + metrics manually with config pointing at", run_dir, flush=True)

    return 0


def _parse_args():
    p = argparse.ArgumentParser(description="Run hackathon evaluation for one submission.")
    p.add_argument("--submission", "-s", required=True, help="Path to participant module (defines get_guardrails()).")
    p.add_argument("--benchmark-csv", "-b", required=True, help="Path to benchmark CSV.")
    p.add_argument("--config", "-c", required=True, help="Path to benchmark YAML config (for judge model, etc.).")
    p.add_argument("--output-dir", "-o", default="results/hackathon", help="Output directory for run.")
    p.add_argument("--max-rows", type=int, default=None, help="Limit number of benchmark rows (for testing).")
    p.add_argument("--skip-judge", action="store_true", help="Only run pipeline; do not run judge/metrics.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(
        main(
            submission_path=args.submission,
            benchmark_csv=args.benchmark_csv,
            config_path=args.config,
            output_dir=args.output_dir,
            max_rows=args.max_rows,
            skip_judge=args.skip_judge,
        )
    )
