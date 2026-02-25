#!/usr/bin/env bash
# Predict script: fixed name and path for hackathon submissions.
# Usage: scripts/predict.sh <path_to_input_file.csv> <path_to_predictions_output_file.csv>
#
# Reads the input CSV and writes predictions to the output CSV using this team's
# guardrail (get_predictions.py). Override submission via SUBMISSION_MODULE env var.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: scripts/predict.sh <path_to_input_file.csv> <path_to_predictions_output_file.csv>" >&2
  exit 1
fi

INPUT_CSV="$1"
OUTPUT_CSV="$2"

# Project root = parent of scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Submission module (each team sets this or edits the default)
SUBMISSION_MODULE="${SUBMISSION_MODULE:-src/benchmark/example_submission_llm_judge.py}"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Input CSV not found: $INPUT_CSV" >&2
  exit 1
fi

# get_predictions.py writes to <output_dir>/predictions.csv; we then move to requested path
OUTPUT_DIR="$(dirname "$OUTPUT_CSV")"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"
PYTHONPATH=. python scripts/get_predictions.py \
  --submission "$SUBMISSION_MODULE" \
  --data "$INPUT_CSV" \
  --output-dir "$OUTPUT_DIR"

# Write to the exact path requested (script writes predictions.csv into output-dir)
if [[ -f "$OUTPUT_DIR/predictions.csv" ]]; then
  mv "$OUTPUT_DIR/predictions.csv" "$OUTPUT_CSV"
  echo "Predictions written to $OUTPUT_CSV"
else
  echo "Predictions file was not produced." >&2
  exit 1
fi
