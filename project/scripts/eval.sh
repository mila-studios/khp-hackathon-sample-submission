#!/usr/bin/env bash
# Eval script: evaluate predictions for a team and write eval metrics.
# Usage: scripts/eval.sh <path_to_predictions.csv> <path_to_eval_metrics.csv>
#
# Reads the team's predictions CSV and writes evaluation metrics (precision,
# recall, F1, latency) to eval_metrics.csv.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: scripts/eval.sh <path_to_predictions.csv> <path_to_eval_metrics.csv>" >&2
  exit 1
fi

PREDICTIONS_CSV="$1"
EVAL_METRICS_CSV="$2"

# Project root = parent of scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$PREDICTIONS_CSV" ]]; then
  echo "Predictions CSV not found: $PREDICTIONS_CSV" >&2
  exit 1
fi

# get_guardrail_metrics.py writes metrics.csv to the same dir as --output (JSON)
OUTPUT_DIR="$(dirname "$EVAL_METRICS_CSV")"
mkdir -p "$OUTPUT_DIR"

cd "$PROJECT_ROOT"
PYTHONPATH=. python scripts/get_guardrail_metrics.py \
  --predictions "$PREDICTIONS_CSV" \
  --output "$OUTPUT_DIR/eval_metrics.json"

# Rename metrics.csv to the requested eval_metrics.csv path
if [[ -f "$OUTPUT_DIR/metrics.csv" ]]; then
  mv "$OUTPUT_DIR/metrics.csv" "$EVAL_METRICS_CSV"
  echo "Eval metrics written to $EVAL_METRICS_CSV"
else
  echo "Metrics CSV was not produced." >&2
  exit 1
fi
