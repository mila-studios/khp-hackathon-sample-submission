#!/usr/bin/env bash
set -euo pipefail

# Repo root = parent of project/ (where pyproject.toml and .venv live)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"

echo "== configure =="
echo "Mode: ${HACKATHON_MODE:-unknown} (gpu=${HACKATHON_NEEDS_GPU:-0} llm_judge=${HACKATHON_NEEDS_LLM_JUDGE:-0})"

# Goal of this step:
# - Install dependencies
# - Download / materialize any model files you need (optional)

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
"$PY" -m pip install -U pip >/dev/null

if [[ -f requirements.txt ]]; then
  echo "Installing Python dependencies from requirements.txt"
  "$PY" -m pip install -r requirements.txt
elif [[ -f pyproject.toml ]]; then
  echo "Installing project and dependencies from pyproject.toml"
  "$PY" -m pip install -e .
else
  echo "No requirements.txt or pyproject.toml found. Skipping dependency install."
fi

echo "Ensuring ipykernel is available for notebooks"
"$PY" -m pip install ipykernel

echo "OK"
