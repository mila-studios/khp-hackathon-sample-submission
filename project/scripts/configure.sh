#!/usr/bin/env bash
# Configure script: prepare environment and model artifacts for submissions.
# Usage: scripts/configure.sh
#
# Installs project dependencies and optionally downloads/materializes model
# artifacts (e.g., from Google Drive or S3), then unpacks archives when enabled.

set -euo pipefail

# Repo root = parent of project/ (where pyproject.toml and .venv live)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv}"
# Participants will need to set the MODEL_PROVIDER and MODEL_URI to the appropriate values.
MODEL_PROVIDER="${MODEL_PROVIDER:-gdrive}"  # none | gdrive | s3
MODEL_URI="${MODEL_URI:-https://drive.google.com/file/d/1YpYbNqFne6RvySVsiz8VBib7J4J5WLr0/view?usp=sharing}"                # gdrive URL/id or s3://bucket/prefix
MODEL_LOCAL_DIR="${MODEL_LOCAL_DIR:-project/models}"
MODEL_UNPACK_ARCHIVES="${MODEL_UNPACK_ARCHIVES:-1}"         # 1=auto-extract archives after download

echo "== configure =="
echo "Mode: ${HACKATHON_MODE:-unknown} (gpu=${HACKATHON_NEEDS_GPU:-0} llm_judge=${HACKATHON_NEEDS_LLM_JUDGE:-0})"

# Goal of this step:
# - Install dependencies
# - Download / materialize any model files you need (optional)

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtualenv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi
PY="$REPO_ROOT/$VENV_DIR/bin/python"
"$PY" -m pip install -U pip >/dev/null

unpack_archives_if_needed() {
  local target_path="$1"
  if [[ "$MODEL_UNPACK_ARCHIVES" != "1" ]]; then
    echo "Archive unpack disabled (MODEL_UNPACK_ARCHIVES=$MODEL_UNPACK_ARCHIVES)"
    return
  fi

  "$PY" - "$target_path" <<'PY'
import gzip
import shutil
import sys
from pathlib import Path

target = Path(sys.argv[1])
suffixes = (".zip", ".tar", ".tgz", ".tar.gz", ".tbz2", ".tar.bz2", ".txz", ".tar.xz", ".gz")
candidates = ([target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()])
candidates = [p for p in candidates if p.name.lower().endswith(suffixes)]

if not candidates:
    print("No archives detected. Using files as-is.")
    raise SystemExit(0)

print(f"Found {len(candidates)} archive(s). Extracting...")
for archive in candidates:
    name = archive.name.lower()
    try:
        if name.endswith(".gz") and not name.endswith(".tar.gz"):
            with gzip.open(archive, "rb") as src, open(archive.with_suffix(""), "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.unpack_archive(str(archive), str(archive.parent))
        print(f" - extracted {archive}")
    except (shutil.ReadError, ValueError):
        print(f" - skipped unsupported archive: {archive}")
PY
}

download_gdrive() {
  local uri="$1"
  local out_dir="$2"
  local file_id=""

  # Primary path: download to the target directory.
  if "$PY" -m gdown --fuzzy "$uri" -O "$out_dir/"; then
    return 0
  fi

  # Fallback: extract file id and use uc export URL.
  if [[ "$uri" =~ /d/([^/]+) ]]; then
    file_id="${BASH_REMATCH[1]}"
  elif [[ "$uri" =~ id=([^&]+) ]]; then
    file_id="${BASH_REMATCH[1]}"
  fi
  [[ -z "$file_id" ]] && return 1

  echo "Retrying Google Drive download via direct uc URL..."
  "$PY" -m gdown "https://drive.google.com/uc?export=download&id=$file_id" -O "$out_dir/"
}

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

# Optional model materialization for submissions.
case "$MODEL_PROVIDER" in
  none|"")
    echo "Model sync disabled (MODEL_PROVIDER=none)"
    ;;
  gdrive)
    if [[ -z "$MODEL_URI" ]]; then
      echo "MODEL_URI is required when MODEL_PROVIDER=gdrive" >&2
      exit 1
    fi
    echo "Downloading model artifacts from Google Drive -> $MODEL_LOCAL_DIR"
    mkdir -p "$MODEL_LOCAL_DIR"
    "$PY" -m pip install gdown
    if [[ "$MODEL_URI" == *"/folders/"* ]]; then
      "$PY" -m gdown --folder --fuzzy "$MODEL_URI" -O "$MODEL_LOCAL_DIR"
    else
      download_gdrive "$MODEL_URI" "$MODEL_LOCAL_DIR"
    fi
    unpack_archives_if_needed "$MODEL_LOCAL_DIR"
    ;;
  s3)
    if [[ -z "$MODEL_URI" ]]; then
      echo "MODEL_URI is required when MODEL_PROVIDER=s3" >&2
      exit 1
    fi
    if ! command -v aws >/dev/null 2>&1; then
      echo "aws CLI is required when MODEL_PROVIDER=s3" >&2
      exit 1
    fi
    echo "Syncing model artifacts from S3 -> $MODEL_LOCAL_DIR"
    mkdir -p "$MODEL_LOCAL_DIR"
    aws s3 sync "$MODEL_URI" "$MODEL_LOCAL_DIR"
    unpack_archives_if_needed "$MODEL_LOCAL_DIR"
    ;;
  *)
    echo "Unsupported MODEL_PROVIDER: $MODEL_PROVIDER (use none|gdrive|s3)" >&2
    exit 1
    ;;
esac

echo "OK"
