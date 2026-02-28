#!/usr/bin/env bash
set -euo pipefail

# Repo root = parent of project/ (where pyproject.toml and .venv live)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR=".venv"
HACKATHON_JSON="$REPO_ROOT/hackathon.json"

echo "== configure =="

# Goal of this step:
# - Install dependencies
# - Download / materialize model artifacts from hackathon.json (optional)

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

if [[ ! -f "$HACKATHON_JSON" ]]; then
  echo "ERROR: hackathon.json not found at $HACKATHON_JSON" >&2
  exit 1
fi

# Parse and apply hackathon runtime configuration.
# - needs_gpu: if true, fail fast when CUDA is unavailable
# - artifacts: optional list of downloadable resources
export REPO_ROOT HACKATHON_JSON
"$PY" - <<'PY'
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if not str(member_path).startswith(str(destination)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tar.extractall(destination)


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


repo_root = Path(os.environ["REPO_ROOT"]).resolve()
hackathon_json = Path(os.environ["HACKATHON_JSON"]).resolve()
cfg = json.loads(hackathon_json.read_text())
if not isinstance(cfg, dict):
    raise RuntimeError("hackathon.json must be a JSON object")

if "needs_gpu" not in cfg:
    raise RuntimeError("hackathon.json missing required field: needs_gpu")
if not isinstance(cfg["needs_gpu"], bool):
    raise RuntimeError("hackathon.json field 'needs_gpu' must be a boolean")
needs_gpu = cfg["needs_gpu"]

if "artifacts" not in cfg:
    raise RuntimeError("hackathon.json missing required field: artifacts")
artifacts = cfg["artifacts"]
if not isinstance(artifacts, list):
    raise RuntimeError("hackathon.json field 'artifacts' must be a list")

for idx, artifact in enumerate(artifacts):
    if not isinstance(artifact, dict):
        raise RuntimeError(f"Artifact #{idx} must be an object")

    uri = artifact.get("uri")
    destination = artifact.get("destination")
    required = bool(artifact.get("required", True))
    expected_sha256 = artifact.get("sha256")
    if "required" in artifact and not isinstance(artifact["required"], bool):
        raise RuntimeError(f"Artifact #{idx} field 'required' must be a boolean")
    if expected_sha256 is not None and not isinstance(expected_sha256, str):
        raise RuntimeError(f"Artifact #{idx} field 'sha256' must be a string when provided")

    if not isinstance(uri, str) or not uri.strip():
        raise RuntimeError(f"Artifact #{idx} missing required string field: uri")
    if not isinstance(destination, str) or not destination.strip():
        raise RuntimeError(f"Artifact #{idx} missing required string field: destination")

    dest_path = Path(destination)
    if not dest_path.is_absolute():
        dest_path = (repo_root / dest_path).resolve()
    else:
        dest_path = dest_path.resolve()

    # Keep artifacts inside the repository for safety.
    if not str(dest_path).startswith(str(repo_root)):
        raise RuntimeError(f"Artifact destination must stay under repo root: {dest_path}")

    print(f"[configure] Fetching artifact #{idx}: {uri} -> {dest_path}", file=sys.stderr)
    with tempfile.TemporaryDirectory() as td:
        tmp_file = Path(td) / "artifact.bin"
        try:
            if uri.startswith("s3://"):
                result = subprocess.run(
                    ["aws", "s3", "cp", uri, str(tmp_file)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.strip() or "aws s3 cp failed")
            elif uri.startswith("http://") or uri.startswith("https://"):
                with urllib.request.urlopen(uri) as r:
                    if r.status < 200 or r.status >= 300:
                        raise RuntimeError(f"HTTP download failed with status {r.status}")
                    with tmp_file.open("wb") as f:
                        shutil.copyfileobj(r, f)
            else:
                raise RuntimeError("Unsupported uri scheme; use s3:// or https://")

            if expected_sha256:
                got = sha256sum(tmp_file)
                if got.lower() != str(expected_sha256).lower():
                    raise RuntimeError(f"SHA256 mismatch for {uri}: expected {expected_sha256}, got {got}")

            lower_uri = uri.lower()
            if lower_uri.endswith(".tar.gz") or lower_uri.endswith(".tgz"):
                dest_path.mkdir(parents=True, exist_ok=True)
                safe_extract_tar(tmp_file, dest_path)
            elif lower_uri.endswith(".zip"):
                dest_path.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(tmp_file, "r") as zf:
                    for member in zf.namelist():
                        member_path = (dest_path / member).resolve()
                        if not str(member_path).startswith(str(dest_path)):
                            raise RuntimeError(f"Unsafe zip member path: {member}")
                    zf.extractall(dest_path)
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp_file, dest_path)

        except Exception as exc:
            if required:
                raise
            print(f"[configure] Optional artifact failed ({uri}): {exc}", file=sys.stderr)

# If a GPU is required, verify CUDA availability after dependencies are installed.
if needs_gpu:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"needs_gpu=true but torch is not available: {exc}")
    if not torch.cuda.is_available():
        raise RuntimeError("needs_gpu=true but CUDA is not available on this machine")
PY

echo "OK"
