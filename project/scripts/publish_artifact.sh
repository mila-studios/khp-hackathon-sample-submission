#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: scripts/publish_artifact.sh <local_path> <s3_uri> <destination_in_repo> [required:true|false]" >&2
  exit 1
fi

LOCAL_PATH="$1"
S3_URI="$2"
DESTINATION="$3"
REQUIRED="${4:-true}"
REQUIRED_LOWER="$(printf '%s' "$REQUIRED" | tr '[:upper:]' '[:lower:]')"
case "$REQUIRED_LOWER" in
  true|false)
    REQUIRED="$REQUIRED_LOWER"
    ;;
  *)
    echo "required must be true or false (got: $REQUIRED)" >&2
    exit 1
    ;;
esac

if [[ ! -e "$LOCAL_PATH" ]]; then
  echo "Local path not found: $LOCAL_PATH" >&2
  exit 1
fi

if [[ "$S3_URI" != s3://* ]]; then
  echo "s3_uri must start with s3:// (got: $S3_URI)" >&2
  exit 1
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required. Install AWS CLI and configure credentials." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

TO_UPLOAD=""
BASENAME="$(basename "$LOCAL_PATH")"
if [[ -d "$LOCAL_PATH" ]]; then
  ARCHIVE_NAME="${BASENAME}.tar.gz"
  TO_UPLOAD="$TMP_DIR/$ARCHIVE_NAME"
  tar -czf "$TO_UPLOAD" -C "$(dirname "$LOCAL_PATH")" "$BASENAME"
else
  TO_UPLOAD="$LOCAL_PATH"
fi

if command -v sha256sum >/dev/null 2>&1; then
  SHA256="$(sha256sum "$TO_UPLOAD" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  SHA256="$(shasum -a 256 "$TO_UPLOAD" | awk '{print $1}')"
else
  echo "Could not find sha256sum or shasum." >&2
  exit 1
fi

SIZE_BYTES="$(wc -c < "$TO_UPLOAD" | tr -d ' ')"

echo "Uploading $TO_UPLOAD -> $S3_URI"
aws s3 cp "$TO_UPLOAD" "$S3_URI"

echo
echo "size_bytes: $SIZE_BYTES (info only; do not include in hackathon.json)"
echo
echo "Copy this into hackathon.json artifacts:"
cat <<EOF
{
  "uri": "$S3_URI",
  "destination": "$DESTINATION",
  "sha256": "$SHA256",
  "required": $REQUIRED
}
EOF
