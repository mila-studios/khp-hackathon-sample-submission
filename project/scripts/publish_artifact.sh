#!/usr/bin/env bash
set -euo pipefail

# Required environment variables:
#   S3_BUCKET_NAME  - bucket name
#   S3_ENDPOINT     - endpoint URL (e.g., https://10.2.16.3)
#   S3_ACCESS_KEY   - access key
#   S3_SECRET_KEY   - secret key

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: scripts/publish_artifact.sh <team_id> <local_path> <destination> [required:true|false]" >&2
  exit 1
fi

TEAM_ID="$1"
LOCAL_PATH="$2"
DESTINATION="$3"
REQUIRED="${4:-true}"

# Validate team_id format: team_XXX where XXX is 001-100
if [[ ! "$TEAM_ID" =~ ^team_[0-9]{3}$ ]]; then
  echo "Invalid team_id format: $TEAM_ID (must be team_XXX where XXX is 001-100)" >&2
  exit 1
fi
TEAM_NUM="${TEAM_ID#team_}"
TEAM_NUM_INT=$((10#$TEAM_NUM))  # Force base-10 interpretation
if [[ $TEAM_NUM_INT -lt 1 || $TEAM_NUM_INT -gt 100 ]]; then
  echo "Invalid team_id: $TEAM_ID (number must be between 001 and 100)" >&2
  exit 1
fi
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

# Read from environment variables
BUCKET_NAME="${S3_BUCKET_NAME:-}"
S3_ENDPOINT="${S3_ENDPOINT:-}"
ACCESS_KEY="${S3_ACCESS_KEY:-}"
SECRET_KEY="${S3_SECRET_KEY:-}"

if [[ -z "$BUCKET_NAME" ]]; then
  echo "Missing environment variable: S3_BUCKET_NAME" >&2
  exit 1
fi

if [[ -z "$S3_ENDPOINT" ]]; then
  echo "Missing environment variable: S3_ENDPOINT" >&2
  exit 1
fi

if [[ -z "$ACCESS_KEY" || -z "$SECRET_KEY" ]]; then
  echo "Missing environment variable: S3_ACCESS_KEY or S3_SECRET_KEY" >&2
  exit 1
fi

# Remove protocol from endpoint for Host header
S3_HOST="${S3_ENDPOINT#https://}"
S3_HOST="${S3_HOST#http://}"

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
  OBJECT_KEY="${TEAM_ID}/${ARCHIVE_NAME}"
else
  TO_UPLOAD="$LOCAL_PATH"
  OBJECT_KEY="${TEAM_ID}/${BASENAME}"
fi

S3_PATH="s3://${BUCKET_NAME}/${OBJECT_KEY}"

# Compute file hash
if command -v sha256sum >/dev/null 2>&1; then
  SHA256="$(sha256sum "$TO_UPLOAD" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  SHA256="$(shasum -a 256 "$TO_UPLOAD" | awk '{print $1}')"
else
  echo "Could not find sha256sum or shasum." >&2
  exit 1
fi

SIZE_BYTES="$(wc -c < "$TO_UPLOAD" | tr -d ' ')"

# S3 Signature Version 4 signing
REGION="us-east-1"
SERVICE="s3"
DATE_STAMP="$(date -u +%Y%m%d)"
AMZ_DATE="$(date -u +%Y%m%dT%H%M%SZ)"
CONTENT_TYPE="application/octet-stream"

# Canonical request components
HTTP_METHOD="PUT"
CANONICAL_URI="/${BUCKET_NAME}/${OBJECT_KEY}"
CANONICAL_QUERYSTRING=""
CANONICAL_HEADERS="content-type:${CONTENT_TYPE}\nhost:${S3_HOST}\nx-amz-content-sha256:${SHA256}\nx-amz-date:${AMZ_DATE}"
SIGNED_HEADERS="content-type;host;x-amz-content-sha256;x-amz-date"

# Create canonical request
CANONICAL_REQUEST="${HTTP_METHOD}
${CANONICAL_URI}
${CANONICAL_QUERYSTRING}
content-type:${CONTENT_TYPE}
host:${S3_HOST}
x-amz-content-sha256:${SHA256}
x-amz-date:${AMZ_DATE}

${SIGNED_HEADERS}
${SHA256}"

# Create string to sign
ALGORITHM="AWS4-HMAC-SHA256"
CREDENTIAL_SCOPE="${DATE_STAMP}/${REGION}/${SERVICE}/aws4_request"
CANONICAL_REQUEST_HASH="$(printf '%s' "$CANONICAL_REQUEST" | openssl dgst -sha256 | awk '{print $NF}')"
STRING_TO_SIGN="${ALGORITHM}
${AMZ_DATE}
${CREDENTIAL_SCOPE}
${CANONICAL_REQUEST_HASH}"

# Calculate signature
hmac_sha256() {
  printf '%s' "$2" | openssl dgst -sha256 -mac HMAC -macopt "hexkey:$1" | awk '{print $NF}'
}

hmac_sha256_key() {
  printf '%s' "$2" | openssl dgst -sha256 -mac HMAC -macopt "key:$1" | awk '{print $NF}'
}

DATE_KEY="$(hmac_sha256_key "AWS4${SECRET_KEY}" "$DATE_STAMP")"
DATE_REGION_KEY="$(hmac_sha256 "$DATE_KEY" "$REGION")"
DATE_REGION_SERVICE_KEY="$(hmac_sha256 "$DATE_REGION_KEY" "$SERVICE")"
SIGNING_KEY="$(hmac_sha256 "$DATE_REGION_SERVICE_KEY" "aws4_request")"
SIGNATURE="$(hmac_sha256 "$SIGNING_KEY" "$STRING_TO_SIGN")"

# Create authorization header
AUTHORIZATION="${ALGORITHM} Credential=${ACCESS_KEY}/${CREDENTIAL_SCOPE}, SignedHeaders=${SIGNED_HEADERS}, Signature=${SIGNATURE}"

# Upload using curl
UPLOAD_URL="${S3_ENDPOINT}/${BUCKET_NAME}/${OBJECT_KEY}"

echo "Uploading $TO_UPLOAD -> $UPLOAD_URL"
curl -X PUT \
  -H "Content-Type: ${CONTENT_TYPE}" \
  -H "Host: ${S3_HOST}" \
  -H "x-amz-content-sha256: ${SHA256}" \
  -H "x-amz-date: ${AMZ_DATE}" \
  -H "Authorization: ${AUTHORIZATION}" \
  --data-binary "@${TO_UPLOAD}" \
  --insecure \
  "${UPLOAD_URL}"

echo
echo "size_bytes: $SIZE_BYTES (info only; do not include in hackathon.json)"
echo
echo "Copy this into hackathon.json artifacts:"
cat <<EOF
uri  "$S3_PATH"
destination  "$DESTINATION"
sha256  "$SHA256"
required  $REQUIRED
EOF
