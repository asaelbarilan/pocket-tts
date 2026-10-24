#!/usr/bin/env bash
# Upload a model to the model store (gs://<project>-hebrew-tts-models/<weights sha256>/).
#
#   deploy/publish_model.sh                          # artifacts/model.safetensors + config + tokenizer
#   deploy/publish_model.sh path/to/model.safetensors
#   NOTE="run hebrew-v3 step 30000" deploy/publish_model.sh
#
# Prints the weights sha256: that hash IS the model. Serve it with
#   deploy/deploy.sh                      (first deploy / code change; uses the local weights' hash)
#   deploy/swap_model.sh <sha256>         (just a new model: no image build)
# A published model is immutable: publishing the same weights again is a no-op, and the same
# weights with a different config or tokenizer is refused.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

WEIGHTS="${1:-$ROOT/artifacts/model.safetensors}"
CONFIG="${CONFIG:-$ROOT/artifacts/hebrew_24l.yaml}"
TOKENIZER="${TOKENIZER:-$ROOT/tokenizers/tokenizer.model}"
NOTE="${NOTE:-}"

for f in "$WEIGHTS" "$CONFIG" "$TOKENIZER"; do
  [ -s "$f" ] || { echo "missing or empty: $f" >&2; exit 1; }
done

echo "hashing $WEIGHTS ..."
SHA="$(sha256_of "$WEIGHTS")"
CONFIG_SHA="$(sha256_of "$CONFIG")"
TOKENIZER_SHA="$(sha256_of "$TOKENIZER")"
size() { wc -c < "$1" | tr -d ' '; }

gcloud services enable storage.googleapis.com --project "$PROJECT" >/dev/null
if ! gcloud storage buckets describe "gs://$MODELS_BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$MODELS_BUCKET" --project "$PROJECT" --location "$REGION" \
    --uniform-bucket-level-access
fi
gcloud storage buckets add-iam-policy-binding "gs://$MODELS_BUCKET" --project "$PROJECT" \
  --member "serviceAccount:$(run_service_account)" --role roles/storage.objectViewer >/dev/null

DEST="gs://$MODELS_BUCKET/$SHA"
if gcloud storage ls "$DEST/manifest.json" --project "$PROJECT" >/dev/null 2>&1; then
  PUBLISHED="$(gcloud storage cat "$DEST/manifest.json" --project "$PROJECT")"
  if ! grep -q "$CONFIG_SHA" <<<"$PUBLISHED" || ! grep -q "$TOKENIZER_SHA" <<<"$PUBLISHED"; then
    echo "$SHA is already published with a different config or tokenizer; published models are" \
      "immutable." >&2
    exit 1
  fi
  echo "already published: $DEST"
  echo "$SHA"
  exit 0
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cat > "$STAGE/manifest.json" <<EOF
{
  "format": 1,
  "created": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "note": "$NOTE",
  "files": {
    "model.safetensors": {"sha256": "$SHA", "size": $(size "$WEIGHTS")},
    "hebrew_24l.yaml": {"sha256": "$CONFIG_SHA", "size": $(size "$CONFIG")},
    "tokenizer.model": {"sha256": "$TOKENIZER_SHA", "size": $(size "$TOKENIZER")}
  }
}
EOF

# The manifest goes last: a model without one is an interrupted upload and is never served.
gcloud storage cp --no-clobber "$WEIGHTS" "$DEST/model.safetensors" --project "$PROJECT"
gcloud storage cp --no-clobber "$CONFIG" "$DEST/hebrew_24l.yaml" --project "$PROJECT"
gcloud storage cp --no-clobber "$TOKENIZER" "$DEST/tokenizer.model" --project "$PROJECT"
gcloud storage cp --no-clobber "$STAGE/manifest.json" "$DEST/manifest.json" --project "$PROJECT"

echo "published $DEST"
echo "$SHA"
