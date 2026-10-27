#!/usr/bin/env bash
# Upload a frozen semantic encoder to the model store (gs://<project>-hebrew-tts-models/
# encoders/<sha256 of its main file>/). Once per encoder revision, not once per model: every model
# whose config names the encoder (flow_lm.semantic_conditioner: encoder, model_name, revision)
# finds it there by itself, and a model without the section needs none.
#
#   deploy/publish_encoder.sh dictabert                     # dicta-il/dictabert-large, pinned revision
#   deploy/publish_encoder.sh renikud_plus                  # notmax123/RenikudPlus model.onnx (fp32)
#   deploy/publish_encoder.sh renikud_plus <commit>         # another revision (then name it in the config)
#   NOTE="..." deploy/publish_encoder.sh dictabert
#
# The files are downloaded from Hugging Face here (not by the server) at the revision the shipped
# configs pin, hashed, and uploaded with a manifest last. A published encoder is immutable:
# publishing the same file again is a no-op, and the same main file with a different set of other
# files is refused. Prints the sha256, which is only needed to pin it (SEMANTIC_ENCODER_SHA256).
# No image build is involved: the server image is the same for every encoder and model.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

ENCODER="${1:?usage: deploy/publish_encoder.sh dictabert|renikud_plus [revision]}"
REVISION="${2:-}"
NOTE="${NOTE:-}"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

ARGS=("$ENCODER" --out "$STAGE" --note "$NOTE")
[ -n "$REVISION" ] && ARGS+=(--revision "$REVISION")
echo "downloading and hashing $ENCODER ..."
FILES=()
while IFS= read -r line; do FILES+=("$line"); done < <(cd "$ROOT" && uv run python -m deploy.publish_encoder "${ARGS[@]}")
SHA="${FILES[0]}"
FILES=("${FILES[@]:1}")

gcloud services enable storage.googleapis.com --project "$PROJECT" >/dev/null
if ! gcloud storage buckets describe "gs://$MODELS_BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$MODELS_BUCKET" --project "$PROJECT" --location "$REGION" \
    --uniform-bucket-level-access
fi
gcloud storage buckets add-iam-policy-binding "gs://$MODELS_BUCKET" --project "$PROJECT" \
  --member "serviceAccount:$(run_service_account)" --role roles/storage.objectViewer >/dev/null

DEST="gs://$MODELS_BUCKET/encoders/$SHA"
if gcloud storage ls "$DEST/manifest.json" --project "$PROJECT" >/dev/null 2>&1; then
  # The same main file, so the same encoder: only the other files' hashes are compared.
  PUBLISHED="$(gcloud storage cat "$DEST/manifest.json" --project "$PROJECT")"
  for name in "${FILES[@]}"; do
    FILE_SHA="$(sha256_of "$STAGE/$name")"
    if ! grep -q "$FILE_SHA" <<<"$PUBLISHED"; then
      echo "$SHA is already published with a different $name; published encoders are immutable." >&2
      exit 1
    fi
  done
  echo "already published: $DEST"
  echo "$SHA"
  exit 0
fi

# The manifest goes last: an encoder without one is an interrupted upload and is never used.
for name in "${FILES[@]}"; do
  gcloud storage cp --no-clobber "$STAGE/$name" "$DEST/$name" --project "$PROJECT"
done
gcloud storage cp --no-clobber "$STAGE/manifest.json" "$DEST/manifest.json" --project "$PROJECT"

echo "published $DEST"
echo "$SHA"
