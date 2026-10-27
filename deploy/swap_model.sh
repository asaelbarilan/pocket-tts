#!/usr/bin/env bash
# Point the running Cloud Run service at another published model. No image build: it only
# updates MODEL_SHA256, which makes a new revision. If that revision cannot load the model
# (its /ready startup probe never passes) Cloud Run keeps the old revision serving.
#
#   deploy/swap_model.sh                  # the model of artifacts/model.safetensors
#   deploy/swap_model.sh <sha256>         # any published model (a rollback is the old hash)
#   GPU=1 deploy/swap_model.sh ...        # the GPU service
#
# Publish first with deploy/publish_model.sh, and, if the model's config has a semantic_conditioner
# section, its encoder with deploy/publish_encoder.sh (once per encoder revision; the server finds it
# by the identity the config names, so the model's hash is all that changes here). Only for weights
# the image's code can run: a new architecture or code change needs deploy/deploy.sh.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SHA="${1:-$(sha256_of "$ROOT/artifacts/model.safetensors")}"
require_published "$SHA"

gcloud run services update "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --update-env-vars "MODEL_SHA256=$SHA"

gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format 'value(status.url)'
