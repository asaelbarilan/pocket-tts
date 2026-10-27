#!/usr/bin/env bash
# Build deploy/Dockerfile (code only, no weights) with Cloud Build and deploy it to Cloud Run,
# serving one published model.
#
#   deploy/publish_model.sh                       # once per model (uploads to the model store)
#   deploy/deploy.sh                              # CPU service "hebrew-tts" (4 vCPU, 16 GiB)
#   GPU=1 deploy/deploy.sh                        # GPU service "hebrew-tts-l4" (1x NVIDIA L4)
#   MODEL_SHA256=<sha256> deploy/deploy.sh        # a specific published model (default: the
#                                                 # local artifacts/model.safetensors)
#   IMAGE=<image> deploy/deploy.sh                # redeploy an existing image, no build
#   PROJECT=... REGION=... deploy/deploy.sh
#
# A new model alone does not need this: deploy/swap_model.sh changes the model without a build,
# whichever semantic encoder it uses (publish it once with deploy/publish_encoder.sh).
#
# The services are private (IAM). To use one from a browser/curl:
#   gcloud run services proxy hebrew-tts --region europe-west1 --port 8080
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODEL_SHA256="${MODEL_SHA256:-$(sha256_of "$ROOT/artifacts/model.safetensors")}"
if [ -n "${IMAGE:-}" ]; then
  BUILD=0
else
  BUILD=1
  IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:$(date +%Y%m%d-%H%M%S)"
fi

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --project "$PROJECT"

# Persistent storage for the validation-set audio. Cloud Run has no attachable disks; a Cloud
# Storage bucket mounted at /cache is the persistent option (a few GB at most, cents per month).
if ! gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION" \
    --uniform-bucket-level-access
fi
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --project "$PROJECT" \
  --member "serviceAccount:$(run_service_account)" --role roles/storage.objectUser >/dev/null

# The model store is mounted read-only; deploy/publish_model.sh creates it and grants access.
require_published "$MODEL_SHA256"

if [ "$BUILD" = "1" ]; then
  if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" \
    >/dev/null 2>&1; then
    gcloud artifacts repositories create "$REPO" --repository-format docker \
      --location "$REGION" --project "$PROJECT"
  fi

  # Stage only the code: the repo has gigabytes of unrelated data, and the weights are not
  # part of the image (they come from the model store at startup).
  STAGE="$(mktemp -d)"
  trap 'rm -rf "$STAGE"' EXIT
  cp "$ROOT/deploy/Dockerfile" "$STAGE/Dockerfile"
  for dir in pocket_tts training deploy; do
    rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store --exclude validation_cache --exclude milim_bench_data "$ROOT/$dir" "$STAGE/"
  done

  cat > "$STAGE/cloudbuild.yaml" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: [build, --build-arg, TORCH_INDEX=${TORCH_INDEX}, -t, ${IMAGE}, .]
images: [${IMAGE}]
options:
  machineType: E2_HIGHCPU_8
timeout: 1800s
EOF

  gcloud builds submit "$STAGE" --config "$STAGE/cloudbuild.yaml" \
    --project "$PROJECT" --region "$REGION"
fi

# --no-cpu-throttling: the model loads in a background thread after the port opens, and
# request-based billing would throttle the CPU between requests and stall the load.
# (Cloud Run GPUs require it anyway.)
# The startup probe (/ready answers 200 only once the model is loaded and verified, up to 240 s)
# keeps a revision that cannot load its model from taking traffic: the old one keeps serving.
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --project "$PROJECT" --region "$REGION" \
  "${RESOURCE_FLAGS[@]}" --cpu-boost --no-cpu-throttling \
  --min-instances 0 --max-instances 1 --concurrency 4 --timeout 900 \
  --execution-environment gen2 \
  --add-volume "name=cache,type=cloud-storage,bucket=$BUCKET" \
  --add-volume-mount "volume=cache,mount-path=/cache" \
  --add-volume "name=models,type=cloud-storage,bucket=$MODELS_BUCKET,readonly=true" \
  --add-volume-mount "volume=models,mount-path=/models" \
  --set-env-vars "MODEL_SHA256=$MODEL_SHA256" \
  --startup-probe "httpGet.path=/ready,periodSeconds=5,timeoutSeconds=3,failureThreshold=48" \
  --port 8080 --no-allow-unauthenticated

gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format 'value(status.url)'
