#!/usr/bin/env bash
# Build deploy/Dockerfile with Cloud Build and deploy it to Cloud Run.
#
#   deploy/deploy.sh                 # CPU service "hebrew-tts" (4 vCPU, 16 GiB)
#   GPU=1 deploy/deploy.sh           # GPU service "hebrew-tts-l4" (1x NVIDIA L4, 4 vCPU, 16 GiB)
#   PROJECT=... REGION=... deploy/deploy.sh
#
# The services are private (IAM). To use one from a browser/curl:
#   gcloud run services proxy hebrew-tts --region europe-west1 --port 8080
set -euo pipefail

PROJECT="${PROJECT:-project-71583fab-5cd0-4bba-a6f}"
REGION="${REGION:-europe-west1}"
REPO="${REPO:-hebrew-tts}"
GPU="${GPU:-0}"

if [ "$GPU" = "1" ]; then
  SERVICE="${SERVICE:-hebrew-tts-l4}"
  TORCH_INDEX=cu130
  RESOURCE_FLAGS=(--cpu 4 --memory 16Gi --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy)
else
  SERVICE="${SERVICE:-hebrew-tts}"
  TORCH_INDEX=cpu
  RESOURCE_FLAGS=(--cpu 4 --memory 16Gi)
fi

BUCKET="${BUCKET:-${PROJECT}-hebrew-tts-cache}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:$(date +%Y%m%d-%H%M%S)"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --project "$PROJECT"

if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" \
  >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" --repository-format docker \
    --location "$REGION" --project "$PROJECT"
fi

# Persistent storage for the validation-set audio. Cloud Run has no attachable disks; a Cloud
# Storage bucket mounted at /cache is the persistent option (a few GB at most, cents per month).
if ! gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://$BUCKET" --project "$PROJECT" --location "$REGION" \
    --uniform-bucket-level-access
fi
# The service runs as the default compute service account.
RUN_SA="$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --project "$PROJECT" \
  --member "serviceAccount:$RUN_SA" --role roles/storage.objectUser >/dev/null

# The weights hash is baked into the image (and verified there); the validation cache is keyed on it.
MODEL_SHA256="$(shasum -a 256 "$ROOT/artifacts/model.safetensors" | cut -d' ' -f1)"

# Stage only what the image needs: artifacts/ and deploy/voices are gitignored, so gcloud
# would otherwise skip them, and the repo has gigabytes of unrelated data.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/artifacts" "$STAGE/tokenizers"
cp "$ROOT/deploy/Dockerfile" "$STAGE/Dockerfile"
cp "$ROOT/artifacts/hebrew_24l.yaml" "$ROOT/artifacts/model.safetensors" "$STAGE/artifacts/"
cp "$ROOT/tokenizers/tokenizer.model" "$STAGE/tokenizers/"
for dir in pocket_tts training deploy; do
  rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude .DS_Store --exclude validation_cache "$ROOT/$dir" "$STAGE/"
done

cat > "$STAGE/cloudbuild.yaml" <<EOF
steps:
  - name: gcr.io/cloud-builders/docker
    args: [build, --build-arg, TORCH_INDEX=${TORCH_INDEX}, --build-arg, MODEL_SHA256=${MODEL_SHA256}, -t, ${IMAGE}, .]
images: [${IMAGE}]
options:
  machineType: E2_HIGHCPU_8
timeout: 1800s
EOF

gcloud builds submit "$STAGE" --config "$STAGE/cloudbuild.yaml" \
  --project "$PROJECT" --region "$REGION"

# --no-cpu-throttling: the model loads in a background thread after the port opens, and
# request-based billing would throttle the CPU between requests and stall the load.
# (Cloud Run GPUs require it anyway.)
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --project "$PROJECT" --region "$REGION" \
  "${RESOURCE_FLAGS[@]}" --cpu-boost --no-cpu-throttling \
  --min-instances 0 --max-instances 1 --concurrency 4 --timeout 900 \
  --execution-environment gen2 \
  --add-volume "name=cache,type=cloud-storage,bucket=$BUCKET" \
  --add-volume-mount "volume=cache,mount-path=/cache" \
  --port 8080 --no-allow-unauthenticated

gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format 'value(status.url)'
