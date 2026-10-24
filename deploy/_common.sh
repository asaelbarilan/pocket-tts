# Sourced by deploy.sh, publish_model.sh and swap_model.sh: project, service and bucket names.
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

# Validation-set audio cache (read-write) and the model store (read-only for the service).
BUCKET="${BUCKET:-${PROJECT}-hebrew-tts-cache}"
MODELS_BUCKET="${MODELS_BUCKET:-${PROJECT}-hebrew-tts-models}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The service runs as the default compute service account.
run_service_account() {
  echo "$(gcloud projects describe "$PROJECT" --format 'value(projectNumber)')-compute@developer.gserviceaccount.com"
}

sha256_of() {
  shasum -a 256 "$1" | cut -d' ' -f1
}

# Fails unless `$1` (a weights sha256) is fully published in the model store (the manifest is
# uploaded last, so its presence means the model is complete).
require_published() {
  if ! gcloud storage ls "gs://$MODELS_BUCKET/$1/manifest.json" --project "$PROJECT" >/dev/null 2>&1; then
    echo "model $1 is not in gs://$MODELS_BUCKET: run deploy/publish_model.sh first" >&2
    exit 1
  fi
}
