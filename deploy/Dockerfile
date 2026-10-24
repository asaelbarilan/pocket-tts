# CPU image for deploy/hebrew_tts_server.py. COPY paths are relative to the repo root:
#   docker build -f deploy/Dockerfile .          (or use deploy/deploy.sh for Cloud Run)
# Code only: a new model never needs a new image (deploy/swap_model.sh).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# TORCH_INDEX: cpu (default) or cu130 (Cloud Run GPUs: driver 580 / CUDA 13.0).
ARG TORCH_INDEX=cpu
RUN pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/${TORCH_INDEX}
COPY deploy/requirements.txt deploy/requirements.txt
RUN pip install -r deploy/requirements.txt

# MILIM-Bench (renikud/MILIM-Bench), the whole dataset repo, downloaded at build time and pinned
# to a revision; the server reads data/gold.tsv from it for the test page's sample picker.
ARG MILIM_REVISION=0cdf63745681f65dc33b5006f781f652c597f4b2
ENV MILIM_DIR=/app/milim-bench
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('renikud/MILIM-Bench', repo_type='dataset', revision='${MILIM_REVISION}', local_dir='${MILIM_DIR}')" \
    && test -s "${MILIM_DIR}/data/gold.tsv"

# No weights in the image. The model (weights, config, tokenizer) is copied at startup from the
# model store, a directory of published models (on Cloud Run a read-only mounted bucket), and
# verified against the hash in the MODEL_SHA256 env var, which must be set at run time:
#   docker run -v /path/to/store:/models:ro -e MODEL_SHA256=<sha256> -p 8080:8080 <image>
# The server keys the validation-set cache on that hash.
ENV MODEL_STORE=/models

# Validation-set audio cache. On Cloud Run /cache is a mounted Cloud Storage bucket (persistent).
ENV VALIDATION_CACHE_DIR=/cache

COPY pocket_tts pocket_tts
COPY training training
COPY deploy deploy

# --device auto picks cuda when the GPU build runs on a GPU, else cpu. 4 torch threads: more
# does not help at batch size 1. The model loads in the background, so the port opens at once
# and /tts answers 503 "warming up..." until it is ready.
CMD ["sh", "-c", "exec python -m deploy.hebrew_tts_server --device auto --host 0.0.0.0 --port ${PORT} --threads 4"]
