# CPU image for deploy/hebrew_tts_server.py. COPY paths are relative to the repo root:
#   docker build -f deploy/Dockerfile .          (or use deploy/deploy.sh for Cloud Run)
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

# Weights and tokenizer first: they are large and change rarely, so this layer stays cached.
COPY artifacts/hebrew_24l.yaml artifacts/hebrew_24l.yaml
COPY artifacts/model.safetensors artifacts/model.safetensors
COPY tokenizers/tokenizer.model tokenizers/tokenizer.model

# Weights hash, baked in and checked against the file at build time. The server keys the
# validation-set cache on it. Empty (a plain `docker build`) = the server hashes the file itself.
ARG MODEL_SHA256=
RUN if [ -n "$MODEL_SHA256" ]; then \
      echo "$MODEL_SHA256  artifacts/model.safetensors" | sha256sum -c -; fi
ENV MODEL_SHA256=$MODEL_SHA256

# Validation-set audio cache. On Cloud Run /cache is a mounted Cloud Storage bucket (persistent).
ENV VALIDATION_CACHE_DIR=/cache

COPY pocket_tts pocket_tts
COPY training training
COPY deploy deploy

# --device auto picks cuda when the GPU build runs on a GPU, else cpu. 4 torch threads: more
# does not help at batch size 1. The model loads in the background, so the port opens at once
# and /tts answers 503 "warming up..." until it is ready.
CMD ["sh", "-c", "exec python -m deploy.hebrew_tts_server --device auto --host 0.0.0.0 --port ${PORT} --threads 4"]
