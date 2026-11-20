#!/usr/bin/env bash
# Run build_numeric_speech_manifest on every GPU, PROCS_PER_GPU processes each, then merge.
#
# The model generates one sample at a time (batch size 1) and is latency-bound, so one process
# leaves a GPU mostly idle; several processes per GPU is the batching. Each process is a shard.
#
#   OUT_DIR=/tmp/numeric-canary PAIRS=... PROMPTS=... CONFIG=... CHECKPOINT=... \
#   PROCS_PER_GPU=2 scripts/build_numeric_speech.sh --max-pairs 100
#
# Extra arguments go to every process. GPUS="0 1 2" picks GPUs (default: all).
# A rerun resumes (finished ids are skipped); logs are in $OUT_DIR/logs.
set -euo pipefail

: "${OUT_DIR:?}" "${PAIRS:?}" "${PROMPTS:?}"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
PY="${PY:-python}"

read -r -a GPU_LIST <<< "$GPUS"
TOTAL=$(( ${#GPU_LIST[@]} * PROCS_PER_GPU ))
mkdir -p "$OUT_DIR/logs"

model_args=()
[[ -n "${CONFIG:-}" ]] && model_args+=(--config "$CONFIG")
[[ -n "${CHECKPOINT:-}" ]] && model_args+=(--checkpoint "$CHECKPOINT")

pids=()
shard=0
for gpu in "${GPU_LIST[@]}"; do
  for ((p = 0; p < PROCS_PER_GPU; p++)); do
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m hebrew_training.build_numeric_speech_manifest \
      --pairs "$PAIRS" --prompts "$PROMPTS" --out-dir "$OUT_DIR" "${model_args[@]}" \
      --device cuda --num-shards "$TOTAL" --shard-index "$shard" "$@" \
      > "$OUT_DIR/logs/shard-$shard.log" 2>&1 &
    pids+=($!)
    shard=$((shard + 1))
  done
done
echo "started $TOTAL processes on GPUs: $GPUS (logs: $OUT_DIR/logs)"

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ $status -eq 0 ]] || echo "some shards failed: see $OUT_DIR/logs" >&2

"$PY" -m hebrew_training.build_numeric_speech_manifest --out-dir "$OUT_DIR" --merge
exit $status
