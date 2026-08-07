# Hebrew training on another server

This runbook moves the experiment to a Linux NVIDIA server without copying machine-specific
paths into Git. The repositories contain code and configuration; raw audio, prepared audio,
latent caches, checkpoints, and Hugging Face credentials must be transferred separately.

## What is ready

- 6-layer Hebrew student fine-tuning from `english`.
- Experimental 24-layer Hebrew teacher fine-tuning from a released multilingual base such as
  `french_24l`, with a separate model-specific latent cache.
- Stable flow-matching training with head batch multiplier 8, smoke tests, checkpoints,
  validation metrics, and logs.

The released English model is already a 6-layer student. Kyutai has not published its
24-layer English teacher. Starting from `french_24l` is therefore an experiment, not the
original teacher recipe. Section 4.7 latent-CFG teacher-to-student distillation is not
implemented yet; train and evaluate a useful 24-layer Hebrew model before adding it.

## 1. Clone and prepare the data

```bash
git clone https://github.com/asaelbarilan/hebrew-tts-data-tools.git
git clone https://github.com/asaelbarilan/pocket-tts.git

cd hebrew-tts-data-tools
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export CROWD_RECITAL_ROOT=/data/crowd-recital
export POCKET_TTS_PREPARED_DIR=/data/prepared/CrowdRecital_pockettts_8s
python data_prep/prepare_ivritai.py \
  --config configs/prepare_pocket_tts_hebrew.yaml \
  --max_source_entries 50 --source_selection spread
```

Listen to the pilot and inspect duration, quality, Latin-letter, and normalization statistics.
Then remove `--max_source_entries 50 --source_selection spread` and run the full preparation.
See the preprocessing repository README for resume and teacher-length profile instructions.

For the teacher-length dataset, use a different output path and profile:

```bash
export POCKET_TTS_PREPARED_DIR=/data/prepared/CrowdRecital_pockettts_teacher_23s
python data_prep/prepare_ivritai.py \
  --config configs/prepare_pocket_tts_hebrew_teacher.yaml \
  --max_source_entries 50 --source_selection spread
# Inspect the pilot, then rerun the profile without --max_source_entries.
```

## 2. Install Pocket TTS with CUDA

Choose the PyTorch CUDA wheel compatible with the server driver. For a CUDA 12.8 setup:

```bash
cd ../pocket-tts
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup_training_env.sh
source .venv/bin/activate
huggingface-cli login
```

Accept any required Kyutai model terms on Hugging Face before downloading weights. A 24-layer
training run has 316M trainable FlowLM parameters and has not been memory-benchmarked here;
use a server GPU with at least 16 GB VRAM as a conservative starting point and confirm with a
smoke test. The local 8 GB RTX 4060 is the validated target for the 6-layer run only.

## 3. Build leakage-safe Pocket manifests and tokenizer

```bash
python -m hebrew_training.prepare_data_v2 \
  --dataset /data/prepared/CrowdRecital_pockettts_8s \
  --output-dir artifacts/hebrew_server_8s

python -m hebrew_training.train_tokenizer \
  --input artifacts/hebrew_server_8s/tokenizer_corpus.txt \
  --output-dir artifacts/hebrew_server_8s
```

The converter splits on `user_id` and pairs every target with a different clip from the same
speaker. Review its printed speaker-overlap and duration checks before continuing.

Prepare the teacher-length artifacts separately; never overwrite the student artifacts:

```bash
python -m hebrew_training.prepare_data_v2 \
  --dataset /data/prepared/CrowdRecital_pockettts_teacher_23s \
  --output-dir artifacts/hebrew_teacher_server_23s \
  --max-seconds 40

python -m hebrew_training.train_tokenizer \
  --input artifacts/hebrew_teacher_server_23s/tokenizer_corpus.txt \
  --output-dir artifacts/hebrew_teacher_server_23s
```

## 4. Student: precompute, smoke, train

Commands are dry-run by default. Add `--execute` only after inspecting them.

```bash
python -m hebrew_training.server_launcher doctor \
  --kind student --base-language english \
  --artifacts-dir artifacts/hebrew_server_8s

python -m hebrew_training.server_launcher precompute \
  --base-language english --artifacts-dir artifacts/hebrew_server_8s --execute

python -m hebrew_training.server_launcher train \
  --kind student --base-language english \
  --artifacts-dir artifacts/hebrew_server_8s \
  --run-dir runs/hebrew-student-server-smoke --smoke --execute

python -m hebrew_training.server_launcher train \
  --kind student --base-language english \
  --artifacts-dir artifacts/hebrew_server_8s \
  --run-dir runs/hebrew-student-server-v1 --steps 12000 --execute
```

## 5. Teacher: separate latents, smoke, train

Do not reuse English latent manifests. The launcher writes `french_24l` latents and caches to
different paths and the trainer verifies the base-language marker in every row.

```bash
python -m hebrew_training.server_launcher doctor \
  --kind teacher --base-language french_24l \
  --artifacts-dir artifacts/hebrew_teacher_server_23s

python -m hebrew_training.server_launcher precompute \
  --base-language french_24l --artifacts-dir artifacts/hebrew_teacher_server_23s --execute

python -m hebrew_training.server_launcher train \
  --kind teacher --base-language french_24l \
  --artifacts-dir artifacts/hebrew_teacher_server_23s \
  --run-dir runs/hebrew-teacher-24l-smoke --smoke --execute

python -m hebrew_training.server_launcher train \
  --kind teacher --base-language french_24l \
  --artifacts-dir artifacts/hebrew_teacher_server_23s \
  --run-dir runs/hebrew-teacher-24l-v1 --steps 12000 --execute
```

Teacher mode defaults to `head_samples=500`; student mode defaults to 128. Both default to
flow matching and head batch multiplier 8. The reconstructed FM/LSD objective remains
available in the low-level trainer, but it performed worse in the controlled 6-layer
ablation and is intentionally not the portable launcher's default.

## Outputs and recovery

- Console logs: `logs/<run-name>.log`
- Curves: `runs/<run-name>/metrics.jsonl`
- Checkpoints: `runs/<run-name>/checkpoint-*`
- Arguments: `runs/<run-name>/arguments.json`

Each fresh experiment must use a new run directory. Transfer artifacts and checkpoints with
rsync or object storage; Git intentionally ignores them. Check free disk before a full latent
cache or teacher run. After training, compare fixed held-out Hebrew WER/CER, speaker groups,
and listening samples rather than selecting only by validation loss.
