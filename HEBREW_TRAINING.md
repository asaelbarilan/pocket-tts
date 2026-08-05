# Experimental Hebrew training

This repository extends Kyutai Pocket TTS with a locally reconstructed Hebrew training
pipeline. Kyutai publishes inference code and weights, but its complete production training
pipeline has not been released.

The reconstruction currently supports:

- a frozen Mimi codec and trainable FlowLM;
- a 4,000-piece Hebrew SentencePiece tokenizer;
- cross-clip, same-speaker voice prompts;
- deterministic speaker-disjoint train/validation manifests;
- flow matching plus EOS supervision;
- an optional 75/25 flow-matching/Lagrangian Self-Distillation (FM/LSD) objective;
- machine-readable training and validation metrics;
- 6-layer and 24-layer Pocket TTS model configurations.

This is experimental research code, not a claim of reproducing Kyutai's training recipe.
Section 4.7 teacher-to-student latent-CFG distillation is not implemented. Always evaluate
generated speech with fixed listening samples and Hebrew CER/WER; loss alone is insufficient.

## Current dataset experiment

The corrected local experiment uses 19,715 clips (38.90 hours) prepared from CrowdRecital:

- training: 19,084 clips, 37.679 hours, 60 speakers;
- validation: 628 clips, 1.211 hours, 20 held-out speakers;
- target clips are paired with a different prompt clip from the same speaker;
- the prepared duration target is approximately 8 seconds, with a 4-16 second envelope.

The dataset, cached latents, checkpoints, logs, and evaluation manifests are intentionally
excluded from Git. Users must supply speech data for which they have the necessary rights.

## Environment

Use Python 3.11 or 3.12 and a CUDA build of PyTorch 2.5 or newer for training:

```powershell
git clone https://github.com/asaelbarilan/pocket-tts-hebrew.git
cd pocket-tts-hebrew
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
python -m pip install pyyaml pyarrow
```

Confirm CUDA before training:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Before latent precomputation, accept Kyutai's model terms on Hugging Face and authenticate:

```powershell
.\.venv\Scripts\hf.exe auth login
```

## Corrected manifest contract

Prepare segmented and normalized audio first. The Pocket TTS converter expects a prepared
dataset directory and preserves `entry_id` and `user_id`. It splits by `user_id`, never by
recording, and writes an explicit `prompt_audio_path` for a different clip from the same
speaker:

```powershell
python -m hebrew_training.prepare_data_v2 `
  --dataset C:\path\to\prepared_hebrew_dataset `
  --output-dir artifacts\hebrew_v2_8s
```

Train the tokenizer and precompute target/prompt latents:

```powershell
python -m hebrew_training.train_tokenizer `
  --input artifacts\hebrew_v2_8s\tokenizer_corpus.txt `
  --output-dir artifacts\hebrew_v2_8s

python -m hebrew_training.precompute_latents `
  --manifest artifacts\hebrew_v2_8s\train.jsonl `
  --output-dir artifacts\hebrew_v2_8s\latents\train `
  --output-manifest artifacts\hebrew_v2_8s\train_latents.jsonl `
  --base-language english `
  --device cuda

python -m hebrew_training.precompute_latents `
  --manifest artifacts\hebrew_v2_8s\validation.jsonl `
  --output-dir artifacts\hebrew_v2_8s\latents\validation `
  --output-manifest artifacts\hebrew_v2_8s\validation_latents.jsonl `
  --base-language english `
  --device cuda
```

Cached latents are model-family-specific. A 24-layer French/German/Italian/Portuguese/Spanish
experiment must recompute latents using the matching `--base-language` in a separate artifact
directory. The trainer rejects known mismatches.

## Smoke test first

Run a very short training probe before a full experiment:

```powershell
python -m hebrew_training.train `
  --train-manifest artifacts\hebrew_v2_8s\train_latents.jsonl `
  --validation-manifest artifacts\hebrew_v2_8s\validation_latents.jsonl `
  --tokenizer artifacts\hebrew_v2_8s\tokenizer.model `
  --run-dir runs\hebrew-v2-smoke `
  --steps 30 `
  --gradient-accumulation 1 `
  --limit-train-samples 16 `
  --save-every 30 `
  --eval-every 10 `
  --eval-samples 8 `
  --device cuda
```

## Controlled evaluation and next training run

The fixed evaluation manifest has four equal groups: seen/unseen speakers crossed with
seen/unseen text. Generation seeds, prompts, and sentences are fixed across checkpoints.
The evaluator reports normalized Hebrew WER/CER with clip-bootstrap confidence intervals.
On 128 genuine held-out recordings, the ASR floor is WER 0.1231 (95% CI 0.1065-0.1411)
and CER 0.0698 (0.0559-0.0839). Generated-model error must be interpreted above this floor.

Build it with preserved prompt audio from both artifact sets:

```powershell
python -m hebrew_training.build_eval_set `
  --artifacts artifacts\hebrew_v2_8s artifacts\hebrew_v2_23s `
  --output runs\evaluation\hebrew-v3-controlled.json `
  --sentences-per-group 8 `
  --speakers-per-group 4 `
  --asr-floor-clips 128
```

The recommended next command is the controlled three-arm ablation, not another blind
20,000-step run. It starts every arm from the identical released English 6-layer model,
trains flow/m1, flow/m8, and FM-LSD/m8 for 6,000 optimizer steps each, then scores all three
final checkpoints on the controlled set:

```powershell
.\run_hebrew_v3_ablation.ps1
```

For a quick end-to-end check without saving a 1 GB checkpoint:

```powershell
.\run_hebrew_v3_fmlsd_training.ps1 -Smoke
```

After the ablation supports FM/LSD, the isolated longer launcher is:

```powershell
.\run_hebrew_v3_fmlsd_training.ps1
```

Its defaults are 12,000 steps, head multiplier 8, FM/LSD 75/25, validation on 64 fixed
latent examples every 250 steps, and checkpoints every 3,000 steps (about 4 GB total).
It writes `runs/hebrew-v3-8s-fmlsd-m8` and never reuses the completed v2 directory.

Use `--init-flow-checkpoint`/`-InitFlowCheckpoint` to change objectives from an existing
flow checkpoint while starting a fresh optimizer. Do not use `--resume` unless the source
checkpoint is already FM/LSD and contains `adaptive_loss_weight.safetensors`.

Checkpoints save FlowLM, optimizer, scheduler, and RNG state. FM/LSD checkpoints additionally
save `adaptive_loss_weight.safetensors`. Metrics are written to `metrics.jsonl` inside the run
directory and console output is written under `logs/` by the launcher.

## Model compatibility

The loss implementation is layer-count agnostic and has completed finite forward/backward
probes on:

- released English 6-layer model: 89,447,489 trainable FlowLM parameters;
- released French 24-layer model: 316,013,633 trainable FlowLM parameters.

That verifies tensor and gradient compatibility, not Hebrew quality. A foreign-language
24-layer starting point still needs its own latent cache, tokenizer-overlap audit, smoke test,
and held-out evaluation.

See [TODO.md](TODO.md) for the next controlled experiments.
