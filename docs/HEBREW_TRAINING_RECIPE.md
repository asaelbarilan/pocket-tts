# Hebrew Pocket TTS — training recipe

End-to-end recipe using Kyutai's released training code (upstream `0140f9c`), not our
earlier reimplementation. Written after the earlier 39-hour attempt reached ~94% WER; the
diagnosis and the numbers behind every choice here are in `../../CHANGES.md`.

---

## The whole pipeline

```bash
# 2. fetch a subset of the corpus (skip if you already have the audio)
python -m hebrew_training.fetch_knesset --out data/knesset_plenums --hours 1200

# 3. corpus -> manifests. Merges Whisper segments into ~12 s utterances; read its output.
PYTHONUTF8=1 python -m hebrew_training.build_official_manifest \
    --corpus data/knesset_plenums --out-dir data/hebrew_official \
    --normalize-text --valid-hours 2.0

# 4. alignment -- NOT needed here, this corpus ships transcript.aligned.json.
#    If needed (a corpus without word timings), the model we tested:
# PYTHONPATH=$(pwd) python -m data_prep.align_hebrew \
#     in.jsonl out.jsonl --model imvladikon/wav2vec2-xls-r-300m-hebrew \
#     --device cuda --batch-size 8

# 5. tokenizer, trained on the manifest text from step 3
uv run training/scripts/train_tokenizer.py tokenizers/hebrew \
    data/hebrew_official/train_aligned.jsonl --vocab-size 4000

# 6. train: 24-layer teacher, then distil to the 6-layer student
uv run training/train.py training/configs/lsd_scratch.yaml
uv run training/train.py training/configs/lsd_depth_distill.yaml

# 7. score by WER, never by loss
python -m hebrew_training.score_wer --runs-dir runs
```

Steps 5 and 6 need config edits, not just the commands — see those sections.

### What has actually been run

Be straight with whoever picks this up:

| step | state |
|---|---|
| 2 fetch | selection verified against the live repo; a full 102 GiB download has not been run |
| 3 manifests | validated end-to-end on CrowdRecital (80/80 utterances through Kyutai's `DataLoader`). The segment-merging path added for the Knesset corpora is **not yet exercised on downloaded audio** |
| 4 alignment | `align_hebrew.py` measured against CrowdRecital timings: 19 ms median word-end error. Not needed for the Knesset corpora |
| 5 tokenizer | Kyutai's script. **Never run here** |
| 6 train | Kyutai's trainer. **Never run here, not even a smoke test.** Every number in the hardware table is theirs, not ours |
| 7 WER | ours, used throughout the earlier run |

The first thing to do on the server is a 100-step run at `max_steps: 100` to prove the
config loads and the loss moves, before committing to a 40-hour job.

---

## Why the earlier attempt failed

Not a bug, and not the stop-token tuning we spent three experiments on. Our reimplemented
trainer had **all four** of the hyperparameters Kyutai call decisive set wrong, and the
corpus was below their stated minimum:

| knob | Kyutai | ours | their note |
|---|---|---|---|
| learning rate | 2e-4 | 2e-5 | "1e-4 never gets there in 400k steps" |
| effective batch | >= 64 | 8 | "below that the quality transition arrives late or not at all" |
| flow_batch_multiplier | 4 | 1 | "1 never completes the transition" |
| t-sampling | lognormal(0.4, 1.0) | uniform | "uniform never gets there" |
| corpus | 1000+ h | 39 h | "100 hours is a minimum" |

Using the official trainer makes the first four correct by construction. **Do not port our
`hebrew_training/train.py` settings onto it.**

---

## Step 0 — what "enough data" means

From `training/README.md`:

- 100 h is the floor. 200 h gives "a model that speaks". 1000+ h for a strong model.
- 2000 h reaches WER 0.94% at 400k steps.

Metric milestones from scratch, so you know when to stop:

| milestone | steps |
|---|---|
| WER starts dropping | ~15k |
| **WER ~1%, then flat forever** | **~50k** |
| UTMOS climbing to 3.7 | ~150k |
| prosody settled | 300–400k |

Intelligibility is done by 50k. Everything past that buys expressivity.

---

## Step 1 — corpus: `ivrit-ai/knesset-plenums`

Numbers below are read from the repo's own `manifest.csv` (all 1,551 recordings), not
estimated:

| | |
|---|---|
| wall-clock audio | **8,816 h** |
| inside transcript segments | **5,449 h** |
| format | 44.1 kHz **stereo** AAC, 128 kbps |
| transcripts | Whisper, refined against the official Knesset protocol |
| word alignment | **ships with the data** (`transcript.aligned.json`) |
| recording `quality_score` | median 0.894, p25 0.861, p10 0.582 |

44.1 kHz is the number that matters: Mimi runs at 24 kHz, so this downsamples with nothing
missing. `ivrit-ai/VoxKnesset` is the same source audio resampled to 16 kHz during its
curation **and it publishes no transcripts at all** — nine columns of speaker demographics
plus `audio`. Do not use it. `ivrit-ai/knesset-committees` is the same shape as plenums and
adds ~339 GiB more; it is a straight `--repo` swap once its gate is accepted.

Other datasets considered and rejected: `crowd-transcribe-v5` (~316 h, but HF parquet with
audio as bytes — needs an ingestion script nobody has written), CrowdRecital (50 h, fine,
already works, just small next to 5,449 h).

**Both repos are gated**, one checkbox, auto-approved. Accept, then `hf auth login`:
<https://huggingface.co/datasets/ivrit-ai/knesset-plenums>

---

## Step 2 — fetch a subset

Never download all 396 GiB. Selection reads one 147 KiB `manifest.csv` and picks
recordings before fetching any audio.

```bash
python -m hebrew_training.fetch_knesset --out data/knesset_plenums --hours 1200
```

1,200 speech-hours is 324 recordings, ~1,907 h of audio, **~102 GiB**. Check with
`--dry-run` first. Selection is deterministic and nested — raising `--hours` later only
downloads what is new — and the whole thing is resumable.

`--min-quality` gates on the recording-level `quality_score` (default 0.8, keeping 1,331 of
1,551). Budget in **speech** hours, not wall-clock: plenary audio is only ~62% speech, the
rest is gavel, procedure and dead air.

---

## Step 3 — manifests

Target schema (`training/dataloader.py`):

```json
{"path": "...", "start": 0.0, "duration": 12.6, "transcript": "...",
 "words": [{"word": "...", "start": 0.12, "end": 0.44}]}
```

Word times are **relative to `start`**.

```bash
PYTHONUTF8=1 python -m hebrew_training.build_official_manifest \
    --corpus data/knesset_plenums \
    --out-dir data/hebrew_official \
    --normalize-text --valid-hours 2.0
```

### The one thing that is easy to get wrong here

The Knesset corpora ship Whisper **decoder** segments, not utterances: median 1.84 s, p90
4.40 s, over 702,646 segments measured. The loader's `MIN_CUT_SEC` requires 1.0 s on both
sides of a cut, so **53.7% of raw segments have no eligible word-boundary cut at all**.
Those rows silently take the fallback branch, where the voice prompt is read from
`entry.start` — overlapping the target audio the model is being asked to predict. That is
prompt leakage, and it is the defect that wrecked the first Hebrew run.

So the builder merges consecutive segments into ~12 s utterances, breaking at silences
longer than `--merge-gap` (1.5 s), and drops anything under `--min-duration` (4.0 s). The
defaults are right; the flags exist to be inspected, not tuned. Result: 12.64 s median row,
retaining ~4,200 h of the 5,449.

**Read these two lines of its output before going further:**

```
row duration : median 12.64 s, p10 ..., p90 ..., max 30.00
uncuttable (<2.0 s, prompt would overlap target): 0 (0.00%)
```

A low median or a non-zero uncuttable count means the merge did not work, and training on
that manifest will repeat the last failure with a bigger bill.

For CrowdRecital, whose segments are already utterance-length, pass `--merge-gap 0`.

---

## Step 4 — alignment: not needed for this corpus

`transcript.aligned.json` already carries per-word `start`, `end` and `probability`; 100% of
words in the sampled recordings were timed. Skip straight to Step 5.

**If needed** — a corpus that arrives without word timings — use
`data_prep/align_hebrew.py` from the `hebrew-tts-data-tools` repo with
`imvladikon/wav2vec2-xls-r-300m-hebrew`, which is the model we measured:

```bash
PYTHONPATH=/path/to/pocket-tts python -m data_prep.align_hebrew \
    manifest.jsonl manifest_aligned.jsonl \
    --model imvladikon/wav2vec2-xls-r-300m-hebrew --device cuda --batch-size 8
```

Check a candidate model before spending GPU hours on it — `--check-model` prints its
vocabulary and warns if the CTC head is not Hebrew:

```bash
python -m data_prep.align_hebrew in.jsonl out.jsonl \
    --model imvladikon/wav2vec2-xls-r-300m-hebrew --check-model
```

Measured against CrowdRecital's own timings: word **ends** within 19 ms median (a quarter of
a 12.5 Hz frame), starts within 74 ms, 5.7% of words untimed. Ends are the accurate side and
are what the trailing-silence trim uses. Budget ~31 GPU-hours per 2,300 h of audio.

---

## Step 5 — tokenizer

### Is this actually necessary? No — but do it anyway

The released tokenizer is SentencePiece **BPE**, not character-level. It does not break on
Hebrew: byte fallback covers everything, round-trip decoding is exact, zero UNK. That is why
the earlier CrowdRecital run learned Hebrew at all with an English tokenizer.

What it does instead is silently degenerate to one token per character:

| | chars | tokens | tokens/char | distinct pieces used |
|---|---|---|---|---|
| Hebrew sentence | 127 | 128 | **1.01** | 25 |
| comparable English | 166 | 53 | 0.32 | 43 |

Only 508 of the 4,000 vocabulary entries are reachable by Hebrew text at all; the remaining
87% are English subwords. So you pay 3.2x the text-conditioning sequence length and leave
most of the lookup table dead.

Training a Hebrew tokenizer is a two-minute CPU job that removes both costs. It is cheap
insurance, not a blocker — if something else is broken, this is not it.

```bash
uv run training/scripts/train_tokenizer.py tokenizers/hebrew \
    data/hebrew_official/train_aligned.jsonl --vocab-size 4000
```

Writes `tokenizers/hebrew.model` and `.vocab`. It reads the manifest directly, so run it
**after** Step 3 — the tokenizer must see the normalized text the trainer will feed it, not
the raw transcripts.

This is a subword tokenizer for text conditioning. It is unrelated to the character-level
CTC vocabulary in the aligner; the two are different objects and neither substitutes for the
other.

Then wire it up. `pocket_tts/config/hebrew.yaml` in this repo already does it — copied from
`english.yaml` with two changes:

| field | value | why |
|---|---|---|
| `flow_lm.lookup_table.tokenizer_path` | `tokenizers/hebrew.model` | the file you just built |
| `flow_lm.lookup_table.n_bins` | must equal `--vocab-size` | silent shape mismatch otherwise |

`weights_path` stays pointed at the released English model on purpose. `training/args.py`
uses it for the **Mimi codec always**, and for the FlowLM only when `start_from_pretrained`
is true — and both training configs set that false. So Mimi is borrowed (it is a
language-agnostic audio codec) and the FlowLM is fresh. That is the intended arrangement.

---

## Step 6 — train

Two stages. Kyutai's note: training in two steps "works better than training a 6-layer model
from scratch."

**Stage 1 — the 24-layer teacher.** In `training/configs/lsd_scratch.yaml`:

```yaml
model_config: pocket_tts/config/hebrew.yaml     # was english.yaml
model_overrides:
  flow_lm.transformer.num_layers: 24            # leave as-is: this is what makes it the teacher
data:
  train_jsonl: data/hebrew_official/train_aligned.jsonl
  valid_jsonl: data/hebrew_official/valid_aligned.jsonl
batch_size: 64                                  # see the batch note below
grad_accum_steps: 1
max_steps: 400000                               # 50000 if you only need intelligibility
```

```bash
uv run training/train.py training/configs/lsd_scratch.yaml
```

**Stage 2 — distil to the 6-layer student**, baking CFG in. In `lsd_depth_distill.yaml`:

```yaml
model_config: pocket_tts/config/hebrew.yaml           # the 6-layer student
distill_teacher_config: pocket_tts/config/hebrew.yaml # same file...
distill_teacher_overrides:
  flow_lm.transformer.num_layers: 24                  # ...plus this, making it the teacher
distill_teacher_weights: runs/lsd_scratch/checkpoint_00400000.pt   # <- Stage 1's output
data:
  train_jsonl: data/hebrew_official/train_aligned.jsonl
  valid_jsonl: data/hebrew_official/valid_aligned.jsonl
```

```bash
uv run training/train.py training/configs/lsd_depth_distill.yaml
```

The teacher is not a second model you find somewhere — it is Stage 1's own checkpoint. The
student has the same `d_model`, so the flow head and every non-backbone weight are copied
from it and the head stays frozen; only the 6-layer backbone is trained, to reproduce the
24-layer backbone's activations.

### What not to touch

`lr: 2e-4`, `flow_batch_multiplier: 4`, `text_dropout: 0.2`, `voice_dropout: 0.2`, and
`flow.type: lsd`. These are the settings Kyutai call decisive, and all four were wrong in
the earlier attempt (see the table at the top). Do not port anything from
`hebrew_training/train.py` onto these configs — that trainer was our reimplementation and it
is superseded.

`text_dropout` and `voice_dropout` are correctly **0.0** in the distillation config, not a
typo: the teacher's targets are computed fully-conditioned, so dropping conditioning there
would ask the student to predict a conditioned target from a null input.

### Effective batch

**`batch_size` x GPUs x `grad_accum_steps` must reach 64.** Below that, per Kyutai, "the
quality transition arrives late or not at all" — our earlier run sat at 8. One GPU in a
single pass wants ~56 GiB; halve `batch_size` and double `grad_accum_steps` until it fits.
A consumer card runs 16 x 4 in ~16 GiB. On 8 GPUs use `batch_size: 8`.

### Hardware and cost

Kyutai trained the released models on **8x H100**. Their measured scaling, with our
50k-step column derived from the same rates (50k is where WER flattens; 200k is where
acoustic quality lifts):

| GPUs | steps/s | VRAM/GPU | to 50k | to 200k | GPU-hours for 200k |
|---|---|---|---|---|---|
| 1x L4 23GB | 0.35 | 15.9 GiB | 40 h | 158 h | 158 |
| 1x L40S 46GB | 0.77 | 42.0 GiB | 18 h | 72 h | 72 |
| 1x H100 80GB | 1.36 | 55.6 GiB | 10 h | 41 h | 41 |
| 2x H100 | 2.24 | 32.6 GiB | 6 h | 25 h | 50 |
| 4x H100 | 3.94 | 20.0 GiB | 3.5 h | 14 h | 56 |
| 8x H100 | 6.20 | 14.9 GiB | 2.5 h | 9 h | 72 |

**Adding GPUs costs more, not less.** The right-hand column is what you pay for: 8 GPUs
finish 4.6x sooner than one but consume 1.76x the GPU-hours, because the per-GPU batch
shrinks. Kyutai say the same: "Scaling falls off because the per-GPU batch shrinks, not
because of communication." Buy parallelism to save wall-clock, never to save money.

Per-GPU-hour rates, checked August 2026:

| provider | H100 80GB | notes |
|---|---|---|
| Vast.ai | from $1.49 | marketplace, variable reliability |
| RunPod | $1.99 PCIe / $2.69 SXM | |
| Lambda | $3.99 | |
| GCP | $10.98 | `a3-highgpu-8g` only, $87.83/hr for the whole 8-GPU node |

Total cost at RunPod SXM ($2.69):

| target | 1x H100 | 2x H100 | 4x H100 | 8x H100 |
|---|---|---|---|---|
| 50k steps (intelligible) | $27 / 10 h | $32 / 6 h | $38 / 3.5 h | $54 / 2.5 h |
| 200k steps (quality lift) | $110 / 41 h | $135 / 25 h | $151 / 14 h | $194 / 9 h |

Distillation adds roughly 3 h on 8x H100, proportionally more on fewer.

**For a half-day budget: 50k steps on 1-2 H100s, about $30.** If you want the full 200k
acoustic-quality run inside half a day, that is 4-8 H100s at $150-195.

GCP is 4-7x the price of the specialist providers here and forces an 8-GPU node, so it is
the wrong venue for this unless you are already committed to it.

---

## Step 7 — evaluate

Score by **WER**, not loss. On this project validation loss, EOS loss and clip duration each
ranked checkpoints differently from what the audio actually sounded like; EOS loss turned
out to be training step in disguise (partial correlation with WER collapses from −0.82 to
+0.10 once step is controlled for).

```bash
python -m hebrew_training.score_wer --runs-dir runs
```

Uses `ivrit-ai/whisper-large-v3-turbo-ct2`. Kyutai's own reference eval runs
`--temp 0.3 --cfg 2.0 --n-steps 1 --eos-threshold -1`, and note that `flow_matching` configs
need `--n-steps >= 16`.

Use a fixed eval set of at least a few hundred clips. Our early 12-clip scores swung 0.1–0.3
between adjacent checkpoints on sampling noise alone.

---

## Gotchas

- **Windows**: `load_entries` opens manifests without an encoding, so cp1252 kills any
  Hebrew manifest. Set `PYTHONUTF8=1`. One-line upstream fix; they accept bugfix PRs.
  Not an issue on a Linux server, which is where this belongs anyway.
- **m4a decoding.** The loader reads audio through `sphn`, which seeks into whatever file a
  manifest names — no transcode step, the same way Kyutai point HiFiTTS-2 manifests at whole
  chapter mp3s. Confirm `sphn.read` opens one `audio.m4a` before building 1,200 h of
  manifests against them; if it cannot, transcode to wav/flac and re-run step 3.
- **Latents are model-specific.** `emb_mean` differs between models by a median factor of
  730x, so latents cached for one config are meaningless to another.
- `pip install sphn transformers soundfile` — the training and alignment code needs these
  and an inference-only venv lacks them.
- Kyutai's diagnostic list in `training/README.md` maps symptoms to causes: a TTS that
  ignores the transcript means inaccurate transcripts; cut-off first or last words mean bad
  alignment. Read it before debugging.
