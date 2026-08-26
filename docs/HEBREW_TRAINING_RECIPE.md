# Hebrew Pocket TTS — training recipe

End-to-end recipe using Kyutai's released training code (upstream `0140f9c`), not our
earlier reimplementation. Written after the earlier 39-hour attempt reached ~94% WER; the
diagnosis and the numbers behind every choice here are in `../../CHANGES.md`.

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

## Step 1 — corpus

| dataset | hours | sample rate | transcripts | alignment |
|---|---|---|---|---|
| `ivrit-ai/VoxKnesset` | 2,307 | 16 kHz | human-verified | not published |
| CrowdRecital (local) | 50.4 | 48 kHz | human recital | ships `transcript.aligned.json` |
| `ivrit-ai/crowd-transcribe-v5` | ~316 est. | 32/44.1/48 kHz | human + retranscribe pass | not published |

Mimi runs at **24 kHz**. Anything above downsamples cleanly; 16 kHz must be upsampled and
the missing 8–12 kHz band never comes back. VoxKnesset supplies the hours but caps
bandwidth — accept a band-limited-sounding v1, or mix in the higher-rate corpora and accept
fewer hours.

VoxKnesset segments average **125 s** against Kyutai's `max_duration_sec: 30`, so they must
be split into utterances at word boundaries — which requires alignment first.

---

## Step 2 — manifests

Target schema (`training/dataloader.py`):

```json
{"path": "...", "start": 0.0, "duration": 4.31, "transcript": "...",
 "words": [{"word": "...", "start": 0.12, "end": 0.44}]}
```

Word times are **relative to `start`**. `words` is optional but costs you the
word-boundary cutting and the trailing-silence trim if omitted.

For CrowdRecital, which already has timings:

```bash
python -m hebrew_training.build_official_manifest \
    --corpus /path/to/crowd-recital \
    --out-dir data/hebrew_official \
    --normalize-text --valid-hours 2.0
```

Validated: 80/80 utterances sampled through `training.dataloader.DataLoader` cleanly.

---

## Step 3 — alignment, for anything without timings

```bash
PYTHONPATH=/path/to/pocket-tts python -m data_prep.align_hebrew \
    manifest.jsonl manifest_aligned.jsonl \
    --model imvladikon/wav2vec2-xls-r-300m-hebrew --device cuda --batch-size 8
```

Measured against CrowdRecital's timings: word **ends** within 19 ms median (a quarter of a
12.5 Hz frame), starts within 74 ms, 5.7% of words untimed. Ends are the accurate side and
are what the silence trim uses.

Budget roughly 31 GPU-hours for 2,307 h at ~75x realtime, about $8 on L4 spot.

---

## Step 4 — tokenizer

```bash
uv run training/scripts/train_tokenizer.py   # see its --help for corpus flags
```

Retrain on the final Hebrew text. A Hebrew vocabulary means the text-conditioning path is
learned from scratch even when starting from released weights, so this is not optional.

---

## Step 5 — train

Two stages, as Kyutai ship them. Their note: training in two steps "works better than
training a 6-layer model from scratch."

```bash
# Stage 1: 24-layer teacher
uv run training/train.py training/configs/lsd_scratch.yaml

# Stage 2: distil to the 6-layer student, baking in CFG
uv run training/train.py training/configs/lsd_depth_distill.yaml
```

Edit in `lsd_scratch.yaml`:

- `data.train_jsonl` / `valid_jsonl` — your manifests
- `batch_size` / `grad_accum_steps` — **effective batch must reach 64**. One GPU in a single
  pass wants ~56 GiB; a consumer card runs `batch_size: 16` with `grad_accum_steps: 4` in
  ~16 GiB.
- `max_steps` — 50k if you only need intelligibility; 400k for the published quality.

Leave `lr: 2e-4`, `flow_batch_multiplier: 4`, `text_dropout: 0.2`, `voice_dropout: 0.2` and
the `lsd` flow type alone. Those are the decisive settings.

`start_from_pretrained` defaults to **true** in `args.py` but both shipped configs set it
false. From scratch is right at 1000+ h; at 39 h it was not.

### Cost

| stage | steps | L4 GPU-hours | on-demand | spot |
|---|---|---|---|---|
| teacher | 200k | 158 | $133 | $40 |
| distil | 200k | ~53 | $45 | $13 |

50k teacher steps — enough for intelligibility — is ~40 h, about $33 on-demand or $10 spot.
On GCP the L4 is `g2-standard-8`; H100 is only sold as an 8-GPU node at $87.83/hr.

---

## Step 6 — evaluate

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
- **Latents are model-specific.** `emb_mean` differs between models by a median factor of
  730x, so latents cached for one config are meaningless to another.
- `pip install sphn transformers soundfile` — the training and alignment code needs these
  and an inference-only venv lacks them.
- Kyutai's diagnostic list in `training/README.md` maps symptoms to causes: a TTS that
  ignores the transcript means inaccurate transcripts; cut-off first or last words mean bad
  alignment. Read it before debugging.
