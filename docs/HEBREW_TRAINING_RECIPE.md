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

# 6. train. Finetune from the released 24-layer teacher -- Kyutai measured this as
#    ~2.5x faster to the same WER than scratch on 976h of Czech. Add torchrun for multi-GPU.
uv run torchrun --nproc-per-node 8 training/train.py training/configs/finetune_language.yaml
# then distil that into the 6-layer student
uv run torchrun --nproc-per-node 8 training/train.py training/configs/depth_distill.yaml

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
| 5 tokenizer | Kyutai's script. **Never run here** (`--vocab-size` now defaults to 4000, so it needs no override) |
| 6 train | Kyutai's trainer. **Never run here, not even a smoke test.** Every number in the hardware table is theirs, not ours |
| 6b finetune path | Kyutai's `finetune_language.yaml`, added upstream `8c98c9b` and measured by them on Czech. **Never run here.** Supersedes our `warm_start_checkpoint.py` |
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

**Those are English numbers on audiobook speech. Do not expect them for Hebrew.** Kyutai's
own new-language result is 976 h of Czech parliamentary speech settling at **10-12% WER** —
an order of magnitude off the 0.94% English figure, on a corpus very like ours. Judge the
run against 10-12%, not against 1%.

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

By default, the builder transcodes every recording that contributes a manifest row to
24 kHz mono PCM WAV under `data/hebrew_official/audio/<recording_id>.wav`, and manifests
point there instead of at the AAC/MKA source. Conversion runs in the bounded recording
worker pool, writes directly to the output volume rather than the temporary spool drive,
and reuses complete WAVs when a build is restarted. Budget output storage at about 165 MiB
per wall-clock audio hour and set `--workers` for the available CPU and output-disk bandwidth.
Use `--no-transcode-audio` only when retaining source paths is intentional.

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

### How long should a row be? Match their data, not their paper

Three sources disagree, so measure rather than infer:

| source | audio length |
|---|---|
| paper, Table 14 (TTS) | 60 s |
| `max_duration_sec` in the shipped configs | 30 s |
| **HiFiTTS-2, what the released model was actually trained on** | **median 9.16 s, p90 15.4 s, max exactly 20.00 s** |
| Mimi's transformer context (`context: 250` at 12.5 Hz) | 20 s |
| our merged Hebrew rows | median 12.64 s, cap 30 s |

The HiFiTTS-2 row is measured from 21,416 utterances of NVIDIA's own manifest. Their
utterances stop dead at 20.00 s, which is exactly Mimi's context window — so the 30 s config
cap never binds on Kyutai's own data, and the paper's 60 s describes neither.

Our 12.64 s median sits comfortably inside their real distribution, a little longer than
their 9.16 s. That is fine. The tail is the question: rows between 20 and 30 s exceed Mimi's
context window, and no released model was trained on anything that long.

If you want to match their operating point exactly, cap at 20 s in both places:

```bash
python -m hebrew_training.build_official_manifest ... --max-duration 20
```

and set `data.max_duration_sec: 20.0` in the training config. It costs little — the merge
target is 12 s, so only the tail moves — and it keeps every row inside the codec's context.

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

### Recommended: finetune from the released teacher, not from scratch

**Kyutai added an official config for exactly this case** (`finetune_language.yaml`,
upstream `8c98c9b`), and they measured it on a language much closer to ours than English:

> on 976 h of Czech this hit **12% WER by 10k steps where a scratch run was still at 46%**,
> and both settled around 10-12%

Czech parliamentary speech at ~1,000 h is the closest published analogue to Hebrew Knesset
speech at ~4,200 h. Same destination, reached about 2.5x sooner. There is no reason to pay
for the scratch run.

`training/configs/finetune_hebrew.yaml` in this repo is ready to run. It is
`finetune_language.yaml` with the Hebrew paths filled in and Hebrew `sample_sentences`;
**every training parameter is upstream's, verified identical**. Nothing to edit unless your
manifests live elsewhere.

```bash
uv run torchrun --nproc-per-node 8 training/train.py training/configs/finetune_hebrew.yaml
```

The parameters that matter, and none of them should be touched:

| | value | note |
|---|---|---|
| `lr` | 2e-4 | "the text embedding starts random, so the backbone has to move with it" |
| `schedule` | **constant** | not `cosine` — that is the scratch config |
| `max_steps` | 250000 | "WER plateaus by ~15k; the rest is acoustic quality" |
| `batch_size` | 64 | per GPU; on 8 GPUs drop to 8 |
| `start_from_pretrained` | true | |
| `reset_text_embedding` | true | the whole trick, see below |

`reset_text_embedding` is the whole trick, and it is the dictionary problem from above:
`builders.py:170` drops the `conditioner.embed.*` keys before loading, so the text table
starts fresh while all 24 backbone layers transfer. Their note on why `lr` stays at 2e-4:
"the text embedding starts random, so the backbone has to move with it."

Note `max_steps: 250000` with "WER plateaus by ~15k; the rest is acoustic quality" — so a
usable Hebrew voice is a far shorter run than the headline number suggests.

Then distil with `training/configs/depth_distill_hebrew.yaml`, also generated from
upstream's and also parameter-identical. The one thing to check before launching it is that
`distill_teacher_weights` names a checkpoint `runs/finetune_hebrew/` actually produced — the
default assumes the full 250k steps.

#### Our `warm_start_checkpoint.py` is superseded — mostly

It predates upstream `8c98c9b` and solved the same problem offline. Use
`reset_text_embedding: true` instead: it is supported, simpler, and it is what Kyutai
measured.

The one thing our script still does that theirs does not: Kyutai **discard** the text table
outright, while ours transplants rows matched on the piece string, so punctuation, digits and
spaces keep their learned embeddings. That is a small, unmeasured edge over a validated
default. Take the validated default.

### Alternative: train the teacher from scratch

Only if you have a reason not to finetune. Kyutai's note on the two-stage shape still
applies: training in two steps "works better than training a 6-layer model from scratch."

**Stage 1 — the 24-layer teacher.** In `training/configs/scratch.yaml`:

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
uv run training/train.py training/configs/scratch.yaml
```

**Stage 2 — distil to the 6-layer student**, baking CFG in. In `depth_distill.yaml`:

```yaml
model_config: pocket_tts/config/hebrew.yaml           # the 6-layer student
distill_teacher_config: pocket_tts/config/hebrew.yaml # same file...
distill_teacher_overrides:
  flow_lm.transformer.num_layers: 24                  # ...plus this, making it the teacher
distill_teacher_weights: runs/finetune_language/checkpoint_00250000.pt  # <- your teacher run
data:
  train_jsonl: data/hebrew_official/train_aligned.jsonl
  valid_jsonl: data/hebrew_official/valid_aligned.jsonl
```

```bash
uv run training/train.py training/configs/depth_distill.yaml
```

### What distillation actually does

Kyutai document this in one line (`training/README.md:155`); the rest is only in the code.
It does two separate things at once.

**1. It shrinks 24 layers to 6.** The teacher is not a second model you find somewhere — it
is Stage 1's own checkpoint. The student starts seeded from it, not randomly: `shrink()`
copies every non-backbone tensor verbatim plus the teacher's bottom and top layers. The flow
head and the EOS head are then **frozen at the teacher's weights and never trained**. Only
the backbone learns, against plain MSE against the teacher's backbone output.

**2. It bakes in classifier-free guidance.** The regression target is not the teacher's
normal output. Each step runs the teacher twice — once fully conditioned, once with
conditioning nulled — and combines them:

```
z_t = z_null + distill_cfg_coef * (z_cond - z_null)
```

The student learns to hit that guided point directly. So at inference it needs **one**
forward pass where the teacher needs two, at the quality of guided sampling.

### The process, concretely

1. **Finish the teacher run.** Point `distill_teacher_weights` at its checkpoint —
   `runs/finetune_language/checkpoint_*.pt` if you took the recommended path, or
   `runs/scratch/checkpoint_00400000.pt` if you trained from scratch. Nothing about
   distillation can start before the teacher exists.
2. **No new data work.** Same manifests, same tokenizer, same audio. Only the config changes.
3. **Edit the five lines above** in `depth_distill.yaml` — the two config paths, the
   override, the teacher checkpoint, and your manifests.
4. **Leave `distill_cfg_coef: 2.0` alone.** At 0 nothing distils; 1.0 would be pure depth
   distillation with no guidance baked in.
5. **Leave `text_dropout: 0.0` and `voice_dropout: 0.0`.** Not a typo — the teacher's targets
   are always fully conditioned, so dropping the student's conditioning would ask it to
   predict a conditioned target from a null input. The trainer warns if you set them.
6. **Run it.** `uv run training/train.py training/configs/depth_distill.yaml`
7. **Sample the student with `--cfg 1`.** Guidance is already in the weights; sampling at
   cfg 2 applies it twice. (`scratch.yaml` sets `sample_cfg_coef: 2.0` because a *teacher*
   must be sampled guided — the student's default is 1.0.)

Notes worth knowing before it fails on you:

- **Both models sit in VRAM.** The frozen 24-layer teacher runs two forward passes per step
  alongside the student. Budget accordingly; the shipped `batch_size: 16` assumes this.
- `distill_teacher_use_ema` defaults to **true**, so the target is the teacher's EMA shadow,
  falling back to raw weights if the checkpoint has none. Leave it.
- WER reaches parity by ~50k steps; speaker similarity and UTMOS climb until ~150k. The
  config's 200k is there because prosody keeps settling. If you only need a working voice,
  stop early.
- Cost: ~3 h on 8x H100, proportionally more on fewer.
- Upstream `65534c9` changed this config: `batch_size` is now **64** in one pass (was 16),
  and `flow_batch_multiplier` was dropped because the flow loss never runs in distill mode.
  Do not re-add it. Parity is quoted as ~50k steps, of the configured 200k.

### What not to touch

`lr: 2e-4`, `flow_batch_multiplier: 4`, `text_dropout: 0.2`, `voice_dropout: 0.2`, and
`flow.type: lsd`. These are the settings Kyutai call decisive, and all four were wrong in
the earlier attempt (see the table at the top). Do not port anything from
`hebrew_training/train.py` onto these configs — that trainer was our reimplementation and it
is superseded.

`text_dropout` and `voice_dropout` are correctly **0.0** in the distillation config, not a
typo: the teacher's targets are computed fully-conditioned, so dropping conditioning there
would ask the student to predict a conditioned target from a null input.

### Multiple GPUs — already handled, do not write anything

DDP is implemented upstream. `train.py:140` wraps the model in
`DistributedDataParallel` whenever it detects torchrun, `distributed.py` initialises the
nccl process group from `LOCAL_RANK`, and `load_entries` shards the manifest across ranks by
line index (`idx % world_size != rank`). Nothing needs porting or patching.

```bash
# one GPU
uv run training/train.py training/configs/scratch.yaml

# eight GPUs
uv run torchrun --nproc-per-node 8 training/train.py training/configs/scratch.yaml
```

The same applies to the distillation stage — swap the config.

Two consequences to keep in mind:

- **`batch_size` is per GPU**, so adding GPUs multiplies the effective batch. On 8 GPUs set
  `batch_size: 8` to land on 64, not 64.
- Each rank reads a different slice of the manifest, and the loader asserts its slice is
  non-empty. Not a concern at ~136k rows, but it is why a tiny debug manifest fails on 8 GPUs
  and works on 1.

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

### Score every checkpoint as it lands, and watch the curve

Start this alongside the training run. It scores each new checkpoint from step 4000 on five
fixed Hebrew sentences and appends WER/CER to `hebrew_eval.jsonl` in the run directory:

```bash
python -m hebrew_training.watch_eval --run-dir runs/finetune_hebrew --watch
```

It writes `hebrew_eval.html` beside the results and **rewrites it after every checkpoint**,
so a browser left open on the page is never more than one checkpoint behind. Serve the run
directory once and leave it:

```bash
python -m http.server 8000 --directory runs/finetune_hebrew   # open hebrew_eval.html
```

The chart is WER and CER against step with the best checkpoint circled, above a table with
every clip playable beside its step and the ASR's transcription under it — so a number that
looks good can be checked by ear immediately.

The voice the model clones is cut automatically from the run's own `valid_jsonl`: the first
row longer than `--voice-sec` (5 s), cached at `hebrew_eval/voice_prompt.wav` so every
checkpoint is judged against the same voice. Delete that file to re-pick, or pass `--voice`
for a specific clip. It comes from the validation split, so it is a voice the model was not
trained on.

This is a second process, with its own GPU memory for the model plus Whisper — start it
alongside training, not inside it. It is resumable and skips checkpoints already scored, so
it can be started midway through a run or restarted after a crash. `build_wer_dashboard.py`
renders the page standalone if you want it without the watcher, and `--run-dir` repeats to
overlay runs. For the distilled student add `--cfg 1.0`, since guidance is baked into its
weights.

**Five sentences is a progress signal, not a measurement.** Our earlier 12-clip scores swung
0.1-0.3 between adjacent checkpoints on sampling noise alone. Use it to see the shape of the
curve and to catch a run going wrong early; confirm the winning checkpoint with
`score_wer.py` over a few hundred clips before shipping it.

### The trainer computes no WER — score it yourself

Nothing in `training/train.py` touches WER, CER or an ASR. It writes a validation loss and,
every `sample_freq` steps, a handful of wavs. Scoring is entirely a separate pass with
`score_wer.py`, run against the finished run directory. On this project loss ranked
checkpoints differently from what the audio sounded like three separate times, so the
validation curve is not a substitute.

The Hebrew configs set `sample_freq: 2500` (upstream default is 10000, which on a run that
plateaus by 15k gives samples at 10k and 20k only) and `ckpt_freq: 2500`.

`num_ckpt_keep` is the one to watch. Upstream keeps **3**, which at `ckpt_freq: 2500` is the
last 7500 steps of a 250k run — the checkpoint worth keeping is deleted long before the end.
On the earlier Hebrew run the best-sounding checkpoint was step 8000. `finetune_hebrew.yaml`
sets 40, covering the first 100k steps, at roughly 2.5 GB each. Lower it if disk is tight;
never back to 3.

### First, measure the ruler

Do this **before** training, on genuine held-out human Hebrew audio:

```bash
python -m hebrew_training.score_asr_floor --eval-set eval/hebrew.json --output eval/floor.json
```

This runs `ivrit-ai/whisper-large-v3-turbo-ct2` against real recordings with known
transcripts. Whatever WER it reports is error the ASR makes on *correct* speech — a floor
your TTS can approach but never beat, because a perfect model still gets marked wrong
wherever the ASR mishears.

It matters because the headline numbers are not comparable. Kyutai's 0.94% is Granite ASR on
LibriSpeech test-clean: clean, read, English audiobooks, the easiest ASR benchmark there is.
Their Czech 10-12% is Whisper on parliamentary speech. They say so themselves:

> When training on a new language, not all of these metrics transfer directly: Word error
> rate: you need an ASR that supports your language.

So an unknown share of that 10-12% is the ASR misreading good audio, not the TTS speaking
badly. **We have never run this script and have no Hebrew floor number.** Without it you
cannot tell a bad model from a bad ruler. Judge the TTS against `floor + delta`, not against
zero, and expect Knesset-style speech to score worse than audiobooks whatever the model does.

### Set `sample_sentences` to Hebrew before you start training

`args.py:101` defaults them to English:

```
"The quick brown fox jumps over the lazy dog."
```

The trainer synthesizes exactly these every `sample_freq` steps. Leave them and every sample
your friend listens to for the whole run is English text pushed through a Hebrew tokenizer —
unusable for judging progress, and unscoreable. Add a Hebrew list to the training config:

```yaml
sample_sentences:
  - "אדוני היושב ראש, אני מבקש להעלות את הנושא לסדר היום."
  - "חברי הכנסת הנכבדים, מדובר בהחלטה משמעותית."
```

`score_wer.py` refuses to score a run whose `sample_sentences` are not Hebrew rather than
reporting a number that looks real.

### Then score the checkpoints

```bash
python -m hebrew_training.score_wer --runs-dir runs
```

Uses `ivrit-ai/whisper-large-v3-turbo-ct2`. Kyutai's own reference eval runs
`--temp 0.3 --cfg 2.0 --n-steps 1 --eos-threshold -1`, and note that `flow_matching` configs
need `--n-steps >= 16`.

`score_wer.py` reads both sample layouts: Kyutai's flat
`runs/<run>/samples/step00010000_<i>.wav` (recovering the text from `sample_sentences` in the
run's `args.yaml`, which is why the previous section matters) and our older
`runs/<run>/samples/step<N>/samples.json`.

Use a fixed eval set of at least a few hundred clips. Our early 12-clip scores swung 0.1–0.3
between adjacent checkpoints on sampling noise alone — and the trainer's own three sample
sentences are far below that, so treat them as a listening check, not a measurement.

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
