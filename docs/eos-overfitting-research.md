# Why the EOS head overfits, and what to do about it

**NO CODE CHANGED.** This is research and a ranked recommendation only.

Papers downloaded to `papers/tts-overfitting/`. The Pocket TTS paper is at
`docs/2025_kyutai_pocket-tts.pdf`.

## The measurement we are explaining

Run `hebrew-v2-8s`, 20,000 steps. The two loss terms went in opposite directions:

| step | flow (audio quality) | eos (when to stop) |
|---|---|---|
| 6,000 | 0.5439 | 0.1279 |
| 9,000 | 0.5269 | 0.1418 |
| 14,000 | 0.5278 | 0.2465 |
| 20,000 | 0.5249 | **0.4413** |

Flow improved to the end. EOS got **3.1x worse**. The user judged checkpoint 8,000 best by
ear, which is the last checkpoint before EOS degrades sharply — so the ear tracks the EOS
curve, not the flow curve.

## First, two hypotheses I killed

**"The paper uses aligned inner-monologue text and we use a plain prefix."** Wrong. Paper
section 5.2: for TTS "the text is fed to the backbone as a prefix with SentencePiece model
with a vocabulary size of 4k". Inner monologue is used for *speech continuation* (5.1), not
TTS. Our conditioning matches theirs.

**"We only trained on short sentences."** Measured and rejected. Generated speech content
holds at 2.6-3.7 s across every checkpoint against an expected 3.3 s. What grows is
*leading silence*, 0.55 s at step 10,000 to 1.33 s at 20,000. The model is not truncating
or rambling.

## What is mine and what is the paper's — read this before trusting the rest

Challenged on my basis for the stop-token framing, I searched the paper text. The result
matters:

- **"EOS" appears 0 times in the paper. So do "stop token", "end of sequence" and "end of
  speech".** The paper never describes how generation terminates.
- The EOS head is real, but it is in the *released Pocket TTS code* (`flow_lm.out_eos`),
  not in the paper.
- The loss numbers above come from **our** `train.py`, which defines
  `flow_loss + 0.1 x eos_loss` itself. The paper's objective has no such term.

So: **the measurement is ours and solid; the "stop-token problem" framing is imported from
other TTS literature (Non-Attentive Tacotron), not from Kyutai.** It is a hypothesis that
fits, not a documented finding about this model.

I also had the page count wrong at first — the PDF is 23 pages, not 15. The appendices
with data and hyperparameters are present, so the numbers below are from the paper.

## What the paper actually specifies (Tables 13-14, Sections D and F)

Text-to-speech column of Table 14, the teacher before distillation:

| | paper TTS teacher | ours (Pocket TTS student) |
|---|---|---|
| model dimension | 1024 | 1024 |
| MLP dimension | 4096 | 4096 |
| heads | 16 | 16 |
| **layers** | **24** | **6** |
| **learning rate** | **1e-4** | **2e-5** |
| parameters | 302M | 89.4M trainable |
| consistency head | 6 layers, MLP 512, SiLU | same |

Pocket TTS (Section F) = that 313M / 24-layer teacher **latent-distilled to 6 layers** with
CFG coefficient α = 1.5, giving 90M parameters plus a 20M VAE. We fine-tune the student.

Speech VAE (Table 13): 24 kHz, 12.5 Hz frame rate, latent dimension 32, trained on 12 s
audio samples with a 10 s transformer context. Our clips run to 14.62 s, past that context.

### Training hyperparameters, theirs vs ours (Table 14, TTS column)

| | paper TTS | ours |
|---|---|---|
| optimizer | AdamW, b1 0.9, b2 0.95 | AdamW, weight decay 0.01 |
| **batch size** | **128** | **8** (1 x grad-accum 8) |
| **audio sample length** | **60 s** | 4-16 s clips, mean 7.1 s |
| **LR schedule** | **cosine** | **cosine**, after 500-step warmup, floor at 10% |
| learning rate | 1e-4 | 2e-5 |
| training steps | 400k | 20k |
| hardware | 8 x H100 | 1 x RTX 4060 8 GB |
| inner monologue | ✗ for TTS | ✗ |

So yes, **the learning rate does decay during training in both** — both use a cosine
schedule. Ours additionally warms up over the first 500 steps and floors at 10% of peak.

Two gaps stand out beyond the learning rate:

1. **Batch 128 vs our effective 8.** A 16x smaller batch means noisier gradients and far
   more update steps per unit of data, which is an overfitting-friendly regime.
2. **60-second training samples vs our 7-second clips.** Their model routinely sees
   long-form audio; ours has never seen an utterance outside 4-16 s. This is the strongest
   support so far for the user's instinct that clip length is implicated — not because the
   sentences are short, but because the *duration distribution is narrow*.

### The objective is not the same loss

- **Paper (Eq. 3):** a continuous-time **consistency** loss with an adaptive weighting
  `w_psi(t)`, trained jointly across backbone, head and weighting. Pocket TTS then adds
  latent distillation from a CFG-guided teacher. There is **no stop/EOS term anywhere**.
- **Ours (`train.py`):** plain flow-matching MSE on the velocity, `MSE(F(x_t), clean - noise)`,
  plus `0.1 x` binary cross-entropy on a stop flag.

These are different training objectives. Our EOS term has no counterpart in the paper at
all, which is why nothing in the paper speaks to its behaviour.

### Per-language data: the paper does not say

- TTS training data is "a mix of public datasets totalizing 88k hours of speech".
  **No per-language breakdown is given.**
- Section D's detail is about *speech continuation*, trained on French and English from a
  Helium-1 2B backbone — a different task from the TTS model.
- The multilingual release (English, French, German, Spanish, Portuguese, Italian) is a
  separate blog post, and it does not publish hours or epochs per language either.
- So there is **no supported answer** to "how much per language", and nothing in the paper
  about changing the stopping mechanism between languages.

## The field's name for this

This is the **stop-token problem** in autoregressive TTS, and it is old and well
documented. Non-Attentive Tacotron states it directly: a network that decides whether to
stop at each frame means "a misprediction on a single frame can result in serious failures
such as early cut-off", and Tacotron 2 suffered "long babbling or long silence, often at
the end (failure to stop)".

Google's answer was to delete the stop token and **predict duration explicitly instead**.

## Why ours degrades: the arithmetic

| quantity | value |
|---|---|
| train clips | 19,084 |
| steps x gradient accumulation | 20,000 x 8 = 160,000 presentations |
| **epochs over the dataset** | **8.4** |
| frames per clip (mean) | 89 |
| EOS positives vs negatives per clip | 1 vs 88 |
| `pos_weight` on the positive frame | up to 20x |

The EOS head sees **one supervised positive frame per clip** and passes over the same
19,084 clips 8.4 times. There are only 19,084 distinct clip lengths to memorise, and by
epoch 8 it has seen each one eight times. The flow head, by contrast, gets up to 128
supervised frames per step — roughly a hundred times more signal per clip — which is
exactly why it keeps generalising while EOS does not.

Add exposure bias on top: the EOS head is trained under teacher forcing on ground-truth
prefixes, but at inference it must judge states the model drifted into itself.

## Scale, against the paper

| | paper (CALM TTS) | Pocket TTS released | ours |
|---|---|---|---|
| training audio | 88,000 h | not separately stated | **38.9 h** |
| speakers | large public corpora | - | **83** |
| backbone | 300M, 24 layers | 6-layer distilled student | 6-layer student |
| objective | 75/25 flow + LSD | - | flow only + 0.1 x EOS BCE |

We have **1/2,260th** of the paper's audio, and we are fine-tuning a model that is already
a distilled student. The literature is consistent that direct full fine-tuning on little
data overfits or catastrophically forgets, and that partial or parameter-efficient
fine-tuning is the standard mitigation.

## Ranked: effort against expected effect

| # | change | effort | expected effect | why |
|---|---|---|---|---|
| 1 | **Early stop / select on EOS, not total.** Use ~8,000 and stop the run there. | none | high | Already validated by ear. Costs nothing and the checkpoints exist. |
| 2 | **Lower `--eos-weight` from 0.1, or freeze the EOS head after ~8k steps.** | one flag | high | Lets flow keep improving to 18k without the stop-predictor rotting. Flow was still improving at 20k. |
| 3 | **Soften the EOS target.** Label the last 2-3 frames positive, or label-smooth, instead of exactly one frame at `pos_weight` 20. | ~10 lines | medium-high | Directly attacks the 1-positive-per-clip sparsity that is the root cause. |
| 4 | **Fewer epochs / more data.** 8.4 epochs over 38.9 h is a lot of repetition. Widen the duration range to recover the clips the 4 s floor excludes. | hours | medium | Standard overfitting remedy; also fixes the leading-silence artifact, since the model has never seen an utterance under 4 s. |
| 5 | **Parameter-efficient fine-tuning** (LoRA or train only later layers). | ~1 day | medium | Literature's standard answer for small-data adaptation of a pretrained TTS model. |
| 6 | **Explicit duration prediction instead of a stop token** (Non-Attentive Tacotron style). | large | high but invasive | The field's actual fix. Changes the architecture and breaks compatibility with the released Pocket TTS inference path. |
| 7 | **Implement LSD to match the paper's 75/25 objective.** | large | unknown for this symptom | Worth doing for fidelity to the paper, but LSD is about the *flow* head, and flow is the part that is working. It would not obviously fix EOS. |

**Recommendation: 1 + 2 + 3 together on the next run.** Select on EOS, drop `eos_weight`,
and soften the EOS target. That is under an hour of work, targets the measured root cause,
and leaves the flow head — which is behaving — untouched. Do 4 next if the symptom
survives.

## Out of scope, deliberately

- **Rewriting to explicit duration prediction (#6).** It is the correct long-term fix and
  the reason Non-Attentive Tacotron exists, but it changes the architecture away from the
  released Pocket TTS and we would lose the ability to load their weights.
- **Chasing the flow loss.** It never stopped improving. There is no evidence of a problem
  there, and LSD (#7) addresses that side.
- **More speakers.** 83 is genuinely few for voice cloning, but this specific symptom is a
  stop-predictor problem, not a speaker-generalisation problem.

## Sources

- Kyutai, *Continuous Audio Language Models*, arXiv:2509.06926 — `docs/2025_kyutai_pocket-tts.pdf`
- Shen et al., *Non-Attentive Tacotron*, arXiv:2010.04301 — https://arxiv.org/abs/2010.04301
- *Bridging the gap between training and inference in LM-based TTS*, arXiv:2509.17021 — https://arxiv.org/pdf/2509.17021
- *KALL-E: Autoregressive Speech Synthesis with Next-Distribution Prediction*, arXiv:2412.16846 — https://arxiv.org/html/2412.16846
- Neekhara et al., *Adapting TTS models for New Speakers using Transfer Learning*, arXiv:2110.05798 — https://arxiv.org/pdf/2110.05798

## Follow-up: what more data and longer samples would actually cost

The user's position: batch size is fixed by the hardware and is fine as long as enough
samples are seen; the real gaps are **sample duration** and **total hours**. Measured, both
are right, but they have very different price tags.

### Longer samples are free. More hours from this corpus are impossible.

| | |
|---|---|
| raw session audio | 50.5 h |
| transcribed segment audio | 36.5 h (72% of raw; the rest is silence and gaps) |
| what we currently keep | 38.9 h (slices include the pauses between segments) |
| hours lost to the 16 s cap | **0.01 h** |

Raising `max_duration` from 16 s to 30 s recovers essentially nothing, because no *source
segment* is longer than 16 s to begin with. But the slicer builds clips by merging
consecutive segments up to the cap, so raising the cap and the `target_duration` **does**
produce longer clips — it just repackages the same 36-39 h into fewer, longer examples
rather than adding audio.

That is worth doing on its own terms: it widens the duration distribution, which is what
the model currently lacks, and moves us toward the paper's 60 s samples. It costs one
config change and a re-run of preprocessing. It will not fix a data-quantity problem.

**The ceiling on CrowdRecital is about 36-39 h. There is no more audio in it.**

### For actual hours: ivrit.ai

The preprocessing tool is literally named `prepare_ivritai.py`, and ivrit.ai is the obvious
source:

- Original release: 3,300+ hours, 1,000+ speakers
- `ivrit-ai/audio-v2` on Hugging Face: **20,000+ hours**, and reported at 22,000+ as of
  July 2025 — the largest Hebrew audio dataset
- Licensed explicitly for AI training, including commercial

Against the paper's 88,000 h, ivrit.ai at 22,000 h puts us in the same order of magnitude
rather than 2,260x short. It also carries far more than 83 speakers, which is the other
scale problem.

Costs to be honest about: it is delivered as raw audio, post-VAD audio, and *partially*
transcribed data. TTS needs accurate transcripts, so the transcribed subset is the usable
part, and it would need the same segmentation, quality filtering and normalisation
pipeline we already built. Disk and preprocessing time both scale accordingly — our 38.9 h
took 8.7 minutes and 6.3 GB, so 1,000 h would be roughly 4 hours and 160 GB.

### Revised ranking, given all of the above

| # | change | effort | effect |
|---|---|---|---|
| 1 | Select checkpoint on EOS, not total loss | none | high, already validated by ear |
| 2 | Lower `--eos-weight` or freeze the EOS head after ~8k | one flag | high |
| 3 | Re-slice with a wider duration target (e.g. target 15 s, cap 30 s) | one config + re-run | medium; fixes the narrow duration distribution |
| 4 | Add ivrit.ai transcribed data | days | **highest**, and the only route past 39 h |
| 5 | Soften the EOS target to the last 2-3 frames | ~10 lines | medium-high |

1 and 2 are free and target the measured symptom. 3 is cheap and targets the duration
point. 4 is the only thing that changes the fundamental data situation.

### Sources

- ivrit.ai dataset paper, arXiv:2307.08720 — https://arxiv.org/abs/2307.08720
- `ivrit-ai/audio-v2` — https://huggingface.co/datasets/ivrit-ai/audio-v2

## Was the ~10 s cap really the loss? Yes — and it costs nothing to lift

The user asked whether the original 8 s target came from the loss window. It did:
`--head-samples 128` at 12.5 Hz = **10.24 s**. Clips longer than that get randomly
subsampled, so only ~10 s of any clip contributes to the flow loss per step. Targeting 8 s
kept ~92% of clips fully covered.

`head_samples` is just a flag. The question is what it costs. Measured on the RTX 4060:

| clip | frames | head_samples | peak VRAM | ms/clip | audio-seconds per compute-second |
|---|---|---|---|---|---|
| 7 s | 87 | 128 | 0.82 GB | 52 ms | 134x |
| 15 s | 187 | 190 | 0.82 GB | 55 ms | 272x |
| 30 s | 375 | 375 | 0.82 GB | 107 ms | **282x** |
| 45 s | 562 | 375 | 0.85 GB | - | - |
| 60 s | 750 | 750 | 1.04 GB | - | - |

**VRAM is a non-issue.** A 60-second clip with full loss coverage uses 1.04 GB of 8 GB. The
model is small enough that clip length barely registers.

**Longer clips are more compute-efficient, not less.** 282x realtime at 30 s versus 134x at
7 s, because the fixed per-step overhead (text encoding, prefix, kernel launches) amortises
over more frames. An epoch over the same 38.9 h is roughly *twice as fast* at 30 s clips.

So the 8 s cap was a reasonable choice given `head_samples=128`, but it was never a
hardware limit, and lifting it is free in memory and cheaper in time.

### Concrete proposal for the next dataset

- `target_duration` 8 -> 20, `max_duration` 16 -> 30, keep `min_duration` 4
- `--head-samples` 128 -> 375, so a 30 s clip is still fully covered by the loss
- Same 38.9 h, repackaged into roughly 3x fewer, 3x longer clips

Two consequences to expect, one good and one to watch:

- Fewer, longer clips means **fewer EOS positives to memorise** (about 6,500 clips instead
  of 19,084) but each is a harder, more varied stopping decision. Whether that helps or
  hurts the EOS overfitting is an empirical question, not a prediction.
- Fewer clips also means fewer distinct prompt/target pairings per speaker, so check the
  pairing statistics after re-slicing.
