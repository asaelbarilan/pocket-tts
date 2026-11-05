# One Hebrew encoder that knows both the sense and the reading

Written 2026-11-05, after the DictaBERT / ReNikud comparison in
[homograph-disambiguation.md](homograph-disambiguation.md).

## The result this starts from

On the twelve ambiguous words in the seven validation sentences:

| | correct | where it fails |
|---|---|---|
| `dicta-il/dictabert-large` + centroid probe | 11 / 12 | `הבוקר` → *morning* (should be cowboy) |
| `renikud/renikud` (g2p, zero-shot) | 10 / 12 | `בוקר` → *cowboy*, `מלך` → *king* |
| oracle: pick whichever is right | 12 / 12 | — |

On all 72 decisions (anchors included): 68/72 and 55/72, they disagree on 21 (29%),
DictaBERT right in 17, ReNikud right in 4, neither right in 0. **Every disagreement has a
winner.** That is the whole reason this document exists: an oracle gate scores 72/72, so
the information needed to be right is present in the pair, and the only missing piece is
something that decides which one to believe.

## First, name what the two models actually are

They are not two semantic models. Calling them that is what makes the fusion look harder
than it is.

- **DictaBERT is a semantic encoder.** It outputs a contextual vector per token. It has no
  notion of sound at all. Its answer about `הספר` is not "barber" — it is a point in
  R^1024 that happens to sit nearer the barber sentences than the book sentences, and only
  a probe we wrote turns that into a word.
- **ReNikud is a reading model.** It outputs IPA. It states the answer outright, which is
  why it needs no anchors; and it loses on `בוקר` because it is a grapheme-to-phoneme
  model being asked a question about cows, which it was never built to answer.

So "capture the good from both worlds" does not mean averaging two embeddings. It means:
**take a semantic encoder and give it a phonological training objective.** One model, one
vector, and the vector's geometry has to encode the reading because the loss made it.

## The binding constraint, before any option

**Twelve decisions cannot tell these options apart.** 11/12 against 10/12 is one word of
difference; a coin lands that way often. This is the same wall the alignment work hit at
72 clips.

Before training anything, mine a real test set — and it can be mined, not annotated:

1. Take a **vocalized** Hebrew corpus (the Dicta vocalized corpora, the Tanakh, the
   Ben-Yehuda project, any ktiv-menukad text).
2. Strip the nikud. Group the sentences by bare form.
3. Any bare form that appears with **two or more distinct vocalizations** is a homograph,
   and each sentence it came from is a labelled decision — the gold label is the nikud that
   was stripped off it.

That yields thousands of decisions across hundreds of homographs for the cost of a script,
with no human in the loop, and it doubles as the training set for options 1, 2 and 5
below. Expect a frequency-skewed set: `אוכל` and `רוצה` will dominate, so report
macro-averaged per-form accuracy as well as the raw number, or a model that always says
*eating* will look good.

The known weakness of this set: it is written, mostly literary Hebrew, and it will
under-represent the loanwords and the spoken register the TTS actually has to say. It is
still four orders of magnitude better than twelve.

## Option 1 — distil the reading into the semantic encoder (multi-task)

Keep DictaBERT-large as the backbone, because it already wins the semantic half. Add a
second head on top of the same hidden states that predicts the **nikud, or the IPA, per
character**, and train both heads on the mined corpus.

- The nikud head's targets are gold — they were stripped off the text — so this does not
  inherit ReNikud's mistakes, which matters, because those mistakes are exactly the cases
  we care about.
- You ship the backbone's hidden states, not the head's output. The head is scaffolding
  whose only job is to force phonology into the vector. At inference it can be deleted.
- One encoder, one vector per word, and it drops straight into the residual connection at
  the model entrance that is already there. No change to the TTS.

Cost: a continued-pretraining run, not a pretraining run. Order of a GPU-day on a few
hundred thousand vocalized sentences, no labelling.

Risk: the nikud objective is dominated by the easy ~95% of words whose vocalization is
determined by spelling alone, and the gradient from those drowns the homographs. Mitigate
by **upweighting the ambiguous forms** — you know which they are; that is exactly what
step 2 of the mining produced.

## Option 2 — nikud as auxiliary task only, never as output

The weaker, cheaper version of option 1: the same auxiliary diacritization head, but nikud
and IPA appear nowhere in the pipeline. It stays text → encoder → vector → TTS, and the
nikud exists only in the loss.

Worth separating from option 1 because of something already measured here: the diacritizer
`dictabert-large-char-menaked` gets `הספר` **wrong** (4 of 7 sentences right), while
`dictabert-large`'s own vectors get it right. Running a diacritizer and feeding the TTS its
output is therefore strictly worse than using the semantic model directly — an *output* of
a diacritizer is a commitment, and a wrong commitment is unrecoverable, while a *loss*
from diacritization is just a gradient. Same supervision, opposite effect.

## Option 3 — gate the two frozen encoders, supervised by the TTS loss

The one to run this week, because it needs no new training data at all.

Two frozen encoders, a per-token gate over `[a; b; a-b]`, output projection zero-init so
the model starts exactly where it is today, independent dropout on each branch so neither
can be ignored. A softmax over two logits is a sigmoid of their difference, so "attention
over two models" and "a gate" are the same layer; use the gate, it is one matmul.

The part worth saying plainly: **you do not need sense labels to train the gate.** The TTS
has the audio. If the gate picks *morning* where the speaker said *bokér* the cowboy, the
predicted mel is wrong and the reconstruction loss says so. The audio is the label, there
are thousands of hours of it, and the supervision is free and already inside the training
loop. This is the cheapest thing on this list by a wide margin.

ReNikud emits IPA, not a vector, so branch B is an embedding of its IPA string — reuse the
phoneme embedding table the TTS already has, averaged over the word's phonemes.

What this option does not give you: one model. You are running two at inference, and one
of them is an ONNX g2p with a hand-written number front end. Which leads to —

## Option 4 — ensemble, then distil

Options 3 and 1 compose. Train the gate (free supervision, measured against the mined
set), then train a single encoder to regress the **gated** vector on a large unlabelled
corpus. The student sees only text and learns to produce what the pair produced, the
gate's choices included. Ship the student.

This is what "capture the good from both worlds" means mechanically: ensemble to get the
accuracy, distil to get back to one model. It is also the only option here whose accuracy
target is known in advance — the gate's number, which you will already have.

## Option 5 — contrastive on the reading, which optimises the metric itself

Instead of predicting nikud, make the embedding space separate the readings.

For each mined homograph you have a bare form, a sentence, and the gold vocalization.
Train with InfoNCE: the vector of the word in its sentence pulls toward an embedding of
its **correct** vocalized form and pushes away from its **other attested** vocalizations.
The negatives come free from the grouping, and they are hard negatives by construction —
same consonants, different vowels.

Why this beats option 1 for a measurable win: option 1 hopes phonology lands in the
geometry as a side effect of a per-character classifier. This makes the geometry the
objective. Its inference-time test is also a nearest neighbour — the same centroid probe
already written in `hebrew_training/renikud_vs_bert.py`, so the evaluation code exists.

Cost is the same continued-pretraining run as option 1, and the two share it: one backbone,
a nikud head and a contrastive head.

## Option 6 — a decoder turned into an encoder

Everything above keeps a 2021-era BERT as the backbone. The ceiling on `הבוקר` is world
knowledge — that a *bokér* is the one who herds the cows — and a 7B decoder holds far more
of it than DictaBERT-large has parameters.

Three ways in, cheapest first:

1. **The LLM as a label factory, not as the model.** Do not put the decoder in the
   pipeline. Use it once, offline, to vocalize or sense-label millions of Hebrew sentences,
   then train the small encoder of options 1/5 on its output. You get the knowledge at the
   encoder's inference cost, and you can check its labels against the mined gold to measure
   how much it is making up. Highest value per hour on this list after option 3.
2. **LLM2Vec on a Hebrew decoder.** `dicta-il/dictalm2.0` (Mistral-7B adapted to Hebrew),
   or Gemma/Qwen. Three steps: switch the causal mask to bidirectional, adapt with masked
   next-token prediction so the model tolerates the new mask, then SimCSE-style contrastive
   on dropout pairs. You get per-token contextual vectors from a model with real world
   knowledge. The free alternative, if you want bidirectionality without any training:
   **echo embeddings** — feed the sentence twice and read the tokens of the second copy,
   which have seen the whole first copy through the causal mask. Worth measuring before
   paying for LLM2Vec.
3. **Distil it down.** A 7B encoder in a TTS front end is not shippable; option 4's
   machinery applies unchanged, with the LLM2Vec'd decoder as the teacher.

The honest objection to all three: Hebrew is a thin slice of every multilingual decoder's
training data, and DictaBERT was trained on Hebrew and nothing else. A 7B model with 2%
Hebrew can lose to a 0.4B model with 100% Hebrew. That is measurable on the mined set
before any of it is built, and the echo-embedding variant gives you the measurement for
the cost of an afternoon.

## What is missing from both models, and from this whole list

Nikud does not mark stress. `שָׂרָה` is *sára* the name and *sará* the verb under identical
nikud, and nikud cannot tell them apart; the same holds for the `בֹּקֶר` / `בּוֹקֵר` family only
because those happen to differ in the vowels as well. Every option above that uses nikud as
its supervision inherits that blind spot. Phonikud exists for exactly this and predicts
stress on top of nikud; and IPA, option 1's other possible target, carries stress where
nikud does not. If the mined set shows stress errors surviving, that is where to go.

## What I would do

In this order, stopping as soon as the numbers stop moving:

1. **Mine the test set.** A script over a vocalized corpus. Nothing else here is
   falsifiable until it exists, and twelve decisions is not a measurement.
2. **Option 3, the gate, supervised by the TTS loss.** No new data, no new model, free
   supervision, and the oracle already says there is headroom to collect. Measure it on the
   mined set, not on twelve words.
3. **Re-measure the premise.** With thousands of decisions, check whether ReNikud still
   wins anything. If its 4-of-21 shrinks to noise at scale, the fusion question dissolves
   and the answer is "use DictaBERT, drop ReNikud" — the cheapest possible outcome, and a
   live possibility on twelve words.
4. **If it survives: options 1 and 5 together.** One continued-pretraining run of
   DictaBERT-large with a gold-nikud head and a contrastive homograph head, ambiguous forms
   upweighted. This is the one model the question asked for.
5. **Then option 6.1**, an LLM used offline as the label factory for step 4, which raises
   the ceiling without putting a 7B model in the pipeline.

Option 4's distillation is only worth building if step 2 wins and step 4 does not reach it
— it is the fallback that buys the ensemble's accuracy at one model's cost.

Not on the list: training a Hebrew semantic encoder from scratch. DictaBERT is a
Hebrew-only encoder with a real pretraining budget behind it, and nothing measured here
suggests its semantics are the bottleneck — it sits at 68/72.
