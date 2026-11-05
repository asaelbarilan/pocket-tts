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

## The binding constraint, and the dataset that removes it

**Twelve decisions cannot tell these options apart.** 11/12 against 10/12 is one word of
difference; a coin lands that way often. This is the same wall the alignment work hit at
72 clips.

**MILIM-Bench removes it.** `renikud/MILIM-Bench` on the Hub (MIT, from the ReNikud paper)
is 1,653 Hebrew sentences in 11 categories with gold IPA for 3,110 target words, written
as `token_index=IPA`. It was built for exactly the confusing parts of Hebrew: homographs,
gender, acronyms, loanwords, names, slang, colloquial register, rare phonemes, penultimate
stress, minimal stress pairs, and the test split of ILSpeech as a control.

Two properties that make it better than the vocalized-corpus mining this document
originally proposed:

- **Gold IPA, not gold nikud, so stress is in the label.** Nikud cannot write stress, and
  stress is two of the eleven categories. A nikud-supervised objective is blind there.
- **Wide and shallow: 1,284 distinct target forms, none appearing more than four times.**
  Mining a corpus gives the opposite — `אוכל` and `רוצה` dominating — where a model that
  always picks the frequent reading looks good. Nothing here needs reweighting.

Split with `hebrew_training/milim_split.py`, 60/20/20, two ways, because they measure
different things:

- `--by sentence` (983 / 318 / 350 sentences, stratified by category): the same form may
  appear in train and test in different sentences. The deployment setting — your TTS will
  meet `בוקר` in the wild and you would of course have trained on it.
- `--by form` (900 / 302 / 287): a form never crosses a split, so the test words are
  unseen. Harder, and the number will be lower. Quote both or the easy one flatters. This
  mode collapses `ilspeech-v2` from 1,539 target words to 11, because its sentences carry
  ten targets each and nearly all of them straddle; score that category under
  `--by sentence` only.

The split is a SHA-1 of the sentence text rather than a seed, so it is identical on any
machine and after any reshuffle; `hebrew_training/milim/*_assignment.tsv` is the audit
trail. Three traps are documented in that script and cost an afternoon otherwise: row 0 of
`predictions.tsv` is a `#meta` row (so the benchmark's own `src/evaluate.py` raises on its
length check against the published `gold.tsv`), rows 940 and 1129 have a target index past
the end of the sentence, and `ilspeech-v2` is half the benchmark by target word, so a
micro-average over target words is 50% one control category. **Quote the macro average
over the eleven categories.**

MILIM is not a substitute for a Hebrew corpus to *train* on — 983 training sentences will
not move an encoder. It is the measurement. Training data still comes from a vocalized
corpus or, better, from the label factory below.

## What MILIM says about the two models — measured

`predictions.tsv` ships nine systems' answers: two Hebrew g2p models, one preview g2p, and
six LLM runs with the prompts in the repo's `PROMPTS.md`. Reproduce with
`hebrew_training/milim_systems.py`, which scores per target word rather than per row.
These are the benchmark authors' runs, not ours; the LLM numbers depend on the prompt as
well as the model.

| system | MACRO | colloquial | acronyms | rare phon. | homographs | ilspeech |
|---|---|---|---|---|---|---|
| gemini-4-argon | **82.7** | 51.0 | 80.9 | 76.8 | 93.0 | 89.1 |
| conikud (preview) | 78.8 | 52.3 | 78.9 | 59.6 | 90.0 | 91.8 |
| gpt-6-astra | 76.8 | 47.7 | 70.4 | 70.9 | 81.1 | 87.3 |
| gemini-3.1-pro | 75.1 | 34.4 | 77.6 | 56.3 | 94.0 | 95.4 |
| claude-fable-5 | 73.4 | 18.5 | 78.9 | 74.2 | 92.0 | 88.4 |
| claude-opus-5 | 71.3 | 17.2 | 68.4 | 65.6 | 78.6 | 83.0 |
| **renikud** | 59.7 | **55.0** | **3.3** | **27.2** | 77.1 | **90.4** |
| phonikud | 49.5 | 9.3 | 35.5 | 17.9 | 77.6 | 84.1 |

Three things fall out, and all three change the plan.

**1. ReNikud is not the second model. It comes seventh of nine.** 59.7 macro against
gemini-4's 82.7. It scores 3.3% on acronyms — it has no acronym handling at all — and
27.2% on rare phonemes. The twelve-word result that started this document had it at 10/12;
at 3,107 words it is nowhere near DictaBERT's class of competitor. Note also that
**MILIM is by ReNikud's own authors and ReNikud loses on it**, which is a point in the
benchmark's favour, not against it.

**2. But its one win is exactly the register a TTS has to speak.** `colloquial` 55.0
against gemini-4's 51.0, beating every LLM in the file; `ilspeech-v2` — real speech
transcripts — 90.4 against 89.1. And the losses are *systematic*, not noise:

```
[slang]      קרוע       gold kaʁˈua     renikud kaʁˈua     gemini-4 kaʁˈuʔa
[slang]      טירוף      gold teʁˈuf     renikud teʁˈuf     gemini-4 tiʁˈuf
[colloquial] הגיע       gold heɡˈia     renikud heɡˈia     gemini-4 hiɡˈiʔa
[colloquial] לשניים     gold leʃnˈajim  renikud leʃnˈajim  gemini-4 liʃnˈajim
```

The LLM inserts glottal stops nobody says and applies literary vowel reduction. **It reads
book Hebrew; ReNikud speaks street Hebrew.** This is the real "good from both worlds", and
it is narrower and sharper than "two semantic models": the LLM knows *which word*, ReNikud
knows *how it is actually said*.

**3. The gate's ceiling is measured, not hoped for.** ReNikud plus gemini-4, oracle:
**90.3%** micro against gemini-4's 85.7% alone, and 94.8% on ilspeech. Over all nine
systems, 95.4%, with only 4.6% of target words that no system gets right (and 80 of those
142 are ilspeech and colloquial). The 143 words where ReNikud alone is right are
concentrated where you would now expect: ilspeech 87, colloquial 24, foreign 11.

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

## The four picked, re-costed against the measurement

The four chosen were 6.1 (LLM as label factory), 4 (ensemble then distil), the contrastive
half of 5, and the TTS loss as the gate's supervision. They compose into one pipeline, and
MILIM changes each of them.

### 6.1, the label factory — confirmed, with a correction

A frontier LLM at 82.7 macro is a far better teacher than ReNikud at 59.7, so the choice
is right. But **its errors are systematic in the direction that hurts a TTS most**: it
writes book Hebrew. Train an encoder on raw LLM labels and you teach it to say *hiɡˈiʔa*
where a speaker says *heɡˈia*, across the whole corpus and not just on hard words.

Two things follow. Pick the teacher on `colloquial` and `ilspeech`, not on macro — on those
two, gemini-4 (51.0) and conikud (52.3) beat the rest and claude-opus-5 is at 17.2, a
spread of 35 points that the macro column hides. And **take ReNikud as the register
corrector**: where the two disagree and the disagreement is a glottal stop or a reduced
vowel, ReNikud is right about 24 times in 42 on colloquial. That is a rule you can write
and check on the validation split today.

### 4, ensemble then distil — now has a number

The target is no longer a guess: **90.3%** for ReNikud + gemini-4, against 85.7% for the
LLM alone. If the distilled student lands below 85.7 the ensemble bought nothing; above
that, the gap to 90.3 tells you how much of the gate survived distillation. Nine systems
oracle at 95.4%, so a larger pool is worth more — but each extra teacher is another LLM
bill, and the three-way gate has to be trained on something.

### 5, the contrastive half — the negatives are already written

The thing option 5 lacked was hard negatives. `predictions.tsv` is a free supply: for each
target word you have the gold IPA and up to eight *distinct wrong* IPA strings that real
systems produced in that exact context. Anchor = the word in its sentence, positive = gold
IPA, negatives = the wrong strings. They are hard by construction — `kaʁˈua` against
`kaʁˈuʔa` differ by one phoneme — and no generation step is needed.

3,107 anchors is small for contrastive training, so this is a fine-tune on top of option
1's run rather than a run of its own. The 142 words no system gets right are the natural
held-out difficulty probe.

### The TTS loss — cheapest, but it cannot be scored on MILIM

The audio-as-label argument stands and is still the best supervision available. One wrinkle
the measurement exposes: **a gate over two vectors produces a vector, and MILIM scores
IPA strings.** A vector gate is only measurable by listening or by TTS loss, neither of
which gives a number per category.

So build the gate over **IPA candidates**, not over vectors: two candidate strings per
word, ReNikud's and the cached LLM's, with the gate choosing. Then

- MILIM scores it directly, per category, today, with the oracle 90.3% as the known
  ceiling and 85.7% as the floor to beat;
- the chosen string feeds the phoneme path the TTS already has, so nothing downstream
  changes;
- the TTS loss can still train it, and the two kinds of supervision become comparable
  because they now score the same object.

Keep the vector residual as it is. Gate the strings.

## What I would do

In this order, stopping as soon as the numbers stop moving:

1. **Take the split.** Done: `hebrew_training/milim_split.py`, 983/318/350 by sentence and
   900/302/287 by form. Report macro over the eleven categories, both splits.
2. **Score the pipeline as it stands on the test split.** You have never had a per-category
   number for what the TTS front end actually does. One afternoon, and it decides
   everything after it.
3. **The IPA-candidate gate, ReNikud against one cached LLM.** Fit on train, select on val,
   report on test. Floor 85.7, ceiling 90.3, and a trivial baseline to beat first: "use the
   LLM except on colloquial and slang, where you use ReNikud". If that hand rule gets most
   of the 4.6 points, the learned gate may not be worth building.
4. **Then the label factory**, chosen on colloquial and ilspeech rather than on macro, with
   the step-3 gate applied to its output so the labels carry the spoken register.
5. **Then one encoder**: option 1's nikud-or-IPA head plus option 5's contrastive head,
   trained on the factory's labels, MILIM held out. This is the single model the question
   asked for, and by this point its target is a measured number.

Option 4's distillation is the last step, and only if step 5 lands below step 3.

Not on the list: training a Hebrew semantic encoder from scratch. DictaBERT is a
Hebrew-only encoder with a real pretraining budget behind it, and nothing measured here
suggests its semantics are the bottleneck — it sits at 68/72 on the homograph probe, while
ReNikud's 59.7 macro says the *reading* side is where the loss is.

One more thing worth a day: **conikud** (preview, 78.8 macro, and the best system in the
file on ilspeech at 91.8 and on colloquial at 52.3) dominates ReNikud everywhere —
per-word, ReNikud wins only 74 of 3,107 against it. If a local ONNX g2p is wanted in the
pipeline, it is probably the wrong one that is in there now.
