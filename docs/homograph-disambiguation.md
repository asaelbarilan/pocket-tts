# Can the diacritizer tell ספר from ספר?

Measured 2026-10-30 on the seven `homograph` rows of
`hebrew_training/eval_sentences_he.jsonl`.

## Why ask

Unvocalized Hebrew leaves the vowels out, so one spelling is often several words. The TTS
model has to pick one, and it has only the sentence to pick from. The obvious thing to try
is to vocalize the text first with `dicta-il/dictabert-large-char-menaked` — already in the
pipeline, as the generator of the nikud slice in `build_tokenizer_corpus.py` — and feed the
model unambiguous input.

That only helps if the diacritizer itself gets the reading right. This is the check.

**The model tested is DictaBERT-char-menaked, the diacritizer. Not DictaLM**, the
generative Hebrew LLM, which has not been run against this set.

## Method

The seven sentences were written so that context, and nothing else, settles the reading;
each was hand-vocalized for the intended reading, and that hand nikud is the gold. The
diacritizer saw only the bare sentence. Thirteen words across the seven carry the
ambiguity, and those thirteen are what is scored — judged by eye, because the model
normalizes ktiv male to ktiv menukad (`עכשיו` → `עַכְשָׁו`) and a string comparison would
score spelling rather than reading.

Reproduce with `probe_homographs.py` (in the session scratchpad; it loads the model, runs
`predict()` with `PreTrainedTokenizerFast`, and prints each target word against its gold).

## Result: 10 of 13 words, 4 of 7 sentences

| id | the ambiguity | gold | DictaBERT | |
|---|---|---|---|---|
| homo_01 | הספר | הַסַּפָּר *sapar*, barber | הַסֵּפֶר *sefer*, book | wrong |
| homo_02 | רוצה / רוצה | רוֹצָה then רוֹצֶה | רוֹצָה then רוֹצֶה | right |
| homo_03 | אוכל | אוּכַל *uchal* | אוּכַל | right |
| homo_04 | המורה / למורה | הַמּוֹרָה then לַמּוֹרֶה | same | right |
| homo_05 | הבוקר | הַבּוֹקֵר *boker*, cowboy | הַבֹּקֶר *boker*, morning | wrong |
| homo_06 | דובר / דובר | דּוֹבֵר then דֻּבַּר | דּוֹבֵר then דֻּבַּר | right |
| homo_07 | מלך | מָלַךְ *malach*, reigned | מֶלַךְ | wrong vowel |

### What it gets right

Grammatical agreement, reliably. `רוצה` after `היא` against after `והוא`, `המורה החדשה`
against `למורה הוותיק` — both resolved from one agreeing word nearby. `דובר` is the
strongest of the seven: it read the first as the noun *dover* and the second, four words
later after `שכבר`, as the pu'al *dubar*. That is a real syntactic decision, not a
frequency prior.

### What it gets wrong

Lexical choice against frequency. All three failures pick the common word over the one the
sentence means:

- **homo_01 is the case this was built to test.** `הספר סיפר לי שהוא קרא ספר על תספורות`
  — "the barber told me he read a book about haircuts". It returned *ha-sefer*, the book.
  The sentence hands it two signals, that the subject *tells* and that the topic is
  haircuts, and it took neither. The second `ספר` is correct, but a model that answers
  *sefer* every time is right there for free.
- **homo_05**: `הבוקר` next to `לפרות שלו ברפת` — cows, in a cowshed — still came back as
  *ha-boker*, the morning.
- **homo_07**: `מלך` after `המלך` was read as a verb, which is the hard half of the
  decision, but vocalized `מֶלַךְ` — not a form of anything. Right analysis, wrong vowels.

## What follows

Diacritizing the input is worth doing and is not sufficient. It fixes agreement, which is
most of the running cost, and leaves the lexical cases — which are the ones a listener
notices, because *the book told me* is not a pronunciation error, it is a different
sentence.

Two things this does **not** license, both unmeasured:

- that DictaLM would do better. It is a generative LLM with a far larger context model and
  may well resolve `הספר ... תספורות`; nobody has run it.
- that the TTS model will inherit these errors. It reads the text it is given; if the text
  is vocalized by hand, as `text_nikud` is in the eval set, none of this applies. That is
  exactly why the set carries both spellings of all seven.

---

# Does `dictabert-large` know, even though it cannot say?

Measured 2026-10-30. `dicta-il/dictabert-large` is the wordpiece model, with no nikud head,
so it cannot be asked to vocalize `הספר`. The question is whether the distinction is in the
representation anyway — because if it is, a small head can read it off, and if it is not,
nothing downstream will recover it.

## The method, and the one thing to get right

Cosine similarity is the right instrument, on **contextual** vectors, not static ones.
Comparing the embedding of `ספר` against `לספר` or `סַפָּר` asks about BERT's input table,
which has one vector per wordpiece and therefore cannot hold a sense distinction at all.
The sense lives in the last hidden layer, where the same wordpiece gets a different vector
in a different sentence. So:

1. Write sentences where the reading is beyond doubt — four barbers, four books, the target
   written `הספר` in every one so the probe compares like with like.
2. Take the last-layer vector of `ספר` in each, averaged over its wordpieces, L2-normalized.
3. Average each sense into a centroid.
4. For the sentence in question, take cosine to both centroids. The larger wins.

**Validate the probe before believing it.** Leave each anchor out, rebuild the centroids
without it, and classify it. A probe that cannot label sentences whose answer is written on
them is not evidence about one whose answer is the open question. This one scores 8/8.

## Result: it knows

    homo_01  הספר סיפר לי שהוא קרא ספר על תספורות.
             -> barber    barber +0.717   book +0.469   margin 0.248

    control  הספר סיפר לי על חייו של המלך.
             -> book      barber +0.803   book +0.897   margin 0.094

The sentence the diacritizer read as *the book told me* is, to `dictabert-large`, clearly
the barber — by a margin two and a half times the control's. Swap the haircut for a king's
life and it moves to the book. The information the menaked model needed was available in
the encoder; its vocalization head did not use it.

A masked-fill probe on the same two sentences is weaker and worth recording as such: mask
`הספר` and the top ten are people in **both** frames (`הוא, בעלי, חבר, מישהו, אחי`), because
`סיפר לי` demands a person whichever reading is meant. It confirms the model parses the
syntax; it does not separate the senses. The centroid probe is the one that carries weight.

## What this does and does not establish

It is not a fair race against the diacritizer. The probe is handed the two candidate senses
and labelled examples of each; the menaked model chooses over all of Hebrew with neither.
The claim is only the one asked for: **the distinction is present in `dictabert-large`'s
contextual representation, and recoverable with a linear read-off.**

It also rests on eight anchors and two test sentences. Before building on it, the next step
is the cheap one: twenty sentences per sense, held out properly, and the same probe over all
seven homographs rather than `ספר` alone.

Rerun with `hebrew_training/probe_sense.py`.

---

# Two encoders at the front: how often do they disagree?

Measured 2026-11-05 with `hebrew_training/sense_disagreement.py` over
`hebrew_training/sense_anchors_he.json` — 42 anchor decisions scored leave-one-out plus the
9 eval-set sentences, 51 word-sense decisions in all, each a two-way choice.

The architecture question behind it: with two semantic encoders feeding the model, a learned
gate only earns its parameters if the two disagree *and* neither is uniformly right. If they
agree everywhere there is nothing to gate. If one wins every disagreement, the gate learns a
constant and you should have dropped the other model.

    A = dicta-il/dictabert-large        (wordpiece)
    B = dicta-il/dictabert-large-char   (character)

                       anchors (42)   tests (9)   all (51)
      A correct           41            8           49  (96%)
      B correct           23            6           29  (57%)
      agree               24 (57%)      7 (78%)     31  (61%)
      DISAGREE            18 (43%)      2 (22%)     20  (39%)
        A right              18            2           20
        B right               0            0            0
      oracle gate         41            8           49  (96%)

## There is nothing to gate

They disagree on **39%** of decisions, far more than the 10% the question assumed. But the
split is **20–0**: on every single disagreement the wordpiece model is right and the
character model is wrong. B scores 57% on a two-way choice, which is a coin flip with a
limp.

The oracle row is the point. A gate that resolved every disagreement perfectly would score
49 of 51 — exactly what A scores alone. **The headroom is zero.** Any gate trained on this
signal converges to "always A", which is a constant, which is not a gate.

## What this does not say

It does not say drop the character model. It says the character model carries no word-sense
signal, which is unsurprising: it was trained to place nikud, not to tell a barber from a
book. The case for keeping two encoders rests on a task where B wins, and this measurement
found none — because it only asked one kind of question.

So the next measurement is the one that would justify the architecture: a decision where the
character model should dominate — vocalized spelling, a word the wordpiece vocabulary has to
break into pieces, a name. If B never wins anywhere, one encoder is the answer and the
residual should carry A alone.

Two limits worth keeping in view: 51 decisions is small, and the anchor sentences are mine,
so they are clean in a way real text is not. And if the two encoders actually wired into the
model are a different pair, rerun with `--models` — nothing here is specific to these two.

---

# ReNikud against DictaBERT-large

Measured 2026-11-05, `hebrew_training/renikud_vs_bert.py`, same 51 decisions.

`renikud/renikud` is grapheme-to-phoneme, not an encoder: it emits IPA, so the
pronunciation it gives the ambiguous word *is* the reading it chose. It needs no anchors and
gets none, while the DictaBERT probe is handed three labelled sentences per sense. ReNikud
is therefore playing on harder terms.

                       anchors (42)   tests (9)   all (51)
      A  dictabert-large   41           8          49  (96%)
      B  renikud           33           8          41  (80%)
      agree                32 (76%)     7 (78%)    39  (76%)
      DISAGREE             10 (24%)     2 (22%)    12  (24%)
        A right             9           1          10
        B right             1           1           2
        neither             0           0           0
      oracle gate          42/42       9/9         51  (100%)

## This is a different answer from the character model

The char model lost every disagreement 20–0 and left no headroom. ReNikud wins two, and an
oracle gate reaches **51/51 against DictaBERT's 49** — so a combiner has something to buy
here, which it did not before. The disagreement rate, 24%, is also near the 10% the question
assumed rather than the char model's 39%.

Both of ReNikud's wins are worth looking at individually, because they are not the same kind
of event:

- `המורה שלי אמר שהמבחן נדחה` — masculine, and DictaBERT said feminine. This is a real win:
  the sentence differs from its feminine twin by one letter, `אמר` against `אמרה`, and
  ReNikud read it.
- `בוקר טוב, אמר הבוקר לפרות שלו ברפת` — cowboy, and DictaBERT said morning. **This one is
  luck.** ReNikud answers `habokˈeʁ` for every single `הבוקר` in the set, including all three
  sentences that mean *this morning*. It was right here by having one fixed answer that
  happened to match.

## Where ReNikud fails, it fails flatly

Per homograph, correct out of all its decisions:

      sefer 6/7   rotze 7/8   ochel 7/7   more 6/7   dover 8/8   boker 4/7   melech 3/7

`dover` is perfect, including *dover* against *dubar* four words apart. But `melech` returns
`mˈeleχ` for all seven sentences and never once produces the verb `malˈaχ`, and `boker`
returns `habokˈeʁ` for all seven. Those are not mistakes of context — the word has one
answer and context cannot move it. Two of seven homographs are simply fixed.

## What to do with it

The honest size of the prize is two decisions out of fifty-one, and one of those is a
coin that landed well. That is real but far too thin to justify a gate on, and far too thin
to train one: a gate has to learn *when* ReNikud is right, and there is one genuine example
of that in the whole set.

So the next step is not architecture, it is more decisions — the same probe over enough
homographs that "B wins" has a shape a gate could fit. If ReNikud's wins stay concentrated
in cases DictaBERT misses for a reason, the gate is worth building; if they stay at one in
fifty, the residual should carry DictaBERT alone.

Rerunning needs `renikud_onnx.py` and `model.onnx`, which are not in this repo: take the
helper from `niobures/ReNikud` (`models/renikud-plus/renikud_onnx.py`) and the model from
`renikud/renikud`, put both in one folder, and pass that folder as the first argument.

---

# ReNikud over the whole validation set, not just the homographs

The comparison above used seven sentences. This is all forty, `hebrew_training/renikud_eval_set.py`.
There is no gold IPA for the other six categories, so these are not scored — they are read,
and three of the categories say something without needing a score.

      plain                   6/6   clean
      nikud                   0/6   BROKEN -- raw Hebrew survives into the IPA
      latin_in_hebrew         6/6   Latin passes through untouched
      english                 4/4   passes through untouched, nothing is phonemized
      loanword_hebrew_script  5/5   clean
      numbers                 6/6   expanded correctly, and well
      homograph               7/7   clean, scored above

## It cannot take vocalized Hebrew

Every one of the six nikud sentences comes back with the diacritics still embedded in the
output string:

    in   אֲנִי רוֹצָה לָלֶכֶת הַבַּיְתָה, תּוֹדָה רַבָּה.
    ipa  ʔaֲnִˈi ʁoֹtsָa leָlֶχֶt haַbeַּiְtָˈa, tּuֹdָa ʁeַbaָּa.

The nikud characters pass through as raw combining marks between the phonemes. This is not
a wrong pronunciation, it is a broken string, and it would reach the TTS as one. The model
card does say *unvocalized* Hebrew, so this is documented rather than surprising — but it
is a hard constraint on where ReNikud can sit in the pipeline. Anything upstream that adds
nikud, including the DictaBERT diacritizer, must not feed it.

Strip the nikud before calling it, and the sentence is handled normally.

## Numbers are its strongest category

    בשנת 1948, בשעה 14:30, עלה השיעור ב-2.5 אחוזים.
    beʃnˈat ʔˈelef tʃˈa meʔˈot ʔaʁbaʔˈim veʃmˈone, beʃaʔˈa ʃtˈajim vaχˈetsi, ʔalˈa haʃiʔˈuʁ
    beʃnˈajim vaχˈetsi ʔaχuzˈim

1948 read as a year, 14:30 as half past two, 2.5 as two and a half, and a phone number read
digit by digit. That is a hand-written Hebrew number front end inside `renikud_onnx.py`
(`expand_year`, `expand_time`, `cardinal`, `noun_gender`), not the neural model — and it is
reusable on its own, whatever is decided about ReNikud.

## English is left alone

`The quick brown fox jumps over the lazy dog.` comes back identical, and `ב-Microsoft`
becomes `be-Microsoft`. ReNikud phonemizes Hebrew and passes everything else through, so
English reaches the TTS as Latin text. Whether that is right depends on what the model is
trained to read, but it is not ReNikud's problem to solve and it will not solve it.

## The strip works, and does not make vocalized input usable

`hebrew_training/renikud_frontend.py` removes the points before calling ReNikud and raises
if a Hebrew letter survives anyway. With it the forty sentences come back **0 failures**,
down from 6.

But the guard only stops the broken string. It does not make a vocalized sentence come out
right, and the homograph twins show why: stripping nikud from properly vocalized Hebrew
leaves *ktiv chaser*, the defective spelling, and ReNikud was trained on ktiv male.

    ktiv male   הספר סיפר לי שהוא קרא ספר על תספורות.
                hasapˈaʁ sipˈeʁ lˈi ʃehˈu kaʁˈa sˈefeʁ ʔˈal tispoʁˈot.      correct

    stripped    הספר ספר לי שהוא קרא ספר על תספרות.
                hasapˈaʁ sapˈaʁ lˈi ʃehˈu kaʁˈa sˈefeʁ ʔˈal tesapʁˈut.      סיפר -> "barber"

    ktiv male   בוקר טוב, אמר הבוקר...     bokˈeʁ tˈov
    stripped    בקר טוב, אמר הבוקר...      bakˈaʁ tˈov                      "cattle"

So the rule is not "strip and proceed". It is: **keep the unvocalized ktiv-male text and
send ReNikud that**. Nikud is for the models that want it; ReNikud wants the ordinary
spelling, and anything that has been through a diacritizer is the wrong input even after
the marks come off.

## Correction: vocalized sentences are out of the comparison

The two sections above are kept because they are the reason for this one, not because they
are results. Running ReNikud on vocalized text was the wrong test and the strip-then-run
advice was wrong with it.

ReNikud's job is to work out vowels that are not written. A sentence that already carries
nikud has them written down, so there is nothing to work out and stripping them throws away
the answer — which is exactly why the stripped sentences read worse.

`renikud_eval_set.py` now drops the `nikud` category and does not touch `text_nikud`. The
comparison is **34 sentences, 0 failures**.

The pipeline rule that follows is a router, not a cleaner:

    no nikud   -> ReNikud
    has nikud  -> nikud-to-IPA directly; ReNikud has nothing to add

with one gap worth naming: nikud does not mark stress, so `בֹּקֶר` cannot tell you *bóker*
from *bokér* on its own. That is the only part of the vocalized path that still wants a
model, and it is what Phonikud is for. `renikud_frontend.phonemize` keeps the strip as a
crash guard for text that reaches ReNikud by mistake; it is not the route anything should
take on purpose.
