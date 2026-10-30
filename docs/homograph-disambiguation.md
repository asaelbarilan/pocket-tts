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
