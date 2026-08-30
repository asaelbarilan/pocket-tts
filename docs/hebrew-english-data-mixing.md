# English, numbers and dates in a Hebrew TTS: what to add and how much

**NO CODE CHANGED.** Research note with measurements. Papers in
`../papers/code-switching/` (gitignored, links at the bottom).

Three questions were asked: which datasets to train the tokenizer on, what to add to the
training audio, and whether synthetic number/date data is needed. Measuring first changed
two of the three answers.

---

## 1. Numbers and dates are already solved. Do not build a synthetic set.

The normalizer in `hebrew-tts-data-tools` converts them to spoken Hebrew before the text
ever reaches the model. Run against it:

```
ב-1995 היו 3 ישיבות        -> ב אלף תשע מאות תשעים וחמש היו שלוש ישיבות
הישיבה תתקיים ב-15 במרץ 2024 -> הישיבה תתקיים ב חמש עשרה במרץ אלפיים עשרים וארבע
הוועדה תתכנס ב-10:30        -> הוועדה תתכנס ב עשר וחצי בבוקר
ב-1.1.2020 נכנס לתוקף       -> ב הראשון לראשון אלפיים ועשרים נכנס לתוקף
אחוז ההצבעה היה 67.4%       -> אחוז ההצבעה היה שישים ושבעה נקודה ארבע אחוז
טלפון 03-1234567            -> טלפון אפס שלוש אחת שתיים שלוש ארבע חמש שש שבע
התקציב הוא 2.5 מיליארד שקל  -> התקציב הוא שתיים וחצי מיליארד שקל
```

Years, dates, times, decimals, percentages and phone numbers all expand. The model therefore
never sees a digit, and cannot mispronounce one. **A synthetic numbers/dates corpus would
teach it to read a form it will never be given.**

Two conditions, and one bug:

- The same normalizer **must run at inference**. If it does not, the model meets digits it
  has never seen. This is a deployment requirement, not a data requirement.
- Knesset speech already supplies the spoken forms in abundance: 1.09% of words carry a
  digit before normalization (1,599 of 147,192), so the expanded forms are well attested.
- One real defect found: `סעיף 12(א)(3) לחוק` -> `סעיף 12(א)(3 לחוק`. A closing parenthesis
  is dropped and `12` is left unexpanded. Worth a fix in the normalizer, not a dataset.

**Effort saved: an entire synthetic corpus.** Spend it on the normalizer's edge cases
instead, which is where the residual risk is.

---

## 2. English is a real gap. The corpus contains almost none.

Measured over the same 147,192 words of Knesset transcripts:

```
words containing Latin : 172   (0.12%)
distinct Latin words   : 99
most common            : is, the, to, This, you, Israel., for, welcome, MRI, CT, BDS
```

One word in 800. Ninety-nine distinct types. Whatever the model learns about English from
this corpus, it learns from ninety-nine words — and the normalizer passes Latin through
untouched (`חברת Microsoft הודיעה` is unchanged), so it reaches the model as-is.

### How much English text the tokenizer needs — measured, not guessed

Trained 4,000-piece SentencePiece BPE models on the real normalized Knesset text with
varying amounts of FineWeb English mixed in, then measured tokens-per-character on held-out
text of each language. Lower is better; 1.0 means one token per character, i.e. total
degeneration to byte fallback.

| English share of tokenizer text | Hebrew tok/char | English tok/char | Hebrew pieces |
|---|---|---|---|
| **0%** | **0.316** | **0.977** | 3,678 |
| 5% | 0.322 | 0.476 | 3,315 |
| 10% | 0.325 | 0.429 | 3,087 |
| 20% | 0.333 | 0.386 | 2,726 |
| 35% | 0.345 | 0.358 | 2,260 |

Read the first row: a tokenizer trained on Hebrew alone spends **0.977 tokens per English
character** — exactly the failure mode the released English tokenizer has on Hebrew, in
reverse. `Microsoft` becomes nine tokens.

**Five percent buys most of the fix.** English more than halves (0.977 -> 0.476) while
Hebrew degrades by 1.9% (0.316 -> 0.322). Ten percent gains a little more English for 2.8%
Hebrew. Past that, English improves slowly while Hebrew keeps paying.

**Recommendation: 5-10% English text in the tokenizer corpus.** Use 10% if you expect
English-heavy input, 5% otherwise.

### Two things that would waste the measurement

- **Normalize the added text with the same normalizer** before training the tokenizer. The
  Hebrew side of the table above is normalized Knesset text, because that is what the model
  is trained on. Feeding raw web text would allocate pieces to digits and unnormalized forms
  that never appear in a manifest.
- **The tokenizer must be trained on the final manifest text**, which
  `train_tokenizer.py` already does by reading the jsonl directly. Adding the English
  sample means concatenating it as a plain-text file alongside — the script accepts both.

### Where to get the text

| source | use | note |
|---|---|---|
| the training manifest itself | the Hebrew side | already what `train_tokenizer.py` reads |
| `HuggingFaceFW/fineweb` (`sample-10BT`) | the English 5-10% | ungated; a few MB is plenty for 4,000 pieces |
| `HuggingFaceFW/fineweb-2` (`heb_Hebr`) | more Hebrew, if wanted | ungated, modern Hebrew; `hbo_Hebr` is Biblical, not what you want |

A 4,000-piece vocabulary saturates on a few MB. There is no reason to stream hundreds of GB
of FineWeb for this.

---

## 3. English *audio* — probably far less than the literature suggests

`2023_code-mixed-tts-low-resource.pdf` is the closest published analogue: Hindi-English
code-mixed TTS from **monolingual data only**. Their finding is the useful one —

> single script bi-lingual training without any code-mixing works well for pure code-mixed
> test sets

They used roughly **65% Hindi / 35% English** and needed **no code-mixed recordings at all**,
which is what makes this tractable: you do not have to find Hebrew-English code-switched
speech, only English speech.

But 35% is probably far more than this project needs, for a reason specific to our setup:

**We warm-start from `english_2026-04_24l`, an English model.** The 24 backbone layers
already know English acoustics; `reset_text_embedding` discards only the text table. So the
question is not "can the model produce English phonemes" — it could before we started — but
"does the new Hebrew tokenizer's English pieces map onto what the backbone already knows".
That is a much smaller thing to learn than English from scratch.

So: start at **5% English audio, or none at all**, and measure. Concretely, add an
English-in-Hebrew stress set to the eval (the `TODO.md` item "difficult-Hebrew stress set
covering numbers, abbreviations, foreign names") and score the finetune with no English
audio first. Add English audio only if that set fails.

If it is needed, LJSpeech (24 h) or a slice of HiFiTTS-2 is the obvious source, and
HiFiTTS-2 is already wired into Kyutai's `prepare_data.py`.

---

## 4. Nikud, and the objection that nobody will run the normalizer

The objection is correct as stated: training only on normalized text makes the normalizer a
**hard inference-time dependency**. Anyone who calls the model with raw text gets digits it
has never seen. That is a deployment failure mode, not a data one, and it is worth removing.

Sort the three problems by whether they actually need the normalizer.

| problem | needs the normalizer? | why |
|---|---|---|
| numbers, dates, times | **yes, or training data that covers raw digits** | `1995` has no pronunciation in its glyphs. Something must decide between "אלף תשע מאות תשעים וחמש" and "תשעים וחמש" |
| nikud | **no** | stripping is a regex over `֑-ׇ` and cannot fail |
| English words | **no** | the normalizer passes Latin through untouched anyway; this is a tokenizer and audio question |

So only numbers genuinely depend on it, and even that is escapable.

### Make the normalizer optional: emit both transcripts

Write each manifest row twice against the same audio — once with the raw text, once
normalized. The model then learns both mappings and the normalizer becomes a quality
improvement rather than a requirement.

This is safe in a way it would not be for ASR. Two texts pointing at one audio is
**many-to-one**, which is benign for TTS: both forms are valid inputs that should produce
that speech. The reverse — one audio with two possible transcripts — is what would be
ambiguous, and that is the ASR direction, not ours.

Is there enough raw-digit data to learn from? The corpus carries **1.09% digit-bearing
words**, which over 53.2M words is roughly **580,000 examples** of a digit string paired with
its spoken form. That is a lot of supervision for a narrow mapping.

Cost: the manifest doubles in rows (not in audio, which is unchanged). `--emit-raw` would be
a small addition to `build_official_manifest.py`.

### Nikud: strip it, and consider training with it

Measured on the corpus: **31 of 147,192 words carry nikud (0.021%)**. The model will
effectively never have seen it. And the normalizer does not remove it — verified:
`שָׁלוֹם עוֹלָם` passes through unchanged, and even `בְּ-1995` keeps its nikud while expanding
the number.

So today, a user who types vocalized Hebrew hands the model characters outside its
experience, which fall to byte fallback and probably produce noise.

The cheap fix is to strip nikud on the way in:

```
שָׁלוֹם עוֹלָם            ->  שלום עולם
הַיּוֹם הַוַּעֲדָה תִּתְכַּנֵּס  ->  היום הועדה תתכנס
```

One regex, cannot fail, and gives back exactly the unvocalized form the model was trained
on. **Do this unconditionally, in the model wrapper rather than the normalizer**, so it
holds whether or not anyone runs normalization.

But note what it costs. Nikud is not decoration — it is precisely the vowel information
Hebrew's abjad omits, and a user who types it is *volunteering the disambiguation the model
otherwise has to guess*. Stripping throws that away. Training on a vocalized fraction so the
model can exploit nikud when present is a real improvement, and needs a diacritizer to
produce the vocalized text: `2026_ReNikud_hebrew-g2p.pdf` and Phonikud (Interspeech 2026,
code released) are the current Hebrew options. Treat that as a later experiment, after
stripping removes the crash.

---

## Deliberately not pursued

- **Synthetic numbers and dates.** The normalizer removes the problem before the model sees
  it. See section 1.
- **Transliterating English into Hebrew letters** (the Hindi-English paper's "common script"
  trick, Roman -> Devanagari). It would remove the need for Latin in the tokenizer entirely,
  and it costs a transliteration step at inference that can fail, plus it makes the model
  unable to read Latin input directly. Reconsider only if the 5-10% tokenizer mix proves
  insufficient.
- **Collecting real Hebrew-English code-switched speech.** The literature says monolingual
  data of both languages suffices for code-mixed test sets, so this expensive path is not
  the first thing to try.
- **Large-scale FineWeb ingestion.** A 4,000-piece tokenizer saturates on megabytes.
- **Phoneme input instead of graphemes.** Phonikud converts Hebrew to IPA and would unify
  nikud, no-nikud and numbers into one representation. It is the most principled answer and
  the largest change: a new text vocabulary, a new inference dependency, and unknown quality
  on Knesset text. Revisit only if grapheme input plateaus.

---

## Sources

- [Code-Mixed Text to Speech Synthesis under Low-Resource Constraints](https://arxiv.org/html/2312.01103v1)
- [Improving Low Resource Code-switched ASR using Augmented Code-switched TTS](https://arxiv.org/pdf/2010.05549)
- [Improve Cross-lingual Voice Cloning Using Low-quality Code-switched Data](https://arxiv.org/pdf/2110.07210)
- [Multilingual Transfer Learning for Code-Switched Language and Speech Neural Modeling](https://arxiv.org/pdf/2104.06268)
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) and [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
