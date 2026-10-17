# English, numbers and dates in a Hebrew TTS: what to add and how much

Research note with measurements (sections 1-5), then how the tokenizer tooling in this
repo implements the decision (section 6). Papers in `../papers/code-switching/` (gitignored,
links at the bottom).

> **Implemented in:** `hebrew_training/build_tokenizer_corpus.py` (mix the text),
> `hebrew_training/train_tokenizer.py` (fit the tokenizer), `hebrew_training/tokenizer_charset.py`
> (the character-set safety filter) and `hebrew_training/configs/tokenizer_hebrew.yaml`. See
> [section 6](#6-building-it-here). Sections 1-5 are the measurements that chose the defaults and
> predate the tooling, so their wording ("the script accepts both") describes the experiments.

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
- **The tokenizer must be trained on the final manifest text**, i.e. the manifest's
  `transcript` field read directly. Adding the English sample means giving
  `train_tokenizer.py` a second `--input`, or letting `build_tokenizer_corpus.py` mix them.

### Where to get the text

| source | use | note |
|---|---|---|
| the training manifest itself | the Hebrew side | a `train.jsonl` is read as-is |
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

### Building the English manifests (about 1,000 h)

Two builders write the same row format as `build_official_manifest`, each into a folder that
holds everything training reads (rows, per-utterance 24 kHz FLAC, no network access needed):

```bash
# diverse half: ~750 h, ~56 GB kept (+~48 GB of MP3 traffic, deleted chapter by chapter)
python -m hebrew_training.build_hifitts2_manifest --out-dir data/english/hifitts2 --hours 750
# clean half: ~250 h, ~23 GB kept (clean subsets only; the pool holds ~240 h after filtering)
python -m hebrew_training.build_libritts_r_manifest --out-dir data/english/libritts_r --hours 250
```

Why 75/25: the clean LibriTTS-R subsets hold only ~245 h (the 310 h "other" subset is the noisy
half), while HiFiTTS-2 has ~5k speakers and long utterances that cut well. `--hours` is the TRAIN
target; raising it later reuses every file already downloaded. Changing the audio format cannot
reuse them: the script says so and asks before deleting and rebuilding. HiFiTTS-2 reads Kyutai's
published alignments (`kyutai/hifitts2-aligned`) and keeps rows with `src_wer <= 0.1`;
LibriTTS-R takes word times from the MFA TextGrids in `cdminix/libritts-r-aligned` and maps them
onto the corpus's cased, punctuated text. Their docstrings give the rest, including the
download sizes (metadata ~6 GiB for HiFiTTS-2, ~130 MB for LibriTTS-R).

Mixing with Hebrew is done in the data config: list each manifest under `data.datasets` and
give the share of each per phase in `data.curriculum`, see
[Mixing datasets](../training/README.md#mixing-datasets-a-training-curriculum). (Before that
existed the trainer read one `train_jsonl`, so manifests were combined with `cat`, repeating a
file to weight it.)

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

### Nikud: put it in the tokenizer. It is free if you raise the vocabulary.

Stripping nikud makes vocalized input *work*, but it throws away exactly the vowel
information Hebrew's abjad omits — a user who types nikud is volunteering the disambiguation
the model otherwise has to guess. If the model should **use** nikud, the marks have to be in
the tokenizer and in the training text.

Two facts make that harder than it sounds:

- The corpus has none: **31 of 147,192 words** carry nikud (0.021%).
- Neither does the web. A FineWeb-2 `heb_Hebr` sample of 263,701 words is **0.11% vocalized**
  — 15 distinct marks, and no usable volume.

So the vocalized text has to be generated. `dicta-il/dictabert-large-char-menaked` (14k
downloads) does it well on Knesset text:

```
הוועדה תתכנס ביום שלישי הקרוב בשעה עשר בבוקר
  ->  הַוַּעֲדָה תִּתְכַּנֵּס בְּיוֹם שְׁלִישִׁי הַקָּרוֹב בְּשָׁעָה עֶשֶׂר בַּבֹּקֶר
```

`Phonikud/phonikud` (Interspeech 2026) is the other option and goes further, to IPA.

#### What nikud costs the tokenizer — measured

Diacritized 6,000 lines of real normalized Knesset text with DictaBERT, then trained
4,000-piece tokenizers on different corpora and measured tokens-per-character on held-out
plain, vocalized and English text:

| tokenizer corpus | plain | nikud | english | nikud pieces |
|---|---|---|---|---|
| plain only | **0.314** | **0.936** | 0.962 | 8 |
| plain + nikud (50/50) | 0.378 | 0.270 | 0.998 | 2,212 |
| nikud only | 0.869 | 0.234 | 0.998 | 3,648 |
| plain + nikud + 5% en | 0.384 | 0.275 | 0.485 | 2,010 |

Row one is the confirmation: **a plain-only tokenizer spends 0.936 tokens per character on
vocalized text.** Nikud is byte-fallback garbage unless it is in the vocabulary. And row
three shows the opposite mistake — training only on vocalized text wrecks plain input
(0.869), so it has to be both.

Supporting all three costs plain Hebrew about 22% at vocab 4,000 (0.314 -> 0.384). That is
the real objection to doing it.

#### Raising the vocabulary removes the cost entirely

`n_bins` is not fixed at 4,000; it just has to match the tokenizer. Same corpus
(plain + nikud + 8% English), swept:

| vocab | plain | nikud | english |
|---|---|---|---|
| 4,000 | 0.386 | 0.277 | 0.452 |
| 6,000 | 0.350 | 0.250 | 0.397 |
| **8,000** | **0.330** | **0.236** | **0.360** |
| 12,000 | 0.311 | 0.217 | 0.308 |

At **8,000 pieces the three-way tokenizer matches the plain-only 4,000 baseline on plain
Hebrew** (0.330 vs 0.314) while handling nikud and English properly. At 12,000 it beats it.

The cost is one tensor: the text table goes from 4,097,024 to 8,193,024 parameters, 15.6 MB
to 31.3 MB in fp32 — **1.3% of a 1.24 GB checkpoint**.

And it is compatible with the finetune path. `build_models()` in `training/modules/builders.py` drops `conditioner.embed.*`
before loading under `reset_text_embedding: true`, then loads with `strict=False`, so the
fresh table can be any size. Changing `n_bins` does not conflict with warm-starting from the
released 24-layer teacher.

#### Recommendation

```
train_tokenizer.py --vocab-size 8000
  corpus: normalized manifest text
        + the same text diacritized with dictabert-large-char-menaked
        + ~8% English from FineWeb
model config: flow_lm.lookup_table.n_bins: 8000
```

And train on both plain and vocalized transcripts against the same audio, exactly as with
raw/normalized numbers — same many-to-one argument, same benefit: the model then handles
whichever form the user types.

---

## 5. The four-way mix, measured

The sections above tested pairs. This is the combination actually proposed, so that nothing
is assumed to compose: Knesset manifest text + 50% general Hebrew + a diacritized copy of
both + 15% English, all normalized the same way.

Tokens per character on five held-out sets:

| vocab | knesset | general he | nikud | gen nikud | english |
|---|---|---|---|---|---|
| 4,000 | 0.350 | 0.394 | 0.310 | 0.353 | 0.483 |
| **8,000** | **0.308** | **0.349** | **0.268** | **0.303** | **0.414** |
| 12,000 | 0.290 | 0.328 | 0.244 | 0.278 | 0.391 |

Baseline for comparison, a Knesset-only tokenizer at 4,000: knesset 0.314, nikud 0.936,
english 0.977.

**Nothing is crowded out.** At 8,000 the four-way tokenizer encodes Knesset text at 0.308 —
slightly *better* than the specialist Knesset-only tokenizer's 0.314 — while also handling
general Hebrew, vocalized Hebrew of both kinds, and English. The specialist wins nothing.

Caveat: this was measured on a 0.49 MB corpus. Real tokenizer training would use far more
text, which only helps; the relative ordering is what transfers.

### The decision

```
--vocab-size 8000
flow_lm.lookup_table.n_bins: 8000
corpus = normalized manifest text
       + ~50% general Hebrew (FineWeb-2 heb_Hebr, normalized)
       + a diacritized copy of both (dictabert-large-char-menaked)
       + ~15% English (FineWeb)
```

---

## 6. Building it here

```bash
# 1. mix the text (charset-filtered), then fit the tokenizer on it
python -m hebrew_training.build_tokenizer_corpus \
    --config hebrew_training/configs/tokenizer_hebrew.yaml --train

# 2. use it: the trainer reads n_bins from the file, there is nothing else to edit
cp artifacts/tokenizer/tokenizer.model tokenizers/hebrew.model
uv run torchrun --nproc-per-node 8 training/train.py training/configs/finetune_hebrew.yaml
```

### The vocabulary size is one number, set once

`--vocab-size` (or `vocab_size:` in the tokenizer config) is the only place it is chosen. The
default is 8000, which is the section 4-5 decision. `training/modules/builders.py` reads
`flow_lm.lookup_table.n_bins` from the tokenizer file whenever `tokenizer_path` is a local
file, so it overrides whatever the model config says (and logs when it does). That matters
beyond finetuning: `depth_distill_hebrew.yaml` builds a 6-layer student from
`english.yaml`, which still says 4000, and would otherwise fail the
`nbins == sp.vocab_size()` assertion. Released `hf://` tokenizers keep their config value.

The inference-side configs (`pocket_tts/config/hebrew.yaml`, and the `hebrew.yaml` that the
legacy `hebrew_training/export.py` writes) still carry an `n_bins`, because `pocket_tts`
asserts it equals the tokenizer's size. `export.py` writes it from the tokenizer;
`pocket_tts/config/hebrew.yaml` is a template and must be edited by hand if the vocabulary
changes.

### Choosing sources

`sources:` in the config has four entries. Each has a `pct` (share of **characters**; the
non-zero ones must add to 100), and `pct: 0` leaves a source out entirely, with nothing else
configured for it:

| source | what it is | where it reads from |
|---|---|---|
| `manifest` | the model's own training text | `path`: a `tokenizer_corpus.txt`, a `train.jsonl`, or a folder |
| `hebrew` | general modern Hebrew | `path`, or `download: true` for FineWeb-2 `heb_Hebr` |
| `hebrew_nikud` | vocalized Hebrew | `path` of supplied text, or `diacritize: true` with `diacritize_sources: [manifest, hebrew]` |
| `english` | general English | `path`, or `download: true` for FineWeb `sample-10BT` |

Every `path` is a local file or folder (`.txt`, `.jsonl` with a `text` or `transcript` field,
`.parquet`, each optionally `.gz`), read in sorted order, so a FineWeb dump downloaded
yourself works unchanged. **Nothing is downloaded unless you opt in** with `download: true`
or `--download hebrew|english`; a source with a share but neither a path nor `download` stops
the run before any work, naming the flag that would fix it. Downloaded and diacritized slices
are cached under `cache_dir`, so a re-run resumes. The streamed Hebrew and English slices
are the first rows of each dataset, which for a few MB is fine; for a more even sample,
download shards yourself and point `path` at them.

`diacritize` runs `diacritizer.model` (default in the sample config:
`dicta-il/dictabert-large-char-menaked`). That is a model download that executes the model
repo's own code (`trust_remote_code`), which is why the model id is written in the config and
not defaulted in code; a local directory works too. It is the slow step and wants a GPU. It
vocalizes extra text from the requested source pools, balanced by input characters. The run stops if the first
batch comes back with under 5% niqqud, since a wrong tokenizer for this model fails silently
by returning plain text.

Shares and everything else can also be overridden on the command line without editing the
config:

```bash
python -m hebrew_training.build_tokenizer_corpus --config hebrew_training/configs/tokenizer_hebrew.yaml \
    --pct manifest=20 --pct hebrew=50 --pct hebrew_nikud=0 --pct english=30 \
    --path hebrew=/data/fineweb2-heb --path english=/data/fineweb-en --vocab-size 12000 --train
```

The run prints the shares it actually got beside the ones it asked for and names any source
that ran short. The manifest slice is the one that has to be rebuilt whenever the training
corpus changes.

The default shares (manifest 20 / hebrew 40 / niqqud 20 / english 20) are a heavier English
and general-Hebrew mix than section 5's measured one. Treat them as a starting point and compare
tokens per character on held-out text of each kind before committing to a long run;
`train_tokenizer.py` prints it for one sentence of each kind and how many fell back to bytes.

For bounded seeded sampling, held-out evaluation and an actual corpus-size sweep, see
[the tokenizer experiment](TOKENIZER_EXPERIMENT.md). Its generated artifacts stay outside Git.

### The character-set safety filter

Everything that reaches SentencePiece passes `hebrew_training/tokenizer_charset.py`, in both
the corpus builder and `train_tokenizer.py` (`--no-charset-filter` to disable it there).
It keeps:

- **latin**: `A-Z a-z` and accented Latin letters (`é ü ø ß` ...), kept as they are.
- **hebrew**: the letters `א-ת` and Hebrew punctuation.
- **niqqud**: vowel points, dagesh, shin/sin dots, meteg, rafe, qamats qatan.
- **cantillation**: the Hebrew cantillation marks (trope, U+0591-U+05AF), kept as they are.
- **digits**: `0-9`.
- **punctuation**: ASCII punctuation, typographic quotes/dashes/ellipsis, and `% $ € ₪ £ ¥ °`.

Zero-width and bidi controls, other control characters and emoji are removed rather than
judged. Anything else is *foreign*, and by default drops the whole line: stripping the Cyrillic
out of a Russian sentence leaves garbage that is worse than no sentence. `on_foreign: strip`
removes just the character. Categories can be switched off with `allow:` / `--allow`; the
filter prints how many lines it dropped and the most common foreign characters, which is the
quickest way to see what a source is really made of.

The allowed set is derived from what the normalizer in `hebrew-tts-data-tools`
(`hebrew_tts_normalizer.py`) accepts: Hebrew letters, `. , ; : ? ! … " ' - ( ) [ ] { }`, digits
and Latin. It is wider in three places the normalizer does not cover: niqqud (which it neither
rejects nor treats specially), digits kept as digits (it expands them to words, so a normalized
manifest has none, but web text and a user typing digits at inference do), and a few currency
and degree symbols. With `fold` on (the default) typographic quotes, Unicode dashes, maqaf,
geresh and gershayim become the same plain `"` `'` `-` that the normalizer's character
conversion produces, so the tokenizer sees what a normalized manifest carries. Turn `fold`
off only if the manifest text is not normalized.

What the filter does **not** do: run the number-expansion normalizer. Added web text keeps its
digits and abbreviations as written, which section 2 flagged as the thing to normalize. If
that matters for the mix you build, normalize the text before passing it in as a `path`.

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
