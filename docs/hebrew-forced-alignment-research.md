# Word-level forced alignment for Hebrew

**NO CODE CHANGED.** This is a research note.

The eight papers are downloaded to `../papers/hebrew-alignment/` with a README saying what
each one gave us — but `papers/` is gitignored, so that directory is local to whoever ran
this. Every source is linked at the bottom; re-fetch them with those links.

Written because the collaborator who produced the ivrit-ai Knesset alignments says they are
not accurate, and he would know — he made them. This checks that claim against the data,
then asks what to use instead.

---

## 1. The claim checks out, and the file says so itself

`transcript.aligned.json` in `ivrit-ai/knesset-plenums` carries these top-level keys:

```
text, segments, language, ori_dict, regroup_history, nonspeech_sections
```

`regroup_history` and `nonspeech_sections` are **stable-ts** fields
([jianfch/stable-ts](https://github.com/jianfch/stable-ts)). So the timings were produced by
Whisper plus stable-ts post-processing, not by forced alignment.

That matters because the two are different in kind:

- **Forced alignment** is given the text and finds where each word is. It solves an
  alignment problem.
- **Whisper timestamping** reads timings out of the decoder's cross-attention via DTW, then
  nudges them with silence heuristics. The timings are a by-product of decoding.

stable-ts's own documentation concedes the starting point: *"initial timestamps are not
pinpoint accuracy"* — silence suppression exists precisely to paper over that. And
per the whisper-timestamped documentation, cross-attention weights differ between Whisper
model sizes, so **the same audio and the same transcript can yield timestamps differing by
100–400 ms** depending only on which model ran.

`2025_Yeh_whisper-internal-word-aligner.pdf` explains the mechanism: only *some* attention
heads align well, and averaging DTW over all of them dilutes the good ones.
`2024_CrisperWhisper_accurate-timestamps.pdf` shows the fix requires retraining Whisper with
an alignment-specific loss — which means the shipped timings cannot be repaired in
post-processing. They have to be replaced.

### Why this matters here specifically

`training/dataloader.py` uses word times for two things, and both degrade with sloppy
timings:

- it cuts each utterance at a word boundary, prompt before / target after. A boundary off by
  200 ms puts part of the next word in the prompt and clips it from the target.
- it trims to `last_word_end + 0.2 s`. Kyutai name a late `last_word_end` as the cause of a
  model that "emits silence instead of EOS, so generations never terminate" — which is
  exactly the failure the first Hebrew run had.

Kyutai's own diagnostic list puts it plainly: *"If your model is generating speech that cuts
off part of the first or last word: the issue is probably with your alignment."*

---

## 2. Options, ranked

Availability checked, not assumed.

| # | Approach | Hebrew support | Effort | Expected effect |
|---|---|---|---|---|
| **1** | **`align_hebrew.py` + `imvladikon/wav2vec2-xls-r-300m-hebrew`** — CTC forced alignment over native Hebrew characters, wrapping Kyutai's `align_data.py` | native; **measured** | **none — already built and tested** | high |
| 2 | `ctc-forced-aligner` + `MahmoudAshraf/mms-300m-1130-forced-aligner` — CTC forced alignment, uroman romanization | 1130+ languages incl. Hebrew | low — pip install, one command | high; useful as a cross-check |
| 3 | CrisperWhisper — Whisper retrained for verbatim timestamps | multilingual, Hebrew unverified | medium | medium; still Whisper-family |
| 4 | MFA 3.0 — HMM, the accuracy leader at **<15 ms** mean boundary error | **none. No Hebrew acoustic model and no pronunciation dictionary** | very high — train an acoustic model and build a dictionary from scratch | unknown |
| 5 | Learned DP over MMS + UnSupSeg (arXiv 2606.10675) — beats MFA and MMS, **evaluated on Hebrew** | yes, in the paper | **not available — no code released** | unknown |

### Recommendation: option 1, which we already have

`hebrew-tts-data-tools/data_prep/align_hebrew.py` exists, and was measured against
CrowdRecital's human-recital timings on 40 utterances / 316 words:

```
word END   error: median 19 ms, p90 71 ms
word START error: median 74 ms, p90 158 ms
5.7% of words left untimed
```

Put that next to the two reference points this research turned up:

- MFA 3.0, the state of the art, reports **<15 ms** mean boundary error (English/Japanese/Korean).
- Whisper DTW timestamps vary by **100–400 ms** between model sizes alone.

So our aligner is within a few milliseconds of the best published system, and roughly an
order of magnitude better than what the corpus ships. One frame at Mimi's 12.5 Hz is 80 ms;
19 ms is a quarter of a frame, and `last_word_end` — the trailing-silence trim — is the
accurate side.

**Action: re-align the Knesset corpus rather than trusting `transcript.aligned.json`.**
Keep the shipped `text` as the transcript (it is Whisper refined against the official Knesset
protocol, which is the better text) and recompute only the word times.

Cost: at the ~75x realtime measured earlier, 4,000 h is ~53 GPU-hours — about 7 h wall clock
on the 8-GPU box, or roughly $15 on an L4 spot instance. Against a multi-day training run
that is noise, and `align_data.py` already shards across GPUs.

Run option 2 on a few hundred utterances as a second opinion before committing 53 GPU-hours.
If two independently-trained CTC models agree to within a frame, the timings are sound; if
they disagree, that is worth knowing before training on either.

---

## 3. The Hebrew-specific limit both CTC options share

Written Hebrew is an abjad: vowels are normally omitted, so the grapheme string
underdetermines the pronunciation. A character-level CTC head is aligning a consonant
skeleton, and `2024_Roth_diacritic-free-hebrew-tts.pdf` is entirely about what that costs a
Hebrew TTS.

This bounds both option 1 (native Hebrew characters) and option 2 (uroman romanization,
which also drops vowels). It is not a reason to prefer one over the other, and it is not a
reason to keep the stable-ts timings. It is a ceiling to be aware of: a phoneme-level
aligner over vocalized text would in principle do better, which is what
`2026_ReNikud_hebrew-g2p.pdf` points toward, but no such pipeline exists off the shelf for
Hebrew today.

---

## 4. Deliberately not pursued

- **MFA.** It is the most accurate published option and it is unavailable: no Hebrew
  acoustic model, no Hebrew pronunciation dictionary. Getting there means training an
  acoustic model and building a dictionary — a project, not a step.
- **Repairing the shipped timings.** CrisperWhisper's result is that accurate Whisper
  timestamps need a retrained model, so post-processing what ivrit-ai shipped cannot recover
  what was lost.
- **arXiv 2606.10675.** Best reported numbers on Hebrew of anything found, and no code. Re-check later.
- **Re-transcribing the audio.** The transcripts are refined against the official Knesset
  protocol and are the strongest part of this corpus. Only the *timings* are in question.

---

## Sources

- [Tradition or Innovation: A Comparison of Modern ASR Methods for Forced Alignment](https://arxiv.org/pdf/2406.19363)
- [Montreal Forced Aligner and the state of speech-to-text alignment in 2026](https://arxiv.org/pdf/2606.18466)
- [Multilingual Word-Level Forced Alignment with Self-Supervised Representations and Learned Dynamic Programming](https://arxiv.org/pdf/2606.10675)
- [Whisper Has an Internal Word Aligner](https://arxiv.org/pdf/2509.09987)
- [CrisperWhisper: Accurate Timestamps on Verbatim Speech Transcriptions](https://www.isca-archive.org/interspeech_2024/zusag24_interspeech.pdf)
- [A Language Modeling Approach to Diacritic-Free Hebrew TTS](https://www.isca-archive.org/interspeech_2024/roth24_interspeech.pdf)
- [jianfch/stable-ts](https://github.com/jianfch/stable-ts)
- [MahmoudAshraf97/ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- [MFA pretrained acoustic models](https://mfa-models.readthedocs.io/en/latest/acoustic/index.html) — Hebrew absent
- [torchaudio CTC forced alignment API](https://docs.pytorch.org/audio/2.8/tutorials/ctc_forced_alignment_api_tutorial.html)
