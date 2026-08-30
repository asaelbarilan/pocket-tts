# Hebrew Pocket TTS TODO

## Alignment gold set — BLOCKING

- [ ] **Build a 100-sentence hand-marked word-alignment gold set.** Nothing about alignment
  can be decided without it. Every ivrit.ai corpus (Knesset and CrowdRecital) carries
  timings from the same stable-ts pipeline, so measuring one against another reports
  agreement, not accuracy — the earlier "19 ms median word-end error" was exactly that
  mistake. No public Hebrew corpus with human word timings exists: ILSpeech has expert IPA
  but no timestamps, Pekar's segmented corpus was never released, MFA's benchmarks are
  English.
  - Source the sentences from Knesset plenums, the domain we train on. Optionally add a
    clean slice from ILSpeech as an upper bound.
  - Pre-fill each with an aligner and correct the boundaries rather than placing them from
    scratch — much faster, and `alignment_disagreement.py` already emits Praat TextGrids
    with both aligners on separate tiers and an empty `gold` tier to fill.
  - Choose which sentences to annotate by three-way disagreement, so the human time lands
    where the ranking is actually decided.
  - Then score all three columns against it: `imvladikon/wav2vec2-xls-r-300m-hebrew`,
    `MahmoudAshraf/mms-300m-1130-forced-aligner`, and the shipped stable-ts timings.
  - Report median and p90 boundary error in **Mimi frames (80 ms)**, not milliseconds, and
    report `last_word_end` separately since it alone drives the trailing-silence trim.

  Why it is blocking: on 112 clips from 10 recordings the three columns agree to 26-37 ms at
  the median, but 28% of word ends differ by more than one frame and 46% of `last_word_end`
  values differ by more than one frame between the Hebrew aligner and stable-ts. Medians look
  fine and the tail does not. Without a gold set there is no way to know which column is
  right, and `last_word_end` is what broke EOS on the first run.

- [ ] Measure the dead-word rate (`start == end`) on 40 random plenums. Current estimates
  come from 25.6 h of the corpus's 8,816 h and disagree threefold between samples: 12.7% on
  8 long recordings, 4.3% on 10 short ones. `clean_segments` already drops them; the open
  question is how much data that costs.

## Controlled evaluation

- [x] Build fixed, equal-sized evaluation groups for seen/unseen speakers and
  seen/unseen sentences, using fixed generation seeds.
- [ ] Add a difficult-Hebrew stress set covering numbers, abbreviations, foreign names,
  prefixes, punctuation, and long sentences.
- [x] Score WER and CER after applying identical Hebrew normalization to reference and ASR
  output.
- [x] Measure the ASR floor on genuine held-out recordings using the same evaluator:
  128 clips, WER 0.1231 (95% CI 0.1065-0.1411), CER 0.0698 (0.0559-0.0839).
- [x] Use equal clip counts and bootstrap confidence intervals across checkpoints.
- [ ] Add speaker-similarity and human listening checks.

## Training comparisons

- [x] Implement the paper's head batch multiplier.
- [x] Implement opt-in 75% flow matching / 25% Lagrangian Self-Distillation with JVP,
  stop-gradient targets, adaptive weighting, metrics, and checkpoint support.
- [x] Verify finite forward/backward behavior on released 6-layer and 24-layer models.
- [x] Compare from the same initialization:
  - flow only, head multiplier 1;
  - flow only, head multiplier 8;
  - FM/LSD 75/25, head multiplier 8.
  All three 6,000-step arms and controlled 128-clip scoring completed on 2026-08-06.
  Overall WER/CER: flow/m1 0.850/0.587, flow/m8 0.755/0.538, and reconstructed
  FM-LSD/m8 0.998/0.875. The flow/m8 confidence interval is better than flow/m1, while
  FM/LSD failed in all four seen/unseen groups. Preserve these runs; do not rerun FM/LSD
  without demonstrating and testing a concrete reconstruction correction first.
- [ ] Select checkpoints using fixed CER/WER and listening evaluation, not validation MSE.
- [x] Correct FM/LSD validation so it uses the configured head batch multiplier; this is a
  metrics-comparability defect and does not explain the failed generated speech.
- [ ] Verify the FM/LSD probability path, loss reduction, adaptive weighting, and backbone
  noise augmentation against Kyutai's unreleased trainer or author guidance. The paper and
  public Flow Maps reference leave implementation choices that cannot be resolved from the
  combined loss scale alone.
- [ ] Run the next flow/m8 duration experiment in a new versioned directory, with fixed
  intermediate CER/WER gates; 6,000 steps improved over m1 but remains far above the
  held-out ASR floor.

## 24-layer experiment

- [ ] Choose a released non-English 24-layer checkpoint while asking Kyutai whether the
  English teacher can be shared.
- [ ] Audit Hebrew tokenizer-piece overlap for each candidate.
- [ ] Recompute latents with the exact candidate `--base-language` in a separate directory.
- [ ] Fine-tune in a separate run directory and compare held-out intelligibility.
- [ ] Consider Section 4.7 latent-CFG teacher-to-student distillation only after a
  teacher-sized Hebrew model performs better. This is separate from LSD.

## Data quality

- [ ] Audit low-value spoken wiki boilerplate while preserving text/audio alignment.
- [ ] Survey clipping and peak amplitude across the full prepared dataset.
- [ ] Decide whether to rebuild after the slicer float-precision correction.
- [ ] Re-run speaker-leakage and prompt-pair checks after any rebuild.
