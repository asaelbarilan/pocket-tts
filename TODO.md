# Hebrew Pocket TTS TODO

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
- [ ] Compare from the same initialization:
  - flow only, head multiplier 1;
  - flow only, head multiplier 8;
  - FM/LSD 75/25, head multiplier 8.
  `run_hebrew_v3_ablation.ps1` automates all three 6,000-step arms and controlled scoring.
- [ ] Select checkpoints using fixed CER/WER and listening evaluation, not validation MSE.

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
