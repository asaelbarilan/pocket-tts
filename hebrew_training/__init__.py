"""Hebrew-specific tooling around Kyutai's Pocket TTS trainer.

Not everything here is live. The project began as a reimplementation of the trainer, before
Kyutai released theirs; those modules are kept so old runs stay reproducible, but nothing in
docs/HEBREW_TRAINING_RECIPE.md uses them and new work should not either.

USE THESE -- the recipe depends on them:
    fetch_knesset.py            download a subset of an ivrit-ai Knesset corpus
    build_official_manifest.py  corpus -> the jsonl Kyutai's DataLoader reads
    score_asr_floor.py          the ASR's own error on real human audio, i.e. the WER floor
    score_wer.py                score a run's samples (reads Kyutai's layout and ours)
    build_eval_set.py           build the fixed eval set those two read
    evaluation.py               shared normalization and scoring helpers
    build_data_report.py        optional corpus dashboard
    server_launcher.py          optional, serves the HTML dashboards

SUPERSEDED by upstream -- do not build on these:
    train.py, lsd.py, smoke_test.py, validate_lsd_compatibility.py
                                -> training/train.py and its flow modules
    prepare_data.py, prepare_data_v2.py
                                -> build_official_manifest.py
    precompute_latents.py       -> Kyutai's DataLoader reads audio directly, no latent cache
    train_tokenizer.py          -> training/scripts/train_tokenizer.py
    warm_start_checkpoint.py    -> `reset_text_embedding: true` (upstream 8c98c9b)
    model_utils.py              install_tokenizer is superseded by that same flag; the rest
                                of the module is still imported by the scripts above

TIED TO OUR OLD CHECKPOINT FORMAT -- will not read Kyutai's runs as-is:
    generate_samples.py, eval_checkpoints.py, export.py
"""
