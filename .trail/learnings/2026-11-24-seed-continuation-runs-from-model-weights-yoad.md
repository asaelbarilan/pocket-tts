---
title: Seed continuation runs from model weights without inheriting old steps
author: Yoad
date: 2026-11-24
branch: yoad/semantic-fusion
id: l-90ff1210
scope: repo
status: active
src: agent
touches:
  - training/args.py
  - training/tests/test_semantic_training.py
  - training/train.py
  - training/checkpointing.py
---

## Decided
- For a short new curriculum that follows a completed 130k-step combined-semantic run, load the source checkpoint's raw model weights into a new run and reset its optimizer, EMA and step counter. Otherwise `training.train` auto-resumes at 130k and fast-forwards past the entire 1,248-step curriculum; warmup cannot occur.
- Keep the new optimizer's 500-step warmup on the previous run's final seven-dataset mix before introducing the synthetic numeric dataset, so a changed optimizer and a new data source are not introduced at the same step.
## Tried and ruled out
- Pointing the new config at the old run directory: it implicitly restores step 130k and optimizer, so no 500-step warmup or new curriculum would execute.
- Using the source run's `model.safetensors` export as a shortcut: it contains EMA weights, not the raw model weights saved in the named checkpoint; the user asked to start from the latest checkpoint.
## Watch out
- An existing checkpoint in the new run directory takes precedence over the initial source checkpoint for safe resume. Launch the initial run in a new, empty directory; use the exact combined semantic encoder pair and matching tokenizer.
In the code: training/args.py init_checkpoint; training/train.py setup; training/configs/finetune_hebrew_numeric_15k.yaml; training/checkpointing.py exports EMA while checkpoint contains raw weights.
