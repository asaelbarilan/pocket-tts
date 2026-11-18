---
title: Recycle Viter GPU workers and retry crashed batches in smaller chunks
author: Yoad
date: 2026-11-18
branch: yoad/semantic-fusion
id: l-13ca5a10
scope: repo
status: active
src: agent
touches:
  - hebrew_training/align_official_manifest.py
  - tests/test_align_official_manifest.py
---

## Decided
- Give each GPU alignment worker its own task queue so the coordinator knows which batch was in flight on a crashed worker, recycle Viter workers cleanly every 250 batches (`MAX_VITER_WORKER_BATCHES = 250`), and respawn any worker that exits mid-batch while retrying its interrupted batch in smaller `_VITER.align_many` sub-chunks (`chunk_size` 16, 4, 1).

## Tried and ruled out
- Assuming `GPU workers exited unexpectedly` was a Linux OOM-killer event or a single bad utterance: the batch around line 1,332,033 aligned in 3.5s on a fresh worker, whereas in the full 3.6M-row run `viter` panicked in Rust (`wgpu error: Out of Memory` in `wgpu_core.rs`) after ~385–916 batches per worker from long-running `wgpu`/Vulkan memory accumulation and `retry_beam=800` spikes.
- Using one shared `multiprocessing.Queue` for all workers without tracking batch ownership: when a worker panicked in Rust, the coordinator could not tell which submitted `batch_id` was lost, stalled `next_write` until all 32 in-flight slots filled, and aborted the entire job on `results.get(timeout=30)`.

In the code:
- `hebrew_training/align_official_manifest.py`
- `tests/test_align_official_manifest.py`
