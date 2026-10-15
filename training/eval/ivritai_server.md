# Local Hebrew evaluation dashboard

From the repository root, using your training/evaluation Python environment:

```bash
python -m training.eval.ivritai_server
```

Open **http://127.0.0.1:8001**. You can also run the script directly:

```bash
python training/eval/ivritai_server.py --port 8001
```

Optional locations:

```bash
python -m training.eval.ivritai_server \
  --runs-dir /path/to/runs \
  --eval-data /path/to/eval_data \
  --output-dir /path/to/runs/evals
```

The dashboard discovers run folders containing `args.yaml`, `checkpoint_*.pt`
files within each run, audio prompts anywhere under `eval_data`, and nonblank
lines in UTF-8 `.txt` files under `eval_data`. Refresh to pick up new assets or
checkpoints. An absolute run path can also be entered in the UI.

`eval_data/` is ignored by Git. Create it locally and supply your own voice prompts
and UTF-8 sentence files (one sentence per line); no evaluation inputs are bundled.

1. Choose a run and any number of checkpoints, voices, and sentences.
2. Optionally add custom sentences, one per line. Deselect the file sentences
   for custom-only evaluation.
3. Detect available devices and select one or more GPUs, or CPU alone. GPU names
   and total memory are shown. With multiple GPUs, the evaluator runs one worker
   per selected GPU and assigns the next checkpoint to whichever worker is free.
4. Adjust generation/ASR settings and start. The dashboard defaults to CFG 2.0,
   EMA enabled, batch size 6, and blank ASR (audio-only). Other defaults follow
   `ivritai.py`. Each setting includes a short explanation in the UI.
5. Watch the process log and per-checkpoint progress/device assignments. Running
   worker log tails are included; individual worker logs can also be opened.
   As each checkpoint finishes, its results appear in the
   matrix, with sample text, prompt playback, generated playback, ASR transcript,
   WER/CER, duration, and silent/no-EOS flags. Filter by voice to shrink the view.
6. Reopen saved evaluations using the history selector, including after a server
   restart. Pending and failed checkpoints remain visible.

Only one evaluation job runs at a time; that job can evaluate multiple checkpoints
concurrently on different GPUs. Cancel stops the evaluator and its workers and
retains artifacts. A normal server shutdown also cancels the active evaluation.

## Export a static results page

Load a saved evaluation and click **export static HTML** above the results table.
The downloaded `<evaluation-id>.zip` contains `index.html` and an `audio/` folder
with generated WAVs and the original voice prompts (preserving their audio format).
Extract its contents into the root of a static web server, or open `index.html`
locally. Styles, JavaScript, and result data are embedded; voice filtering,
metrics, transcripts, and playback work without the evaluation server.

The export includes all voices, regardless of the current view filter, and takes
a snapshot of the results available when clicked. Running evaluations can be
exported with pending checkpoints still visible. ZIPs are built in the system
temporary directory and removed after the download response finishes. Missing
referenced audio files cause the export to fail rather than produce broken players.

## Standalone evaluator and device discovery

Scheduling, device validation, per-checkpoint isolation, progress, and result
aggregation belong to `ivritai.py`. The dashboard launches a single evaluator
process and reads its output files. It delegates GPU discovery to the same CLI:

```bash
uv run python -m training.eval.ivritai --list-devices
```

This prints JSON containing CPU and visible CUDA device IDs, names, and total
GPU memory. Discovery imports Torch but does not load TTS, Mimi, or Whisper.
Device IDs respect `CUDA_VISIBLE_DEVICES`.

Standalone CLI evaluations saved directly under `runs/evals/<name>` also appear
in dashboard history. The dashboard reads their `evaluation.json` and combined
results without requiring a dashboard manifest. For CLI evaluations, prompt
playback uses the original voice files; dashboard-launched jobs use snapshots.
CLI-started jobs can be viewed live but are cancelled from their own terminal.

To evaluate checkpoints on all visible GPUs, add `--devices auto` to your usual
command. To use a specific subset:

```bash
uv run python -m training.eval.ivritai runs/finetune_hebrew_recital \
  --sentences eval_data/hebrew_sentences.txt \
  --voices eval_data/voice-prompt-1.wav eval_data/voice-prompt-2.wav \
  --checkpoints \
    runs/finetune_hebrew_recital/checkpoint_00019377.pt \
    runs/finetune_hebrew_recital/checkpoint_00020000.pt \
    runs/finetune_hebrew_recital/checkpoint_00022500.pt \
  --devices cuda:0 cuda:1 \
  --cfg 2.0 --use-ema --batch-size 6 --asr "" \
  --output-dir runs/evals/my-comparison
```

`--devices auto` falls back to CPU when there are no CUDA GPUs. Omitting
`--devices` preserves the single-device CLI behavior controlled by `--device`.
Batch size is **per worker**, not divided between GPUs. The number of workers is
capped at the number of selected checkpoints. Each worker loads one checkpoint
at a time; optional ASR is loaded once per worker and reused. Generated audio is
still the full sentence/voice matrix for every checkpoint.

The parent writes combined results atomically as checkpoints finish, maintaining
the original checkpoint selection order even when completion order differs.
A checkpoint failure is recorded while other checkpoints continue; the CLI
returns a nonzero exit code if any checkpoint fails. Unexpected worker crashes
stop the job and retain results already collected. Ctrl+C or termination cancels
the job and stops its workers.

## Output and reproducibility

Each evaluation gets a short random directory, such as `runs/evals/e-1234abcd`:

```text
manifest.json          # selections, source paths/file stats, settings, command,
                       # UTC creation/start/end times, status and exit code
evaluation.log         # evaluator stdout and stderr
results.json           # original evaluator results (one entry per checkpoint)
evaluation.json        # evaluator-owned settings, inputs, job/device assignments,
                       # sample progress, errors, and UTC start/end timestamps
inputs/args.yaml       # snapshot of the run's training arguments
inputs/sentences.txt   # exact evaluated text, including custom sentences
inputs/voice01.wav     # copied voice prompts, used by the evaluator
checkpoints/cp001/     # isolated output for the first selected checkpoint
  results.json         # local result with audio paths relative to this directory
  worker.log           # multi-device worker stdout/stderr and model logs
  <checkpoint-tag>/... # generated WAV files
```

Combined `results.json` audio paths are relative to the evaluation root and
include their checkpoint subdirectory. Existing saved evaluations using the
earlier flat directory layout remain viewable.

The manifest also records each sentence's original file/line (or custom input),
voice names and original paths, and checkpoint sizes/modification times.
Checkpoint weights and model/tokenizer files referenced by `args.yaml` are not
copied. Keep those source artifacts for rerunning an evaluation. A running job
left by an unclean shutdown is marked `interrupted` on the next server startup.

The evaluator is launched as `python -u -m training.eval.ivritai` with the server's
Python interpreter and the repository as its working directory. Install its
training/evaluation dependencies in that environment. Leave ASR blank to generate
audio without transcription or WER/CER. To enable scoring, install `faster-whisper`
and enter `ivrit-ai/whisper-large-v3-turbo-ct2` (or another compatible model) in
the ASR field. For CPU scoring, select CPU and set ASR compute type to `int8`.

The CLI evaluator still defaults to its existing ASR model if `--asr` is omitted;
use `--asr ""` for the dashboard's audio-only behavior.

The server uses the project's existing FastAPI/Uvicorn dependencies, requires no
frontend build or external assets, and binds to localhost by default.
