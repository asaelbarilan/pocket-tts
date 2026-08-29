"""Stream a run's metrics to Weights & Biases, without patching the trainer.

Kyutai's trainer has no experiment tracking -- no wandb, no tensorboard. It writes
`progress.jsonl` in the run directory (see ProgressLog in training/train_utils.py) and prints
to stdout. Rather than patch `training/train.py` and carry a merge conflict against every
upstream pull, this tails that file and forwards it.

It follows three things, all of which the trainer or watch_eval.py already produce:

    progress.jsonl      loss, grad norm, learning rate, validation -- whatever the trainer logs
    args.yaml           the full config, recorded as the wandb run config
    hebrew_eval.jsonl   WER/CER per checkpoint from watch_eval.py, plus the generated audio

    wandb login
    python -m hebrew_training.wandb_sync --run-dir runs/finetune_hebrew --follow

Start it any time -- at launch, or halfway through. It replays the file from the beginning,
so the history is complete either way, and `--follow` then keeps pace with the run.

Safe to run on rank 0 only; ProgressLog is already rank-0-only, so there is one file per run
regardless of how many GPUs are training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", default="hebrew-pocket-tts")
    parser.add_argument("--entity", help="wandb team; omit for your default.")
    parser.add_argument("--name", help="Run name. Defaults to the run directory's name.")
    parser.add_argument(
        "--follow", action="store_true", help="Keep tailing instead of exiting at end of file."
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--no-audio", action="store_true", help="Skip uploading generated wavs.")
    return parser.parse_args()


def flatten(record: dict) -> dict:
    """One jsonl record to wandb scalars.

    The trainer nests measurements under "metrics" and puts the event kind in "type", so
    `train/loss` and `valid/loss` stay distinct series rather than overwriting each other.
    """
    kind = record.get("type", "train")
    out: dict[str, float] = {}
    for key, value in (record.get("metrics") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"{kind}/{key}"] = value
    for key, value in record.items():
        if key in {"metrics", "type", "step", "time"}:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"{kind}/{key}"] = value
    return out


def read_config(run_dir: Path) -> dict:
    path = run_dir / "args.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 -- a bad config must not stop metric streaming
        return {}


def tail(path: Path, offset: int) -> tuple[list[dict], int]:
    """New whole records since `offset`, and the offset to resume from.

    Stops at the first partial line so a record still being written is picked up whole on
    the next pass rather than dropped.
    """
    if not path.exists():
        return [], offset
    records = []
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                offset = position
                break
            if not line.endswith("\n"):
                offset = position
                break
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records, offset


def main() -> None:
    args = parse_args()
    if not args.run_dir.exists():
        raise SystemExit(f"no such run directory: {args.run_dir}")

    import wandb

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name or args.run_dir.name,
        config=read_config(args.run_dir),
        resume="allow",
        id=args.run_dir.name,
    )
    print(f"streaming {args.run_dir} -> {run.url}", flush=True)

    progress = args.run_dir / "progress.jsonl"
    evaluation = args.run_dir / "hebrew_eval.jsonl"
    offsets = {progress: 0, evaluation: 0}
    uploaded: set[str] = set()
    logged = 0

    while True:
        for record in tail(progress, offsets[progress])[0]:
            payload = flatten(record)
            if payload:
                run.log(payload, step=int(record.get("step", 0)))
                logged += 1
        offsets[progress] = tail(progress, offsets[progress])[1]

        for record in tail(evaluation, offsets[evaluation])[0]:
            step = int(record.get("step", 0))
            payload = {
                "eval/wer": record.get("wer"),
                "eval/cer": record.get("cer"),
                "eval/empty_outputs": record.get("empty_outputs"),
                "eval/mean_seconds": record.get("mean_seconds"),
            }
            run.log({k: v for k, v in payload.items() if v is not None}, step=step)
            logged += 1
            if args.no_audio:
                continue
            # The sample text is the same every checkpoint, so the table is comparable
            # down a column: one row per checkpoint, one column per sentence.
            clips = []
            for clip in record.get("clips") or []:
                path = args.run_dir / (clip.get("file") or "")
                if not clip.get("file") or not path.exists() or str(path) in uploaded:
                    continue
                uploaded.add(str(path))
                clips.append(wandb.Audio(str(path), caption=f"{step}: {clip['reference']}"))
            if clips:
                run.log({"eval/samples": clips}, step=step)
        offsets[evaluation] = tail(evaluation, offsets[evaluation])[1]

        if not args.follow:
            break
        time.sleep(args.poll_seconds)

    print(f"logged {logged} records to {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
