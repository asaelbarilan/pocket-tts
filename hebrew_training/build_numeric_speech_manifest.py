"""Build the "numeric-speech" dataset: speech of spelled-out numbers, trained against digits.

Input: numeric pairs (`digits` / `cardinal`, same meaning; `data_local/numeric_pairs/*.jsonl`) and
a voice prompt library (`python -m hebrew_training.voice_prompts extract`). For every pair, a
prompt is drawn from the library, the Hebrew model speaks the CARDINAL text in that voice (the form the
model reads well), and the row stores `prompt WAV + generated WAV` as one file. The row's text is
the DIGITS form, so training teaches the model to say digits the way it says the cardinal.

The cut between prompt and target is known exactly, so the row names it (`prompt_end`, seconds)
and training/dataloader.py cuts there and nowhere else. No alignment, alts or niqqud: the row has
no `words`.

    uv run python -m hebrew_training.build_numeric_speech_manifest \
        --pairs data_local/numeric_pairs --prompts data_local/voice_prompts_out \
        --out-dir data_local/numeric_speech --config artifacts/hebrew_24l.yaml --device cuda

`OUT_DIR` gets `audio/<id[:2]>/<id>.wav`, `train.jsonl`, `valid.jsonl` and `failures.jsonl`.

SPEAKERS. The library's speakers are split into train and valid HERE (`--valid-speaker-fraction`,
by a hash of the speaker, so a speaker is on one side only); the library's own `split` field is
ignored. A speaker is `speaker`, else `user_id`, else the source recording. Pairs of the splits in
`--valid-splits` (dev, test) take valid speakers, the others train speakers. Within a split every
pair gets a speaker that differs from the previous ones: the speakers are dealt out in a shuffled
cycle (each speaker once before any repeats) and a speaker's own prompts rotate. The assignment is
a pure function of (--seed, pair ids, library), so it is the same on every run and every shard.

RESUMING AND SCALE. Rows are appended one at a time; a rerun skips the ids already written (a torn
last line is ignored). For a large set run one process per GPU with `--shard-index I
--num-shards N` (each writes `train.shard-I.jsonl`; give each its `CUDA_VISIBLE_DEVICES`), then
`--merge` folds the shard files into `train.jsonl` / `valid.jsonl`. Nothing is held in memory but
the pair list and the library manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from hebrew_training.data import read_jsonl

TRAIN, VALID = "train", "valid"
SPLIT_NAMES = {"train", "dev", "test", "valid", "validation"}
# Silence kept after the last generated sample, same role as DataLoader.TRAIL_SEC.
TRAIL_SEC = 0.2
SILENCE_FLOOR = 10 ** (-45 / 20)  # amplitude under which the end of the audio counts as silence


def _hash(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big")


def load_pairs(directory: Path) -> list[dict]:
    """Every pair of every `*.jsonl` in `directory` (any split), each id once, sorted by id.

    An id can be in several files (an aggregate `pairs.jsonl` next to `train/dev/test.jsonl`).
    The copy with its own `split` field wins, then one from a file named after a split, then any
    other, so an aggregate without splits can never relabel a dev or test pair as train.
    """
    best: dict[str, tuple[int, dict]] = {}
    for path in sorted(directory.glob("*.jsonl")):
        for row in read_jsonl(path):
            if not row.get("digits") or not row.get("cardinal"):
                continue
            rank = 2 if row.get("split") else int(path.stem in SPLIT_NAMES)
            if row["id"] in best and best[row["id"]][0] >= rank:
                continue
            row.setdefault("split", path.stem)
            best[row["id"]] = (rank, row)
    if not best:
        raise SystemExit(f"no pairs with `digits` and `cardinal` in {directory}/*.jsonl")
    return sorted((row for _, row in best.values()), key=lambda p: p["id"])


def sample_pairs(pairs: list[dict], count: int, seed: int) -> list[dict]:
    """A seeded random subset of `count` pairs (for a canary), still sorted by id."""
    chosen = sorted(pairs, key=lambda p: _hash(seed, "subset", p["id"]))[:count]
    return sorted(chosen, key=lambda p: p["id"])


def speaker_of(prompt: dict) -> str:
    return str(
        prompt.get("speaker")
        or prompt.get("user_id")
        or f"{prompt.get('dataset', '')}:{prompt.get('source_path', prompt['id'])}"
    )


def load_prompts(directory: Path, max_prompt_sec: float) -> dict[str, list[dict]]:
    """The library's prompts by speaker, those the loader would not truncate, ordered by id."""
    by_speaker: dict[str, list[dict]] = {}
    skipped = 0
    for row in read_jsonl(directory / "manifest.jsonl"):
        if row["duration"] > max_prompt_sec:
            skipped += 1
            continue
        row = {**row, "audio": str((directory / row["audio"]).resolve())}
        by_speaker.setdefault(speaker_of(row), []).append(row)
    if skipped:
        print(f"{skipped} prompts over {max_prompt_sec}s left out", file=sys.stderr)
    for rows in by_speaker.values():
        rows.sort(key=lambda r: str(r["id"]))
    return by_speaker


def split_speakers(
    speakers: list[str], valid_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """(train, valid) speakers, each side shuffled by hash; the valid side has at least one."""
    if len(speakers) < 2:
        raise SystemExit(f"need at least 2 speakers in the library, found {len(speakers)}")
    ordered = sorted(speakers, key=lambda s: _hash(seed, "speaker", s))
    n_valid = min(len(ordered) - 1, max(1, round(valid_fraction * len(ordered))))
    return ordered[n_valid:], ordered[:n_valid]


def assign(
    pairs: list[dict],
    by_speaker: dict[str, list[dict]],
    valid_splits: set[str],
    valid_fraction: float,
    seed: int,
) -> dict[str, tuple[str, dict]]:
    """pair id -> (TRAIN | VALID, prompt). Pure function of the arguments (see the module doc)."""
    train_speakers, valid_speakers = split_speakers(list(by_speaker), valid_fraction, seed)
    pools = {TRAIN: train_speakers, VALID: valid_speakers}
    chosen: dict[str, tuple[str, dict]] = {}
    for side, speakers in pools.items():
        members = [p for p in pairs if (p["split"] in valid_splits) == (side == VALID)]
        members.sort(key=lambda p: _hash(seed, "pair", p["id"]))
        deals: dict[int, list[str]] = {}  # cycle number -> that cycle's speaker order
        for k, pair in enumerate(members):
            cycle, slot = divmod(k, len(speakers))
            if cycle not in deals:
                deals[cycle] = random.Random(_hash(seed, side, "cycle", cycle)).sample(
                    speakers, len(speakers)
                )
            prompts = by_speaker[deals[cycle][slot]]
            chosen[pair["id"]] = (side, prompts[cycle % len(prompts)])
    return chosen


def trim_tail(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Cut trailing silence down to TRAIL_SEC, so the target ends where the speech does."""
    loud = np.flatnonzero(np.abs(audio) > SILENCE_FLOOR)
    if not len(loud):
        return audio
    return audio[: min(len(audio), loud[-1] + 1 + int(TRAIL_SEC * sample_rate))]


class NoEosWatch(logging.Handler):
    """Notices the model's "reached maximum length without EOS" warning (the speech ran to the
    length cap instead of ending), which usually means trailing babble or a repeated phrase."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.hit = False

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "without EOS" in message or "EOS never fired" in message:
            self.hit = True


def audio_path(out_dir: Path, pair_id: str) -> Path:
    return out_dir / "audio" / pair_id[:2] / f"{pair_id}.wav"


def build_row(
    pair: dict,
    prompt: dict,
    prompt_audio: np.ndarray,
    speech: np.ndarray,
    path: Path,
    sr: int,
    no_eos: bool = False,
) -> dict:
    """Write the wav and return the manifest row; `prompt_end` is the prompt's exact length."""
    import soundfile

    audio = np.concatenate([prompt_audio, speech]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), audio, sr, subtype="PCM_16")
    return {
        "path": str(path.resolve()),
        "start": 0.0,
        "duration": round(len(audio) / sr, 4),
        "prompt_end": len(prompt_audio) / sr,
        "transcript": pair["digits"],
        "id": pair["id"],
        "split": pair["split"],
        "archetype_id": pair.get("archetype_id"),
        "cardinal": pair["cardinal"],
        "prompt_id": prompt["id"],
        "prompt_text": prompt.get("text"),
        "speaker": speaker_of(prompt),
        "sample_rate": sr,
        "no_eos": no_eos,
    }


def _rows(path: Path):
    """Rows of a jsonl that may end in a torn line (a killed run): that line is dropped."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def done_ids(out_dir: Path) -> set[str]:
    return {
        row["id"]
        for pattern in ("train*.jsonl", "valid*.jsonl")
        for path in out_dir.glob(pattern)
        for row in _rows(path)
    }


def merge_shards(out_dir: Path) -> None:
    for side in (TRAIN, VALID):
        shards = sorted(out_dir.glob(f"{side}.shard-*.jsonl"))
        if not shards:
            continue
        final = out_dir / f"{side}.jsonl"
        seen = {row["id"] for row in _rows(final)} if final.exists() else set()
        added = 0
        with final.open("a", encoding="utf-8", newline="\n") as out:
            for shard in shards:
                for row in _rows(shard):
                    if row["id"] not in seen:
                        seen.add(row["id"])
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        added += 1
        for shard in shards:
            shard.unlink()
        print(f"{side}: merged {len(shards)} shards, +{added} rows, {len(seen)} total")


def append(path: Path, row: dict) -> None:
    """One row per call, so an interrupted run loses nothing already generated."""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_prompt(path: str, sr: int) -> np.ndarray:
    import soundfile

    audio, file_sr = soundfile.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if file_sr != sr:
        import torch

        from pocket_tts.data.audio_utils import convert_audio

        audio = convert_audio(torch.from_numpy(audio)[None], file_sr, sr, 1)[0].numpy()
    return audio


def load_tts(config: str, device: str, encoder_dirs: list[str] | None):
    """The deploy server's sampler (`HebrewTTS`): classifier-free guidance and the EOS countdown,
    which `TTSModel.generate_audio` does not have. It changes into the project root, so every
    path given on the command line is made absolute first (see `main`)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
    from hebrew_tts_infer import HebrewTTS

    return HebrewTTS(config, device, encoder_dirs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pairs", type=Path, help="directory of pair *.jsonl files")
    p.add_argument("--prompts", type=Path, help="voice prompt library directory")
    p.add_argument("--out-dir", type=Path, required=True, help="gets audio/ and the jsonl files")
    p.add_argument("--config", help="model YAML, as deploy/hebrew_tts_infer.py takes it")
    p.add_argument(
        "--semantic-encoder-dir", nargs="+", help="local semantic encoder dirs (config order)"
    )
    p.add_argument("--device", default="cuda")
    gen = p.add_argument_group(
        "generation (the deploy server's sampler, deploy/hebrew_tts_infer.py)"
    )
    gen.add_argument("--temp", type=float, default=0.3)
    gen.add_argument("--cfg", type=float, default=2.0, help="classifier-free guidance")
    gen.add_argument("--decode-steps", type=int, default=1)
    gen.add_argument("--eos-threshold", type=float, default=-4.0)
    gen.add_argument("--eos-countdown", type=int, default=5, help="frames kept after EOS fires")
    gen.add_argument("--max-seconds", type=float, default=30.0)
    gen.add_argument("--prompt-seconds", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--valid-splits", default="dev,test", help="pair splits that go to valid.jsonl")
    p.add_argument(
        "--valid-speaker-fraction",
        type=float,
        default=0.1,
        help="share of the library's speakers kept for valid (at least one)",
    )
    p.add_argument(
        "--max-prompt-sec", type=float, default=5.0, help="loader's max_voice_prompt_sec"
    )
    p.add_argument(
        "--drop-no-eos",
        action="store_true",
        help="fail (to failures.jsonl) samples whose generation hit the length cap without EOS; "
        "they are otherwise kept and marked `no_eos: true`",
    )
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="fold shard files into train/valid.jsonl")
    p.add_argument(
        "--max-pairs",
        type=int,
        help="use only a seeded random subset of this many pairs (a canary); the speaker "
        "assignment is then computed on the subset, so use a separate --out-dir",
    )
    p.add_argument("--limit", type=int, help="stop each process after this many generated rows")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--dry-run", action="store_true", help="show the assignment, generate nothing")
    args = p.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        p.error("--shard-index must be in [0, --num-shards)")
    if not args.merge and not (args.pairs and args.prompts):
        p.error("--pairs and --prompts are required")
    if not args.merge and not args.dry_run and not args.config:
        p.error("--config is required to generate")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.merge:
        merge_shards(args.out_dir)
        return
    args.pairs, args.prompts, args.out_dir = (
        path.resolve() if path else path for path in (args.pairs, args.prompts, args.out_dir)
    )
    pairs = load_pairs(args.pairs)
    if args.max_pairs:
        pairs = sample_pairs(pairs, args.max_pairs, args.seed)
    by_speaker = load_prompts(args.prompts, args.max_prompt_sec)
    chosen = assign(
        pairs, by_speaker, set(args.valid_splits.split(",")), args.valid_speaker_fraction, args.seed
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    finished = done_ids(args.out_dir)
    mine = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard_index]
    todo = [p for p in mine if p["id"] not in finished]
    print(
        f"{len(pairs)} pairs {dict(Counter(chosen[p['id']][0] for p in pairs))} over "
        f"{len(by_speaker)} speakers; shard {args.shard_index}/{args.num_shards}: "
        f"{len(mine)} pairs, {len(mine) - len(todo)} done, {len(todo)} to generate"
    )
    if args.dry_run:
        for pair in todo[:5]:
            side, prompt = chosen[pair["id"]]
            print(pair["id"], "->", side, speaker_of(prompt), prompt["id"], prompt["duration"], "s")
        return

    tts = load_tts(str(Path(args.config).resolve()), args.device, args.semantic_encoder_dir)
    sr = tts.sample_rate
    suffix = "" if args.num_shards == 1 else f".shard-{args.shard_index}"

    watch = NoEosWatch()
    logging.getLogger("hebrew_tts_infer").addHandler(watch)
    written, started = 0, time.monotonic()
    for pair in todo:
        if args.limit is not None and written >= args.limit:
            break
        side, prompt = chosen[pair["id"]]
        watch.hit = False
        try:
            voice = tts.encode_voice(prompt["audio"], args.prompt_seconds)
            wave = tts.synthesize(
                pair["cardinal"],
                voice,
                temp=args.temp,
                cfg=args.cfg,
                n_steps=args.decode_steps,
                eos_threshold=args.eos_threshold,
                eos_countdown=args.eos_countdown,
                max_seconds=args.max_seconds,
                seed=_hash(args.seed, "generate", pair["id"]) % 2**31,
            )
            if wave is None:
                raise RuntimeError("the model ended speech at once (no frames)")
            speech = wave.numpy()
            if watch.hit and args.drop_no_eos:
                raise RuntimeError("no_eos: generation hit the length cap without EOS")
            row = build_row(
                pair,
                prompt,
                read_prompt(prompt["audio"], sr),
                trim_tail(speech, sr),
                audio_path(args.out_dir, pair["id"]),
                sr,
                no_eos=watch.hit,
            )
        except Exception as exc:  # noqa: BLE001 - one bad sample must not stop the run
            append(args.out_dir / f"failures{suffix}.jsonl", {"id": pair["id"], "error": repr(exc)})
            print(f"failed {pair['id']}: {exc}", file=sys.stderr)
            continue
        append(args.out_dir / f"{side}{suffix}.jsonl", row)
        written += 1
        if written % args.log_every == 0 or written == len(todo):
            rate = written / (time.monotonic() - started)
            eta = (len(todo) - written) / rate if rate else math.inf
            print(f"[{written}/{len(todo)}] {rate:.2f} rows/s, ~{eta / 60:.0f} min left")
    print(f"wrote {written} rows to {args.out_dir}")


if __name__ == "__main__":
    main()
