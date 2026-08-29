"""Convert Hebrew corpora into the manifest format Kyutai's trainer expects.

Replaces our own slicing pipeline for training. The official DataLoader does its own
windowing: given `words`, it cuts each utterance at a random word boundary, uses the audio
before the cut as voice conditioning and the audio after it as the target, paired with only
the remaining words as text. So we hand it whole aligned utterances and let it decide the
split, rather than pre-cutting clips and pairing prompts ourselves.

Row schema (see `get_entry` in training/dataloader.py):

    {"path": str, "start": float, "duration": float, "transcript": str,
     "words": [{"word": str, "start": float, "end": float}]}

`words` is optional but strongly preferred; without it the loader falls back to a random
window for the voice prompt. Word times are RELATIVE to `start`, because align_data.py
keys on (path, start) and aligns inside that window.

`words` IS THE TEXT. training/dataloader.py:150 builds what it trains on by joining
`words`, and reads `transcript` only in the fallback branch where no word-boundary cut
exists. So anything done to the text -- normalization above all -- has to be done to
`words`, or it is silently discarded. `transcript` here is derived from `words` so the two
cannot drift apart.

WHY SEGMENTS ARE MERGED
-----------------------
CrowdRecital ships utterance-length segments. The ivrit-ai Knesset corpora ship Whisper
DECODER segments: median 1.84 s, p90 4.40 s, measured over 702,646 segments in 120 random
plenum recordings. That length is not usable as a manifest row, because of `MIN_CUT_SEC`
in training/dataloader.py:

    a cut must leave >= 1.0 s of audio on BOTH sides

so a row shorter than 2.0 s has no eligible word-boundary cut at all -- 53.7% of raw
segments. Those rows take the loader's fallback branch, where the voice prompt is a window
read from `entry.start`, i.e. it OVERLAPS the target audio the model is asked to predict.
That is the same prompt leakage prepare_data_v2 was written to remove.

So consecutive segments are merged into utterances of ~12 s before a row is emitted, and
merging stops at a gap longer than --merge-gap (parliamentary audio is 63.7% speech; the
rest is gavel, procedure and dead air, and merging across it would train the model on
silence). Measured on the same 120 recordings, --merge-gap 1.5 --min-duration 4 yields an
11.88 s median row and retains ~5,490 h of the plenum corpus.

This is the same job prepare_ivritai.py's generate_slices did for the earlier dataset --
including drawing each group's target around the aim rather than fixing it, which
DurationController did there and --merge-jitter does here. The difference is only the
output: that wrote one wav per utterance, this writes offsets into one transcoded wav per
recording (or the source audio when --no-transcode-audio is used).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from array import array
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

_WS = re.compile(r"\s+")
_HEBREW = re.compile(r"[֐-׿]")

# Resolved from this file, not the working directory: the earlier relative default broke
# whenever the command was run from anywhere but the repo root. Assumes the two repos are
# checked out side by side, which is what the recipe tells you to do.
DEFAULT_NORMALIZER = Path(__file__).resolve().parents[2] / "hebrew-tts-data-tools" / "normalizer"
PROGRESS_SECONDS = 10.0
PROGRESS_RECORDINGS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Kyutai-format training manifests from an aligned Hebrew corpus."
    )
    parser.add_argument(
        "--corpus", type=Path, required=True, help="Root holding one directory per recording."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--audio-glob",
        default="audio.wav,audio.m4a,audio.mka",
        help="Comma-separated candidates; the first that matches wins. "
        "CrowdRecital ships audio.mka, the Knesset corpora audio.m4a.",
    )
    parser.add_argument("--align-glob", default="transcript.aligned.json")
    parser.add_argument("--metadata-glob", default="metadata.json")
    parser.add_argument(
        "--transcode-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transcode each used recording to a 24 kHz mono WAV in OUT_DIR/audio "
        "(default: enabled). Completed files are reused; pass --no-transcode-audio "
        "to keep source paths in the manifest.",
    )
    parser.add_argument(
        "--speaker-field",
        default="user_id",
        help="metadata.json key identifying the speaker. Falls back to the "
        "recording directory name, which for the Knesset corpora means "
        "the split is recording-disjoint but not speaker-disjoint.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="Post-merge floor. Must stay above 2 x MIN_CUT_SEC (2.0 s) or "
        "the loader cannot cut the row and leaks the prompt.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Should match data.max_duration_sec in the training config.",
    )
    parser.add_argument(
        "--merge-target",
        type=float,
        default=12.0,
        help="Stop merging once a group reaches this length.",
    )
    parser.add_argument(
        "--merge-jitter",
        type=float,
        default=0.3,
        help="Draw each group's target uniformly from target*(1 +/- this), the "
        "way prepare_ivritai.py's DurationController did. 0 fixes the "
        "target and narrows the duration distribution.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seeds the merge jitter, so a rebuild is reproducible."
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=1.5,
        help="Break a merge at a silence longer than this. 0 disables merging, "
        "which is right only for already-utterance-length corpora.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.6,
        help="Median word probability required to keep a segment.",
    )
    parser.add_argument("--valid-hours", type=float, default=2.0)
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Apply the Hebrew TTS normalizer (number expansion etc).",
    )
    parser.add_argument(
        "--normalizer-dir",
        type=Path,
        default=DEFAULT_NORMALIZER,
        help="The normalizer package from the hebrew-tts-data-tools repo. "
        "Defaults to a checkout sitting beside this one.",
    )
    parser.add_argument("--limit-recordings", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of processes for the per-recording work. Output is reproducible "
        "and independent of worker count.",
    )
    parser.add_argument(
        "--spool-dir",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="Parent for a temporary manifest spool (default: the system temporary directory). "
        "Use local storage; the temporary tree is removed after use.",
    )
    parser.add_argument(
        "--max-spool-gib",
        type=float,
        default=200.0,
        help="Hard limit on JSON payload stored in the temporary spool (default: 200 GiB).",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_spool_gib <= 0:
        parser.error("--max-spool-gib must be positive")
    return args


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def stable_bucket(key: str) -> float:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64


def format_seconds(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def load_normalizer(directory: Path):
    """The same Hebrew normalizer the earlier pipeline used -- number expansion,
    word replacements, punctuation handling. Only the slicing was replaced; this was not."""
    if not directory.exists():
        raise SystemExit(
            f"--normalize-text needs the normalizer, and {directory} does not exist.\n"
            "Clone it beside this repo:\n"
            "    git clone https://github.com/asaelbarilan/hebrew-tts-data-tools\n"
            "or pass --normalizer-dir explicitly."
        )
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory.resolve()))
    from hebrew_tts_normalizer import (  # type: ignore
        TTSNormalizeOptions,
        load_word_replacements,
        normalize_tts_text,
    )

    options = TTSNormalizeOptions(
        apply_word_replacements=True,
        expand_numbers=True,
        keep_punctuations=True,
        attach_punctuations_to_token=True,
        stt_compat_mode=False,
        remove_parentheses=False,
    )
    replacements = load_word_replacements()
    return lambda text: normalize_tts_text(text, options=options, word_replacements=replacements)


def normalize_words(words: list[dict], normalize) -> list[dict]:
    """Normalize the word list itself, not just the joined transcript.

    This is not optional polish. training/dataloader.py:150 builds the text it trains on
    from `words`, and only falls back to `transcript` when there is no usable word-boundary
    cut. So normalizing `transcript` alone -- which is what this script did until now --
    was thrown away on every ordinary sample, and the model would have been trained on
    "1995" instead of "אלף תשע מאות תשעים וחמש".

    A normalized word can expand into several ("1995" -> five words). The expansion is
    spoken across the original word's span, so the span is divided evenly between them.
    That is an approximation, but the loader only uses word times to pick a cut point and
    to trim trailing silence, and both stay correct at the group's outer boundaries.
    """
    out: list[dict] = []
    for word in words:
        text = clean(normalize(word["word"]))
        if not text:
            continue
        parts = text.split()
        start, end = float(word["start"]), float(word["end"])
        if len(parts) == 1:
            out.append({**word, "word": parts[0]})
            continue
        step = (end - start) / len(parts)
        for index, part in enumerate(parts):
            out.append(
                {"word": part, "start": start + index * step, "end": start + (index + 1) * step}
            )
    return out


def find_audio(recording: Path, globs: str) -> Path | None:
    """First matching audio file. Corpora differ: audio.wav here, audio.m4a there."""
    for pattern in (g.strip() for g in globs.split(",") if g.strip()):
        match = next(iter(recording.glob(pattern)), None)
        if match is not None:
            return match
    return None


def transcode_audio(audio: Path, recording: Path, out_dir: Path) -> tuple[Path, bool]:
    """Write one restart-safe WAV per recording directly to the durable output volume."""
    target = out_dir / "audio" / f"{recording.name}.wav"
    if target.is_file() and target.stat().st_size > 44:
        return target.resolve(), False

    # Deterministic so a restart overwrites, rather than strands, a potentially huge partial.
    partial = target.with_name(f".{target.name}.partial")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio),
                "-map",
                "0:a:0",
                "-map_metadata",
                "-1",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                "-threads",
                "1",
                "-f",
                "wav",
                str(partial),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit status {result.returncode}"
            raise RuntimeError(f"ffmpeg failed for {audio}: {detail}")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target.resolve(), True


def clean_segments(segments: list[dict], args, stats: dict) -> list[dict]:
    """Segments that pass the quality and language filters, times still absolute."""
    kept = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            stats["no_words"] += 1
            continue

        probabilities = [w["probability"] for w in words if w.get("probability") is not None]
        if probabilities and statistics.median(probabilities) < args.min_quality:
            stats["low_quality"] += 1
            continue

        timed = [w for w in words if w.get("start") is not None and w.get("end") is not None]
        if not timed:
            stats["no_words"] += 1
            continue

        text = clean(segment.get("text", ""))
        if not text or not _HEBREW.search(text):
            stats["no_hebrew"] += 1
            continue

        kept.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text,
                "words": timed,
            }
        )
    kept.sort(key=lambda s: s["start"])
    return kept


def merge_segments(segments: list[dict], args, rng: random.Random) -> list[list[dict]]:
    """Group consecutive segments into utterance-length runs.

    A group closes when it reaches its target length, when the next segment would push it
    past --max-duration, or when the silence before it exceeds --merge-gap. See the module
    docstring for why merging is required and not a tuning knob.

    Each group draws its own target around --merge-target rather than using it fixed, which
    is what prepare_ivritai.py's DurationController did. Measured over 120 plenum
    recordings, jitter 0.3 widens the p10-p90 duration spread from 9.70 s to 10.65 s for the
    same total hours. A narrow duration distribution was implicated in the earlier run's
    overfitting (docs/eos-overfitting-research.md), so the spread is worth having free.
    """
    if args.merge_gap <= 0:
        return [[segment] for segment in segments]

    def next_target() -> float:
        if args.merge_jitter <= 0:
            return args.merge_target
        low = max(args.min_duration, args.merge_target * (1 - args.merge_jitter))
        high = min(args.max_duration, args.merge_target * (1 + args.merge_jitter))
        return rng.uniform(low, high)

    groups: list[list[dict]] = []
    current: list[dict] = []
    target = next_target()
    for segment in segments:
        if not current:
            current = [segment]
            continue
        gap = segment["start"] - current[-1]["end"]
        span = segment["end"] - current[0]["start"]
        if gap <= args.merge_gap and span <= args.max_duration:
            current.append(segment)
            if current[-1]["end"] - current[0]["start"] >= target:
                groups.append(current)
                current = []
                target = next_target()
        else:
            groups.append(current)
            current = [segment]
            target = next_target()
    if current:
        groups.append(current)
    return groups


def segment_rows(recording: Path, args, normalize, rng: random.Random) -> tuple[list[dict], dict]:
    """One manifest row per merged run of aligned segments."""
    stats = {
        "no_words": 0,
        "low_quality": 0,
        "bad_duration": 0,
        "no_hebrew": 0,
        "kept": 0,
        "audio_transcoded": 0,
        "audio_reused": 0,
    }
    audio = find_audio(recording, args.audio_glob)
    align = next(iter(recording.glob(args.align_glob)), None)
    if audio is None or align is None:
        return [], stats

    # No speaker field in the Knesset corpora, so fall back to the recording id. Without
    # this every row would share one speaker key and the validation split would collapse.
    speaker = None
    meta_file = next(iter(recording.glob(args.metadata_glob)), None)
    if meta_file is not None:
        try:
            speaker = json.loads(meta_file.read_text(encoding="utf-8")).get(args.speaker_field)
        except (json.JSONDecodeError, OSError):
            speaker = None
    speaker = str(speaker) if speaker else recording.name

    try:
        segments = json.loads(align.read_text(encoding="utf-8"))["segments"]
    except (json.JSONDecodeError, KeyError, OSError):
        return [], stats

    rows = []
    for group in merge_segments(clean_segments(segments, args, stats), args, rng):
        start = group[0]["start"]
        duration = group[-1]["end"] - start
        if not args.min_duration <= duration <= args.max_duration:
            stats["bad_duration"] += 1
            continue

        words = [word for segment in group for word in segment["words"]]
        if normalize is not None:
            words = normalize_words(words, normalize)
        if not words:
            stats["no_hebrew"] += 1
            continue

        # Derived from `words` so the two can never disagree -- and because `words` is what
        # the loader actually trains on.
        text = clean(" ".join(word["word"] for word in words))
        if not text:
            stats["no_hebrew"] += 1
            continue

        # Relative to `start`: the loader compares these against entry.duration, and
        # align_data.py aligns inside the (path, start) window.
        timed = [
            {
                "word": clean(word["word"]),
                "start": round(float(word["start"]) - start, 4),
                "end": round(float(word["end"]) - start, 4),
            }
            for word in words
        ]

        rows.append(
            {
                "path": str(audio.resolve()),
                "start": round(start, 4),
                "duration": round(duration, 4),
                "transcript": text,
                "words": timed,
                "speaker": speaker,
            }
        )
        stats["kept"] += 1
    if rows and args.transcode_audio:
        manifest_audio, created = transcode_audio(audio, recording, args.out_dir)
        for row in rows:
            row["path"] = str(manifest_audio)
        stats["audio_transcoded" if created else "audio_reused"] += 1
    return rows, stats


_WORKER_NORMALIZE = None
_WORKER_ARGS = None
_WORKER_SPOOL_DIR = None
_WORKER_MAX_SPOOL_BYTES = 0
_WORKER_SPOOL_BYTES = None
_WORKER_SPOOL_LOCK = None


def _init_worker(
    args: argparse.Namespace, spool_dir: Path, max_spool_bytes: int, spool_bytes, spool_lock
) -> None:
    """Load the normalizer once per worker process (lambdas do not pickle)."""
    global _WORKER_ARGS, _WORKER_MAX_SPOOL_BYTES, _WORKER_NORMALIZE
    global _WORKER_SPOOL_BYTES, _WORKER_SPOOL_DIR, _WORKER_SPOOL_LOCK
    _WORKER_ARGS = args
    _WORKER_SPOOL_DIR = spool_dir
    _WORKER_MAX_SPOOL_BYTES = max_spool_bytes
    _WORKER_SPOOL_BYTES = spool_bytes
    _WORKER_SPOOL_LOCK = spool_lock
    if args.normalize_text:
        _WORKER_NORMALIZE = load_normalizer(args.normalizer_dir)


def _recording_rng(seed: int, recording: Path) -> random.Random:
    # str hashing is salted per process, so use a digest for cross-process stability.
    digest = hashlib.sha256(f"{seed}:{recording.name}".encode()).digest()
    return random.Random(digest)


def _write_spool(
    index: int, rows: list[dict], spool_dir: Path, max_spool_bytes: int, spool_bytes, spool_lock
) -> tuple[Path, str | None, float, array]:
    path = spool_dir / f"{index:09d}.jsonl"
    reserved = 0
    durations = array("d")
    speaker = rows[0]["speaker"] if rows else None
    seconds = 0.0
    try:
        with path.open("wb") as handle:
            for row in rows:
                if row["speaker"] != speaker:
                    raise RuntimeError(f"recording {index} contains multiple speakers")
                payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                with spool_lock:
                    wanted = spool_bytes.value + len(payload)
                    if wanted > max_spool_bytes:
                        raise RuntimeError(
                            f"manifest spool exceeded {max_spool_bytes / 2**30:.1f} GiB; "
                            "free local space, raise --max-spool-gib, or use a smaller corpus"
                        )
                    spool_bytes.value = wanted
                reserved += len(payload)
                handle.write(payload)
                duration = float(row["duration"])
                durations.append(duration)
                seconds += duration
    except BaseException:
        with spool_lock:
            spool_bytes.value -= reserved
        path.unlink(missing_ok=True)
        raise
    return path, speaker, seconds, durations


def _process_recording(
    index: int, recording: Path
) -> tuple[int, Path, str | None, float, int, array, dict]:
    rows, stats = segment_rows(
        recording, _WORKER_ARGS, _WORKER_NORMALIZE, _recording_rng(_WORKER_ARGS.seed, recording)
    )
    path, speaker, seconds, durations = _write_spool(
        index,
        rows,
        _WORKER_SPOOL_DIR,
        _WORKER_MAX_SPOOL_BYTES,
        _WORKER_SPOOL_BYTES,
        _WORKER_SPOOL_LOCK,
    )
    return index, path, speaker, seconds, len(durations), durations, stats


def main() -> None:
    args = parse_args()
    recordings = sorted(p for p in args.corpus.iterdir() if p.is_dir())
    if args.limit_recordings:
        recordings = recordings[: args.limit_recordings]

    if args.transcode_audio:
        if shutil.which("ffmpeg") is None:
            raise SystemExit("--transcode-audio requires ffmpeg on PATH")
        (args.out_dir / "audio").mkdir(parents=True, exist_ok=True)

    if not args.spool_dir.is_dir():
        raise SystemExit(f"--spool-dir is not a directory: {args.spool_dir}")
    configured_spool_bytes = int(args.max_spool_gib * 2**30)
    free_bytes = shutil.disk_usage(args.spool_dir).free
    max_spool_bytes = min(configured_spool_bytes, int(free_bytes * 0.9))
    if max_spool_bytes < configured_spool_bytes:
        print(
            f"note: limiting the spool to {max_spool_bytes / 2**30:.1f} GiB, 90% of the "
            f"{free_bytes / 2**30:.1f} GiB free on {args.spool_dir}"
        )

    spool_dir = Path(tempfile.mkdtemp(prefix="pocket-tts-manifest-", dir=args.spool_dir))
    context = multiprocessing.get_context()
    spool_bytes = context.Value("Q", 0, lock=False)
    spool_lock = context.Lock()
    results: list[tuple[Path, str | None, float, int] | None] = [None] * len(recordings)
    duration_counts: dict[float, int] = {}
    uncuttable = 0
    progress_started = time.monotonic()
    progress_last_report = progress_started
    progress_last_count = 0
    completed_recordings = 0
    totals = {
        "no_words": 0,
        "low_quality": 0,
        "bad_duration": 0,
        "no_hebrew": 0,
        "kept": 0,
        "audio_transcoded": 0,
        "audio_reused": 0,
    }

    def add_stats(stats: dict) -> None:
        for key, value in stats.items():
            totals[key] += value

    def collect(result) -> None:
        nonlocal completed_recordings, progress_last_count, progress_last_report, uncuttable
        index, path, speaker, seconds, count, durations, stats = result
        results[index] = (path, speaker, seconds, count)
        for duration in durations:
            duration_counts[duration] = duration_counts.get(duration, 0) + 1
            uncuttable += duration < 2.0
        add_stats(stats)
        completed_recordings += 1

        now = time.monotonic()
        report = (
            completed_recordings == len(recordings)
            or completed_recordings - progress_last_count >= PROGRESS_RECORDINGS
            or now - progress_last_report >= PROGRESS_SECONDS
        )
        if not report:
            return
        elapsed = now - progress_started
        remaining = len(recordings) - completed_recordings
        eta = elapsed * remaining / completed_recordings
        audio_progress = ""
        if args.transcode_audio:
            audio_progress = (
                f", WAVs {totals['audio_transcoded']} new/{totals['audio_reused']} reused"
            )
        print(
            f"progress : {completed_recordings}/{len(recordings)} recordings "
            f"({100 * completed_recordings / len(recordings):.1f}%), "
            f"{remaining} waiting, elapsed {format_seconds(elapsed)}, ETA {format_seconds(eta)}, "
            f"{totals['kept']} utterances{audio_progress}",
            flush=True,
        )
        progress_last_report = now
        progress_last_count = completed_recordings

    try:
        if args.workers == 1:
            normalize = load_normalizer(args.normalizer_dir) if args.normalize_text else None
            for index, recording in enumerate(recordings):
                rows, stats = segment_rows(
                    recording, args, normalize, _recording_rng(args.seed, recording)
                )
                path, speaker, seconds, durations = _write_spool(
                    index, rows, spool_dir, max_spool_bytes, spool_bytes, spool_lock
                )
                collect((index, path, speaker, seconds, len(durations), durations, stats))
        else:
            initargs = (args, spool_dir, max_spool_bytes, spool_bytes, spool_lock)
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=context,
                initializer=_init_worker,
                initargs=initargs,
            ) as pool:
                recording_iter = iter(enumerate(recordings))
                pending = {}
                for index, recording in recording_iter:
                    future = pool.submit(_process_recording, index, recording)
                    pending[future] = recording
                    if len(pending) >= args.workers * 2:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for completed in done:
                            pending.pop(completed)
                            collect(completed.result())
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for completed in done:
                        pending.pop(completed)
                        collect(completed.result())

        if not totals["kept"]:
            raise SystemExit("no usable utterances found")

        # Speaker-disjoint split, deterministic: whole speakers go to validation until the
        # hour budget is met, so no voice appears on both sides.
        seconds_by_speaker: dict[str, float] = {}
        for result in results:
            assert result is not None
            _, speaker, seconds, _ = result
            if speaker is not None:
                seconds_by_speaker[speaker] = seconds_by_speaker.get(speaker, 0.0) + seconds
        budget = args.valid_hours * 3600
        ordered = sorted(seconds_by_speaker, key=stable_bucket)
        valid_speakers: set[str] = set()
        accumulated = 0.0
        for speaker in ordered:
            if accumulated >= budget:
                break
            if seconds_by_speaker[speaker] > budget * 1.5:
                continue
            valid_speakers.add(speaker)
            accumulated += seconds_by_speaker[speaker]

        # A whole Knesset sitting runs several hours, so every candidate can exceed the
        # oversize guard above and leave validation empty.
        if not valid_speakers:
            smallest = min(ordered, key=lambda s: seconds_by_speaker[s])
            valid_speakers.add(smallest)
            accumulated = seconds_by_speaker[smallest]
            print(
                f"note: no speaker fit under {args.valid_hours} h; validation is the single "
                f"smallest ({smallest}, {accumulated / 3600:.2f} h)"
            )

        train_speakers = set(seconds_by_speaker) - valid_speakers
        assert not (train_speakers & valid_speakers), "speaker leaked across split"
        assert train_speakers and valid_speakers, "one side of the split is empty"

        args.out_dir.mkdir(parents=True, exist_ok=True)
        train_count = valid_count = 0
        train_seconds = valid_seconds = 0.0
        train_path = args.out_dir / "train_aligned.jsonl"
        valid_path = args.out_dir / "valid_aligned.jsonl"
        with train_path.open("wb") as train_handle, valid_path.open("wb") as valid_handle:
            for result in results:
                assert result is not None
                path, speaker, seconds, count = result
                is_valid = speaker in valid_speakers
                with path.open("rb") as source:
                    shutil.copyfileobj(source, valid_handle if is_valid else train_handle)
                path.unlink()
                if is_valid:
                    valid_count += count
                    valid_seconds += seconds
                else:
                    train_count += count
                    train_seconds += seconds
        assert train_count + valid_count == totals["kept"]

        print(f"recordings scanned : {len(recordings)}")
        print(f"utterances kept    : {totals['kept']}")
        if args.transcode_audio:
            print(
                f"audio wavs          : {totals['audio_transcoded']} transcoded, "
                f"{totals['audio_reused']} reused"
            )
        print(
            f"  dropped: no_words {totals['no_words']}, low_quality {totals['low_quality']}, "
            f"duration {totals['bad_duration']}, no_hebrew {totals['no_hebrew']}"
        )
        print(
            f"train : {train_count:6d} utterances, {train_seconds / 3600:6.2f} h, "
            f"{len(train_speakers)} speakers"
        )
        print(
            f"valid : {valid_count:6d} utterances, {valid_seconds / 3600:6.2f} h, "
            f"{len(valid_speakers)} speakers"
        )

        # Compute exact quantiles from bounded unique-duration counts rather than retaining
        # one heavyweight Python object per row.
        ordered_durations = sorted(duration_counts.items())

        def duration_at(index: int) -> float:
            seen = 0
            for duration, count in ordered_durations:
                seen += count
                if seen > index:
                    return duration
            raise AssertionError("duration index is out of range")

        row_count = totals["kept"]
        middle = row_count // 2
        median = duration_at(middle)
        if row_count % 2 == 0:
            median = (duration_at(middle - 1) + median) / 2
        print(
            f"row duration : median {median:.2f} s, "
            f"p10 {duration_at(row_count // 10):.2f}, "
            f"p90 {duration_at(9 * row_count // 10):.2f}, max {ordered_durations[-1][0]:.2f}"
        )
        print(
            f"uncuttable (<2.0 s, prompt would overlap target): {uncuttable} "
            f"({100 * uncuttable / row_count:.2f}%)"
        )
        print(f"wrote {train_path} and {valid_path}")
    finally:
        shutil.rmtree(spool_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
