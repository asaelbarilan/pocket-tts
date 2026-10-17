"""Pieces shared by every builder of Kyutai-format manifests.

`build_official_manifest` (Hebrew), `build_hifitts2_manifest` and `build_libritts_r_manifest`
(English, kept to preserve the base model's English) all produce the same row, described in
build_official_manifest.py:

    {"path", "start", "duration", "transcript", "words": [{"word", "start", "end"}], "speaker"}

What they share lives here so the rules cannot drift between corpora: text cleaning, the
deterministic hash used for every "random but reproducible" choice, row validation (word
times relative to `start`, `transcript` derived from `words` because `words` is what the
loader trains on), the duration report, atomic audio writes, and the policy for reusing the
output of an earlier build.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path

from hebrew_training.manifest_files import BUILT

SAMPLE_RATE = 24_000
# training/dataloader.py keeps this much audio on both sides of a cut, so a row shorter than
# twice it has no word-boundary cut and falls back to a prompt that overlaps the target.
MIN_CUT_SEC = 1.0
STATE_FILE = "build_state.json"
AUDIO_FORMATS = ("flac", "wav")

_WS = re.compile(r"\s+")
_SPOKEN = re.compile(r"[^\W_]")


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def stable_bucket(key: str) -> float:
    """A fixed pseudo-random number in [0, 1) for `key`: the same in every process and run."""
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


class DurationStats:
    """Exact duration quantiles from a bounded number of distinct values, not one object per row."""

    def __init__(self) -> None:
        self.counts: dict[float, int] = {}
        self.total = 0
        self.uncuttable = 0

    def add(self, duration: float) -> None:
        self.counts[duration] = self.counts.get(duration, 0) + 1
        self.total += 1
        self.uncuttable += duration < 2 * MIN_CUT_SEC

    def _at(self, index: int) -> float:
        seen = 0
        for duration, count in sorted(self.counts.items()):
            seen += count
            if seen > index:
                return duration
        raise AssertionError("duration index is out of range")

    def lines(self) -> list[str]:
        if not self.total:
            return []
        middle = self.total // 2
        median = self._at(middle)
        if self.total % 2 == 0:
            median = (self._at(middle - 1) + median) / 2
        spread = (
            f"row duration : median {median:.2f} s, "
            f"p10 {self._at(self.total // 10):.2f}, "
            f"p90 {self._at(9 * self.total // 10):.2f}, max {max(self.counts):.2f}"
        )
        short = (
            f"uncuttable (<{2 * MIN_CUT_SEC:.1f} s, prompt would overlap target): "
            f"{self.uncuttable} ({100 * self.uncuttable / self.total:.2f}%)"
        )
        return [spread, short]


def fill_untimed_words(
    words: list[dict], duration: float, max_gap_per_word: float = 1.5
) -> list[dict]:
    """Give every word a start and end.

    The forced aligner leaves a word untimed when none of its characters is in the CTC
    vocabulary: in practice a lone dash, quote or full stop (9% of HiFiTTS-2 rows have one).
    Such a token is silent, so it becomes a zero-width word at the current position inside the
    gap between its timed neighbours; the text is kept exactly as written. A token with letters
    or digits that was not timed is spread over the gap with its neighbours. If that would give
    it more than `max_gap_per_word` seconds the alignment is not trustworthy and ValueError is
    raised, so the caller drops the row.
    """
    out = [dict(word) for word in words]
    untimed = [word.get("start") is None or word.get("end") is None for word in out]
    index = 0
    while index < len(out):
        if not untimed[index]:
            index += 1
            continue
        stop = index
        while stop < len(out) and untimed[stop]:
            stop += 1
        low = float(out[index - 1]["end"]) if index else 0.0
        high = float(out[stop]["start"]) if stop < len(out) else float(duration)
        high = max(high, low)
        spoken = {k for k in range(index, stop) if _SPOKEN.search(out[k]["word"])}
        step = (high - low) / len(spoken) if spoken else 0.0
        if step > max_gap_per_word:
            raise ValueError("an untimed word sits in an implausibly long gap")
        cursor = low
        for k in range(index, stop):
            width = step if k in spoken else 0.0
            out[k]["start"], out[k]["end"] = cursor, cursor + width
            cursor += width
        index = stop
    return out


def normalize_words(words: list[dict], duration: float) -> list[dict]:
    """Validate and round `words` (times relative to the row start). Raises ValueError."""
    if not words:
        raise ValueError("no words")
    out = []
    previous_start = 0.0
    for word in words:
        text = word.get("word")
        if not isinstance(text, str) or not text or text != text.strip() or _WS.search(text):
            raise ValueError(f"bad word {text!r}")
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"word {text!r} has no usable times") from error
        if not (math.isfinite(start) and math.isfinite(end)) or end < start:
            raise ValueError(f"word {text!r} has times {start}..{end}")
        if start < previous_start - 1e-6:
            raise ValueError(f"word {text!r} starts before the previous word")
        previous_start = start
        start = min(max(start, 0.0), duration)
        out.append({"word": text, "start": round(start, 4), "end": round(min(end, duration), 4)})
        out[-1]["end"] = max(out[-1]["end"], out[-1]["start"])
    return out


def make_row(
    path: Path | str,
    duration: float,
    words: list[dict],
    speaker: str,
    start: float = 0.0,
    **extra: object,
) -> dict:
    """One manifest row. `transcript` is derived from `words`, so they cannot disagree."""
    timed = normalize_words(words, duration)
    return {
        "path": str(path),
        "start": round(start, 4),
        "duration": round(duration, 4),
        "transcript": clean(" ".join(word["word"] for word in timed)),
        "words": timed,
        "speaker": str(speaker),
        **extra,
    }


def write_audio(samples, path: Path, audio_format: str, sample_rate: int = SAMPLE_RATE) -> None:
    """Write mono PCM16 audio so that a file at `path` is always a complete one."""
    import soundfile

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        soundfile.write(
            str(partial), samples, sample_rate, format=audio_format.upper(), subtype="PCM_16"
        )
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def parallel(func: Callable, items: list, workers: int):
    """Yield `func(item)` for every item as each finishes, in processes (inline for one worker).

    `func` and the items must be picklable. Running inline for `workers == 1` keeps a build
    debuggable and lets tests replace the network calls, which a child process would not see."""
    if workers == 1 or len(items) <= 1:
        for item in items:
            yield func(item)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=min(workers, len(items))) as pool:
        for future in as_completed([pool.submit(func, item) for item in items]):
            yield future.result()


def audio_suffix(audio_format: str) -> str:
    return f".{audio_format}"


def prepare_output(
    out_dir: Path,
    signature: dict,
    *,
    assume_yes: bool,
    wipe: tuple[str, ...] = ("audio", "rows"),
    ask: Callable[[str], str] | None = None,
) -> None:
    """Decide whether what is already in `out_dir` can be reused, and record `signature`.

    Raising the hours target, or lowering it, always reuses everything: the selections are
    nested, and every audio file and processed shard that is already there is skipped. Only a
    change to something baked into the files themselves (`signature`: the audio format, the
    sample rate, the layout version) makes them unusable. Then the audio and row files are
    deleted and rebuilt, after the user agrees; `meta/` (the downloaded metadata) is kept.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / STATE_FILE
    previous = None
    if state_path.is_file():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
    existing = [name for name in wipe if (out_dir / name).exists()]
    if previous is not None:
        changed = {key: (previous.get(key), value) for key, value in signature.items()}
        changed = {key: pair for key, pair in changed.items() if pair[0] != pair[1]}
    elif existing:
        changed = {"state file": ("missing", "present")}
    else:
        changed = {}
    if changed:
        details = "; ".join(f"{key}: {old!r} -> {new!r}" for key, (old, new) in changed.items())
        print(
            f"WARNING: {out_dir} was built with different settings ({details}).\n"
            f"The existing files cannot be reused. Continuing DELETES {', '.join(existing)} "
            "in it (and the old manifests) and REDOWNLOADS everything. Downloaded metadata "
            "under meta/ is kept.",
            file=sys.stderr,
        )
        if not assume_yes:
            if ask is None:
                if not sys.stdin.isatty():
                    raise SystemExit(
                        "not a terminal: rerun with --yes to confirm, or use a new --out-dir"
                    )
                ask = input
            if ask("Delete and rebuild? [y/N] ").strip().lower() not in {"y", "yes"}:
                raise SystemExit("aborted; nothing was deleted")
        for name in existing:
            shutil.rmtree(out_dir / name, ignore_errors=True)
        for split in ("train", "valid"):
            (out_dir / BUILT.format(split=split)).unlink(missing_ok=True)
    state_path.write_text(json.dumps(signature, indent=2) + "\n", encoding="utf-8")


def write_manifests(out_dir: Path, train: list[dict], valid: list[dict]) -> tuple[Path, Path]:
    """Write train.jsonl and valid.jsonl, sorted by path so a rebuild is byte-identical."""
    paths = []
    for split, rows in (("train", train), ("valid", valid)):
        target = out_dir / BUILT.format(split=split)
        partial = target.with_name(f".{target.name}.partial")
        with partial.open("w", encoding="utf-8") as handle:
            for row in sorted(rows, key=lambda r: (r["path"], r["start"])):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        partial.replace(target)
        paths.append(target)
    return paths[0], paths[1]


def describe_split(name: str, rows: list[dict]) -> None:
    seconds = sum(row["duration"] for row in rows)
    speakers = len({row["speaker"] for row in rows})
    print(f"{name:5s} : {len(rows):7d} utterances, {seconds / 3600:7.2f} h, {speakers} speakers")


def print_duration_report(rows: list[dict]) -> None:
    stats = DurationStats()
    for row in rows:
        stats.add(row["duration"])
    for line in stats.lines():
        print(line)
