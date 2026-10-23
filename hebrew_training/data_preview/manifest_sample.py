"""A random subset of a manifest, written to a small jsonl the unmodified DataLoader can read.

`training.dataloader.DataLoader` reads its whole manifest into memory when it is built, which
is the right thing for training and far too slow for a quick look at a manifest of millions
of rows. Rather than touching the loader, the preview hands it a smaller file.

For a large manifest the rows are picked without reading it: random byte offsets, each moved
forward to the next line start. That costs a seek per row (so seconds for tens of thousands
of rows, not a pass over gigabytes), and favours lines that follow long lines a little. Small
manifests are read whole and sampled exactly. The subset keeps the file's order, so a loader
without shuffle reads it in manifest order.

The result is cached on disk, keyed by the source file (path, size, mtime), the row count and
the sample seed, so restarting the loader or the server does not sample again.
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("data_preview")

SMALL_MANIFEST_BYTES = 64 << 20  # below this, read it whole and sample exactly
DRAWS_PER_ROW = 20  # offsets tried per wanted row before settling for fewer


@dataclass(frozen=True)
class SampledManifest:
    path: Path
    rows: int
    source_bytes: int


def _read_all(source: Path, rows: int, rng: random.Random) -> list[bytes]:
    with source.open("rb") as f:
        lines = [line for line in f if line.strip()]
    if len(lines) <= rows:
        return lines
    return [lines[i] for i in sorted(rng.sample(range(len(lines)), rows))]


def _seek_sample(source: Path, size: int, rows: int, rng: random.Random) -> list[bytes]:
    picked: dict[int, bytes] = {}  # line start offset -> line
    with source.open("rb") as f:
        for _ in range(rows * DRAWS_PER_ROW):
            if len(picked) >= rows:
                break
            position = rng.randrange(size)
            f.seek(position)
            if position:
                f.readline()  # the rest of the line the offset landed in
            start = f.tell()
            line = f.readline()
            if line.strip():
                picked.setdefault(start, line)
    if len(picked) < rows:
        logger.warning(
            "%s: only %d of the %d requested rows were found; the manifest is probably smaller",
            source,
            len(picked),
            rows,
        )
    return [picked[start] for start in sorted(picked)]


def sample_manifest(path: str, rows: int, seed: int, cache_dir: Path) -> SampledManifest:
    source = Path(path)
    stat = source.stat()
    key = f"{source.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}\0{rows}\0{seed}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    target = cache_dir / f"{source.stem}-{rows}-{digest}.jsonl"
    if not target.exists():
        rng = random.Random(f"manifest:{seed}")
        if stat.st_size <= SMALL_MANIFEST_BYTES:
            lines = _read_all(source, rows, rng)
        else:
            lines = _seek_sample(source, stat.st_size, rows, rng)
        cache_dir.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        with partial.open("wb") as f:
            for line in lines:
                f.write(line if line.endswith(b"\n") else line + b"\n")
        partial.replace(target)
        logger.info("sampled %d rows of %s into %s", len(lines), source, target)
    with target.open("rb") as f:
        count = sum(1 for _ in f)
    return SampledManifest(target, count, stat.st_size)
