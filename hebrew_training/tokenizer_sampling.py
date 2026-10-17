"""Bounded, seeded byte-window sampling of local tokenizer text (never a full scan).

Windows are byte-weighted, not uniform over documents. JSONL recording IDs are used to
keep recordings out of both splits; text records are split by content hash. A document
contribution cap and line deduplication prevent long pages/boilerplate dominating.
Plain-text lines are treated as records; original web-page boundaries may be unavailable.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from hebrew_training.tokenizer_text import _record_text, find_files, lines_of


@dataclass
class SamplingSpec:
    mode: str = "prefix"
    seed: int = 1729
    holdout_chars: int = 150_000
    holdout_fraction: float = 0.1
    window_bytes: int = 65_536
    max_record_bytes: int = 1_048_576
    max_document_chars: int = 4000
    max_read_mb: float = 2048


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def sample_source(path, want_chars, charset, keep, spec, cache_dir):
    """Return a repeatable train pool; save held-out text and sampling provenance beside it."""
    files = find_files(Path(path))
    if any(p.suffix.lower() not in (".txt", ".jsonl") for p in files):
        raise ValueError("bounded sampling needs seekable, uncompressed .txt/.jsonl files")
    if (
        not 0 < spec.holdout_fraction < 1
        or min(spec.window_bytes, spec.max_record_bytes, spec.max_document_chars, spec.max_read_mb)
        <= 0
        or spec.holdout_chars < 0
    ):
        raise ValueError("invalid bounded sampling limits or holdout fraction")
    provenance = {
        "version": 1,
        "files": [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
        "sampling": asdict(spec),
        "charset": charset.describe(),
        "source": cache_dir.name,
    }
    fingerprint = digest(provenance)
    meta = cache_dir / "sampling.json"
    train_path, held_path = cache_dir / "train.txt", cache_dir / "heldout.txt"
    if meta.exists():
        previous = json.loads(meta.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError(f"stale sampling cache {cache_dir}; use a new cache directory")
        if previous["train_chars"] >= want_chars:
            print(f"  reusing bounded sample {train_path}", flush=True)
            return train_path.read_text(encoding="utf-8").splitlines()

    rng = random.Random(spec.seed)
    sizes = [p.stat().st_size for p in files]
    if not sum(sizes):
        raise ValueError(f"empty source: {path}")
    train, held, seen, documents = [], [], set(), set()
    before = {key: value for key, value in vars(charset.stats).items() if key != "foreign"}
    foreign_before = charset.stats.foreign.copy()
    train_chars = held_chars = read_bytes = windows = duplicates = 0
    train_groups, held_groups = set(), set()
    limit = int(spec.max_read_mb * 1e6)
    while train_chars < want_chars or held_chars < spec.holdout_chars:
        if read_bytes >= limit:
            raise RuntimeError(
                f"sampling {path} hit {limit} bytes: train={train_chars}/{want_chars}, "
                f"heldout={held_chars}/{spec.holdout_chars}; increase max_read_mb or reduce budgets"
            )
        file = rng.choices(files, weights=sizes, k=1)[0]
        with file.open("rb") as handle:
            offset = rng.randrange(file.stat().st_size)
            handle.seek(offset)
            # Always discard the partial first record, even if this lands on a newline.
            partial = handle.readline(min(spec.max_record_bytes + 1, limit - read_bytes))
            read_bytes += len(partial)
            if not partial.endswith(b"\n"):
                continue
            used = 0
            windows += 1
            while used < spec.window_bytes and read_bytes < limit:
                raw = handle.readline(min(spec.max_record_bytes + 1, limit - read_bytes))
                used += len(raw)
                read_bytes += len(raw)
                if not raw:
                    break
                if len(raw) > spec.max_record_bytes or not raw.endswith(b"\n"):
                    break
                try:
                    text = raw.decode("utf-8").strip()
                    row = json.loads(text) if file.suffix == ".jsonl" else None
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                group = None
                if row is not None:
                    if not isinstance(row, dict):
                        continue
                    group = row.get("speaker") or row.get("user_id") or row.get("path")
                    text = _record_text(row)
                document = digest(text)
                if document in documents:
                    continue
                documents.add(document)
                group_hash = digest(str(group)) if group is not None else document
                is_held = int(group_hash[:16], 16) / 2**64 < spec.holdout_fraction
                if is_held and held_chars >= spec.holdout_chars:
                    continue
                if not is_held and train_chars >= want_chars:
                    continue
                contribution = 0
                for line in lines_of(text):
                    line = charset.clean(line)
                    if not line or not keep(line):
                        continue
                    key = digest(line)
                    if key in seen:
                        duplicates += 1
                        continue
                    seen.add(key)
                    if is_held:
                        held.append(line)
                        held_chars += len(line)
                        held_groups.add(group_hash)
                    else:
                        train.append(line)
                        train_chars += len(line)
                        train_groups.add(group_hash)
                    contribution += len(line)
                    if contribution >= spec.max_document_chars:
                        break
                if train_chars >= want_chars and held_chars >= spec.holdout_chars:
                    break
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_path.write_text("\n".join(train) + "\n", encoding="utf-8")
    held_path.write_text("\n".join(held) + "\n", encoding="utf-8")
    report = {
        **provenance,
        "fingerprint": fingerprint,
        "train_chars": train_chars,
        "heldout_chars": held_chars,
        "read_bytes": read_bytes,
        "windows": windows,
        "duplicates_skipped": duplicates,
        "train_groups": len(train_groups),
        "heldout_groups": len(held_groups),
        "group_overlap": len(train_groups & held_groups),
        "filter_stats": {
            **{key: getattr(charset.stats, key) - value for key, value in before.items()},
            "foreign": dict(charset.stats.foreign - foreign_before),
        },
    }
    meta.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"  sampled {train_chars:,} train / {held_chars:,} held-out chars from "
        f"{read_bytes / 1e6:.1f} MB read; {len(train_groups)} train groups, "
        f"{len(held_groups)} held-out groups",
        flush=True,
    )
    return train
