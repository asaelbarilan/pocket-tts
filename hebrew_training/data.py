from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

_INVISIBLE_MARKS = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e"), None
)
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ManifestRow:
    audio_path: str
    text: str
    duration: float
    source_id: str


def normalize_hebrew_text(text: str) -> str:
    """Apply conservative Unicode/whitespace cleanup without changing pronunciation."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE_MARKS)
    return _WHITESPACE.sub(" ", text).strip()


def source_id_from_row(row: dict) -> str:
    original_path = row.get("original_path")
    if original_path:
        return Path(str(original_path)).parent.name
    return Path(str(row["audio_path"])).stem.split("_seg", maxsplit=1)[0]


def resolve_audio_path(row: dict, wavs_dir: Path | None) -> Path:
    original = Path(str(row["audio_path"]))
    if original.exists():
        return original.resolve()
    if wavs_dir is not None:
        rewritten = wavs_dir / original.name
        if rewritten.exists():
            return rewritten.resolve()
    return original


def read_arrow_rows(path: Path) -> Iterator[dict]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        raise RuntimeError("Reading F5 Arrow data requires pyarrow: pip install pyarrow") from exc

    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.open_file(source)
        except pa.ArrowInvalid:
            reader = ipc.open_stream(source)
        batches = (
            (reader.get_batch(index) for index in range(reader.num_record_batches))
            if hasattr(reader, "num_record_batches")
            else reader
        )
        for batch in batches:
            for row in batch.to_pylist():
                yield row


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc


def write_jsonl(path: Path, rows: list[ManifestRow] | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = asdict(row) if isinstance(row, ManifestRow) else row
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
