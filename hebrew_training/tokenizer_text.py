"""Read tokenizer text from a local file or folder, whatever shape it comes in.

A source is a path. A folder is searched recursively and its files read in sorted order, so a
run is repeatable. Supported files (each also as `.gz`):

    .txt      one text per line
    .jsonl    one JSON object per line; the text is its "text" or "transcript" field
              (FineWeb dumps and this project's manifests both fit). A line that is not
              JSON is taken as plain text.
    .parquet  a table with a "text" or "transcript" column (the format FineWeb is published
              in; needs pyarrow)

Nothing here touches the network.
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
from collections.abc import Iterator
from pathlib import Path

TEXT_FIELDS = ("text", "transcript")
SUFFIXES = (".txt", ".jsonl", ".parquet")

# A SentencePiece line should be a sentence, not a web page. Beyond this the diacritizer
# also starts truncating, which would silently drop the tail of every long paragraph.
MAX_LINE = 400
_WS = re.compile(r"\s+")


def lines_of(text: str, max_line: int = MAX_LINE) -> list[str]:
    """One sentence-ish line per entry: NFC, whitespace collapsed, long lines split at spaces."""
    out = []
    for raw in unicodedata.normalize("NFC", text).splitlines():
        line = _WS.sub(" ", raw).strip()
        while len(line) > max_line:
            cut = line.rfind(" ", 0, max_line)
            cut = cut if cut > max_line // 2 else max_line
            out.append(line[:cut].strip())
            line = line[cut:].strip()
        if len(line) > 1:
            out.append(line)
    return out


def _kind(path: Path) -> str | None:
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        suffixes = suffixes[:-1]
    return suffixes[-1] if suffixes and suffixes[-1] in SUFFIXES else None


def find_files(path: Path) -> list[Path]:
    """The readable files under `path` (or `path` itself), sorted. Raises if there are none."""
    if not path.exists():
        raise FileNotFoundError(f"text source {path} does not exist")
    if path.is_file():
        if _kind(path) is None:
            raise ValueError(f"{path}: unsupported file type, expected one of {SUFFIXES} (or .gz)")
        return [path]
    files = sorted(p for p in path.rglob("*") if p.is_file() and _kind(p))
    if not files:
        raise FileNotFoundError(f"{path} contains no {'/'.join(SUFFIXES)} files")
    return files


def _open(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _record_text(record: object) -> str:
    if isinstance(record, dict):
        for name in TEXT_FIELDS:
            value = record.get(name)
            if isinstance(value, str):
                return value
    return ""


def _parquet_texts(path: Path) -> Iterator[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise SystemExit(
            f"{path} is parquet, which needs pyarrow: `uv sync --extra training` "
            "or `pip install pyarrow`."
        ) from error
    handle = pq.ParquetFile(path)
    column = next((n for n in TEXT_FIELDS if n in handle.schema_arrow.names), None)
    if column is None:
        raise ValueError(
            f"{path} has none of the columns {TEXT_FIELDS}: {handle.schema_arrow.names}"
        )
    for batch in handle.iter_batches(columns=[column]):
        for value in batch.column(0).to_pylist():
            if isinstance(value, str):
                yield value


def iter_documents(path: Path) -> Iterator[str]:
    """Every text under `path`, one string per line / record / row, in a stable order."""
    for file in find_files(Path(path)):
        kind = _kind(file)
        if kind == ".parquet":
            yield from _parquet_texts(file)
            continue
        with _open(file) as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if kind == ".jsonl" and line.startswith("{"):
                    try:
                        text = _record_text(json.loads(line))
                    except json.JSONDecodeError:
                        text = line
                else:
                    text = line
                if text:
                    yield text
