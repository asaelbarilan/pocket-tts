"""File names and input checks shared by the manifest stages.

One folder, one name per stage, so a file's name says which stages produced it:

    train.jsonl / valid.jsonl                  build_official_manifest
    train_diac.jsonl / valid_diac.jsonl        diacritize_official_manifest   (optional)
    train_aligned.jsonl / valid_aligned.jsonl  align_official_manifest        (optional)

Every stage after the first reads the row a previous stage wrote and returns it with its own
keys added -- it never rewrites what another stage owns. That is what makes the two optional
stages independent: `words[i]` keeps its text and its position (the aligner changes only the
times), `nikud[i]` is a parallel list of strings (the diacritizer adds it and never touches
`words`), and `alt_words` indices stay valid because neither stage adds or drops a word.

The aligner runs from its own virtualenv by file path, so it cannot import this module and
keeps its own copy of the names and checks; tests/test_align_official_manifest.py pins the two
together.
"""

from __future__ import annotations

import json
from pathlib import Path

SPLITS = ("train", "valid")
BUILT = "{split}.jsonl"
DIACRITIZED = "{split}_diac.jsonl"
ALIGNED = "{split}_aligned.jsonl"

VERIFY_ROWS = 20


def head_rows(path: Path, count: int = VERIFY_ROWS) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path} is not a JSON-lines manifest: {error}") from error
            rows.append(row)
            if len(rows) == count:
                break
    return rows


def check_manifest_head(
    path: Path, *, expected_from: str, forbidden: tuple[str, ...] = (), count: int = VERIFY_ROWS
) -> None:
    """Fail up front, naming the file, when its first rows are not what the stage reads.

    `expected_from` names the stage that should have written it. `forbidden` lists keys that
    mean the stage already ran on this file."""
    rows = head_rows(path, count)
    if not rows:
        raise SystemExit(f"{path} is empty; expected the output of {expected_from}")
    for number, row in enumerate(rows, 1):
        problem = _row_problem(row, forbidden)
        if problem:
            raise SystemExit(
                f"{path} does not look like the output of {expected_from}: row {number} {problem}"
            )


def _row_problem(row: object, forbidden: tuple[str, ...]) -> str | None:
    if not isinstance(row, dict):
        return "is not a JSON object"
    if not isinstance(row.get("path"), str):
        return "has no string `path`"
    for key in ("start", "duration"):
        if not isinstance(row.get(key), int | float):
            return f"has no numeric `{key}`"
    words = row.get("words")
    if not isinstance(words, list) or not words:
        return "has no `words` list"
    for word in words:
        if not (
            isinstance(word, dict)
            and isinstance(word.get("word"), str)
            and isinstance(word.get("start"), int | float)
            and isinstance(word.get("end"), int | float)
        ):
            return "has a `words` entry that is not {word, start, end}"
    for key in forbidden:
        if key in row:
            return f"already has `{key}`, so this stage has run on it"
    nikud = row.get("nikud")
    if nikud is not None and (not isinstance(nikud, list) or len(nikud) != len(words)):
        return "has a `nikud` list that is not one entry per word"
    return None
