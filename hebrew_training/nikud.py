"""Vocalized-word rows: stripping marks, validating the parallel `nikud` list, vocalizing.

A diacritized row carries `nikud`, a list with one string per entry of `words`: the same word
with niqqud (and whatever else the diacritizer adds) on it. `words` is never changed. The
invariant that makes stripping a plain character filter is

    strip_marks(nikud[i]) == words[i]

The diacritizer is asked to keep matres lectionis (`preserve_matres`), so it adds marks and
nothing else; it does not respell. Spelling differences (ktiv male against haser) are not
diacritics and are not handled here.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Sequence

from hebrew_training.tokenizer_charset import CANTILLATION, HEBREW_LETTERS, NIQQUD
from hebrew_training.tokenizer_text import MAX_LINE

MARKS = NIQQUD | CANTILLATION


def strip_marks(text: str) -> str:
    """NFC first, so presentation forms (U+FB2A ...) and mark order cannot hide a mark.

    Maqaf, paseq, sof pasuq and the geresh signs are punctuation, not marks, and stay."""
    return "".join(c for c in unicodedata.normalize("NFC", text) if c not in MARKS)


def has_marks(text: str) -> bool:
    return any(c in MARKS for c in unicodedata.normalize("NFC", text))


def has_hebrew_letter(text: str) -> bool:
    return any(c in HEBREW_LETTERS for c in text)


def validate_nikud(words: Sequence[str], nikud: Sequence[str]) -> None:
    """Raise ValueError unless `nikud` is `words` with marks added, word for word."""
    if len(nikud) != len(words):
        raise ValueError(f"nikud has {len(nikud)} entries for {len(words)} words")
    for index, (word, marked) in enumerate(zip(words, nikud, strict=True)):
        if not isinstance(marked, str) or not marked or marked != marked.strip() or " " in marked:
            raise ValueError(f"nikud[{index}] is not a single word: {marked!r}")
        if strip_marks(marked) != word:
            raise ValueError(f"nikud[{index}] {marked!r} does not strip back to {word!r}")


def word_chunks(words: Sequence[str], max_chars: int = MAX_LINE) -> list[tuple[int, int]]:
    """`(start, stop)` word slices whose joined text fits `max_chars`.

    The diacritizer silently truncates a line past about 400 characters (see
    tokenizer_text.MAX_LINE), which would drop the tail of a long row unnoticed. A single
    word longer than the limit gets a chunk of its own."""
    chunks, start, length = [], 0, 0
    for index, word in enumerate(words):
        added = len(word) + (1 if index > start else 0)
        if index > start and length + added > max_chars:
            chunks.append((start, index))
            start, length, added = index, 0, len(word)
        length += added
    if words:
        chunks.append((start, len(words)))
    return chunks


Predict = Callable[[list[str]], list[str]]


def vocalize_word_lists(
    word_lists: Sequence[Sequence[str]],
    predict: Predict,
    batch_size: int = 32,
    max_chars: int = MAX_LINE,
) -> list[list[str] | str]:
    """Vocalize many rows at once. One result per row: the marked words, or a reason string.

    Rows are cut into chunks that fit the diacritizer, the chunks of all rows are sent in
    batches, and each row is put back together and checked with `validate_nikud`. A row whose
    result does not strip back to its words is reported, never repaired."""
    texts: list[str] = []
    owners: list[tuple[int, int, int]] = []
    for row, words in enumerate(word_lists):
        for start, stop in word_chunks(words, max_chars):
            texts.append(" ".join(words[start:stop]))
            owners.append((row, start, stop))
    batches = [texts[first : first + batch_size] for first in range(0, len(texts), batch_size)]
    # A predictor that spreads work over devices takes all the batches at once.
    spread = getattr(predict, "map_batches", None)
    outputs = spread(batches) if spread else [predict(batch) for batch in batches]
    marked: list[str] = []
    for batch, out in zip(batches, outputs, strict=True):
        if len(out) != len(batch):
            raise RuntimeError(f"diacritizer returned {len(out)} lines for {len(batch)}")
        marked.extend(out)

    pieces: list[list[str] | str] = [[] for _ in word_lists]
    for (row, start, stop), line in zip(owners, marked, strict=True):
        if isinstance(pieces[row], str):
            continue
        parts = unicodedata.normalize("NFC", line).split()
        if len(parts) != stop - start:
            pieces[row] = "word_count_changed"
        else:
            pieces[row].extend(parts)
    for row, words in enumerate(word_lists):
        if isinstance(pieces[row], str):
            continue
        try:
            validate_nikud(words, pieces[row])
        except ValueError:
            pieces[row] = "content_changed"
    return pieces
