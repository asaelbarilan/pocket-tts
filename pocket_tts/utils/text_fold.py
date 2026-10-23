"""Character-level folds that training text went through, so inference can apply the same.

The Hebrew manifests are built with hebrew-tts-data-tools' normalizer, and the tokenizer is
trained on text folded the same way (`hebrew_training/tokenizer_charset.py`), with
`normalization_rule_name="identity"` and byte fallback. A curly apostrophe (U+2019) therefore
never appeared in training; at inference it is three byte tokens the model has not learned.
`fold_characters` maps those characters to the plain forms the model did see.

This module is the single definition: the tokenizer filter imports `FOLD` from here, the
model's text preparation calls `fold_characters` (when the config sets `fold_text: true`), and
`tests/test_text_fold.py` checks `NORMALIZER_CHAR_MAP` against the normalizer's own
`_CHAR_CONVERSION_DICT`, so a change on either side fails a test instead of drifting.

Only characters are folded. Rewrites that need context (numbers, abbreviations, `ה-API`,
punctuation collapsing) are not applied at inference: the training loader teaches the model
the raw forms instead (`alt_words`, see hebrew_training/build_official_manifest.py).

Pure Python on purpose: no torch, so it can be imported anywhere.
"""

from __future__ import annotations

# Visible characters with a plain equivalent. Same conversions as the normalizer's
# `_CHAR_CONVERSION_DICT`, minus its removals (those are in NORMALIZER_CHAR_MAP).
FOLD: dict[str, str] = {
    **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u05be", "-"),
    **dict.fromkeys("\u00ab\u00bb\u201c\u201d\u201e\u201f\u05f4", '"'),
    **dict.fromkeys("\u2018\u2019\u201b\u05f3", "'"),
    "\\": "",
}

# Characters the normalizer deletes or turns into a blank: control characters, zero-width and
# bidi marks, and tab/newline. Not folded to a visible character, so not in FOLD.
_REMOVED = "\u0001\u0007\u200a\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
_BLANKS = "\t\n"

# Everything `hebrew_tts_normalizer._CHAR_CONVERSION_DICT` does, character for character.
NORMALIZER_CHAR_MAP: dict[str, str] = {
    **FOLD,
    **dict.fromkeys(_REMOVED, ""),
    **dict.fromkeys(_BLANKS, " "),
}

_TABLE = str.maketrans(NORMALIZER_CHAR_MAP)


def fold_characters(text: str) -> str:
    """`text` with every character the training normalizer rewrites mapped to its plain form."""
    return text.translate(_TABLE)
