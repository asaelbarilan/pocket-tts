"""The character set a Hebrew tokenizer is allowed to see, and the filter that enforces it.

Tokenizer text comes from places nobody curated: web crawls, other people's corpora, model
output. A single Cyrillic or CJK page is enough to spend vocabulary pieces on scripts a Hebrew
TTS will never be asked to read, so every text that reaches SentencePiece goes through here.

Allowed, by category (each can be switched off with `allow`):

    latin        A-Z a-z and the accented Latin letters (é, ü, ø, ß, ...), which are English
                 and European loanword text rather than another script. Every letter whose
                 Unicode name starts with LATIN, plus combining accents U+0300-U+036F.
    hebrew       the letters U+05D0-U+05EA, plus the Hebrew punctuation marks
                 (maqaf, geresh, gershayim, sof pasuq, paseq).
    niqqud       the vowel points and dagesh/shin/sin dots U+05B0-U+05BD, U+05BF, U+05C1,
                 U+05C2, U+05C7.
    cantillation the Hebrew cantillation marks (trope), U+0591-U+05AF.
    digits       0-9.
    punctuation  ASCII punctuation, the typographic quotes/dashes/ellipsis, and the symbols
                 a sentence reads aloud (% $ € ₪ £ ¥ °).

Always removed rather than judged: zero-width and bidi control characters, other control
characters, variation selectors and emoji. Every other kind of space becomes a plain space.

Anything else is *foreign*. By default a line containing a foreign character is dropped
whole -- stripping the Cyrillic out of a Russian sentence leaves word-shaped garbage that is
worse than no sentence. `on_foreign="strip"` removes just the character instead.

`fold` rewrites typographic equivalents to the plain form hebrew-tts-data-tools'
normalizer produces (maqaf and Unicode dashes -> "-", curly/guillemet/gershayim quotes -> '"',
curly apostrophes and geresh -> "'"), so tokenizer text matches what a normalized manifest
carries. It is on by default; turn it off only if the manifest text is not normalized.
"""

from __future__ import annotations

import argparse
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

CATEGORIES = ("latin", "hebrew", "niqqud", "cantillation", "digits", "punctuation")
ON_FOREIGN = ("drop-line", "strip")

HEBREW_LETTERS = frozenset(chr(c) for c in range(0x05D0, 0x05EB))
NIQQUD = frozenset(chr(c) for c in (*range(0x05B0, 0x05BE), 0x05BF, 0x05C1, 0x05C2, 0x05C7))
# Hebrew punctuation that is not folded away: sof pasuq and paseq. Maqaf, geresh and gershayim
# are here too, so that `fold=False` still accepts them.
HEBREW_PUNCTUATION = frozenset("\u05be\u05c0\u05c3\u05f3\u05f4")
CANTILLATION = frozenset(chr(c) for c in range(0x0591, 0x05B0))
LATIN_LETTERS = frozenset(string.ascii_letters)
DIGITS = frozenset(string.digits)
PUNCTUATION = frozenset(
    string.punctuation
    + "\u2010\u2011\u2012\u2013\u2014\u2015"  # hyphens and dashes
    + "\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f"  # curly quotes
    + "\u00ab\u00bb\u2039\u203a"  # guillemets
    + "\u2026"  # ellipsis
    + "\u20aa\u20ac\u00a3\u00a5\u00b0"  # ₪ € £ ¥ °
)

# Same conversions as hebrew_tts_normalizer._CHAR_CONVERSION_DICT.
FOLD = {
    **dict.fromkeys("\u2010\u2011\u2012\u2013\u2014\u2015\u05be", "-"),
    **dict.fromkeys("\u00ab\u00bb\u201c\u201d\u201e\u201f\u05f4", '"'),
    **dict.fromkeys("\u2018\u2019\u201b\u05f3", "'"),
    "\\": "",
}

_REMOVED_RANGES = (
    (0x05C4, 0x05C6),  # upper/lower dot and nun hafukha: not niqqud, not cantillation
    (0xFE00, 0xFE0F),  # variation selectors
    # Emoji and pictographs, the same ranges hebrew_tts_normalizer._EMOJI_RE drops.
    (0x1F300, 0x1FAFF),
    (0x2600, 0x27BF),
)
_PRESENTATION_FORMS = (0xFB1D, 0xFB4F)  # precomposed Hebrew letter+point, split by NFKD

_NIQQUD_RATE = re.compile("[\u05b0-\u05bd\u05bf\u05c1\u05c2\u05c7]")
_WS = re.compile(r"\s+")


def _removed(ch: str) -> bool:
    code = ord(ch)
    if any(lo <= code <= hi for lo, hi in _REMOVED_RANGES):
        return True
    return unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co", "Cn") and ch not in "\t\n\r"


def is_latin_letter(ch: str) -> bool:
    """A-Z, a-z, and every accented or extended Latin letter (é, ø, ß, ﬁ, ...)."""
    return ch in LATIN_LETTERS or (
        unicodedata.category(ch).startswith("L") and unicodedata.name(ch, "").startswith("LATIN")
    )


def _is_combining_accent(ch: str) -> bool:
    return 0x0300 <= ord(ch) <= 0x036F


@dataclass
class FilterStats:
    lines: int = 0
    kept: int = 0
    changed: int = 0  # kept, but with something removed or folded
    dropped_foreign: int = 0
    dropped_empty: int = 0
    foreign: Counter = field(default_factory=Counter)

    def report(self, top: int = 8) -> str:
        out = [
            (
                f"charset filter: {self.lines} lines in, {self.kept} kept "
                f"({self.changed} rewritten), {self.dropped_foreign} dropped for foreign "
                f"characters, {self.dropped_empty} empty"
            )
        ]
        if self.foreign:
            seen = ", ".join(
                f"{ch!r} U+{ord(ch):04X} {unicodedata.name(ch, '?')} x{n}"
                for ch, n in self.foreign.most_common(top)
            )
            out.append(f"  most common foreign characters: {seen}")
        return "\n".join(out)


class CharsetFilter:
    """`clean(line)` -> the line restricted to the allowed characters, or None if it has none."""

    def __init__(
        self,
        allow: tuple[str, ...] | list[str] = CATEGORIES,
        on_foreign: str = "drop-line",
        fold: bool = True,
    ) -> None:
        unknown = set(allow) - set(CATEGORIES)
        if unknown:
            raise ValueError(
                f"unknown charset categories {sorted(unknown)}; choose from {CATEGORIES}"
            )
        if on_foreign not in ON_FOREIGN:
            raise ValueError(f"on_foreign must be one of {ON_FOREIGN}, got {on_foreign!r}")
        self.allow = tuple(c for c in CATEGORIES if c in set(allow))
        self.on_foreign = on_foreign
        self.fold = fold
        allowed: set[str] = {" "}
        if "latin" in self.allow:
            allowed |= LATIN_LETTERS
        if "hebrew" in self.allow:
            allowed |= HEBREW_LETTERS | HEBREW_PUNCTUATION
        if "niqqud" in self.allow:
            allowed |= NIQQUD
        if "cantillation" in self.allow:
            allowed |= CANTILLATION
        if "digits" in self.allow:
            allowed |= DIGITS
        if "punctuation" in self.allow:
            allowed |= PUNCTUATION
        self.allowed = frozenset(allowed)
        self.stats = FilterStats()
        self._cache: dict[str, str | None] = {}

    def _map(self, ch: str) -> str | None:
        """What `ch` becomes: a string (possibly empty), or None if it is foreign."""
        if self.fold and ch in FOLD:
            ch = FOLD[ch]
        if ch in self.allowed:
            return ch
        if ch in ("", " "):
            return ch
        if _removed(ch):
            return ""
        if ch.isspace():  # tabs, newlines, NBSP and every other Unicode space
            return " "
        if _PRESENTATION_FORMS[0] <= ord(ch) <= _PRESENTATION_FORMS[1]:
            parts = [self._map(c) for c in unicodedata.normalize("NFKD", ch)]
            return None if any(p is None for p in parts) else "".join(parts)
        if "latin" in self.allow and (is_latin_letter(ch) or _is_combining_accent(ch)):
            return ch
        return None

    def clean(self, line: str) -> str | None:
        self.stats.lines += 1
        original = unicodedata.normalize("NFC", line)
        out: list[str] = []
        bad: list[str] = []
        for ch in original:
            mapped = self._cache.get(ch, 0)
            if mapped == 0:
                mapped = self._cache[ch] = self._map(ch)
            if mapped is None:
                bad.append(ch)
            else:
                out.append(mapped)
        if bad:
            self.stats.foreign.update(bad)
            if self.on_foreign == "drop-line":
                self.stats.dropped_foreign += 1
                return None
        text = _WS.sub(" ", "".join(out)).strip()
        if not text:
            self.stats.dropped_empty += 1
            return None
        self.stats.kept += 1
        if text != _WS.sub(" ", original).strip():
            self.stats.changed += 1
        return text

    def __call__(self, line: str) -> str | None:
        return self.clean(line)

    def describe(self) -> str:
        return (
            f"allow={','.join(self.allow) or 'nothing'} on_foreign={self.on_foreign} "
            f"fold={'on' if self.fold else 'off'}"
        )


def niqqud_rate(text: str) -> float:
    """Niqqud marks as a share of all characters; vocalized Hebrew is roughly 25%."""
    return len(_NIQQUD_RATE.findall(text)) / max(1, len(text))


def has_hebrew(text: str) -> bool:
    return any(c in HEBREW_LETTERS for c in text)


def has_latin(text: str) -> bool:
    return any(is_latin_letter(c) for c in text)


def add_charset_args(parser: argparse.ArgumentParser, allow_disable: bool = True) -> None:
    group = parser.add_argument_group("charset safety filter")
    group.add_argument(
        "--allow",
        default=",".join(CATEGORIES),
        help=f"Comma-separated categories to keep (default: all of {', '.join(CATEGORIES)}).",
    )
    group.add_argument(
        "--on-foreign",
        choices=ON_FOREIGN,
        default="drop-line",
        help="What to do with a line that has a character outside the allowed set.",
    )
    group.add_argument(
        "--no-fold",
        action="store_true",
        help="Keep typographic quotes/dashes as they are instead of folding them to the "
        "forms hebrew-tts-data-tools' normalizer produces.",
    )
    if allow_disable:
        group.add_argument(
            "--no-charset-filter",
            action="store_true",
            help="Disable the filter entirely. Anything in the input reaches the tokenizer.",
        )


def filter_from_args(args: argparse.Namespace) -> CharsetFilter | None:
    if getattr(args, "no_charset_filter", False):
        return None
    return CharsetFilter(
        allow=[c.strip() for c in args.allow.split(",") if c.strip()],
        on_foreign=args.on_foreign,
        fold=not args.no_fold,
    )
