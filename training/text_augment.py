"""Per-sample text variants for training: the corpus's own spelling, vocalized words.

Pure Python on purpose: no audio, torch or pocket_tts import, so the logic (and its tests)
loads anywhere. `training/dataloader.py` calls `render_text` once per sample, after it has
chosen its cut.

Manifest rows can carry two optional keys next to `words` (see hebrew_training/NIQQUD.md and
the docstring of hebrew_training/build_official_manifest.py):

    "alt_words": {"norms": [{"from": i, "to": j, "replace": ["14"], "k": "n"}, ...]}
    "nikud": ["word with marks", ...]        # one per entry of `words`

`words` is always the plain, spoken text. The two features only ever change what is
trained on, never the audio:

* Alternate words. `words[from..to]` is something the normalizer rewrote (a spelled-out number,
  an expanded abbreviation, a spelled Latin letter, a split hyphen, a collapsed `...`, a
  stripped quote, ...) and `replace` is what the corpus wrote. `k`, a one-letter tag of the
  kind of rewrite, is for tooling and is not read here. With probability `alt_words_sample_prob` a sample gets alternates at
  all; then each span is swapped with probability `alt_word_swap_prob`, and if none was, one
  random span is swapped, so a chosen sample always has at least one swap. Only spans that
  start at or after the cut are candidates: a span the cut falls inside is left spoken (its
  first words' audio is gone), and a span before the cut is not in the text.
* Vocalized words. Each word after the cut is replaced by its `nikud` form with probability
  `nikud_word_prob`. No row-level gate and no at-least-one rule.

The nikud draw comes first and the swap works on those strings, so a swapped-in span is always
the corpus's own plain tokens.

REPRODUCIBILITY. Every draw comes from `random.Random` instances seeded from one integer per
sample (`text_seed`), never from a shared generator, so the result does not depend on which
thread ran first. Draws are also keyed by position, not by cut: word `j` always gets the same
uniform number, span `k` the same one, whatever the cut is. Two samples of one row with
different cuts therefore agree on every word they share. A row without the keys has no seed
and consumes nothing.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_ALT_WORDS_SAMPLE_PROB = 0.30
DEFAULT_ALT_WORD_SWAP_PROB = 0.60
DEFAULT_NIKUD_WORD_PROB = 0.10

# The only label of `alt_words` that exists. Others are ignored, not guessed at.
NORMS = "norms"


def check_probability(name: str, value: float) -> float:
    """`value` as a float, or ValueError unless it is within [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        # ValueError, like the range check: a config typo should fail the same way
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")  # noqa: TRY004
    if not 0.0 <= value <= 1.0:  # also rejects NaN
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return float(value)


@dataclass(frozen=True)
class TextAugment:
    # P(a sample with `alt_words` gets any alternate at all). 0 disables them.
    alt_words_sample_prob: float = DEFAULT_ALT_WORDS_SAMPLE_PROB
    # P(a span is swapped | the sample was chosen). The sample is forced to swap one span if
    # none was drawn, so 0 here means exactly one span per chosen sample, not "off".
    alt_word_swap_prob: float = DEFAULT_ALT_WORD_SWAP_PROB
    # P(a word of a sample with `nikud` is vocalized). 0 disables vocalization.
    nikud_word_prob: float = DEFAULT_NIKUD_WORD_PROB

    def __post_init__(self) -> None:
        for name in ("alt_words_sample_prob", "alt_word_swap_prob", "nikud_word_prob"):
            object.__setattr__(self, name, check_probability(name, getattr(self, name)))

    def wants(self, has_alt_words: bool, has_nikud: bool) -> bool:
        """Whether a row with these keys can be changed at all."""
        return (has_alt_words and self.alt_words_sample_prob > 0) or (
            has_nikud and self.nikud_word_prob > 0
        )


def derive_seed(seed: int, path: str, start: float) -> int:
    """A stable per-row seed. Built from SHA-256, not `hash()`, which varies per process."""
    digest = hashlib.sha256(f"{seed}\0{path}\0{start!r}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def swap_alternates(
    words: Sequence[str],
    spans: Sequence[dict],
    first_word: int = 0,
    use: Sequence[bool] | None = None,
) -> list[str]:
    """`words[first_word:]` with the chosen `spans` swapped for their `replace` words.

    Same behavior as `swap_alternates` in hebrew_training/build_official_manifest.py (the
    reference; training/tests/test_text_augment.py checks they agree). It is copied rather than
    imported because that module pulls in multiprocessing and subprocess. A span the cut
    falls inside (`from < first_word <= to`) is left spoken. `use[k]` says whether to swap
    `spans[k]`; default all.
    """
    out: list[str] = []
    position = first_word
    for number, span in enumerate(spans):
        if span["from"] < first_word or (use is not None and not use[number]):
            continue
        out.extend(words[position : span["from"]])
        out.extend(span["replace"])
        position = span["to"] + 1
    out.extend(words[position:])
    return out


def _check_spans(spans: Sequence[dict], word_count: int) -> None:
    """The structural half of the builder's `validate_alt_words`: in range, ordered, disjoint."""
    previous_to = -1
    for span in spans:
        first, last = span["from"], span["to"]
        if not 0 <= first <= last < word_count or first <= previous_to:
            raise ValueError(f"alt_words span {first}..{last} is bad for {word_count} words")
        if not span["replace"]:
            raise ValueError(f"alt_words span {first}..{last} has no replace words")
        previous_to = last


def vocalized_forms(
    words: Sequence[str], nikud: Sequence[str], first_word: int, prob: float, rng: random.Random
) -> list[str]:
    """`words`, with each one at or after `first_word` replaced by its `nikud` form at `prob`.

    One uniform number per word of the whole row, so a word's fate does not depend on the cut.
    """
    if len(nikud) != len(words):
        raise ValueError(f"nikud has {len(nikud)} entries for {len(words)} words")
    draws = [rng.random() for _ in words]
    return [
        nikud[index] if index >= first_word and draws[index] < prob else word
        for index, word in enumerate(words)
    ]


def choose_swaps(
    spans: Sequence[dict], first_word: int, sample_prob: float, swap_prob: float, rng: random.Random
) -> list[bool]:
    """`use` for `swap_alternates`: which spans to swap, honouring the at-least-one rule.

    The fixed set of draws (sample, one per span, forced pick) is taken up front, so the
    outcome for a span does not depend on the cut either; only the candidate set does.
    """
    sample_draw = rng.random()
    span_draws = [rng.random() for _ in spans]
    forced_draw = rng.random()
    candidates = [k for k, span in enumerate(spans) if span["from"] >= first_word]
    use = [False] * len(spans)
    if sample_draw >= sample_prob or not candidates:
        return use
    for k in candidates:
        use[k] = span_draws[k] < swap_prob
    if not any(use):
        use[candidates[int(forced_draw * len(candidates))]] = True
    return use


def render_text(
    words: Sequence[str],
    nikud: Sequence[str] | None,
    alt_words: dict | None,
    first_word: int,
    text_seed: int | None,
    aug: TextAugment,
) -> str:
    """The text for a sample whose audio starts at word `first_word`.

    `text_seed` None means no augmentation applies (the row has neither key, or both features
    are off for it): the plain words, no draw taken.
    """
    forms = list(words)
    spans = None
    if text_seed is not None:
        if nikud and aug.nikud_word_prob > 0:
            rng = random.Random(f"{text_seed}:nikud")
            forms = vocalized_forms(words, nikud, first_word, aug.nikud_word_prob, rng)
        spans = (alt_words or {}).get(NORMS)
    if spans and aug.alt_words_sample_prob > 0:
        _check_spans(spans, len(words))
        rng = random.Random(f"{text_seed}:alt")
        use = choose_swaps(
            spans, first_word, aug.alt_words_sample_prob, aug.alt_word_swap_prob, rng
        )
        return " ".join(swap_alternates(forms, spans, first_word, use))
    return " ".join(forms[first_word:])
