"""Digit and niqqud text variants: the pure logic, no audio and no pocket_tts needed."""

import json
import random
from pathlib import Path

import pytest

from hebrew_training import build_official_manifest as builder
from hebrew_training.nikud import strip_marks
from training import text_augment
from training.text_augment import (
    TextAugment,
    check_probability,
    choose_swaps,
    derive_seed,
    render_text,
    swap_alternates,
    vocalized_forms,
)

FIXTURE = Path(__file__).parent / "fixtures" / "valid_diac.jsonl"

WORDS = ["a", "b", "c", "d", "e", "f", "g", "h"]
NIKUD = [f"{w}\u05b8" for w in WORDS]  # a mark on every word, so a changed word is visible
# Three spans: b..c (2 words) -> "1", e (1 word) -> "2", g..h -> "3".
SPANS = [
    {"from": 1, "to": 2, "replace": ["1"]},
    {"from": 4, "to": 4, "replace": ["2"]},
    {"from": 6, "to": 7, "replace": ["3"]},
]
ALT = {"digits": SPANS}

OFF = TextAugment(0.0, 0.0, 0.0)


def real_rows() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def render(first_word=0, seed=1, nikud=NIKUD, alt=ALT, **probs):
    aug = TextAugment(**{"alt_words_sample_prob": 0, "alt_word_swap_prob": 0, **probs})
    return render_text(WORDS, nikud, alt, first_word, seed, aug).split()


# ---------------------------------------------------------------- range validation


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2, float("nan"), float("inf"), "0.5", None, True])
def test_probabilities_outside_zero_one_are_rejected(bad):
    with pytest.raises(ValueError):
        check_probability("p", bad)
    for name in ("alt_words_sample_prob", "alt_word_swap_prob", "nikud_word_prob"):
        with pytest.raises(ValueError, match=name):
            TextAugment(**{name: bad})


@pytest.mark.parametrize("edge", [0, 1, 0.0, 1.0, 0.5])
def test_the_closed_range_is_accepted(edge):
    assert check_probability("p", edge) == float(edge)
    TextAugment(edge, edge, edge)


def test_defaults_are_the_specified_ones():
    aug = TextAugment()
    assert (aug.alt_words_sample_prob, aug.alt_word_swap_prob, aug.nikud_word_prob) == (
        0.30,
        0.60,
        0.10,
    )


# ---------------------------------------------------------------- swap_alternates


def test_swap_alternates_matches_the_builders_reference():
    rng = random.Random(7)
    for _ in range(300):
        words = [f"w{i}" for i in range(12)]
        spans, position = [], 0
        while position < 11 and rng.random() < 0.8:
            first = rng.randint(position, 10)
            last = rng.randint(first, min(first + 2, 11))
            spans.append(
                {"from": first, "to": last, "replace": [f"d{first}", "x"][: rng.randint(1, 2)]}
            )
            position = last + 2
        builder.validate_alt_words({"digits": spans}, len(words))
        first_word = rng.randint(0, 12)
        use = [rng.random() < 0.5 for _ in spans]
        assert swap_alternates(words, spans, first_word, use) == builder.swap_alternates(
            words, spans, first_word, use
        )
        assert swap_alternates(words, spans, first_word) == builder.swap_alternates(
            words, spans, first_word
        )


# ---------------------------------------------------------------- vocalized words


def test_nikud_prob_zero_changes_nothing_and_one_changes_every_word():
    assert render(nikud_word_prob=0) == WORDS
    assert render(nikud_word_prob=1) == NIKUD


def test_nikud_only_applies_to_words_after_the_cut():
    assert render(first_word=3, nikud_word_prob=1) == NIKUD[3:]
    assert render(first_word=3, nikud_word_prob=0) == WORDS[3:]


def test_nikud_has_no_at_least_one_rule():
    # A tiny probability over a short row: some seeds must come out entirely plain.
    plain = [render(seed=s, nikud_word_prob=0.01) == WORDS for s in range(200)]
    assert sum(plain) > 150


def test_a_words_fate_does_not_depend_on_the_cut():
    for seed in range(50):
        whole = render(first_word=0, seed=seed, nikud_word_prob=0.5)
        for cut in (2, 5):
            assert render(first_word=cut, seed=seed, nikud_word_prob=0.5) == whole[cut:]


def test_nikud_of_the_wrong_length_is_an_error():
    with pytest.raises(ValueError, match="nikud has"):
        vocalized_forms(WORDS, NIKUD[:-1], 0, 1.0, random.Random(0))


# ---------------------------------------------------------------- alternate words


def test_alt_sample_prob_zero_never_swaps():
    for seed in range(100):
        assert render(seed=seed, nikud=None, alt_words_sample_prob=0, alt_word_swap_prob=1) == WORDS


def test_alt_prob_one_swaps_every_span():
    out = render(nikud=None, alt_words_sample_prob=1, alt_word_swap_prob=1)
    assert out == ["a", "1", "d", "2", "f", "3"]


def test_swap_prob_zero_still_swaps_exactly_one_span_of_a_chosen_sample():
    for seed in range(100):
        out = render(seed=seed, nikud=None, alt_words_sample_prob=1, alt_word_swap_prob=0)
        swapped = [t for t in out if t in {"1", "2", "3"}]
        assert len(swapped) == 1, out


def test_the_forced_swap_picks_among_surviving_candidates_only():
    # Cut at 3: span 0 (b..c) is before the cut, so only "2" and "3" can ever be forced.
    seen = set()
    for seed in range(300):
        out = render(first_word=3, seed=seed, nikud=None, alt_words_sample_prob=1,
                     alt_word_swap_prob=0)  # fmt: skip
        assert "1" not in out
        assert not {"b", "c"} & set(out)
        swapped = [t for t in out if t in {"2", "3"}]
        assert len(swapped) == 1
        seen.add(swapped[0])
    assert seen == {"2", "3"}, "the forced pick should reach every candidate"


def test_a_span_cut_through_by_the_boundary_stays_spoken():
    # first_word=2 falls inside b..c: "c" stays as spoken, and the span is no candidate.
    for seed in range(50):
        out = render(first_word=2, seed=seed, nikud=None, alt_words_sample_prob=1,
                     alt_word_swap_prob=1)  # fmt: skip
        assert out[0] == "c"
        assert "1" not in out
        assert out == ["c", "d", "2", "f", "3"]


def test_no_surviving_candidate_means_no_swap_even_at_probability_one():
    assert render(first_word=8, nikud=None, alt_words_sample_prob=1, alt_word_swap_prob=1) == []
    out = render(first_word=7, nikud=None, alt_words_sample_prob=1, alt_word_swap_prob=1)
    assert out == ["h"]  # span g..h is cut through; spoken


def test_unknown_alt_labels_are_ignored():
    alt = {"percent": SPANS}
    assert render(nikud=None, alt=alt, alt_words_sample_prob=1, alt_word_swap_prob=1) == WORDS


def test_broken_spans_are_an_error_not_a_misaligned_text():
    bad = {"digits": [{"from": 6, "to": 9, "replace": ["3"]}]}
    with pytest.raises(ValueError, match="alt_words span"):
        render(nikud=None, alt=bad, alt_words_sample_prob=1)
    unordered = {"digits": [SPANS[1], SPANS[0]]}
    with pytest.raises(ValueError, match="alt_words span"):
        render(nikud=None, alt=unordered, alt_words_sample_prob=1)


def test_choose_swaps_outcomes_do_not_depend_on_the_cut_for_shared_spans():
    # Same per-sample draws, a later cut: a span that is still a candidate keeps its fate
    # unless the forced pick had to move because earlier candidates disappeared.
    for seed in range(100):
        before = choose_swaps(SPANS, 0, 1.0, 0.5, random.Random(seed))
        after = choose_swaps(SPANS, 3, 1.0, 0.5, random.Random(seed))
        assert after[0] is False
        if any(before[1:]):
            assert before[1:] == after[1:] or not any(after[1:])


# ---------------------------------------------------------------- composition


def test_nikud_is_drawn_first_and_a_swapped_span_is_plain_digits():
    out = render(nikud_word_prob=1, alt_words_sample_prob=1, alt_word_swap_prob=1)
    assert out == [NIKUD[0], "1", NIKUD[3], "2", NIKUD[5], "3"]
    assert not any(strip_marks(t) != t for t in ("1", "2", "3"))


def test_unswapped_spans_keep_their_niqqud():
    out = render(nikud_word_prob=1, alt_words_sample_prob=1, alt_word_swap_prob=0, seed=3)
    assert len([t for t in out if t in {"1", "2", "3"}]) == 1
    assert all(t in NIKUD or t in {"1", "2", "3"} for t in out)


def test_the_two_features_draw_independently():
    """Changing one probability must not change the other feature's outcome."""
    for seed in range(30):
        a = render(seed=seed, nikud=None, alt_words_sample_prob=1, alt_word_swap_prob=0.5)
        b = render(seed=seed, nikud=NIKUD, nikud_word_prob=0.5, alt_words_sample_prob=1,
                   alt_word_swap_prob=0.5)  # fmt: skip
        # Same spans swapped (digits are identical in both), whatever the niqqud draw did.
        assert [t for t in a if t.isdigit()] == [t for t in b if t.isdigit()]


# ---------------------------------------------------------------- rows without keys


def test_a_row_without_a_seed_is_plain_words_and_takes_no_draw(monkeypatch):
    def no_rng(*args, **kwargs):
        raise AssertionError("a row without alt_words/nikud must not create a generator")

    monkeypatch.setattr(text_augment.random, "Random", no_rng)
    state = random.getstate()
    aug = TextAugment(1.0, 1.0, 1.0)
    assert render_text(WORDS, None, None, 0, None, aug) == " ".join(WORDS)
    assert render_text(WORDS, None, None, 3, None, aug) == " ".join(WORDS[3:])
    # even if the keys were present, no seed means no augmentation and no draw
    assert render_text(WORDS, NIKUD, ALT, 3, None, aug) == " ".join(WORDS[3:])
    assert random.getstate() == state


def test_wants_is_false_for_rows_with_nothing_to_vary():
    aug = TextAugment()
    assert not aug.wants(False, False)
    assert aug.wants(True, False)
    assert aug.wants(False, True)
    assert not OFF.wants(True, True)
    assert not TextAugment(alt_words_sample_prob=0).wants(True, False)
    assert TextAugment(alt_words_sample_prob=0).wants(True, True)


# ---------------------------------------------------------------- reproducibility


def test_the_same_seed_gives_the_same_text():
    aug = TextAugment(0.7, 0.5, 0.4)
    for seed in range(50):
        first = render_text(WORDS, NIKUD, ALT, 1, seed, aug)
        assert render_text(WORDS, NIKUD, ALT, 1, seed, aug) == first


def test_the_global_random_state_is_never_used():
    random.seed(1234)
    state = random.getstate()
    render_text(WORDS, NIKUD, ALT, 0, 5, TextAugment(1, 0.5, 0.5))
    assert random.getstate() == state


def test_derive_seed_is_stable_across_processes_and_separates_rows():
    # A literal, so a change of the hash (which would reshuffle every validation set) is seen.
    assert derive_seed(0, "/a/b.wav", 12.5) == 4089429811943107491
    seeds = {
        derive_seed(0, "/a/b.wav", 12.5),
        derive_seed(1, "/a/b.wav", 12.5),
        derive_seed(0, "/a/c.wav", 12.5),
        derive_seed(0, "/a/b.wav", 12.6),
    }
    assert len(seeds) == 4


# ---------------------------------------------------------------- empirical rates


def test_default_rates_over_many_draws():
    """The defaults: 10% of words vocalized, 30% of samples with a swap, 60% per span."""
    draws = 4000
    defaults = TextAugment()
    # Niqqud alone, so words inside swapped spans do not hide marks from the count.
    nikud_only = TextAugment(0.0, defaults.alt_word_swap_prob, defaults.nikud_word_prob)
    marked = sum(
        token in NIKUD
        for seed in range(draws)
        for token in render_text(WORDS, NIKUD, ALT, 0, seed, nikud_only).split()
    )
    assert abs(marked / (draws * len(WORDS)) - 0.10) < 0.01

    chosen = spans_swapped = 0
    alt_only = TextAugment(defaults.alt_words_sample_prob, defaults.alt_word_swap_prob, 0.0)
    for seed in range(draws):
        out = render_text(WORDS, None, ALT, 0, seed, alt_only).split()
        swaps = [t for t in out if t in {"1", "2", "3"}]
        chosen += bool(swaps)
        spans_swapped += len(swaps)
    assert abs(chosen / draws - 0.30) < 0.03
    # P(span swapped | chosen) = 0.6, plus the forced pick when none was: 0.4**3 / 3 each.
    expected = 0.6 + 0.4**3 / 3
    assert abs(spans_swapped / (chosen * len(SPANS)) - expected) < 0.04


# ---------------------------------------------------------------- a real row


@pytest.fixture(scope="module")
def digit_row():
    row = real_rows()[0]
    assert row["alt_words"] and row["nikud"], "fixture row must carry both keys"
    return row


def test_fixture_rows_keep_the_manifest_invariants(digit_row):
    words = [w["word"] for w in digit_row["words"]]
    builder.validate_alt_words(digit_row["alt_words"], len(words))
    assert [strip_marks(x) for x in digit_row["nikud"]] == words
    assert "alt_words" not in real_rows()[1]


def test_real_row_nikud_only(digit_row):
    words = [w["word"] for w in digit_row["words"]]
    out = render_text(words, digit_row["nikud"], digit_row["alt_words"], 5, 1, TextAugment(0, 0, 1))
    assert out == " ".join(digit_row["nikud"][5:])
    assert strip_marks(out) == " ".join(words[5:])


def test_real_row_digits_only(digit_row):
    words = [w["word"] for w in digit_row["words"]]
    spans = digit_row["alt_words"]["digits"]
    out = render_text(words, digit_row["nikud"], digit_row["alt_words"], 0, 1, TextAugment(1, 1, 0))
    assert out == " ".join(builder.swap_alternates(words, spans))
    assert "2025" in out.split() and "ב-14:30" in out.split()


def test_real_row_both_features_compose(digit_row):
    words = [w["word"] for w in digit_row["words"]]
    nikud = digit_row["nikud"]
    spans = digit_row["alt_words"]["digits"]
    cut = 8  # inside the second span's neighbourhood: spans 0 (5..6) is before the cut
    out = render_text(words, nikud, digit_row["alt_words"], cut, 9, TextAugment(1, 1, 1)).split()
    expected = builder.swap_alternates(nikud, spans, cut)
    assert out == expected
    assert out[:2] == [nikud[8], nikud[9]]  # words before the first surviving span keep marks


def test_real_row_every_default_draw_is_a_valid_variant(digit_row):
    words = [w["word"] for w in digit_row["words"]]
    nikud = digit_row["nikud"]
    spans = digit_row["alt_words"]["digits"]
    replacements = {tuple(s["replace"]) for s in spans}
    for seed in range(200):
        cut = seed % 12
        out = render_text(words, nikud, digit_row["alt_words"], cut, seed, TextAugment()).split()
        # Every token is a plain word, a vocalized word, or a span's digits; no span is cut up.
        allowed = set(words) | set(nikud) | {t for r in replacements for t in r}
        assert set(out) <= allowed
        survivors = [s for s in spans if s["from"] >= cut]
        if not survivors:
            assert strip_marks(" ".join(out)) == " ".join(words[cut:])
