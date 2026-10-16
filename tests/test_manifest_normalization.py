"""Context-sensitive text normalization in build_official_manifest.

These run the real normalizer from the sibling hebrew-tts-data-tools checkout (the numbers'
gender comes from its tables, so a stub would test nothing) and are skipped without it.
"""

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from hebrew_training import build_official_manifest as manifest


@pytest.fixture(scope="module")
def normalize():
    if not (manifest.DEFAULT_NORMALIZER / "hebrew_tts_normalizer.py").is_file():
        pytest.skip("hebrew-tts-data-tools checkout not found beside this repo")
    try:
        return manifest.load_normalizer(manifest.DEFAULT_NORMALIZER)
    except SystemExit as error:
        pytest.skip(str(error))


def _words(text: str) -> list[dict]:
    return [
        {"word": word, "start": float(index), "end": float(index + 1), "probability": 0.9}
        for index, word in enumerate(text.split())
    ]


def _spoken(words: list[dict]) -> list[str]:
    return [word["word"] for word in words]


def _normalized(normalize, text: str, **context) -> str:
    return " ".join(_spoken(manifest.normalize_words(_words(text), normalize, **context)))


def test_number_gender_follows_the_next_word(normalize) -> None:
    assert _normalized(normalize, "14 ימים") == "ארבעה עשר ימים"
    assert _normalized(normalize, "14 שנה") == "ארבע עשרה שנה"
    # "שנים" looks masculine but is a feminine noun.
    assert _normalized(normalize, "14 שנים") == "ארבע עשרה שנים"
    assert _normalized(normalize, "2 ימים") == "שני ימים"
    assert _normalized(normalize, "הגיעו 21 פצועים") == "הגיעו עשרים ואחד פצועים"


def test_each_number_uses_its_own_following_word(normalize) -> None:
    assert (
        _normalized(normalize, "אחרי 14 ימים ו-3 שנים וגם 2 אנשים")
        == "אחרי ארבעה עשר ימים ושלוש שנים וגם שני אנשים"
    )


def test_prefix_attached_to_a_number_still_sees_the_next_word(normalize) -> None:
    assert _normalized(normalize, "ב-15 ימים") == "בחמישה עשר ימים"
    assert _normalized(normalize, "ו-15 שנים") == "וחמש עשרה שנים"
    # The prefix and the number come from one input word and stay attached to it.
    out = manifest.normalize_words(_words("ב-15 ימים"), normalize)
    assert _spoken(out) == ["בחמישה", "עשר", "ימים"]
    assert [word["start"] for word in out] == pytest.approx([0.0, 0.5, 1.0])


def test_previous_word_changes_the_gender(normalize) -> None:
    assert _normalized(normalize, "פי 12") == "פי שנים עשר"
    assert _normalized(normalize, "במאה 3") == "במאה שלישית"
    assert _normalized(normalize, "12") == "שתים עשרה"
    assert _normalized(normalize, "3") == "שלוש"


def test_numbers_at_the_edges(normalize) -> None:
    assert _normalized(normalize, "14 ימים") == "ארבעה עשר ימים"
    assert _normalized(normalize, "ימים 14") == "ימים ארבע עשרה"
    # No neighbour at all: the feminine default.
    assert _normalized(normalize, "14") == "ארבע עשרה"


def test_context_decides_a_number_at_the_edge(normalize) -> None:
    assert _normalized(normalize, "14", after=["ימים"]) == "ארבעה עשר"
    assert _normalized(normalize, "14", after=["שנה"]) == "ארבע עשרה"
    assert _normalized(normalize, "12", before=["פי"]) == "שנים עשר"
    assert _normalized(normalize, "3", before=["נולד", "במאה"]) == "שלישית"
    assert _normalized(normalize, "ימים 14", before=["כך"], after=["ימים"]) == "ימים ארבעה עשר"


def test_context_words_never_leak_into_the_output(normalize) -> None:
    out = manifest.normalize_words(
        _words("14"), normalize, before=["פי", "מאה", "5"], after=["ימים", "20", "."]
    )
    assert _spoken(out) == ["ארבעה", "עשר"]

    out = manifest.normalize_words(
        _words("הלך"), normalize, before=["פי", "3"], after=[".", "ואז", "7"]
    )
    assert _spoken(out) == ["הלך"]

    # Nothing outside `words` shows up in the timed words either.
    out = manifest.normalize_words(_words("שלום 14"), normalize, before=["x"], after=["ימים"])
    assert all(0.0 <= word["start"] < word["end"] <= 2.0 for word in out)


def test_context_changes_nothing_for_words_that_need_none(normalize) -> None:
    words = _words("שלום עולם, מה שלומך")
    plain = manifest.normalize_words(words, normalize)
    with_context = manifest.normalize_words(words, normalize, before=["פי", "3"], after=["14"])
    assert plain == with_context
    assert _spoken(plain) == ["שלום", "עולם,", "מה", "שלומך"]


def test_expansion_divides_the_original_span(normalize) -> None:
    out = manifest.normalize_words([{"word": "1995", "start": 2.0, "end": 3.0}], normalize)
    assert _spoken(out) == ["אלף", "תשע", "מאות", "תשעים", "וחמש"]
    assert [word["start"] for word in out] == pytest.approx([2.0, 2.2, 2.4, 2.6, 2.8])
    assert [word["end"] for word in out] == pytest.approx([2.2, 2.4, 2.6, 2.8, 3.0])

    # A gender-dependent expansion splits the same way once the next word is known.
    out = manifest.normalize_words(
        [{"word": "14", "start": 4.0, "end": 5.0}, {"word": "ימים", "start": 5.0, "end": 6.0}],
        normalize,
    )
    assert _spoken(out) == ["ארבעה", "עשר", "ימים"]
    assert [word["start"] for word in out] == pytest.approx([4.0, 4.5, 5.0])
    assert [word["end"] for word in out] == pytest.approx([4.5, 5.0, 6.0])


def test_words_without_special_handling_are_returned_untouched(normalize) -> None:
    words = [
        {"word": "שלום", "start": 0.0, "end": 0.4, "probability": 0.91},
        {"word": "עולם", "start": 0.4, "end": 1.0, "probability": 0.88},
    ]
    assert manifest.normalize_words(words, normalize) == words


def test_dropped_words_disappear(normalize) -> None:
    words = [
        {"word": "שלום", "start": 0.0, "end": 1.0},
        {"word": '"', "start": 1.0, "end": 1.1},
        {"word": "עולם", "start": 1.1, "end": 2.0},
    ]
    assert _spoken(manifest.normalize_words(words, normalize)) == ["שלום", "עולם"]


def test_lone_punctuation_word_joins_the_previous_word(normalize) -> None:
    words = [
        {"word": "הנוער", "start": 0.0, "end": 1.0},
        {"word": ".", "start": 1.0, "end": 1.2},
        {"word": "ואז", "start": 1.2, "end": 2.0},
    ]
    out = manifest.normalize_words(words, normalize)
    assert _spoken(out) == ["הנוער.", "ואז"]
    assert out[0]["start"] == 0.0
    assert out[0]["end"] == 1.2
    assert out[1] == {"word": "ואז", "start": 1.2, "end": 2.0}


def test_matches_whole_sentence_normalization(normalize) -> None:
    sentence = (
        "אנחנו סופרים למעלה מ-65 פצועים רק בשנת 2025 לעומת 21 פצועים . "
        "אמר פי 12 במאה 3 ו-14 ימים , 2 שנים ."
    )
    words = _words(sentence)
    expected = (
        "אנחנו סופרים למעלה משישים וחמישה פצועים רק בשנת אלפיים עשרים וחמש "
        "לעומת עשרים ואחד פצועים. אמר פי שנים עשר במאה שלישית וארבעה עשר ימים, "
        "שתי שנים."
    )
    normalized = manifest.normalize_words(words, normalize)
    assert " ".join(_spoken(normalized)) == expected
    # Each input word keeps its own place on the timeline, in order.
    starts = [word["start"] for word in normalized]
    assert starts == sorted(starts)
    assert starts[0] == 0.0
    assert normalized[-1]["end"] == pytest.approx(float(len(words)))


def _group(start: float, end: float, text: str) -> list[dict]:
    return [{"start": start, "end": end, "text": text, "words": _words(text)}]


def test_context_words_use_adjacent_groups_within_the_merge_gap() -> None:
    args = SimpleNamespace(merge_gap=1.5)
    groups = [_group(0.0, 6.0, "א ב ג ד ה"), _group(6.5, 12.0, "ו ז"), _group(20.0, 26.0, "ח ט")]
    assert manifest.context_words(groups, 0, args) == ([], ["ו", "ז"])
    assert manifest.context_words(groups, 1, args) == (["ג", "ד", "ה"], [])
    assert manifest.context_words(groups, 2, args) == ([], [])


def test_context_words_are_off_when_merging_is_off() -> None:
    args = SimpleNamespace(merge_gap=0.0)
    groups = [_group(0.0, 6.0, "א ב"), _group(6.1, 12.0, "ג ד")]
    assert manifest.context_words(groups, 0, args) == ([], [])
    assert manifest.context_words(groups, 1, args) == ([], [])


def _recording_with_split_number(tmp_path: Path, gap: float) -> Path:
    recording = tmp_path / "session"
    recording.mkdir()
    (recording / "audio.wav").touch()
    first = {
        "start": 0.0,
        "end": 6.0,
        "text": "אחרי 14",
        "words": [
            {"word": "אחרי", "start": 0.0, "end": 3.0, "probability": 0.9},
            {"word": "14", "start": 3.0, "end": 6.0, "probability": 0.9},
        ],
    }
    second = {
        "start": 6.0 + gap,
        "end": 12.0 + gap,
        "text": "ימים עברו",
        "words": [
            {"word": "ימים", "start": 6.0 + gap, "end": 9.0 + gap, "probability": 0.9},
            {"word": "עברו", "start": 9.0 + gap, "end": 12.0 + gap, "probability": 0.9},
        ],
    }
    (recording / "transcript.aligned.json").write_text(
        json.dumps({"segments": [first, second]}), encoding="utf-8"
    )
    return recording


def _segment_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        audio_glob="audio.wav",
        align_glob="transcript.aligned.json",
        metadata_glob="metadata.json",
        speaker_field="user_id",
        min_quality=0.6,
        min_duration=4.0,
        # Too short to hold both segments, so each becomes its own row.
        max_duration=10.0,
        merge_target=12.0,
        merge_jitter=0.0,
        merge_gap=1.5,
        transcode_audio=False,
        out_dir=tmp_path,
    )


def test_rows_use_the_neighbouring_row_as_context(normalize, tmp_path: Path) -> None:
    recording = _recording_with_split_number(tmp_path, gap=0.1)
    rows, _ = manifest.segment_rows(recording, _segment_args(tmp_path), normalize, random.Random(0))

    assert [row["transcript"] for row in rows] == ["אחרי ארבעה עשר", "ימים עברו"]
    assert [word["word"] for word in rows[0]["words"]] == ["אחרי", "ארבעה", "עשר"]
    assert [word["word"] for word in rows[1]["words"]] == ["ימים", "עברו"]


def test_rows_do_not_borrow_context_across_a_long_silence(normalize, tmp_path: Path) -> None:
    recording = _recording_with_split_number(tmp_path, gap=3.0)
    rows, _ = manifest.segment_rows(recording, _segment_args(tmp_path), normalize, random.Random(0))

    assert [row["transcript"] for row in rows] == ["אחרי ארבע עשרה", "ימים עברו"]
