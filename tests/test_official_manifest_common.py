import json

import pytest

from hebrew_training import official_manifest_common as common


def word(text, start=None, end=None):
    return {"word": text, "start": start, "end": end}


def test_untimed_punctuation_becomes_a_zero_width_word_and_text_is_kept():
    words = [word("all", 1.0, 1.4), word("-"), word("not", 2.0, 2.3)]
    filled = common.fill_untimed_words(words, 5.0)
    assert [w["word"] for w in filled] == ["all", "-", "not"]
    assert filled[1]["start"] == filled[1]["end"] == 1.4
    assert words[1]["start"] is None  # the input is not modified


def test_untimed_word_with_letters_is_spread_over_the_gap_and_a_huge_gap_is_refused():
    filled = common.fill_untimed_words([word("a", 0.0, 1.0), word("foo"), word("b", 2.0, 4.0)], 5.0)
    assert (filled[1]["start"], filled[1]["end"]) == (1.0, 2.0)
    with pytest.raises(ValueError):
        common.fill_untimed_words([word("a", 0.0, 1.0), word("foo"), word("b", 9.0, 9.5)], 10.0)


def test_leading_and_trailing_untimed_words_use_the_row_edges():
    filled = common.fill_untimed_words([word('"'), word("hi", 0.5, 0.9), word(".")], 2.0)
    assert (filled[0]["start"], filled[0]["end"]) == (0.0, 0.0)
    assert (filled[2]["start"], filled[2]["end"]) == (0.9, 0.9)


@pytest.mark.parametrize(
    "words",
    [
        [],
        [word("two words", 0.0, 1.0)],
        [word("a", 1.0, 0.5)],
        [word("a", 2.0, 3.0), word("b", 1.0, 1.5)],
        [word("a", None, None)],
    ],
)
def test_normalize_words_rejects_bad_input(words):
    with pytest.raises(ValueError):
        common.normalize_words(words, 10.0)


def test_normalize_words_clamps_to_the_duration_and_rounds():
    out = common.normalize_words([word("a", 0.123456, 12.0)], 5.0)
    assert out == [{"word": "a", "start": 0.1235, "end": 5.0}]


def test_make_row_derives_the_transcript_from_the_words():
    row = common.make_row(
        "/x/a.flac", 4.0, [word("Hello,", 0.1, 0.5), word("world.", 0.6, 1.0)], 7, dataset="d"
    )
    assert row["transcript"] == "Hello, world."
    assert row["speaker"] == "7"
    assert row["start"] == 0.0 and row["dataset"] == "d"


def test_duration_stats_report_matches_the_hebrew_builder_format():
    stats = common.DurationStats()
    for duration in (1.0, 3.0, 5.0, 7.0):
        stats.add(duration)
    spread, short = stats.lines()
    assert spread == "row duration : median 4.00 s, p10 1.00, p90 7.00, max 7.00"
    assert short == "uncuttable (<2.0 s, prompt would overlap target): 1 (25.00%)"


def test_prepare_output_reuses_a_matching_build_and_guards_an_incompatible_one(tmp_path):
    signature = {"builder": "x", "audio_format": "flac"}
    common.prepare_output(tmp_path, signature, assume_yes=False)
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio" / "a.flac").write_text("keep")
    (tmp_path / "meta").mkdir()
    (tmp_path / "train.jsonl").write_text("{}\n")

    common.prepare_output(tmp_path, signature, assume_yes=False)  # same settings: untouched
    assert (tmp_path / "audio" / "a.flac").exists()

    changed = {**signature, "audio_format": "wav"}
    with pytest.raises(SystemExit, match="--yes"):  # no terminal, no consent
        common.prepare_output(tmp_path, changed, assume_yes=False)
    assert (tmp_path / "audio" / "a.flac").exists()
    with pytest.raises(SystemExit, match="aborted"):
        common.prepare_output(tmp_path, changed, assume_yes=False, ask=lambda _: "n")
    assert (tmp_path / "audio" / "a.flac").exists()

    common.prepare_output(tmp_path, changed, assume_yes=True)
    assert not (tmp_path / "audio").exists()
    assert not (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "meta").exists()  # downloaded metadata survives
    assert json.loads((tmp_path / common.STATE_FILE).read_text()) == changed


def test_prepare_output_treats_unrecorded_audio_as_unusable(tmp_path):
    (tmp_path / "audio").mkdir()
    with pytest.raises(SystemExit):
        common.prepare_output(tmp_path, {"builder": "x"}, assume_yes=False)


def test_write_manifests_sorts_rows_by_path(tmp_path):
    rows = [
        common.make_row(f"/a/{name}.flac", 3.0, [word("w", 0.0, 1.0)], "s") for name in ("b", "a")
    ]
    train, valid = common.write_manifests(tmp_path, rows, [])
    assert [json.loads(line)["path"] for line in train.read_text().splitlines()] == [
        "/a/a.flac",
        "/a/b.flac",
    ]
    assert valid.read_text() == ""


def test_parallel_gives_the_same_results_inline_and_in_processes():
    assert sorted(common.parallel(abs, [-3, -1, -2], 1)) == [1, 2, 3]
    assert sorted(common.parallel(abs, [-3, -1, -2], 2)) == [1, 2, 3]
