import io
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from hebrew_training import align_official_manifest as aligner
from hebrew_training import build_crowd_transcribe_manifest as crowd
from hebrew_training.official_manifest_common import MIN_CUT_SEC

SETTINGS = {"min_duration": 3.0, "max_duration": 30.0, "duplicates": "agree"}


def record(uuid="Show/2021.01.01 episode/7", text="שלום עולם יפה", duration=4.0, **changes):
    return {
        "shard": "data/train-0.parquet",
        "row": 0,
        "split": "train",
        "uuid": uuid,
        "text": text,
        "retranscribed": False,
        "duration": duration,
        "flagged": False,
        **changes,
    }


def reasons(*records, settings=SETTINGS):
    kept, dropped = crowd.choose_clips(list(records), settings)
    return kept, dict(dropped)


def test_a_clean_clip_is_kept_with_its_uuid_parts():
    (clip,), dropped = reasons(record())
    assert dropped == {}
    assert (clip["show"], clip["episode"], clip["index"]) == (
        "Show",
        "Show/2021.01.01 episode",
        "7",
    )


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"flagged": True}, "flagged"),
        ({"duration": 2.4}, "too_short"),
        ({"duration": 31.0}, "too_long"),
        ({"text": "hello there"}, "no_hebrew"),
        ({"text": "שלום"}, "few_words"),
        ({"uuid": "no-index-here"}, "bad_uuid"),
    ],
)
def test_choose_clips_names_the_reason_a_clip_is_dropped(changes, reason):
    kept, dropped = reasons(record(**changes))
    assert kept == [] and dropped == {reason: 1}


def test_duplicates_are_one_clip_and_disagreeing_transcribers_drop_it():
    same = [record(row=0), record(row=1, text="שלום, עולם יפה!")]
    (clip,), dropped = reasons(*same)
    assert dropped == {} and clip["row"] == 0
    differing = [record(row=0), record(row=1, text="שלום עולם נחמד")]
    assert reasons(*differing) == ([], {"transcribers_disagree": 1})
    kept, _ = reasons(*differing, settings={**SETTINGS, "duplicates": "first"})
    assert len(kept) == 1


def test_a_retranscription_is_preferred_when_taking_the_first():
    group = [record(row=0), record(row=1, retranscribed=True, text="שלום עולם נחמד")]
    (clip,), _ = reasons(*group, settings={**SETTINGS, "duplicates": "first"})
    assert clip["row"] == 1


def test_bidi_marks_are_removed_from_the_text():
    (clip,), _ = reasons(record(text="\u202bשלום \u202bעולם יפה"))
    assert clip["text"] == "שלום עולם יפה"


def test_bare_punctuation_joins_the_word_before_it():
    assert crowd.glue_bare_punctuation(["שלום", "-", "עולם", "..."]) == ["שלום-", "עולם..."]
    assert crowd.glue_bare_punctuation(["-", "שלום"]) == ["שלום"]


@pytest.mark.parametrize(
    ("text", "reason"),
    [("שלום 2026 עולם", "digits"), ("שלום Microsoft", "latin"), ("שלום עולם ٣", "digits")],
)
def test_a_word_viter_cannot_spell_drops_the_row_with_the_reason(text, reason):
    with pytest.raises(ValueError, match=reason):
        crowd.prepare_text(text, None)


def test_text_that_viter_can_spell_keeps_its_punctuation():
    tokens, alt_words = crowd.prepare_text('אמר: "שלום, עולם!" ג\'ינס', None)
    assert tokens == ["אמר:", '"שלום,', 'עולם!"', "ג'ינס"] and alt_words is None


@pytest.mark.parametrize(
    "word",
    [
        "שלום",
        "שלום,",
        '"שלום"',
        "ג'ינס",
        "וילה'",
        'גרמ"ן',
        "-",
        "...",
        "2026",
        "abc",
        "שלוֹם",
        "'",
        "",
    ],
)
def test_viter_spellable_agrees_with_the_aligner(word):
    expected = aligner.split_graphemes(aligner.clean_viter_word(word)) is not None
    assert crowd.viter_spellable(word) is expected


def test_estimated_times_cover_the_clip_in_order():
    words = crowd.estimated_words(["א", "בבבב", "גג"], 6.0)
    assert words[0]["start"] == 0.0 and words[-1]["end"] == pytest.approx(6.0)
    assert all(a["end"] <= b["start"] + 1e-9 for a, b in itertools.pairwise(words))
    assert words[1]["end"] - words[1]["start"] > words[0]["end"] - words[0]["start"]


def test_min_cut_matches_the_loader_and_the_aligner():
    assert aligner.MIN_CUT_SEC == MIN_CUT_SEC


def test_train_selection_takes_whole_episodes_and_a_smaller_one_is_a_prefix():
    clips = [
        {"episode": f"s/e{n}", "duration": 600.0, "uuid": f"s/e{n}/{i}"}
        for n in range(30)
        for i in range(2)
    ]
    small, large = (crowd.take_train(clips, hours) for hours in (1.0, 3.0))
    assert {c["episode"] for c in small} < {c["episode"] for c in large}
    assert len(small) % 2 == 0 and crowd.take_train(clips, None) == clips


def make_shard(path: Path, rows: list[tuple[str, str, float, dict]], seed: int):
    rng = np.random.default_rng(seed)
    blobs = []
    for _, _, seconds, _ in rows:
        buffer = io.BytesIO()
        samples = (rng.standard_normal(int(seconds * 16000)) * 0.1).astype("float32")
        soundfile.write(buffer, np.stack([samples, samples], axis=1), 16000, format="FLAC")
        blobs.append({"bytes": buffer.getvalue(), "path": "x.mp3"})
    extra = [
        {
            "duration": seconds,
            "foreign_language": False,
            "multiple_speakers": False,
            "noisy": False,
            "skipped": False,
            "unintelligible": False,
            "too_long": False,
            **flags,
        }
        for _, _, seconds, flags in rows
    ]
    pq.write_table(
        pa.table(
            {
                "uuid": [r[0] for r in rows],
                "audio": blobs,
                "sentence": [r[1] for r in rows],
                "is_retranscribe": [False] * len(rows),
                "extra_data": extra,
            }
        ),
        path,
        row_group_size=2,
    )


def test_build_end_to_end_from_local_shards(tmp_path):
    source = tmp_path / "source" / "data"
    source.mkdir(parents=True)
    train = [
        ("A/2021 one/1", "שלום עולם יפה מאוד", 4.0, {}),
        ("A/2021 one/2", "מה שלומך היום", 4.5, {}),
        ("A/2021 one/3", "קצר מדי", 2.2, {}),
        ("A/2021 one/4", "עם מספר 2026 כאן", 4.0, {}),
        ("B/2022 two/1", "זה משפט שלישי טוב", 5.0, {}),
        ("B/2022 two/2", "משפט רועש מאוד", 4.0, {"noisy": True}),
    ]
    test = [
        ("C/2023 three/1", "אנחנו בודקים את הסיפור", 4.0, {}),
        ("C/2023 three/2", "ועוד משפט לבדיקה", 3.5, {}),
    ]
    make_shard(source / "train-00000-of-00001.parquet", train, 0)
    make_shard(source / "test-00000-of-00001.parquet", test, 1)
    out = tmp_path / "out"
    crowd.main(["--out-dir", str(out), "--source-dir", str(source), "--workers", "1", "--yes"])

    train_rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    valid_rows = [json.loads(line) for line in (out / "valid.jsonl").read_text().splitlines()]
    assert sorted(r["clip"] for r in train_rows) == ["A/2021 one/1", "A/2021 one/2", "B/2022 two/1"]
    assert len(valid_rows) == 2
    assert {r["show"] for r in train_rows}.isdisjoint({r["show"] for r in valid_rows})
    for row in train_rows + valid_rows:
        audio = Path(row["path"])
        assert audio.is_file() and audio.suffix == ".flac"
        info = soundfile.info(audio)
        assert (info.samplerate, info.channels) == (24000, 1)
        assert row["duration"] == pytest.approx(info.frames / info.samplerate, abs=1e-3)
        assert row["timing"] == "estimated" and row["start"] == 0
        assert row["words"][-1]["end"] == pytest.approx(row["duration"], abs=1e-3)
        assert row["transcript"] == " ".join(w["word"] for w in row["words"])
        assert row["speaker"].startswith(row["show"] + "/")
    assert [r["path"] for r in train_rows] == sorted(r["path"] for r in train_rows)

    before = {p: p.stat().st_mtime_ns for p in (out / "audio").rglob("*.flac")}
    crowd.main(["--out-dir", str(out), "--source-dir", str(source), "--workers", "1", "--yes"])
    assert before == {p: p.stat().st_mtime_ns for p in (out / "audio").rglob("*.flac")}
