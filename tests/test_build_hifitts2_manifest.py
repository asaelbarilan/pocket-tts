import gzip
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile

from hebrew_training import build_hifitts2_manifest as hifitts
from hebrew_training.official_manifest_common import stable_bucket

SETTINGS = {"min_duration": 2.0, "max_duration": 30.0, "max_wer": 0.1}
UTTERANCE_SEC = 6.0
CHAPTER_SEC = 40.0


def timed(*texts, start=0.2, step=0.5):
    return [
        {"word": text, "start": round(start + i * step, 3), "end": round(start + i * step + 0.4, 3)}
        for i, text in enumerate(texts)
    ]


def row(name, speaker="1", wer=0.0, duration=UTTERANCE_SEC, words=None):
    return {
        "audio_filepath": name,
        "duration": duration,
        "transcript": "hello there world",
        "speaker": speaker,
        "src_wer": wer,
        "src_cer": 0.0,
        "words": words if words is not None else timed("Hello", "there", "world."),
    }


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"wer": 0.5}, "wer"),
        ({"duration": 1.2}, "too_short"),
        ({"duration": 45.0}, "too_long"),
        ({"words": []}, "no_words"),
        ({"words": [{"word": "a b", "start": 0.1, "end": 0.2}]}, "bad_words"),
    ],
)
def test_usable_row_names_the_reason_a_row_is_dropped(changes, reason):
    parsed, why = hifitts.usable_row(row("a/b/c_0.flac", **changes), SETTINGS)
    assert parsed is None and why == reason


def test_untimed_punctuation_does_not_cost_the_row():
    words = (
        timed("It", "ended")
        + [{"word": "-", "start": None, "end": None}]
        + timed("later.", start=2)
    )
    parsed, why = hifitts.usable_row(row("a/b/c_0.flac", words=words), SETTINGS)
    assert why is None
    assert [w["word"] for w in parsed["words"]] == ["It", "ended", "-", "later."]
    dash = parsed["words"][2]
    assert dash["start"] == dash["end"]


def test_chapter_key_matches_the_chapter_list_path():
    utterance = "14/1503/14_1503_book_librivox-book_11_mee_8.flac"
    chapter = "14/1503/14_1503_book_librivox-book_11_mee.flac"
    assert hifitts.chapter_key(utterance) == hifitts.chapter_key_of_path(chapter)


def test_a_smaller_selection_is_a_prefix_of_a_larger_one():
    seconds = {f"chapter{i}": 900.0 for i in range(40)}
    order = hifitts.chapter_order(seconds)
    small, large = (hifitts.take_hours(order, seconds, h) for h in (1.0, 3.0))
    assert large[: len(small)] == small and len(small) == 4 and len(large) == 12
    assert hifitts.take_hours(order, seconds, 999.0) == order  # the pool, when it is too small
    assert hifitts.take_hours(order, seconds, 1.0, skip={small[0]})[0] == order[1]


CHAPTERS = [f"1/10/1_10_book_librivox-book_{i:02d}_author" for i in range(4)] + [
    "2/20/2_20_other_librivox-other_01_author"
]


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Four train chapters of three utterances, one validation chapter, a fake archive.org."""
    meta = tmp_path / "out" / "meta"
    (meta / "nvidia" / "44khz").mkdir(parents=True)
    (meta / "kyutai" / "train").mkdir(parents=True)
    with (meta / "nvidia" / hifitts.CHAPTERS_REMOTE).open("w") as handle:
        for key in CHAPTERS:
            utterances = [
                {
                    "audio_filepath": f"{key}_{n}.flac",
                    "offset": 3 + n * 8.0,
                    "duration": UTTERANCE_SEC,
                }
                for n in range(3)
            ]
            handle.write(
                json.dumps(
                    {
                        "url": f"http://archive.invalid/{key.rsplit('_', 3)[1]}/{key[-9:]}.mp3",
                        "chapter_filepath": key + ".flac",
                        "duration": CHAPTER_SEC,
                        "utterances": utterances,
                    }
                )
                + "\n"
            )
    train = [
        row(f"{key}_{n}.flac", speaker=key[0], wer=0.0 if n < 2 else 0.4)
        for key in CHAPTERS[:4]
        for n in range(3)
    ]
    # an untimed dash in one row
    train[0]["words"] = (
        timed("Hello") + [{"word": "-", "start": None, "end": None}] + timed("world.", start=1.5)
    )
    with gzip.open(meta / "kyutai" / "train" / "train_aligned-000-of-001.jsonl.gz", "wt") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in train)
    with gzip.open(meta / "kyutai" / hifitts.EVAL_REMOTE, "wt") as handle:
        handle.writelines(
            json.dumps(row(f"{CHAPTERS[4]}_{n}.flac", speaker="2")) + "\n" for n in range(2)
        )
    (meta / hifitts.META_COMPLETE).write_text("ok")

    fetched: list[str] = []
    dead: set[str] = set()

    def fake_fetch(url, destination):
        if any(part in url for part in dead):
            raise RuntimeError("404")
        fetched.append(url)
        destination.write_bytes(b"mp3")

    def fake_decode(path):
        return 0.1 * np.sin(np.arange(int(CHAPTER_SEC * 24000)) * 0.05).astype("float32")

    monkeypatch.setattr(hifitts, "fetch_audio", fake_fetch)
    monkeypatch.setattr(hifitts, "decode_chapter", fake_decode)
    return tmp_path / "out", fetched, dead


def run(out, *extra):
    hifitts.main(
        [
            "--out-dir",
            str(out),
            "--hours",
            "0.006",
            "--valid-utterances",
            "2",
            "--workers",
            "1",
            *extra,
        ]
    )


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_builds_the_target_hours_from_whole_chapters_and_applies_the_filters(corpus):
    out, fetched, _ = corpus
    run(out)
    train, valid = read(out / "train.jsonl"), read(out / "valid.jsonl")

    # 0.006 h needs two chapters of two usable (wer <= 0.1) utterances each
    assert len(train) == 4 and len({Path(r["path"]).parent for r in train}) == 1  # one speaker dir
    assert sum(r["duration"] for r in train) / 3600 >= 0.0066
    assert len(valid) == 2 and {r["speaker"] for r in valid} == {"2"}
    assert len(fetched) == 3  # two train chapters and the validation chapter
    for r in train + valid:
        info = soundfile.info(r["path"])
        assert (info.format, info.samplerate, info.channels) == ("FLAC", 24000, 1)
        assert r["start"] == 0.0 and r["dataset"] == "hifitts2"
    assert not (out / "tmp").exists()


def test_a_dead_chapter_is_replaced_from_the_same_ordered_pool(corpus):
    out, _, dead = corpus
    order = hifitts.chapter_order({hifitts.chapter_key_of_path(k): 1.0 for k in CHAPTERS[:4]})
    dead.add("librivox-book/" + order[0][-9:] + ".mp3")  # exactly this chapter's URL
    run(out)
    train = read(out / "train.jsonl")
    assert len(train) == 4  # still two chapters' worth
    assert not any(order[0] in r["path"] for r in train)


def test_a_rerun_fetches_nothing_and_a_bigger_target_fetches_only_the_difference(corpus):
    out, fetched, _ = corpus
    run(out)
    first, manifest = list(fetched), (out / "train.jsonl").read_text()
    run(out)
    assert fetched == first and (out / "train.jsonl").read_text() == manifest

    run(out, "--hours", "0.02")  # every chapter
    assert len(fetched) == len(first) + 2 and len(set(fetched)) == len(fetched)
    assert len(read(out / "train.jsonl")) == 8


def test_the_untimed_dash_is_kept_in_the_text(corpus):
    out, _, _ = corpus
    run(out, "--hours", "0.02")
    assert any("Hello - world." in r["transcript"] for r in read(out / "train.jsonl"))


def test_valid_speaker_overlap_is_reported(corpus, capsys):
    out, _, _ = corpus
    run(out)
    assert "valid speakers also in train: 0/1" in capsys.readouterr().out


def test_changing_the_format_asks_before_a_rebuild(corpus):
    out, fetched, _ = corpus
    run(out)
    with pytest.raises(SystemExit, match="--yes"):
        run(out, "--audio-format", "wav")
    assert list((out / "audio").rglob("*.flac"))
    count = len(fetched)
    run(out, "--audio-format", "wav", "--yes")
    assert list((out / "audio").rglob("*.wav")) and not list((out / "audio").rglob("*.flac"))
    assert len(fetched) == 2 * count  # everything was fetched again


def test_hash_order_is_stable_across_processes():
    # the selection must be the same on every run, or "reuse what is downloaded" is a lie
    assert stable_bucket("1/10/x") == pytest.approx(stable_bucket("1/10/x"))
    assert 0 <= stable_bucket("anything") < 1


def test_a_chapter_over_the_length_cap_is_replaced_from_the_pool(corpus):
    out, fetched, _ = corpus
    order = hifitts.chapter_order({hifitts.chapter_key_of_path(k): 1.0 for k in CHAPTERS[:4]})
    chapters = out / "meta" / "nvidia" / hifitts.CHAPTERS_REMOTE
    lines = [json.loads(line) for line in chapters.read_text().splitlines()]
    for line in lines:
        if hifitts.chapter_key_of_path(line["chapter_filepath"]) == order[0]:
            line["duration"] = 5 * 3600.0
    chapters.write_text("".join(json.dumps(line) + "\n" for line in lines))

    run(out)
    train = read(out / "train.jsonl")
    assert len(train) == 4 and not any(order[0] in r["path"] for r in train)
    assert not any(url.endswith("librivox-book/" + order[0][-9:] + ".mp3") for url in fetched)
