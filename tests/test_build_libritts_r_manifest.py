import io
import json
import shutil
import tarfile
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import soundfile

from hebrew_training import build_libritts_r_manifest as libritts

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

RATE = 24000


def textgrid(intervals, xmax):
    """A TextGrid the way MFA writes it: a words tier, then a phones tier."""
    lines = ['File type = "ooTextFile"', "", "xmin = 0 ", f"xmax = {xmax} ", "size = 2 "]
    for number, name in enumerate(("words", "phones"), 1):
        lines += [
            f"    item [{number}]:",
            '        class = "IntervalTier" ',
            f'        name = "{name}" ',
        ]
        lines += [
            "        xmin = 0 ",
            f"        xmax = {xmax} ",
            f"        intervals: size = {len(intervals)} ",
        ]
        for index, (start, end, text) in enumerate(intervals, 1):
            lines += [f"        intervals [{index}]:", f"            xmin = {start} "]
            lines += [f"            xmax = {end} ", f'            text = "{text}" ']
    return "\n".join(lines) + "\n"


def mfa(*words, start=0.1, step=0.4):
    out, at = [(0, start, "")], start
    for text in words:
        out.append((round(at, 3), round(at + step, 3), text))
        at += step
    return out


@pytest.mark.parametrize(
    ("text", "intervals", "expected"),
    [
        ("Hello, World!", mfa("hello", "world"), ["Hello,", "World!"]),
        (
            "a self-contained man",
            mfa("a", "self", "contained", "man"),
            ["a", "self-contained", "man"],
        ),
        ("It's Mr. Smith's", mfa("it's", "mr", "smith's"), ["It's", "Mr.", "Smith's"]),
        ("'tis so", mfa("tis", "so"), ["'tis", "so"]),
        ("Wait - what", mfa("wait", "what"), ["Wait", "-", "what"]),
    ],
)
def test_map_words_keeps_the_corpus_tokens(text, intervals, expected):
    words = libritts.map_words(text.split(), intervals)
    assert [w["word"] for w in words] == expected
    assert all(a["end"] <= b["start"] + 1e-9 for a, b in pairwise(words))


def test_map_words_spans_every_interval_of_a_hyphenated_token():
    words = libritts.map_words(["self-contained"], mfa("self", "contained"))
    assert (words[0]["start"], words[0]["end"]) == (0.1, 0.9)


def test_map_words_puts_a_lone_dash_at_the_end_of_the_previous_word():
    words = libritts.map_words(["wait", "-", "what"], mfa("wait", "what"))
    assert words[1]["start"] == words[1]["end"] == words[0]["end"]


@pytest.mark.parametrize(
    "intervals",
    [mfa("hello"), mfa("hello", "there", "world"), mfa("hello", "<unk>"), mfa("hi", "world")],
)
def test_map_words_refuses_text_that_does_not_line_up(intervals):
    assert libritts.map_words(["Hello", "world"], intervals) is None


def test_parse_textgrid_reads_only_the_words_tier():
    xmax, intervals = libritts.parse_textgrid(textgrid(mfa("a", "b"), 1.5))
    assert xmax == 1.5
    assert [w for _, _, w in intervals] == ["", "a", "b"]


def make_shard(path: Path, utterances):
    """utterances: (id, speaker, chapter, text, seconds)"""
    records = {k: [] for k in ("id", "audio", "text_normalized", "speaker_id", "chapter_id")}
    for uid, speaker, chapter, text, seconds in utterances:
        buffer = io.BytesIO()
        tone = 0.1 * np.sin(np.arange(int(seconds * RATE)) * 0.05).astype("float32")
        soundfile.write(buffer, tone, RATE, format="WAV", subtype="PCM_16")
        records["id"].append(uid)
        records["audio"].append({"bytes": buffer.getvalue(), "path": uid + ".wav"})
        records["text_normalized"].append(text)
        records["speaker_id"].append(speaker)
        records["chapter_id"].append(chapter)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(records), path)


def make_tarball(path: Path, grids: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for uid, text in grids.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{uid}.TextGrid")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def utterance(uid, speaker, text, seconds):
    return (uid, speaker, "100", text, seconds)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Two train shards and one dev shard, a fake hub, and a counter of shard downloads."""
    hub = tmp_path / "hub"
    meta = tmp_path / "out" / "meta"
    layout = {
        "train.clean.100": {
            "data/train.clean.100/s0.parquet": [
                utterance("1_100_000001_000001", "1", "Hello, world.", 4.0),
                utterance("1_100_000001_000002", "1", "Too short.", 1.5),
                utterance("1_100_000001_000003", "1", "A self-contained man.", 4.0),
                utterance("1_100_000001_000004", "1", "No such words.", 4.0),
            ],
            "data/train.clean.100/s1.parquet": [
                utterance("2_100_000001_000001", "2", "Good day to you.", 4.5)
            ],
        },
        "dev.clean": {
            "data/dev.clean/d0.parquet": [
                utterance("9_100_000001_000001", "9", "Hello, world.", 4.0)
            ]
        },
    }
    grids = {
        "1_100_000001_000001": textgrid(mfa("hello", "world"), 4.0),
        "1_100_000001_000002": textgrid(mfa("too", "short"), 1.5),
        "1_100_000001_000003": textgrid(mfa("a", "self", "contained", "man"), 4.0),
        "1_100_000001_000004": textgrid(mfa("different", "words", "entirely"), 4.0),
        "2_100_000001_000001": textgrid(mfa("good", "day", "to", "you"), 4.5),
        "9_100_000001_000001": textgrid(mfa("hello", "world"), 4.0),
    }
    shards = {}
    for subset, files in layout.items():
        for repo_path, utterances in files.items():
            make_shard(hub / repo_path, utterances)
            shards.setdefault(subset, []).append([repo_path, (hub / repo_path).stat().st_size])
        ids = {u[0] for utterances in files.values() for u in utterances}
        make_tarball(meta / "align" / libritts.tarball_of(subset), {i: grids[i] for i in ids})
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "shards.json").write_text(json.dumps(shards))

    downloads = []

    def fake_download(repo, filename, repo_type, local_dir):
        downloads.append(filename)
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hub / filename, target)
        return str(target)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    return tmp_path / "out", downloads


def run(out, *extra):
    libritts.main(
        [
            "--out-dir", str(out),
            "--train-subsets", "train.clean.100",
            "--valid-subset", "dev.clean",
            "--hours", "0.003",
            "--valid-hours", "0.0005",
            "--workers", "1",
            *extra,
        ]
    )  # fmt: skip


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_builds_rows_from_the_corpus_text_with_mfa_times(corpus, capsys):
    out, _ = corpus
    run(out)
    train = read(out / "train.jsonl")
    valid = read(out / "valid.jsonl")

    # the 1.5 s utterance and the one whose words MFA heard differently are dropped
    assert sorted(r["transcript"] for r in train) == [
        "A self-contained man.",
        "Good day to you.",
        "Hello, world.",
    ]
    assert [r["transcript"] for r in valid] == ["Hello, world."]
    assert not {r["speaker"] for r in train} & {r["speaker"] for r in valid}
    for row in train + valid:
        assert row["start"] == 0.0 and row["duration"] >= 2.0
        assert Path(row["path"]).is_file() and row["path"].endswith(".flac")
        assert soundfile.info(row["path"]).samplerate == 24000
    hyphen = next(r for r in train if r["transcript"] == "A self-contained man.")
    assert [w["word"] for w in hyphen["words"]] == ["A", "self-contained", "man."]
    out_text = capsys.readouterr().out
    assert "'too_short': 1" in out_text and "'text_mismatch': 1" in out_text
    assert not (out / "tmp").exists()  # shards are deleted once written out


def test_a_rerun_and_a_bigger_target_reuse_the_shards_already_processed(corpus):
    out, downloads = corpus
    run(out, "--hours", "0.0001")  # one train shard is enough
    first = list(downloads)
    assert len([d for d in first if "train.clean.100" in d]) == 1

    before = (out / "train.jsonl").read_text()
    run(out, "--hours", "0.0001")
    assert downloads == first and (out / "train.jsonl").read_text() == before

    run(out)  # more hours: only the missing shard is fetched
    assert len(downloads) == len(first) + 1
    assert len({d for d in downloads}) == len(downloads)  # nothing was fetched twice
    assert len(read(out / "train.jsonl")) == 3


def test_changing_the_audio_format_asks_before_deleting(corpus):
    out, _ = corpus
    run(out)
    with pytest.raises(SystemExit, match="--yes"):
        run(out, "--audio-format", "wav")
    assert (out / "train.jsonl").is_file() and list((out / "audio").rglob("*.flac"))

    run(out, "--audio-format", "wav", "--yes")
    assert not list((out / "audio").rglob("*.flac"))
    assert list((out / "audio").rglob("*.wav"))
