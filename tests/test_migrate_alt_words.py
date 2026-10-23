"""migrate_alt_words: new `alt_words` into manifests that were already built and aligned.

Runs the real builder and normalizer on a small synthetic corpus (skipped without the sibling
hebrew-tts-data-tools checkout), turns the built rows into what the later stages leave behind
(aligner: moved `start`, `original_start`, new word times; diacritizer: `nikud`) with the
legacy `digits` alternates, and checks that the migration changes the `alt_words` and nothing
else.
"""

import json
import sys
from pathlib import Path

import pytest

from hebrew_training import build_official_manifest as builder
from hebrew_training import migrate_alt_words as migrate

SEGMENTS = {
    "plenum-a": [
        ('אחרי 14 ימים אמר: "שלום" ל-ה-API ממש...', 3.0),
        ('זה ד"ר כהן 😀 You’ve well-known', 3.0),
    ],
    "plenum-b": [("בשעה 14:30 שלום ,כן", 3.0), ("וכו' [הערה] אחרי", 3.0)],
}


@pytest.fixture(scope="module", autouse=True)
def _needs_normalizer():
    if not (builder.DEFAULT_NORMALIZER / "hebrew_tts_normalizer.py").is_file():
        pytest.skip("hebrew-tts-data-tools checkout not found beside this repo")


def _write_corpus(corpus: Path) -> None:
    corpus.mkdir()
    for name, segments in SEGMENTS.items():
        recording = corpus / name
        recording.mkdir()
        (recording / "audio.wav").touch()
        (recording / "metadata.json").write_text(json.dumps({"user_id": name}), encoding="utf-8")
        built, clock = [], 0.0
        for text, length in segments:
            tokens = text.split()
            step = length / len(tokens)
            built.append(
                {
                    "start": clock,
                    "end": clock + length,
                    "text": text,
                    "words": [
                        {
                            "word": token,
                            "start": clock + step * i,
                            "end": clock + step * (i + 1),
                            "probability": 0.9,
                        }
                        for i, token in enumerate(tokens)
                    ],
                }
            )
            clock += length + 0.1
        (recording / "transcript.aligned.json").write_text(
            json.dumps({"segments": built}, ensure_ascii=False), encoding="utf-8"
        )


def _flags(corpus: Path, out_dir: Path) -> list[str]:
    return [
        "--corpus",
        str(corpus),
        "--out-dir",
        str(out_dir),
        "--normalize-text",
        "--no-transcode-audio",
        "--min-duration",
        "4",
        "--merge-jitter",
        "0",
    ]


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def built(monkeypatch, tmp_path: Path):
    """(corpus, built folder, aligned-and-diacritized manifests with legacy `digits`)."""
    corpus = tmp_path / "corpus"
    _write_corpus(corpus)
    built_dir = tmp_path / "built"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build",
            *_flags(corpus, built_dir),
            "--spool-dir",
            str(tmp_path),
            "--valid-hours",
            "0.001",
        ],
    )
    builder.main()

    later = tmp_path / "later"
    later.mkdir()
    for split in ("train", "valid"):
        rows = _read(built_dir / f"{split}.jsonl")
        assert rows, split
        old = []
        for row in rows:
            row = dict(row)
            row["nikud"] = [word["word"] for word in row["words"]]
            row["original_start"], row["original_duration"] = row["start"], row["duration"]
            row["original_words"] = row["words"]
            row["start"] = round(row["start"] + 0.25, 4)
            row["words"] = [
                {**word, "start": word["start"] + 0.01, "end": word["end"] + 0.01}
                for word in row["words"]
            ]
            row["alignment_backend"] = "viter"
            row["alt_words"] = {"digits": [{"from": 0, "to": 0, "replace": ["stale"]}]}
            old.append(row)
        (later / f"{split}_aligned.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in old), encoding="utf-8"
        )
    return corpus, built_dir, later


def _migrate(corpus: Path, later: Path, out_dir: Path, *extra: str) -> None:
    migrate.main(
        [
            *_flags(corpus, out_dir),
            "--manifest",
            str(later / "train_aligned.jsonl"),
            str(later / "valid_aligned.jsonl"),
            *extra,
        ]
    )


def test_alt_words_are_replaced_and_nothing_else_changes(built, tmp_path: Path, capsys) -> None:
    corpus, built_dir, later = built
    out = tmp_path / "migrated"
    _migrate(corpus, later, out)

    spans = 0
    for split in ("train", "valid"):
        fresh = _read(built_dir / f"{split}.jsonl")
        before = _read(later / f"{split}_aligned.jsonl")
        after = _read(out / f"{split}_aligned.jsonl")
        assert len(fresh) == len(before) == len(after)
        for new_row, old_row, row in zip(fresh, before, after):
            assert row["alt_words"] == new_row["alt_words"]
            assert "digits" not in row["alt_words"]
            spans += len(row["alt_words"]["norms"])
            assert {k: v for k, v in row.items() if k != "alt_words"} == {
                k: v for k, v in old_row.items() if k != "alt_words"
            }
            assert list(row) == list(old_row)  # `alt_words` kept its place among the keys
    assert spans > 6
    summary = capsys.readouterr().out
    assert "with alternates" in summary
    assert "the replacements table" in summary or "number:" in summary


def test_the_migrated_alternates_swap_back_to_what_the_corpus_wrote(built, tmp_path: Path) -> None:
    corpus, _, later = built
    out = tmp_path / "migrated"
    _migrate(corpus, later, out)
    written = {
        name: " ".join(text for text, _ in segments).split() for name, segments in SEGMENTS.items()
    }
    for row in _read(out / "train_aligned.jsonl") + _read(out / "valid_aligned.jsonl"):
        words = [word["word"] for word in row["words"]]
        swapped = builder.swap_alternates(words, row["alt_words"]["norms"])
        recording = Path(row["path"]).parent.name
        # The row is a run of the recording's words; every one of them is the corpus's own.
        assert " ".join(swapped) in " ".join(written[recording])


def test_workers_do_not_change_the_output(built, tmp_path: Path) -> None:
    corpus, _, later = built
    _migrate(corpus, later, tmp_path / "one")
    _migrate(corpus, later, tmp_path / "two", "--workers", "2")
    for split in ("train", "valid"):
        name = f"{split}_aligned.jsonl"
        assert (tmp_path / "one" / name).read_bytes() == (tmp_path / "two" / name).read_bytes()


def test_a_manifest_not_built_from_this_corpus_with_these_options_stops(
    built, tmp_path: Path
) -> None:
    corpus, _, later = built
    # Another merge length makes other rows: the starts no longer line up.
    with pytest.raises(SystemExit, match="not built from this corpus"):
        migrate.main(
            [
                *_flags(corpus, tmp_path / "out"),
                "--min-duration",
                "2.5",
                "--merge-gap",
                "0",
                "--manifest",
                str(later / "train_aligned.jsonl"),
            ]
        )
    assert not list((tmp_path / "out").iterdir())  # no partial file left behind


def test_skip_keeps_the_row_without_stale_alternates(built, tmp_path: Path, capsys) -> None:
    corpus, _, later = built
    out = tmp_path / "out"
    migrate.main(
        [
            *_flags(corpus, out),
            "--min-duration",
            "2.5",
            "--merge-gap",
            "0",
            "--on-mismatch",
            "skip",
            "--manifest",
            str(later / "train_aligned.jsonl"),
        ]
    )
    kept = _read(out / "train_aligned.jsonl")
    original = _read(later / "train_aligned.jsonl")
    assert len(kept) == len(original)
    assert all("digits" not in row.get("alt_words", {}) for row in kept)
    assert "left without alt_words" in capsys.readouterr().out


def test_overwriting_the_input_needs_in_place(built, tmp_path: Path) -> None:
    corpus, _, later = built
    with pytest.raises(SystemExit):
        _migrate(corpus, later, later)
    before = (later / "train_aligned.jsonl").read_bytes()
    assert b"stale" in before  # untouched by the refused run

    _migrate(corpus, later, later, "--in-place")
    after = (later / "train_aligned.jsonl").read_bytes()
    assert b"stale" not in after
    assert not list(later.glob(".*.migrating"))


def test_it_refuses_to_run_without_the_normalizer_flag(built, tmp_path: Path) -> None:
    corpus, _, later = built
    with pytest.raises(SystemExit):
        migrate.main(
            [
                "--corpus",
                str(corpus),
                "--out-dir",
                str(tmp_path / "o"),
                "--manifest",
                str(later / "train_aligned.jsonl"),
            ]
        )
