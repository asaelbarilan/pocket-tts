import json
import multiprocessing
import sys
from pathlib import Path

import pytest

from hebrew_training import build_official_manifest


def _make_recording(corpus: Path, name: str, speaker: str) -> None:
    recording = corpus / name
    recording.mkdir()
    (recording / "audio.wav").touch()
    (recording / "metadata.json").write_text(json.dumps({"user_id": speaker}), encoding="utf-8")
    segments = [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "שלום",
            "words": [{"word": "שלום", "start": 0.0, "end": 3.0, "probability": 0.9}],
        },
        {
            "start": 3.1,
            "end": 6.1,
            "text": "עולם",
            "words": [{"word": "עולם", "start": 3.1, "end": 6.1, "probability": 0.9}],
        },
    ]
    (recording / "transcript.aligned.json").write_text(
        json.dumps({"segments": segments}), encoding="utf-8"
    )


def _run(monkeypatch, corpus: Path, output: Path, spool_parent: Path, workers: int) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_official_manifest",
            "--corpus",
            str(corpus),
            "--out-dir",
            str(output),
            "--spool-dir",
            str(spool_parent),
            "--workers",
            str(workers),
            "--valid-hours",
            "0.001",
        ],
    )
    build_official_manifest.main()


def test_parallel_output_matches_serial_and_removes_spool(monkeypatch, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index in range(4):
        _make_recording(corpus, f"recording-{index}", f"speaker-{index}")

    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _run(monkeypatch, corpus, serial, tmp_path, workers=1)
    _run(monkeypatch, corpus, parallel, tmp_path, workers=2)

    for name in ("train_aligned.jsonl", "valid_aligned.jsonl"):
        assert (serial / name).read_bytes() == (parallel / name).read_bytes()
    assert (
        sum(
            len((serial / name).read_text().splitlines())
            for name in ("train_aligned.jsonl", "valid_aligned.jsonl")
        )
        == 4
    )
    assert not list(tmp_path.glob("pocket-tts-manifest-*"))


def test_spool_limit_removes_partial_file(tmp_path: Path) -> None:
    context = multiprocessing.get_context()
    spool_bytes = context.Value("Q", 0, lock=False)
    spool_lock = context.Lock()
    rows = [{"speaker": "speaker", "duration": 5.0, "transcript": "שלום"}]

    with pytest.raises(RuntimeError, match="spool exceeded"):
        build_official_manifest._write_spool(0, rows, tmp_path, 1, spool_bytes, spool_lock)

    assert spool_bytes.value == 0
    assert not (tmp_path / "000000000.jsonl").exists()
