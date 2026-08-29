import json
import multiprocessing
import random
import sys
from pathlib import Path
from types import SimpleNamespace

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
            "--no-transcode-audio",
            "--valid-hours",
            "0.001",
        ],
    )
    build_official_manifest.main()


def test_parallel_output_matches_serial_and_removes_spool(
    monkeypatch, capsys, tmp_path: Path
) -> None:
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
    output = capsys.readouterr().out
    assert "progress : 4/4 recordings (100.0%), 0 waiting" in output
    assert "ETA 0s, 4 utterances" in output


def test_spool_limit_removes_partial_file(tmp_path: Path) -> None:
    context = multiprocessing.get_context()
    spool_bytes = context.Value("Q", 0, lock=False)
    spool_lock = context.Lock()
    rows = [{"speaker": "speaker", "duration": 5.0, "transcript": "שלום"}]

    with pytest.raises(RuntimeError, match="spool exceeded"):
        build_official_manifest._write_spool(0, rows, tmp_path, 1, spool_bytes, spool_lock)

    assert spool_bytes.value == 0
    assert not (tmp_path / "000000000.jsonl").exists()


def test_transcodes_to_flat_session_named_wav_and_reuses_it(monkeypatch, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _make_recording(corpus, "session-123", "speaker")
    output = tmp_path / "output"
    (output / "audio").mkdir(parents=True)
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        Path(command[-1]).write_bytes(b"RIFF" + b"\0" * 100)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(build_official_manifest.subprocess, "run", fake_run)
    args = SimpleNamespace(
        audio_glob="audio.wav,audio.m4a,audio.mka",
        align_glob="transcript.aligned.json",
        metadata_glob="metadata.json",
        speaker_field="user_id",
        min_quality=0.6,
        min_duration=4.0,
        max_duration=30.0,
        merge_target=12.0,
        merge_jitter=0.0,
        merge_gap=1.5,
        transcode_audio=True,
        out_dir=output,
    )

    rows, stats = build_official_manifest.segment_rows(
        corpus / "session-123", args, None, random.Random(0)
    )
    target = (output / "audio" / "session-123.wav").resolve()

    assert {row["path"] for row in rows} == {str(target)}
    assert stats["audio_transcoded"] == 1
    assert len(commands) == 1
    command, kwargs = commands[0]
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-ar") + 1] == "24000"
    assert command[command.index("-threads") + 1] == "1"
    assert kwargs == {"capture_output": True, "text": True, "check": False}
    assert not list((output / "audio").glob("*.partial"))

    rows, stats = build_official_manifest.segment_rows(
        corpus / "session-123", args, None, random.Random(0)
    )
    assert {row["path"] for row in rows} == {str(target)}
    assert stats["audio_reused"] == 1
    assert len(commands) == 1


def test_failed_transcode_removes_partial_file(monkeypatch, tmp_path: Path) -> None:
    recording = tmp_path / "session"
    recording.mkdir()
    source = recording / "audio.m4a"
    source.touch()
    output = tmp_path / "output"
    (output / "audio").mkdir(parents=True)

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stderr="decode failed")

    monkeypatch.setattr(build_official_manifest.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="decode failed"):
        build_official_manifest.transcode_audio(source, recording, output)

    assert not list((output / "audio").iterdir())
