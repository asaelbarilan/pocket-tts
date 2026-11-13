import json
import multiprocessing
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hebrew_training import build_official_manifest
from hebrew_training.audio_v2_tagger import transcode_selected
from hebrew_training.audio_v2_tagger.build_selected import remove_sources_after_wavs


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

    for name in ("train.jsonl", "valid.jsonl"):
        assert (serial / name).read_bytes() == (parallel / name).read_bytes()
    assert (
        sum(
            len((serial / name).read_text().splitlines()) for name in ("train.jsonl", "valid.jsonl")
        )
        == 4
    )
    assert not list(tmp_path.glob("pocket-tts-manifest-*"))
    output = capsys.readouterr().out
    assert "progress : 4/4 recordings (100.0%), 0 waiting" in output
    assert "ETA 0s, 4 utterances" in output


def test_row_selection_filters_builder_output_before_manifest_split(monkeypatch, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index in range(4):
        _make_recording(corpus, f"recording-{index}", f"speaker-{index}")
    selected = tmp_path / "rows.npz"
    np.savez_compressed(
        selected,
        episode_ids=np.array([f"recording-{index}" for index in range(4)]),
        episode=np.array([0, 2]),
        start=np.array([0.0, 0.0]),
        end=np.array([6.1, 6.1]),
    )
    output = tmp_path / "selected"
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
            str(tmp_path),
            "--row-selection",
            str(selected),
            "--no-transcode-audio",
            "--workers",
            "2",
            "--valid-hours",
            "0.001",
        ],
    )
    build_official_manifest.main()
    rows = [
        json.loads(line)
        for name in ("train.jsonl", "valid.jsonl")
        for line in (output / name).read_text().splitlines()
    ]
    assert len(rows) == 2
    assert {Path(row["path"]).parent.name for row in rows} == {"recording-0", "recording-2"}


def test_builder_reuses_saved_wav_after_source_audio_removed(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _make_recording(corpus, "episode", "speaker")
    (corpus / "episode" / "audio.wav").unlink()
    wavs = tmp_path / "saved-wavs"
    wavs.mkdir()
    saved = wavs / "episode.wav"
    saved.write_bytes(b"RIFF" + b"\0" * 100)
    args = _transcode_args(tmp_path / "new-manifest", wavs)
    rows, stats = build_official_manifest.segment_rows(
        corpus / "episode", args, None, random.Random(0)
    )
    assert len(rows) == 1 and rows[0]["path"] == str(saved.resolve())
    assert stats["audio_reused"] == 1


def test_source_audio_removed_only_after_all_wavs_complete(tmp_path):
    corpus = tmp_path / "corpus"
    wavs = tmp_path / "wavs"
    wavs.mkdir()
    episodes = [{"id": "a", "ext": ".mp3"}, {"id": "b", "ext": ".m4a"}]
    for episode in episodes:
        folder = corpus / episode["id"]
        folder.mkdir(parents=True)
        (folder / f"audio{episode['ext']}").write_bytes(b"source")
        (folder / "transcript.aligned.json").write_text("{}")
    (wavs / "a.wav").write_bytes(b"RIFF" + b"\0" * 100)
    with pytest.raises(RuntimeError, match="retaining sources"):
        remove_sources_after_wavs(episodes, corpus, wavs)
    assert (corpus / "a/audio.mp3").exists()
    (wavs / "b.wav").write_bytes(b"RIFF" + b"\0" * 100)
    assert remove_sources_after_wavs(episodes, corpus, wavs) == 2
    assert not (corpus / "a/audio.mp3").exists()
    assert not (corpus / "b/audio.m4a").exists()
    assert (corpus / "a/transcript.aligned.json").exists()


def test_transcoder_waits_for_complete_download_and_reuses_existing_wavs(monkeypatch, tmp_path):
    corpus, wavs = tmp_path / "corpus", tmp_path / "wavs"
    corpus.mkdir()
    wavs.mkdir()
    episodes = [
        {"id": "first", "ext": ".mp3", "audio_bytes": 6},
        {"id": "second", "ext": ".m4a", "audio_bytes": 6},
    ]
    for episode in episodes:
        folder = corpus / episode["id"]
        folder.mkdir()
        (folder / f"audio{episode['ext']}").write_bytes(b"source")
        (folder / "transcript.aligned.json").write_text("{}")
    (corpus / "first/metadata.json").write_text("{}")
    converted = []

    def fake_transcode(source, folder, output):
        converted.append(folder.name)
        (output / f"{folder.name}.wav").write_bytes(b"RIFF" + b"\0" * 100)
        (corpus / "second/metadata.json").write_text("{}")
        return output / f"{folder.name}.wav", True

    monkeypatch.setattr(transcode_selected, "transcode_audio", fake_transcode)
    assert transcode_selected.transcode_ready(episodes, corpus, wavs, workers=2, interval=0.01) == 2
    assert converted == ["first", "second"]
    assert (corpus / "first/audio.mp3").exists()
    assert (corpus / "second/audio.m4a").exists()
    assert transcode_selected.transcode_ready(episodes, corpus, wavs, workers=2) == 0


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
        build_official_manifest.transcode_audio(source, recording, output / "audio")

    assert not list((output / "audio").iterdir())


def _transcode_args(out_dir: Path, audio_dir: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
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
        out_dir=out_dir,
        audio_dir=audio_dir,
    )


def test_existing_wavs_are_kept_and_reused_by_a_rebuild_into_a_new_folder(
    monkeypatch, tmp_path: Path
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _make_recording(corpus, "session-1", "speaker")
    first = tmp_path / "build-1"
    (first / "audio").mkdir(parents=True)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"RIFF" + b"\0" * 100)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(build_official_manifest.subprocess, "run", fake_run)
    rows, _ = build_official_manifest.segment_rows(
        corpus / "session-1", _transcode_args(first), None, random.Random(0)
    )
    wav = first / "audio" / "session-1.wav"
    before = (wav.read_bytes(), wav.stat().st_mtime_ns)
    assert len(commands) == 1

    # a new manifest version in a new folder, pointed at the old WAVs
    second = tmp_path / "build-2"
    second.mkdir()
    rows2, stats = build_official_manifest.segment_rows(
        corpus / "session-1", _transcode_args(second, first / "audio"), None, random.Random(0)
    )
    assert len(commands) == 1 and stats["audio_reused"] == 1 and stats["audio_transcoded"] == 0
    assert {row["path"] for row in rows2} == {str(wav.resolve())} == {row["path"] for row in rows}
    assert (wav.read_bytes(), wav.stat().st_mtime_ns) == before
    assert not (second / "audio").exists()  # nothing was made in the new folder


def test_only_a_partial_file_is_replaced_never_a_finished_wav(monkeypatch, tmp_path: Path) -> None:
    recording = tmp_path / "session"
    recording.mkdir()
    source = recording / "audio.m4a"
    source.touch()
    audio = tmp_path / "audio"
    audio.mkdir()
    (audio / ".session.wav.partial").write_bytes(b"half written")

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"RIFF" + b"\1" * 100)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(build_official_manifest.subprocess, "run", fake_run)
    target, created = build_official_manifest.transcode_audio(source, recording, audio)
    assert created and target.read_bytes().startswith(b"RIFF")
    assert not list(audio.glob("*.partial"))

    target.write_bytes(b"RIFF" + b"\2" * 100)  # a finished wav stays, even if the source changed
    source.write_bytes(b"new source")
    again, created = build_official_manifest.transcode_audio(source, recording, audio)
    assert not created and again.read_bytes() == b"RIFF" + b"\2" * 100
