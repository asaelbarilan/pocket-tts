from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import pytest
from starlette.testclient import TestClient

import hebrew_training.dataset_viewer as dv


def _create_synthetic_wav(path: Path, duration_sec: float = 10.0, sample_rate: int = 24000) -> None:
    """Create a sine-wave PCM 16-bit mono WAV file for testing."""
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio_data.tobytes())


@pytest.fixture
def temp_dataset(tmp_path: Path):
    """Fixture providing a synthetic dataset with audio and manifest."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    wav1 = audio_dir / "clip1.wav"
    wav2 = audio_dir / "clip2.wav"
    _create_synthetic_wav(wav1, duration_sec=12.0)
    _create_synthetic_wav(wav2, duration_sec=8.0)

    rows = [
        {
            "path": str(wav1),
            "start": 2.0,
            "duration": 5.0,
            "transcript": "שלום עולם בדיקה ראשונה",
            "speaker": "speaker_A",
            "words": [
                {"word": "שלום", "start": 0.0, "end": 1.0},
                {"word": "עולם", "start": 1.1, "end": 2.2},
                {"word": "בדיקה", "start": 2.4, "end": 3.5},
                {"word": "ראשונה", "start": 3.6, "end": 5.0},
            ],
        },
        {
            "path": str(wav1),
            "start": 7.5,
            "duration": 3.0,
            "transcript": "משפט קצר נוסף",
            "speaker": "speaker_A",
            "words": [
                {"word": "משפט", "start": 0.0, "end": 0.8},
                {"word": "קצר", "start": 0.9, "end": 1.7},
                {"word": "נוסף", "start": 1.8, "end": 3.0},
            ],
        },
        {
            "path": str(wav2),
            "start": 1.0,
            "duration": 4.0,
            "transcript": "קובץ שני שמע שונה",
            "speaker": "speaker_B",
            "words": [
                {"word": "קובץ", "start": 0.0, "end": 0.9},
                {"word": "שני", "start": 1.0, "end": 1.8},
                {"word": "שמע", "start": 2.0, "end": 2.8},
                {"word": "שונה", "start": 3.0, "end": 4.0},
            ],
        },
    ]

    manifest_file = tmp_path / "manifest.jsonl"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    return {"root": tmp_path, "manifest": manifest_file, "audio_dir": audio_dir, "rows": rows}


def test_line_indexing_and_caching(temp_dataset):
    manifest_path = temp_dataset["manifest"]

    # 1. First build (creates cache)
    idx1 = dv.ManifestIndex(manifest_path)
    assert len(idx1) == 3
    row0 = idx1.get_row(0)
    assert row0["speaker"] == "speaker_A"
    assert row0["duration"] == 5.0

    row2 = idx1.get_row(2)
    assert row2["speaker"] == "speaker_B"
    assert row2["duration"] == 4.0

    # 2. Second build (loads from cache)
    idx2 = dv.ManifestIndex(manifest_path)
    assert len(idx2) == 3
    assert idx2.offsets == idx1.offsets

    with pytest.raises(IndexError):
        idx1.get_row(99)


def test_sample_data_and_word_timings(temp_dataset):
    manifest_path = temp_dataset["manifest"]
    idx = dv.ManifestIndex(manifest_path)

    sample = idx.get_sample_data(0, padding=3.0)
    assert sample["index"] == 0
    assert sample["line_number"] == 1
    assert sample["total_samples"] == 3
    assert sample["speaker"] == "speaker_A"
    assert sample["start"] == 2.0
    assert sample["duration"] == 5.0
    assert sample["pre_padding"] == 2.0  # Clamped to start (start was 2.0s < 3.0s padding)
    assert sample["post_padding"] == 3.0
    assert sample["total_audio_duration"] == 10.0
    assert sample["sample_audio_start"] == 2.0
    assert sample["sample_audio_end"] == 7.0

    words = sample["words"]
    assert len(words) == 4
    # Word 0: "שלום" [0.0, 1.0] -> audio [2.0, 3.0]
    assert words[0]["word"] == "שלום"
    assert words[0]["start"] == 0.0
    assert words[0]["end"] == 1.0
    assert words[0]["duration"] == 1.0
    assert words[0]["audio_start"] == 2.0
    assert words[0]["audio_end"] == 3.0

    # Word 1: "עולם" [1.1, 2.2] -> audio [3.1, 4.2]
    assert words[1]["word"] == "עולם"
    assert words[1]["audio_start"] == 3.1
    assert words[1]["audio_end"] == 4.2


@pytest.mark.parametrize("backend", ["viter", "mms", "original", None])
@pytest.mark.parametrize("original_start", [1.8, 2.3])
def test_saved_alignment_uses_shared_audio_origin(temp_dataset, backend, original_start):
    row = dict(temp_dataset["rows"][0])
    row.update(
        original_start=original_start,
        original_duration=5.0,
        original_words=[dict(word) for word in row["words"]],
    )
    if backend is not None:
        row["alignment_backend"] = backend
    manifest = temp_dataset["root"] / "realigned.jsonl"
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    sample = dv.ManifestIndex(manifest).get_sample_data(0)
    original = sample["original_words"][0]
    saved = sample["alternative_words"][0]
    assert original["start"] == pytest.approx(original_start - row["start"])
    assert original["audio_start"] == pytest.approx(original_start)
    assert original["duration"] == saved["duration"]
    assert saved["audio_start"] - original["audio_start"] == pytest.approx(2.0 - original_start)
    assert sample["alignment_backend"] == (backend or "unknown")
    assert sample["word_comparisons"] == [
        {"original_index": i, "alternative_index": i} for i in range(4)
    ]


def test_comparison_does_not_pair_different_words(temp_dataset):
    row = dict(temp_dataset["rows"][0])
    row["original_words"] = row["words"]
    row["original_start"] = row["start"]
    row["words"] = [row["words"][0], *row["words"][2:]]
    manifest = temp_dataset["root"] / "realigned.jsonl"
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    sample = dv.ManifestIndex(manifest).get_sample_data(0)
    assert sample["word_comparisons"] == [
        {"original_index": 0, "alternative_index": 0},
        {"original_index": 1, "alternative_index": None},
        {"original_index": 2, "alternative_index": 1},
        {"original_index": 3, "alternative_index": 2},
    ]


def test_extract_audio_slice(temp_dataset):
    sample_wav = temp_dataset["audio_dir"] / "clip1.wav"

    # Total duration is 12.0s
    # Slice [2.0, 7.0] with 3.0s padding -> audio [0.0, 10.0]
    wav_bytes, pre_pad, post_pad, total_dur = dv.extract_audio_slice(
        sample_wav, start=2.0, duration=5.0, padding=3.0
    )

    assert pre_pad == 2.0  # start 2.0 - 0.0
    assert post_pad == 3.0
    assert total_dur == pytest.approx(10.0, abs=0.01)

    # Validate WAV bytes
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        dur = w.getnframes() / w.getframerate()
        assert dur == pytest.approx(10.0, abs=0.01)


def test_extract_audio_slice_end_clamping(temp_dataset):
    sample_wav = temp_dataset["audio_dir"] / "clip2.wav"

    # clip2 is 8.0s. Request start=5.0, dur=2.5, pad=3.0.
    # req_end = 10.5 > 8.0, so end is clamped to 8.0s!
    wav_bytes, pre_pad, post_pad, total_dur = dv.extract_audio_slice(
        sample_wav, start=5.0, duration=2.5, padding=3.0
    )

    assert pre_pad == 3.0
    assert post_pad == pytest.approx(0.5, abs=0.01)  # 8.0 - (5.0 + 2.5) = 0.5s
    assert total_dur == pytest.approx(6.0, abs=0.01)  # [2.0s to 8.0s] = 6.0s

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        dur = w.getnframes() / w.getframerate()
        assert dur == pytest.approx(6.0, abs=0.01)


def test_fastapi_endpoints(temp_dataset):
    dv.state = dv.ServerState(default_manifest=temp_dataset["manifest"])
    client = TestClient(dv.app)

    # Health check
    res = client.get("/healthz")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

    # UI HTML page
    res = client.get("/")
    assert res.status_code == 200
    assert "Hebrew Dataset Sample Viewer" in res.text
    assert '<canvas id="waveform-canvas"' in res.text
    assert '<canvas id="overview-canvas"' in res.text
    assert 'id="zoom-slider"' in res.text
    assert 'id="fit-sample-btn"' in res.text
    assert 'id="zoom-in-btn"' in res.text
    assert 'id="zoom-out-btn"' in res.text
    assert 'id="alignment-select"' in res.text
    assert 'id="align-btn"' not in res.text
    for endpoint in ("align", "calibrate-last-word", "calibrate-last-word-emission"):
        assert client.post(f"/api/sample/0/{endpoint}").status_code == 404

    # Manifests list
    res = client.get("/api/manifests")
    assert res.status_code == 200
    data = res.json()
    assert data["total_samples"] == 3
    assert data["current_name"] == "manifest.jsonl"

    # Sample by index
    res = client.get("/api/sample/0")
    assert res.status_code == 200
    sample0 = res.json()
    assert sample0["speaker"] == "speaker_A"
    assert len(sample0["words"]) == 4
    assert sample0["original_words"] == sample0["words"]
    assert sample0["alternative_words"] is None

    # Sample out of bounds
    res = client.get("/api/sample/999")
    assert res.status_code == 404

    # Random sample
    res = client.get("/api/sample/random/get")
    assert res.status_code == 200
    rand_data = res.json()
    assert 0 <= rand_data["index"] < 3

    # Audio slice
    res = client.get("/api/audio/0")
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert "X-Pre-Padding" in res.headers
    assert len(res.content) > 0


def test_switch_manifest(temp_dataset, tmp_path: Path):
    dv.state = dv.ServerState(default_manifest=temp_dataset["manifest"])
    client = TestClient(dv.app)

    # Create a second manifest with 1 sample
    man2 = tmp_path / "second.jsonl"
    with open(man2, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "path": str(temp_dataset["audio_dir"] / "clip1.wav"),
                    "start": 0.0,
                    "duration": 2.0,
                    "transcript": "קצר",
                    "words": [{"word": "קצר", "start": 0.0, "end": 2.0}],
                }
            )
            + "\n"
        )

    res = client.post("/api/set_manifest", json={"path": str(man2)})
    assert res.status_code == 200
    data = res.json()
    assert data["total_samples"] == 1
    assert data["current_name"] == "second.jsonl"

    # Check that sample 0 reflects the new manifest
    res = client.get("/api/sample/0")
    assert res.status_code == 200
    assert res.json()["transcript"] == "קצר"


@pytest.mark.skipif(
    not dv.DEFAULT_VALID_MANIFEST_PATH.exists(), reason="Plenum dataset not present on machine"
)
def test_real_plenum_dataset_if_present():
    dv.state = dv.ServerState(default_manifest=dv.DEFAULT_VALID_MANIFEST_PATH)
    client = TestClient(dv.app)

    res = client.get("/api/manifests")
    assert res.status_code == 200
    assert res.json()["total_samples"] > 0

    res = client.get("/api/sample/0")
    assert res.status_code == 200
    sample = res.json()
    assert sample["audio_exists"] is True
    assert len(sample["words"]) > 0

    res = client.get("/api/audio/0")
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert len(res.content) > 100000


def test_calibration_mode_dataset_and_audio():
    cd = dv.CalibrationDataset()
    assert len(cd) == 10

    sample0, wav_bytes = cd.generate_sample(0)
    assert sample0["total_audio_duration"] == 10.0
    assert sample0["is_calibration"] is True
    assert len(sample0["words"]) > 0

    sr = 24000
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        assert w.getframerate() == sr
        assert w.getnframes() == sr * 10
        raw_frames = w.readframes(w.getnframes())
        samples = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32767.0

    for word in sample0["words"]:
        assert word["word"] in ("high", "low", "silence")
        # 10ms boundary checks
        assert word["start"] == round(word["start"], 2)
        assert word["end"] == round(word["end"], 2)
        assert word["duration"] == round(word["duration"], 2)

        s_idx = round(word["audio_start"] * sr)
        e_idx = round(word["audio_end"] * sr)
        slice_audio = samples[s_idx:e_idx]

        if word["word"] == "silence":
            assert word["duration"] >= 0.099 and word["duration"] <= 2.001
            assert np.max(np.abs(slice_audio)) == 0.0
        elif word["word"] == "high":
            assert word["duration"] <= 1.501
            fft = np.abs(np.fft.rfft(slice_audio))
            freqs = np.fft.rfftfreq(len(slice_audio), 1 / sr)
            peak_freq = freqs[np.argmax(fft)]
            assert abs(peak_freq - 440) < 6.0
        elif word["word"] == "low":
            assert word["duration"] <= 1.501
            fft = np.abs(np.fft.rfft(slice_audio))
            freqs = np.fft.rfftfreq(len(slice_audio), 1 / sr)
            peak_freq = freqs[np.argmax(fft)]
            assert abs(peak_freq - 220) < 6.0


def test_calibration_fastapi_endpoints():
    dv.state = dv.ServerState(default_manifest=Path("calibrate"))
    client = TestClient(dv.app)

    # 1. Manifests endpoint
    res = client.get("/api/manifests")
    assert res.status_code == 200
    data = res.json()
    assert data["is_calibration"] is True
    assert data["total_samples"] == 10

    # 2. Get sample 0
    res = client.get("/api/sample/0")
    assert res.status_code == 200
    s0 = res.json()
    assert s0["is_calibration"] is True
    assert s0["total_audio_duration"] == 10.0
    assert len(s0["words"]) > 0

    # 3. Random sample in calibration mode
    res = client.get("/api/sample/random/get")
    assert res.status_code == 200
    r_sample = res.json()
    assert r_sample["is_calibration"] is True

    # 4. Audio bytes endpoint
    res = client.get("/api/audio/0")
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert len(res.content) == 480044

    # 5. Switch back to a real manifest
    res = client.post("/api/set_manifest", json={"path": "calibrate"})
    assert res.status_code == 200
    assert res.json()["is_calibration"] is True
