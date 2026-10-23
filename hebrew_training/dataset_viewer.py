"""Self-contained server and viewer for Hebrew TTS dataset manifests.

Provides:
- Fast, memory-efficient line indexing with disk-cached byte offsets.
- On-demand audio slicing (±3s padding) without reading full multi-hour audio files.
- REST API for dataset browsing, random sampling, seeking, and audio streaming.
- Modern, responsive interactive HTML/JS UI with canvas waveform visualization,
  padding demarcation, aligned word durations, and synchronized audio playback.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import io
import json
import logging
import os
import random
import shutil
import struct
import subprocess
import threading
import wave
from collections import OrderedDict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response

try:
    import soundfile as sf
except ImportError:
    sf = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dataset_viewer")

# DEFAULT_DATASET_DIR = Path("/mlspeech/hebrew/pockettts-data/plenum")
DEFAULT_DATASET_DIR = Path("/mlspeech/hebrew/normalized/knesset/committee_sample")
DEFAULT_MANIFEST_PATH = DEFAULT_DATASET_DIR / "train_aligned.jsonl"
DEFAULT_VALID_MANIFEST_PATH = DEFAULT_DATASET_DIR / "valid_aligned.jsonl"
INDEX_MAGIC = b"JSONLIDX"
INDEX_VERSION = 1


# ---------------------------------------------------------------------------
# Indexing Engine
# ---------------------------------------------------------------------------


def _resolve_index_cache_path(manifest_path: Path) -> Path:
    """Determine cache path for .idx file (alongside file or in /tmp)."""
    sibling = manifest_path.with_suffix(manifest_path.suffix + ".idx")
    try:
        # Check if manifest directory is writable
        if os.access(manifest_path.parent, os.W_OK):
            return sibling
    except OSError:
        pass

    hash_key = hashlib.sha256(str(manifest_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(f"/tmp/{manifest_path.stem}_{hash_key}.idx")


def build_or_load_line_index(manifest_path: Path) -> array.array:
    """
    Build or load line byte offsets for a JSONL file.
    Uses array.array('Q') (64-bit unsigned integers) for extreme memory efficiency:
    ~11 MB of RAM for 1.5 million lines.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    stat = manifest_path.stat()
    file_size = stat.st_size
    mtime = int(stat.st_mtime)

    idx_path = _resolve_index_cache_path(manifest_path)

    # Attempt to load from binary cache
    if idx_path.exists():
        try:
            with open(idx_path, "rb") as f:
                header = f.read(36)
                if len(header) == 36:
                    magic, version, c_size, c_mtime, count = struct.unpack("<8sIQQQ", header)
                    if (
                        magic == INDEX_MAGIC
                        and version == INDEX_VERSION
                        and c_size == file_size
                        and c_mtime == mtime
                    ):
                        offsets = array.array("Q")
                        offsets.fromfile(f, count)
                        logger.info(
                            "Loaded %d line offsets from cache %s", len(offsets), idx_path.name
                        )
                        return offsets
        except (OSError, struct.error, ValueError) as exc:
            logger.warning("Cache load failed for %s, rebuilding: %s", idx_path, exc)

    logger.info(
        "Scanning line offsets for %s (%0.1f MB)...", manifest_path, file_size / (1024 * 1024)
    )
    offsets = array.array("Q")
    if file_size == 0:
        return offsets

    chunk_size = 16 * 1024 * 1024  # 16 MB chunks for high throughput
    pos = 0
    offsets.append(0)

    with open(manifest_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            idx = 0
            while True:
                idx = chunk.find(b"\n", idx)
                if idx == -1:
                    break
                next_pos = pos + idx + 1
                if next_pos < file_size:
                    offsets.append(next_pos)
                idx += 1
            pos += len(chunk)

    logger.info("Indexed %d lines in %s", len(offsets), manifest_path.name)

    # Attempt to save cache for fast startup next time
    try:
        header = struct.pack("<8sIQQQ", INDEX_MAGIC, INDEX_VERSION, file_size, mtime, len(offsets))
        temp_cache = idx_path.with_suffix(".tmp")
        with open(temp_cache, "wb") as f:
            f.write(header)
            offsets.tofile(f)
        temp_cache.replace(idx_path)
        logger.info("Saved index cache to %s", idx_path)
    except OSError as exc:
        logger.debug("Could not write cache file %s: %s", idx_path, exc)

    return offsets


class ManifestIndex:
    """Memory-efficient JSONL dataset index supporting instant line seeking."""

    def __init__(self, path: Path, audio_dir: Path | None = None) -> None:
        self.path = Path(path).resolve()
        self.audio_dir = Path(audio_dir).resolve() if audio_dir else None
        self.offsets = build_or_load_line_index(self.path)
        self.count = len(self.offsets)

    def __len__(self) -> int:
        return self.count

    def resolve_audio_path(self, raw_path: str) -> Path:
        """Resolve audio path with graceful fallbacks for moved/relative datasets."""
        p = Path(raw_path)
        if p.is_file():
            return p.resolve()
        if self.audio_dir and (self.audio_dir / p.name).is_file():
            return (self.audio_dir / p.name).resolve()
        manifest_dir = self.path.parent
        if (manifest_dir / p).is_file():
            return (manifest_dir / p).resolve()
        if (manifest_dir / "audio" / p.name).is_file():
            return (manifest_dir / "audio" / p.name).resolve()
        if (manifest_dir / p.name).is_file():
            return (manifest_dir / p.name).resolve()
        return p

    def get_row(self, index: int) -> dict[str, Any]:
        """Fetch raw JSON parsed dictionary at 0-based index."""
        if index < 0 or index >= self.count:
            raise IndexError(f"Sample index {index} out of range [0, {self.count - 1}]")
        offset = self.offsets[index]
        with open(self.path, "rb") as f:
            f.seek(offset)
            line = f.readline()
        if not line:
            raise ValueError(f"Empty line read at index {index} (offset {offset})")
        return json.loads(line.decode("utf-8"))

    def get_sample_data(self, index: int, padding: float = 3.0) -> dict[str, Any]:
        """
        Produce a normalized sample payload with exact timestamps and word boundaries
        both relative to sample start and relative to served audio (with padding).
        """
        row = self.get_row(index)

        raw_path = str(row.get("path") or row.get("audio_path") or "")
        resolved_path = self.resolve_audio_path(raw_path)
        audio_exists = resolved_path.is_file()

        start = float(row.get("start", 0.0))
        duration = float(row.get("duration", 0.0))
        transcript = str(row.get("transcript") or row.get("text") or "").strip()
        speaker = str(row.get("speaker") or row.get("speaker_id") or row.get("user_id") or "")

        # Compute audio slice boundaries
        pre_padding = min(start, padding)
        sample_audio_start = pre_padding
        sample_audio_end = pre_padding + duration
        post_padding = padding  # Nominal; clamped to file length upon actual read
        total_audio_duration = pre_padding + duration + post_padding

        # Words processing
        def normalize_words(raw_words: Any, origin: float) -> list[dict[str, Any]]:
            words: list[dict[str, Any]] = []
            if not isinstance(raw_words, list):
                return words
            for w in raw_words:
                if not isinstance(w, dict):
                    continue
                w_text = str(w.get("word", "")).strip()
                try:
                    w_start = round(float(w.get("start", 0.0)) + origin - start, 3)
                except (ValueError, TypeError):
                    w_start = 0.0
                try:
                    w_end = round(float(w.get("end", w.get("start", 0.0))) + origin - start, 3)
                except (ValueError, TypeError):
                    w_end = w_start
                w_end = max(w_end, w_start)
                w_dur = round(w_end - w_start, 3)

                words.append(
                    {
                        "word": w_text,
                        "start": w_start,
                        "end": w_end,
                        "duration": w_dur,
                        "audio_start": round(pre_padding + w_start, 3),
                        "audio_end": round(pre_padding + w_end, 3),
                    }
                )
            return words

        words = normalize_words(row.get("words", []), start)
        has_snapshot = isinstance(row.get("original_words"), list)
        original_start = float(row.get("original_start", start)) if has_snapshot else start
        original_duration = (
            float(row.get("original_duration", duration)) if has_snapshot else duration
        )
        original_words = (
            normalize_words(row["original_words"], original_start) if has_snapshot else words
        )
        comparisons = []
        if has_snapshot:
            matcher = SequenceMatcher(
                a=[w["word"] for w in original_words], b=[w["word"] for w in words], autojunk=False
            )
            for tag, a0, a1, b0, b1 in matcher.get_opcodes():
                if tag == "equal":
                    comparisons.extend(
                        {"original_index": a, "alternative_index": b}
                        for a, b in zip(range(a0, a1), range(b0, b1))
                    )
                else:
                    comparisons.extend(
                        {"original_index": a, "alternative_index": None} for a in range(a0, a1)
                    )
                    comparisons.extend(
                        {"original_index": None, "alternative_index": b} for b in range(b0, b1)
                    )

        return {
            "index": index,
            "line_number": index + 1,
            "total_samples": self.count,
            "manifest_name": self.path.name,
            "manifest_path": str(self.path),
            "audio_path": str(resolved_path),
            "audio_filename": resolved_path.name,
            "audio_exists": audio_exists,
            "speaker": speaker,
            "start": round(start, 3),
            "duration": round(duration, 3),
            "transcript": transcript,
            "padding": round(padding, 3),
            "pre_padding": round(pre_padding, 3),
            "post_padding": round(post_padding, 3),
            "total_audio_duration": round(total_audio_duration, 3),
            "sample_audio_start": round(sample_audio_start, 3),
            "sample_audio_end": round(sample_audio_end, 3),
            "words": words,
            "original_words": original_words,
            "original_start": original_start,
            "original_duration": original_duration,
            "alternative_words": words if has_snapshot else None,
            "alignment_backend": row.get(
                "alignment_backend", "unknown" if has_snapshot else "original"
            ),
            "alignment_fallback_reason": row.get("alignment_fallback_reason"),
            "alignment_error": row.get("alignment_error"),
            "word_comparisons": comparisons,
        }


class CalibrationDataset:
    """In-memory pseudo dataset for timing and leakage calibration."""

    COUNT = 10
    SR = 24000
    TOTAL_DUR = 10.0

    def __init__(self) -> None:
        self.count = self.COUNT
        self.path = Path("calibrate")
        self._samples: dict[int, tuple[dict[str, Any], bytes]] = {}

    def __len__(self) -> int:
        return self.count

    def generate_sample(self, index: int, randomize: bool = False) -> tuple[dict[str, Any], bytes]:
        if not randomize and index in self._samples:
            return self._samples[index]

        seed = None if randomize else (42 + index * 1009)
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        sr = self.SR
        total_dur = self.TOTAL_DUR
        total_samples = int(total_dur * sr)  # 240000

        # Pre-padding and post-padding aligned to 10ms boundary
        if index == 5:
            pre_pad = 0.0
            post_pad = 0.0
        elif index % 2 == 1:
            pre_pad = 1.5
            post_pad = 1.5
        else:
            pre_pad = 2.0
            post_pad = 2.0

        sample_dur = round(total_dur - pre_pad - post_pad, 3)
        sample_steps = round(sample_dur / 0.010)

        # Fill whole buffer with low-intensity white noise
        audio = np_rng.uniform(-0.035, 0.035, size=total_samples).astype(np.float32)

        curr_step = 0
        words: list[dict[str, Any]] = []
        events: list[tuple[str, int, int]] = []

        while curr_step < sample_steps:
            remaining = sample_steps - curr_step
            if remaining < 10:
                events.append(("noise", curr_step, sample_steps))
                break

            # In-between sections: low-intensity white noise
            need_noise = (len(events) > 0 and events[-1][0] != "noise" and rng.random() < 0.8) or (
                len(events) == 0 and rng.random() < 0.5
            )

            if need_noise:
                max_noise_steps = min(remaining - 10, rng.randint(10, 80))
                if max_noise_steps >= 10:
                    noise_dur_steps = rng.randint(10, max_noise_steps)
                    events.append(("noise", curr_step, curr_step + noise_dur_steps))
                    curr_step += noise_dur_steps
                    continue
                if max_noise_steps > 0:
                    events.append(("noise", curr_step, curr_step + max_noise_steps))
                    curr_step += max_noise_steps
                    continue

            tone_type = rng.choice(["high", "low", "silence"])
            if tone_type in ("high", "low"):
                # high and low are up to 1.5 seconds (150 steps = 1500ms)
                max_s = min(remaining, 150)
                if max_s < 10:
                    events.append(("noise", curr_step, sample_steps))
                    break
                dur_steps = rng.randint(10, max_s)
            else:  # silence
                # silence no less than 100ms (10 steps) and up to 2 seconds (200 steps)
                max_s = min(remaining, 200)
                if max_s < 10:
                    events.append(("noise", curr_step, sample_steps))
                    break
                dur_steps = rng.randint(10, max_s)

            events.append((tone_type, curr_step, curr_step + dur_steps))

            w_start = round(curr_step * 0.010, 3)
            w_end = round((curr_step + dur_steps) * 0.010, 3)
            w_dur = round(w_end - w_start, 3)

            audio_s = round(pre_pad + w_start, 3)
            audio_e = round(pre_pad + w_end, 3)

            start_sample = round(audio_s * sr)
            end_sample = round(audio_e * sr)
            num_s = end_sample - start_sample
            t = np.arange(num_s) / sr

            if tone_type == "high":
                audio[start_sample:end_sample] = 0.35 * np.sin(2 * np.pi * 440 * t)
            elif tone_type == "low":
                audio[start_sample:end_sample] = 0.35 * np.sin(2 * np.pi * 220 * t)
            elif tone_type == "silence":
                audio[start_sample:end_sample] = 0.0

            words.append(
                {
                    "word": tone_type,
                    "start": w_start,
                    "end": w_end,
                    "duration": w_dur,
                    "audio_start": audio_s,
                    "audio_end": audio_e,
                }
            )
            curr_step += dur_steps

        pcm_data = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm_data.tobytes())
        wav_bytes = buf.getvalue()

        transcript = " ".join(w["word"] for w in words)
        sample_data = {
            "index": index,
            "line_number": index + 1,
            "total_samples": self.COUNT,
            "manifest_name": "🎯 Calibration Dataset (Synthetic)",
            "manifest_path": "calibrate",
            "audio_path": f"synthetic://calibrate_{index}.wav",
            "audio_filename": f"calibrate_{index}.wav",
            "audio_exists": True,
            "speaker": "CALIBRATION",
            "start": round(pre_pad, 3),
            "duration": round(sample_dur, 3),
            "transcript": transcript,
            "padding": round(pre_pad, 3),
            "pre_padding": round(pre_pad, 3),
            "post_padding": round(post_pad, 3),
            "total_audio_duration": round(total_dur, 3),
            "sample_audio_start": round(pre_pad, 3),
            "sample_audio_end": round(pre_pad + sample_dur, 3),
            "words": words,
            "is_calibration": True,
        }

        self._samples[index] = (sample_data, wav_bytes)
        return sample_data, wav_bytes

    def get_sample_data(self, index: int, padding: float = 3.0) -> dict[str, Any]:
        data, _ = self.generate_sample(index, randomize=False)
        return data

    def get_audio_bytes(self, index: int) -> bytes:
        _, wav_bytes = self.generate_sample(index, randomize=False)
        return wav_bytes


# ---------------------------------------------------------------------------
# Audio Slicing Engine
# ---------------------------------------------------------------------------

_audio_slice_cache: OrderedDict[str, tuple[bytes, float, float, float]] = OrderedDict()
_CACHE_MAX_ENTRIES = 64
_cache_lock = threading.Lock()


def extract_audio_slice(
    audio_path: Path | str, start: float, duration: float, padding: float = 3.0
) -> tuple[bytes, float, float, float]:
    """
    Extract a slice of WAV audio around [start, start + duration] with +- padding.
    Returns (wav_bytes, pre_padding, post_padding, total_duration).
    """
    p = Path(audio_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Audio file not found: {p}")

    cache_key = f"{p}:{start:.3f}:{duration:.3f}:{padding:.3f}:{p.stat().st_mtime}"
    with _cache_lock:
        if cache_key in _audio_slice_cache:
            _audio_slice_cache.move_to_end(cache_key)
            return _audio_slice_cache[cache_key]

    req_start = max(0.0, start - padding)
    pre_pad = start - req_start
    req_end = start + duration + padding

    wav_bytes: bytes | None = None
    post_pad = padding
    total_dur = duration + pre_pad + post_pad

    # 1. Try soundfile
    if sf is not None:
        try:
            with sf.SoundFile(str(p)) as snd:
                sr = snd.samplerate
                total_frames = snd.frames
                file_dur = total_frames / sr

                end_time = min(file_dur, req_end)
                post_pad = max(0.0, end_time - (start + duration))

                start_frame = min(round(req_start * sr), total_frames)
                end_frame = min(round(end_time * sr), total_frames)
                num_frames = max(0, end_frame - start_frame)

                snd.seek(start_frame)
                data = snd.read(num_frames, dtype="int16")

                buf = io.BytesIO()
                sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
                wav_bytes = buf.getvalue()
                total_dur = len(data) / sr
        except (OSError, RuntimeError, ValueError) as exc:
            logger.debug("soundfile extraction failed: %s, trying wave", exc)

    # 2. Fallback to stdlib wave for standard PCM WAV files
    if wav_bytes is None:
        try:
            with wave.open(str(p), "rb") as w:
                sr = w.getframerate()
                nchannels = w.getnchannels()
                sampwidth = w.getsampwidth()
                total_frames = w.getnframes()
                file_dur = total_frames / sr

                end_time = min(file_dur, req_end)
                post_pad = max(0.0, end_time - (start + duration))

                start_frame = min(round(req_start * sr), total_frames)
                end_frame = min(round(end_time * sr), total_frames)
                num_frames = max(0, end_frame - start_frame)

                w.setpos(start_frame)
                pcm_bytes = w.readframes(num_frames)

                buf = io.BytesIO()
                with wave.open(buf, "wb") as out_w:
                    out_w.setnchannels(nchannels)
                    out_w.setsampwidth(sampwidth)
                    out_w.setframerate(sr)
                    out_w.writeframes(pcm_bytes)
                wav_bytes = buf.getvalue()
                total_dur = num_frames / sr
        except (OSError, RuntimeError, ValueError, wave.Error) as exc:
            logger.debug("wave extraction failed: %s, trying ffmpeg", exc)

    # 3. Fallback to ffmpeg for other audio codecs (e.g. m4a, mp3, flac)
    if wav_bytes is None:
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError(f"Unable to read audio {p} without soundfile, wave, or ffmpeg")
        req_len = duration + pre_pad + padding
        cmd = [
            ffmpeg_bin,
            "-ss",
            f"{req_start:.3f}",
            "-i",
            str(p),
            "-t",
            f"{req_len:.3f}",
            "-ar",
            "24000",
            "-ac",
            "1",
            "-f",
            "wav",
            "pipe:1",
        ]
        res = subprocess.run(cmd, capture_output=True, check=True)
        wav_bytes = res.stdout
        with wave.open(io.BytesIO(wav_bytes), "rb") as test_w:
            total_dur = test_w.getnframes() / test_w.getframerate()
            post_pad = max(0.0, total_dur - (pre_pad + duration))

    result = (wav_bytes, pre_pad, post_pad, total_dur)
    with _cache_lock:
        _audio_slice_cache[cache_key] = result
        if len(_audio_slice_cache) > _CACHE_MAX_ENTRIES:
            _audio_slice_cache.popitem(last=False)

    return result


# ---------------------------------------------------------------------------
# Web Application & REST API
# ---------------------------------------------------------------------------


class ServerState:
    def __init__(self, default_manifest: Path, audio_dir: Path | None = None) -> None:
        self.manifest_indexes: dict[str, ManifestIndex | CalibrationDataset] = {}
        self.audio_dir = audio_dir
        self.calibration_dataset = CalibrationDataset()
        self.manifest_indexes["calibrate"] = self.calibration_dataset

        if str(default_manifest) in ("calibrate", "calibration"):
            self.current_manifest_path = Path("calibrate")
        else:
            self.current_manifest_path = default_manifest.resolve()
            self._ensure_manifest_loaded(self.current_manifest_path)

    @property
    def is_calibration(self) -> bool:
        return str(self.current_manifest_path) in ("calibrate", "calibration")

    def _ensure_manifest_loaded(self, path: Path) -> ManifestIndex | CalibrationDataset:
        key = str(path)
        if key in ("calibrate", "calibration"):
            return self.calibration_dataset
        resolved = path.resolve()
        res_key = str(resolved)
        if res_key not in self.manifest_indexes:
            self.manifest_indexes[res_key] = ManifestIndex(resolved, audio_dir=self.audio_dir)
        return self.manifest_indexes[res_key]

    @property
    def current_index(self) -> ManifestIndex | CalibrationDataset:
        return self._ensure_manifest_loaded(self.current_manifest_path)

    def set_manifest(self, path: Path | str) -> ManifestIndex | CalibrationDataset:
        raw_str = str(path).strip()
        if raw_str in ("calibrate", "calibration"):
            self.current_manifest_path = Path("calibrate")
            return self.calibration_dataset

        resolved = Path(raw_str).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Manifest not found: {resolved}")
        idx = self._ensure_manifest_loaded(resolved)
        self.current_manifest_path = resolved
        return idx


app = FastAPI(title="Pocket TTS Hebrew Dataset Viewer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state: ServerState | None = None


def get_state() -> ServerState:
    global state
    if state is None:
        target = (
            DEFAULT_MANIFEST_PATH
            if DEFAULT_MANIFEST_PATH.exists()
            else (
                DEFAULT_VALID_MANIFEST_PATH
                if DEFAULT_VALID_MANIFEST_PATH.exists()
                else Path("train_aligned.jsonl")
            )
        )
        state = ServerState(default_manifest=target)
    return state


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/manifests")
async def get_manifests() -> dict[str, Any]:
    st = get_state()
    discovered: list[dict[str, Any]] = [
        {"name": "🎯 Calibration Dataset (Synthetic)", "path": "calibrate"}
    ]

    manifest_dir = st.current_manifest_path.parent if not st.is_calibration else DEFAULT_DATASET_DIR
    if manifest_dir.is_dir():
        for f in sorted(manifest_dir.glob("*.jsonl")):
            discovered.append({"name": f.name, "path": str(f.resolve())})

    return {
        "current_manifest": str(st.current_manifest_path),
        "current_name": (
            "🎯 Calibration Dataset" if st.is_calibration else st.current_manifest_path.name
        ),
        "total_samples": len(st.current_index),
        "is_calibration": st.is_calibration,
        "available": discovered,
    }


@app.post("/api/set_manifest")
async def set_manifest_endpoint(payload: dict[str, str]) -> dict[str, Any]:
    st = get_state()
    raw_path = payload.get("path", "").strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail="Missing path")
    try:
        idx = st.set_manifest(raw_path)
        return {
            "status": "ok",
            "current_manifest": str(idx.path),
            "current_name": "🎯 Calibration Dataset" if st.is_calibration else idx.path.name,
            "total_samples": len(idx),
            "is_calibration": st.is_calibration,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sample/{index}")
async def get_sample(index: int, padding: float = Query(3.0, ge=0.0, le=10.0)) -> dict[str, Any]:
    st = get_state()
    idx = st.current_index
    if index < 0 or index >= len(idx):
        raise HTTPException(
            status_code=404, detail=f"Index {index} out of range [0, {len(idx) - 1}]"
        )
    return idx.get_sample_data(index, padding=padding)


@app.get("/api/sample/random/get")
async def get_random_sample(padding: float = Query(3.0, ge=0.0, le=10.0)) -> dict[str, Any]:
    st = get_state()
    idx = st.current_index
    if len(idx) == 0:
        raise HTTPException(status_code=404, detail="Dataset is empty")
    rand_index = random.randint(0, len(idx) - 1)
    if st.is_calibration:
        data, _ = st.calibration_dataset.generate_sample(rand_index, randomize=True)
        return data
    return idx.get_sample_data(rand_index, padding=padding)


@app.get("/api/audio/{index}")
async def get_audio_slice(index: int, padding: float = Query(3.0, ge=0.0, le=10.0)) -> Response:
    st = get_state()
    idx = st.current_index
    if index < 0 or index >= len(idx):
        raise HTTPException(status_code=404, detail=f"Index {index} out of range")

    if st.is_calibration:
        sample = st.calibration_dataset.get_sample_data(index)
        wav_bytes = st.calibration_dataset.get_audio_bytes(index)
        headers = {
            "Content-Type": "audio/wav",
            "Content-Length": str(len(wav_bytes)),
            "Accept-Ranges": "bytes",
            "X-Pre-Padding": f"{sample['pre_padding']:.3f}",
            "X-Post-Padding": f"{sample['post_padding']:.3f}",
            "X-Total-Duration": f"{sample['total_audio_duration']:.3f}",
        }
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)

    sample = idx.get_sample_data(index, padding=padding)
    audio_path = Path(sample["audio_path"])
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

    try:
        wav_bytes, pre_pad, post_pad, total_dur = extract_audio_slice(
            audio_path, sample["start"], sample["duration"], padding=padding
        )
        headers = {
            "Content-Type": "audio/wav",
            "Content-Length": str(len(wav_bytes)),
            "Accept-Ranges": "bytes",
            "X-Pre-Padding": f"{pre_pad:.3f}",
            "X-Post-Padding": f"{post_pad:.3f}",
            "X-Total-Duration": f"{total_dur:.3f}",
        }
        return Response(content=wav_bytes, media_type="audio/wav", headers=headers)
    except Exception as exc:
        logger.exception("Error extracting audio for sample %d", index)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Single-Page Frontend HTML/JS/CSS
# ---------------------------------------------------------------------------

HTML_UI = """<!DOCTYPE html>
<html lang="he" dir="ltr" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pocket TTS • Hebrew Dataset Sample Viewer</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#eef2ff',
              500: '#6366f1',
              600: '#4f46e5',
              700: '#4338ca',
            }
          }
        }
      }
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    body {
      font-family: 'Assistant', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .font-mono {
      font-family: 'JetBrains Mono', monospace;
    }

    .word-token {
      transition: all 0.15s ease-in-out;
      cursor: pointer;
    }
    .word-token:hover {
      background-color: #3b82f6;
      color: #ffffff;
      transform: translateY(-1px);
    }
    .word-token.active {
      background-color: #10b981 !important;
      color: #ffffff !important;
      font-weight: 700;
      box-shadow: 0 0 12px rgba(16, 185, 129, 0.6);
      transform: scale(1.06);
    }
    .word-token.highlighted {
      background-color: #f59e0b;
      color: #111827;
      font-weight: 600;
    }
    .timing-lane .word-token {
      font-size: 70%;
    }

    .word-row.active {
      background-color: rgba(16, 185, 129, 0.18) !important;
      border-left: 3px solid #10b981;
    }
    .word-row.highlighted {
      background-color: rgba(245, 158, 11, 0.15) !important;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    ::-webkit-scrollbar-track {
      background: #1e293b;
    }
    ::-webkit-scrollbar-thumb {
      background: #475569;
      border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #64748b;
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col selection:bg-indigo-500 selection:text-white">

  <!-- Top Navigation Bar -->
  <header class="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-30 px-4 py-3">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-3">
      
      <!-- Brand & Status -->
      <div class="flex items-center space-x-3">
        <div class="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-600/30 text-white font-black text-lg">
          🎙️
        </div>
        <div>
          <h1 class="text-base font-bold text-slate-100 flex items-center gap-2">
            Hebrew Dataset Sample Viewer
            <span class="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 font-normal">Pocket TTS</span>
          </h1>
          <p id="manifest-badge" class="text-xs text-slate-400 font-mono truncate max-w-sm">Loading manifest...</p>
        </div>
      </div>

      <!-- Manifest Chooser & Calibration -->
      <div class="flex items-center gap-2">
        <select id="manifest-select" class="bg-slate-800 border border-slate-700 text-xs rounded-lg px-2.5 py-1.5 text-slate-200 focus:outline-none focus:ring-1 focus:ring-indigo-500">
          <option value="">(Switch manifest...)</option>
        </select>
        <button id="custom-manifest-btn" title="Open custom path" class="px-2 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs border border-slate-700">
          📂 Path
        </button>
        <button id="calibrate-btn" title="Switch to Calibration Mode" class="px-2.5 py-1.5 bg-emerald-700 hover:bg-emerald-600 text-white rounded-lg text-xs font-semibold flex items-center gap-1 transition shadow-sm">
          🎯 <span>Calibrate</span>
        </button>
      </div>

      <!-- Sample Navigation -->
      <div class="flex items-center gap-1.5">
        <button id="prev-btn" title="Previous Sample (Left Arrow)" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium flex items-center gap-1 transition">
          ◀ <span class="hidden sm:inline">Prev</span>
        </button>
        
        <div class="flex items-center bg-slate-900 border border-slate-700 rounded-lg px-2 py-1">
          <span class="text-xs text-slate-400 mr-1.5">#</span>
          <input id="sample-index-input" type="number" min="0" class="w-20 bg-transparent text-center text-xs font-mono text-slate-100 focus:outline-none" value="0">
          <span id="total-samples-label" class="text-xs text-slate-400 ml-1 font-mono">/ ...</span>
          <button id="go-btn" class="ml-2 text-xs bg-indigo-600 hover:bg-indigo-500 text-white px-2 py-0.5 rounded font-medium transition">
            Go
          </button>
        </div>

        <button id="next-btn" title="Next Sample (Right Arrow)" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-medium flex items-center gap-1 transition">
          <span class="hidden sm:inline">Next</span> ▶
        </button>

        <button id="random-btn" title="Random Sample (R)" class="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold shadow-md shadow-indigo-600/30 flex items-center gap-1 transition">
          🎲 <span>Random</span>
        </button>
      </div>

    </div>
  </header>

  <!-- Main Workspace -->
  <main class="max-w-7xl mx-auto w-full p-4 space-y-4 flex-1">

    <!-- Metadata Banner -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-3.5 shadow-sm">
      <div class="flex flex-wrap items-center justify-between gap-3 text-xs">
        
        <div class="flex flex-wrap items-center gap-2">
          <!-- Speaker -->
          <div class="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/60">
            <span class="text-slate-400">Speaker:</span>
            <span id="speaker-badge" class="font-mono font-semibold text-indigo-400">-</span>
          </div>

          <!-- Audio File -->
          <div class="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/60">
            <span class="text-slate-400">Audio:</span>
            <span id="audio-filename-badge" class="font-mono text-slate-200 truncate max-w-xs" title="">-</span>
          </div>

          <!-- In-file offset -->
          <div class="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/60">
            <span class="text-slate-400">File Offset:</span>
            <span id="start-offset-badge" class="font-mono text-slate-200">-</span>
          </div>
        </div>

        <div class="flex flex-wrap items-center gap-2">
          <!-- Duration -->
          <div class="flex items-center gap-1.5 bg-emerald-950/40 text-emerald-300 border border-emerald-800/50 px-2.5 py-1 rounded-md">
            <span>Sample Duration:</span>
            <span id="sample-dur-badge" class="font-mono font-bold">0.00s</span>
          </div>

          <!-- Padding -->
          <div class="flex items-center gap-1.5 bg-amber-950/40 text-amber-300 border border-amber-800/50 px-2.5 py-1 rounded-md">
            <span>Padding:</span>
            <span id="padding-badge" class="font-mono font-bold">±3.00s</span>
          </div>

          <!-- Words count -->
          <div class="flex items-center gap-1.5 bg-slate-800/80 px-2.5 py-1 rounded-md border border-slate-700/60 text-slate-300">
            <span>Words:</span>
            <span id="words-count-badge" class="font-mono font-bold">0</span>
          </div>
        </div>

      </div>
    </div>

    <!-- Hebrew Transcript Card -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg relative">
      <div class="flex items-center justify-between mb-3 text-xs text-slate-400 pb-2 border-b border-slate-800">
        <span class="font-semibold uppercase tracking-wider text-slate-400">Hebrew Transcription</span>
        <div class="flex items-center gap-3">
          <span class="text-[11px] text-slate-500">Click any word to play it individually</span>
          <button id="copy-transcript-btn" class="hover:text-indigo-400 text-slate-400 transition" title="Copy transcript">
            📋 Copy
          </button>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div class="timing-lane rounded-lg border border-slate-800 bg-slate-950/40 p-3">
          <div id="original-track-label" class="text-[10px] uppercase tracking-wider text-indigo-400 font-bold mb-2">Original timings</div>
          <div id="transcript-container" dir="rtl" class="text-[1.05rem] sm:text-[1.35rem] font-normal leading-relaxed text-slate-100 flex flex-wrap gap-x-1.5 gap-y-2 min-h-[50px] items-center">
            <!-- Original word tokens inserted dynamically -->
          </div>
        </div>
        <div class="timing-lane rounded-lg border border-fuchsia-900/50 bg-fuchsia-950/10 p-3">
          <div class="flex items-center justify-between text-[10px] uppercase tracking-wider text-fuchsia-300 font-bold mb-2">
             <span id="alternative-track-label">Saved realignment</span>
             <span id="alignment-status" class="normal-case tracking-normal text-slate-500 font-normal"></span>
          </div>
          <div id="alternative-transcript-container" dir="rtl" class="text-[1.05rem] sm:text-[1.35rem] font-normal leading-relaxed text-slate-100 flex flex-wrap gap-x-1.5 gap-y-2 min-h-[50px] items-center">
              <span class="text-sm text-slate-500">No saved realignment in this manifest.</span>
          </div>
        </div>
      </div>
      <p class="text-xs text-slate-500 mt-3">All times use the displayed sample's start as zero, including original timings. Negative times are before that start. Differences are saved result minus original.</p>
    </div>

    <!-- Waveform & Playback Visualizer Card -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-4 shadow-lg space-y-3">
      
      <!-- Legend & Padding Banner -->
      <div class="flex items-center justify-between text-xs px-1 text-slate-400">
        <div class="flex items-center gap-4">
          <div class="flex items-center gap-1.5">
            <span class="inline-block w-3 h-3 rounded bg-slate-700/60 border border-slate-600"></span>
            <span>Padding Region (±3s)</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="inline-block w-3 h-3 rounded bg-indigo-500 border border-indigo-400"></span>
            <span>Target Sample Audio</span>
          </div>
          <div class="flex items-center gap-1.5">
            <span class="inline-block w-3 h-3 rounded bg-emerald-500"></span>
            <span>Word Markers</span>
          </div>
        </div>
        <div id="loading-spinner" class="hidden flex items-center gap-1.5 text-indigo-400">
          <svg class="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24">
            <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"></circle>
            <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
          </svg>
          <span>Loading Audio...</span>
        </div>
      </div>

      <!-- Waveform Overview / Minimap -->
      <div class="relative w-full h-[26px] bg-slate-950 rounded-md border border-slate-800 overflow-hidden cursor-pointer select-none" id="overview-wrapper" title="Click or drag to pan zoomed view">
        <canvas id="overview-canvas" class="w-full h-full block"></canvas>
      </div>

      <!-- Zoom and Display Controls Bar -->
      <label class="flex items-center gap-2 text-xs text-slate-300">
        Main waveform / word track:
        <select id="alignment-select" class="bg-slate-800 border border-slate-700 rounded px-2 py-1">
          <option value="original">Original</option>
        </select>
        <span class="text-slate-500">Double-click a word token to zoom into its interval. Shift-drag to pan.</span>
      </label>
      <div class="flex flex-wrap items-center justify-between gap-2 text-xs bg-slate-950/70 px-3 py-1.5 rounded-lg border border-slate-800/80">
        <div class="flex items-center gap-2">
          <span class="text-slate-400 font-medium flex items-center gap-1">
            🔍 <span>Zoom:</span>
          </span>
          <button id="zoom-out-btn" title="Zoom Out (-)" class="w-6 h-6 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold flex items-center justify-center transition">
            −
          </button>
          <input id="zoom-slider" type="range" min="1" max="10" step="0.25" value="1" class="w-24 sm:w-32 h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer">
          <span id="zoom-label" class="font-mono text-indigo-400 font-bold w-12 text-center">100%</span>
          <button id="zoom-in-btn" title="Zoom In (+)" class="w-6 h-6 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold flex items-center justify-center transition">
            +
          </button>
          
          <div class="hidden sm:flex items-center gap-1 ml-1">
            <button class="zoom-preset-btn px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-mono transition" data-zoom="1">1x</button>
            <button class="zoom-preset-btn px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-mono transition" data-zoom="2">2x</button>
            <button class="zoom-preset-btn px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-mono transition" data-zoom="4">4x</button>
            <button class="zoom-preset-btn px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-mono transition" data-zoom="8">8x</button>
          </div>
        </div>

        <div class="flex items-center gap-2">
          <button id="fit-sample-btn" title="Fit Sample to Viewport (F)" class="px-2.5 py-1 bg-indigo-600/80 hover:bg-indigo-600 text-white rounded text-[11px] font-medium transition flex items-center gap-1 shadow-sm">
            🎯 <span>Fit Sample</span>
          </button>
          <button id="zoom-reset-btn" title="Reset Zoom to 100% (0)" class="px-2 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[11px] font-medium transition">
            Reset (1x)
          </button>
          <label class="flex items-center gap-1.5 cursor-pointer text-slate-400 hover:text-slate-200 ml-1">
            <input id="auto-follow-checkbox" type="checkbox" checked class="rounded bg-slate-800 border-slate-700 text-indigo-600 focus:ring-0">
            <span class="text-[11px]">Follow Playhead</span>
          </label>
        </div>
      </div>

      <!-- Canvas Container -->
      <div class="relative w-full rounded-lg bg-slate-950 border border-slate-800 overflow-x-auto overflow-y-hidden select-none cursor-pointer" id="canvas-wrapper">
        <canvas id="waveform-canvas" class="h-[185px] block"></canvas>
        <div id="canvas-tooltip" class="absolute hidden pointer-events-none bg-slate-900/90 text-slate-200 text-xs px-2 py-1 rounded shadow-lg border border-slate-700 z-10 font-mono"></div>
      </div>

      <!-- Audio Controls & Transport -->
      <div class="flex flex-wrap items-center justify-between gap-3 pt-2">
        
        <!-- Main Play Buttons -->
        <div class="flex items-center gap-2">
          <!-- Play Sample Only (Recommended Default) -->
          <button id="play-sample-btn" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg text-xs flex items-center gap-2 shadow-md shadow-emerald-600/30 transition">
            <span id="play-sample-icon">▶</span>
            <span>Play Sample (P)</span>
          </button>

          <button id="last-word-btn" class="px-3.5 py-2 bg-amber-600 hover:bg-amber-500 text-white font-semibold rounded-lg text-xs flex items-center gap-1.5 transition">
            ▶ Last Word + Tail
          </button>

          <!-- Play Full with Padding -->
          <button id="play-full-btn" class="px-3.5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold rounded-lg text-xs flex items-center gap-1.5 shadow-md shadow-indigo-600/30 transition">
            <span id="play-full-icon">▶</span>
            <span>Play Full [±3s] (Space)</span>
          </button>

          <!-- Stop -->
          <button id="stop-btn" class="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium transition">
            ⏹ Stop (Esc)
          </button>

          <!-- Loop Sample -->
          <button id="loop-btn" class="px-2.5 py-2 bg-slate-800 text-slate-400 hover:text-slate-200 rounded-lg text-xs border border-transparent font-medium transition" title="Toggle Loop (L)">
            🔁 Loop
          </button>
        </div>

        <!-- Time Display -->
        <div class="flex items-center gap-2 font-mono text-xs">
          <div class="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
            <span class="text-slate-500">Audio: </span>
            <span id="current-time-label" class="text-emerald-400 font-bold">00.00s</span>
            <span class="text-slate-500"> / </span>
            <span id="total-time-label" class="text-slate-300">00.00s</span>
          </div>

          <div class="bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800">
            <span class="text-slate-500">Sample: </span>
            <span id="sample-rel-time-label" class="text-indigo-400 font-bold">+00.00s</span>
          </div>
        </div>

        <!-- Secondary Controls: Speed & Volume -->
        <div class="flex items-center gap-3 text-xs">
          <div class="flex items-center gap-1">
            <span class="text-slate-400">Speed:</span>
            <select id="playback-rate-select" class="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-slate-200 focus:outline-none">
              <option value="0.5">0.5x</option>
              <option value="0.75">0.75x</option>
              <option value="1.0" selected>1.0x</option>
              <option value="1.25">1.25x</option>
              <option value="1.5">1.5x</option>
            </select>
          </div>

          <div class="flex items-center gap-1.5">
            <span class="text-slate-400">🔊</span>
            <input id="volume-slider" type="range" min="0" max="1" step="0.05" value="1" class="w-16 h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer">
          </div>
        </div>

      </div>

    </div>

    <!-- Word Breakdown Timing Table -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
      <div class="px-4 py-3 bg-slate-800/60 border-b border-slate-800 flex items-center justify-between text-xs">
        <div class="flex items-center gap-2">
          <span class="font-bold uppercase tracking-wider text-slate-300">Original Word Timings</span>
          <span id="table-words-count" class="px-2 py-0.5 rounded-full bg-slate-700 text-slate-300 text-[11px] font-mono">0 words</span>
        </div>
        <span class="text-slate-400 text-[11px]">Click row or play icon to isolate word</span>
      </div>

      <div class="max-h-72 overflow-auto">
        <table class="w-full text-left text-xs">
          <thead class="bg-slate-950/80 sticky top-0 text-slate-400 uppercase font-mono text-[11px] border-b border-slate-800">
            <tr>
              <th class="py-2 px-3 w-12 text-center">#</th>
              <th class="py-2 px-3 text-right">Word (Hebrew)</th>
              <th class="py-2 px-3 text-center font-mono">Sample Start</th>
              <th class="py-2 px-3 text-center font-mono">Sample End</th>
              <th class="py-2 px-3 text-center font-mono">Duration</th>
              <th class="py-2 px-3 text-center font-mono">Audio Interval</th>
              <th class="py-2 px-3 text-center w-20">Play</th>
            </tr>
          </thead>
          <tbody id="words-table-body" class="divide-y divide-slate-800/60 font-mono">
            <!-- Dynamic rows -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- Alternative Word Breakdown Timing Table -->
    <div id="alternative-timing-card" class="hidden bg-slate-900 border border-fuchsia-900/50 rounded-xl overflow-hidden shadow-lg">
      <div class="px-4 py-3 bg-fuchsia-950/30 border-b border-fuchsia-900/50 flex items-center justify-between text-xs">
        <div class="flex items-center gap-2">
            <span id="alternative-timing-title" class="font-bold uppercase tracking-wider text-fuchsia-200">Saved vs Original Word Timings</span>
          <span id="alternative-table-words-count" class="px-2 py-0.5 rounded-full bg-fuchsia-950 text-fuchsia-200 text-[11px] font-mono">0 words</span>
        </div>
          <span class="text-slate-400 text-[11px]">Δ = saved − original, in milliseconds</span>
      </div>
      <div class="max-h-72 overflow-auto">
        <table class="w-full text-left text-xs">
          <thead class="bg-slate-950/80 sticky top-0 text-slate-400 uppercase font-mono text-[11px] border-b border-slate-800">
            <tr>
              <th class="py-2 px-3 w-12 text-center">#</th>
              <th class="py-2 px-3 text-right">Word (Hebrew)</th>
              <th class="py-2 px-3 text-center font-mono">Sample Start</th>
              <th class="py-2 px-3 text-center font-mono">Sample End</th>
              <th class="py-2 px-3 text-center font-mono">Duration</th>
              <th class="py-2 px-3 text-center font-mono">Audio Interval</th>
              <th class="py-2 px-3 text-center w-20">Play</th>
            </tr>
          </thead>
          <tbody id="alternative-words-table-body" class="divide-y divide-fuchsia-900/30 font-mono"></tbody>
        </table>
      </div>
    </div>

  </main>

  <!-- Footer / Shortcuts Guide -->
  <footer class="border-t border-slate-800 bg-slate-950 text-slate-500 text-[11px] py-2 px-4">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
      <div class="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span><kbd class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">Space</kbd> Play Full</span>
        <span><kbd class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">P</kbd> Play Sample</span>
        <span><kbd class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">Esc</kbd> Stop</span>
        <span><kbd class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">R</kbd> Random</span>
        <span><kbd class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">← / →</kbd> Prev / Next</span>
        <span><kbd class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">+ / -</kbd> Zoom</span>
        <span><kbd class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">F</kbd> Fit Sample</span>
        <span><kbd class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">0</kbd> Reset (1x)</span>
        <span><kbd class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">Ctrl+Wheel</kbd> Zoom at Cursor</span>
        <span><kbd class="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">L</kbd> Loop</span>
      </div>
      <div>
        <span>Plenum Hebrew Dataset • Kyutai Pocket TTS</span>
      </div>
    </div>
  </footer>

  <!-- Application Logic -->
  <script>
    // State variables
    let currentSample = null;
    let audioContext = null;
    let audioBuffer = null;
    let currentSource = null;
    let isPlaying = false;
    let playbackMode = null; // 'sample', 'full', 'word'
    let playStartTime = 0;
    let playStartOffset = 0;
    let playEndTime = 0;
    let loopMode = false;
    let hoveredWordIndex = null;
    let animationFrameId = null;
    let zoomLevel = 1.0;
    let autoFollowPlayhead = true;
    let isDraggingOverview = false;
     let isPanningCanvas = false;
     let panStartX = 0;
     let panStartScrollLeft = 0;
      let selectedAlignment = 'alternative';

    // DOM Elements
    const manifestSelect = document.getElementById('manifest-select');
    const customManifestBtn = document.getElementById('custom-manifest-btn');
    const calibrateBtn = document.getElementById('calibrate-btn');
    const manifestBadge = document.getElementById('manifest-badge');
    const sampleIndexInput = document.getElementById('sample-index-input');
    const totalSamplesLabel = document.getElementById('total-samples-label');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const randomBtn = document.getElementById('random-btn');
    const goBtn = document.getElementById('go-btn');

    const speakerBadge = document.getElementById('speaker-badge');
    const audioFilenameBadge = document.getElementById('audio-filename-badge');
    const startOffsetBadge = document.getElementById('start-offset-badge');
    const sampleDurBadge = document.getElementById('sample-dur-badge');
    const paddingBadge = document.getElementById('padding-badge');
    const wordsCountBadge = document.getElementById('words-count-badge');
    const transcriptContainer = document.getElementById('transcript-container');
    const alternativeTranscriptContainer = document.getElementById('alternative-transcript-container');
    const copyTranscriptBtn = document.getElementById('copy-transcript-btn');
      const alignmentSelect = document.getElementById('alignment-select');
      const alternativeTrackLabel = document.getElementById('alternative-track-label');
     const alignmentStatus = document.getElementById('alignment-status');
     const alternativeTimingTitle = document.getElementById('alternative-timing-title');
     const lastWordBtn = document.getElementById('last-word-btn');

    const overviewWrapper = document.getElementById('overview-wrapper');
    const overviewCanvas = document.getElementById('overview-canvas');
    const zoomSlider = document.getElementById('zoom-slider');
    const zoomLabel = document.getElementById('zoom-label');
    const zoomInBtn = document.getElementById('zoom-in-btn');
    const zoomOutBtn = document.getElementById('zoom-out-btn');
    const zoomResetBtn = document.getElementById('zoom-reset-btn');
    const fitSampleBtn = document.getElementById('fit-sample-btn');
    const autoFollowCheckbox = document.getElementById('auto-follow-checkbox');
    const zoomPresetBtns = document.querySelectorAll('.zoom-preset-btn');

    const canvasWrapper = document.getElementById('canvas-wrapper');
    const canvas = document.getElementById('waveform-canvas');
    const canvasTooltip = document.getElementById('canvas-tooltip');
    const loadingSpinner = document.getElementById('loading-spinner');

    const playSampleBtn = document.getElementById('play-sample-btn');
    const playSampleIcon = document.getElementById('play-sample-icon');
    const playFullBtn = document.getElementById('play-full-btn');
    const playFullIcon = document.getElementById('play-full-icon');
    const stopBtn = document.getElementById('stop-btn');
    const loopBtn = document.getElementById('loop-btn');
    const currentTimeLabel = document.getElementById('current-time-label');
    const totalTimeLabel = document.getElementById('total-time-label');
    const sampleRelTimeLabel = document.getElementById('sample-rel-time-label');
    const playbackRateSelect = document.getElementById('playback-rate-select');
    const volumeSlider = document.getElementById('volume-slider');

    const tableWordsCount = document.getElementById('table-words-count');
    const wordsTableBody = document.getElementById('words-table-body');
    const alternativeTimingCard = document.getElementById('alternative-timing-card');
    const alternativeTableWordsCount = document.getElementById('alternative-table-words-count');
    const alternativeWordsTableBody = document.getElementById('alternative-words-table-body');

    // Initialize AudioContext on first interaction
    function getAudioContext() {
      if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
      }
      if (audioContext.state === 'suspended') {
        audioContext.resume();
      }
      return audioContext;
    }

    // Format seconds as SS.ss or MM:SS.ss
    function formatTime(sec) {
      if (isNaN(sec) || sec < 0) sec = 0;
      const m = Math.floor(sec / 60);
      const s = (sec % 60).toFixed(2);
      if (m > 0) {
        return `${m}:${s.padStart(5, '0')}`;
      }
      return `${s.padStart(5, '0')}s`;
    }

    // Initial setup
    async function init() {
      await loadManifests();
      await loadSample(0);
      setupEvents();
      setupKeyboard();
    }

    // Manifest management
    async function loadManifests() {
      try {
        const res = await fetch('/api/manifests');
        const data = await res.json();
        if (data.is_calibration) {
          manifestBadge.textContent = '🎯 Calibration Mode (10 synthetic clips)';
        } else {
          manifestBadge.textContent = `${data.current_name} (${data.total_samples.toLocaleString()} samples)`;
        }
        totalSamplesLabel.textContent = `/ ${(data.total_samples - 1).toLocaleString()}`;
        sampleIndexInput.max = data.total_samples - 1;

        manifestSelect.innerHTML = '<option value="">(Switch manifest...)</option>';
        data.available.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.path;
          opt.textContent = m.name;
          if (m.path === data.current_manifest) {
            opt.selected = true;
          }
          manifestSelect.appendChild(opt);
        });
      } catch (err) {
        console.error('Failed to load manifests list:', err);
      }
    }

    // Load sample by index
    async function loadSample(index) {
      stopAudio();
      audioBuffer = null;
      loadingSpinner.classList.remove('hidden');
      try {
        const res = await fetch(`/api/sample/${index}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        currentSample = data;
        loadSavedAlignments();
        renderSampleMetadata(data);
        renderTranscript(data);
        renderWordsTable(data);
        sampleIndexInput.value = data.index;
        
        // Fetch and decode audio
        await fetchAndDecodeAudio(data.index);
      } catch (err) {
        alert('Failed to load sample: ' + err.message);
      } finally {
        loadingSpinner.classList.add('hidden');
      }
    }

    // Load random sample
    async function loadRandomSample() {
      stopAudio();
      audioBuffer = null;
      loadingSpinner.classList.remove('hidden');
      try {
        const res = await fetch('/api/sample/random/get');
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        currentSample = data;
        loadSavedAlignments();
        renderSampleMetadata(data);
        renderTranscript(data);
        renderWordsTable(data);
        sampleIndexInput.value = data.index;
        await fetchAndDecodeAudio(data.index);
      } catch (err) {
        alert('Failed to load random sample: ' + err.message);
      } finally {
        loadingSpinner.classList.add('hidden');
      }
    }

    // Render metadata tags
    function renderSampleMetadata(data) {
      speakerBadge.textContent = data.speaker || 'unknown';
      audioFilenameBadge.textContent = data.audio_filename;
      audioFilenameBadge.title = data.audio_path;
      startOffsetBadge.textContent = `${data.start.toFixed(2)}s (${formatTime(data.start)})`;
      sampleDurBadge.textContent = `${data.duration.toFixed(2)}s`;
      paddingBadge.textContent = `-${data.pre_padding.toFixed(2)}s / +${data.post_padding.toFixed(2)}s`;
      wordsCountBadge.textContent = data.words.length;
      totalTimeLabel.textContent = `${data.total_audio_duration.toFixed(2)}s`;
      currentTimeLabel.textContent = '00.00s';
      sampleRelTimeLabel.textContent = '+00.00s';
    }

    // Render interactive transcript
    function renderTranscript(data) {
      transcriptContainer.innerHTML = '';
      if (data.is_calibration) {
        transcriptContainer.setAttribute('dir', 'ltr');
      } else {
        transcriptContainer.setAttribute('dir', 'rtl');
      }

      if (!data.words || data.words.length === 0) {
        transcriptContainer.textContent = data.transcript || '(Empty transcript)';
        return;
      }

      renderWordTokens(transcriptContainer, data.words, 'original');
    }

    function renderWordTokens(container, words, track) {
      if (!words || words.length === 0) {
        container.textContent = '(No words aligned)';
        return;
      }

      words.forEach((w, idx) => {
        const token = document.createElement('span');
        token.className = 'word-token px-2 py-1 rounded-md bg-slate-800/80 text-slate-200 border border-slate-700/60 inline-flex items-center gap-1';
        token.dataset.index = idx;
        token.dataset.track = track;
        const text = document.createElement('span');
        text.className = 'font-semibold';
        text.textContent = w.word;
        const timing = document.createElement('span');
        timing.className = 'text-[10px] text-slate-400 font-mono';
        timing.textContent = `${w.start.toFixed(3)}–${w.end.toFixed(3)}s • ${(w.duration * 1000).toFixed(0)}ms`;
        token.append(text, timing);
        token.addEventListener('dblclick', () => zoomToWord(w));

        token.addEventListener('click', (e) => {
          e.stopPropagation();
          track === 'alternative' ? playAlternativeWord(idx) : playWord(idx);
        });

        token.addEventListener('mouseenter', () => {
          hoveredWordIndex = idx;
          highlightWord(idx, true);
          drawWaveform();
        });

        token.addEventListener('mouseleave', () => {
          hoveredWordIndex = null;
          highlightWord(idx, false);
          drawWaveform();
        });

        container.appendChild(token);
      });
    }

    function loadSavedAlignments() {
      currentSample.words = currentSample.original_words || currentSample.words;
      document.getElementById('original-track-label').textContent =
        `Original timings • file offset ${(currentSample.original_start ?? currentSample.start).toFixed(3)}s`;
      const backend = currentSample.alignment_backend;
      const hasSaved = Array.isArray(currentSample.alternative_words);
      const label = !hasSaved ? 'Saved realignment' :
        backend === 'original' ? 'Retained original (realignment failed)' :
        backend === 'unknown' ? 'Saved realignment (backend unspecified)' : `Saved ${backend.toUpperCase()}`;
      alternativeTrackLabel.textContent = label;
      alternativeTimingTitle.textContent = `${label} vs Original Word Timings`;
      alignmentSelect.replaceChildren(new Option('Original', 'original'));
      if (hasSaved) {
        alignmentSelect.add(new Option(label, 'alternative'));
        alignmentSelect.value = selectedAlignment;
        renderAlternativeAlignment(currentSample.alternative_words);
        alignmentStatus.textContent = currentSample.alignment_fallback_reason || '';
        alignmentStatus.title = currentSample.alignment_error || '';
      } else {
        alignmentSelect.value = 'original';
        alternativeTimingCard.classList.add('hidden');
        alternativeTranscriptContainer.textContent = 'No saved realignment in this manifest.';
        alternativeWordsTableBody.replaceChildren();
        alignmentStatus.textContent = '';
        alignmentStatus.title = '';
      }
      hoveredWordIndex = null;
    }

    function getViewerWords() {
      if (!currentSample) return [];
      return alignmentSelect.value === 'alternative' ?
        (currentSample.alternative_words || []) : currentSample.words;
    }

    function zoomToWord(word) {
      stopAudio();
      const span = Math.max(0.2, word.duration + 0.4);
      setZoom(currentSample.total_audio_duration / span,
        (word.audio_start + word.audio_end) / 2, canvasWrapper.clientWidth / 2);
    }

    function renderAlternativeAlignment(words) {
      alternativeTranscriptContainer.innerHTML = '';
      alternativeTranscriptContainer.setAttribute('dir', currentSample.is_calibration ? 'ltr' : 'rtl');
      renderWordTokens(alternativeTranscriptContainer, words, 'alternative');
      renderAlternativeWordsTable(words);
      alternativeTimingCard.classList.remove('hidden');
    }

    // Render table
    function renderWordsTable(data) {
      tableWordsCount.textContent = `${data.words.length} words`;
      wordsTableBody.innerHTML = '';

      const wordDir = data.is_calibration ? 'ltr' : 'rtl';
      const alignClass = data.is_calibration ? 'text-left' : 'text-right';

      data.words.forEach((w, idx) => {
        const tr = document.createElement('tr');
        tr.id = `word-row-${idx}`;
        tr.className = 'word-row hover:bg-slate-800/60 transition cursor-pointer';

        tr.innerHTML = `
          <td class="py-2 px-3 text-center text-slate-400 text-[11px]">${idx + 1}</td>
          <td class="py-2 px-3 ${alignClass} font-sans font-bold text-slate-200 text-sm" dir="${wordDir}">${w.word}</td>
          <td class="py-2 px-3 text-center text-slate-300 font-mono">${w.start.toFixed(3)}s</td>
          <td class="py-2 px-3 text-center text-slate-300 font-mono">${w.end.toFixed(3)}s</td>
          <td class="py-2 px-3 text-center text-indigo-400 font-bold font-mono">${(w.duration * 1000).toFixed(0)} ms</td>
          <td class="py-2 px-3 text-center text-slate-400 font-mono">${w.audio_start.toFixed(2)}s - ${w.audio_end.toFixed(2)}s</td>
          <td class="py-2 px-3 text-center">
            <button class="play-word-btn px-2 py-0.5 rounded bg-indigo-600/80 hover:bg-indigo-500 text-white text-[11px] font-mono transition" data-index="${idx}">
              ▶ Play
            </button>
          </td>
        `;

        tr.addEventListener('click', () => playWord(idx));
        tr.addEventListener('mouseenter', () => {
          hoveredWordIndex = idx;
          highlightWord(idx, true);
          drawWaveform();
        });
        tr.addEventListener('mouseleave', () => {
          hoveredWordIndex = null;
          highlightWord(idx, false);
          drawWaveform();
        });

        wordsTableBody.appendChild(tr);
      });
    }

    function renderAlternativeWordsTable(words) {
      alternativeTableWordsCount.textContent = `${words.length} words`;
      alternativeWordsTableBody.innerHTML = '';
      const wordDir = currentSample.is_calibration ? 'ltr' : 'rtl';
      const alignClass = currentSample.is_calibration ? 'text-left' : 'text-right';

      const header = alternativeTimingCard.querySelector('thead tr');
      header.innerHTML = ['#', 'Word', 'Original start', 'Original end', 'Original duration',
        'Saved start', 'Saved end', 'Saved duration', 'Δ start', 'Δ end', 'Δ duration', 'Play']
        .map(label => `<th class="py-2 px-3 text-center whitespace-nowrap">${label}</th>`).join('');
      const timeCell = (word, key) => word ? `${word[key].toFixed(3)}s` : '—';
      const deltaCell = (original, saved, key) => {
        if (!original || !saved) return '—';
        const ms = Math.round((saved[key] - original[key]) * 1000);
        return `${ms > 0 ? '+' : ''}${ms} ms`;
      };
      (currentSample.word_comparisons || []).forEach((pair, rowIndex) => {
        const original = pair.original_index === null ? null : currentSample.words[pair.original_index];
        const saved = pair.alternative_index === null ? null : words[pair.alternative_index];
        const tr = document.createElement('tr');
        if (saved) tr.id = `alternative-word-row-${pair.alternative_index}`;
        tr.className = 'word-row hover:bg-fuchsia-950/40 transition cursor-pointer';
        tr.innerHTML = `
          <td class="py-2 px-3 text-center text-slate-400">${rowIndex + 1}</td>
          <td class="py-2 px-3 ${alignClass} font-sans font-bold text-sm" dir="${wordDir}"></td>
          ${['start', 'end', 'duration'].map(key => `<td class="py-2 px-3 text-center whitespace-nowrap">${timeCell(original, key)}</td>`).join('')}
          ${['start', 'end', 'duration'].map(key => `<td class="py-2 px-3 text-center text-fuchsia-300 whitespace-nowrap">${timeCell(saved, key)}</td>`).join('')}
          ${['start', 'end', 'duration'].map(key => `<td class="py-2 px-3 text-center text-amber-300 whitespace-nowrap">${deltaCell(original, saved, key)}</td>`).join('')}
          <td class="py-2 px-3 text-center whitespace-nowrap"></td>
        `;
        tr.children[1].textContent = (saved || original).word +
          (!original ? ' (saved only)' : !saved ? ' (original only)' : '');
        for (const [word, label, play] of [
          [original, 'Original', () => playWord(pair.original_index)],
          [saved, 'Saved', () => playAlternativeWord(pair.alternative_index)],
        ]) {
          if (!word) continue;
          const button = document.createElement('button');
          button.className = 'px-2 py-1 rounded bg-slate-700 hover:bg-slate-600 mr-1';
          button.textContent = `▶ ${label}`;
          button.addEventListener('click', e => { e.stopPropagation(); play(); });
          tr.lastElementChild.appendChild(button);
        }
        tr.addEventListener('click', () => saved ? playAlternativeWord(pair.alternative_index) : playWord(pair.original_index));
        alternativeWordsTableBody.appendChild(tr);
      });
    }

    // Audio fetching and decoding
    async function fetchAndDecodeAudio(index) {
      const ctx = getAudioContext();
      const res = await fetch(`/api/audio/${index}?padding=${currentSample.padding}`);
      if (!res.ok) throw new Error(await res.text());
      const arrayBuffer = await res.arrayBuffer();
      audioBuffer = await ctx.decodeAudioData(arrayBuffer);
      currentSample.total_audio_duration = audioBuffer.duration;
      currentSample.post_padding = Math.max(0, audioBuffer.duration - currentSample.sample_audio_end);
      totalTimeLabel.textContent = `${audioBuffer.duration.toFixed(2)}s`;
      paddingBadge.textContent = `-${currentSample.pre_padding.toFixed(2)}s / +${currentSample.post_padding.toFixed(2)}s`;
      setZoom(1.0);
      drawWaveform();
      drawOverview();
    }

    // Zoom Controls
    function setZoom(newZoom, anchorTime = null, anchorScreenX = null) {
      if (!currentSample) return;
      if (anchorTime === null) {
        anchorScreenX = canvasWrapper.clientWidth / 2;
        anchorTime = ((canvasWrapper.scrollLeft + anchorScreenX) /
          (canvas.clientWidth || canvasWrapper.clientWidth)) * currentSample.total_audio_duration;
      }
      const clamped = Math.max(1.0, Math.min(10.0, parseFloat(newZoom) || 1.0));
      zoomLevel = clamped;
      zoomSlider.value = zoomLevel;
      zoomLabel.textContent = `${Math.round(zoomLevel * 100)}%`;

      const containerWidth = canvasWrapper.clientWidth;
      if (zoomLevel <= 1.01) {
        zoomLevel = 1.0;
        canvas.style.width = '100%';
        canvasWrapper.scrollLeft = 0;
      } else {
        const newCanvasWidth = Math.round(containerWidth * zoomLevel);
        canvas.style.width = `${newCanvasWidth}px`;
        if (anchorTime !== null && anchorScreenX !== null) {
          const anchorCanvasX = (anchorTime / currentSample.total_audio_duration) * newCanvasWidth;
          canvasWrapper.scrollLeft = anchorCanvasX - anchorScreenX;
        }
      }

      drawWaveform();
      drawOverview();
    }

    function fitSample() {
      if (!currentSample) return;
      const sampleDur = currentSample.duration;
      const totalDur = currentSample.total_audio_duration;
      if (sampleDur <= 0 || totalDur <= 0) return;

      const idealZoom = (totalDur / sampleDur) * 0.85;
      const targetZoom = Math.max(1.0, Math.min(10.0, idealZoom));
      const sampleCenterSec = currentSample.pre_padding + (sampleDur / 2);
      const containerWidth = canvasWrapper.clientWidth;
      setZoom(targetZoom, sampleCenterSec, containerWidth / 2);
    }

    // Overview / Minimap rendering
    function drawOverview(currentPlaybackSec = null) {
      if (!overviewCanvas || !currentSample) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = overviewWrapper.getBoundingClientRect();
      if (rect.width <= 0) return;

      overviewCanvas.width = rect.width * dpr;
      overviewCanvas.height = rect.height * dpr;
      const ctx = overviewCanvas.getContext('2d');
      ctx.scale(dpr, dpr);

      const width = rect.width;
      const height = rect.height;

      ctx.fillStyle = '#020617';
      ctx.fillRect(0, 0, width, height);

      const totalDur = currentSample.total_audio_duration;
      if (totalDur <= 0) return;

      const prePad = currentSample.pre_padding;
      const sampleDur = currentSample.duration;
      const sampleEnd = prePad + sampleDur;

      const xPreEnd = (prePad / totalDur) * width;
      const xSampleEnd = (sampleEnd / totalDur) * width;

      // Draw background zones
      ctx.fillStyle = 'rgba(30, 41, 59, 0.5)';
      ctx.fillRect(0, 0, xPreEnd, height);

      ctx.fillStyle = 'rgba(79, 70, 229, 0.15)';
      ctx.fillRect(xPreEnd, 0, xSampleEnd - xPreEnd, height);

      ctx.fillStyle = 'rgba(30, 41, 59, 0.5)';
      ctx.fillRect(xSampleEnd, 0, width - xSampleEnd, height);

      // Peaks in miniature
      if (audioBuffer) {
        const rawData = audioBuffer.getChannelData(0);
        const amp = height / 2;
        const totalSamples = rawData.length;

        for (let x = 0; x < width; x++) {
          const sampleIdxStart = Math.min(totalSamples - 1, Math.floor((x / width) * totalSamples));
          const sampleIdxEnd = Math.min(
            totalSamples,
            Math.max(sampleIdxStart + 1, Math.floor(((x + 1) / width) * totalSamples))
          );
          let min = 1.0;
          let max = -1.0;
          for (let j = sampleIdxStart; j < sampleIdxEnd; j++) {
            const datum = rawData[j] || 0;
            if (datum < min) min = datum;
            if (datum > max) max = datum;
          }
          if (max < min) { min = 0; max = 0; }

          const pxStart = (x / width) * totalDur;
          const pxEnd = ((x + 1) / width) * totalDur;
          const isSample = pxEnd > prePad && pxStart < sampleEnd;
          ctx.fillStyle = isSample ? '#818cf8' : '#475569';

          const yLow = (1 + min) * amp;
          const yHigh = Math.max(1, (max - min) * amp);
          ctx.fillRect(x, yLow, 1, yHigh);
        }
      }

      // Divider lines
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(xPreEnd, 0);
      ctx.lineTo(xPreEnd, height);
      ctx.stroke();

      ctx.strokeStyle = '#f43f5e';
      ctx.beginPath();
      ctx.moveTo(xSampleEnd, 0);
      ctx.lineTo(xSampleEnd, height);
      ctx.stroke();

      // Playhead in overview
      if (currentPlaybackSec !== null && currentPlaybackSec >= 0) {
        const px = (currentPlaybackSec / totalDur) * width;
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, height);
        ctx.stroke();
      }

      // Viewport lens showing visible zoom section
      const containerW = canvasWrapper.clientWidth;
      const cWidth = canvas.clientWidth || containerW;
      const scrollLeft = canvasWrapper.scrollLeft;

      const lensLeft = (scrollLeft / cWidth) * width;
      const lensWidth = Math.max(6, (containerW / cWidth) * width);

      ctx.fillStyle = 'rgba(99, 102, 241, 0.28)';
      ctx.fillRect(lensLeft, 0, lensWidth, height);

      ctx.strokeStyle = '#818cf8';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(lensLeft, 0.5, lensWidth, height - 1);
    }

    function handleOverviewSeek(e) {
      if (!currentSample) return;
      const rect = overviewWrapper.getBoundingClientRect();
      const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const targetSec = pct * currentSample.total_audio_duration;
      const cWidth = canvas.clientWidth;
      const targetX = (targetSec / currentSample.total_audio_duration) * cWidth;
      canvasWrapper.scrollLeft = targetX - (canvasWrapper.clientWidth / 2);
      drawOverview();
    }

    // Waveform rendering
    function drawWaveform(currentPlaybackSec = null) {
      if (!canvas || !currentSample) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = canvas.clientWidth || rect.width;
      const height = rect.height;
      
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      const ctx = canvas.getContext('2d');
      ctx.scale(dpr, dpr);

      // Heights breakdown
      const rulerHeight = 24;
      const wordTrackHeight = 36;
      const waveHeight = height - rulerHeight - wordTrackHeight;

      // Clear canvas
      ctx.fillStyle = '#020617';
      ctx.fillRect(0, 0, width, height);

      const totalDur = currentSample.total_audio_duration;
      if (totalDur <= 0) return;

      const prePad = currentSample.pre_padding;
      const sampleDur = currentSample.duration;
      const sampleEnd = prePad + sampleDur;

      const xPreEnd = (prePad / totalDur) * width;
      const xSampleEnd = (sampleEnd / totalDur) * width;

      // 1. Draw Background Regions
      // Pre-padding background (shaded/diagonal pattern)
      ctx.fillStyle = 'rgba(30, 41, 59, 0.45)';
      ctx.fillRect(0, 0, xPreEnd, waveHeight + wordTrackHeight);

      // Target Sample background (highlighted)
      ctx.fillStyle = 'rgba(79, 70, 229, 0.08)';
      ctx.fillRect(xPreEnd, 0, xSampleEnd - xPreEnd, waveHeight + wordTrackHeight);

      // Post-padding background (shaded)
      ctx.fillStyle = 'rgba(30, 41, 59, 0.45)';
      ctx.fillRect(xSampleEnd, 0, width - xSampleEnd, waveHeight + wordTrackHeight);

      // 2. Draw Waveform Peaks
      if (audioBuffer) {
        const rawData = audioBuffer.getChannelData(0);
        const amp = waveHeight / 2;
        const totalSamples = rawData.length;

        for (let x = 0; x < width; x++) {
          const sampleIdxStart = Math.min(totalSamples - 1, Math.floor((x / width) * totalSamples));
          const sampleIdxEnd = Math.min(
            totalSamples,
            Math.max(sampleIdxStart + 1, Math.floor(((x + 1) / width) * totalSamples))
          );
          let min = 1.0;
          let max = -1.0;
          for (let j = sampleIdxStart; j < sampleIdxEnd; j++) {
            const datum = rawData[j] || 0;
            if (datum < min) min = datum;
            if (datum > max) max = datum;
          }
          if (max < min) { min = 0; max = 0; }

          const pxStart = (x / width) * totalDur;
          const pxEnd = ((x + 1) / width) * totalDur;
          const isSample = pxEnd > prePad && pxStart < sampleEnd;

          // Color based on region
          if (isSample) {
            ctx.fillStyle = '#6366f1'; // Indigo for sample
          } else {
            ctx.fillStyle = '#475569'; // Muted slate for padding
          }

          // If inside hovered word
          if (hoveredWordIndex !== null && getViewerWords()[hoveredWordIndex]) {
            const hw = getViewerWords()[hoveredWordIndex];
            if (pxStart < hw.audio_end && pxEnd > hw.audio_start) {
              ctx.fillStyle = '#f59e0b'; // Amber for hovered word
            }
          }

          const yLow = (1 + min) * amp;
          const yHigh = Math.max(1, (max - min) * amp);
          ctx.fillRect(x, yLow, 1, yHigh);
        }
      }

      // 3. Draw Vertical Boundary Lines for Padding vs Sample
      // Sample Start Line (Green)
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(xPreEnd, 0);
      ctx.lineTo(xPreEnd, waveHeight + wordTrackHeight);
      ctx.stroke();

      // Sample End Line (Amber/Red)
      ctx.strokeStyle = '#f43f5e';
      ctx.beginPath();
      ctx.moveTo(xSampleEnd, 0);
      ctx.lineTo(xSampleEnd, waveHeight + wordTrackHeight);
      ctx.stroke();
      ctx.setLineDash([]);

      // Region Top Badges
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(`◀ PRE-PAD (${prePad.toFixed(1)}s)`, 6, 14);

      ctx.fillStyle = '#818cf8';
      ctx.fillText(`★ SAMPLE AUDIO (${sampleDur.toFixed(2)}s) ★`, xPreEnd + 8, 14);

      ctx.fillStyle = '#94a3b8';
      ctx.fillText(`POST-PAD ▶`, xSampleEnd + 8, 14);

      // 4. Draw Word Blocks in Word Track
      const trackY = waveHeight;
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, trackY, width, wordTrackHeight);
      ctx.strokeStyle = '#1e293b';
      ctx.beginPath();
      ctx.moveTo(0, trackY);
      ctx.lineTo(width, trackY);
      ctx.stroke();

       if (currentSample.words) {
         getViewerWords().forEach((w, idx) => {
          const wStartSec = w.audio_start;
          const wEndSec = w.audio_end;
          const wx = (wStartSec / totalDur) * width;
          const ww = Math.max(2, ((wEndSec - wStartSec) / totalDur) * width);

          const isHovered = hoveredWordIndex === idx;
          const isPlayingWord = (
            currentPlaybackSec !== null &&
            currentPlaybackSec >= w.audio_start &&
            currentPlaybackSec <= w.audio_end
          );

          if (isPlayingWord) {
            ctx.fillStyle = 'rgba(16, 185, 129, 0.4)';
          } else if (isHovered) {
            ctx.fillStyle = 'rgba(245, 158, 11, 0.35)';
          } else {
            ctx.fillStyle = idx % 2 === 0 ? 'rgba(79, 70, 229, 0.18)' : 'rgba(99, 102, 241, 0.12)';
          }

          ctx.fillRect(wx, trackY + 2, ww - 1, wordTrackHeight - 4);

          ctx.strokeStyle = isPlayingWord ? '#10b981' : (isHovered ? '#f59e0b' : '#3730a3');
          ctx.strokeRect(wx, trackY + 2, ww - 1, wordTrackHeight - 4);

          // Word text
          if (ww > 14) {
            ctx.font = '11px Assistant, sans-serif';
            ctx.fillStyle = isPlayingWord ? '#34d399' : (isHovered ? '#fbbf24' : '#e2e8f0');
            ctx.textAlign = 'center';
            ctx.fillText(w.word, wx + ww / 2, trackY + 16);

            // Duration
            if (ww > 32) {
              ctx.font = '9px JetBrains Mono, monospace';
              ctx.fillStyle = '#94a3b8';
              ctx.fillText(`${(w.duration * 1000).toFixed(0)}ms`, wx + ww / 2, trackY + 28);
            }
           }
         });
       }

       // 5. Timeline Ruler
      const rulerY = trackY + wordTrackHeight;
      ctx.fillStyle = '#020617';
      ctx.fillRect(0, rulerY, width, rulerHeight);
      ctx.strokeStyle = '#334155';
      ctx.beginPath();
      ctx.moveTo(0, rulerY);
      ctx.lineTo(width, rulerY);
      ctx.stroke();

      let tickStep = 1.0;
      let labelStep = 2.0;
      if (zoomLevel >= 4.0) {
        tickStep = 0.1;
        labelStep = 0.5;
      } else if (zoomLevel >= 2.0) {
        tickStep = 0.5;
        labelStep = 1.0;
      }

      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillStyle = '#64748b';
      ctx.textAlign = 'center';

      for (let t = 0; t <= totalDur + 0.001; t += tickStep) {
        const tx = (t / totalDur) * width;
        const isMajor = Math.abs(t % labelStep) < 0.005 || Math.abs(t % labelStep - labelStep) < 0.005;
        ctx.strokeStyle = isMajor ? '#475569' : '#1e293b';
        ctx.beginPath();
        ctx.moveTo(tx, rulerY);
        ctx.lineTo(tx, rulerY + (isMajor ? 7 : 3));
        ctx.stroke();

        if (isMajor) {
          ctx.fillText(`${t.toFixed(zoomLevel >= 4.0 ? 1 : 0)}s`, tx, rulerY + 16);
        }
      }

      // 6. Playhead Cursor
      if (currentPlaybackSec !== null && currentPlaybackSec >= 0) {
        const px = (currentPlaybackSec / totalDur) * width;
        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, height);
        ctx.stroke();

        // Cursor knob
        ctx.fillStyle = '#f43f5e';
        ctx.beginPath();
        ctx.moveTo(px - 5, 0);
        ctx.lineTo(px + 5, 0);
        ctx.lineTo(px, 8);
        ctx.closePath();
        ctx.fill();
      }
    }

    // Playback Engine
    function stopAudio() {
      if (currentSource) {
        try {
          currentSource.stop();
          currentSource.disconnect();
        } catch (e) {}
        currentSource = null;
      }
      isPlaying = false;
      playbackMode = null;
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
      }
      updatePlayIcons(false);
      clearActiveHighlights();
      drawWaveform(null);
      drawOverview(null);
    }

    function playRange(startSec, endSec, mode) {
      if (!audioBuffer) return;
      startSec = Math.max(0, Math.min(audioBuffer.duration, startSec));
      endSec = Math.max(0, Math.min(audioBuffer.duration, endSec));
      if (endSec <= startSec) return;
      stopAudio();

      const ctx = getAudioContext();
      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.playbackRate.value = parseFloat(playbackRateSelect.value) || 1.0;

      const gainNode = ctx.createGain();
      gainNode.gain.value = parseFloat(volumeSlider.value) || 1.0;

      source.connect(gainNode);
      gainNode.connect(ctx.destination);

      currentSource = source;
      isPlaying = true;
      playbackMode = mode;
      playStartTime = ctx.currentTime;
      playStartOffset = startSec;
      playEndTime = endSec;

      const duration = Math.max(0.01, endSec - startSec);
      source.start(0, startSec, duration);
      updatePlayIcons(true, mode);

      source.onended = () => {
        if (isPlaying) {
          if (loopMode && playbackMode === 'sample') {
            playRange(currentSample.sample_audio_start, currentSample.sample_audio_end, 'sample');
          } else {
            stopAudio();
          }
        }
      };

      animatePlayhead();
    }

    // Play entire sample (target audio without padding)
    function playSampleOnly() {
      if (!currentSample) return;
      if (isPlaying && playbackMode === 'sample') {
        stopAudio();
        return;
      }
      playRange(currentSample.sample_audio_start, currentSample.sample_audio_end, 'sample');
    }

    // Play full audio (including ±3s padding)
    function playFullAudio() {
      if (!currentSample) return;
      if (isPlaying && playbackMode === 'full') {
        stopAudio();
        return;
      }
      playRange(0, currentSample.total_audio_duration, 'full');
    }

    // Play single word
     function playWord(wordIndex) {
      if (!currentSample || !currentSample.words[wordIndex]) return;
      const w = currentSample.words[wordIndex];
      const start = w.audio_start;
      const end = w.audio_end;
      if (end <= start) return;
      playRange(start, end, 'word');
       highlightWord(wordIndex, true);
     }

     function playLastWordWithTail() {
       const words = getViewerWords();
       if (!words.length) return;
       const index = words.length - 1;
       const word = words[index];
       const end = Math.min(currentSample.total_audio_duration, word.audio_end + 0.15);
       playRange(word.audio_start, end, 'word');
       highlightWord(index, true);
     }

    function playAlternativeWord(wordIndex) {
      if (!currentSample || !currentSample.alternative_words || !currentSample.alternative_words[wordIndex]) return;
      const w = currentSample.alternative_words[wordIndex];
      if (w.audio_end <= w.audio_start) return;
      playRange(w.audio_start, w.audio_end, 'word');
      highlightWord(wordIndex, true);
    }

    // Smooth animation loop during playback
    function animatePlayhead() {
      if (!isPlaying || !audioContext) return;

      const ctx = audioContext;
      const rate = parseFloat(playbackRateSelect.value) || 1.0;
      const elapsed = (ctx.currentTime - playStartTime) * rate;
      const currentSec = playStartOffset + elapsed;

      currentTimeLabel.textContent = `${currentSec.toFixed(2)}s`;
      
      const relSec = currentSec - currentSample.sample_audio_start;
      sampleRelTimeLabel.textContent = (relSec >= 0 ? `+${relSec.toFixed(2)}s` : `-${Math.abs(relSec).toFixed(2)}s`);

      // Update active word highlight
      updateActiveWord(currentSec);

      // Redraw canvas
      drawWaveform(currentSec);
      drawOverview(currentSec);

      // Auto-follow playhead when zoomed
      if (autoFollowPlayhead && zoomLevel > 1.0) {
        const playheadCanvasX = (currentSec / currentSample.total_audio_duration) * canvas.clientWidth;
        const scrollLeft = canvasWrapper.scrollLeft;
        const viewW = canvasWrapper.clientWidth;
        if (playheadCanvasX < scrollLeft + 30 || playheadCanvasX > scrollLeft + viewW - 30) {
          canvasWrapper.scrollLeft = playheadCanvasX - (viewW / 2);
        }
      }

      if (currentSec <= playEndTime) {
        animationFrameId = requestAnimationFrame(animatePlayhead);
      }
    }

    // Word highlighting synchronization
    function updateActiveWord(currentSec) {
      if (!currentSample || !currentSample.words) return;
      currentSample.words.forEach((w, idx) => {
        const isActive = currentSec >= w.audio_start && currentSec <= w.audio_end;
        const token = document.querySelector(`.word-token[data-track="original"][data-index="${idx}"]`);
        const row = document.getElementById(`word-row-${idx}`);

        if (token) token.classList.toggle('active', isActive);
        if (row) row.classList.toggle('active', isActive);
      });
      if (currentSample.alternative_words) {
        currentSample.alternative_words.forEach((w, idx) => {
          const isActive = currentSec >= w.audio_start && currentSec <= w.audio_end;
          const token = document.querySelector(`.word-token[data-track="alternative"][data-index="${idx}"]`);
          const row = document.getElementById(`alternative-word-row-${idx}`);
          if (token) token.classList.toggle('active', isActive);
          if (row) row.classList.toggle('active', isActive);
        });
      }
    }

    function highlightWord(idx, enable) {
      const tokens = document.querySelectorAll(`.word-token[data-index="${idx}"]`);
      const row = document.getElementById(`word-row-${idx}`);
      const alternativeRow = document.getElementById(`alternative-word-row-${idx}`);
      tokens.forEach(token => token.classList.toggle('highlighted', enable));
      if (row) row.classList.toggle('highlighted', enable);
      if (alternativeRow) alternativeRow.classList.toggle('highlighted', enable);
    }

    function clearActiveHighlights() {
      document.querySelectorAll('.word-token.active, .word-token.highlighted').forEach(el => {
        el.classList.remove('active', 'highlighted');
      });
      document.querySelectorAll('.word-row.active, .word-row.highlighted').forEach(el => {
        el.classList.remove('active', 'highlighted');
      });
    }

    function updatePlayIcons(playing, mode) {
      playSampleIcon.textContent = (playing && mode === 'sample') ? '⏸' : '▶';
      playFullIcon.textContent = (playing && mode === 'full') ? '⏸' : '▶';
    }

    // Seek via canvas click
    function handleCanvasClick(e) {
      if (!currentSample || !audioBuffer || isPanningCanvas) return;
      const rect = canvas.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const pct = Math.max(0, Math.min(1, clickX / canvas.clientWidth));
      const seekSec = pct * currentSample.total_audio_duration;
      playRange(seekSec, currentSample.total_audio_duration, 'full');
    }

    // Tooltip on canvas hover
    function handleCanvasMouseMove(e) {
      if (!currentSample) return;
      const rect = canvas.getBoundingClientRect();
      const wrapRect = canvasWrapper.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const pct = Math.max(0, Math.min(1, clickX / canvas.clientWidth));
      const timeSec = pct * currentSample.total_audio_duration;

      const prePad = currentSample.pre_padding;
      const sampleEnd = prePad + currentSample.duration;

      let label = `${timeSec.toFixed(2)}s`;
      if (timeSec < prePad) {
        label += ` (Pre-Padding -${(prePad - timeSec).toFixed(2)}s)`;
      } else if (timeSec > sampleEnd) {
        label += ` (Post-Padding +${(timeSec - sampleEnd).toFixed(2)}s)`;
      } else {
        label += ` (Sample +${(timeSec - prePad).toFixed(2)}s)`;
      }

      // Check if inside a word
      if (currentSample.words) {
        const found = getViewerWords().find(w => timeSec >= w.audio_start && timeSec <= w.audio_end);
        if (found) {
          label += ` • "${found.word}" (${(found.duration * 1000).toFixed(0)}ms)`;
        }
      }

      const mouseX = e.clientX - wrapRect.left;
      const mouseY = e.clientY - wrapRect.top;
      canvasTooltip.textContent = label;
      canvasTooltip.style.left = `${Math.min(canvasWrapper.clientWidth - 180, Math.max(10, mouseX - 60))}px`;
      canvasTooltip.style.top = `${Math.max(10, mouseY - 30)}px`;
      canvasTooltip.classList.remove('hidden');
    }

    function handleCanvasMouseLeave() {
      canvasTooltip.classList.add('hidden');
    }

    // Setup event listeners
    function setupEvents() {
      prevBtn.addEventListener('click', () => {
        if (currentSample && currentSample.index > 0) {
          loadSample(currentSample.index - 1);
        }
      });

      nextBtn.addEventListener('click', () => {
        if (currentSample && currentSample.index < currentSample.total_samples - 1) {
          loadSample(currentSample.index + 1);
        }
      });

      randomBtn.addEventListener('click', loadRandomSample);

      goBtn.addEventListener('click', () => {
        const idx = parseInt(sampleIndexInput.value, 10);
        if (!isNaN(idx)) loadSample(idx);
      });

      sampleIndexInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          const idx = parseInt(sampleIndexInput.value, 10);
          if (!isNaN(idx)) loadSample(idx);
        }
      });

      manifestSelect.addEventListener('change', async () => {
        const path = manifestSelect.value;
        if (!path) return;
        try {
          const res = await fetch('/api/set_manifest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
          });
          if (!res.ok) throw new Error(await res.text());
          await loadManifests();
          await loadSample(0);
        } catch (err) {
          alert('Could not switch manifest: ' + err.message);
        }
      });

      customManifestBtn.addEventListener('click', async () => {
        const p = prompt('Enter absolute path to manifest JSONL:', currentSample ? currentSample.manifest_path : '');
        if (!p) return;
        try {
          const res = await fetch('/api/set_manifest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: p }),
          });
          if (!res.ok) throw new Error(await res.text());
          await loadManifests();
          await loadSample(0);
        } catch (err) {
          alert('Error loading manifest: ' + err.message);
        }
      });

      calibrateBtn.addEventListener('click', async () => {
        try {
          const res = await fetch('/api/set_manifest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: 'calibrate' }),
          });
          if (!res.ok) throw new Error(await res.text());
          await loadManifests();
          await loadSample(0);
        } catch (err) {
          alert('Error loading calibration mode: ' + err.message);
        }
      });

       playSampleBtn.addEventListener('click', playSampleOnly);
       lastWordBtn.addEventListener('click', playLastWordWithTail);
       playFullBtn.addEventListener('click', playFullAudio);
      stopBtn.addEventListener('click', stopAudio);

      loopBtn.addEventListener('click', () => {
        loopMode = !loopMode;
        loopBtn.classList.toggle('text-indigo-400', loopMode);
        loopBtn.classList.toggle('border-indigo-500', loopMode);
      });

      copyTranscriptBtn.addEventListener('click', () => {
        if (currentSample && currentSample.transcript) {
          navigator.clipboard.writeText(currentSample.transcript);
          copyTranscriptBtn.textContent = '✓ Copied';
          setTimeout(() => copyTranscriptBtn.textContent = '📋 Copy', 1500);
        }
      });

      alignmentSelect.addEventListener('change', () => {
        stopAudio();
        selectedAlignment = alignmentSelect.value;
        hoveredWordIndex = null;
        drawWaveform();
      });

      // Zoom Controls Events
      zoomSlider.addEventListener('input', (e) => setZoom(e.target.value));
      zoomInBtn.addEventListener('click', () => setZoom(zoomLevel * 1.3));
      zoomOutBtn.addEventListener('click', () => setZoom(zoomLevel / 1.3));
      zoomResetBtn.addEventListener('click', () => setZoom(1.0));
      fitSampleBtn.addEventListener('click', fitSample);
      autoFollowCheckbox.addEventListener('change', (e) => { autoFollowPlayhead = e.target.checked; });

      zoomPresetBtns.forEach(btn => {
        btn.addEventListener('click', () => setZoom(parseFloat(btn.dataset.zoom)));
      });

      // Overview Drag / Seek Events
      overviewWrapper.addEventListener('mousedown', (e) => {
        isDraggingOverview = true;
        handleOverviewSeek(e);
      });
      window.addEventListener('mousemove', (e) => {
        if (isDraggingOverview) handleOverviewSeek(e);
      });
      window.addEventListener('mouseup', () => {
        isDraggingOverview = false;
      });

      // Waveform Canvas Events
      canvasWrapper.addEventListener('scroll', () => drawOverview());

      canvasWrapper.addEventListener('wheel', (e) => {
        if (!currentSample) return;
        if (e.ctrlKey || e.altKey || e.metaKey) {
          e.preventDefault();
          const wrapRect = canvasWrapper.getBoundingClientRect();
          const anchorScreenX = e.clientX - wrapRect.left;
          const cWidth = canvas.clientWidth;
          const anchorTime = ((canvasWrapper.scrollLeft + anchorScreenX) / cWidth) * currentSample.total_audio_duration;
          const factor = e.deltaY < 0 ? 1.25 : 0.8;
          setZoom(zoomLevel * factor, anchorTime, anchorScreenX);
        }
      }, { passive: false });

      canvasWrapper.addEventListener('mousedown', (e) => {
        if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
          isPanningCanvas = true;
          panStartX = e.clientX;
          panStartScrollLeft = canvasWrapper.scrollLeft;
          canvasWrapper.style.cursor = 'grabbing';
          e.preventDefault();
        }
      });
      window.addEventListener('mousemove', (e) => {
        if (isPanningCanvas) {
          const dx = e.clientX - panStartX;
          canvasWrapper.scrollLeft = panStartScrollLeft - dx;
        }
      });
      window.addEventListener('mouseup', () => {
        if (isPanningCanvas) {
          isPanningCanvas = false;
          canvasWrapper.style.cursor = '';
        }
      });

      canvasWrapper.addEventListener('click', e => {
        if (!e.shiftKey) handleCanvasClick(e);
      });
      canvasWrapper.addEventListener('mousemove', handleCanvasMouseMove);
      canvasWrapper.addEventListener('mouseleave', handleCanvasMouseLeave);
      window.addEventListener('resize', () => { setZoom(zoomLevel); });
    }

    // Keyboard navigation
    function setupKeyboard() {
      window.addEventListener('keydown', (e) => {
        // Ignore if user is typing in an input
        if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
          return;
        }

        if (e.code === 'Space') {
          e.preventDefault();
          playFullAudio();
        } else if (e.key === 'p' || e.key === 'P') {
          e.preventDefault();
          playSampleOnly();
        } else if (e.key === 'Escape') {
          e.preventDefault();
          stopAudio();
        } else if (e.key === 'r' || e.key === 'R') {
          e.preventDefault();
          loadRandomSample();
        } else if (e.key === 'ArrowLeft' || e.key === 'k') {
          e.preventDefault();
          if (currentSample && currentSample.index > 0) {
            loadSample(currentSample.index - 1);
          }
        } else if (e.key === 'ArrowRight' || e.key === 'j') {
          e.preventDefault();
          if (currentSample && currentSample.index < currentSample.total_samples - 1) {
            loadSample(currentSample.index + 1);
          }
        } else if (e.key === 'l' || e.key === 'L') {
          e.preventDefault();
          loopMode = !loopMode;
          loopBtn.classList.toggle('text-indigo-400', loopMode);
          loopBtn.classList.toggle('border-indigo-500', loopMode);
        } else if (e.key === '+' || e.key === '=') {
          e.preventDefault();
          setZoom(zoomLevel * 1.25);
        } else if (e.key === '-' || e.key === '_') {
          e.preventDefault();
          setZoom(zoomLevel / 1.25);
        } else if (e.key === '0') {
          e.preventDefault();
          setZoom(1.0);
        } else if (e.key === 'f' || e.key === 'F') {
          e.preventDefault();
          fitSample();
        }
      });
    }

    // Start everything
    window.addEventListener('DOMContentLoaded', init);
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> str:
    """Serve single-page HTML/JS application."""
    return HTML_UI


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pocket TTS Hebrew Dataset Manifest Viewer Server")
    parser.add_argument(
        "--manifest",
        "-m",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to jsonl manifest (default: {DEFAULT_MANIFEST_PATH})",
    )
    parser.add_argument(
        "--audio-dir",
        "-a",
        type=Path,
        default=None,
        help="Optional override directory holding WAV audio files",
    )
    parser.add_argument(
        "--padding",
        "-p",
        type=float,
        default=3.0,
        help="Padding in seconds around target audio (default: 3.0)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind server on (default: 0.0.0.0)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--vad-python",
        type=Path,
        help=argparse.SUPPRESS,  # Accepted for compatibility with old viewer commands; unused.
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global state

    # If default train_aligned does not exist, try valid_aligned
    manifest_path = args.manifest
    if not manifest_path.exists() and DEFAULT_VALID_MANIFEST_PATH.exists():
        logger.warning(
            "Specified manifest %s does not exist; falling back to %s",
            manifest_path,
            DEFAULT_VALID_MANIFEST_PATH,
        )
        manifest_path = DEFAULT_VALID_MANIFEST_PATH

    state = ServerState(default_manifest=manifest_path, audio_dir=args.audio_dir)
    logger.info(
        "Serving %s (%d samples) on http://%s:%d",
        manifest_path,
        len(state.current_index),
        args.host,
        args.port,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
