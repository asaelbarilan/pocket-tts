from __future__ import annotations

import io
import json
import queue
import shutil
import sys
import wave
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from hebrew_training import align_official_manifest as aligner


def sample(word="שלום"):
    return {
        "path": "recording.wav",
        "start": 3.0,
        "duration": 2.0,
        "transcript": word,
        "speaker": "speaker-a",
        "words": [{"word": word, "start": 0.1, "end": 1.9}],
    }


def options():
    return Namespace(decode_workers=1, beam=100, retry_beam=800)


@pytest.mark.parametrize("digit", ["2026", "ב12", "١٢", "１２"])
def test_digits_bypass_viter_and_keep_original_on_mms_failure(monkeypatch, digit):
    row = sample(digit)
    before = deepcopy(row)
    viter = Mock()
    monkeypatch.setattr(aligner, "_VITER", viter)
    fallback = Mock()
    fallback.align.return_value = [{"error": "no CTC path"}]
    outcomes = aligner.align_viter_batch([(7, row)], options(), fallback)
    viter.align_many.assert_not_called()
    fallback.align.assert_called_once_with([(7, row)])
    result = outcomes[0]["row"]
    for key in ("start", "duration", "words", "transcript"):
        assert result[key] == before[key]
    assert result["original_words"] == before["words"]
    assert result["alignment_backend"] == "original"
    assert row == before
    output, failed = io.BytesIO(), io.BytesIO()
    assert aligner.write_batch(output, failed, outcomes, [(7, row)]) == 1
    assert json.loads(output.getvalue())["alignment_backend"] == "original"
    assert json.loads(failed.getvalue())["input_line"] == 7


def test_mixed_batch_preserves_order_labels_and_speaker_keys(monkeypatch):
    rows = [(1, sample("שלום,")), (2, sample("2026")), (3, sample("עולם"))]
    rows[2][1]["speaker"] = "speaker-b"
    monkeypatch.setattr(aligner, "_VITER_WORDS", {"שלום", "עולם"})
    monkeypatch.setattr(aligner, "decode_audio", lambda _: ([0.0] * 64000, 2.5))
    viter = Mock()
    viter.align_many.return_value = [
        SimpleNamespace(words=[SimpleNamespace(label="שלום", start=0.2, end=1.7)]),
        SimpleNamespace(words=[SimpleNamespace(label="עולם", start=0.3, end=1.8)]),
    ]
    monkeypatch.setattr(aligner, "_VITER", viter)
    fallback = Mock()
    reply = aligner.original_snapshot(rows[1][1])
    reply["alignment_backend"] = "mms"
    fallback.align.return_value = [{"row": reply}]
    outcomes = aligner.align_viter_batch(rows, options(), fallback)
    assert [item["row"]["alignment_backend"] for item in outcomes] == ["viter", "mms", "viter"]
    assert outcomes[0]["row"]["words"][0]["word"] == "שלום,"
    assert outcomes[0]["row"]["start"] == 2.7
    assert outcomes[0]["row"]["words"][0]["start"] == 0
    assert viter.align_many.call_args.kwargs["speakers"] == ["speaker-a", "speaker-b"]
    fallback.align.assert_called_once_with([rows[1]])


def test_failed_viter_alignment_routes_to_mms(monkeypatch):
    row = sample()
    monkeypatch.setattr(aligner, "_VITER_WORDS", {"שלום"})
    monkeypatch.setattr(aligner, "decode_audio", lambda _: ([0.0] * 64000, 2.5))
    viter = Mock()
    viter.align_many.return_value = [None]
    monkeypatch.setattr(aligner, "_VITER", viter)
    fallback = Mock()
    fallback.align.side_effect = RuntimeError("worker died")
    result = aligner.align_viter_batch([(1, row)], options(), fallback)[0]["row"]
    assert result["alignment_backend"] == "original"
    assert "worker died" in result["alignment_error"]
    assert result["words"] == row["words"]


def test_mms_rejects_unspellable_words_instead_of_dropping_them(monkeypatch):
    romanizer = Mock()
    romanizer.romanize_string.side_effect = ["shalom", "123"]
    monkeypatch.setattr(aligner, "_ROMANIZER", romanizer)
    monkeypatch.setattr(aligner, "_VOCAB", {letter: i for i, letter in enumerate("shalom")})
    row = sample("שלום 123")
    row["words"] = []
    with pytest.raises(ValueError, match="refusing to drop words"):
        aligner.word_targets(row)


def test_dictionary_extension_is_grapheme_only_and_source_is_read_only(tmp_path):
    dictionary = tmp_path / "hebrew_graphemes.dict"
    dictionary.write_text("שלום\tש\tל\tו\tמ\n", encoding="utf-8")
    source = dictionary.read_bytes()
    manifest = tmp_path / "valid.jsonl"
    manifest.write_text(json.dumps(sample("ג'ץ"), ensure_ascii=False) + "\n", encoding="utf-8")
    args = Namespace(
        viter_dictionary=dictionary, splits=["valid"], folder=tmp_path, input_stage="auto"
    )
    merged = tmp_path / "merged.tsv"
    aligner.prepare_viter_dictionary(args, merged)
    assert "ג'ץ\tג' צ" in merged.read_text(encoding="utf-8")
    assert dictionary.read_bytes() == source
    assert aligner.split_graphemes("םןץףך") == list("מנצפכ")
    assert aligner.split_graphemes("123") is None
    dictionary.write_text("שלום\tʃ a l o m\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Hebrew grapheme"):
        aligner.prepare_viter_dictionary(args, merged)


def test_resume_rejects_backend_change_without_truncating_output(tmp_path):
    output = tmp_path / "valid_aligned.jsonl"
    old = {"aligner": "mms"}
    handle, failed, _ = aligner.open_resumable_files(output, False, old)
    handle.write(b"preserve me\n")
    handle.close()
    failed.close()
    with pytest.raises(ValueError, match="Resume configuration differs"):
        aligner.open_resumable_files(output, False, {"aligner": "viter"})
    assert output.read_bytes() == b"preserve me\n"


def test_multiple_workers_on_same_gpu_have_distinct_readiness_ids(monkeypatch):
    monkeypatch.setattr(aligner, "initialize_worker", Mock())
    monkeypatch.setattr(aligner, "limit_auxiliary_threads", Mock())
    monkeypatch.setattr(aligner.signal, "signal", Mock())
    args = Namespace(aligner="mms", model="model", dtype="float16")
    results = queue.Queue()
    for worker_id in ("mms-cuda:0-0", "mms-cuda:0-1"):
        tasks = queue.Queue()
        tasks.put(None)
        aligner.worker_main("cuda:0", worker_id, args, tasks, results)
    assert results.get_nowait() == ("ready", "mms-cuda:0-0")
    assert results.get_nowait() == ("ready", "mms-cuda:0-1")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def test_the_input_is_the_diacritized_file_when_there_is_one_else_the_built_one(tmp_path):
    write_rows(tmp_path / "train.jsonl", [sample()])
    assert aligner.source_path(tmp_path, "train") == tmp_path / "train.jsonl"
    write_rows(tmp_path / "train_diac.jsonl", [sample()])
    assert aligner.source_path(tmp_path, "train") == tmp_path / "train_diac.jsonl"
    assert aligner.source_path(tmp_path, "train", "built") == tmp_path / "train.jsonl"
    with pytest.raises(FileNotFoundError):
        aligner.source_path(tmp_path, "valid")
    with pytest.raises(FileNotFoundError):
        aligner.source_path(tmp_path, "train_x", "diac")


def test_the_first_rows_must_look_like_a_manifest_that_has_not_been_aligned(tmp_path):
    path = tmp_path / "train.jsonl"
    write_rows(path, [sample(), {"path": "x"}])
    with pytest.raises(SystemExit, match="row 2 has no numeric `start`"):
        aligner.source_path(tmp_path, "train")
    already = {**sample(), "alignment_backend": "viter", "original_words": []}
    write_rows(path, [already])
    with pytest.raises(SystemExit, match="already has `alignment_backend`"):
        aligner.source_path(tmp_path, "train")
    write_rows(path, [{**sample(), "nikud": []}])
    with pytest.raises(SystemExit, match="`nikud` list"):
        aligner.source_path(tmp_path, "train")
    write_rows(path, [{**sample(), "nikud": ["שָׁלוֹם"]}])
    assert aligner.source_path(tmp_path, "train") == path
    path.write_text("")
    with pytest.raises(SystemExit, match="is empty"):
        aligner.source_path(tmp_path, "train")


def test_a_row_the_aligner_cannot_place_keeps_its_nikud(monkeypatch):
    row = {**sample("שלום"), "nikud": ["שָׁלוֹם"], "alt_words": {"norms": []}}
    monkeypatch.setattr(aligner, "_VITER_WORDS", set())  # nothing is in the dictionary
    fallback = Mock()
    fallback.align.return_value = [{"error": "no CTC path"}]
    (outcome,) = aligner.align_viter_batch([(1, row)], options(), fallback)
    assert outcome["row"]["nikud"] == ["שָׁלוֹם"]
    assert outcome["row"]["alt_words"] == {"norms": []}


def placeholder_row(*words, duration=4.0):
    step = duration / len(words)
    return {
        "path": "clip.flac",
        "start": 0.0,
        "duration": duration,
        "transcript": " ".join(words),
        "speaker": "episode",
        "timing": "estimated",
        "words": [
            {"word": w, "start": i * step, "end": (i + 1) * step} for i, w in enumerate(words)
        ],
    }


def strict_options():
    return Namespace(decode_workers=1, beam=100, retry_beam=800, strict=True)


def viter_result(*spans):
    return SimpleNamespace(
        words=[SimpleNamespace(label=word, start=start, end=end) for word, start, end in spans]
    )


def test_strict_keeps_a_viter_row_and_clears_the_placeholder_marker(monkeypatch):
    row = placeholder_row("שלום", "עולם", "יפה")
    monkeypatch.setattr(aligner, "_VITER_WORDS", {"שלום", "עולם", "יפה"})
    monkeypatch.setattr(aligner, "decode_audio", lambda _: ([0.0] * 80000, 0.0))
    viter = Mock()
    viter.align_many.return_value = [
        viter_result(("שלום", 0.2, 0.9), ("עולם", 1.2, 2.0), ("יפה", 2.4, 3.5))
    ]
    monkeypatch.setattr(aligner, "_VITER", viter)
    fallback = Mock()
    (outcome,) = aligner.align_viter_batch([(1, row)], strict_options(), fallback)
    assert outcome["row"]["alignment_backend"] == "viter"
    assert "timing" not in outcome["row"] and "error" not in outcome
    fallback.align.assert_not_called()


@pytest.mark.parametrize("reason", ["digits", "unsupported", "viter"])
def test_strict_drops_a_row_viter_cannot_align_and_never_calls_mms(monkeypatch, reason):
    word = "2026" if reason == "digits" else "עולם"
    row = placeholder_row("שלום", word)
    monkeypatch.setattr(
        aligner, "_VITER_WORDS", set() if reason == "unsupported" else {"שלום", "עולם"}
    )
    monkeypatch.setattr(aligner, "decode_audio", lambda _: ([0.0] * 80000, 0.0))
    viter = Mock()
    viter.align_many.return_value = [None]
    monkeypatch.setattr(aligner, "_VITER", viter)
    fallback = Mock()
    outcomes = aligner.align_viter_batch([(5, row)], strict_options(), fallback)
    assert "row" not in outcomes[0] and outcomes[0]["error"].startswith("strict: ")
    fallback.align.assert_not_called()
    output, failed = io.BytesIO(), io.BytesIO()
    assert aligner.write_batch(output, failed, outcomes, [(5, row)]) == 1
    assert output.getvalue() == b"" and json.loads(failed.getvalue())["input_line"] == 5


def test_strict_drops_a_row_whose_aligned_span_cannot_be_cut(monkeypatch):
    row = placeholder_row("שלום", "עולם", duration=3.0)
    monkeypatch.setattr(aligner, "_VITER_WORDS", {"שלום", "עולם"})
    monkeypatch.setattr(aligner, "decode_audio", lambda _: ([0.0] * 48000, 0.0))
    viter = Mock()
    # 1.6 s of speech: the one boundary is at 0.8 s, under MIN_CUT_SEC from the start
    viter.align_many.return_value = [viter_result(("שלום", 0.3, 0.7), ("עולם", 0.9, 1.9))]
    monkeypatch.setattr(aligner, "_VITER", viter)
    fallback = Mock()
    (outcome,) = aligner.align_viter_batch([(1, row)], strict_options(), fallback)
    assert "row" not in outcome and "no word boundary" in outcome["error"]
    fallback.align.assert_not_called()
    # the same alignment is kept without --strict: the default behaviour is unchanged
    viter.align_many.return_value = [viter_result(("שלום", 0.3, 0.7), ("עולם", 0.9, 1.9))]
    (kept,) = aligner.align_viter_batch([(1, row)], options(), fallback)
    assert kept["row"]["alignment_backend"] == "viter"


def test_has_eligible_cut_follows_the_loaders_rule():
    def row(duration, *spans):
        return {
            "duration": duration,
            "words": [{"word": "w", "start": s, "end": e} for s, e in spans],
        }

    assert aligner.has_eligible_cut(row(3.0, (0.0, 1.0), (1.2, 3.0)))  # cut 1.1, 1.9 left
    assert not aligner.has_eligible_cut(row(2.0, (0.0, 0.9), (1.2, 2.0)))  # cut 1.05, 0.95 left
    assert not aligner.has_eligible_cut(row(3.0, (0.0, 3.0)))  # one word


def write_pcm_wav(path, rate=24000, channels=1, width=2, seconds=4):
    samples = np.random.default_rng(17).integers(
        -32000, 32000, int(rate * seconds * channels), dtype="<i2"
    )
    raw = samples.tobytes() if width == 2 else bytes(int(rate * seconds * channels * width))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(raw)
    return path


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 44100, 48000, 96000, 192000])
@pytest.mark.parametrize("start", [0.0, 0.500012, 0.6234567, 3.3])
def test_native_wav_matches_ffmpeg_samples_seek_padding_and_eof(tmp_path, monkeypatch, rate, start):
    if not shutil.which("ffmpeg") or aligner.swresample_library() is None:
        pytest.skip("ffmpeg and its optional legacy libswresample API are needed")
    path = write_pcm_wav(tmp_path / "audio.wav", rate=rate)
    before = path.read_bytes()
    row = {"path": str(path), "start": start, "duration": 0.2378912}
    origin = max(0.0, start - aligner.PRE_PADDING)
    padded = start + row["duration"] + aligner.POST_PADDING - origin
    expected = aligner.decode_audio_ffmpeg(path, origin, padded)
    fallback = Mock(side_effect=AssertionError("a supported WAV must use the native decoder"))
    monkeypatch.setattr(aligner, "decode_audio_ffmpeg", fallback)
    actual, actual_origin = aligner.decode_audio(row)
    assert actual_origin == origin
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    fallback.assert_not_called()
    assert path.read_bytes() == before


@pytest.mark.parametrize("channels,width", [(2, 2), (1, 1), (1, 3), (1, 4)])
def test_unsupported_wav_layout_uses_ffmpeg_without_loading_native_library(
    tmp_path, monkeypatch, channels, width
):
    path = write_pcm_wav(tmp_path / "audio.wav", channels=channels, width=width)
    native = Mock(side_effect=AssertionError("unsupported layout"))
    monkeypatch.setattr(aligner, "swresample_library", native)
    expected = np.array([0.25], dtype=np.float32)
    fallback = Mock(return_value=expected)
    monkeypatch.setattr(aligner, "decode_audio_ffmpeg", fallback)
    actual, origin = aligner.decode_audio({"path": str(path), "start": 1.0, "duration": 0.5})
    assert actual is expected and origin == 0.5
    fallback.assert_called_once_with(path, 0.5, 2.0)
    native.assert_not_called()


@pytest.mark.parametrize("filename", ["audio.flac", "broken.wav"])
def test_other_formats_and_malformed_wav_retain_ffmpeg_fallback(tmp_path, monkeypatch, filename):
    path = tmp_path / filename
    path.write_bytes(b"not a PCM WAV")
    before = path.read_bytes()
    fallback = Mock(return_value=np.ones(3, dtype=np.float32))
    monkeypatch.setattr(aligner, "decode_audio_ffmpeg", fallback)
    audio, _ = aligner.decode_audio({"path": str(path), "start": 0.0, "duration": 1.0})
    assert len(audio) == 3
    fallback.assert_called_once()
    assert path.read_bytes() == before


@pytest.mark.parametrize("variant", ["extensible", "rf64", "rifx", "junk-first"])
def test_nonstandard_wav_headers_are_left_to_ffmpeg(tmp_path, monkeypatch, variant):
    path = write_pcm_wav(tmp_path / "audio.wav")
    raw = bytearray(path.read_bytes())
    if variant == "extensible":
        # Even if wave accepts the format, it doesn't expose its channel mask/valid bits.
        raw[20:22] = b"\xfe\xff"
    elif variant == "rf64":
        raw[:4] = b"RF64"
    elif variant == "rifx":
        raw[:4] = b"RIFX"
    else:
        raw[12:12] = b"JUNK\x00\x00\x00\x00"
        raw[4:8] = (len(raw) - 8).to_bytes(4, "little")
    path.write_bytes(raw)
    native = Mock(side_effect=AssertionError("leave WAV metadata interpretation to ffmpeg"))
    monkeypatch.setattr(aligner, "swresample_library", native)
    fallback = Mock(return_value=np.ones(3, dtype=np.float32))
    monkeypatch.setattr(aligner, "decode_audio_ffmpeg", fallback)
    aligner.decode_audio({"path": str(path), "start": 1.0, "duration": 0.5})
    fallback.assert_called_once()
    native.assert_not_called()


def test_missing_native_library_uses_ffmpeg(tmp_path, monkeypatch):
    path = write_pcm_wav(tmp_path / "audio.wav")
    monkeypatch.setattr(aligner, "swresample_library", lambda: None)
    fallback = Mock(return_value=np.ones(3, dtype=np.float32))
    monkeypatch.setattr(aligner, "decode_audio_ffmpeg", fallback)
    aligner.decode_audio({"path": str(path), "start": 1.0, "duration": 0.5})
    fallback.assert_called_once_with(path, 0.5, 2.0)


def test_big_endian_hosts_use_ffmpeg(tmp_path, monkeypatch):
    path = write_pcm_wav(tmp_path / "audio.wav")
    monkeypatch.setattr(aligner.sys, "byteorder", "big")
    native = Mock(side_effect=AssertionError("PCM16 native sample format is little-endian"))
    monkeypatch.setattr(aligner, "swresample_library", native)
    fallback = Mock(return_value=np.ones(3, dtype=np.float32))
    monkeypatch.setattr(aligner, "decode_audio_ffmpeg", fallback)
    aligner.decode_audio({"path": str(path), "start": 1.0, "duration": 0.5})
    fallback.assert_called_once()
    native.assert_not_called()


def test_ffmpeg_failure_still_reports_stderr_and_caps_subprocess_threads(tmp_path, monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=1, stderr=b"invalid audio format"))
    monkeypatch.setattr(aligner.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="ffmpeg exited 1: invalid audio format"):
        aligner.decode_audio_ffmpeg(tmp_path / "audio.flac", 0.5, 2.0)
    command = run.call_args.args[0]
    for option in ("-threads", "-filter_threads", "-filter_complex_threads"):
        assert command[command.index(option) + 1] == "1"
    assert "-y" not in command  # ffmpeg has only a stdout destination, never a source output
    assert command[-1] == "pipe:1"


@pytest.mark.parametrize("missing", ["library", "symbols"])
def test_missing_resampler_library_or_symbols_are_cached(monkeypatch, missing):
    monkeypatch.setattr(aligner, "_SWRESAMPLE", None)
    monkeypatch.setattr(aligner, "_SWRESAMPLE_CHECKED", False)
    find = Mock(return_value=None if missing == "library" else "libswresample.so")
    load = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(aligner.ctypes.util, "find_library", find)
    monkeypatch.setattr(aligner.ctypes, "CDLL", load)
    assert aligner.swresample_library() is None
    assert aligner.swresample_library() is None
    find.assert_called_once_with("swresample")
    assert load.call_count == (missing == "symbols")


def test_resampler_library_initializes_once_for_concurrent_decoders(monkeypatch):
    monkeypatch.setattr(aligner, "_SWRESAMPLE", None)
    monkeypatch.setattr(aligner, "_SWRESAMPLE_CHECKED", False)
    monkeypatch.setattr(aligner.ctypes.util, "find_library", lambda _: "libswresample.so")
    library = Mock()
    load = Mock(return_value=library)
    monkeypatch.setattr(aligner.ctypes, "CDLL", load)
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: aligner.swresample_library(), range(32)))
    assert all(result is library for result in results)
    load.assert_called_once_with("libswresample.so")


@pytest.mark.parametrize("stage", ["init", "size", "convert", "flush"])
def test_native_resampler_errors_free_context_and_fall_back(tmp_path, monkeypatch, stage):
    path = write_pcm_wav(tmp_path / "audio.wav")
    library = SimpleNamespace(
        swr_alloc_set_opts=Mock(return_value=1),
        swr_init=Mock(return_value=-1 if stage == "init" else 0),
        swr_get_out_samples=Mock(return_value=-1 if stage == "size" else 100000),
        swr_convert=Mock(side_effect=[-1] if stage == "convert" else [100, -1]),
        swr_free=Mock(),
    )
    monkeypatch.setattr(aligner, "swresample_library", lambda: library)
    fallback = Mock(return_value=np.ones(3, dtype=np.float32))
    monkeypatch.setattr(aligner, "decode_audio_ffmpeg", fallback)
    aligner.decode_audio({"path": str(path), "start": 1.0, "duration": 0.5})
    library.swr_free.assert_called_once()
    fallback.assert_called_once()


@pytest.mark.parametrize(
    "start,duration", [(-1.0, 1.0), (0.0, 0.0), (float("nan"), 1.0), (0.0, float("inf"))]
)
def test_invalid_spans_are_rejected_before_native_or_ffmpeg(tmp_path, monkeypatch, start, duration):
    path = write_pcm_wav(tmp_path / "audio.wav")
    native, fallback = Mock(), Mock()
    monkeypatch.setattr(aligner, "decode_pcm_wav", native)
    monkeypatch.setattr(aligner, "decode_audio_ffmpeg", fallback)
    with pytest.raises(ValueError, match="invalid span"):
        aligner.decode_audio({"path": str(path), "start": start, "duration": duration})
    native.assert_not_called()
    fallback.assert_not_called()


def test_batch_dictionary_keeps_normalized_aliases_and_pronunciation_order(tmp_path, monkeypatch):
    full = tmp_path / "merged.tsv"
    content = (
        'שלום\tש ל ו מ\nשלום(2)\tש ל מ\n"שלום,"\tש ל ו מ\n'
        "WORD\tו ר ד\nword\tו ו ר ד\nעולם\tע ו ל מ\n"
    )
    full.write_text(content, encoding="utf-8")
    words, entries = aligner.load_viter_dictionary(full)
    assert words == {"שלום", "שלום(2)", '"שלום,"', "WORD", "word", "עולם"}
    destination = tmp_path / "worker.tsv"
    monkeypatch.setattr(aligner, "_VITER_DICTIONARY", entries)
    monkeypatch.setattr(aligner, "_VITER_DICTIONARY_PATH", destination)
    aligner.write_viter_batch_dictionary(["שלום", "word", "שלום"])
    assert destination.read_text(encoding="utf-8") == (
        'WORD\tו ר ד\nword\tו ו ר ד\nשלום\tש ל ו מ\nשלום(2)\tש ל מ\n"שלום,"\tש ל ו מ\n'
    )
    aligner.write_viter_batch_dictionary(["עולם"])
    assert destination.read_text(encoding="utf-8") == "עולם\tע ו ל מ\n"
    assert full.read_text(encoding="utf-8") == content


def test_batch_dictionary_excludes_decode_failures_and_unsupported_words(tmp_path, monkeypatch):
    full = tmp_path / "merged.tsv"
    content = "שלום\tש ל ו מ\nעולם\tע ו ל מ\n"
    full.write_text(content, encoding="utf-8")
    words, entries = aligner.load_viter_dictionary(full)
    destination = tmp_path / "worker.tsv"
    monkeypatch.setattr(aligner, "_VITER_WORDS", words)
    monkeypatch.setattr(aligner, "_VITER_DICTIONARY", entries)
    monkeypatch.setattr(aligner, "_VITER_DICTIONARY_PATH", destination)

    def decode(row):
        if row["transcript"] == "עולם":
            raise ValueError("bad audio")
        return np.zeros(64000, dtype=np.float32), 2.5

    monkeypatch.setattr(aligner, "decode_audio", decode)

    def align_many(*args, **kwargs):
        assert destination.read_text(encoding="utf-8") == "שלום\tש ל ו מ\n"
        return [viter_result(("שלום", 0.2, 1.7))]

    monkeypatch.setattr(aligner, "_VITER", SimpleNamespace(align_many=align_many))
    fallback = Mock()
    fallback.align.return_value = [{"error": "cannot align"}] * 2
    rows = [(1, sample("שלום")), (2, sample("עולם")), (3, sample("זר"))]
    outcomes = aligner.align_viter_batch(rows, options(), fallback)
    assert [outcome["row"]["alignment_backend"] for outcome in outcomes] == [
        "viter",
        "original",
        "original",
    ]
    fallback.align.assert_called_once_with(rows[1:])
    assert full.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("fail_startup", [False, True])
def test_viter_workers_use_unique_scratch_dictionaries_and_clean_up(
    tmp_path, monkeypatch, fail_startup
):
    full = tmp_path / "merged.tsv"
    content = "שלום\tש ל ו מ\nעולם\tע ו ל מ\n"
    full.write_text(content, encoding="utf-8")
    paths = []

    def model_factory(*args, **kwargs):
        path = Path(kwargs["dict"])
        paths.append(path)
        assert path != full and path.is_file()
        if fail_startup:
            raise RuntimeError("model load failed")

        def align_many(*args, **kwargs):
            assert path.read_text(encoding="utf-8") == "שלום\tש ל ו מ\n"
            return [viter_result(("שלום", 0.2, 1.7))]

        return SimpleNamespace(phones=["ש", "ל", "ו", "מ"], align_many=align_many)

    monkeypatch.setitem(sys.modules, "viter", SimpleNamespace(Model=model_factory))
    monkeypatch.setattr(aligner.signal, "signal", Mock())
    monkeypatch.setattr(aligner, "limit_auxiliary_threads", Mock())
    monkeypatch.setattr(aligner, "_VITER", None)
    monkeypatch.setattr(aligner, "_VITER_WORDS", set())
    monkeypatch.setattr(aligner, "_VITER_DICTIONARY", {})
    monkeypatch.setattr(aligner, "_VITER_DICTIONARY_PATH", None)
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    monkeypatch.setattr(aligner, "decode_audio", lambda _: (np.zeros(64000), 2.5))
    args = Namespace(
        aligner="viter",
        viter_threads=4,
        viter_model=tmp_path / "model.viter",
        merged_dictionary=full,
        decode_workers=1,
        beam=100,
        retry_beam=800,
    )
    for worker in range(2):
        tasks, results = queue.Queue(), queue.Queue()
        tasks.put((0, [(1, sample())]))
        tasks.put(None)
        aligner.worker_main("cpu", f"worker-{worker}", args, tasks, results)
        if fail_startup:
            assert results.get_nowait()[0] == "fatal"
        else:
            assert results.get_nowait() == ("ready", f"worker-{worker}")
            assert results.get_nowait()[0] == "batch"
        assert not paths[-1].exists()
        assert aligner._VITER_DICTIONARY_PATH is None
        assert full.read_text(encoding="utf-8") == content
    assert paths[0] != paths[1]


def test_auxiliary_thread_caps_do_not_change_the_viter_thread_budget(monkeypatch):
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    for name in names:
        monkeypatch.setenv(name, "64")
    monkeypatch.setenv("RAYON_NUM_THREADS", "4")
    aligner.limit_auxiliary_threads()
    assert all(aligner.os.environ[name] == "1" for name in names)
    assert aligner.os.environ["RAYON_NUM_THREADS"] == "4"
