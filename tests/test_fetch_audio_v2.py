"""Offline tests for the audio-v2 survey/selection logic and the builder's Whisper filters."""

from __future__ import annotations

import json
from collections import defaultdict
from types import SimpleNamespace

from hebrew_training import fetch_audio_v2 as av2
from hebrew_training.build_official_manifest import clean_segments


def segment(start, end, logprob, compression, no_speech=0.0, probability=0.95, text="שלום עולם"):
    words = [
        {"word": " שלום", "start": start, "end": (start + end) / 2, "probability": probability},
        {"word": " עולם", "start": (start + end) / 2, "end": end, "probability": probability},
    ]
    return {
        "start": start,
        "end": end,
        "text": text,
        "avg_logprob": logprob,
        "compression_ratio": compression,
        "no_speech_prob": no_speech,
        "words": words,
    }


TRANSCRIPT = {
    "segments": [
        segment(0.0, 10.0, -0.10, 1.5),  # clean
        segment(10.0, 20.0, -0.55, 1.5),  # low logprob
        segment(20.0, 30.0, -0.10, 2.8),  # repetition loop
        segment(30.0, 40.0, -0.60, 3.0),  # both
        {"start": 40.0, "end": 45.0, "text": "", "words": []},  # no words
    ]
}


def test_episode_stats_totals():
    stats = av2.episode_stats(TRANSCRIPT)
    assert stats["segments"] == 4
    assert stats["speech_s"] == 40.0
    assert stats["no_words_s"] == 5.0
    assert stats["wall_s"] == 45.0
    assert sum(stats["joint"].values()) == 40.0
    assert sum(stats["wp"].values()) == 40.0
    json.dumps(stats)  # sparse and serializable


def test_kept_seconds_thresholds_snap_to_bin_edges():
    stats = av2.episode_stats(TRANSCRIPT)
    assert av2.kept_seconds(stats, None, None) == 40.0
    assert av2.kept_seconds(stats, -0.5, None) == 20.0  # drops both -0.55 and -0.60
    assert av2.kept_seconds(stats, None, 2.4) == 20.0  # drops 2.8 and 3.0
    assert av2.kept_seconds(stats, -0.5, 2.4) == 10.0  # only the clean one
    assert av2.kept_seconds(stats, -3.0, 9.0) == 40.0  # outside the axes keeps everything


def test_marginals_and_quantile():
    stats = av2.episode_stats(TRANSCRIPT)
    lp = av2.marginal([stats, stats], "lp")
    assert sum(lp.values()) == 80.0
    # Half the speech sits at -0.10, half at -0.55/-0.60.
    assert av2.quantile(lp, av2.LOGPROB, 0.25) < -0.5
    assert -0.2 < av2.quantile(lp, av2.LOGPROB, 0.75) < 0.0
    cr = av2.marginal([stats], "cr")
    assert sum(cr.values()) == 40.0


def test_bins_underflow_overflow():
    assert av2.bin_of(av2.LOGPROB, -9.0) == 0
    assert av2.bin_of(av2.LOGPROB, 0.0) == av2.LOGPROB.n + 1
    assert av2.bin_of(av2.COMPRESSION, 2.4) == av2.edge(av2.COMPRESSION, 2.4) + 1


def test_recording_id_is_ascii_and_stable():
    a = av2.recording_id("ישיבת הר עציון", "2020.01.01 שיעור 🎨 'א'")
    assert a.isascii() and a == av2.recording_id("ישיבת הר עציון", "2020.01.01 שיעור 🎨 'א'")
    assert a != av2.recording_id("ישיבת הר עציון", "other")


def test_selection_order_is_nested():
    ids = [f"av2_{i:04d}" for i in range(200)]
    order = sorted(ids, key=av2.stable_bucket)
    assert order[:50] == sorted(ids, key=av2.stable_bucket)[:50]
    assert len(set(order[:50]) & set(order[:100])) == 50


def test_audio_download_retries_rate_limit_and_keeps_only_complete_file(monkeypatch, tmp_path):
    import huggingface_hub
    import huggingface_hub.utils

    statuses = iter([429, 200])

    class Response:
        def __init__(self, status):
            self.status_code = status
            self.headers = {"retry-after": "3"} if status == 429 else {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            if self.status_code != 200:
                raise RuntimeError("rate limited")

        def iter_bytes(self, size):
            yield b"audio"

    class Session:
        def stream(self, *args, **kwargs):
            return Response(next(statuses))

    delays = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_url", lambda *args, **kwargs: "https://test")
    monkeypatch.setattr(huggingface_hub.utils, "build_hf_headers", dict)
    monkeypatch.setattr(huggingface_hub.utils, "get_session", lambda: Session())
    monkeypatch.setattr(av2.time, "sleep", delays.append)
    monkeypatch.setattr(av2.random, "random", lambda: 0.0)
    target = tmp_path / "audio.m4a"
    av2._download_audio(
        {"id": "episode", "source": "source", "episode": "name", "ext": ".m4a", "audio_bytes": 5},
        target,
    )
    assert target.read_bytes() == b"audio"
    assert delays == [3.0]
    assert not (tmp_path / "audio.m4a.part").exists()


def test_builder_whisper_filters_are_opt_in():
    stats: dict = defaultdict(int)
    base = {"min_quality": 0.6}
    off = clean_segments(TRANSCRIPT["segments"], SimpleNamespace(**base), stats)
    assert len(off) == 4  # default path unchanged: only the empty segment is dropped

    on = SimpleNamespace(**base, min_avg_logprob=-0.5, max_compression_ratio=2.4)
    stats = defaultdict(int)
    kept = clean_segments(TRANSCRIPT["segments"], on, stats)
    assert [s["start"] for s in kept] == [0.0]
    assert stats["asr_filtered"] == 3
    assert stats["no_words"] == 1

    # A corpus without these fields (CrowdRecital) passes the filters untouched.
    bare = {
        k: v for k, v in TRANSCRIPT["segments"][0].items() if k in ("start", "end", "text", "words")
    }
    assert len(clean_segments([bare], on, defaultdict(int))) == 1


def test_row_segments_matches_the_builders_cleaning():
    """The tagger runs without the builder; this pins its segment rule to clean_segments."""
    from hebrew_training.audio_v2_table import MIN_SEGMENT_QUALITY, row_segments

    segments = [
        segment(0.0, 5.0, -0.1, 1.5),
        segment(5.0, 10.0, -0.1, 1.5, probability=0.4),  # median word probability too low
        segment(10.0, 15.0, -0.1, 1.5, text="hello world"),  # no Hebrew
        {"start": 15.0, "end": 20.0, "text": "שלום", "words": []},  # no words
        segment(20.0, 25.0, -0.1, 1.5),
        {**segment(25.0, 30.0, -0.1, 1.5), "words": [{"word": "x", "start": None, "end": None}]},
    ]
    args = SimpleNamespace(min_quality=MIN_SEGMENT_QUALITY)
    builder = clean_segments(segments, args, defaultdict(int))
    ours = row_segments(segments, 0.0, 30.0)
    assert [(s["start"], s["end"]) for s in ours] == [(s["start"], s["end"]) for s in builder]
    assert [len(s["words"]) for s in ours] == [len(s["words"]) for s in builder]
    # Only segments fully inside the row are returned.
    assert [s["start"] for s in row_segments(segments, 0.0, 12.0)] == [0.0]
