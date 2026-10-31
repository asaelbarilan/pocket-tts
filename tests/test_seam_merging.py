import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from hebrew_training import align_official_manifest as aligner
from hebrew_training import build_official_manifest as builder


def _segment(start: float, end: float) -> dict:
    return {
        "start": start,
        "end": end,
        "text": "מילה",
        "words": [{"word": "מילה", "start": start, "end": end}],
    }


def _run(bounds: list[tuple[float, float]]) -> list[dict]:
    return [_segment(start, end) for start, end in bounds]


def _args(**overrides) -> SimpleNamespace:
    values = {
        "min_duration": 4.0,
        "max_duration": 30.0,
        "merge_target": 12.0,
        "merge_jitter": 0.0,
        "merge_gap": 1.5,
        "seam_window": 4.0,
        "greedy_merge": False,
        "seam_gap": 0.3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _merge(segments: list[dict], **overrides) -> list[list[dict]]:
    return builder.merge_segments(segments, _args(**overrides), random.Random(0))


def _ends(groups: list[list[dict]]) -> list[float]:
    return [group[-1]["end"] for group in groups]


def test_seam_first_skips_the_greedy_boundary_inside_running_speech() -> None:
    # Boundaries every 4 s. Greedy closes at 12.0 (gap 0.05, running speech); the pause is at 16.0.
    bounds = [(0.0, 4.0), (4.05, 8.0), (8.05, 12.0), (12.05, 16.0), (16.9, 20.0), (20.05, 24.0)]
    segments = _run(bounds)

    assert _ends(_merge(segments, greedy_merge=True)) == [12.0, 24.0]
    assert _ends(_merge(segments)) == [16.0, 24.0]


def test_seam_first_prefers_the_pause_nearest_the_target_then_the_longest_pause() -> None:
    bounds = [(0.0, 6.0), (6.5, 9.0), (9.05, 11.0), (11.5, 14.0), (14.05, 18.0), (18.05, 22.0)]
    # Window 8..16 s. Pauses of 0.5 s close at 6.0 (outside the window) and 11.0 and 14.0 are
    # inside it; 11.0 is nearer the 12 s target.
    assert _ends(_merge(_run(bounds)))[0] == 11.0

    no_pause = _run([(0.0, 4.0), (4.1, 8.0), (8.2, 12.0), (12.25, 16.0), (16.05, 20.0)])
    # No gap reaches 0.3 s. Inside the window, 8.0 -> 8.2 and 12.0 -> 12.25 are the longest: 0.2
    # and 0.25, so the cut is at 12.0.
    assert _ends(_merge(no_pause))[0] == 12.0


def test_a_cut_that_leaves_an_unusable_remainder_is_avoided() -> None:
    # Target 12, window 4: 15.0 is the nearest pause but leaves only 2.0 s, under --min-duration.
    segments = _run([(0.0, 8.0), (8.5, 11.0), (11.5, 15.0), (15.5, 17.0)])
    groups = _merge(segments, merge_target=12.0)

    assert [(g[0]["start"], g[-1]["end"]) for g in groups] == [(0.0, 11.0), (11.5, 17.0)]


def test_a_run_that_fits_the_window_is_one_group_and_gaps_over_the_limit_split_runs() -> None:
    segments = _run([(0.0, 5.0), (5.1, 10.0), (12.0, 16.0)])

    assert [len(g) for g in _merge(segments)] == [2, 1]


def test_a_single_segment_longer_than_the_window_still_makes_progress() -> None:
    segments = _run([(0.0, 40.0), (40.5, 45.0)])
    groups = _merge(segments)

    assert [len(g) for g in groups] == [1, 1]


def test_seam_first_never_overlaps_or_drops_a_segment() -> None:
    rng = random.Random(3)
    clock, bounds = 0.0, []
    for _ in range(300):
        length = rng.uniform(0.5, 5.0)
        bounds.append((clock, clock + length))
        clock += length + rng.choice([0.0, 0.05, 0.4, 0.9, 1.4, 2.5])
    segments = _run(bounds)
    groups = _merge(segments, merge_jitter=0.3)

    assert [segment for group in groups for segment in group] == segments


def _write_recording(tmp_path: Path, segments: list[dict]) -> Path:
    recording = tmp_path / "recording"
    recording.mkdir()
    (recording / "audio.wav").touch()
    (recording / "transcript.aligned.json").write_text(
        json.dumps({"segments": segments}), encoding="utf-8"
    )
    return recording


def _row_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        audio_glob="audio.wav",
        align_glob="transcript.aligned.json",
        metadata_glob="metadata.json",
        speaker_field="user_id",
        min_quality=0.6,
        min_duration=4.0,
        max_duration=30.0,
        merge_target=12.0,
        merge_jitter=0.0,
        merge_gap=1.5,
        transcode_audio=False,
        out_dir=tmp_path,
    )


def _spoken(start: float, end: float, probability: float = 0.9) -> dict:
    segment = _segment(start, end)
    segment["words"][0]["probability"] = probability
    return segment


def test_rows_report_the_silence_beside_them_including_dropped_neighbours(tmp_path) -> None:
    segments = [
        _spoken(0.0, 0.2, probability=0.1),
        _spoken(0.3, 0.9, probability=0.1),  # dropped: untranscribed speech before the row
        _spoken(1.4, 6.0),
        _spoken(6.1, 9.0),
        _spoken(9.7, 13.0, probability=0.1),  # dropped: untranscribed speech after the row
    ]
    recording = _write_recording(tmp_path, segments)

    rows, _ = builder.segment_rows(recording, _row_args(tmp_path), None, random.Random(0))

    assert len(rows) == 1
    assert rows[0]["lead_gap"] == 500  # 1.4 - 0.9, to the dropped segment before it
    assert rows[0]["tail_gap"] == 700  # 9.7 - 9.0


def test_rows_at_the_edge_of_the_recording_have_no_gap_on_that_side(tmp_path) -> None:
    recording = _write_recording(tmp_path, [_spoken(0.0, 5.0), _spoken(5.1, 9.0)])

    rows, _ = builder.segment_rows(recording, _row_args(tmp_path), None, random.Random(0))

    assert "lead_gap" not in rows[0]
    assert "tail_gap" not in rows[0]


def test_the_aligner_shifts_the_gaps_by_how_far_it_moved_the_edge_words() -> None:
    row = {
        "path": "a.wav",
        "start": 10.0,
        "duration": 5.0,
        "words": [{"word": "א", "start": 0.0, "end": 1.0}, {"word": "ב", "start": 1.0, "end": 5.0}],
        "lead_gap": 400,
        "tail_gap": 300,
    }
    # The aligner finds speech starting 0.25 s later and the last word ending 0.1 s earlier.
    timed = [
        {"word": "א", "start": 10.25 - 9.5, "end": 11.5 - 9.5},
        {"word": "ב", "start": 11.5 - 9.5, "end": 14.9 - 9.5},
    ]

    updated = aligner.update_timings(row, 9.5, timed, "viter")

    assert updated["lead_gap"] == 650
    assert updated["tail_gap"] == 400


def test_the_aligner_never_reports_a_negative_gap_and_ignores_rows_without_them() -> None:
    row = {
        "path": "a.wav",
        "start": 0.0,
        "duration": 4.0,
        "words": [{"word": "א", "start": 0.0, "end": 4.0}],
        "tail_gap": 50,
    }
    updated = aligner.update_timings(row, 0.0, [{"word": "א", "start": 0.0, "end": 4.3}], "viter")

    assert updated["tail_gap"] == 0
    assert "lead_gap" not in updated


def _parse(*flags: str):
    return builder.build_parser().parse_args(["--corpus", "c", "--out-dir", "o", *flags])


def test_seam_aware_merging_is_the_default_and_greedy_is_opt_in() -> None:
    args = _parse()
    assert (args.seam_window, args.seam_gap, args.greedy_merge) == (5.0, 0.3, False)
    assert _parse("--greedy-merge").greedy_merge is True


def test_a_non_positive_seam_window_is_rejected() -> None:
    parser = builder.build_parser()
    with pytest.raises(SystemExit):
        builder.check_merge_args(parser, _parse("--seam-window", "0"))
    builder.check_merge_args(parser, _parse("--seam-window", "3"))
    builder.check_merge_args(parser, _parse("--greedy-merge"))
