import itertools
import json
import random

import numpy as np
import pytest
import soundfile as sf

from hebrew_training.voice_prompts import dedupe, extract

RATE = 24_000


def make_row(path, words, **extra):
    row = {"path": str(path), "start": 1.0, "duration": 14.0, "speaker": "spk", "words": words}
    row.update(extra)
    return row


def word(text, start, end):
    return {"word": text, "start": start, "end": end}


def args(**overrides):
    parsed = extract.argparse.ArgumentParser()
    extract.add_arguments(parsed)
    namespace = parsed.parse_args(["x", "--output", "y"])
    for key, value in overrides.items():
        setattr(namespace, key, value)
    return namespace


# Words every 0.5 s (0.4 s long, 0.1 s apart) except pauses of 0.4 s after "d" and "j".
WORDS = []
cursor = 0.0
for letter in "abcdefghijklmn":
    WORDS.append(word(letter, cursor, cursor + 0.4))
    cursor += 0.4 + (0.4 if letter in "dj" else 0.1)


def test_cuts_sit_in_the_middle_of_long_gaps_only(tmp_path):
    row = make_row(tmp_path / "a.wav", WORDS)
    plans = extract.plan_row(row, args(per_row=5), random.Random(0))
    assert plans
    for window, _ in plans:
        assert 3.0 <= window.source_end - window.source_start <= 5.0
        assert window.seam_before_ms >= 300 and window.seam_after_ms >= 300
        assert window.gap_before_ms == pytest.approx(window.seam_before_ms / 2, abs=1)
        assert window.gap_after_ms == pytest.approx(window.seam_after_ms / 2, abs=1)
        # no word straddles an edge
        for item in row["words"]:
            start, end = 1.0 + item["start"], 1.0 + item["end"]
            assert not (start < window.source_start < end)
            assert not (start < window.source_end < end)


def test_per_row_windows_do_not_overlap(tmp_path):
    row = make_row(tmp_path / "a.wav", WORDS, lead_gap=900, tail_gap=900)
    plans = extract.plan_row(row, args(per_row=5, min_duration=1.0, min_words=1), random.Random(1))
    spans = sorted((w.source_start, w.source_end) for w, _ in plans)
    assert len(spans) > 1
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))


def test_rows_without_measured_times_or_gaps_are_skipped(tmp_path):
    estimated = make_row(tmp_path / "a.wav", WORDS, timing="estimated")
    assert extract.plan_row(estimated, args(), random.Random(0)) == []
    kept_original = make_row(tmp_path / "a.wav", WORDS, alignment_backend="original")
    assert extract.plan_row(kept_original, args(), random.Random(0)) == []
    tight = make_row(tmp_path / "a.wav", [word(str(i), i * 0.5, i * 0.5 + 0.45) for i in range(12)])
    assert extract.plan_row(tight, args(), random.Random(0)) == []


def test_edge_cuts_need_the_neighbour_gap(tmp_path):
    words = [word(str(i), i * 0.5, i * 0.5 + 0.45) for i in range(8)]  # 4 s, no inner seams
    row = make_row(tmp_path / "a.wav", words, lead_gap=600, tail_gap=500)
    ((window, _),) = extract.plan_row(row, args(min_words=3), random.Random(0))
    assert window.seam_before_ms == 600 and window.seam_after_ms == 500
    assert window.source_start == pytest.approx(1.0 - 0.3)
    assert extract.plan_row(row, args(edge_cuts=False), random.Random(0)) == []
    assert extract.plan_row(dict(row, lead_gap=100), args(), random.Random(0)) == []


def test_extract_writes_numbered_wavs_and_manifest(tmp_path):
    folder = tmp_path / "ds"
    folder.mkdir()
    rng = np.random.default_rng(0)
    sf.write(folder / "a.wav", rng.uniform(-0.1, 0.1, 20 * RATE).astype("float32"), RATE)
    rows = [make_row("a.wav", WORDS, user_id="u1")]
    (folder / "train_aligned.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "out"
    extract.main([f"ds={folder}", "--output", str(out), "--splits", "train", "--workers", "2"])

    manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert [row["id"] for row in manifest] == list(range(len(manifest))) and manifest
    row = manifest[0]
    assert row["dataset"] == "ds" and row["split"] == "train" and row["stage"] == "aligned"
    assert row["user_id"] == "u1" and row["speaker"] == "spk" and row["line"] == 1
    info = sf.info(out / row["audio"])
    assert info.samplerate == RATE and info.channels == 1
    assert info.duration == pytest.approx(row["duration"], abs=1e-3)
    assert 3.0 <= row["duration"] <= 5.0
    assert row["source_end"] - row["source_start"] == pytest.approx(row["duration"], abs=1e-3)
    assert {"text", "num_words", "gap_before_ms", "gap_after_ms"} <= row.keys()

    with pytest.raises(SystemExit):
        extract.main([f"ds={folder}", "--output", str(out)])  # no silent overwrite


def test_per_speaker_cap(tmp_path):
    folder = tmp_path / "ds"
    folder.mkdir()
    rows = [make_row(f"{i}.wav", WORDS) for i in range(6)]
    (folder / "train_aligned.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    windows, stats = extract.plan(args(per_speaker=2, splits=["train"]), [("ds", folder)])
    assert len(windows) == 2 and stats["windows"] == 6 and stats["speakers"] == 1


def unit(*rows):
    matrix = np.array(rows, dtype="float32")
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def test_find_duplicates_keeps_the_first_of_a_cluster():
    matrix = unit([1, 0], [0.99, 0.1], [0, 1])
    found = dedupe.find_duplicates([0, 1, 2], matrix, 0.9)
    assert list(found) == [1] and found[1][0] == 0


def test_select_diverse_spreads_over_clusters():
    matrix = unit([1, 0], [0.99, 0.1], [0.98, 0.15], [0, 1], [-1, 0.05])
    removed = dedupe.select_diverse([10, 11, 12, 13, 14], matrix, 3)
    assert set(removed) == {11, 12} and removed[11][0] == 10


def test_standardize_gives_unit_vectors_and_evens_out_scale():
    pooled = np.random.default_rng(0).normal(size=(20, 4)) * [1, 100, 1, 1]
    matrix = dedupe.standardize(pooled)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)


def test_pooled_embedding_is_mean_and_std_of_the_prompt_sequence(tmp_path):
    torch = pytest.importorskip("torch")
    sf.write(tmp_path / "a.wav", np.zeros(RATE, dtype="float32"), RATE)
    sequence = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)

    class Mimi:
        def encode_to_latent(self, audio):
            return sequence

    class Model:
        mimi = Mimi()
        device = "cpu"

        class config:
            class mimi:
                sample_rate = RATE

    vector = dedupe.pooled_embedding(Model(), tmp_path / "a.wav", "latent")
    expected = np.concatenate(
        [sequence[0].mean(0).numpy(), sequence[0].std(0, unbiased=False).numpy()]
    )
    assert np.allclose(vector, expected)


def test_append_continues_numbering_and_avoids_earlier_spans(tmp_path):
    folder = tmp_path / "ds"
    folder.mkdir()
    rng = np.random.default_rng(0)
    sf.write(folder / "a.wav", rng.uniform(-0.1, 0.1, 20 * RATE).astype("float32"), RATE)
    row = make_row("a.wav", WORDS, lead_gap=900, tail_gap=900)
    (folder / "train_aligned.jsonl").write_text(json.dumps(row) + "\n")
    out = tmp_path / "out"
    base = [f"ds={folder}", "--output", str(out), "--splits", "train", "--per-speaker", "0"]
    base += ["--min-duration", "1", "--max-duration", "2.6", "--min-words", "1"]
    extract.main([*base, "--seed", "1"])
    extract.main([*base, "--seed", "2", "--append"])

    manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert [r["id"] for r in manifest] == [0, 1]
    first, second = (r for r in manifest)
    assert (
        first["source_end"] <= second["source_start"]
        or second["source_end"] <= first["source_start"]
    )
    assert len(list((out / "prompts").glob("*.wav"))) == 2

    capped = extract.load_existing(out)
    assert capped.next_id == 2 and sum(capped.per_speaker.values()) == 2
    with pytest.raises(SystemExit):
        extract.main([*base, "--sample-rate", "16000", "--append"])  # would mix rates
