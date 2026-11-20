"""Fixed-cut rows (`prompt_end`) and the numeric-speech builder's pure parts."""

import json

import numpy as np
import pytest
import sphn

from hebrew_training import build_numeric_speech_manifest as ns
from training.tests.test_dataloader import SR, _loader


def _fixed_cut_manifest(tmp_path, prompt_end=3.5, duration=8.0, **extra):
    """3.5 s of 220 Hz (prompt) followed by 4.5 s of 880 Hz (target)."""
    t = np.arange(int(duration * SR)) / SR
    tone = np.where(
        t < 3.5, 0.2 * np.sin(2 * np.pi * 220 * t), 0.2 * np.sin(2 * np.pi * 880 * t)
    ).astype(np.float32)
    wav = tmp_path / "row.wav"
    sphn.write_wav(str(wav), tone, SR)
    row = {
        "path": str(wav),
        "duration": duration,
        "transcript": "יש 143 משתתפים",
        "prompt_end": prompt_end,
        **extra,
    }
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n" * 1)
    return str(manifest)


def _dominant_hz(wav):
    spectrum = np.abs(np.fft.rfft(wav))
    return np.fft.rfftfreq(len(wav), 1 / SR)[spectrum.argmax()]


def test_prompt_end_cuts_exactly_there(tmp_path):
    loader = _loader(_fixed_cut_manifest(tmp_path), max_voice_prompt_sec=5.0, batch_size=1)
    wav, tokens, prompt, length, text = loader._sample(loader.get_entry(0))
    assert length == len(prompt) == int(3.5 * SR)
    assert len(wav) == pytest.approx(4.5 * SR, abs=2)
    assert _dominant_hz(prompt) == pytest.approx(220, abs=2)
    assert _dominant_hz(wav) == pytest.approx(880, abs=2)
    assert text == "יש 143 משתתפים"  # the digits form, untouched
    assert len(tokens) == max(1, len(text) // 3)


def test_prompt_end_ignores_words_and_min_cut(tmp_path):
    # words would put cuts elsewhere, and 0.5 s is under MIN_CUT_SEC: the sentinel still wins.
    words = [{"word": "a", "start": 1.0, "end": 2.0}, {"word": "b", "start": 4.0, "end": 5.0}]
    loader = _loader(_fixed_cut_manifest(tmp_path, prompt_end=0.5, words=words), batch_size=1)
    _, _, _, length, _ = loader._sample(loader.get_entry(0))
    assert length == int(0.5 * SR)


def test_prompt_over_the_cap_is_an_error_not_a_truncation(tmp_path):
    loader = _loader(_fixed_cut_manifest(tmp_path), max_voice_prompt_sec=3.0, batch_size=1)
    with pytest.raises(ValueError, match="max_voice_prompt_sec"):
        loader._sample(loader.get_entry(0))


@pytest.mark.parametrize("prompt_end", [0, 8.0, 9.0])
def test_prompt_end_outside_the_audio_is_an_error(tmp_path, prompt_end):
    loader = _loader(_fixed_cut_manifest(tmp_path, prompt_end=prompt_end), batch_size=1)
    with pytest.raises(ValueError, match="prompt_end"):
        loader._sample(loader.get_entry(0))


def test_pairs_load_from_every_jsonl_once(tmp_path):
    row = {"id": "a", "digits": "יש 5", "cardinal": "יש חמש", "split": "train"}
    (tmp_path / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n")
    dev = {"id": "b", "digits": "7", "cardinal": "שבע", "split": "dev"}
    dup = {"id": "a", "digits": "x", "cardinal": "y"}
    (tmp_path / "dev.jsonl").write_text(
        json.dumps(dev) + "\n" + json.dumps(dup) + "\n" + json.dumps({"id": "c"}) + "\n"
    )
    (tmp_path / "manifest.json").write_text("{}")
    assert [p["id"] for p in ns.load_pairs(tmp_path)] == ["a", "b"]


def _library(speakers=10, per=3):
    return {
        f"s{i}": [{"id": f"s{i}-{j}", "speaker": f"s{i}", "duration": 3.0} for j in range(per)]
        for i in range(speakers)
    }


def _pairs(n, split="train"):
    return [{"id": f"{split}{i:03d}", "split": split} for i in range(n)]


def test_speakers_are_split_and_never_shared_between_train_and_valid():
    lib = _library(20)
    pairs = _pairs(60) + _pairs(6, "dev") + _pairs(2, "test")
    got = ns.assign(pairs, lib, {"dev", "test"}, 0.2, seed=1)
    train = {p["speaker"] for side, p in got.values() if side == "train"}
    valid = {p["speaker"] for side, p in got.values() if side == "valid"}
    assert valid and train and not train & valid
    assert len(valid) <= 4  # 20% of 20 speakers
    assert {got[p["id"]][0] for p in pairs if p["split"] != "train"} == {"valid"}


def test_each_speaker_is_used_once_before_any_repeats():
    lib = _library(10)
    got = ns.assign(_pairs(18), lib, {"dev"}, 0.2, seed=0)  # 8 train speakers
    order = [got[p["id"]] for p in sorted(_pairs(18), key=lambda p: ns._hash(0, "pair", p["id"]))]
    first_cycle = [prompt["speaker"] for _, prompt in order[:8]]
    assert len(set(first_cycle)) == 8
    second = [prompt for _, prompt in order[8:16]]
    assert len({p["speaker"] for p in second}) == 8
    assert all(p["id"].endswith("-1") for p in second)  # the speaker's next prompt


def test_assignment_depends_only_on_seed_and_inputs():
    lib, pairs = _library(12), _pairs(30)
    first = ns.assign(pairs, lib, {"dev"}, 0.1, seed=3)
    assert ns.assign(list(reversed(pairs)), lib, {"dev"}, 0.1, seed=3) == first
    assert ns.assign(pairs, lib, {"dev"}, 0.1, seed=4) != first


def test_one_speaker_library_is_refused():
    with pytest.raises(SystemExit, match="at least 2"):
        ns.assign(_pairs(3), _library(1), {"dev"}, 0.1, seed=0)


def test_trim_tail_keeps_a_short_silence():
    audio = np.concatenate([np.full(SR, 0.3), np.zeros(3 * SR)]).astype(np.float32)
    trimmed = ns.trim_tail(audio, SR)
    assert len(trimmed) == pytest.approx(SR + ns.TRAIL_SEC * SR, abs=2)


def test_build_row_names_the_exact_cut(tmp_path):
    prompt_audio = np.full(int(3.2 * SR), 0.1, dtype=np.float32)
    speech = np.full(SR, 0.1, dtype=np.float32)
    pair = {"id": "x", "digits": "יש 5", "cardinal": "יש חמש", "split": "train"}
    row = ns.build_row(pair, {"id": 7}, prompt_audio, speech, tmp_path / "x.wav", SR)
    assert row["prompt_end"] == pytest.approx(3.2)
    assert row["duration"] == pytest.approx(4.2)
    assert row["transcript"] == "יש 5"
    assert "words" not in row
    wav, sr = sphn.read(row["path"])
    assert sr == SR and wav.shape[-1] == int(4.2 * SR)


class _FakeModel:
    sample_rate = SR

    @classmethod
    def load_model(cls, **_):
        return cls()

    def to(self, _):
        return self

    def eval(self):
        return self

    def get_state_for_audio_prompt(self, path):
        return path

    def generate_audio(self, state, text, max_tokens):
        import torch

        assert "143" not in text  # the model is fed the cardinal form
        return torch.full((SR,), 0.1)


def _dataset(tmp_path):
    pairs, lib = tmp_path / "pairs", tmp_path / "lib"
    pairs.mkdir()
    (lib / "prompts").mkdir(parents=True)
    rows = [
        {"id": f"t{i}", "split": "train", "digits": f"עד {i}43", "cardinal": f"עד מאה {i}"}
        for i in range(6)
    ] + [{"id": "d1", "split": "dev", "digits": "עד 7", "cardinal": "עד שבע"}]
    (pairs / "all.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    library = []
    for i in range(5):  # five speakers, no `split` field: the builder splits them itself
        sphn.write_wav(str(lib / "prompts" / f"{i}.wav"), np.full(3 * SR, 0.1, np.float32), SR)
        library.append(
            {"id": i, "audio": f"prompts/{i}.wav", "speaker": f"spk{i}", "duration": 3.0}
        )
    (lib / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in library))
    return ["--pairs", str(pairs), "--prompts", str(lib)]


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_main_writes_split_manifests_and_resumes(tmp_path, monkeypatch):
    import pocket_tts

    monkeypatch.setattr(pocket_tts, "TTSModel", _FakeModel)
    out = tmp_path / "out"
    argv = [*_dataset(tmp_path), "--out-dir", str(out)]
    ns.main(argv)
    train, valid = _read(out / "train.jsonl"), _read(out / "valid.jsonl")
    assert len(train) == 6 and [r["id"] for r in valid] == ["d1"]
    assert not {r["speaker"] for r in train} & {r["speaker"] for r in valid}
    assert len({r["speaker"] for r in train}) == 4  # all four train speakers get used
    assert all(r["prompt_end"] == pytest.approx(3.0) for r in train)
    assert train[0]["transcript"].startswith("עד ") and "audio" in train[0]["path"]
    (out / "train.jsonl").open("a").write('{"id": "torn')  # a killed run's last line
    ns.main(argv)  # nothing left to do, nothing duplicated
    assert len(_read_ok(out / "train.jsonl")) == 6


def _read_ok(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.endswith("}")]


def test_shards_partition_the_work_and_merge(tmp_path, monkeypatch):
    import pocket_tts

    monkeypatch.setattr(pocket_tts, "TTSModel", _FakeModel)
    out = tmp_path / "out"
    base = [*_dataset(tmp_path), "--out-dir", str(out), "--num-shards", "2"]
    ns.main([*base, "--shard-index", "0"])
    ns.main([*base, "--shard-index", "1"])
    ids = [r["id"] for f in out.glob("*.shard-*.jsonl") for r in _read(f)]
    assert sorted(ids) == sorted([f"t{i}" for i in range(6)] + ["d1"])
    ns.main(["--out-dir", str(out), "--merge"])
    assert not list(out.glob("*.shard-*"))
    assert len(_read(out / "train.jsonl")) == 6 and len(_read(out / "valid.jsonl")) == 1
    ns.main([*base, "--shard-index", "0"])  # merged rows still count as done
    assert not list(out.glob("*.shard-*"))


def test_aggregate_file_without_splits_cannot_relabel_pairs(tmp_path):
    def row(i, **extra):
        return {"id": i, "digits": "1", "cardinal": "אחת", **extra}

    # `pairs.jsonl` sorts before `test.jsonl`: it must not claim the test pair for itself.
    (tmp_path / "pairs.jsonl").write_text(
        "\n".join(json.dumps(row(i)) for i in ("a", "b", "c", "new"))
    )
    (tmp_path / "test.jsonl").write_text(json.dumps(row("b", split="test")))
    (tmp_path / "train.jsonl").write_text(json.dumps(row("a")))  # split from the file name
    (tmp_path / "dev.jsonl").write_text(json.dumps(row("c", split="dev")))
    got = {p["id"]: p["split"] for p in ns.load_pairs(tmp_path)}
    assert got == {"a": "train", "b": "test", "c": "dev", "new": "pairs"}


def test_pair_subset_is_seeded_and_sorted():
    pairs = [{"id": f"p{i:03d}", "split": "train"} for i in range(50)]
    a = ns.sample_pairs(pairs, 10, seed=1)
    assert a == ns.sample_pairs(pairs, 10, seed=1) and a != ns.sample_pairs(pairs, 10, seed=2)
    assert [p["id"] for p in a] == sorted(p["id"] for p in a) and len(a) == 10
