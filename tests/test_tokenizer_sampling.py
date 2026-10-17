import json
from dataclasses import asdict, replace

import pytest

from hebrew_training import build_tokenizer_corpus as builder
from hebrew_training import train_tokenizer
from hebrew_training.tokenizer_charset import CharsetFilter
from hebrew_training.tokenizer_experiment import choose_size
from hebrew_training.tokenizer_sampling import SamplingSpec, sample_source


def test_bounded_sampling_repeatable_disjoint_and_cached(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"speaker": f"recording-{j // 3}", "transcript": f"טקסט עברי לדוגמה מספר {j} " * 8}
        for j in range(600)
    ]
    source.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    spec = SamplingSpec(mode="bounded", holdout_chars=800, window_bytes=500, max_read_mb=2)
    cache = tmp_path / "cache"
    lines = sample_source(source, 4000, CharsetFilter(), lambda s: True, spec, cache)
    held = (cache / "heldout.txt").read_text().splitlines()
    report = json.loads((cache / "sampling.json").read_text())
    assert report["group_overlap"] == 0
    assert set(lines).isdisjoint(held)
    assert len(lines) == len(set(lines))
    assert sample_source(source, 2000, CharsetFilter(), lambda s: True, spec, cache) == lines
    again = sample_source(source, 4000, CharsetFilter(), lambda s: True, spec, tmp_path / "other")
    assert again == lines
    with pytest.raises(ValueError, match="stale"):
        sample_source(source, 4000, CharsetFilter(), lambda s: True, replace(spec, seed=4), cache)


def test_sampling_serializes_foreign_character_counts(tmp_path):
    path = tmp_path / "mixed.txt"
    path.write_text("\n".join(f"English document {i}\nПривет {i}" for i in range(200)) + "\n")
    spec = SamplingSpec(mode="bounded", holdout_chars=100, max_read_mb=1)
    cache = tmp_path / "cache"
    sample_source(path, 500, CharsetFilter(), lambda s: True, spec, cache)
    report = json.loads((cache / "sampling.json").read_text())
    assert report["filter_stats"]["foreign"]["П"] > 0


def test_sampling_enforces_io_budget(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("unsupported text\n" * 100)
    spec = SamplingSpec(mode="bounded", holdout_chars=0, max_read_mb=0.002)
    with pytest.raises(RuntimeError, match="hit"):
        sample_source(path, 100, CharsetFilter(), lambda s: False, spec, tmp_path / "cache")


def test_builder_vocalizes_both_domains_from_extra_text(tmp_path, monkeypatch):
    config = builder.TokenizerConfig(
        target_mb=0.004, output_dir=tmp_path / "out", cache_dir=tmp_path / "cache"
    )
    config.sampling = SamplingSpec(mode="bounded", holdout_chars=100, max_read_mb=1)
    for name, pct in (("manifest", 20), ("hebrew", 40), ("english", 20)):
        path = tmp_path / f"{name}.txt"
        path.write_text(
            "\n".join(
                f"{name} טקסט עברי מספר {i}" if name != "english" else f"English text number {i}"
                for i in range(300)
            )
        )
        config.sources[name] = builder.SourceSpec(pct=pct, path=str(path))
    config.sources["hebrew_nikud"] = builder.SourceSpec(
        pct=20, diacritize=True, diacritize_sources=["manifest", "hebrew"]
    )
    config.diacritizer.model = "fake"
    captured = []

    def fake(cached, lines, want, spec, charset):
        captured.extend(lines)
        return [line + "ָ" for line in lines]

    monkeypatch.setattr(builder, "diacritize", fake)
    builder.build(config, builder.validate(config))
    assert any(line.startswith("manifest") for line in captured)
    assert any(line.startswith("hebrew") for line in captured)
    plain = set((config.output_dir / "source_slices/manifest.txt").read_text().splitlines())
    assert plain.isdisjoint(captured)
    assert (config.output_dir / "source_slices/hebrew_nikud.txt").exists()


def test_size_selection_checks_every_domain_and_roundtrips():
    def metrics(value, failures=0):
        return {"tokens_per_char": value, "roundtrip_failures": failures, "unknown_tokens": 0}

    results = {
        4: {"he": metrics(0.31), "en": metrics(0.42)},
        8: {"he": metrics(0.30), "en": metrics(0.401)},
        16: {"he": metrics(0.30), "en": metrics(0.4)},
    }
    assert choose_size(results) == 8
    results[8]["en"]["roundtrip_failures"] = 1
    assert choose_size(results) == 16
    results[16]["en"]["roundtrip_failures"] = 1
    with pytest.raises(RuntimeError, match="no tokenizer passed"):
        choose_size(results)


def test_diacritizer_resume_tracks_inputs_not_only_accepted_outputs(tmp_path, monkeypatch):
    import transformers

    calls = []

    class FakeModel:
        def eval(self):
            return self

        def to(self, **kwargs):
            return self

        def predict(self, lines, tokenizer, **kwargs):
            assert kwargs["mark_matres_lectionis"] == ""
            calls.extend(lines)
            return [line + "ָָָ" for line in lines]

    class RejectOneLine(CharsetFilter):
        def clean(self, line):
            return None if line.startswith("שניים") else super().clean(line)

    monkeypatch.setattr(
        transformers.PreTrainedTokenizerFast, "from_pretrained", lambda *a, **k: None
    )
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **k: FakeModel())
    lines = ["אחד", "שניים", "שלושה", "ארבעה"]
    spec = builder.DiacritizerSpec(model="fake", device="cpu", batch_size=2)
    path = tmp_path / "marked.txt"
    first = builder.diacritize(path, lines, 1, spec, RejectOneLine())
    assert len(first) == 1
    resumed = builder.diacritize(path, lines, 1000, spec, RejectOneLine())
    assert len(resumed) == 3
    assert calls == lines  # rejected output does not make the input cursor go backwards
    with pytest.raises(ValueError, match="stale"):
        builder.diacritize(path, lines + ["אחר"], 1000, spec, CharsetFilter())


def test_diacritizer_python_offsets_adapter():
    import torch

    class FakeTokenizer:
        model_max_length = 512

        def __call__(self, text):
            return {"offset_mapping": torch.tensor([[[0, 1], [1, 2]]]), "input_ids": [1, 2]}

    adapted = builder._PythonOffsetsTokenizer(FakeTokenizer())
    assert adapted.model_max_length == 512
    assert adapted("אב") == {"offset_mapping": [[[0, 1], [1, 2]]], "input_ids": [1, 2]}


def test_diacritizer_deduplicates_normalization_expanded_offsets():
    import torch

    class FakeTokenizer:
        def __call__(self, text):
            return {"offset_mapping": torch.tensor([[[0, 1], [0, 1], [0, 1], [1, 2]]])}

    encoded = builder._PythonOffsetsTokenizer(FakeTokenizer())("…!")
    assert encoded["offset_mapping"] == [[[0, 1], [0, 0], [0, 0], [1, 2]]]


def test_diacritizer_strict_legacy_ellipsis_cache_migration(tmp_path):
    lines = ["אחד…", "אחר"]
    spec = builder.DiacritizerSpec(model="not-downloaded", device="cpu")
    charset = CharsetFilter()
    path = tmp_path / "marked.txt"
    path.write_text("אַחַד………\nאַחֵר\n")
    progress = {
        "provenance": {
            "inputs": builder.digest(lines),
            "model": asdict(spec),
            "charset": charset.describe(),
        },
        "next_input": 2,
        "output_lines": 2,
    }
    path.with_suffix(".progress.json").write_text(json.dumps(progress))
    result = builder.diacritize(path, lines, 1, spec, charset)
    assert result == ["אַחַד…", "אַחֵר"]
    assert json.loads(path.with_suffix(".progress.json").read_text())["deduplicated_offsets"]

    path.write_text("אַחַד………\nאַחֵרX\n")
    path.with_suffix(".progress.json").write_text(json.dumps(progress))
    with pytest.raises(ValueError, match="unexpected content change"):
        builder.diacritize(path, lines, 1, spec, charset)
    assert path.read_text() == "אַחַד………\nאַחֵרX\n"  # fail before modifying anything


def test_vocalization_seed_checks_provenance_and_processed_prefix(tmp_path):
    config = builder.TokenizerConfig(vocalization_seed_dir=tmp_path)
    config.diacritizer.model = "fake"
    inputs = ["אחד", "אחר", "נוסף"]
    outputs = ["אַחַד", "אַחֵר"]
    charset = CharsetFilter()
    (tmp_path / "vocalization_input.txt").write_text("\n".join(inputs) + "\n")
    (tmp_path / "hebrew_nikud.txt").write_text("\n".join(outputs) + "\n")
    (tmp_path / "hebrew_nikud.progress.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "inputs": builder.digest(inputs),
                    "model": asdict(config.diacritizer),
                    "charset": charset.describe(),
                },
                "next_input": 2,
                "output_lines": 2,
                "deduplicated_offsets": True,
            }
        )
    )
    assert builder._vocalization_seed(config, charset) == (inputs[:2], outputs)
    config.diacritizer.model = "other"
    with pytest.raises(ValueError, match="provenance"):
        builder._vocalization_seed(config, charset)


def test_required_rare_niqqud_does_not_need_synthetic_training_lines(tmp_path):
    import sentencepiece as spm

    source = tmp_path / "plain.txt"
    source.write_text("\n".join(f"טקסט עברי רגיל ללא ניקוד מספר {i}" for i in range(200)))
    model = train_tokenizer.train([source], tmp_path / "model", 290, required_chars="ֿ")
    sp = spm.SentencePieceProcessor(model_file=str(model))
    assert not any(sp.is_byte(i) for i in sp.encode("אֿ"))
    with pytest.raises(ValueError, match="required_chars"):
        train_tokenizer.train([source], tmp_path / "bad", 340, CharsetFilter(), "Ж")
