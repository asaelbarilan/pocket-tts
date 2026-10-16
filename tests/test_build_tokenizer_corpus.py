"""Mixing tokenizer text from local folders: shares, opt-in downloads, the charset filter."""

import json
import sys
from pathlib import Path

import pytest
import sentencepiece as spm
import yaml

from hebrew_training import build_tokenizer_corpus as builder
from hebrew_training import train_tokenizer
from hebrew_training.tokenizer_text import iter_documents

HEBREW = [
    "אדוני היושב ראש אני מבקש להעלות את הנושא לסדר היום",
    "חברי הכנסת הנכבדים מדובר בהחלטה משמעותית עבור אזרחי המדינה",
    "הוועדה תתכנס ביום שלישי הקרוב בשעה עשר בבוקר",
    "תודה רבה אני מסיים כאן את דבריי",
]
NIKUD = ["אֲנִי רוֹצָה לָלֶכֶת הַבַּיְתָה", "הַוַּעֲדָה תִּתְכַּנֵּס בְּיוֹם שְׁלִישִׁי הַקָּרוֹב"]
ENGLISH = [
    "The quick brown fox jumps over the lazy dog.",
    "He joined Microsoft before moving to the platform team.",
    "A café in Zürich served 12 customers.",
]


def _write(path: Path, lines: list[str], suffix: str = ".txt") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    file = path / f"part{suffix}"
    if suffix == ".jsonl":
        file.write_text("\n".join(json.dumps({"text": x}, ensure_ascii=False) for x in lines))
    else:
        file.write_text("\n".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def folders(tmp_path: Path) -> dict[str, Path]:
    # Foreign-script lines mixed into every folder: none of them may survive.
    junk = ["Привет мир это текст", "你好世界这是中文"]
    return {
        "manifest": _write(tmp_path / "manifest", HEBREW[:2] * 40 + junk),
        "hebrew": _write(tmp_path / "hebrew", HEBREW * 40 + junk, ".jsonl"),
        "hebrew_nikud": _write(tmp_path / "nikud", NIKUD * 60 + junk),
        "english": _write(tmp_path / "english", ENGLISH * 40 + junk),
    }


def _config(tmp_path: Path, folders, **pcts) -> builder.TokenizerConfig:
    config = builder.TokenizerConfig(
        target_mb=0.004, output_dir=tmp_path / "out", cache_dir=tmp_path / "cache"
    )
    for name, spec in config.sources.items():
        spec.pct = pcts.get(name, 0)
        spec.path = str(folders[name]) if spec.pct else None
    return config


def _run(config) -> str:
    return builder.build(config, builder.validate(config)).read_text(encoding="utf-8")


def test_mix_follows_the_requested_shares(tmp_path, folders):
    config = _config(tmp_path, folders, manifest=25, hebrew=25, hebrew_nikud=25, english=25)
    text = _run(config)
    summary = json.loads((config.output_dir / "corpus_summary.json").read_text())
    for got in summary["got_pct"].values():
        assert got == pytest.approx(25, abs=4), summary
    assert summary["short"] == []
    assert "Привет" not in text and "你好" not in text
    assert "café in Zürich" in text  # accented Latin is kept


def test_zero_percent_sources_are_not_used_and_need_nothing(tmp_path, folders):
    config = _config(tmp_path, folders, hebrew=60, english=40)
    # the other two have no path at all, and that is fine at 0%
    assert config.sources["manifest"].path is None
    text = _run(config)
    assert "אֲנִי" not in text
    assert (
        "hebrew_nikud"
        not in json.loads((config.output_dir / "corpus_summary.json").read_text())["chars"]
    )


def test_a_single_source_is_a_valid_mix(tmp_path, folders):
    text = _run(_config(tmp_path, folders, english=100))
    assert "quick brown fox" in text and "שלום" not in text


def test_shares_must_add_to_100(tmp_path, folders):
    with pytest.raises(SystemExit, match="add to 100"):
        builder.validate(_config(tmp_path, folders, hebrew=50, english=40))


def test_missing_path_and_no_download_fails_up_front(tmp_path, folders, monkeypatch):
    config = _config(tmp_path, folders, hebrew=50, english=50)
    config.sources["english"].path = None
    monkeypatch.setitem(sys.modules, "datasets", None)  # any import of it would explode
    with pytest.raises(SystemExit, match="nowhere to read from") as error:
        builder.validate(config)
    assert "--download english" in str(error.value)


def test_nonexistent_folder_fails_up_front(tmp_path, folders):
    config = _config(tmp_path, folders, hebrew=50, english=50)
    config.sources["english"].path = str(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        builder.validate(config)


def test_download_is_opt_in_and_never_attempted_for_a_path(tmp_path, folders, monkeypatch):
    # With paths given, `datasets` must never be imported, even if installed.
    monkeypatch.setitem(sys.modules, "datasets", None)
    _run(_config(tmp_path, folders, hebrew=50, english=50))


def test_download_flag_streams_through_datasets_and_caches(tmp_path, folders, monkeypatch):
    config = _config(tmp_path, folders, hebrew=50, english=50)
    for name in ("hebrew", "english"):
        config.sources[name].path = None
        config.sources[name].download = True
    calls = []

    def fake_stream(repo, subset):
        calls.append((repo, subset))
        yield from (HEBREW * 40 if "fineweb-2" in repo else ENGLISH * 40)

    monkeypatch.setattr(builder, "_stream", fake_stream)
    _run(config)
    assert sorted(calls) == [
        ("HuggingFaceFW/fineweb", "sample-10BT"),
        ("HuggingFaceFW/fineweb-2", "heb_Hebr"),
    ]
    assert (config.cache_dir / "hebrew.txt").exists()
    _run(config)  # second run reads the cache
    assert len(calls) == 2


def test_path_and_download_together_are_ambiguous(tmp_path, folders):
    config = _config(tmp_path, folders, hebrew=50, english=50)
    config.sources["english"].download = True
    with pytest.raises(SystemExit, match="pick one"):
        builder.validate(config)


def test_manifest_cannot_be_downloaded(tmp_path, folders):
    config = _config(tmp_path, folders, manifest=50, english=50)
    config.sources["manifest"].download = True
    config.sources["manifest"].path = None
    with pytest.raises(SystemExit, match="cannot be downloaded"):
        builder.validate(config)


def test_vocalized_source_must_actually_be_vocalized(tmp_path, folders):
    config = _config(tmp_path, folders, hebrew_nikud=50, english=50)
    config.sources["hebrew_nikud"].path = str(folders["hebrew"])  # plain Hebrew
    with pytest.raises(SystemExit, match="does not look vocalized"):
        _run(config)


def test_diacritize_needs_a_model_and_a_hebrew_source(tmp_path, folders):
    config = _config(tmp_path, folders, hebrew_nikud=50, english=50)
    config.sources["hebrew_nikud"].path = None
    config.sources["hebrew_nikud"].diacritize = True
    with pytest.raises(SystemExit, match="diacritizer.model"):
        builder.validate(config)
    config.diacritizer.model = "some/model"
    with pytest.raises(SystemExit, match="`hebrew` source"):
        builder.validate(config)
    config.sources["hebrew"].path = str(folders["hebrew"])  # pct stays 0
    assert "hebrew_nikud" in builder.validate(config)


def test_short_source_is_reported(tmp_path, folders):
    config = _config(tmp_path, folders, hebrew=50, english=50)
    config.target_mb = 1.0  # far more than the folders hold
    _run(config)
    summary = json.loads((config.output_dir / "corpus_summary.json").read_text())
    assert set(summary["short"]) == {"hebrew", "english"}


def test_config_file_and_cli_overrides(tmp_path, folders):
    path = tmp_path / "tok.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "vocab_size": 300,
                "target_mb": 0.004,
                "output_dir": str(tmp_path / "out"),
                "charset": {"allow": ["latin", "hebrew"], "on_foreign": "strip"},
                "sources": {
                    "hebrew": {"pct": 100, "path": str(folders["hebrew"])},
                    "english": {"pct": 0},
                },
            }
        )
    )
    args = builder.parse_args(
        [
            "--config",
            str(path),
            "--vocab-size",
            "250",
            "--pct",
            "hebrew=50",
            "--pct",
            "english=50",
            "--path",
            f"english={folders['english']}",
            "--no-fold",
        ]
    )
    config = builder.load_config(args.config)
    builder.apply_overrides(config, args)
    assert config.vocab_size == 250
    assert [config.sources[n].pct for n in ("hebrew", "english")] == [50, 50]
    assert config.charset.allow == ["latin", "hebrew"]
    assert config.charset.on_foreign == "strip"
    assert config.charset.fold is False
    assert set(builder.validate(config)) == {"hebrew", "english"}


def test_unknown_config_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("vocab_sise: 1\n")
    with pytest.raises(ValueError, match="vocab_sise"):
        builder.load_config(path)
    path.write_text("sources: {klingon: {pct: 5}}\n")
    with pytest.raises(ValueError, match="klingon"):
        builder.load_config(path)


def test_shipped_config_parses():
    shipped = Path(builder.__file__).parent / "configs" / "tokenizer_hebrew.yaml"
    config = builder.load_config(shipped)
    assert sum(s.pct for s in config.sources.values()) == 100


def test_readers_handle_txt_jsonl_gz_and_folders(tmp_path):
    import gzip

    (tmp_path / "a.txt").write_text("one\ntwo\n")
    (tmp_path / "b.jsonl").write_text('{"text": "three"}\n{"transcript": "four"}\nnot json\n')
    with gzip.open(tmp_path / "c.jsonl.gz", "wt") as handle:
        handle.write('{"text": "five"}\n')
    (tmp_path / "ignored.csv").write_text("x")
    assert list(iter_documents(tmp_path)) == ["one", "two", "three", "four", "not json", "five"]
    with pytest.raises(FileNotFoundError):
        list(iter_documents(tmp_path / "missing"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="contains no"):
        list(iter_documents(empty))


def test_parquet_source(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    pq.write_table(pa.table({"text": ENGLISH, "id": [1, 2, 3]}), tmp_path / "shard.parquet")
    assert list(iter_documents(tmp_path)) == ENGLISH


def test_training_filters_its_inputs_and_takes_the_vocab_size_as_a_parameter(tmp_path, folders):
    junk_in = _write(tmp_path / "raw", HEBREW * 30 + ENGLISH * 30 + ["Привет мир это текст"] * 30)
    model = train_tokenizer.train(
        [junk_in], tmp_path / "tok", vocab_size=340, charset=builder.CharsetFilter()
    )
    sp = spm.SentencePieceProcessor(model_file=str(model))
    assert sp.vocab_size() == 340
    # No piece was spent on the Cyrillic that was in the input.
    assert not any("\u0400" <= ch <= "\u04ff" for i in range(340) for ch in sp.id_to_piece(i))
    unfiltered = train_tokenizer.train(
        [junk_in], tmp_path / "raw_tok", vocab_size=340, charset=None
    )
    sp_raw = spm.SentencePieceProcessor(model_file=str(unfiltered))
    assert any("\u0400" <= ch <= "\u04ff" for i in range(340) for ch in sp_raw.id_to_piece(i))

    with pytest.raises(SystemExit, match="no text left"):
        train_tokenizer.train(
            [_write(tmp_path / "ru", ["Привет мир"] * 5)],
            tmp_path / "x",
            100,
            builder.CharsetFilter(),
        )


def test_a_cli_path_replaces_the_configs_diacritize_and_download(tmp_path, folders):
    config = builder.TokenizerConfig()
    config.sources["hebrew_nikud"].diacritize = True
    config.sources["english"].download = True
    args = builder.parse_args(
        [
            "--path",
            f"hebrew_nikud={folders['hebrew_nikud']}",
            "--path",
            f"english={folders['english']}",
        ]
    )
    builder.apply_overrides(config, args)
    assert not config.sources["hebrew_nikud"].diacritize
    assert not config.sources["english"].download
    args = builder.parse_args(["--download", "english"])
    builder.apply_overrides(config, args)
    assert config.sources["english"].download and config.sources["english"].path is None
