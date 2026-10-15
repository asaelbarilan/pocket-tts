import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import training.args
from training.eval import ivritai
from training.eval.ivritai import checkpoint_tag, read_sentences


def test_read_sentences_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "sentences.txt"
    path.write_text("משפט ראשון\n\n  משפט שני  \n", encoding="utf-8")

    assert read_sentences(path) == ["משפט ראשון", "משפט שני"]


def test_read_sentences_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "sentences.txt"
    path.write_text("\n  \n", encoding="utf-8")

    with pytest.raises(ValueError, match="no non-empty sentences"):
        read_sentences(path)


def test_checkpoint_tag_identifies_run_step_and_weights() -> None:
    checkpoint = Path("runs/finetune hebrew/checkpoint_00002000.pt")

    assert checkpoint_tag(checkpoint, 2_000, use_ema=True) == ("finetune_hebrew_step00002000_ema")


@pytest.mark.parametrize("asr_name", ["", "  "])
def test_blank_asr_generates_without_importing_whisper(tmp_path, monkeypatch, asr_name):
    run = tmp_path / "run"
    run.mkdir()
    (run / "args.yaml").write_text("", encoding="utf-8")
    voice = tmp_path / "voice.wav"
    voice.touch()
    checkpoint = run / "checkpoint_00000100.pt"
    checkpoint.touch()
    sentences = tmp_path / "sentences.txt"
    sentences.write_text("שלום עולם\n", encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ivritai",
            str(run),
            "--sentences",
            str(sentences),
            "--voices",
            str(voice),
            "--checkpoints",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--asr",
            asr_name,
            "--device",
            "cpu",
        ],
    )
    monkeypatch.setattr(
        training.args, "load_args", lambda path: SimpleNamespace(model_config="test")
    )
    original_import = builtins.__import__

    def forbid_whisper(name, *args, **kwargs):
        assert name != "faster_whisper", "blank ASR must not load Whisper, even if installed"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_whisper)

    def generate(checkpoint, texts, args, asr, progress):
        assert asr is None
        assert texts == ["שלום עולם"]
        progress(1)
        return {
            "checkpoint": str(checkpoint),
            "step": 100,
            "silent": 0,
            "no_eos": 0,
            "asr": None,
            "records": [{"text": texts[0], "audio": None}],
        }

    monkeypatch.setattr(ivritai, "evaluate_checkpoint", generate)
    ivritai.main()
    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert results[0]["asr"] is None
