from __future__ import annotations

import io
import json
import queue
from argparse import Namespace
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

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
