from __future__ import annotations

from pathlib import Path

import pytest

from hebrew_training.evaluation import bootstrap_interval, normalize_hebrew_for_asr
from hebrew_training.train import ensure_safe_run_directory


def test_hebrew_normalization_is_shared_and_stable() -> None:
    assert normalize_hebrew_for_asr("  שָׁלוֹם, עולם!  ") == "שלום עולם"


def test_bootstrap_interval_is_deterministic() -> None:
    references = ["א ב", "ג ד", "ה ו"]
    hypotheses = ["א ב", "ג", "ז ו"]

    def mismatch_rate(refs: list[str], hyps: list[str]) -> float:
        return sum(reference != hypothesis for reference, hypothesis in zip(refs, hyps)) / len(refs)

    first = bootstrap_interval(references, hypotheses, mismatch_rate, samples=100, seed=9)
    second = bootstrap_interval(references, hypotheses, mismatch_rate, samples=100, seed=9)
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_fresh_run_refuses_nonempty_directory(tmp_path: Path) -> None:
    (tmp_path / "metrics.jsonl").write_text("old run", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing a fresh run"):
        ensure_safe_run_directory(tmp_path, resume=None)


def test_resume_allows_existing_directory(tmp_path: Path) -> None:
    (tmp_path / "metrics.jsonl").write_text("old run", encoding="utf-8")
    ensure_safe_run_directory(tmp_path, resume=tmp_path / "checkpoint-0001000")
