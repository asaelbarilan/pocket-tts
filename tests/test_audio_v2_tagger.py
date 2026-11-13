"""Offline tests for the audio-v2 tagger's model and statistics."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("pandas")

from hebrew_training.audio_v2_tagger import model as tm

NAMES = ["lp_mean", "cr_max", "noise"]


def synthetic(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    lp = rng.normal(-0.2, 0.15, n)
    cr = rng.normal(2.1, 0.3, n)
    noise = rng.normal(0, 1, n)
    x = np.stack([lp, cr, noise], axis=1)
    # Truth: good when logprob is high and compression is not extreme.
    score = (lp + 0.2) * 6 - np.maximum(cr - 2.4, 0) * 4 + rng.normal(0, 0.3, n)
    y = np.where(score > 0.5, 3, np.where(score > -0.5, 2, 1))
    return x, y


def test_wilson_interval_contains_estimate_and_shrinks():
    low, high = tm.wilson(30, 100)
    assert low < 0.3 < high
    low2, high2 = tm.wilson(300, 1000)
    assert (high2 - low2) < (high - low)
    assert tm.wilson(0, 0) == (0.0, 1.0)


def test_p_at_least():
    probs = np.array([[0.2, 0.3, 0.5]])
    assert tm.p_at_least(probs, 1)[0] == pytest.approx(1.0)
    assert tm.p_at_least(probs, 2)[0] == pytest.approx(0.8)
    assert tm.p_at_least(probs, 3)[0] == pytest.approx(0.5)


def test_too_few_labels_reports_direct_only():
    x, y = synthetic(10)
    df = tm.frame(x, np.zeros(10, dtype=int), 1, NAMES, False)
    population = tm.frame(x, np.zeros(10, dtype=int), 1, NAMES, False)
    result = tm.analyze(df, y, population, total_hours=1000.0, bootstrap=2)
    assert not result["ready"] and result["model"] is None
    assert 0 <= result["direct"][3]["hours"] <= 1000


def test_model_beats_prior_and_estimates_kept_share():
    x, y = synthetic(400, seed=1)
    px, py = synthetic(20000, seed=2)
    zeros = np.zeros(len(y), dtype=int)
    df = tm.frame(x, zeros, 1, NAMES, False)
    population = tm.frame(px, np.zeros(len(py), dtype=int), 1, NAMES, False)
    result = tm.analyze(df, y, population, total_hours=10000.0, bootstrap=8)
    assert result["ready"]
    cv = result["cv"]
    assert cv["log_loss"] < cv["baseline_log_loss"]
    assert cv["accuracy"] > cv["baseline_accuracy"]
    truth = float((py >= 3).mean()) * 10000.0
    estimate = result["model"][3]
    assert abs(estimate["hours"] - truth) < 0.08 * 10000
    assert estimate["hours_low"] <= estimate["hours"] <= estimate["hours_high"]
    assert [name for name, _ in result["importance"]][0] in ("lp_mean", "cr_max")


def test_source_feature_is_categorical_and_fits():
    x, y = synthetic(120, seed=3)
    source = np.random.default_rng(0).integers(0, 5, len(y))
    df = tm.frame(x, source, 5, NAMES, True)
    assert "source" in df.columns and str(df["source"].dtype) == "category"
    booster = tm.fit(df, y)
    probs = tm.predict(booster, df)
    assert probs.shape == (len(y), 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
