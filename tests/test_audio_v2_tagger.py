"""Offline tests for the audio-v2 tagger's model and statistics."""

from __future__ import annotations

import collections
import threading

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("pandas")

from hebrew_training.audio_v2_tagger import model as tm
from hebrew_training.audio_v2_tagger.server import Tagger

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
    assert next(name for name, _ in result["importance"]) in ("lp_mean", "cr_max")


def test_source_feature_is_categorical_and_fits():
    x, y = synthetic(120, seed=3)
    source = np.random.default_rng(0).integers(0, 5, len(y))
    df = tm.frame(x, source, 5, NAMES, True)
    assert "source" in df.columns and str(df["source"].dtype) == "category"
    booster = tm.fit(df, y)
    probs = tm.predict(booster, df)
    assert probs.shape == (len(y), 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_targeted_ratings_do_not_bias_label_only_hours():
    x, y = synthetic(120, seed=9)
    df = tm.frame(x, np.zeros(120, dtype=int), 1, NAMES, False)
    # Targeted selections are all class 3; only the random control is unbiased.
    y[60:] = 3
    result = tm.analyze(df, y, df, 1000.0, bootstrap=2, control_mask=np.arange(120) < 60)
    assert result["random_n"] == 60 and result["active_n"] == 60
    assert result["direct"][3]["hours"] == pytest.approx((y[:60] == 3).mean() * 1000)
    assert result["cv_population_holdout"] is True
    assert result["confusion_ranges"]


def test_active_scores_favor_uncertainty_and_error_ranges():
    df = tm.frame(
        np.array([[0.1, 2.0, 0.0], [0.2, 2.0, 0.0], [0.2, 2.0, 0.0]]),
        np.zeros(3, dtype=int),
        1,
        NAMES,
        False,
    )
    probs = np.array([[0.98, 0.01, 0.01], [0.34, 0.33, 0.33], [0.34, 0.33, 0.33]])
    scores = tm.active_scores(
        probs, df, [{"feature": "lp_mean", "low": 0.15, "high": 0.25, "excess": 0.2}]
    )
    assert scores[1] == pytest.approx(scores[2])
    assert scores[1] > scores[0]


def test_confusion_ranges_weight_heldout_error_by_population_hours():
    # The error-prone low range is common in the population; the high range is rare.
    x = np.arange(80, dtype=float).reshape(-1, 1)
    df = tm.frame(x, np.zeros(80, dtype=int), 1, ["feature"], False)
    population = tm.frame(
        np.r_[np.arange(20), np.arange(20), np.arange(80)],
        np.zeros(120, dtype=int),
        1,
        ["feature"],
        False,
    )
    y = np.ones(80, dtype=int)
    probs = np.full((80, 3), [0.8, 0.1, 0.1])
    probs[:20] = [0.2, 0.4, 0.4]
    ranges = tm.confusion_ranges(df, y, probs, population)
    assert ranges and ranges[0]["low"] == 0.0
    assert ranges[0]["population_share"] > 0.4
    assert ranges[0]["impact"] > 0


def test_undo_reconstructs_evicted_previous_clip_and_can_go_back_twice(tmp_path):
    tagger = Tagger.__new__(Tagger)
    tagger.lock = threading.RLock()
    tagger.clips = tmp_path
    tagger.labels_path = tmp_path / "labels.jsonl"
    tagger.ready = collections.deque()
    tagger.served = collections.OrderedDict([("current", {"id": "current", "row": 3})])
    tagger.history = ["first", "second"]
    tagger.ratings = {
        "first": {"row": 1, "rating": 1, "selection": "random"},
        "second": {"row": 2, "rating": 3, "selection": "active"},
    }
    made = []

    def prepare(row, selection):
        key = "first" if row == 1 else "second"
        (tmp_path / f"{key}.ogg").write_bytes(b"audio")
        made.append((row, selection))
        return {"id": key, "row": row, "selection": selection}

    tagger.prepare = prepare
    tagger.maybe_refit = lambda **kwargs: None
    second = tagger.undo("current")
    assert second["id"] == "second"
    assert made == [(2, "active")]
    assert tagger.ready[0]["id"] == "current"
    first = tagger.undo("second")
    assert first["id"] == "first"
    assert tagger.ready[0]["id"] == "second"
    assert not tagger.ratings
    assert [
        r["undo"]
        for r in map(__import__("json").loads, tagger.labels_path.read_text().splitlines())
    ] == ["second", "first"]


def test_settings_persist_and_invalid_cutoff_does_not_overwrite(tmp_path):
    import json

    tagger = Tagger.__new__(Tagger)
    tagger.lock = threading.RLock()
    tagger.settings_path = tmp_path / "settings.json"
    tagger.maybe_refit = lambda **kwargs: None
    tagger.update_settings(True, 0.65)
    assert json.loads(tagger.settings_path.read_text()) == {"use_source": True, "cutoff": 0.65}
    with pytest.raises(ValueError):
        tagger.update_settings(False, 1.0)
    assert json.loads(tagger.settings_path.read_text()) == {"use_source": True, "cutoff": 0.65}
