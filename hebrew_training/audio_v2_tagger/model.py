"""Rating model and hour estimates for the audio-v2 tagger. Pure functions, no network.

Ratings are 1 (clearly not a training sample), 2 (acceptable in a bunch), 3 (clearly yes).
Random-control rows are drawn uniformly over speech HOURS (probability proportional to
duration); targeted rows are drawn from uncertain or high-error ranges. Only the random
control's share of labels >= k directly estimates the share of hours a "rating >= k" rule keeps.
The model adds two things: it explains what drives the rating
(feature importance) and, applied to a fixed hour-weighted sample of the whole population, it
estimates kept hours more tightly than the labels alone once it predicts better than the prior.

Uncertainty is reported two ways, because they answer different questions:
  direct  Wilson interval on the *random-control* labelled share.
  model   bootstrap over the labels (refit, repredict the population sample): 2.5-97.5% of the
          kept hours. Under adaptive sampling this is a heuristic, not a coverage guarantee.
          Cross-validated loss is measured on random controls once at least 30 are available.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "max_depth": 3,
    "eta": 0.15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 3.0,
    "tree_method": "hist",
    # The CUDA wheel picks the GPU by default and fails on hosts with an old driver; a few
    # hundred rows gain nothing from one.
    "device": "cpu",
    "max_cat_to_onehot": 1,
    "verbosity": 0,
    "nthread": 4,
}
ROUNDS = 60
MIN_LABELS = 20
WAV_GB_PER_HOUR = 24000 * 2 * 3600 / 1e9  # 24 kHz mono 16-bit


def frame(x: np.ndarray, source: np.ndarray, n_sources: int, names, use_source: bool):
    import pandas as pd

    df = pd.DataFrame(np.asarray(x, dtype=np.float32), columns=list(names))
    if use_source:
        df["source"] = pd.Categorical(np.asarray(source), categories=np.arange(n_sources))
    return df


def fit(df, y: np.ndarray, seed: int = 0, rounds: int = ROUNDS):
    import xgboost as xgb

    dtrain = xgb.DMatrix(df, label=np.asarray(y) - 1, enable_categorical=True)
    return xgb.train({**PARAMS, "seed": seed}, dtrain, num_boost_round=rounds)


def predict(booster, df) -> np.ndarray:
    import xgboost as xgb

    return booster.predict(xgb.DMatrix(df, enable_categorical=True)).reshape(-1, 3)


def p_at_least(probabilities: np.ndarray, k: int) -> np.ndarray:
    """P(rating >= k) for k in 1..3."""
    return probabilities[:, k - 1 :].sum(axis=1)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def cross_validate(df, y: np.ndarray, folds: int = 5, seed: int = 0, predictions=False):
    """Stratified k-fold: accuracy and log loss of the model against the class-prior baseline."""
    rng = np.random.default_rng(seed)
    n = len(y)
    fold_of = np.empty(n, dtype=int)
    for label in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == label))
        fold_of[members] = np.arange(len(members)) % folds
    probabilities = np.zeros((n, 3))
    baseline = np.zeros((n, 3))
    for fold in range(folds):
        test = fold_of == fold
        train = ~test
        if not test.any() or len(np.unique(y[train])) < 2:
            probabilities[test] = baseline[test] = 1 / 3
            continue
        booster = fit(df[train], y[train], seed=seed)
        probabilities[test] = predict(booster, df[test])
        counts = np.bincount(y[train] - 1, minlength=3) + 0.5
        baseline[test] = counts / counts.sum()
    index = np.arange(n)
    chosen = np.clip(probabilities[index, y - 1], 1e-6, 1)
    prior = np.clip(baseline[index, y - 1], 1e-6, 1)
    result = {
        "folds": folds,
        "accuracy": float((probabilities.argmax(axis=1) + 1 == y).mean()),
        "baseline_accuracy": float((baseline.argmax(axis=1) + 1 == y).mean()),
        "log_loss": float(-np.log(chosen).mean()),
        "baseline_log_loss": float(-np.log(prior).mean()),
        "mae": float(np.abs((probabilities * np.array([1, 2, 3])).sum(axis=1) - y).mean()),
    }
    return (result, probabilities) if predictions else result


def confusion_ranges(
    df, y: np.ndarray, probabilities: np.ndarray, population_df=None
) -> list[dict]:
    """Rank elevated held-out error by the fraction of population hours it affects."""
    errors = 1 - probabilities[np.arange(len(y)), y - 1]
    baseline = float(errors.mean())
    ranges = []
    for name in df.select_dtypes(include=[np.number]).columns:
        values = df[name].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 32:
            continue
        edges = np.unique(np.quantile(values[finite], [0, 0.25, 0.5, 0.75, 1]))
        for lo, hi in pairwise(edges):
            mask = finite & (values >= lo) & (values <= hi if hi == edges[-1] else values < hi)
            count = int(mask.sum())
            if count < 8:
                continue
            error = float(errors[mask].mean())
            # Shrink tiny bins toward the population mean to avoid noisy hot spots.
            excess = max(0.0, error - baseline) * count / (count + 15)
            if excess:
                population_share = (
                    float(((population_df[name] >= lo) & (population_df[name] <= hi)).mean())
                    if population_df is not None
                    else count / len(y)
                )
                ranges.append(
                    {
                        "feature": name,
                        "low": float(lo),
                        "high": float(hi),
                        "n": count,
                        "error": error,
                        "excess": excess,
                        "population_share": population_share,
                        "impact": excess * population_share,
                    }
                )
    return sorted(ranges, key=lambda r: -r["impact"])[:8]


def active_scores(
    probabilities: np.ndarray, features, ranges: list[dict], cutoff: float = 0.5
) -> np.ndarray:
    """Score population-drawn candidates by model uncertainty and CV-error hot spots."""
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-8, 1))).sum(axis=1) / math.log(3)
    # A small change in prediction near either keep/reject boundary changes kept hours.
    p2, p3 = p_at_least(probabilities, 2), p_at_least(probabilities, 3)
    boundary = np.maximum(np.exp(-np.abs(p2 - cutoff) / 0.15), np.exp(-np.abs(p3 - cutoff) / 0.15))
    hotspots = np.zeros(len(probabilities))
    for r in ranges[:5]:
        values = features[r["feature"]].to_numpy(dtype=float)
        hotspots += (np.isfinite(values) & (values >= r["low"]) & (values <= r["high"])) * min(
            r["excess"], 0.5
        )
    return 0.10 + 0.5 * entropy + 0.35 * boundary + hotspots


def analyze(
    df,
    y: np.ndarray,
    population_df,
    total_hours: float,
    cutoff: float = 0.5,
    bootstrap: int = 20,
    seed: int = 0,
    control_mask: np.ndarray | None = None,
) -> dict:
    """Everything the UI shows. `population_df` is an hour-weighted sample of every row."""
    y = np.asarray(y)
    n = len(y)
    control_mask = (
        np.ones(n, dtype=bool) if control_mask is None else np.asarray(control_mask, dtype=bool)
    )
    control = y[control_mask]
    counts = {k: int((y == k).sum()) for k in (1, 2, 3)}
    result: dict = {
        "n": n,
        "counts": counts,
        "total_hours": total_hours,
        "cutoff": cutoff,
        "direct": {},
        "model": None,
        "ready": False,
        "random_n": len(control),
        "active_n": n - len(control),
    }
    for k in (2, 3):
        hits = int((control >= k).sum())
        low, high = wilson(hits, len(control))
        share = hits / len(control) if len(control) else 0.0
        result["direct"][k] = _row(share, low, high, total_hours)

    classes_present = sum(1 for c in counts.values() if c >= 3)
    if n < MIN_LABELS or classes_present < 2:
        result["note"] = (
            f"need {MIN_LABELS}+ ratings with at least two classes rated 3+ times to fit "
            f"({n} so far)"
        )
        return result

    booster = fit(df, y, seed=seed)
    probabilities = predict(booster, population_df)
    model = {}
    shares = {k: float((p_at_least(probabilities, k) >= cutoff).mean()) for k in (2, 3)}

    rng = np.random.default_rng(seed + 1)
    boot = {2: [], 3: []}
    for b in range(bootstrap):
        pick = rng.integers(0, n, n)
        if len(np.unique(y[pick])) < 2:
            continue
        fitted = fit(df.iloc[pick], y[pick], seed=seed + b)
        probs = predict(fitted, population_df)
        for k in (2, 3):
            boot[k].append(float((p_at_least(probs, k) >= cutoff).mean()))
    for k in (2, 3):
        low, high = (
            (float(np.quantile(boot[k], 0.025)), float(np.quantile(boot[k], 0.975)))
            if len(boot[k]) >= 5
            else (0.0, 1.0)
        )
        model[k] = _row(shares[k], low, high, total_hours)

    importance = booster.get_score(importance_type="gain")
    result["model"] = model
    result["importance"] = sorted(importance.items(), key=lambda kv: -kv[1])[:12]
    # Only the random stream estimates population generalization without sampling bias.
    eval_mask = (
        control_mask
        if len(control) >= 30 and len(np.unique(control)) >= 2
        else np.ones(n, dtype=bool)
    )
    result["cv_population_holdout"] = bool(np.array_equal(eval_mask, control_mask))
    result["cv"], heldout = cross_validate(
        df[eval_mask],
        y[eval_mask],
        folds=5 if eval_mask.sum() >= 30 else 3,
        seed=seed,
        predictions=True,
    )
    result["confusion_ranges"] = confusion_ranges(
        df[eval_mask], y[eval_mask], heldout, population_df
    )
    result["ready"] = True
    return result


def _row(share: float, low: float, high: float, total_hours: float) -> dict:
    return {
        "share": share,
        "hours": share * total_hours,
        "hours_low": low * total_hours,
        "hours_high": high * total_hours,
        "margin_hours": (high - low) / 2 * total_hours,
        "wav_tb": share * total_hours * WAV_GB_PER_HOUR / 1000,
    }
