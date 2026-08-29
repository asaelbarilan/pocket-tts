from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Callable

_PUNCT = re.compile(r"[^\w\s\u0590-\u05ff]", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalize_hebrew_for_asr(text: str) -> str:
    """Normalize references and ASR hypotheses identically before WER/CER."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(character for character in text if not "\u0591" <= character <= "\u05c7")
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def bootstrap_interval(
    references: list[str],
    hypotheses: list[str],
    metric: Callable[[list[str], list[str]], float],
    *,
    samples: int = 1_000,
    seed: int = 1337,
) -> tuple[float, float]:
    """Return a deterministic 95% clip-bootstrap confidence interval."""
    if len(references) != len(hypotheses) or not references:
        raise ValueError("references and hypotheses must be non-empty and have equal length")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = random.Random(seed)
    count = len(references)
    values = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        values.append(
            metric(
                [references[index] for index in indices], [hypotheses[index] for index in indices]
            )
        )
    values.sort()
    low = values[int(0.025 * (samples - 1))]
    high = values[int(0.975 * (samples - 1))]
    return float(low), float(high)


def score_transcripts(
    references: list[str],
    hypotheses: list[str],
    *,
    bootstrap_samples: int = 1_000,
    seed: int = 1337,
) -> dict:
    import jiwer

    normalized_references = [normalize_hebrew_for_asr(text) for text in references]
    normalized_hypotheses = [normalize_hebrew_for_asr(text) for text in hypotheses]
    if not normalized_references or any(not reference for reference in normalized_references):
        raise ValueError("every normalized reference must be non-empty")
    wer = float(jiwer.wer(normalized_references, normalized_hypotheses))
    cer = float(jiwer.cer(normalized_references, normalized_hypotheses))
    wer_low, wer_high = bootstrap_interval(
        normalized_references,
        normalized_hypotheses,
        jiwer.wer,
        samples=bootstrap_samples,
        seed=seed,
    )
    cer_low, cer_high = bootstrap_interval(
        normalized_references,
        normalized_hypotheses,
        jiwer.cer,
        samples=bootstrap_samples,
        seed=seed + 1,
    )
    return {
        "wer": wer,
        "wer_ci95": [wer_low, wer_high],
        "cer": cer,
        "cer_ci95": [cer_low, cer_high],
        "clips": len(normalized_references),
        "empty_outputs": sum(not hypothesis for hypothesis in normalized_hypotheses),
    }
