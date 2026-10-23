"""Build a tiny mock dataset + training configs to try the data preview server on.

    uv run python -m hebrew_training.data_preview.make_mock_data [--wav my_voice.wav]

Writes `hebrew_training/data_preview/mock/` (git-ignored):

    audio/voice.wav          copy of the given wav, used by every row
    hebrew_{train,valid}.jsonl, english_{train,valid}.jsonl
    curriculum.yaml          two datasets, two phases (hebrew only, then hebrew + english)
    single.yaml              plain train_jsonl / valid_jsonl config (no curriculum)

Paths inside the manifests and configs are relative to the repo root, like the real configs:
run the server from the repo root (it changes to it anyway). The alignments are fake (words
spread evenly over the clip) but they exercise the word-boundary cuts, and some rows carry
`alt_words` (digits) and `nikud` so the text augmentations are visible.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import sphn
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_DIR = Path(__file__).resolve().parent / "mock"

HEBREW = [
    "אדוני היושב ראש אני מבקש להעלות את הנושא לסדר היום",
    "חברי הכנסת הנכבדים מדובר בהחלטה משמעותית עבור אזרחי המדינה",
    "תודה רבה אני מסיים כאן את דבריי ומבקש את תמיכתכם",
    "מי עושה בחינה נוספת זה לא נעשה כלאחר יד",
    "אנחנו מדברים על תקציב של שלושה מיליון שקלים לשנה הבאה",
    "הוועדה תדון בהצעה בשבוע הבא ותפרסם את מסקנותיה",
]
# Spelled-out numbers: (sentence, [(first_word, last_word, digits)]), indexes into the words.
HEBREW_NUMBERS = [
    ("בשנת אלפיים עשרים וחמש נפצעו עשרים ואחד אנשים באזור", [(1, 3, "2025"), (5, 6, "21")]),
    ("הגענו למלון בשעה שלוש ועוד ארבעה עשר אנשים הצטרפו אלינו", [(3, 3, "3"), (5, 6, "14")]),
]
ENGLISH = [
    "the quick brown fox jumps over the lazy dog near the river bank",
    "she sells sea shells by the sea shore every single summer morning",
    "training is still running and this voice is a work in progress",
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood",
]

NIQQUD = "\u05b7"  # patah, appended after the first letter: enough to see vocalization in the UI


def _words(sentence: str, duration: float) -> list[dict]:
    words = sentence.split()
    start, end = 0.3, duration - 0.7
    step = (end - start) / len(words)
    return [
        {"word": w, "start": round(start + i * step, 3), "end": round(start + (i + 0.85) * step, 3)}
        for i, w in enumerate(words)
    ]


def _row(path: str, sentence: str, duration: float, *, hebrew: bool, kind: str, spans=None):
    row = {"path": path, "start": 0.0, "duration": duration, "transcript": sentence}
    if kind == "plain_transcript":  # no alignment: the loader's fallback path
        return row
    spoken = sentence
    if spans:
        # `words` is the spoken text; `transcript` is what the corpus wrote, with digits.
        row["words"] = _words(sentence, duration)
        written = row["words"][:]
        out, position = [], 0
        for first, last, digits in spans:
            out.extend(w["word"] for w in written[position:first])
            out.append(digits)
            position = last + 1
        out.extend(w["word"] for w in written[position:])
        row["transcript"] = " ".join(out)
        row["alt_words"] = {
            "digits": [{"from": f, "to": last, "replace": [d]} for f, last, d in spans]
        }
    else:
        row["words"] = _words(spoken, duration)
    if hebrew and kind in {"nikud", "numbers_nikud"}:
        row["nikud"] = [w["word"][0] + NIQQUD + w["word"][1:] for w in row["words"]]
    return row


def _dataset(
    name: str, sentences, numbers, audio_path: str, duration: float, count: int, seed: int
):
    rng = random.Random(f"{name}:{seed}")
    hebrew = name == "hebrew"
    rows = []
    for i in range(count):
        if hebrew and numbers and i % 4 == 1:
            sentence, spans = numbers[(i // 4) % len(numbers)]
            kind = "numbers_nikud" if i % 8 == 1 else "numbers"
            rows.append(_row(audio_path, sentence, duration, hebrew=hebrew, kind=kind, spans=spans))
            continue
        sentence = sentences[i % len(sentences)]
        kind = "plain_transcript" if i % 9 == 8 else rng.choice(["plain", "nikud", "plain"])
        rows.append(_row(audio_path, sentence, duration, hebrew=hebrew, kind=kind))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def build(wav: Path, tokenizer: str) -> None:
    MOCK_DIR.mkdir(parents=True, exist_ok=True)
    (MOCK_DIR / "audio").mkdir(exist_ok=True)
    voice = MOCK_DIR / "audio" / "voice.wav"
    shutil.copyfile(wav, voice)
    samples, sample_rate = sphn.read(str(voice))
    duration = round(samples.shape[-1] / sample_rate, 3)
    audio_path = _rel(voice)

    sizes = {"hebrew": (32, 12), "english": (20, 8)}
    for name, sentences, numbers in (("hebrew", HEBREW, HEBREW_NUMBERS), ("english", ENGLISH, [])):
        train, valid = sizes[name]
        _write_jsonl(
            MOCK_DIR / f"{name}_train.jsonl",
            _dataset(name, sentences, numbers, audio_path, duration, train, 1),
        )
        _write_jsonl(
            MOCK_DIR / f"{name}_valid.jsonl",
            _dataset(name, sentences, numbers, audio_path, duration, valid, 2),
        )

    common = {
        "model_config": "pocket_tts/config/english_2026-04_24l.yaml",
        "model_overrides": {"flow_lm.lookup_table.tokenizer_path": tokenizer},
        "start_from_pretrained": True,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "seed": 42,
        "run_dir": "runs/mock_preview",
    }
    data_common = {
        "max_duration_sec": 30.0,
        "max_voice_prompt_sec": 5.0,
        "alt_words_sample_prob": 0.5,
        "alt_word_swap_prob": 0.6,
        "nikud_word_prob": 0.3,
    }
    h, e = MOCK_DIR / "hebrew_{}.jsonl", MOCK_DIR / "english_{}.jsonl"
    curriculum = {
        **common,
        "data": {
            **data_common,
            "datasets": {
                "hebrew": {
                    "train_jsonl": _rel(Path(str(h).format("train"))),
                    "valid_jsonl": _rel(Path(str(h).format("valid"))),
                    "valid_weight": 0.7,
                },
                "english": {
                    "train_jsonl": _rel(Path(str(e).format("train"))),
                    "valid_jsonl": _rel(Path(str(e).format("valid"))),
                    "valid_weight": 0.3,
                },
            },
            "curriculum": [
                {"steps": 100, "mix": {"hebrew": 1}},
                {"steps": 400, "mix": {"hebrew": 0.7, "english": 0.3}},
            ],
        },
    }
    single = {
        **common,
        "max_steps": 100,
        "data": {
            **data_common,
            "train_jsonl": _rel(Path(str(h).format("train"))),
            "valid_jsonl": _rel(Path(str(h).format("valid"))),
        },
    }
    for name, config in (("curriculum", curriculum), ("single", single)):
        with (MOCK_DIR / f"{name}.yaml").open("w") as f:
            f.write("# Generated by hebrew_training/data_preview/make_mock_data.py\n")
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"wrote {MOCK_DIR} ({duration}s clip at {sample_rate} Hz)")
    print(
        f"try: uv run python -m hebrew_training.data_preview.server {_rel(MOCK_DIR)}/curriculum.yaml"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wav", type=Path, default=REPO_ROOT / "my_voice.wav")
    parser.add_argument(
        "--tokenizer",
        default="tokenizers/tokenizer.model",
        help="sentencepiece model the mock configs point at (default: the repo's English one)",
    )
    args = parser.parse_args()
    build(args.wav, args.tokenizer)


if __name__ == "__main__":
    main()
