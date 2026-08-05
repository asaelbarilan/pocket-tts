from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

# Build one fixed evaluation set used to score every checkpoint.
#
# The old set was 4 sentences x 3 speakers = 12 clips, which could not separate models:
# swings of 0.1-0.3 WER between adjacent checkpoints were pure sampling noise. This builds
# a larger set and, critically, freezes it — the same sentences and the same voice prompts
# for every model, forever, or the numbers are not comparable.
#
# Sentences are taken from held-out validation transcripts rather than written by hand, so
# they match the real text distribution. Every candidate is checked against the training
# corpora to make sure it never appeared in training.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed WER evaluation set.")
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True,
                        help="Artifact dirs; the first supplies the voice prompts.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sentences", type=int, default=40)
    parser.add_argument("--speakers", type=int, default=10)
    parser.add_argument("--min-chars", type=int, default=45)
    parser.add_argument("--max-chars", type=int, default=110)
    parser.add_argument("--prompt-min-seconds", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    def load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []

    val_rows, train_rows, val_speaker_sets, train_speakers = [], [], [], set()
    for art in args.artifacts:
        v = load(art / "validation.jsonl")
        t = load(art / "train.jsonl")
        val_rows.append(v)
        train_rows.extend(t)
        val_speaker_sets.append({r["speaker_id"] for r in v})
        train_speakers |= {r["speaker_id"] for r in t}

    # A speaker only counts as held out if no dataset trained on them.
    held_out = set.intersection(*val_speaker_sets) - train_speakers
    if len(held_out) < args.speakers:
        raise SystemExit(f"only {len(held_out)} speakers are held out everywhere")

    # One prompt clip per speaker: the longest available, so the voice is well conditioned.
    prompts = {}
    for rows in val_rows:
        for r in rows:
            s = r["speaker_id"]
            if s not in held_out or r["duration"] < args.prompt_min_seconds:
                continue
            if not Path(r["audio_path"]).exists():
                continue
            if s not in prompts or r["duration"] > prompts[s]["duration"]:
                prompts[s] = r
    usable = sorted(prompts, key=lambda s: -prompts[s]["duration"])[: args.speakers]
    if len(usable) < args.speakers:
        raise SystemExit(f"only {len(usable)} held-out speakers have a usable prompt wav")

    # Candidate sentences from held-out transcripts, rejected if they appear in any
    # training text. Substring, not equality: our long clips concatenate sentences.
    train_blob = "\n".join(clean(r["text"]) for r in train_rows)
    seen, candidates = set(), []
    for rows in val_rows:
        for r in rows:
            for sentence in _SENT_SPLIT.split(r["text"]):
                s = clean(sentence)
                if not (args.min_chars <= len(s) <= args.max_chars):
                    continue
                if re.search(r"[A-Za-z0-9]", s):        # keep the set purely Hebrew
                    continue
                if not re.search(r"[֐-׿]", s):
                    continue
                if s in seen or s in train_blob:
                    continue
                seen.add(s)
                candidates.append(s)
    if len(candidates) < args.sentences:
        raise SystemExit(f"only {len(candidates)} usable sentences found")
    rng.shuffle(candidates)
    sentences = sorted(candidates[: args.sentences])

    payload = {
        "sentences": sentences,
        "speakers": [
            {
                "speaker_id": s,
                "prompt_audio_path": prompts[s]["audio_path"],
                "prompt_duration": round(prompts[s]["duration"], 2),
            }
            for s in usable
        ],
        "clips_per_checkpoint": len(sentences) * len(usable),
        "built_from": [str(a) for a in args.artifacts],
        "seed": args.seed,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(sentences)} sentences x {len(usable)} speakers "
          f"= {payload['clips_per_checkpoint']} clips per checkpoint")
    print(f"held-out speakers available: {len(held_out)}, used {len(usable)}")
    print(f"sentence length: {min(len(s) for s in sentences)}-{max(len(s) for s in sentences)} chars")
    print(f"wrote {args.output}")
    for s in sentences[:3]:
        print("  ", s)


if __name__ == "__main__":
    main()
