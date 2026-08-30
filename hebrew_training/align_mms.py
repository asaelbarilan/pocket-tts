"""Align a manifest with Meta's MMS forced-aligner, as the second opinion.

This is the B column of the alignment experiment. The A column is
`data_prep/align_hebrew.py` (imvladikon/wav2vec2-xls-r-300m-hebrew, CTC over native Hebrew
letters), and the C column is whatever timings the corpus already ships. The point of B is
that it shares as little as possible with A: different training data, different alphabet,
different team.

MMS aligns over **romanized Latin** -- its vocabulary is 26 letters plus an apostrophe -- so
Hebrew text goes through uroman first. Worth seeing what that means concretely:

    שלום עולם   ->   shlvm 'vlm

Unvocalized Hebrew romanizes to its consonant skeleton, because the vowels are not written.
Both CTC options face that; it is a ceiling on the whole approach, not a fault of this one.

The upstream `ctc-forced-aligner` package wraps the same model, but it ships a C++ extension
that does not build on Windows. This calls `torchaudio.functional.forced_align` directly,
which is the same Viterbi the package uses and needs no compiler.

    python -m hebrew_training.align_mms --manifest clips.jsonl --out clips_mms.jsonl

Rows keep `path`, `start` and `duration` untouched so the output can be compared against
another aligner with `alignment_disagreement.py`, which keys on (path, start).
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

MODEL = "MahmoudAshraf/mms-300m-1130-forced-aligner"
SAMPLE_RATE = 16000
_WS = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cuda is faster but this model is small; cpu is fine for a few hundred clips.",
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def source_words(row: dict) -> list[str]:
    """The words to align, preferring the existing word list over the transcript.

    When a row already carries `words`, those are the tokens the other aligners used, and
    reusing them keeps the three columns word-for-word comparable. Falling back to splitting
    `transcript` would risk a different tokenization and make the comparison meaningless.
    """
    words = [clean(w["word"]) for w in (row.get("words") or []) if clean(w.get("word", ""))]
    return words or clean(row.get("transcript", "")).split()


def romanize_words(words: list[str], uroman, vocab: dict[str, int]) -> list[list[int]]:
    """Token ids per word; an empty list marks a word that cannot be represented.

    Romanizing word by word rather than the whole line keeps the word boundaries known --
    romanizing the line and re-splitting would silently drift if uroman merged or split
    anything.
    """
    out = []
    for word in words:
        roman = uroman.romanize_string(word).lower()
        ids = [vocab[c] for c in roman if c in vocab]
        out.append(ids)
    return out


def main() -> None:
    args = parse_args()
    import torch
    import torchaudio.functional as AF
    import uroman as uroman_module
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    from training.dataloader import _load_window

    print(f"loading {args.model} on {args.device} ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(args.device).eval()
    vocab = {k: v for k, v in processor.tokenizer.get_vocab().items() if len(k) == 1}
    blank = processor.tokenizer.get_vocab().get("<blank>", 0)
    romanizer = uroman_module.Uroman()

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows, 1):
            words = source_words(row)
            token_ids = romanize_words(words, romanizer, vocab)
            keep = [i for i, ids in enumerate(token_ids) if ids]
            if not keep:
                skipped += 1
                continue

            wav = _load_window(
                row["path"], float(row.get("start", 0.0)), float(row["duration"]), SAMPLE_RATE
            )
            audio = torch.from_numpy(wav).float()[None].to(args.device)
            with torch.inference_mode():
                logits = model(audio).logits
                emission = torch.log_softmax(logits, dim=-1)

            flat = [t for i in keep for t in token_ids[i]]
            targets = torch.tensor([flat], dtype=torch.int32, device=args.device)
            try:
                aligned, scores = AF.forced_align(emission, targets, blank=blank)
            except Exception as exc:  # noqa: BLE001 -- one unalignable row must not stop the run
                print(f"  row {index}: {type(exc).__name__} {exc}", flush=True)
                skipped += 1
                continue
            spans = AF.merge_tokens(aligned[0], scores[0].exp())

            # emission frames -> seconds, relative to the row's start, matching the schema
            # the other aligners write.
            ratio = audio.shape[-1] / emission.shape[1] / SAMPLE_RATE
            timed, cursor = [], 0
            for i in keep:
                count = len(token_ids[i])
                chunk = spans[cursor : cursor + count]
                cursor += count
                if not chunk:
                    continue
                timed.append(
                    {
                        "word": words[i],
                        "start": round(chunk[0].start * ratio, 4),
                        "end": round(chunk[-1].end * ratio, 4),
                        "score": round(float(sum(s.score for s in chunk) / len(chunk)), 4),
                    }
                )
            handle.write(json.dumps({**row, "words": timed}, ensure_ascii=False) + "\n")
            done += 1
            if index % 10 == 0 or index == len(rows):
                print(f"  {index}/{len(rows)} aligned ({skipped} skipped)", flush=True)

    print(f"\naligned {done}, skipped {skipped} -> {args.out}")


if __name__ == "__main__":
    main()
