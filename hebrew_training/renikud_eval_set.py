"""Run ReNikud over the whole validation set and flag what it cannot handle.

    python -m hebrew_training.renikud_eval_set <renikud folder> \
        hebrew_training/eval_sentences_he.jsonl out.txt

There is no gold IPA for most of the forty sentences, so this does not score them. It
checks the three things that can be checked without one, each of which is a broken string
rather than a debatable pronunciation:

    HEBREW-LEFT        a Hebrew character survived into the IPA. Always a failure, and the
                       way vocalized input fails -- the nikud marks pass straight through.
    DIGITS-LEFT        a digit survived, meaning the number front end did not fire.
    latin-passthrough  Latin went in and came out unchanged. Not a failure: ReNikud
                       phonemizes Hebrew and leaves the rest alone. Recorded so that what
                       reaches the TTS is a decision rather than an accident.

The ReNikud folder must hold `model.onnx` (from `renikud/renikud`) and `renikud_onnx.py`
(from `niobures/ReNikud`, under `models/renikud-plus/`). Neither is vendored here.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HEBREW = re.compile(r"[֐-׿]")
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9]")


def main() -> None:
    folder, sentences, out_path = (Path(a) for a in sys.argv[1:4])
    sys.path.insert(0, str(folder))
    from renikud_onnx import G2P

    g2p = G2P(str(folder / "model.onnx"))
    rows = [
        json.loads(line)
        for line in sentences.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    out, failures = [], 0
    for row in rows:
        ipa = g2p.phonemize(row["text"])
        flags = []
        if HEBREW.search(ipa):
            flags.append("HEBREW-LEFT")
        if DIGIT.search(ipa):
            flags.append("DIGITS-LEFT")
        if LATIN.search(row["text"]) and LATIN.search(ipa):
            flags.append("latin-passthrough")
        failures += any(f.isupper() for f in flags)
        out.append(f"[{row['category']:22s}] {row['id']:10s} {' '.join(flags) or 'ok'}")
        out.append(f"   in  {row['text']}")
        out.append(f"   ipa {ipa}")
        if row.get("text_nikud"):
            # The vocalized twin of a homograph, which is the case that breaks it.
            out.append(f"   nik {g2p.phonemize(row['text_nikud'])}")

    out.append(f"\n{failures} of {len(rows)} sentences came back with Hebrew or digits left in.")
    text = "\n".join(out)
    out_path.write_text(text + "\n", encoding="utf-8")
    # Not print(): a Windows console is cp1252 and would raise on the IPA.
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")


if __name__ == "__main__":
    main()
