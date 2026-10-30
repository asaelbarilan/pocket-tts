"""Does the Dicta diacritizer pick the right reading of a Hebrew homograph?

Runs dicta-il/dictabert-large-char-menaked over the seven homograph sentences in
eval_sentences_he.jsonl and prints what it put on the ambiguous words, next to the reading
the sentence is supposed to force. Judgement is by eye on seven rows -- small enough that a
string comparison would only hide the ktiv-male / ktiv-menukad spelling difference.

    .venv/Scripts/python.exe probe_homographs.py <jsonl> <out.txt>
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, PreTrainedTokenizerFast

MODEL = "dicta-il/dictabert-large-char-menaked"

# The word positions that carry the ambiguity, and what each should come out as.
TARGETS = {
    "homo_01": [(0, "ha-SAPAR, barber"), (5, "SEFER, book")],
    "homo_02": [(1, "rotzA, feminine"), (5, "rotzE, masculine")],
    "homo_03": [(2, "uCHAL, I will be able")],
    "homo_04": [(0, "ha-morA, feminine"), (3, "la-morE, masculine")],
    "homo_05": [(0, "BOker, morning"), (3, "boKER, cowboy")],
    "homo_06": [(0, "doVER, spokesman"), (4, "duBAR, was discussed")],
    "homo_07": [(0, "ha-MElech, king"), (1, "maLACH, reigned")],
}

rows = [json.loads(x) for x in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if x.strip()]
rows = [r for r in rows if r["id"] in TARGETS]

# PreTrainedTokenizerFast, not AutoTokenizer: see build_tokenizer_corpus.diacritize.
tok = PreTrainedTokenizerFast.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).eval()
with torch.inference_mode():
    predicted = model.predict([r["text"] for r in rows], tok)

out = [f"{MODEL}\n"]
for row, marked in zip(rows, predicted):
    out.append(f"\n=== {row['id']} -- {row.get('note', '')}")
    out.append(f"  in      {row['text']}")
    out.append(f"  dicta   {marked}")
    out.append(f"  gold    {row['text_nikud']}")
    words = marked.split()
    for position, want in TARGETS[row["id"]]:
        got = words[position] if position < len(words) else "<missing>"
        out.append(f"    word {position}: dicta={got}   should be {want}")
Path(sys.argv[2]).write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"{len(rows)} sentences -> {sys.argv[2]}")
