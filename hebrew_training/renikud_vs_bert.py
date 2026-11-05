"""ReNikud against DictaBERT-large on the same 51 word-sense decisions.

    python renikud_vs_bert.py <renikud dir> <anchors.json> <out.txt>

The two answer in different currencies, so each is read in its own:

  DictaBERT  is an encoder. It gets the nearest-centroid probe, anchors scored
             leave-one-out, exactly as in docs/homograph-disambiguation.md.
  ReNikud    is grapheme-to-phoneme. It states the answer outright -- the IPA it emits for
             the word *is* the reading -- so it needs no anchors and gets no help from them.
             RULES below turn its IPA back into a sense label.

ReNikud therefore competes on strictly harder terms: it was never shown an example, while
the probe is handed three labelled sentences per sense.
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

BERT = "dicta-il/dictabert-large"
# IPA substring -> sense. First match wins; "" means no reading could be read off.
RULES = {
    "sefer": [("sap", "barber"), ("ef", "book")],
    "rotze": [("ʁotsˈa", "feminine"), ("ʁotsˈe", "masculine")],
    "ochel": [("uχ", "able"), ("oχ", "eating")],
    "more": [("ʁˈa", "feminine"), ("ʁˈe", "masculine")],
    "boker": [("bˈoke", "morning"), ("bokˈe", "cowboy")],
    "dover": [("dub", "discussed"), ("dovˈe", "spokesman")],
    "melech": [("malˈa", "reigned"), ("mˈele", "king")],
    "sefer_bare": [("sap", "barber"), ("ef", "book")],
    "lamore": [("ʁˈa", "feminine"), ("ʁˈe", "masculine")],
    "boker_bare": [("bˈoke", "morning"), ("bokˈe", "cowboy")],
}

folder, anchors_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(folder))
from renikud_onnx import G2P  # noqa: E402

g2p = G2P(str(folder / "model.onnx"))
data = json.loads(anchors_path.read_text(encoding="utf-8"))["homographs"]

tokenizer = AutoTokenizer.from_pretrained(BERT)
bert = AutoModel.from_pretrained(BERT).eval()


def vector(sentence: str, word_index: int) -> torch.Tensor:
    words = sentence.split()
    start = len(" ".join(words[:word_index])) + (1 if word_index else 0)
    stop = start + len(words[word_index])
    encoded = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    with torch.inference_mode():
        hidden = bert(**encoded, output_hidden_states=True).hidden_states[-1][0]
    rows = [i for i, (a, b) in enumerate(offsets) if a != b and a < stop and b > start]
    return F.normalize(hidden[rows].mean(0), dim=-1)


def renikud_pick(word: str, sentence: str, index: int) -> str:
    pieces = g2p.phonemize(sentence).split()
    got = pieces[index] if index < len(pieces) else ""
    for needle, sense in RULES[word]:
        if needle in got:
            return sense
    return "?"


rows = []
for word, entry in data.items():
    anchors = entry["anchors"]
    vectors = [vector(s, i) for s, i, _ in anchors]
    senses = [s for _, _, s in anchors]
    for held_out, (sentence, index, truth) in enumerate(anchors):
        centroids = {
            s: torch.stack(
                [v for j, v in enumerate(vectors) if senses[j] == s and j != held_out]
            ).mean(0)
            for s in entry["senses"]
        }
        rows.append(
            {
                "word": word,
                "kind": "anchor",
                "truth": truth,
                "text": sentence,
                "bert": max(
                    centroids,
                    key=lambda s: float(
                        F.cosine_similarity(vectors[held_out], centroids[s], dim=-1)
                    ),
                ),
                "renikud": renikud_pick(word, sentence, index),
            }
        )
    centroids = {
        s: torch.stack([v for j, v in enumerate(vectors) if senses[j] == s]).mean(0)
        for s in entry["senses"]
    }
    for sentence, index, truth in entry["tests"]:
        v = vector(sentence, index)
        rows.append(
            {
                "word": word,
                "kind": "test",
                "truth": truth,
                "text": sentence,
                "bert": max(
                    centroids, key=lambda s: float(F.cosine_similarity(v, centroids[s], dim=-1))
                ),
                "renikud": renikud_pick(word, sentence, index),
            }
        )

lines = [
    f"A = {BERT}  (centroid probe, 3 labelled anchors per sense)",
    "B = renikud/renikud  (g2p, zero-shot, no anchors)",
]
for kind in ("anchor", "test", None):
    group = [r for r in rows if kind is None or r["kind"] == kind]
    n = len(group)
    agree = [r for r in group if r["bert"] == r["renikud"]]
    split = [r for r in group if r["bert"] != r["renikud"]]
    oracle = sum(r["bert"] == r["truth"] for r in agree) + len(split)
    lines += [
        f"\n--- {kind or 'ALL'}  ({n} decisions) ---",
        f"  A correct          {sum(r['bert'] == r['truth'] for r in group):3d}/{n}",
        f"  B correct          {sum(r['renikud'] == r['truth'] for r in group):3d}/{n}",
        f"  agree              {len(agree):3d}/{n} ({100 * len(agree) / n:.0f}%), "
        f"{sum(r['bert'] == r['truth'] for r in agree)} of them correct",
        f"  DISAGREE           {len(split):3d}/{n} ({100 * len(split) / n:.0f}%)",
        f"    A right            {sum(r['bert'] == r['truth'] for r in split):3d}",
        f"    B right            {sum(r['renikud'] == r['truth'] for r in split):3d}",
        f"    neither            {sum(r['bert'] != r['truth'] and r['renikud'] != r['truth'] for r in split):3d}",
        f"  oracle gate        {oracle:3d}/{n} ({100 * oracle / n:.0f}%)",
    ]

lines.append("\nper homograph (A / B correct):")
for word in data:
    group = [r for r in rows if r["word"] == word]
    lines.append(
        f"  {word:8s} {sum(r['bert'] == r['truth'] for r in group)}/{len(group)}"
        f"   {sum(r['renikud'] == r['truth'] for r in group)}/{len(group)}"
    )

lines.append("\nPER WORD on the validation sentences:")
for r in rows:
    if r["kind"] != "test":
        continue
    a_ok = "   " if r["bert"] == r["truth"] else " X "
    b_ok = "   " if r["renikud"] == r["truth"] else " X "
    lines.append(
        f"  truth={r['truth']:11s} dictabert={r['bert']:11s}{a_ok}renikud={r['renikud']:11s}{b_ok}"
    )
    lines.append(f"      {r['text']}")

lines.append("\ndisagreements:")
for r in rows:
    if r["bert"] != r["renikud"]:
        who = "A" if r["bert"] == r["truth"] else ("B" if r["renikud"] == r["truth"] else "neither")
        lines.append(
            f"  [{r['kind']:6s}] {r['word']:7s} truth={r['truth']:10s} "
            f"A={r['bert']:10s} B={r['renikud']:10s} -> {who}\n      {r['text']}"
        )

text = "\n".join(lines)
out_path.write_text(text + "\n", encoding="utf-8")
sys.stdout.buffer.write(text.encode("utf-8") + b"\n")
