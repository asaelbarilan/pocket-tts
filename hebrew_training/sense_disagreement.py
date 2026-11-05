"""How often do two semantic encoders disagree, and when they do, which one is right?

    .venv/Scripts/python.exe -m hebrew_training.sense_disagreement \
        --anchors hebrew_training/sense_anchors_he.json \
        --models dicta-il/dictabert-large dicta-il/dictabert-large-char

The question this answers is the one that decides whether combining two encoders is worth
any architecture at all. If they agree everywhere, a gate has nothing to learn and the
second model is dead weight. If they disagree often and one is simply better, drop the
other. A gate only pays when they disagree *and* neither is uniformly right -- and then
the disagreement rate is also the fraction of tokens the gate ever gets gradient from.

Both models are asked the same question in the same way, so the comparison is fair: the
nearest-centroid word-sense probe from docs/homograph-disambiguation.md. Each model builds
its own centroids in its own space -- the two spaces are never compared directly, which
would be meaningless for independently trained encoders.

Every anchor is scored leave-one-out, so the anchors contribute real decisions rather than
being memorised, and the eval-set sentences are scored against all anchors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--anchors", type=Path, required=True)
    p.add_argument("--models", nargs=2, required=True)
    p.add_argument("--out", type=Path)
    return p.parse_args()


class Prober:
    """One encoder, and the contextual vector it gives a chosen word of a sentence."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).eval()

    def vector(self, sentence: str, word_index: int) -> torch.Tensor:
        # Character offsets, so the same code works for a wordpiece model and a character
        # model: whatever the units are, the ones overlapping this word are the word.
        words = sentence.split()
        start = len(" ".join(words[:word_index])) + (1 if word_index else 0)
        stop = start + len(words[word_index])
        encoded = self.tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.inference_mode():
            hidden = self.model(**encoded, output_hidden_states=True).hidden_states[-1][0]
        rows = [i for i, (a, b) in enumerate(offsets) if a != b and a < stop and b > start]
        return F.normalize(hidden[rows].mean(0), dim=-1)


def pick(vector: torch.Tensor, centroids: dict[str, torch.Tensor]) -> str:
    return max(centroids, key=lambda s: float(F.cosine_similarity(vector, centroids[s], dim=-1)))


def main() -> None:
    args = parse_args()
    data = json.loads(args.anchors.read_text(encoding="utf-8"))["homographs"]
    probers = [Prober(name) for name in args.models]

    decisions: list[dict] = []
    for word, entry in data.items():
        anchors = entry["anchors"]
        vectors = [[p.vector(s, i) for s, i, _ in anchors] for p in probers]
        senses = [sense for _, _, sense in anchors]

        for held_out, (sentence, index, truth) in enumerate(anchors):
            row = {"word": word, "kind": "anchor", "text": sentence, "truth": truth}
            for p, group in zip(probers, vectors):
                centroids = {
                    s: torch.stack(
                        [v for j, v in enumerate(group) if senses[j] == s and j != held_out]
                    ).mean(0)
                    for s in entry["senses"]
                }
                row[p.name] = pick(group[held_out], centroids)
            decisions.append(row)

        for sentence, index, truth in entry["tests"]:
            row = {"word": word, "kind": "test", "text": sentence, "truth": truth}
            for p, group in zip(probers, vectors):
                centroids = {
                    s: torch.stack([v for j, v in enumerate(group) if senses[j] == s]).mean(0)
                    for s in entry["senses"]
                }
                row[p.name] = pick(p.vector(sentence, index), centroids)
            decisions.append(row)

    a, b = (p.name for p in probers)
    lines = [f"A = {a}\nB = {b}"]
    for kind in ("anchor", "test", None):
        group = [d for d in decisions if kind is None or d["kind"] == kind]
        n = len(group)
        agree = [d for d in group if d[a] == d[b]]
        split = [d for d in group if d[a] != d[b]]
        label = kind or "ALL"
        lines.append(f"\n--- {label}  ({n} decisions) ---")
        lines.append(f"  A correct            {sum(d[a] == d['truth'] for d in group):3d}/{n}")
        lines.append(f"  B correct            {sum(d[b] == d['truth'] for d in group):3d}/{n}")
        lines.append(
            f"  agree                {len(agree):3d}/{n}  "
            f"({100 * len(agree) / n:.0f}%), of those "
            f"{sum(d[a] == d['truth'] for d in agree)} correct"
        )
        lines.append(f"  DISAGREE             {len(split):3d}/{n}  ({100 * len(split) / n:.0f}%)")
        if split:
            lines.append(f"    A right, B wrong   {sum(d[a] == d['truth'] for d in split):3d}")
            lines.append(f"    B right, A wrong   {sum(d[b] == d['truth'] for d in split):3d}")
            # The ceiling a perfect gate could reach: every agreement it must accept, every
            # disagreement it could in principle resolve correctly.
            best = sum(d[a] == d["truth"] for d in agree) + len(split)
            lines.append(
                f"    oracle gate        {best:3d}/{n} ({100 * best / n:.0f}%) "
                f"-- the most any combiner could score"
            )
    lines.append("\ndisagreements in detail:")
    for d in decisions:
        if d[a] != d[b]:
            winner = "A" if d[a] == d["truth"] else ("B" if d[b] == d["truth"] else "neither")
            lines.append(
                f"  [{d['kind']:6s}] {d['word']:7s} truth={d['truth']:10s} "
                f"A={d[a]:10s} B={d[b]:10s} -> {winner}\n      {d['text']}"
            )

    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    # Not print(): a Windows console is cp1252 and raises on the Hebrew, which would throw
    # away a report that is already computed.
    sys.stdout.buffer.write(text.encode("utf-8") + b"\n")


if __name__ == "__main__":
    main()
