"""Does dicta-il/dictabert-large represent ספר-the-barber differently from ספר-the-book?

dictabert-large is a wordpiece model with no nikud head, so it cannot be asked the question
directly the way the menaked model can. Two probes that do not need it to:

  fill  -- mask the ambiguous word and read the top predictions. If the model has understood
           that the subject of "___ סיפר לי" is a person, it fills the slot with people.
           This asks what the model expects, in its own vocabulary, with no anchors to argue
           about.

  sense -- the nearest-centroid word-sense test. Take the contextual vector of ספר in
           sentences where the sense is beyond doubt, average each sense into a centroid,
           and see which centroid the test sentence's vector is closer to. This asks the
           sharper question: is the distinction *present* in the encoder, whether or not
           the model can name it. If it is, a small head can read it off; if it is not,
           nothing downstream will recover it.

The sense probe is validated leave-one-out over the anchors before it is believed. A probe
that cannot classify sentences whose answer is written on them cannot be trusted on one
whose answer is the open question.

    .venv/Scripts/python.exe probe_sense.py <out.txt>
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL = "dicta-il/dictabert-large"
WORD = "ספר"

# Sentences where only one reading is possible. The target word is written ה+ספר in all of
# them so the probe compares like with like.
ANCHORS = {
    "barber": [
        "הספר גזר לי את השיער קצר מדי.",
        "קבעתי תור אצל הספר ליום ראשון בבוקר.",
        "הספר במספרה המליץ על תספורת אחרת.",
        "הספר החזיק את המספריים ביד רועדת.",
    ],
    "book": [
        "הספר מונח על המדף בספרייה.",
        "קראתי את הספר פעמיים ולא הבנתי את הסוף.",
        "הספר הזה יצא לאור בשנה שעברה.",
        "הספר נפתח בעמוד הראשון ונקרע.",
    ],
}
# The sentence the eval set actually asks about, and a minimal control that means the book.
TESTS = {
    "homo_01 (barber intended)": "הספר סיפר לי שהוא קרא ספר על תספורות.",
    "control (book intended)": "הספר סיפר לי על חייו של המלך.",
}

out: list[str] = [MODEL]
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForMaskedLM.from_pretrained(MODEL).eval()


def target_span(sentence: str) -> tuple[int, int]:
    """The token indices covering the first ה+ספר, found by character offset.

    Offsets, not a token-id search: ספר may be one wordpiece or two, and with the ה prefix
    it is a different id again. The characters are the only stable handle.
    """
    start = sentence.index(WORD)
    encoded = tokenizer(sentence, return_offsets_mapping=True)
    covering = [
        i
        for i, (a, b) in enumerate(encoded["offset_mapping"])
        if a != b and a < start + len(WORD) and b > start
    ]
    return covering[0], covering[-1] + 1


def word_vector(sentence: str) -> torch.Tensor:
    """The contextual vector of ספר: last hidden layer, averaged over its wordpieces."""
    batch = tokenizer(sentence, return_tensors="pt")
    with torch.inference_mode():
        hidden = model(**batch, output_hidden_states=True).hidden_states[-1][0]
    first, last = target_span(sentence)
    return F.normalize(hidden[first:last].mean(0), dim=-1)


# --- fill probe ------------------------------------------------------------------------
out.append("\n\n=== fill: mask the word, read the top 10 ===")
for label, sentence in TESTS.items():
    first, last = target_span(sentence)
    ids = tokenizer(sentence, return_tensors="pt")
    ids["input_ids"][0, first:last] = tokenizer.mask_token_id
    with torch.inference_mode():
        logits = model(**ids).logits[0, first]
    top = logits.topk(10).indices.tolist()
    out.append(f"\n{label}\n  {sentence}")
    out.append("  " + ", ".join(tokenizer.convert_ids_to_tokens(top)))

# --- sense probe -----------------------------------------------------------------------
vectors = {sense: [word_vector(s) for s in group] for sense, group in ANCHORS.items()}

out.append("\n\n=== sense: leave-one-out over the anchors (the probe's own check) ===")
correct = 0
for sense, group in vectors.items():
    for i, vector in enumerate(group):
        centroids = {
            other: torch.stack(
                [v for j, v in enumerate(vs) if not (other == sense and j == i)]
            ).mean(0)
            for other, vs in vectors.items()
        }
        scores = {k: float(F.cosine_similarity(vector, c, dim=-1)) for k, c in centroids.items()}
        pick = max(scores, key=scores.get)
        correct += pick == sense
        out.append(
            f"  {sense:7s} #{i} -> {pick:7s}  "
            f"barber {scores['barber']:+.3f}  book {scores['book']:+.3f}"
        )
total = sum(len(v) for v in vectors.values())
out.append(f"  held-out accuracy {correct}/{total}")

out.append("\n=== sense: the sentences in question ===")
centroids = {sense: torch.stack(v).mean(0) for sense, v in vectors.items()}
for label, sentence in TESTS.items():
    vector = word_vector(sentence)
    scores = {k: float(F.cosine_similarity(vector, c, dim=-1)) for k, c in centroids.items()}
    pick = max(scores, key=scores.get)
    out.append(
        f"  {label}\n    {sentence}\n"
        f"    -> {pick}   barber {scores['barber']:+.3f}  book {scores['book']:+.3f}"
        f"   margin {abs(scores['barber'] - scores['book']):.3f}"
    )

Path(sys.argv[1]).write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"-> {sys.argv[1]}")
