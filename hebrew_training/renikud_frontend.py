"""The guard that has to sit in front of ReNikud.

ReNikud takes unvocalized Hebrew. Give it vocalized text and it does not fail loudly: the
nikud characters pass straight through into the IPA string, so

    אֲנִי רוֹצָה לָלֶכֶת       becomes     ʔaֲnִˈi ʁoֹtsָa leָlֶχֶt

which is not a mispronunciation but a broken string, and it reaches the TTS as one. All six
vocalized sentences in eval_sentences_he.jsonl failed this way before this module existed.

That matters because the obvious pipeline -- diacritize with DictaBERT, then phonemize --
is exactly the one that breaks it. So the strip is not a convenience, it is the thing that
makes the two models composable at all.

    from hebrew_training.renikud_frontend import phonemize
    phonemize(g2p, "אֲנִי רוֹצָה לָלֶכֶת הַבַּיְתָה")
"""

from __future__ import annotations

import re
import unicodedata

# Hebrew points and cantillation. Not [֐-׿], which would also eat the letters.
NIKUD = re.compile(r"[֑-ׇ]")
# Anything left in the output from this block means the strip missed something.
HEBREW_LETTER = re.compile(r"[א-ת]")


def strip_nikud(text: str) -> str:
    """Vocalized Hebrew to the bare consonantal spelling ReNikud expects.

    NFC first: the same vocalized word can arrive with its marks composed or decomposed,
    and a regex over code points only sees one of those spellings.
    """
    return NIKUD.sub("", unicodedata.normalize("NFC", text))


def phonemize(g2p, text: str, strict: bool = True) -> str:
    """ReNikud's IPA for `text`, with the nikud removed first and the result checked.

    `strict` raises when a Hebrew letter survives into the IPA. That is always a bug --
    either input this module does not know how to clean, or a word ReNikud declined to
    phonemize -- and it is worth a crash rather than a string that reaches the TTS and is
    read out as noise.
    """
    out = g2p.phonemize(strip_nikud(text))
    left = HEBREW_LETTER.findall(out)
    if left and strict:
        raise ValueError(
            f"Hebrew survived into the IPA ({''.join(sorted(set(left)))}). "
            f"Input: {text[:80]!r}\nOutput: {out[:120]!r}"
        )
    return out
