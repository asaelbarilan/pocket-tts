"""Convert Hebrew corpora into the manifest format Kyutai's trainer expects.

Replaces our own slicing pipeline for training. The official DataLoader does its own
windowing: given `words`, it cuts each utterance at a random word boundary, uses the audio
before the cut as voice conditioning and the audio after it as the target, paired with only
the remaining words as text. So we hand it whole aligned utterances and let it decide the
split, rather than pre-cutting clips and pairing prompts ourselves.

Row schema (see `get_entry` in training/dataloader.py):

    {"path": str, "start": float, "duration": float, "transcript": str,
     "words": [{"word": str, "start": float, "end": float}]}

`words` is optional but strongly preferred; without it the loader falls back to a random
window for the voice prompt. Word times are RELATIVE to `start`, because align_data.py
keys on (path, start) and aligns inside that window.

Optional, written when the row has a neighbouring segment in the recording:

    "lead_gap": 840, "tail_gap": 120

Milliseconds of natural silence next to the row: from the previous raw segment's last word to
this row's first word, and from this row's last word to the next raw segment's first word (by
Whisper's word times; the neighbour counts even if a filter dropped it, since that is speech
nobody transcribed). Small values mean the row was cut out of running speech or sits against
untranscribed speech, so its edge words are the ones whose timing to doubt; large ones mean it
ends in a pause. They describe the source, not how much silence the row's audio includes. The
aligner moves the first and last word, and `update_timings` shifts both gaps by the same amount
so they stay true (align_official_manifest.py).

Optional, written only for rows where --normalize-text changed a word:

    "alt_words": {"norms": [{"from": 1, "to": 2, "replace": ["ב-14:30"], "k": "n"}]}

Each label (only "norms" so far) maps to spans of `words`, `from`..`to` inclusive, that
can be swapped for the `replace` words -- the form the corpus wrote before normalization.
EVERY change the normalizer makes gets a span, not just numbers: a spelled-out number, an
expanded abbreviation, a spelled Latin letter, a split hyphen, a collapsed `...`, a stripped
quote or bracket, a folded curly apostrophe, a deleted emoji. Words it left alone have none.
Alternates carry no timings; a loader derives them from the words they replace, or swaps
after it has chosen its cut (see `swap_alternates`). Loaders that do not know the key
ignore it. The indices are only valid for the exact `words` list of the row, so any step
that rewrites `words` must keep it one-to-one or drop `alt_words`.

`k` is an optional one-letter tag for the group of normalizer rules the span came from, kept
to one character so manifests stay small. The loader does not read it; it is for counting
(`migrate_alt_words` prints a census) and for weighting groups later. See NORM_KINDS.

A word that produces no output of its own (a lone quote, an emoji, a removed bracket, the
unit a number swallowed, a lone `.` glued onto the word before) has nothing to anchor a span
to, so it is absorbed by the span of the word before it, whose `replace` then lists both.
Words at the very start of a row with no word before them get no span.

Writes `train.jsonl` and `valid.jsonl`. The optional later stages add to these rows and name
their output after themselves (see manifest_files.py): `diacritize_official_manifest` adds a
parallel `nikud` list (`train_diac.jsonl`), `align_official_manifest` refines the word times
(`train_aligned.jsonl`). Neither changes the text of `words` or the number of words, so
`alt_words` indices stay valid through both.

`words` IS THE TEXT. training/dataloader.py:150 builds what it trains on by joining
`words`, and reads `transcript` only in the fallback branch where no word-boundary cut
exists. So anything done to the text -- normalization above all -- has to be done to
`words`, or it is silently discarded. `transcript` here is derived from `words` so the two
cannot drift apart.

WHY SEGMENTS ARE MERGED
-----------------------
CrowdRecital ships utterance-length segments. The ivrit-ai Knesset corpora ship Whisper
DECODER segments: median 1.84 s, p90 4.40 s, measured over 702,646 segments in 120 random
plenum recordings. That length is not usable as a manifest row, because of `MIN_CUT_SEC`
in training/dataloader.py:

    a cut must leave >= 1.0 s of audio on BOTH sides

so a row shorter than 2.0 s has no eligible word-boundary cut at all -- 53.7% of raw
segments. Those rows take the loader's fallback branch, where the voice prompt is a window
read from `entry.start`, i.e. it OVERLAPS the target audio the model is asked to predict.
That is the same prompt leakage prepare_data_v2 was written to remove.

So consecutive segments are merged into utterances of ~12 s before a row is emitted, and
merging stops at a gap longer than --merge-gap (parliamentary audio is 63.7% speech; the
rest is gavel, procedure and dead air, and merging across it would train the model on
silence). Measured on the same 120 recordings, --merge-gap 1.5 --min-duration 4 yields an
11.88 s median row and retains ~5,490 h of the plenum corpus.

WHERE MERGED ROWS ARE CUT
-------------------------
Greedy merging closes a group at the first segment boundary past its target, so ~28% of cuts
between touching rows land in running speech (measured: gap <= 0.3 s between segments, by word
times). Those are the cuts where a word can be clipped or a neighbour's syllable left in.
The default merge is therefore `seam_groups`: close at a pause of at least `--seam-gap` within
+/- `--seam-window` of the target. `--greedy-merge` restores the earlier behaviour, which is only
needed to reproduce a manifest built with it. Whisper's gaps are good enough to find pauses: of gaps
>= 0.3 s, 75-86% were 12+ dB under the surrounding speech in the audio, and of gaps <= 0.1 s
only 5% were.

This is the same job prepare_ivritai.py's generate_slices did for the earlier dataset --
including drawing each group's target around the aim rather than fixing it, which
DurationController did there and --merge-jitter does here. The difference is only the
output: that wrote one wav per utterance, this writes offsets into one transcoded wav per
recording (or the source audio when --no-transcode-audio is used).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from hebrew_training.manifest_files import BUILT
from hebrew_training.official_manifest_common import (
    DurationStats,
    clean,
    format_seconds,
    stable_bucket,
)
from pocket_tts.utils.text_fold import FOLD, NORMALIZER_CHAR_MAP

_HEBREW = re.compile(r"[֐-׿]")

# The label of `alt_words` that holds every normalizer replacement.
NORMS = "norms"
# `k` tags, one letter each: the group of normalizer rules a span came from. `_norm_kind`
# decides from what the corpus wrote and what the normalizer made of it, in this priority.
NORM_KINDS = {
    "n": "number: digits, times, dates, ranges, units, identifiers",
    "l": "Latin: a letter or acronym spelled out in Hebrew, `ה-API`",
    "a": 'abbreviation or respelling: ד"ר, ת"א, the replacements table',
    "h": "hyphen or dash that split a word in two",
    "x": "vanished: an emoji, a lone quote or bracket, a deleted [note]",
    "q": "quotes or brackets around a word that were stripped",
    "p": "punctuation: `...` to `.`, `;` and `:` to `,`, a lone `.` glued on",
    "c": "character fold: curly quotes, Unicode dashes, invisible marks",
    "o": "anything else",
}
_DASHES = frozenset({"-"} | {char for char, plain in FOLD.items() if plain == "-"})
_QUOTES_AND_BRACKETS = frozenset('"“”„‟«»()[]{}')
_WRAPPED_TOKEN = re.compile(r"^[\"“”„‟«»(\[{]|[\"“”„‟«»)\]}]+[.,;:!?…]*$")
_SENTENCE_PUNCTUATION = ".,;:!?…"

# Resolved from this file, not the working directory: the earlier relative default broke
# whenever the command was run from anywhere but the repo root. Assumes the two repos are
# checked out side by side, which is what the recipe tells you to do.
DEFAULT_NORMALIZER = Path(__file__).resolve().parents[2] / "hebrew-tts-data-tools" / "normalizer"
# The normalizer reads one token ahead (number gender) and the last token it emitted behind
# ("פי", "...מאה"). Words that vanish (a lone quote, an emoji) are skipped, so a few words of
# context on each side cover every case.
CONTEXT_WORDS = 3
_GLUED_PUNCTUATION = ".,?!"
PROGRESS_SECONDS = 10.0
PROGRESS_RECORDINGS = 10


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """The builder's options. `migrate_alt_words` reuses them (`add_help=False`, as a parent
    parser) so it can be given exactly the flags the manifest was built with."""
    parser = argparse.ArgumentParser(
        description="Build Kyutai-format training manifests from an aligned Hebrew corpus.",
        add_help=add_help,
    )
    parser.add_argument(
        "--corpus", type=Path, required=True, help="Root holding one directory per recording."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--audio-glob",
        default="audio.wav,audio.m4a,audio.mka",
        help="Comma-separated candidates; the first that matches wins. "
        "CrowdRecital ships audio.mka, the Knesset corpora audio.m4a.",
    )
    parser.add_argument("--align-glob", default="transcript.aligned.json")
    parser.add_argument("--metadata-glob", default="metadata.json")
    parser.add_argument(
        "--transcode-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transcode each used recording to a 24 kHz mono WAV in OUT_DIR/audio "
        "(default: enabled). Completed files are reused; pass --no-transcode-audio "
        "to keep source paths in the manifest.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        help="Where the transcoded WAVs live (default: OUT_DIR/audio). Point a rebuild of the "
        "manifest at the folder of an earlier build and every WAV already there is reused, "
        "not transcoded again.",
    )
    parser.add_argument(
        "--speaker-field",
        default="user_id",
        help="metadata.json key identifying the speaker. Falls back to the "
        "recording directory name, which for the Knesset corpora means "
        "the split is recording-disjoint but not speaker-disjoint.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=4.0,
        help="Post-merge floor. Must stay above 2 x MIN_CUT_SEC (2.0 s) or "
        "the loader cannot cut the row and leaks the prompt.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Should match data.max_duration_sec in the training config.",
    )
    parser.add_argument(
        "--merge-target",
        type=float,
        default=12.0,
        help="Stop merging once a group reaches this length.",
    )
    parser.add_argument(
        "--merge-jitter",
        type=float,
        default=0.3,
        help="Draw each group's target uniformly from target*(1 +/- this), the "
        "way prepare_ivritai.py's DurationController did. 0 fixes the "
        "target and narrows the duration distribution.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seeds the merge jitter, so a rebuild is reproducible."
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=1.5,
        help="Break a merge at a silence longer than this. 0 disables merging, "
        "which is right only for already-utterance-length corpora.",
    )
    parser.add_argument(
        "--seam-window",
        type=float,
        default=5.0,
        help="Merging closes a group at the best boundary within +/- this many seconds of its "
        "target (see --merge-target): a pause of at least --seam-gap if there is one, rather "
        "than the first boundary past the target. On 30 plenum recordings the share of splits "
        "through running speech (gap <= 0.3 s) fell from 28%% to 10%% at 4 and 6%% at 5, with "
        "the same hours.",
    )
    parser.add_argument(
        "--greedy-merge",
        action="store_true",
        help="Close each group at the first segment boundary past its target, ignoring pauses: "
        "the merge every manifest built before seam-aware merging used. Needed only to "
        "reproduce one -- pass it to migrate_alt_words for such a manifest, which must "
        "rebuild the same grouping. Ignores --seam-window and --seam-gap.",
    )
    parser.add_argument(
        "--seam-gap",
        type=float,
        default=0.3,
        help="A silence this long (seconds, between the last word of one segment and the first "
        "of the next, by Whisper's word times) is a natural seam. In the plenum audio 75-86%% "
        "of gaps >= 0.3 s measured 12+ dB below the speech around them, and 83%% of gaps "
        "<= 0.1 s were not silent at all.",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.6,
        help="Median word probability required to keep a segment.",
    )
    parser.add_argument("--valid-hours", type=float, default=2.0)
    parser.add_argument(
        "--normalize-text",
        action="store_true",
        help="Apply the Hebrew TTS normalizer (number expansion etc).",
    )
    parser.add_argument(
        "--normalizer-dir",
        type=Path,
        default=DEFAULT_NORMALIZER,
        help="The normalizer package from the hebrew-tts-data-tools repo. "
        "Defaults to a checkout sitting beside this one.",
    )
    parser.add_argument("--limit-recordings", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of processes for the per-recording work. Output is reproducible "
        "and independent of worker count.",
    )
    parser.add_argument(
        "--spool-dir",
        type=Path,
        default=Path(tempfile.gettempdir()),
        help="Parent for a temporary manifest spool (default: the system temporary directory). "
        "Use local storage; the temporary tree is removed after use.",
    )
    parser.add_argument(
        "--max-spool-gib",
        type=float,
        default=200.0,
        help="Hard limit on JSON payload stored in the temporary spool (default: 200 GiB).",
    )
    return parser


def check_merge_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.seam_window <= 0:
        parser.error("--seam-window must be positive; use --greedy-merge for the earlier merge")
    if args.seam_gap < 0:
        parser.error("--seam-gap must not be negative")


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    check_merge_args(parser, args)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_spool_gib <= 0:
        parser.error("--max-spool-gib must be positive")
    return args


def load_normalizer(directory: Path):
    """The same Hebrew normalizer the earlier pipeline used -- number expansion,
    word replacements, punctuation handling. Only the slicing was replaced; this was not.

    Returns `normalize(words: list[str]) -> list[WordNormalization]`, one result per word:
    `.text` is what the word became, `.consumed` the later words it swallowed (the unit in
    `2.5 kg`). The earlier pipeline normalized a whole slice at once; calling it word by
    word is what drops the context it depends on, so the whole word list must go in
    together."""
    if not directory.exists():
        raise SystemExit(
            f"--normalize-text needs the normalizer, and {directory} does not exist.\n"
            "Clone it beside this repo:\n"
            "    git clone https://github.com/asaelbarilan/hebrew-tts-data-tools\n"
            "or pass --normalizer-dir explicitly."
        )
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory.resolve()))
    try:
        from hebrew_tts_normalizer import (  # type: ignore
            TTSNormalizeOptions,
            load_word_replacements,
            normalize_tts_words_detailed,
        )
    except ImportError as error:
        raise SystemExit(
            f"The normalizer in {directory} is too old or incomplete ({error}).\n"
            "It must provide normalize_tts_words_detailed(), which normalizes words with "
            "their neighbours as context and reports the words a result absorbed. "
            "Update the hebrew-tts-data-tools checkout."
        ) from error

    options = TTSNormalizeOptions(
        apply_word_replacements=True,
        expand_numbers=True,
        keep_punctuations=True,
        attach_punctuations_to_token=True,
        remove_parentheses=False,
    )
    replacements = load_word_replacements()
    return lambda words: normalize_tts_words_detailed(
        words, options=options, word_replacements=replacements
    )


def normalize_words(
    words: list[dict], normalize, before: Sequence[str] = (), after: Sequence[str] = ()
) -> list[dict]:
    """Normalize the word list itself, not just the joined transcript.

    This is not optional polish. training/dataloader.py:150 builds the text it trains on
    from `words`, and only falls back to `transcript` when there is no usable word-boundary
    cut. So normalizing `transcript` alone -- which is what this script did until now --
    was thrown away on every ordinary sample, and the model would have been trained on
    "1995" instead of "אלף תשע מאות תשעים וחמש".

    The words are normalized TOGETHER, never one at a time. The normalizer is context
    sensitive: a number takes its gender from the next token ("14 ימים" -> "ארבעה עשר",
    "14 שנה" -> "ארבע עשרה"), "פי" and "...מאה" change the gender of the number after them,
    and punctuation is glued to the word before it. A word normalized alone has no
    neighbours, so every number silently came out feminine.

    `before` and `after` are plain words just outside `words` (the end of the previous
    row, the start of the next). They are only context: their output is discarded, and
    only the result the normalizer attributes to each of `words` is used.

    A normalized word can expand into several ("1995" -> five words). The expansion is
    spoken across the original word's span, so the span is divided evenly between them.
    That is an approximation, but the loader only uses word times to pick a cut point and
    to trim trailing silence, and both stay correct at the group's outer boundaries.

    A word whose output starts with punctuation (a lone "." that Whisper timed as its own
    word) is glued onto the previous word, as the normalizer does for a whole sentence.
    """
    return normalize_words_with_alternates(words, normalize, before, after)[0]


@dataclass
class _Unit:
    """The output words one input word produced, and the corpus words they stand for."""

    head: int  # index of the input word that produced it
    first: int  # its first and last words in the normalized list
    last: int
    raw: list[str]  # the corpus tokens, extended by the silent words that follow it
    vanished: bool = False  # a word with no output of its own was absorbed
    valid: bool = True


def normalize_words_with_alternates(
    words: list[dict], normalize, before: Sequence[str] = (), after: Sequence[str] = ()
) -> tuple[list[dict], dict | None]:
    """normalize_words, plus the form the corpus wrote for every word the normalizer changed.

    Returns `(words, alt_words)`. `alt_words` is None when nothing changed, else
    `{"norms": [{"from": i, "to": j, "replace": [...], "k": "n"}]}`: `words[i..j]` (inclusive)
    is what the corpus words `replace` were turned into. The indices are into the returned
    `words`, so they stay valid only for that exact list. `k` is a one-letter tag from
    NORM_KINDS.

    There is one span per input word whose output differs from what was written, so a
    spelled-out number, an expanded abbreviation, a split `ה-API`, a `...` collapsed to `.` and
    a curly apostrophe all get one, and a word left alone gets none. `replace` holds the source
    words as written, wrapper quotes and trailing punctuation included, so the swapped-in text
    is what the corpus really said. Cases that need care:

    - a unit spoken with its number (`2.5 kg`) is part of one span; the normalizer reports it as
      consumed, otherwise it would look like a dropped word. When the unit falls outside
      `words` (in the next row's context), or another spoken word sits between the two, the
      span is not emitted, because swapping it back would lose the unit;
    - a word with no output of its own (a lone `"`, an emoji, a deleted `[note]`) or one whose
      whole output was glued onto the word before (a lone `.`) is appended to the span of that
      word, which is then a span even if the word itself was unchanged;
    - a word whose first output piece was glued onto the word before but that says more
      (`,שלום`) makes a span of its own, and the word before gets one too, since its output
      changed;
    - silent words before the first spoken word of the row have no word to attach to and get
      no span.
    """
    offset = len(before)
    results = normalize([*before, *(word["word"] for word in words), *after])
    results = results[offset : offset + len(words)]
    source = [clean(word["word"]) for word in words]
    # `.consumed` counts from the first context word; units[...] below count from the row's.
    consumed_by = {
        later - offset: head for head, result in enumerate(results) for later in result.consumed
    }
    outside = {
        head
        for head, result in enumerate(results)
        if any(later - offset >= len(words) for later in result.consumed)
    }
    out: list[dict] = []
    units: list[_Unit] = []
    by_head: dict[int, _Unit] = {}
    anchor: _Unit | None = None
    for index, (word, result) in enumerate(zip(words, results)):
        parts = clean(result.text).split()
        raw = source[index].split()
        if not parts:
            if anchor is None:
                continue
            head = consumed_by.get(index)
            if head is not None and head != anchor.head:
                # A spoken word sits between a number and the unit it swallowed.
                if head in by_head:
                    by_head[head].valid = False
                continue
            anchor.raw.extend(raw)
            anchor.vanished = anchor.vanished or head is None
            continue
        if len(parts) == 1:
            pieces = [{**word, "word": parts[0]}]
        else:
            start, end = float(word["start"]), float(word["end"])
            step = (end - start) / len(parts)
            pieces = [
                {"word": part, "start": start + n * step, "end": start + (n + 1) * step}
                for n, part in enumerate(parts)
            ]
        if out and pieces[0]["word"][0] in _GLUED_PUNCTUATION:
            glued = pieces.pop(0)
            out[-1] = {**out[-1], "word": out[-1]["word"] + glued["word"], "end": glued["end"]}
            if not pieces:
                if anchor is not None:
                    anchor.raw.extend(raw)
                continue
        unit = _Unit(
            index, len(out), len(out) + len(pieces) - 1, list(raw), valid=index not in outside
        )
        units.append(unit)
        by_head[index] = unit
        anchor = unit
        out.extend(pieces)

    spans = []
    for unit in units:
        if not unit.valid or not unit.raw:
            continue
        got = [piece["word"] for piece in out[unit.first : unit.last + 1]]
        if got == unit.raw:
            continue
        spans.append(
            {
                "from": unit.first,
                "to": unit.last,
                "replace": unit.raw,
                "k": _norm_kind(unit.raw, got, unit.vanished),
            }
        )
    return out, ({NORMS: spans} if spans else None)


def _letters(text: str) -> str:
    return "".join(char for char in text if char.isalpha())


def _norm_kind(raw: list[str], got: list[str], vanished: bool) -> str:
    """The NORM_KINDS tag for a span: what the normalizer did to `raw` to make `got`.

    A heuristic read off the two forms, in priority order, since one span can be several
    things at once (`ב-14:30` is a number and a hyphen; it is a number)."""
    before, after = " ".join(raw), " ".join(got)
    if vanished:
        return "x"
    if any(char.isdigit() for char in before):
        return "n"
    if _letters(before) != _letters(after):
        return "l" if any(char.isascii() and char.isalpha() for char in before) else "a"
    if sum(char in _DASHES for char in before) > sum(char in _DASHES for char in after):
        return "h"
    if "".join(raw) == "".join(got):
        return "p"  # only the spacing changed: a lone `.` glued onto the word before
    if any(_WRAPPED_TOKEN.search(token) for token in raw) and sum(
        char in _QUOTES_AND_BRACKETS for char in before
    ) > sum(char in _QUOTES_AND_BRACKETS for char in after):
        return "q"
    if [c for c in before if c in _SENTENCE_PUNCTUATION] != [
        c for c in after if c in _SENTENCE_PUNCTUATION
    ]:
        return "p"
    if any(char in NORMALIZER_CHAR_MAP for char in before):
        return "c"
    if before.count("'") + before.count('"') != after.count("'") + after.count('"'):
        return "a"  # a geresh or gershayim added or dropped: the replacements table
    return "o"


def validate_alt_words(alt_words: dict, word_count: int) -> None:
    """Raise ValueError unless every label's spans are in range, ordered and disjoint."""
    for label, spans in alt_words.items():
        previous_to = -1
        for span in spans:
            first, last, replace = span["from"], span["to"], span["replace"]
            if not 0 <= first <= last < word_count:
                raise ValueError(
                    f"alt_words[{label!r}] span {first}..{last} outside {word_count} words"
                )
            if first <= previous_to:
                raise ValueError(
                    f"alt_words[{label!r}] span {first}..{last} overlaps or is unordered"
                )
            if not replace or any(
                not token or token != token.strip() or " " in token for token in replace
            ):
                raise ValueError(
                    f"alt_words[{label!r}] span {first}..{last} has a bad replace {replace!r}"
                )
            tag = span.get("k")
            if tag is not None and not (isinstance(tag, str) and len(tag) == 1):
                raise ValueError(
                    f"alt_words[{label!r}] span {first}..{last} has a bad tag {tag!r}; "
                    "it must be one character"
                )
            previous_to = last


def swap_alternates(
    words: Sequence[str],
    spans: Sequence[dict],
    first_word: int = 0,
    use: Sequence[bool] | None = None,
) -> list[str]:
    """`words[first_word:]` with the chosen `spans` swapped for their `replace` words.

    This is the reference for what a training loader should do: cut first, then swap. A
    span the cut falls inside (`from < first_word <= to`) is left spoken, since the
    audio for its first words is gone. `use[i]` says whether to swap `spans[i]`; default all.
    """
    out: list[str] = []
    position = first_word
    for number, span in enumerate(spans):
        if span["from"] < first_word or (use is not None and not use[number]):
            continue
        out.extend(words[position : span["from"]])
        out.extend(span["replace"])
        position = span["to"] + 1
    out.extend(words[position:])
    return out


def context_words(groups: list[list[dict]], index: int, args) -> tuple[list[str], list[str]]:
    """The words just before and after group `index` that belong to the same sentence flow.

    Merged groups are cut at a target length, not at a sentence end, so a number can end one
    row while its noun opens the next. Neighbours count only when the silence between them
    is within --merge-gap, the same test merging used. With merging off each segment is an
    utterance of its own and its neighbours are unrelated.
    """
    if args.merge_gap <= 0:
        return [], []
    group = groups[index]
    before: list[str] = []
    after: list[str] = []
    if index > 0 and group[0]["start"] - groups[index - 1][-1]["end"] <= args.merge_gap:
        before = [w["word"] for s in groups[index - 1] for w in s["words"]][-CONTEXT_WORDS:]
    if index + 1 < len(groups) and groups[index + 1][0]["start"] - group[-1]["end"] <= (
        args.merge_gap
    ):
        after = [w["word"] for s in groups[index + 1] for w in s["words"]][:CONTEXT_WORDS]
    return before, after


def find_audio(recording: Path, globs: str) -> Path | None:
    """First matching audio file. Corpora differ: audio.wav here, audio.m4a there."""
    for pattern in (g.strip() for g in globs.split(",") if g.strip()):
        match = next(iter(recording.glob(pattern)), None)
        if match is not None:
            return match
    return None


def audio_dir_of(args) -> Path:
    return getattr(args, "audio_dir", None) or args.out_dir / "audio"


def transcode_audio(audio: Path, recording: Path, audio_dir: Path) -> tuple[Path, bool]:
    """Write one restart-safe WAV per recording directly to the durable output volume.

    A WAV that is already there is never rewritten or removed: the rename below is the last
    step, so a file at the final name is a complete one, and a rebuild -- into this folder or,
    with --audio-dir, into a new one -- finds it and skips the transcode. Only a half-written
    `.partial` from an interrupted run is replaced."""
    target = audio_dir / f"{recording.name}.wav"
    if target.is_file() and target.stat().st_size > 44:
        return target.resolve(), False

    # Deterministic so a restart overwrites, rather than strands, a potentially huge partial.
    partial = target.with_name(f".{target.name}.partial")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(audio),
                "-map",
                "0:a:0",
                "-map_metadata",
                "-1",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "24000",
                "-c:a",
                "pcm_s16le",
                "-threads",
                "1",
                "-f",
                "wav",
                str(partial),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or f"exit status {result.returncode}"
            raise RuntimeError(f"ffmpeg failed for {audio}: {detail}")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target.resolve(), True


def clean_segments(segments: list[dict], args, stats: dict) -> list[dict]:
    """Segments that pass the quality and language filters, times still absolute."""
    kept = []
    for segment in segments:
        words = segment.get("words") or []
        if not words:
            stats["no_words"] += 1
            continue

        probabilities = [w["probability"] for w in words if w.get("probability") is not None]
        if probabilities and statistics.median(probabilities) < args.min_quality:
            stats["low_quality"] += 1
            continue

        timed = [w for w in words if w.get("start") is not None and w.get("end") is not None]
        if not timed:
            stats["no_words"] += 1
            continue

        text = clean(segment.get("text", ""))
        if not text or not _HEBREW.search(text):
            stats["no_hebrew"] += 1
            continue

        kept.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text,
                "words": timed,
            }
        )
    kept.sort(key=lambda s: s["start"])
    return kept


def merge_segments(segments: list[dict], args, rng: random.Random) -> list[list[dict]]:
    """Group consecutive segments into utterance-length runs.

    By default the groups are cut at pauses near the target (`seam_groups`). With
    --greedy-merge a group closes when it reaches its target length, when the next segment
    would push it past --max-duration, or when the silence before it exceeds --merge-gap. See
    the module docstring for why merging is required and not a tuning knob.

    Each group draws its own target around --merge-target rather than using it fixed, which
    is what prepare_ivritai.py's DurationController did. Measured over 120 plenum
    recordings, jitter 0.3 widens the p10-p90 duration spread from 9.70 s to 10.65 s for the
    same total hours. A narrow duration distribution was implicated in the earlier run's
    overfitting (docs/eos-overfitting-research.md), so the spread is worth having free.
    """
    if args.merge_gap <= 0:
        return [[segment] for segment in segments]

    def next_target() -> float:
        if args.merge_jitter <= 0:
            return args.merge_target
        low = max(args.min_duration, args.merge_target * (1 - args.merge_jitter))
        high = min(args.max_duration, args.merge_target * (1 + args.merge_jitter))
        return rng.uniform(low, high)

    if not getattr(args, "greedy_merge", False):
        return seam_groups(segments, args, next_target)

    groups: list[list[dict]] = []
    current: list[dict] = []
    target = next_target()
    for segment in segments:
        if not current:
            current = [segment]
            continue
        gap = segment["start"] - current[-1]["end"]
        span = segment["end"] - current[0]["start"]
        if gap <= args.merge_gap and span <= args.max_duration:
            current.append(segment)
            if current[-1]["end"] - current[0]["start"] >= target:
                groups.append(current)
                current = []
                target = next_target()
        else:
            groups.append(current)
            current = [segment]
            target = next_target()
    if current:
        groups.append(current)
    return groups


def segment_gap(before: dict, after: dict) -> float:
    """Silence between the last word of `before` and the first word of `after`, by the
    segments' word times. Segment times are looser: Whisper pads them."""
    return after["words"][0]["start"] - before["words"][-1]["end"]


def seam_groups(segments: list[dict], args, next_target) -> list[list[dict]]:
    """Group segments like the greedy merge, but cut where the speaker paused.

    Greedy merging closes a group at the first segment boundary past its target, whatever is
    happening there. About 30% of those boundaries sit in running speech (gap <= 0.3 s), where
    Whisper's timings are least trustworthy and a split can clip the last word of one row or
    leave the first syllable of the next in its tail. Whisper segments are short (median 1.8 s),
    so a window around the target almost always holds a real pause.

    A run is a stretch of segments with no silence over --merge-gap between them. It is
    consumed from the left: the group takes everything left if that fits the window, else it
    closes at a boundary inside [target - w, target + w] (clamped to --min/--max-duration),
    preferring a seam of at least --seam-gap and, among those, the one nearest the target. With
    no seam in the window it takes the longest pause there. A cut that would leave a remainder
    shorter than --min-duration (a row the loader could not cut) is avoided when another
    boundary exists. The target is still drawn per group, so --merge-jitter still widens the
    duration distribution.
    """
    window = getattr(args, "seam_window", 5.0)
    seam = getattr(args, "seam_gap", 0.3)
    groups: list[list[dict]] = []
    run: list[dict] = []

    def flush() -> None:
        index = 0
        while index < len(run):
            first = run[index]["start"]
            target = next_target()
            low = max(args.min_duration, target - window)
            high = min(args.max_duration, target + window)
            if run[-1]["end"] - first <= high:
                groups.append(run[index:])
                return
            inside = [j for j in range(index, len(run) - 1) if low <= run[j]["end"] - first <= high]
            usable = [
                j for j in inside if run[-1]["end"] - run[j + 1]["start"] >= args.min_duration
            ]
            inside = usable or inside
            if inside:
                gaps = {j: segment_gap(run[j], run[j + 1]) for j in inside}
                natural = [j for j in inside if gaps[j] >= seam]
                if natural:
                    last = min(natural, key=lambda j: abs(run[j]["end"] - first - target))
                else:
                    last = max(
                        inside, key=lambda j: (gaps[j], -abs(run[j]["end"] - first - target))
                    )
            else:
                # One long segment spans the window: close as late as --max-duration allows.
                fits = [
                    j
                    for j in range(index, len(run) - 1)
                    if run[j]["end"] - first <= args.max_duration
                ]
                last = fits[-1] if fits else index
            groups.append(run[index : last + 1])
            index = last + 1

    for segment in segments:
        if run and segment["start"] - run[-1]["end"] > args.merge_gap:
            flush()
            run.clear()
        run.append(segment)
    flush()
    return groups


def _speech_bounds(segment: dict) -> tuple[float, float] | None:
    """When speech starts and stops in a raw Whisper segment: its first and last word times,
    else the segment's own. None if it has no usable times."""
    words = [
        w
        for w in segment.get("words") or []
        if w.get("start") is not None and w.get("end") is not None
    ]
    try:
        if words:
            return float(words[0]["start"]), float(words[-1]["end"])
        return float(segment["start"]), float(segment["end"])
    except (KeyError, TypeError, ValueError):
        return None


def neighbour_gaps(segments: list[dict]):
    """A function `(group) -> (lead_gap_ms, tail_gap_ms)` over the raw `segments`.

    A row's boundary is only a natural seam if the audio next to it is quiet, and that depends
    on the next RAW segment, including the ones the quality and language filters dropped:
    those are speech nobody transcribed. In 30 plenum recordings 56% of rows end next to a
    dropped segment, and 27% of those within 0.3 s of it. Each gap is the silence between the
    row's first (last) word and the previous (next) raw segment's last (first) word, None at the
    ends of the recording."""
    raw = sorted(
        (
            (float(segment["start"]), bounds)
            for segment in segments
            if isinstance(segment.get("start"), (int, float))
            and (bounds := _speech_bounds(segment)) is not None
        ),
        key=lambda item: item[0],
    )
    starts = [start for start, _ in raw]

    def gaps(group: list[dict]) -> tuple[int | None, int | None]:
        first, last = group[0], group[-1]
        after = bisect_right(starts, last["start"])
        before = bisect_left(starts, first["start"]) - 1
        lead = tail = None
        if before >= 0:
            lead = max(0.0, first["words"][0]["start"] - raw[before][1][1])
        if after < len(raw):
            tail = max(0.0, raw[after][1][0] - last["words"][-1]["end"])
        return (
            None if lead is None else round(lead * 1000),
            None if tail is None else round(tail * 1000),
        )

    return gaps


def segment_rows(recording: Path, args, normalize, rng: random.Random) -> tuple[list[dict], dict]:
    """One manifest row per merged run of aligned segments."""
    stats = {
        "no_words": 0,
        "low_quality": 0,
        "bad_duration": 0,
        "no_hebrew": 0,
        "kept": 0,
        "audio_transcoded": 0,
        "audio_reused": 0,
    }
    audio = find_audio(recording, args.audio_glob)
    align = next(iter(recording.glob(args.align_glob)), None)
    if audio is None or align is None:
        return [], stats

    # No speaker field in the Knesset corpora, so fall back to the recording id. Without
    # this every row would share one speaker key and the validation split would collapse.
    speaker = None
    meta_file = next(iter(recording.glob(args.metadata_glob)), None)
    if meta_file is not None:
        try:
            speaker = json.loads(meta_file.read_text(encoding="utf-8")).get(args.speaker_field)
        except (json.JSONDecodeError, OSError):
            speaker = None
    speaker = str(speaker) if speaker else recording.name

    try:
        segments = json.loads(align.read_text(encoding="utf-8"))["segments"]
    except (json.JSONDecodeError, KeyError, OSError):
        return [], stats

    rows = []
    groups = merge_segments(clean_segments(segments, args, stats), args, rng)
    neighbours = neighbour_gaps(segments)
    for index, group in enumerate(groups):
        start = group[0]["start"]
        duration = group[-1]["end"] - start
        if not args.min_duration <= duration <= args.max_duration:
            stats["bad_duration"] += 1
            continue

        words = [word for segment in group for word in segment["words"]]
        alt_words = None
        if normalize is not None:
            before, after = context_words(groups, index, args)
            words, alt_words = normalize_words_with_alternates(words, normalize, before, after)
        if not words:
            stats["no_hebrew"] += 1
            continue

        # Derived from `words` so the two can never disagree -- and because `words` is what
        # the loader actually trains on.
        text = clean(" ".join(word["word"] for word in words))
        if not text:
            stats["no_hebrew"] += 1
            continue

        # Relative to `start`: the loader compares these against entry.duration, and
        # align_data.py aligns inside the (path, start) window.
        timed = [
            {
                "word": clean(word["word"]),
                "start": round(float(word["start"]) - start, 4),
                "end": round(float(word["end"]) - start, 4),
            }
            for word in words
        ]

        row = {
            "path": str(audio.resolve()),
            "start": round(start, 4),
            "duration": round(duration, 4),
            "transcript": text,
            "words": timed,
            "speaker": speaker,
        }
        lead_gap, tail_gap = neighbours(group)
        if lead_gap is not None:
            row["lead_gap"] = lead_gap
        if tail_gap is not None:
            row["tail_gap"] = tail_gap
        if alt_words:
            validate_alt_words(alt_words, len(timed))
            row["alt_words"] = alt_words
        rows.append(row)
        stats["kept"] += 1
    if rows and args.transcode_audio:
        manifest_audio, created = transcode_audio(audio, recording, audio_dir_of(args))
        for row in rows:
            row["path"] = str(manifest_audio)
        stats["audio_transcoded" if created else "audio_reused"] += 1
    return rows, stats


_WORKER_NORMALIZE = None
_WORKER_ARGS = None
_WORKER_SPOOL_DIR = None
_WORKER_MAX_SPOOL_BYTES = 0
_WORKER_SPOOL_BYTES = None
_WORKER_SPOOL_LOCK = None


def _init_worker(
    args: argparse.Namespace, spool_dir: Path, max_spool_bytes: int, spool_bytes, spool_lock
) -> None:
    """Load the normalizer once per worker process (lambdas do not pickle)."""
    global _WORKER_ARGS, _WORKER_MAX_SPOOL_BYTES, _WORKER_NORMALIZE
    global _WORKER_SPOOL_BYTES, _WORKER_SPOOL_DIR, _WORKER_SPOOL_LOCK
    _WORKER_ARGS = args
    _WORKER_SPOOL_DIR = spool_dir
    _WORKER_MAX_SPOOL_BYTES = max_spool_bytes
    _WORKER_SPOOL_BYTES = spool_bytes
    _WORKER_SPOOL_LOCK = spool_lock
    if args.normalize_text:
        _WORKER_NORMALIZE = load_normalizer(args.normalizer_dir)


def _recording_rng(seed: int, recording: Path) -> random.Random:
    # str hashing is salted per process, so use a digest for cross-process stability.
    digest = hashlib.sha256(f"{seed}:{recording.name}".encode()).digest()
    return random.Random(digest)


def _write_spool(
    index: int, rows: list[dict], spool_dir: Path, max_spool_bytes: int, spool_bytes, spool_lock
) -> tuple[Path, str | None, float, array]:
    path = spool_dir / f"{index:09d}.jsonl"
    reserved = 0
    durations = array("d")
    speaker = rows[0]["speaker"] if rows else None
    seconds = 0.0
    try:
        with path.open("wb") as handle:
            for row in rows:
                if row["speaker"] != speaker:
                    raise RuntimeError(f"recording {index} contains multiple speakers")
                payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                with spool_lock:
                    wanted = spool_bytes.value + len(payload)
                    if wanted > max_spool_bytes:
                        raise RuntimeError(
                            f"manifest spool exceeded {max_spool_bytes / 2**30:.1f} GiB; "
                            "free local space, raise --max-spool-gib, or use a smaller corpus"
                        )
                    spool_bytes.value = wanted
                reserved += len(payload)
                handle.write(payload)
                duration = float(row["duration"])
                durations.append(duration)
                seconds += duration
    except BaseException:
        with spool_lock:
            spool_bytes.value -= reserved
        path.unlink(missing_ok=True)
        raise
    return path, speaker, seconds, durations


def _process_recording(
    index: int, recording: Path
) -> tuple[int, Path, str | None, float, int, array, dict]:
    rows, stats = segment_rows(
        recording, _WORKER_ARGS, _WORKER_NORMALIZE, _recording_rng(_WORKER_ARGS.seed, recording)
    )
    path, speaker, seconds, durations = _write_spool(
        index,
        rows,
        _WORKER_SPOOL_DIR,
        _WORKER_MAX_SPOOL_BYTES,
        _WORKER_SPOOL_BYTES,
        _WORKER_SPOOL_LOCK,
    )
    return index, path, speaker, seconds, len(durations), durations, stats


def main() -> None:
    args = parse_args()
    recordings = sorted(p for p in args.corpus.iterdir() if p.is_dir())
    if args.limit_recordings:
        recordings = recordings[: args.limit_recordings]

    if args.transcode_audio:
        if shutil.which("ffmpeg") is None:
            raise SystemExit("--transcode-audio requires ffmpeg on PATH")
        audio_dir_of(args).mkdir(parents=True, exist_ok=True)

    if not args.spool_dir.is_dir():
        raise SystemExit(f"--spool-dir is not a directory: {args.spool_dir}")
    configured_spool_bytes = int(args.max_spool_gib * 2**30)
    free_bytes = shutil.disk_usage(args.spool_dir).free
    max_spool_bytes = min(configured_spool_bytes, int(free_bytes * 0.9))
    if max_spool_bytes < configured_spool_bytes:
        print(
            f"note: limiting the spool to {max_spool_bytes / 2**30:.1f} GiB, 90% of the "
            f"{free_bytes / 2**30:.1f} GiB free on {args.spool_dir}"
        )

    spool_dir = Path(tempfile.mkdtemp(prefix="pocket-tts-manifest-", dir=args.spool_dir))
    context = multiprocessing.get_context()
    spool_bytes = context.Value("Q", 0, lock=False)
    spool_lock = context.Lock()
    results: list[tuple[Path, str | None, float, int] | None] = [None] * len(recordings)
    duration_stats = DurationStats()
    progress_started = time.monotonic()
    progress_last_report = progress_started
    progress_last_count = 0
    completed_recordings = 0
    totals = {
        "no_words": 0,
        "low_quality": 0,
        "bad_duration": 0,
        "no_hebrew": 0,
        "kept": 0,
        "audio_transcoded": 0,
        "audio_reused": 0,
    }

    def add_stats(stats: dict) -> None:
        for key, value in stats.items():
            totals[key] += value

    def collect(result) -> None:
        nonlocal completed_recordings, progress_last_count, progress_last_report
        index, path, speaker, seconds, count, durations, stats = result
        results[index] = (path, speaker, seconds, count)
        for duration in durations:
            duration_stats.add(duration)
        add_stats(stats)
        completed_recordings += 1

        now = time.monotonic()
        report = (
            completed_recordings == len(recordings)
            or completed_recordings - progress_last_count >= PROGRESS_RECORDINGS
            or now - progress_last_report >= PROGRESS_SECONDS
        )
        if not report:
            return
        elapsed = now - progress_started
        remaining = len(recordings) - completed_recordings
        eta = elapsed * remaining / completed_recordings
        audio_progress = ""
        if args.transcode_audio:
            audio_progress = (
                f", WAVs {totals['audio_transcoded']} new/{totals['audio_reused']} reused"
            )
        print(
            f"progress : {completed_recordings}/{len(recordings)} recordings "
            f"({100 * completed_recordings / len(recordings):.1f}%), "
            f"{remaining} waiting, elapsed {format_seconds(elapsed)}, ETA {format_seconds(eta)}, "
            f"{totals['kept']} utterances{audio_progress}",
            flush=True,
        )
        progress_last_report = now
        progress_last_count = completed_recordings

    try:
        if args.workers == 1:
            normalize = load_normalizer(args.normalizer_dir) if args.normalize_text else None
            for index, recording in enumerate(recordings):
                rows, stats = segment_rows(
                    recording, args, normalize, _recording_rng(args.seed, recording)
                )
                path, speaker, seconds, durations = _write_spool(
                    index, rows, spool_dir, max_spool_bytes, spool_bytes, spool_lock
                )
                collect((index, path, speaker, seconds, len(durations), durations, stats))
        else:
            initargs = (args, spool_dir, max_spool_bytes, spool_bytes, spool_lock)
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=context,
                initializer=_init_worker,
                initargs=initargs,
            ) as pool:
                recording_iter = iter(enumerate(recordings))
                pending = {}
                for index, recording in recording_iter:
                    future = pool.submit(_process_recording, index, recording)
                    pending[future] = recording
                    if len(pending) >= args.workers * 2:
                        done, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for completed in done:
                            pending.pop(completed)
                            collect(completed.result())
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for completed in done:
                        pending.pop(completed)
                        collect(completed.result())

        if not totals["kept"]:
            raise SystemExit("no usable utterances found")

        # Speaker-disjoint split, deterministic: whole speakers go to validation until the
        # hour budget is met, so no voice appears on both sides.
        seconds_by_speaker: dict[str, float] = {}
        for result in results:
            assert result is not None
            _, speaker, seconds, _ = result
            if speaker is not None:
                seconds_by_speaker[speaker] = seconds_by_speaker.get(speaker, 0.0) + seconds
        budget = args.valid_hours * 3600
        ordered = sorted(seconds_by_speaker, key=stable_bucket)
        valid_speakers: set[str] = set()
        accumulated = 0.0
        for speaker in ordered:
            if accumulated >= budget:
                break
            if seconds_by_speaker[speaker] > budget * 1.5:
                continue
            valid_speakers.add(speaker)
            accumulated += seconds_by_speaker[speaker]

        # A whole Knesset sitting runs several hours, so every candidate can exceed the
        # oversize guard above and leave validation empty.
        if not valid_speakers:
            smallest = min(ordered, key=lambda s: seconds_by_speaker[s])
            valid_speakers.add(smallest)
            accumulated = seconds_by_speaker[smallest]
            print(
                f"note: no speaker fit under {args.valid_hours} h; validation is the single "
                f"smallest ({smallest}, {accumulated / 3600:.2f} h)"
            )

        train_speakers = set(seconds_by_speaker) - valid_speakers
        assert not (train_speakers & valid_speakers), "speaker leaked across split"
        assert train_speakers and valid_speakers, "one side of the split is empty"

        args.out_dir.mkdir(parents=True, exist_ok=True)
        train_count = valid_count = 0
        train_seconds = valid_seconds = 0.0
        train_path = args.out_dir / BUILT.format(split="train")
        valid_path = args.out_dir / BUILT.format(split="valid")
        with train_path.open("wb") as train_handle, valid_path.open("wb") as valid_handle:
            for result in results:
                assert result is not None
                path, speaker, seconds, count = result
                is_valid = speaker in valid_speakers
                with path.open("rb") as source:
                    shutil.copyfileobj(source, valid_handle if is_valid else train_handle)
                path.unlink()
                if is_valid:
                    valid_count += count
                    valid_seconds += seconds
                else:
                    train_count += count
                    train_seconds += seconds
        assert train_count + valid_count == totals["kept"]

        print(f"recordings scanned : {len(recordings)}")
        print(f"utterances kept    : {totals['kept']}")
        if args.transcode_audio:
            print(
                f"audio wavs          : {totals['audio_transcoded']} transcoded, "
                f"{totals['audio_reused']} reused"
            )
        print(
            f"  dropped: no_words {totals['no_words']}, low_quality {totals['low_quality']}, "
            f"duration {totals['bad_duration']}, no_hebrew {totals['no_hebrew']}"
        )
        print(
            f"train : {train_count:6d} utterances, {train_seconds / 3600:6.2f} h, "
            f"{len(train_speakers)} speakers"
        )
        print(
            f"valid : {valid_count:6d} utterances, {valid_seconds / 3600:6.2f} h, "
            f"{len(valid_speakers)} speakers"
        )

        for line in duration_stats.lines():
            print(line)
        print(f"wrote {train_path} and {valid_path}")
    finally:
        shutil.rmtree(spool_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
