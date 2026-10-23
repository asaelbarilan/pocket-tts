"""The character fold shared by training (tokenizer corpus, manifests) and inference.

`pocket_tts/utils/text_fold.py` is the single definition. These tests pin it to the normalizer
it mirrors (the sibling hebrew-tts-data-tools checkout; skipped without it) and to the
tokenizer, and check that the model's prompt preparation applies it when the config asks.
"""

import sys
from pathlib import Path

import pytest

from hebrew_training import build_official_manifest as manifest
from hebrew_training import tokenizer_charset
from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences
from pocket_tts.utils import text_fold
from pocket_tts.utils.config import CONFIGS_DIR, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = REPO_ROOT / "tokenizers" / "tokenizer.model"


def test_a_curly_apostrophe_is_folded() -> None:
    assert text_fold.fold_characters("You’ve been signed out of the AWS console.") == (
        "You've been signed out of the AWS console."
    )


@pytest.mark.parametrize(
    "written, plain",
    [
        ("‘a’ ‛b", "'a' 'b"),
        ("“a” „b‟ «c»", '"a" "b" "c"'),
        ("a‐b a‑b a‒b a–b a—b a―b a־b", "a-b a-b a-b a-b a-b a-b a-b"),
        ("ג׳ירפה צה״ל", "ג'ירפה צה\"ל"),
        ("a\u200bb\u200fc\u202ad", "abcd"),
        ("a\tb\nc", "a b c"),
        ("a\\b", "ab"),
    ],
)
def test_folds(written: str, plain: str) -> None:
    assert text_fold.fold_characters(written) == plain


def test_plain_text_is_unchanged_and_the_fold_is_idempotent() -> None:
    for text in ("שלום עולם, מה שלומך?", "Hello, world! (ok) 50% a-b 'x' \"y\"", "שָׁלוֹם"):
        assert text_fold.fold_characters(text) == text
    once = text_fold.fold_characters("“x” — ’y’ \u200fz")
    assert text_fold.fold_characters(once) == once


def test_the_tokenizer_filter_uses_the_same_map() -> None:
    assert tokenizer_charset.FOLD is text_fold.FOLD


def test_the_map_is_what_the_normalizer_does() -> None:
    if not (manifest.DEFAULT_NORMALIZER / "hebrew_tts_normalizer.py").is_file():
        pytest.skip("hebrew-tts-data-tools checkout not found beside this repo")
    manifest.load_normalizer(manifest.DEFAULT_NORMALIZER)  # puts the checkout on sys.path
    assert str(manifest.DEFAULT_NORMALIZER.resolve()) in sys.path
    import hebrew_tts_normalizer  # type: ignore

    assert text_fold.NORMALIZER_CHAR_MAP == hebrew_tts_normalizer._CHAR_CONVERSION_DICT


def test_folded_text_never_falls_back_to_bytes() -> None:
    if not TOKENIZER.is_file():
        pytest.skip("tokenizers/tokenizer.model is not here")
    import sentencepiece

    sp = sentencepiece.SentencePieceProcessor(model_file=str(TOKENIZER))
    curly = "You’ve “quoted” a–b ג׳ירפה"
    assert any(piece.startswith("<0x") for piece in sp.encode(curly, out_type=str))
    folded = sp.encode(text_fold.fold_characters(curly), out_type=str)
    assert not any(piece.startswith("<0x") for piece in folded), folded


def test_prepare_text_prompt_folds_only_when_asked() -> None:
    text = "You’ve been signed out"
    plain, _ = prepare_text_prompt(text, False, False)
    folded, _ = prepare_text_prompt(text, False, False, fold_text=True)
    assert plain == "You’ve been signed out."
    assert folded == "You've been signed out."


def test_split_into_best_sentences_passes_the_flag_through() -> None:
    seen = []

    def tokenizer(text):
        seen.append(text)
        raise StopIteration  # the prepared text is all this test reads

    with pytest.raises(StopIteration):
        split_into_best_sentences(tokenizer, "It’s fine", 50, False, False, fold_text=True)
    assert seen == ["It's fine."]


def test_only_the_hebrew_config_turns_the_fold_on() -> None:
    assert load_config(CONFIGS_DIR / "hebrew.yaml").fold_text is True
    assert load_config(CONFIGS_DIR / "english.yaml").fold_text is False
