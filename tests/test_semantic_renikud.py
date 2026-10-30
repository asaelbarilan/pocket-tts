"""The RenikudPlus semantic encoder and the encoder choice (pocket_tts/conditioners/
semantic_renikud.py, SEMANTIC_CONDITIONING.md).

No network and no model download, except `test_real_model_*`, which is skipped unless the pinned
fp32 ONNX file is in the Hugging Face cache (or RENIKUD_ONNX points at it).
"""

import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from test_semantic_conditioning import CORPUS, DIM, HIDDEN, flow_lm, make_sp, small_config

from pocket_tts.conditioners import semantic as semantic_module
from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.conditioners.semantic import (
    CHAR_CLASSES,
    ENCODER_METADATA_KEY,
    SemanticFusion,
    char_owner,
    check_encoder_record,
    encoder_metadata,
    extend_over_marks,
    load_state_dict_allowing_semantic,
    piece_class,
    pool_to_tokens,
    read_encoder_record,
)
from pocket_tts.conditioners.semantic_renikud import (
    RenikudPlusEncoder,
    RenikudTokenizer,
    build_encoder_view,
    find_layer_outputs,
    split_windows,
    tap_layer_outputs,
)
from pocket_tts.utils.config import (
    CONFIGS_DIR,
    DICTABERT_LARGE,
    DICTABERT_LARGE_REVISION,
    RENIKUD_PLUS,
    RENIKUD_PLUS_REVISION,
    Config,
    SemanticConditionerConfig,
    load_config,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sp():
    return make_sp()


# -------------------------------------------------------------------------------- config


def test_missing_encoder_means_dictabert_with_todays_defaults():
    for config in (
        SemanticConditionerConfig(),
        SemanticConditionerConfig(**{}),  # noqa: PIE804 - exercise an explicit empty config mapping
        SemanticConditionerConfig.model_validate({"first_layer": 3}),
    ):
        assert config.encoder == "dictabert"
        assert (config.model_name, config.revision) == (DICTABERT_LARGE, DICTABERT_LARGE_REVISION)
        assert (config.max_tokens, config.filename, config.onnx_threads) == (512, None, None)
    assert SemanticConditionerConfig() == SemanticConditionerConfig(encoder="dictabert")


def test_old_yaml_without_encoder_loads_as_dictabert():
    raw = yaml.safe_load((CONFIGS_DIR / "hebrew_semantic.yaml").read_text())
    section = raw["flow_lm"]["semantic_conditioner"]
    assert "encoder" not in section  # the shipped DictaBERT config is unchanged
    config = Config(**raw)
    assert config.flow_lm.semantic_conditioner == SemanticConditionerConfig()
    assert config.flow_lm.semantic_conditioner.encoder == "dictabert"


def test_renikud_defaults():
    config = SemanticConditionerConfig(encoder="renikud_plus")
    assert (config.model_name, config.revision) == (RENIKUD_PLUS, RENIKUD_PLUS_REVISION)
    assert config.filename == "model.onnx"  # fp32, never the int8 file
    assert (config.max_tokens, config.onnx_threads, config.hidden_size) == (2048, 1, 1024)
    assert config.onnx_provider == "cuda"
    assert (config.first_layer, config.last_layer) == (8, 20)
    # Explicit values win over the encoder's defaults.
    config = SemanticConditionerConfig(
        encoder="renikud_plus", max_tokens=1024, onnx_threads=4, revision="abc"
    )
    assert (config.max_tokens, config.onnx_threads, config.revision) == (1024, 4, "abc")


def test_encoder_specific_keys_are_checked():
    with pytest.raises(ValueError, match="encoder"):
        SemanticConditionerConfig(encoder="bert")
    with pytest.raises(ValueError, match="renikud_plus"):
        SemanticConditionerConfig(filename="model.onnx")
    with pytest.raises(ValueError, match="renikud_plus"):
        SemanticConditionerConfig(onnx_threads=2)
    with pytest.raises(ValueError, match="onnx_threads"):
        SemanticConditionerConfig(encoder="renikud_plus", onnx_threads=0)


def test_identity_names_the_encoder_and_its_weights():
    dicta = SemanticConditionerConfig().identity()
    assert dicta == {
        "encoder": "dictabert",
        "model_name": DICTABERT_LARGE,
        "revision": DICTABERT_LARGE_REVISION,
        "layers": "8-20",
    }
    renikud = SemanticConditionerConfig(encoder="renikud_plus").identity()
    assert renikud["filename"] == "model.onnx" and renikud["encoder"] == "renikud_plus"


def test_example_model_config_selects_renikud():
    config = load_config(CONFIGS_DIR / "hebrew_semantic_renikud.yaml")
    section = config.flow_lm.semantic_conditioner
    assert section == SemanticConditionerConfig(encoder="renikud_plus")
    # Everything but the section is hebrew_semantic.yaml's.
    base = yaml.safe_load((CONFIGS_DIR / "hebrew_semantic.yaml").read_text())
    other = yaml.safe_load((CONFIGS_DIR / "hebrew_semantic_renikud.yaml").read_text())
    base["flow_lm"].pop("semantic_conditioner")
    other["flow_lm"].pop("semantic_conditioner")
    assert base == other


def test_example_training_config_selects_renikud(tmp_path):
    sys.path.insert(0, str(REPO))
    from training.args import TrainArgs
    from training.modules.builders import load_model_config

    raw = yaml.safe_load(
        (REPO / "training/configs/finetune_hebrew_semantic_renikud.yaml").read_text()
    )
    assert raw["semantic_conditioner"]["encoder"] == "renikud_plus"
    assert SemanticConditionerConfig(**raw["semantic_conditioner"]) == SemanticConditionerConfig(
        encoder="renikud_plus"
    )
    assert TrainArgs().semantic_conditioner is None

    # A model config with a DictaBERT section written out in full (like hebrew_semantic.yaml).
    base = yaml.safe_load((CONFIGS_DIR / "english.yaml").read_text())
    base["flow_lm"]["semantic_conditioner"] = SemanticConditionerConfig().model_dump(
        exclude_none=True
    )
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(base))
    # Naming another encoder replaces that section, so DictaBERT's model name and limits do not
    # leak into a RenikudPlus run.
    config = load_model_config(str(path), {}, {"encoder": "renikud_plus"})
    assert config.flow_lm.semantic_conditioner == SemanticConditionerConfig(encoder="renikud_plus")
    # The same encoder (or none named) is still a merge.
    config = load_model_config(str(path), {}, {"first_layer": 3})
    assert config.flow_lm.semantic_conditioner.encoder == "dictabert"
    assert config.flow_lm.semantic_conditioner.first_layer == 3
    # And a plain model config gets the RenikudPlus section from the training config.
    english = "pocket_tts/config/english.yaml"
    config = load_model_config(english, {}, raw["semantic_conditioner"])
    assert config.flow_lm.semantic_conditioner.encoder == "renikud_plus"


# --------------------------------------------------------------- mode off, mode dictabert


def test_default_import_loads_nothing_new():
    """Mode 1 (off) and mode 2 (DictaBERT) import neither the RenikudPlus module nor ONNX."""
    code = (
        "import sys\n"
        "import pocket_tts.models.tts_model\n"
        "import pocket_tts.conditioners.semantic\n"
        "bad = [m for m in ('onnx', 'onnxruntime', "
        "'pocket_tts.conditioners.semantic_renikud') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_load_encoder_dispatches_on_the_config(monkeypatch, tmp_path):
    calls = []

    class Fake(semantic_module.SemanticEncoder):
        hidden_size = HIDDEN
        num_layers = 24

    monkeypatch.setattr(
        semantic_module,
        "DictaBertEncoder",
        lambda name, revision, max_tokens, keep_layers=None, source=None: (
            calls.append(("dictabert", name, revision, max_tokens, keep_layers, source)) or Fake()
        ),
    )
    monkeypatch.setattr(
        RenikudPlusEncoder,
        "from_config",
        classmethod(
            lambda cls, config, source=None: (
                calls.append(("renikud", config.filename, source)) or Fake()
            )
        ),
    )
    SemanticFusion(small_config(), DIM).load_encoder()
    SemanticFusion(small_config(encoder="renikud_plus"), DIM).load_encoder()
    assert calls == [
        ("dictabert", DICTABERT_LARGE, DICTABERT_LARGE_REVISION, 512, 3, None),
        ("renikud", "model.onnx", None),
    ]
    # With a local directory (the deploy server's copy of the encoder store) it is passed on.
    calls.clear()
    SemanticFusion(small_config(), DIM).load_encoder(tmp_path)
    SemanticFusion(small_config(encoder="renikud_plus"), DIM).load_encoder(tmp_path)
    assert [c[-1] for c in calls] == [tmp_path, tmp_path]
    with pytest.raises(FileNotFoundError, match="directory not found"):
        SemanticFusion(small_config(), DIM).load_encoder(tmp_path / "missing")


def test_missing_onnx_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnx", None)
    config = SemanticConditionerConfig(encoder="renikud_plus")
    with pytest.raises(ImportError, match="semantic-renikud"):
        RenikudPlusEncoder.from_config(config)


# ----------------------------------------------------------------------- encoder view

VOCAB = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3}


def make_vocab(*texts: str) -> dict[str, int]:
    vocab = dict(VOCAB)
    for text in texts:
        for ch in build_encoder_view(text).chars:
            vocab.setdefault(ch, len(vocab))
    return vocab


def test_view_of_plain_text_is_the_text():
    text = "שלום עולם, מה שלומך? 15 ק״ג"
    view = build_encoder_view(text)
    assert view.chars == text.replace("\u05f4", '"')  # only the gershayim is rewritten
    assert view.source == list(range(len(text)))
    assert view.spans == [(i, i + 1) for i in range(len(text))]


def test_view_deletes_niqqud_and_keeps_a_map_back():
    text = "\u05e9\u05c1\u05b8\u05dc\u05d5\u05b9\u05dd"  # שָׁלוֹם: shin + dot + qamats ...
    view = build_encoder_view(text)
    assert view.chars == "שלום"
    assert view.source == [0, 3, 4, 6]
    assert view.spans == [(0, 1), (3, 4), (4, 5), (6, 7)]
    # The deleted marks have no token; they take the token of the character before them.
    owner = extend_over_marks(text, char_owner(len(text), view.spans))
    assert owner.tolist() == [0, 0, 0, 1, 2, 2, 3]


def test_view_trailing_marks_join_the_last_letter():
    text = "אַתְּ כן"  # the dagesh and sheva after the final letter
    view = build_encoder_view(text)
    assert view.chars == "את כן"
    owner = extend_over_marks(text, char_owner(len(text), view.spans))
    assert owner.tolist() == [0, 0, 1, 1, 1, 2, 3, 4]  # the space is a token of its own


def test_view_maqaf_paseq_and_sof_pasuq_are_kept():
    text = "בית\u05beספר\u05c0\u05c3"
    assert build_encoder_view(text).chars == text


def test_view_presentation_forms():
    # U+FB2A shin with shin dot, U+FB2E alef with patah, U+FB1F yod-yod patah.
    text = "\ufb2a\u05dc \ufb2e\u05dc \ufb1f"
    view = build_encoder_view(text)
    assert view.chars == "של אל \u05f2"
    assert view.source == [0, 1, 2, 3, 4, 5, 6]  # each form maps to exactly its own index
    assert view.spans == [(i, i + 1) for i in range(7)]


def test_view_lowercases_latin_and_leaves_digits():
    view = build_encoder_view("NASA KNU a1B2")
    assert view.chars == "nasa knu a1b2"
    assert build_encoder_view("İ").chars == "i\u0307"  # NFD first, then lowercase per character


def test_view_one_to_one_replacements():
    # Ellipsis and non-breaking space are not in the vocabulary.
    view = build_encoder_view("a\u00a0b\u2026")
    assert view.chars == "a b."
    assert view.source == [0, 1, 2, 3]
    # Other whitespace becomes a plain space; the sentence structure is untouched.
    assert build_encoder_view("a\tb\nc\u2009d").chars == "a b c d"
    # Geresh, gershayim, the acute accent and the backtick: renikud_onnx.py's own folds.
    assert build_encoder_view("ג\u05f3ינס צה\u05f4ל ` \u00b4").chars == "ג'ינס צה\"ל ' '"


def test_view_keeps_what_the_vocabulary_has():
    # RTL/LTR marks, curly quotes, en/em dashes: in the vocabulary, so passed through. (With
    # `fold_text` the final string has the plain forms already.)
    text = "\u200fא\u200e \u201cב\u201d \u2013 \u2014 \u2019"
    assert build_encoder_view(text).chars == text


def test_view_nfd_of_accented_latin_keeps_offsets():
    view = build_encoder_view("xé!")  # é is one character, e + U+0301 after NFD
    assert view.chars == "xe\u0301!"
    assert view.source == [0, 1, 1, 2]
    assert view.spans == [(0, 1), (1, 2), (1, 1), (2, 3)]  # the accent does not vote
    # The same text written decomposed: the accent is an uncovered mark of the letter.
    decomposed = "xe\u0301!"
    view = build_encoder_view(decomposed)
    assert view.chars == "xe\u0301!"
    assert view.spans == [(0, 1), (1, 2), (1, 1), (3, 4)]
    owner = extend_over_marks(decomposed, char_owner(len(decomposed), view.spans))
    assert owner.tolist() == [0, 1, 1, 3]


@pytest.mark.parametrize(
    "text",
    [
        "e\u0301\u0323 a\u0323\u0301",  # two accents in canonical order and not
        "\u1e69 \u1e0b\u0323",  # precomposed with two parts
        "\ufb2c\u05bc",  # shin with dagesh and shin dot (three marks)
        "\u0301e",  # a string that starts with a mark
        "고 \ud55c",  # Hangul
    ],
)
def test_view_is_nfd_of_the_whole_string(text):
    view = build_encoder_view(text)
    reference = unicodedata.normalize("NFD", text)
    reference = "".join(c for c in reference if not (0x591 <= ord(c) <= 0x5C7 and c in _MARKS))
    assert view.chars == reference.lower()
    assert len(view.chars) == len(view.source) == len(view.spans)
    assert all(0 <= b <= e <= len(text) for b, e in view.spans)


_MARKS = {chr(c) for c in [*range(0x591, 0x5BE), 0x5BF, 0x5C1, 0x5C2, 0x5C4, 0x5C5, 0x5C7]}


def test_tokenizer_one_token_per_character_unknown_does_not_vote():
    vocab = make_vocab("שלום ab")
    tokenizer = RenikudTokenizer(vocab, 1, 2)
    ids, spans = tokenizer.tokenize("של ZZ\u2603")  # Z -> z (unknown here), snowman unknown
    assert ids[:2] == [vocab["ש"], vocab["ל"]]
    assert ids[2] == vocab[" "]
    assert ids[3:] == [0, 0, 0]
    assert spans[:3] == [(0, 1), (1, 2), (2, 3)]
    assert spans[3:] == [(3, 3), (4, 4), (5, 5)]
    assert tokenizer.space_id == vocab[" "]


# ------------------------------------------------------------------ voting with chars


def _encoder(vocab, hidden=HIDDEN, tapped=(2, 3), max_tokens=2048, calls=None):
    """A RenikudPlusEncoder whose 'model' returns, for token t at position p, a vector that
    identifies (layer, p, t)."""
    tokenizer = RenikudTokenizer(vocab, 1, 2)

    def runner(ids):
        if calls is not None:
            calls.append(list(ids))
        out = np.zeros((len(tapped), len(ids), hidden), dtype=np.float32)
        for li, layer in enumerate(tapped):
            for pos, token in enumerate(ids):
                out[li, pos, 0] = layer
                out[li, pos, 1] = pos
                out[li, pos, 2] = token
        return out

    return RenikudPlusEncoder(tokenizer, runner, tapped, hidden, max_tokens)


def test_char_level_voting_covers_every_visible_character(sp):
    encoder = _encoder(make_vocab(*CORPUS))
    for text in CORPUS:
        ids, spans = encoder.tokenize(text)
        sp_ids, sp_spans = semantic_module.sentencepiece_spans(sp, text)
        owner = extend_over_marks(text, char_owner(len(text), spans))
        visible = torch.tensor([not c.isspace() for c in text])
        assert bool((owner[visible] >= 0).all()), text
        hidden = encoder.encode(ids, [2], torch.device("cpu"))[0]
        pooled, cover = pool_to_tokens(
            hidden,
            owner,
            char_owner(len(text), sp_spans),
            len(sp_ids),
            torch.zeros(HIDDEN),
            visible,
        )
        has_chars = torch.tensor([any(not c.isspace() for c in text[b:e]) for b, e in sp_spans])
        assert torch.allclose(cover[has_chars], torch.ones(int(has_chars.sum()))), text
        assert (pooled[~has_chars] == 0).all()


def test_voting_means_the_characters_of_a_piece(sp):
    text = "שלום עולם"
    vocab = make_vocab(text)
    encoder = _encoder(vocab)
    ids, spans = encoder.tokenize(text)
    sp_ids, sp_spans = semantic_module.sentencepiece_spans(sp, text)
    hidden = encoder.encode(ids, [3], torch.device("cpu"))[0]
    owner = char_owner(len(text), spans)
    pooled, _cover = pool_to_tokens(
        hidden,
        owner,
        char_owner(len(text), sp_spans),
        len(sp_ids),
        torch.zeros(HIDDEN),
        torch.tensor([not c.isspace() for c in text]),
    )
    for i, (b, e) in enumerate(sp_spans):
        letters = [j for j in range(b, e) if not text[j].isspace()]
        if letters:
            # Component 1 of a state is its position in the chunk: the mean over the letters.
            assert pooled[i, 1] == pytest.approx(float(np.mean(letters)) + 1)  # +1: [CLS]


def test_fusion_runs_with_a_renikud_encoder_and_is_identity_at_init(sp):
    model = flow_lm(sp, small_config(encoder="renikud_plus"))
    fusion = model.semantic_fusion
    fusion.set_encoder(_encoder(make_vocab(*CORPUS)))
    text = CORPUS[4]  # vocalized
    tokens = model.conditioner.prepare(text).tokens
    plain = model.conditioner(TokenizedText(tokens))
    assert torch.equal(fusion(text, tokens, plain, sp), plain)
    with torch.no_grad():
        torch.nn.init.normal_(fusion.proj.weight)
    fusion.train()
    fused = fusion(text, tokens, plain, sp)
    assert not torch.equal(fused, plain)
    fused.square().sum().backward()
    assert fusion.proj.weight.grad.abs().sum() > 0


def test_renikud_encoder_rejects_layers_it_did_not_tap():
    encoder = _encoder(make_vocab("ab"), tapped=(2, 3))
    with pytest.raises(ValueError, match="not tapped"):
        encoder.encode([5], [4], torch.device("cpu"))
    out = encoder.encode([5, 6], [3, 2], torch.device("cpu"))  # any order, from the tapped set
    assert out.shape == (2, 2, HIDDEN) and out[0, 0, 0] == 3 and out[1, 0, 0] == 2


# ----------------------------------------------------------------------------- windowing


def test_split_windows_are_contiguous_and_bounded():
    for n in (0, 1, 5, 6, 7, 40):
        ids = [7 if i % 4 == 3 else 5 for i in range(n)]  # a space every fourth token
        ranges = split_windows(ids, 6, space_id=7)
        assert all(0 < e - b <= 6 for b, e in ranges)
        assert [b for b, _ in ranges][1:] == [e for _, e in ranges][:-1]
        assert (ranges[0][0], ranges[-1][1]) == (0, n) if n else ranges == []
        for _, end in ranges[:-1]:
            assert ids[end - 1] == 7  # a window that has a space in its second half ends on it


def test_split_windows_without_a_space_cuts_at_the_limit():
    assert split_windows([5] * 14, 6, space_id=7) == [(0, 6), (6, 12), (12, 14)]
    assert split_windows([5] * 14, 6, space_id=None) == [(0, 6), (6, 12), (12, 14)]


def test_long_input_is_encoded_in_windows_with_cls_and_sep(caplog):
    text = ("שלום עולם " * 10).strip()
    vocab = make_vocab(text)
    calls: list[list[int]] = []
    encoder = _encoder(vocab, max_tokens=12, calls=calls)  # windows of 10 tokens
    ids, _ = encoder.tokenize(text)
    with caplog.at_level("WARNING"):
        hidden = encoder.encode(ids, [2, 3], torch.device("cpu"))
        encoder.encode(ids, [2], torch.device("cpu"))
    assert hidden.shape == (2, len(ids), HIDDEN)
    assert all(len(chunk) <= 12 and chunk[0] == 1 and chunk[-1] == 2 for chunk in calls)
    assert [t for chunk in calls[: len(calls) // 2] for t in chunk[1:-1]] == ids
    assert caplog.text.count("independent windows") == 1  # reported once
    # Positions restart in every window: component 1 is the position within the chunk.
    assert hidden[0, 0, 1] == 1 and hidden[0, -1, 1] >= 1
    # A text that fits runs in one pass.
    calls.clear()
    short = _encoder(vocab, calls=calls)
    short.encode(ids, [2], torch.device("cpu"))
    assert len(calls) == 1


def test_empty_text_has_no_tokens():
    encoder = _encoder(make_vocab("a"))
    ids, spans = encoder.tokenize("")
    assert ids == [] and spans == []
    assert encoder.encode(ids, [2], torch.device("cpu")).shape == (1, 0, HIDDEN)


# ------------------------------------------------------------------------- gate metrics


@pytest.mark.parametrize(
    "piece,expected",
    [
        ("▁שלום", "hebrew"),
        ("שָׁלוֹם", "hebrew"),
        ("15", "digit"),
        ("▁1,5", "digit"),
        ("▁NASA", "latin"),
        ("é", "latin"),
        ("ב-1", "hebrew"),  # tie: Hebrew wins
        ("ב-15", "digit"),
        ("ה-API", "latin"),  # more Latin letters than Hebrew
        ("▁", "other"),
        (".", "other"),
        ("", "other"),
        ("₪", "other"),
    ],
)
def test_piece_class(piece, expected):
    assert piece_class(piece) == expected


def test_gate_is_reported_per_character_class(sp):
    model = flow_lm(sp, small_config())
    fusion = model.semantic_fusion
    fusion.train()
    text = "שלום 15 NASA, שלום!"
    tokens = model.conditioner.prepare(text).tokens
    fusion(text, tokens, model.conditioner(TokenizedText(tokens)), sp)
    stats = fusion.pop_stats()
    for key in (
        "semantic_gate",
        "semantic_gate_latin",
        "semantic_gate_other",
        "semantic_cover",
        *(f"semantic_gate_class_{name}" for name in CHAR_CLASSES),
    ):
        assert key in stats, key
        assert stats[key].ndim == 0
    assert stats["semantic_gate_class_hebrew"] == pytest.approx(0.5)  # the init
    # A text without a class does not report it.
    fusion(CORPUS[0], *_prepared(model, sp, CORPUS[0]))
    stats = fusion.pop_stats()
    assert "semantic_gate_class_digit" not in stats and "semantic_gate_class_latin" not in stats
    assert "semantic_gate_class_hebrew" in stats and "semantic_gate_latin" not in stats


def _prepared(model, sp, text):
    tokens = model.conditioner.prepare(text).tokens
    return tokens, model.conditioner(TokenizedText(tokens)), sp


def test_per_class_gate_values_follow_the_pieces(sp):
    """The class means are over the right pieces: set the gate from a hand-made function of the
    piece class and read it back."""
    model = flow_lm(sp, small_config())
    fusion = model.semantic_fusion
    fusion.train()
    text = "שלום 15 NASA"
    tokens = model.conditioner.prepare(text).tokens
    _, spans = semantic_module.sentencepiece_spans(sp, text)
    classes = [piece_class(text[b:e]) for b, e in spans]
    assert {"hebrew", "digit", "latin"} <= set(classes)
    level = {"hebrew": 2.0, "digit": -2.0, "latin": 0.0, "other": 4.0}
    values = torch.tensor([level[c] for c in classes])

    class Gate(torch.nn.Module):
        def forward(self, x):
            return values[:, None]

    fusion.gate = Gate()
    fusion(text, tokens, model.conditioner(TokenizedText(tokens)), sp)
    stats = fusion.pop_stats()
    for name, raw in level.items():
        if name in classes:
            assert stats[f"semantic_gate_class_{name}"] == pytest.approx(
                float(torch.sigmoid(torch.tensor(raw)))
            )


# --------------------------------------------------------------------- checkpoint identity


def test_check_encoder_record():
    dicta = SemanticConditionerConfig().identity()
    renikud = SemanticConditionerConfig(encoder="renikud_plus").identity()
    check_encoder_record(dicta, dicta)
    check_encoder_record(renikud, renikud)
    check_encoder_record(dicta, None)  # written before the choice existed: DictaBERT's
    with pytest.raises(ValueError, match="renikud_plus.*dictabert|dictabert.*renikud_plus"):
        check_encoder_record(renikud, dicta, "a.safetensors")
    with pytest.raises(ValueError, match="trained with encoder dictabert"):
        check_encoder_record(renikud, None)
    with pytest.raises(ValueError, match="trained with encoder renikud_plus"):
        check_encoder_record(dicta, renikud)
    # The same family but other weights is a mismatch too.
    other_revision = {**dicta, "revision": "0" * 40}
    with pytest.raises(ValueError, match="revision"):
        check_encoder_record(dicta, other_revision)
    other_file = {**renikud, "filename": "model_int8.onnx"}
    with pytest.raises(ValueError, match="filename"):
        check_encoder_record(renikud, other_file)


def test_layer_band_mismatch_only_warns(caplog):
    dicta = SemanticConditionerConfig().identity()
    with caplog.at_level("WARNING"):
        check_encoder_record(dicta, {**dicta, "layers": "9-21"})
    assert "layers 9-21" in caplog.text


def test_loading_across_encoders_fails_before_anything_is_loaded(sp):
    dicta, renikud = flow_lm(sp, small_config()), flow_lm(sp, small_config(encoder="renikud_plus"))
    with torch.no_grad():
        dicta.semantic_fusion.proj.weight.fill_(0.5)
    state = dicta.state_dict()
    record = dicta.semantic_fusion.config.identity()
    with pytest.raises(ValueError, match="trained with encoder dictabert"):
        load_state_dict_allowing_semantic(renikud, state, recorded_encoder=record)
    with pytest.raises(ValueError, match="trained with encoder dictabert"):
        load_state_dict_allowing_semantic(renikud, state, recorded_encoder=None)  # legacy
    assert renikud.semantic_fusion.proj.weight.abs().sum() == 0
    # Same encoder, and a caller that does not pass a record, load.
    load_state_dict_allowing_semantic(dicta, state, recorded_encoder=record)
    load_state_dict_allowing_semantic(flow_lm(sp, small_config()), state, recorded_encoder=None)
    load_state_dict_allowing_semantic(renikud, state)
    # Plain <-> semantic rules are unchanged whatever the encoder.
    plain = flow_lm(sp, None)
    load_state_dict_allowing_semantic(plain, state, recorded_encoder=record)
    load_state_dict_allowing_semantic(renikud, plain.state_dict(), recorded_encoder=None)


def test_safetensors_export_records_the_encoder(sp, tmp_path):
    import safetensors.torch

    model = flow_lm(sp, small_config(encoder="renikud_plus"))
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file(model.state_dict(), str(path), metadata=encoder_metadata(model))
    record = read_encoder_record(path)
    assert record == model.semantic_fusion.config.identity()
    assert json.loads(safetensors.safe_open(str(path), "pt").metadata()[ENCODER_METADATA_KEY])
    # No record without semantic conditioning (and no metadata written at all).
    plain = flow_lm(sp, None)
    assert encoder_metadata(plain) is None
    path = tmp_path / "plain.safetensors"
    safetensors.torch.save_file(plain.state_dict(), str(path), metadata=encoder_metadata(plain))
    assert read_encoder_record(path) is None


def test_tts_model_loading_checks_the_export(sp, tmp_path):
    import safetensors.torch

    from pocket_tts.models.tts_model import TTSModel

    dicta = flow_lm(sp, small_config())
    path = tmp_path / "f.safetensors"
    safetensors.torch.save_file(dicta.state_dict(), str(path), metadata=encoder_metadata(dicta))
    renikud = flow_lm(sp, small_config(encoder="renikud_plus"))
    with pytest.raises(ValueError, match="dictabert"):
        load_state_dict_allowing_semantic(
            renikud,
            safetensors.torch.load_file(path),
            recorded_encoder=read_encoder_record(path),
            source=str(path),
        )
    assert TTSModel  # the wiring itself is exercised by training/tests/test_semantic_training.py


# --------------------------------------------------------------------------- graph surgery

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def toy_graph(layers=3, hidden=4, vocab=12, extra_norm=False, broken=False):
    """A post-LayerNorm BERT-shaped graph: embedding LN, two LNs per layer, FiLM from a
    `speaker` input, a final LN and a head. Layer k's output tensor is named `ln_{k}`."""
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    inits, nodes = [], []

    def weight(name, *shape):
        inits.append(numpy_helper.from_array(rng.normal(size=shape).astype(np.float32), name))
        return name

    def ln(x, out):
        scale = weight(out + "_s", hidden)
        bias = weight(out + "_b", hidden)
        nodes.append(
            helper.make_node("LayerNormalization", [x, scale, bias], [out], name="n_" + out)
        )
        return out

    nodes.append(helper.make_node("Gather", [weight("emb", vocab, hidden), "input_ids"], ["x0"]))
    nodes.append(helper.make_node("Cast", ["attention_mask"], ["m"], to=TensorProto.FLOAT))
    inits.append(numpy_helper.from_array(np.array([2], dtype=np.int64), "ax"))
    nodes.append(helper.make_node("Unsqueeze", ["m", "ax"], ["m3"]))
    nodes.append(helper.make_node("Mul", ["x0", "m3"], ["x0m"]))
    prev = ln("x0m", "emb_ln")
    for k in range(1, layers + 1):
        nodes.append(
            helper.make_node("MatMul", [prev, weight(f"wa{k}", hidden, hidden)], [f"a{k}"])
        )
        residual = "emb_ln" if broken and k == 2 else prev
        nodes.append(helper.make_node("Add", [residual, f"a{k}"], [f"ra{k}"]))
        mid = ln(f"ra{k}", f"mid_{k}")
        nodes.append(helper.make_node("MatMul", [mid, weight(f"wf{k}", hidden, hidden)], [f"f{k}"]))
        nodes.append(helper.make_node("Add", [mid, f"f{k}"], [f"rf{k}"]))
        prev = ln(f"rf{k}", f"ln_{k}")
        if extra_norm and k == 2:
            prev = ln(prev, "stray")  # breaks the embedding / 2 per layer / final structure
    nodes.append(helper.make_node("Gather", [weight("spk", 3, hidden), "speaker"], ["s"]))
    nodes.append(helper.make_node("Unsqueeze", ["s", "ax1"], ["s3"]))
    inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "ax1"))
    nodes.append(helper.make_node("Mul", [prev, "s3"], ["film_m"]))
    nodes.append(helper.make_node("Add", [prev, "film_m"], ["film"]))
    final = ln("film", "final_ln")
    nodes.append(helper.make_node("MatMul", [final, weight("head", hidden, 5)], ["logits"]))
    graph = helper.make_graph(
        nodes,
        "toy",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq_len"]),
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, ["batch", "seq_len"]
            ),
            helper.make_tensor_value_info("speaker", TensorProto.INT64, ["batch"]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "seq_len", 5])],
        initializer=inits,
    )
    # ir_version 10: what the real model uses, and below onnxruntime's maximum.
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10)


def run(model, names, ids=(3, 1, 4, 1, 5), speaker=0):
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        "speaker": np.array([speaker], dtype=np.int64),
    }
    feeds = {k: v for k, v in feeds.items() if k in {i.name for i in session.get_inputs()}}
    return session.run(names, feeds), [i.name for i in session.get_inputs()]


def test_layer_outputs_are_found_by_structure():
    model = toy_graph()
    assert find_layer_outputs(model.graph, 3) == ["ln_1", "ln_2", "ln_3"]
    with pytest.raises(ValueError, match="expected 6"):
        find_layer_outputs(model.graph, 2)
    with pytest.raises(ValueError, match="expected 50"):
        find_layer_outputs(model.graph)  # 24 layers is the default
    with pytest.raises(ValueError, match="found 9"):
        find_layer_outputs(toy_graph(extra_norm=True).graph, 3)
    with pytest.raises(ValueError, match="does not follow"):
        find_layer_outputs(toy_graph(broken=True).graph, 3)


def test_pruned_graph_gives_the_same_layer_states_without_the_heads():
    reference = toy_graph()
    full, names = tap_layer_outputs(toy_graph(), [1, 3, 2], expected_layers=3, prune=False)
    assert names == ["ln_1", "ln_3", "ln_2"]
    (logits0,), _ = run(reference, ["logits"])
    outputs, _ = run(full, ["logits", *names])
    np.testing.assert_array_equal(outputs[0], logits0)  # the original output is untouched

    pruned, names = tap_layer_outputs(toy_graph(), [2, 3], expected_layers=3)
    states, inputs = run(pruned, names)
    assert inputs == ["input_ids", "attention_mask"]  # no speaker: FiLM is above the band
    assert [s.shape for s in states] == [(1, 5, 4)] * 2
    np.testing.assert_allclose(states[0], outputs[3], rtol=1e-5, atol=1e-6)  # layer 2
    np.testing.assert_allclose(states[1], outputs[2], rtol=1e-5, atol=1e-6)  # layer 3
    initializers = {i.name for i in pruned.graph.initializer}
    assert "head" not in initializers and "spk" not in initializers
    assert "wa3" in initializers and "wf3" in initializers
    # Pruning to a lower layer drops the layers above it.
    low, names = tap_layer_outputs(toy_graph(), [1], expected_layers=3)
    assert "wa2" not in {i.name for i in low.graph.initializer}
    assert sum(n.op_type == "LayerNormalization" for n in low.graph.node) == 3


def test_states_do_not_depend_on_the_speaker_input():
    full, names = tap_layer_outputs(toy_graph(), [1, 2, 3], expected_layers=3, prune=False)
    a, _ = run(full, names, speaker=0)
    b, _ = run(full, names, speaker=2)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    (la,), _ = run(full, ["logits"], speaker=0)
    (lb,), _ = run(full, ["logits"], speaker=2)
    assert not np.allclose(la, lb)  # the heads do


def test_invalid_layers_are_rejected():
    with pytest.raises(ValueError, match="layers"):
        tap_layer_outputs(toy_graph(), [0], expected_layers=3)
    with pytest.raises(ValueError, match="layers"):
        tap_layer_outputs(toy_graph(), [4], expected_layers=3)


# ------------------------------------------------------------------ the real model (slow)


def _real_onnx() -> str | None:
    path = os.environ.get("RENIKUD_ONNX")
    if path:
        return path if Path(path).exists() else None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            RENIKUD_PLUS, "model.onnx", revision=RENIKUD_PLUS_REVISION, local_files_only=True
        )
    except Exception:  # noqa: BLE001 - an unavailable optional cache must skip the integration test
        return None


@pytest.mark.skipif(_real_onnx() is None, reason="the pinned fp32 RenikudPlus ONNX is not cached")
def test_real_model_taps_all_24_layers_and_leaves_the_outputs_alone():
    path = _real_onnx()
    original = onnx.load(path)
    meta = {e.key: e.value for e in original.metadata_props}
    vocab = json.loads(meta["vocab"])
    tokenizer = RenikudTokenizer(vocab, int(meta["cls_token_id"]), int(meta["sep_token_id"]))
    text = "הילדה רצה הביתה כי התחיל לרדת גשם."
    ids, _ = tokenizer.tokenize(text)
    chunk = [tokenizer.cls_id, *ids, tokenizer.sep_id]
    seq = len(chunk)

    outputs = find_layer_outputs(original.graph)
    assert len(outputs) == 24 and len(set(outputs)) == 24
    assert outputs[0] == "layer_norm_2" and outputs[23] == "layer_norm_48"  # as observed

    def feeds(session):
        values = {
            "input_ids": np.array([chunk], dtype=np.int64),
            "attention_mask": np.ones((1, seq), dtype=np.int64),
            "speaker": np.zeros(1, dtype=np.int64),
            "target_speaker": np.zeros(1, dtype=np.int64),
        }
        return {i.name: values[i.name] for i in session.get_inputs()}

    heads = ["consonant_logits", "vowel_logits", "stress_logits"]
    reference = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    expected = reference.run(heads, feeds(reference))
    del reference

    full, names = tap_layer_outputs(original, list(range(1, 25)), prune=False)
    session = ort.InferenceSession(full.SerializeToString(), providers=["CPUExecutionProvider"])
    got = session.run([*heads, *names], feeds(session))
    del session
    for a, b in zip(expected, got[:3], strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)  # unchanged outputs
    assert [g.shape for g in got[3:]] == [(1, seq, 1024)] * 24
    assert all(np.isfinite(g).all() for g in got[3:])

    config = SemanticConditionerConfig(
        encoder="renikud_plus", first_layer=8, last_layer=20, onnx_provider="cpu"
    )
    encoder = RenikudPlusEncoder.from_config(config)
    assert encoder.hidden_size == 1024 and len(encoder.tapped) == 13
    states = encoder.encode(ids, list(range(8, 21)), torch.device("cpu"))
    assert states.shape == (13, len(ids), 1024)
    for k in range(8, 21):  # the pruned graph agrees with the full one, layer by layer
        np.testing.assert_allclose(
            states[k - 8].numpy(), got[3 + k - 1][0, 1:-1], rtol=1e-3, atol=1e-3
        )


# -------------------------------------------------------------- loading from a local directory


def test_renikud_source_directory_must_hold_the_file(tmp_path):
    config = SemanticConditionerConfig(encoder="renikud_plus")
    with pytest.raises(FileNotFoundError, match="model.onnx not found"):
        RenikudPlusEncoder.from_config(config, tmp_path)


def test_dictabert_loads_from_a_local_directory(tmp_path, monkeypatch):
    """The files the deploy server copies from its store are all from_pretrained needs: no Hub,
    no revision (offline mode proves nothing is fetched)."""
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "[PAD]": 3, "[MASK]": 4, "שלום": 5, "עולם": 6}
    core = tokenizers.Tokenizer(tokenizers.models.WordPiece(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=core,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )
    fast.save_pretrained(tmp_path)
    transformers.BertModel(
        transformers.BertConfig(
            vocab_size=len(vocab),
            hidden_size=HIDDEN,
            num_hidden_layers=4,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
        ),
        add_pooling_layer=False,
    ).save_pretrained(tmp_path)

    from pocket_tts.conditioners.semantic import DictaBertEncoder

    encoder = DictaBertEncoder(
        "dicta-il/dictabert-large", "pinned", 16, keep_layers=3, source=tmp_path
    )
    assert encoder.hidden_size == HIDDEN and len(encoder.model.encoder.layer) == 3
    ids, spans = encoder.tokenize("שלום עולם")
    assert ids == [5, 6] and spans == [(0, 4), (5, 9)]
    assert encoder.encode(ids, [1, 3], torch.device("cpu")).shape == (2, 2, HIDDEN)


@pytest.mark.skipif(_real_onnx() is None, reason="the pinned fp32 RenikudPlus ONNX is not cached")
def test_real_model_loads_from_a_local_directory():
    """What the deploy server does: the file comes from a directory, not the Hub."""
    directory = Path(_real_onnx()).parent
    config = SemanticConditionerConfig(
        encoder="renikud_plus", first_layer=8, last_layer=9, onnx_provider="cpu"
    )
    local = RenikudPlusEncoder.from_config(config, directory)
    ids, _ = local.tokenize("הילדה רצה הביתה.")
    assert local.encode(ids, [8, 9], torch.device("cpu")).shape == (2, len(ids), 1024)
