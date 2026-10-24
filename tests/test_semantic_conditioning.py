"""Optional DictaBERT semantic conditioning (pocket_tts/conditioners/semantic.py).

No network and no GPU: a tiny SentencePiece model is trained in-process and a fake encoder
stands in for DictaBERT.
"""

import copy
import io
import queue
from types import SimpleNamespace

import pytest
import sentencepiece
import torch
from torch import nn

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.conditioners.semantic import (
    SemanticEncoder,
    SemanticFusion,
    byte_to_char_spans,
    char_owner,
    extend_over_marks,
    fuse_rows,
    load_state_dict_allowing_semantic,
    pool_to_tokens,
    sentencepiece_spans,
)
from pocket_tts.conditioners.text import LUTConditioner
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.modules.mlp import SimpleMLPAdaLN
from pocket_tts.modules.transformer import StreamingTransformer
from pocket_tts.utils.config import CONFIGS_DIR, SemanticConditionerConfig, load_config

HIDDEN, DIM, LDIM = 16, 32, 8

CORPUS = [
    "שלום עולם, מה שלומך היום?",
    "הילדה רצה הביתה כי התחיל לרדת גשם.",
    "הוא רצה ללכת הביתה אבל נשאר.",
    "ב-15 ק״ג של תפוחים ו-3 ליטר מים.",
    "שָׁלוֹם עוֹלָם, אַתְּ רוֹצָה לָלֶכֶת הַבַּיְתָה?",
    "He worked at Microsoft and NASA before joining the team.",
    "The quick brown fox jumps over the lazy dog.",
]


def make_sp() -> sentencepiece.SentencePieceProcessor:
    model = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(CORPUS * 20),
        model_writer=model,
        vocab_size=340,  # byte fallback alone takes 256
        byte_fallback=True,
        normalization_rule_name="identity",  # as hebrew_training/train_tokenizer.py
        character_coverage=1.0,
        minloglevel=2,
    )
    return sentencepiece.SentencePieceProcessor(model_proto=model.getvalue())


@pytest.fixture(scope="module")
def sp() -> sentencepiece.SentencePieceProcessor:
    return make_sp()


class FakeEncoder(SemanticEncoder):
    """Whitespace words, trailing combining marks left uncovered like DictaBERT's offsets.
    Hidden states are a fixed function of (token id, position, layer)."""

    hidden_size = HIDDEN
    num_layers = 4

    def __init__(self):
        self.calls = 0

    def tokenize(self, text):
        ids, spans, i = [], [], 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            j = i
            while j < len(text) and not text[j].isspace():
                j += 1
            end = j
            while end > i + 1 and 0x0591 <= ord(text[end - 1]) <= 0x05C7:
                end -= 1  # niqqud after the last letter is not in the span
            ids.append(sum(map(ord, text[i:j])) % 997)
            spans.append((i, end))
            i = j
        return ids, spans

    def encode(self, ids, layers, device):
        self.calls += 1
        out = torch.zeros(len(layers), len(ids), HIDDEN, device=device)
        for li, layer in enumerate(layers):
            for pos, token in enumerate(ids):
                g = torch.Generator().manual_seed(token * 131 + pos * 7 + layer)
                out[li, pos] = torch.randn(HIDDEN, generator=g)
        return out


def small_config(**kw) -> SemanticConditionerConfig:
    return SemanticConditionerConfig(
        **{"hidden_size": HIDDEN, "first_layer": 2, "last_layer": 3, "gate_hidden": 8, **kw}
    )


class SPConditioner(LUTConditioner):
    """LUTConditioner over an in-memory SentencePiece model."""

    def __init__(self, sp: sentencepiece.SentencePieceProcessor, dim: int):
        nn.Module.__init__(self)
        self.dim = dim
        self.output_dim = dim
        self.tokenizer = SimpleNamespace(sp=sp)
        self.embed = nn.Embedding(sp.vocab_size() + 1, dim)

    def prepare(self, x: str) -> TokenizedText:
        return TokenizedText(torch.tensor(self.tokenizer.sp.encode(x))[None])


def flow_lm(sp, semantic: SemanticConditionerConfig | None) -> FlowLMModel:
    torch.manual_seed(0)
    fusion = None
    if semantic is not None:
        fusion = SemanticFusion(semantic, DIM)
        fusion.set_encoder(FakeEncoder())
    torch.manual_seed(1)
    model = FlowLMModel(
        conditioner=SPConditioner(sp, DIM),
        flow_net=SimpleMLPAdaLN(LDIM, 16, LDIM, DIM, 2, 2),
        transformer=StreamingTransformer(
            d_model=DIM, num_heads=2, num_layers=2, dim_feedforward=64
        ),
        dim=DIM,
        ldim=LDIM,
        insert_bos_before_voice=True,
        semantic_fusion=fusion,
    )
    model.speaker_proj_weight = nn.Parameter(torch.randn(DIM, LDIM) * 0.1)
    return model


# ---------------------------------------------------------------------------------- offsets


def test_byte_to_char_spans_with_hebrew_and_niqqud():
    text = "שָׁלוֹם a"  # 7 Hebrew code points (2 bytes each), a space, a Latin letter
    assert len(text) == 9
    assert byte_to_char_spans(text, [(0, 14), (14, 15), (15, 16)]) == [(0, 7), (7, 8), (8, 9)]
    # A letter and its marks inside a word: שָׁ is bytes 0..6, characters 0..3.
    assert byte_to_char_spans(text, [(0, 6), (6, 14)]) == [(0, 3), (3, 7)]
    # Empty spans (byte-fallback pieces) stay empty.
    assert byte_to_char_spans(text, [(6, 6)]) == [(3, 3)]


@pytest.mark.parametrize("text", CORPUS + ["  הוא עבד ב-Microsoft  NASA ", "a\u00a0b\tc ﬁve ２０"])
def test_sentencepiece_spans_match_the_encoding(sp, text):
    ids, spans = sentencepiece_spans(sp, text)
    assert ids == sp.encode(text)
    assert len(spans) == len(ids)
    previous = 0
    for begin, end in spans:
        assert previous <= begin <= end <= len(text)
        previous = end
    # The pieces cover the text, give or take the whitespace SentencePiece drops at the ends.
    assert "".join(text[b:e] for b, e in spans) == text.strip()


# -------------------------------------------------------------------------------- alignment


def test_partial_overlap_worked_example():
    text = "ב-15 ק״ג"
    # s_0 = "ב-15" (chars 0..4), s_1 = " ק״ג" (4..8); d_0 = "ב" (0..1), d_1 = "15 ק״ג" (2..8).
    char_to_s = char_owner(len(text), [(0, 4), (4, 8)])
    char_to_d = char_owner(len(text), [(0, 1), (2, 8)])
    latents = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    visible = torch.tensor([not c.isspace() for c in text])
    pooled, cover = pool_to_tokens(latents, char_to_d, char_to_s, 2, torch.zeros(2), visible)
    # s_0: chars 0 (d_0), 1 (uncovered "-"), 2 and 3 (d_1) -> (L0 + 2 L1) / 3.
    torch.testing.assert_close(pooled[0], torch.tensor([1 / 3, 2 / 3]))
    # s_1: the space is invisible, the three letters are all d_1.
    torch.testing.assert_close(pooled[1], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(cover, torch.tensor([3 / 4, 1.0]))


def test_uncovered_tokens_get_the_null_vector():
    text = "ab - cd"
    char_to_s = char_owner(len(text), [(0, 2), (2, 4), (4, 7)])
    char_to_d = char_owner(len(text), [(0, 2), (5, 7)])  # "-" is not covered ([UNK], dropped)
    null = torch.tensor([9.0, 9.0])
    visible = torch.tensor([not c.isspace() for c in text])
    latents = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    pooled, cover = pool_to_tokens(latents, char_to_d, char_to_s, 3, null, visible)
    torch.testing.assert_close(pooled[1], null)
    torch.testing.assert_close(cover, torch.tensor([1.0, 0.0, 1.0]))
    torch.testing.assert_close(pooled[2], latents[1])


def test_whitespace_only_token_has_zero_cover():
    text = "ab  cd"
    char_to_s = char_owner(len(text), [(0, 2), (2, 3), (3, 6)])
    char_to_d = char_owner(len(text), [(0, 2), (4, 6)])
    visible = torch.tensor([not c.isspace() for c in text])
    pooled, cover = pool_to_tokens(
        torch.eye(2), char_to_d, char_to_s, 3, torch.full((2,), 5.0), visible
    )
    assert cover[1] == 0 and torch.equal(pooled[1], torch.full((2,), 5.0))


def test_trailing_niqqud_joins_the_word_before_it():
    text = "אַתְּ כן"  # אַתְּ = א ַ ת ְ ּ : the last two marks follow the final letter
    owner = char_owner(len(text), [(0, 3), (6, 8)])
    assert owner.tolist()[:5] == [0, 0, 0, -1, -1]
    extended = extend_over_marks(text, owner)
    assert extended.tolist() == [0, 0, 0, 0, 0, -1, 1, 1]  # the space stays uncovered


# ----------------------------------------------------------------------------------- fusion


def test_zero_init_fusion_returns_the_lookup_table_exactly(sp):
    model = flow_lm(sp, small_config())
    text = CORPUS[4]
    tokens = model.conditioner.prepare(text).tokens
    plain = model.conditioner(TokenizedText(tokens))
    fused = model.semantic_fusion(text, tokens, plain, sp)
    assert fused.shape == plain.shape
    assert torch.equal(fused, plain)


def test_plain_and_semantic_models_agree_at_init(sp):
    """Same weights everywhere else: the semantic model is the plain model at step 0."""
    plain = flow_lm(sp, None)
    semantic = flow_lm(sp, small_config())
    missing, unexpected = load_state_dict_allowing_semantic(semantic, plain.state_dict())
    assert not missing and not unexpected
    text = CORPUS[1]
    tokens = plain.conditioner.prepare(text).tokens
    a = plain.conditioner(TokenizedText(tokens))
    b = semantic.semantic_fusion(text, tokens, semantic.conditioner(TokenizedText(tokens)), sp)
    assert torch.equal(a, b)


@pytest.mark.parametrize("gate,width", [("scalar", 1), ("vector", DIM)])
def test_gate_shape(sp, gate, width):
    model = flow_lm(sp, small_config(gate=gate))
    fusion = model.semantic_fusion
    assert fusion.gate[-1].out_features == width
    nn.init.normal_(fusion.proj.weight)
    nn.init.normal_(fusion.gate[-1].weight)
    text = CORPUS[5]
    tokens = model.conditioner.prepare(text).tokens
    plain = model.conditioner(TokenizedText(tokens))
    fused = fusion(text, tokens, plain, sp)
    assert fused.shape == plain.shape
    assert not torch.equal(fused, plain)


def test_gradients_reach_the_fusion(sp):
    model = flow_lm(sp, small_config())
    fusion = model.semantic_fusion
    fusion.train()
    text = CORPUS[2]
    tokens = model.conditioner.prepare(text).tokens
    fused = fusion(text, tokens, model.conditioner(TokenizedText(tokens)), sp)
    fused.square().sum().backward()
    assert fusion.proj.weight.grad is not None and fusion.proj.weight.grad.abs().sum() > 0
    # In the graph (DDP needs a gradient for every trainable parameter), possibly zero.
    for name, param in fusion.named_parameters():
        assert param.grad is not None, name
    stats = fusion.pop_stats()
    assert set(stats) >= {"semantic_gate", "semantic_cover"}
    assert fusion.pop_stats() == {}


def test_tokens_that_are_not_the_texts_encoding_are_rejected(sp):
    model = flow_lm(sp, small_config())
    tokens = model.conditioner.prepare(CORPUS[0]).tokens
    emb = model.conditioner(TokenizedText(tokens))
    with pytest.raises(ValueError, match="exact string"):
        model.semantic_fusion(CORPUS[0] + "!", tokens, emb, sp)


def test_fuse_rows_keeps_the_padded_shape(sp):
    model = flow_lm(sp, small_config())
    texts = CORPUS[:3]
    tokens = [torch.tensor(sp.encode(t)) for t in texts]
    width = max(t.shape[0] for t in tokens)
    padded = torch.randn(3, width, DIM)
    fused = fuse_rows(model.semantic_fusion, texts, tokens, padded, sp)
    assert fused.shape == padded.shape
    for b, t in enumerate(tokens):
        torch.testing.assert_close(fused[b, : t.shape[0]], padded[b, : t.shape[0]])
    with pytest.raises(ValueError, match="text of every row"):
        fuse_rows(model.semantic_fusion, None, tokens, padded, sp)


def test_encoder_is_not_part_of_the_module(sp):
    model = flow_lm(sp, small_config())
    keys = [k for k in model.state_dict() if k.startswith("semantic_fusion.")]
    assert keys and not any("encoder" in k for k in keys)
    assert isinstance(model.semantic_fusion.encoder, FakeEncoder)


def test_deepcopy_shares_the_encoder_but_not_the_weights(sp):
    model = flow_lm(sp, small_config())
    clone = copy.deepcopy(model)
    assert clone.semantic_fusion.encoder is model.semantic_fusion.encoder
    assert clone.semantic_fusion.proj.weight is not model.semantic_fusion.proj.weight
    with torch.no_grad():
        clone.semantic_fusion.proj.weight.fill_(1.0)
    assert model.semantic_fusion.proj.weight.abs().sum() == 0


def test_encoder_size_mismatch_is_rejected():
    fusion = SemanticFusion(small_config(hidden_size=HIDDEN * 2), DIM)
    with pytest.raises(ValueError, match="hidden size"):
        fusion.set_encoder(FakeEncoder())
    fusion = SemanticFusion(small_config(last_layer=9), DIM)
    with pytest.raises(ValueError, match="last_layer"):
        fusion.set_encoder(FakeEncoder())


# ------------------------------------------------------------------------- checkpoint compat


def test_plain_checkpoint_loads_into_a_semantic_model(sp):
    plain, semantic = flow_lm(sp, None), flow_lm(sp, small_config())
    load_state_dict_allowing_semantic(semantic, plain.state_dict(), strict=True)
    assert torch.equal(
        semantic.semantic_fusion.proj.weight, torch.zeros_like(semantic.semantic_fusion.proj.weight)
    )


def test_semantic_checkpoint_loads_into_a_plain_model(sp):
    plain, semantic = flow_lm(sp, None), flow_lm(sp, small_config())
    with torch.no_grad():
        semantic.out_eos.bias.fill_(3.0)
    load_state_dict_allowing_semantic(plain, semantic.state_dict(), strict=True)
    assert plain.out_eos.bias.item() == 3.0
    with pytest.raises(RuntimeError):  # without the helper, strict loading refuses it
        plain.load_state_dict(semantic.state_dict(), strict=True)


def test_other_missing_keys_still_fail_strict_loading(sp):
    semantic = flow_lm(sp, small_config())
    state = {k: v for k, v in flow_lm(sp, None).state_dict().items() if k != "bos_emb"}
    with pytest.raises(RuntimeError, match="bos_emb"):
        load_state_dict_allowing_semantic(semantic, state, strict=True)


# ------------------------------------------------------------------------ default unchanged


def test_shipped_configs_have_no_semantic_section_except_its_own():
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        config = load_config(path)
        if path.name == "hebrew_semantic.yaml":
            assert config.flow_lm.semantic_conditioner == SemanticConditionerConfig()
        else:
            assert config.flow_lm.semantic_conditioner is None, path.name


def test_plain_model_has_no_semantic_state(sp):
    model = flow_lm(sp, None)
    assert model.semantic_fusion is None
    assert not any("semantic" in k for k in model.state_dict())
    # Loading through the helper is exactly load_state_dict when neither side has the keys.
    other = flow_lm(sp, None)
    assert load_state_dict_allowing_semantic(other, model.state_dict()) == ([], [])


def test_layer_band_is_validated():
    with pytest.raises(ValueError, match="first_layer"):
        SemanticConditionerConfig(first_layer=0)
    with pytest.raises(ValueError, match="first_layer"):
        SemanticConditionerConfig(first_layer=10, last_layer=9)


def test_generate_passes_the_text_only_with_semantic_conditioning():
    calls = []
    model = SimpleNamespace(
        _flow_lm_current_end=lambda model_state: 0,
        _expand_kv_cache=lambda model_state, sequence_length: None,
        _run_flow_lm_and_increment_step=lambda **kw: calls.append(kw),
        _autoregressive_generation=lambda *a: None,
    )
    for text in (None, "שלום."):
        TTSModel._generate(
            model,
            model_state={},
            prepared=TokenizedText(torch.zeros((1, 2), dtype=torch.long)),
            max_gen_len=1,
            frames_after_eos=1,
            latents_queue=queue.Queue(),
            result_queue=queue.Queue(),
            text=text,
        )
    assert "text" not in calls[0]
    assert calls[1]["text"] == "שלום."


def test_tts_run_flow_lm_fuses_the_prompt_text(sp):
    model = flow_lm(sp, small_config())
    with torch.no_grad():
        nn.init.normal_(model.semantic_fusion.proj.weight)
    seen = {}

    def sample(sequence, text_embeddings, **kw):
        seen["emb"] = text_embeddings
        return torch.zeros(1, LDIM), torch.zeros(1, 1, dtype=torch.bool)

    model._sample_next_latent = sample
    tts = SimpleNamespace(
        flow_lm=model, sampler_decode_steps=1, temp=0.0, noise_clamp=None, eos_threshold=0.0
    )
    text = CORPUS[1]
    tokens = model.conditioner.prepare(text).tokens
    empty_audio = torch.empty(1, 0, DIM)
    latents = torch.empty(1, 0, LDIM)
    TTSModel._run_flow_lm(tts, {}, tokens, latents, empty_audio, text=text)
    plain = model.conditioner(TokenizedText(tokens))
    assert seen["emb"].shape == plain.shape and not torch.equal(seen["emb"], plain)
    with pytest.raises(ValueError, match="needs the text"):
        TTSModel._run_flow_lm(tts, {}, tokens, latents, empty_audio)
    # Audio steps carry no text tokens and need no text.
    TTSModel._run_flow_lm(tts, {}, tokens[:, :0], latents, empty_audio)
