"""Optional semantic conditioning: frozen encoder states fused into the FlowLM text prefix.

An experimental local extension for Hebrew (see SEMANTIC_CONDITIONING.md). It is not part of
Kyutai's released models or training recipe, and it is off unless the model config has a
`flow_lm.semantic_conditioner` section. That section selects the encoder: DictaBERT-large
(the default; `DictaBertEncoder` here) or RenikudPlus (`encoder: renikud_plus`, in
semantic_renikud.py, imported only when selected).

The backbone keeps reading one vector per SentencePiece token, so sequence length, positions
and every length heuristic are unchanged. A frozen bidirectional encoder reads the same string;
its states are normalized, mixed over a band of layers, mapped onto the SentencePiece tokens by
character overlap and added through a gate:

    p_i = mean over the characters of SentencePiece token i of the encoder state covering them
    g_i = sigmoid(MLP([es_i ; p_i ; cover_i]))
    x_i = es_i + g_i * W p_i          W zero-initialized, so x_i == es_i at initialization

With two encoders configured (`flow_lm.semantic_conditioners`, `CombinedSemanticFusion`) each
encoder keeps its own mix, null vector and zero-initialized projection, and one gate network
reads the embedding and both encoders' vectors and emits one gate per encoder:

    g_i = sigmoid(MLP([es_i ; p_i^1 ; p_i^2 ; cover_i^1 ; cover_i^2]))      # one value per encoder
    x_i = es_i + g_i^1 * W1 p_i^1 + g_i^2 * W2 p_i^2

The embedding path stays ungated, so x_i == es_i at initialization here too.

The encoders are deliberately not nn.Module attributes: their weights never enter state_dict,
the optimizer, EMA or DDP, and are always loaded from the pinned Hugging Face revision.
"""

import bisect
import copy
import json
import logging
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import sentencepiece
import torch
from torch import nn
from torch.nn import functional as F

from pocket_tts.utils.config import SemanticConditionerConfig

logger = logging.getLogger(__name__)

# The attribute name on FlowLMModel, and so the key prefix of every fusion tensor in
# checkpoints ("flow_lm.semantic_fusion." in training checkpoints and exports).
SEMANTIC_ATTR = "semantic_fusion"
SEMANTIC_PREFIX = SEMANTIC_ATTR + "."


# ------------------------------------------------------------------------------------------
# SentencePiece character spans
# ------------------------------------------------------------------------------------------


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return value, pos


def _proto_fields(data: bytes) -> list[tuple[int, int | bytes]]:
    """(field number, value) pairs of a serialized protobuf message, one level deep."""
    fields: list[tuple[int, int | bytes]] = []
    pos = 0
    while pos < len(data):
        key, pos = _varint(data, pos)
        number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, pos = _varint(data, pos)
        elif wire_type == 2:
            length, pos = _varint(data, pos)
            value, pos = data[pos : pos + length], pos + length
        elif wire_type == 5:
            value, pos = data[pos : pos + 4], pos + 4
        elif wire_type == 1:
            value, pos = data[pos : pos + 8], pos + 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type}")
        fields.append((number, value))
    return fields


def byte_to_char_spans(text: str, byte_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """UTF-8 byte offsets into `text` -> character offsets. A begin inside a character moves to
    that character's start and an end inside one to its end (SentencePiece never does either,
    but byte-fallback pieces carry empty spans, which stay empty)."""
    starts = [0]
    for ch in text:
        starts.append(starts[-1] + len(ch.encode("utf-8")))
    spans = []
    for begin, end in byte_spans:
        first = bisect.bisect_right(starts, begin) - 1
        last = min(bisect.bisect_left(starts, end), len(text))
        spans.append((first, max(first, last)))
    return spans


def sentencepiece_spans(
    sp: sentencepiece.SentencePieceProcessor, text: str
) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids of `sp.encode(text)` and each piece's character span in `text`.

    SentencePiece reports byte offsets into the original (unnormalized) input in its proto.
    Reading the serialized form directly avoids a dependency on the protobuf package.
    """
    ids: list[int] = []
    byte_spans: list[tuple[int, int]] = []
    for number, value in _proto_fields(sp.encode(text, out_type="serialized_proto")):
        if number != 2:  # SentencePieceText.pieces
            continue
        # piece: 1 piece, 2 id, 3 surface, 4 begin, 5 end. Zero values are omitted on the wire.
        piece = dict(_proto_fields(value))
        ids.append(int(piece.get(2, 0)))
        byte_spans.append((int(piece.get(4, 0)), int(piece.get(5, 0))))
    return ids, byte_to_char_spans(text, byte_spans)


# ------------------------------------------------------------------------------------------
# Alignment: per-character voting from encoder tokens onto SentencePiece tokens
# ------------------------------------------------------------------------------------------


def char_owner(length: int, spans: list[tuple[int, int]]) -> torch.Tensor:
    """[length] index of the token whose span covers each character, -1 where none does."""
    owner = torch.full((length,), -1, dtype=torch.long)
    for index, (begin, end) in enumerate(spans):
        if end > begin:
            owner[begin:end] = index
    return owner


def extend_over_marks(text: str, owner: torch.Tensor) -> torch.Tensor:
    """Give uncovered combining marks (niqqud, accents) the token of the character before them.

    DictaBERT's tokenizer strips niqqud and its offsets then end at the last base letter, so the
    marks after a word's final letter (the dagesh and sheva of אַתְּ) would otherwise be uncovered.
    """
    owner = owner.clone()
    for i in range(1, len(text)):
        if owner[i] < 0 and owner[i - 1] >= 0 and unicodedata.category(text[i]).startswith("M"):
            owner[i] = owner[i - 1]
    return owner


def pool_to_tokens(
    latents: torch.Tensor,
    char_to_d: torch.Tensor,
    char_to_s: torch.Tensor,
    num_tokens: int,
    null: torch.Tensor,
    visible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-character voting.

    Every visible character covered both by an encoder token d and a SentencePiece token s
    votes with the latent of d for s. A token's vector is the mean of its votes (overlap-weighted
    pooling: one d over many s copies the vector, one s over many d averages them). A token with
    no vote gets `null`. `cover` is the share of the token's visible characters that voted.

    latents [Nd, H]; char_to_d, char_to_s, visible [num_chars]. Returns ([num_tokens, H],
    [num_tokens]).
    """
    device = latents.device
    char_to_d, char_to_s, visible = char_to_d.to(device), char_to_s.to(device), visible.to(device)
    voting = (char_to_d >= 0) & (char_to_s >= 0) & visible
    s_idx, d_idx = char_to_s[voting], char_to_d[voting]
    counts = torch.zeros(num_tokens, device=device).index_add_(
        0, s_idx, torch.ones(s_idx.shape[0], device=device)
    )
    sums = latents.new_zeros(num_tokens, latents.shape[-1]).index_add_(0, s_idx, latents[d_idx])
    mean = sums / counts.clamp(min=1)[:, None]
    # torch.where keeps `null` in the graph even when no token uses it, so DDP never sees a
    # trainable parameter without a gradient.
    pooled = torch.where((counts > 0)[:, None], mean, null.to(mean.dtype).expand_as(mean))
    owned = (char_to_s >= 0) & visible
    sizes = torch.zeros(num_tokens, device=device).index_add_(
        0, char_to_s[owned], torch.ones(int(owned.sum()), device=device)
    )
    cover = counts / sizes.clamp(min=1)
    return pooled, cover


# ------------------------------------------------------------------------------------------
# The frozen encoder
# ------------------------------------------------------------------------------------------


class SemanticEncoder:
    """A frozen text encoder. Subclasses: DictaBertEncoder, RenikudPlusEncoder
    (semantic_renikud.py), and fakes in tests.

    Each encoder supplies the two things the shared voting needs: the token spans in the final
    string (`tokenize`) and the hidden states of its layers (`encode`)."""

    hidden_size: int
    num_layers: int

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Token ids without special tokens, and each token's character span in `text` (the
        final string the SentencePiece tokenizer reads, even where the encoder reads an edited
        view of it). Tokens that must not vote (unknown tokens) get an empty span."""
        raise NotImplementedError

    def encode(self, ids: list[int], layers: list[int], device: torch.device) -> torch.Tensor:
        """[len(layers), len(ids), hidden_size] float32 hidden states of `ids` for each of
        `layers` (1-based: hidden state k is the output of transformer layer k)."""
        raise NotImplementedError


class DictaBertEncoder(SemanticEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        max_tokens: int,
        keep_layers: int | None = None,
        source: str | Path | None = None,
    ):
        """`keep_layers`: drop the transformer layers above it, which nothing reads (layers
        21-24 with the default band: a sixth of the encoder's compute).

        `source`: a local directory holding the files `from_pretrained` reads (config.json,
        model.safetensors, tokenizer.json, tokenizer_config.json, vocab.txt) instead of the
        Hugging Face Hub; `model_name` and `revision` are then only names for the logs. The
        deploy server copies the pinned files from its model store and passes them here."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "semantic conditioning needs the `transformers` package: "
                "pip install 'pocket-tts[semantic]'"
            ) from exc
        if source is not None:
            logger.info("loading semantic encoder %s@%s from %s", model_name, revision, source)
            location, extra = str(source), {"local_files_only": True}
        else:
            logger.info("loading semantic encoder %s@%s", model_name, revision)
            location, extra = model_name, {"revision": revision}
        self.tokenizer = AutoTokenizer.from_pretrained(location, **extra)
        if not self.tokenizer.is_fast:
            raise ValueError(f"{model_name} has no fast tokenizer; character offsets need one")
        # BertModel without the pooler: only hidden states are used, and the checkpoint has no
        # pooler weights (it is a masked-LM checkpoint).
        model = AutoModel.from_pretrained(location, add_pooling_layer=False, **extra)
        model.eval()
        model.requires_grad_(False)
        self.model = model
        self.hidden_size = int(model.config.hidden_size)
        self.num_layers = int(model.config.num_hidden_layers)
        if keep_layers is not None and keep_layers < self.num_layers:
            model.encoder.layer = model.encoder.layer[:keep_layers]
        self.max_tokens = min(max_tokens, int(model.config.max_position_embeddings))
        self.cls_id = int(self.tokenizer.cls_token_id)
        self.sep_id = int(self.tokenizer.sep_token_id)
        self.unk_id = int(self.tokenizer.unk_token_id)
        self._warned_long = False

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        # verbose=False: texts over the position limit are windowed in encode(), so the
        # tokenizer's own length warning does not apply.
        encoded = self.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, verbose=False
        )
        ids = [int(i) for i in encoded["input_ids"]]
        spans = [
            (0, 0) if i == self.unk_id else (int(begin), int(end))
            for i, (begin, end) in zip(ids, encoded["offset_mapping"], strict=True)
        ]
        return ids, spans

    @torch.no_grad()
    def encode(self, ids: list[int], layers: list[int], device: torch.device) -> torch.Tensor:
        if next(self.model.parameters()).device != device:
            self.model.to(device)
        window = self.max_tokens - 2
        if len(ids) > window and not self._warned_long:
            self._warned_long = True
            logger.warning(
                "semantic encoder input of %d tokens exceeds its %d-token limit: encoding it in "
                "independent windows of %d tokens (no context across windows). Reported once.",
                len(ids),
                self.max_tokens,
                window,
            )
        outputs = []
        for start in range(0, len(ids), window):
            chunk = [self.cls_id, *ids[start : start + window], self.sep_id]
            hidden = self.model(
                input_ids=torch.tensor([chunk], device=device), output_hidden_states=True
            ).hidden_states
            outputs.append(torch.stack([hidden[layer][0, 1:-1] for layer in layers]).float())
        if not outputs:
            return torch.zeros(len(layers), 0, self.hidden_size, device=device)
        return torch.cat(outputs, dim=1)


# ------------------------------------------------------------------------------------------
# Character classes of a SentencePiece piece (for the per-class gate metrics)
# ------------------------------------------------------------------------------------------

CHAR_CLASSES = ("hebrew", "digit", "latin", "other")


def piece_class(piece: str) -> str:
    """The class of the letters or digits most common in a piece: "hebrew" (U+05D0-05EA, so
    the niqqud on a letter does not count), "digit", "latin", else "other" (punctuation, symbols,
    an empty piece). Ties go to the order Hebrew, digit, Latin."""
    counts = dict.fromkeys(CHAR_CLASSES[:3], 0)
    for ch in piece:
        if "\u05d0" <= ch <= "\u05ea":
            counts["hebrew"] += 1
        elif ch.isdecimal():
            counts["digit"] += 1
        elif ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN"):
            counts["latin"] += 1
    best = max(counts.values())
    return next(name for name, n in counts.items() if n == best) if best else "other"


# ------------------------------------------------------------------------------------------
# The trainable fusion
# ------------------------------------------------------------------------------------------


class _EncoderSlot:
    """What SemanticFusion and CombinedSemanticFusion share: the frozen encoders, kept out of
    the module tree, and the loading, size checks and deepcopy that go with that.

    A subclass lists its encoders' configs in `configs` and keeps the encoders in
    `self.__dict__["_encoders"]`, one slot per config."""

    configs: list[SemanticConditionerConfig]

    @staticmethod
    def _check_fits(encoder: SemanticEncoder, config: SemanticConditionerConfig) -> None:
        if encoder.hidden_size != config.hidden_size:
            raise ValueError(
                f"semantic encoder hidden size {encoder.hidden_size} != "
                f"semantic_conditioner.hidden_size {config.hidden_size}"
            )
        if config.last_layer > encoder.num_layers:
            raise ValueError(
                f"semantic_conditioner.last_layer {config.last_layer} > the encoder's "
                f"{encoder.num_layers} layers"
            )

    @staticmethod
    def _build_encoder(
        config: SemanticConditionerConfig, source: str | Path | None
    ) -> SemanticEncoder:
        if source is not None and not Path(source).is_dir():
            raise FileNotFoundError(f"semantic encoder directory not found: {source}")
        if config.encoder == "renikud_plus":
            # Imported only here: onnx and onnxruntime are needed by this encoder alone.
            from pocket_tts.conditioners.semantic_renikud import RenikudPlusEncoder

            return RenikudPlusEncoder.from_config(config, source)
        return DictaBertEncoder(
            config.model_name,
            config.revision,
            config.max_tokens,
            keep_layers=config.last_layer,
            source=source,
        )

    def __deepcopy__(self, memo: dict):
        # Copies (a distillation teacher, for one) share the frozen encoders instead of
        # duplicating 435M-parameter models.
        for encoder in self._encoders:
            if encoder is not None:
                memo[id(encoder)] = encoder
        clone = self.__class__.__new__(self.__class__)
        memo[id(self)] = clone
        clone.__dict__.update({k: copy.deepcopy(v, memo) for k, v in self.__dict__.items()})
        return clone

    def identity(self) -> dict:
        """What the trained tensors are tied to, as recorded in checkpoints and exports."""
        raise NotImplementedError


class SemanticFusion(_EncoderSlot, nn.Module):
    def __init__(self, config: SemanticConditionerConfig, d_model: int):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.layers = list(range(config.first_layer, config.last_layer + 1))
        # Softmax weights over the layer band, uniform at init.
        self.mix_logits = nn.Parameter(torch.zeros(len(self.layers)))
        self.out_norm = nn.LayerNorm(hidden)
        # The latent of a SentencePiece token no encoder token covers (spaces, [UNK], byte
        # fallback pieces).
        self.null = nn.Parameter(torch.zeros(hidden))
        gate_out = 1 if config.gate == "scalar" else d_model
        self.gate = nn.Sequential(
            nn.Linear(d_model + hidden + 1, config.gate_hidden),
            nn.SiLU(),
            nn.Linear(config.gate_hidden, gate_out),
        )
        # g = 0.5 everywhere at init; its gradient is zero until proj moves away from zero.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.proj = nn.Linear(hidden, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)
        # Plain attributes, not submodules (see the module docstring).
        self.__dict__["_encoders"] = [None]
        self.__dict__["_stats"] = []

    @property
    def configs(self) -> list[SemanticConditionerConfig]:
        return [self.config]

    def identity(self) -> dict:
        return self.config.identity()

    @property
    def encoder(self) -> SemanticEncoder:
        if self._encoders[0] is None:
            self.load_encoder()
        return self._encoders[0]

    def set_encoder(self, encoder: SemanticEncoder) -> None:
        self._check_fits(encoder, self.config)
        self.__dict__["_encoders"] = [encoder]

    def load_encoder(self, source: str | Path | Sequence[str | Path | None] | None = None) -> None:
        """Load the pinned encoder now (otherwise it loads on first use).

        `source` is a local directory with the encoder's files (what the config's `model_name`
        and `revision` would download), for deployments that keep them in their own store: for
        DictaBERT the files `from_pretrained` reads, for RenikudPlus the config's `filename`.
        The caller vouches that they are the files the config pins."""
        if self._encoders[0] is not None:
            return
        if source is not None and not isinstance(source, str | Path):
            if len(source) != 1:
                raise ValueError("one semantic encoder is configured: give one directory")
            source = source[0]
        self.set_encoder(self._build_encoder(self.config, source))

    def mix(self, hidden: torch.Tensor) -> torch.Tensor:
        return _mix_layers(hidden, self.mix_logits, self.out_norm)

    def forward(
        self,
        text: str,
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
        sp: sentencepiece.SentencePieceProcessor,
    ) -> torch.Tensor:
        """Fuse `embeddings` ([L, D] or [1, L, D], the lookup-table output for `tokens`) with
        the encoder's reading of `text`. `tokens` must be exactly `sp.encode(text)`: both
        tokenizers have to see the same string or the character alignment is wrong."""
        squeeze = embeddings.dim() == 3
        es = embeddings[0] if squeeze else embeddings
        sp_spans = _check_tokens(text, tokens, es, sp)
        char_to_s, visible = char_owner(len(text), sp_spans), _visible(text)
        pooled, cover = _pooled_latents(
            self.encoder, self.layers, self.mix, self.null, text, char_to_s, visible, es
        )
        gate = torch.sigmoid(self.gate(torch.cat([es, pooled, cover[:, None]], dim=-1)))
        fused = es + gate * self.proj(pooled)
        if self.training:
            groups = _token_groups(text, sp_spans, gate.device)
            per_token = gate.detach().float().mean(dim=-1)
            stats = _token_stats(per_token, groups, "gate")
            stats["cover"] = cover.detach().float().mean()
            self._stats.append(stats)
        return fused[None] if squeeze else fused

    def pop_stats(self) -> dict[str, torch.Tensor]:
        """Mean gate (overall, Latin-letter tokens, the rest; and per character class of the
        token: Hebrew, digit, Latin, other) and cover since the last call, as `semantic_*`
        metrics. Recorded in training mode only."""
        return _pop_stats(self)


class _Branch(nn.Module):
    """One encoder's trainable side inside CombinedSemanticFusion: the same pieces as
    SemanticFusion's own tensors (layer mix, null vector, zero-initialized projection)."""

    def __init__(self, config: SemanticConditionerConfig, d_model: int):
        super().__init__()
        self.layers = list(range(config.first_layer, config.last_layer + 1))
        self.mix_logits = nn.Parameter(torch.zeros(len(self.layers)))
        self.out_norm = nn.LayerNorm(config.hidden_size)
        self.null = nn.Parameter(torch.zeros(config.hidden_size))
        self.proj = nn.Linear(config.hidden_size, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)

    def mix(self, hidden: torch.Tensor) -> torch.Tensor:
        return _mix_layers(hidden, self.mix_logits, self.out_norm)


def branch_names(configs: Sequence[SemanticConditionerConfig]) -> list[str]:
    """Metric names of the encoders: the encoder's own name, numbered when two share one."""
    names = [c.encoder for c in configs]
    return [n if names.count(n) == 1 else f"{n}{i}" for i, n in enumerate(names)]


class CombinedSemanticFusion(_EncoderSlot, nn.Module):
    """Two (or more) frozen encoders fused through one gate network.

        g_i = sigmoid(MLP([es_i ; p_i^1 ; p_i^2 ; cover_i^1 ; cover_i^2]))   one value per encoder
        x_i = es_i + sum_k g_i^k * W_k p_i^k                                  every W_k zero at init

    The gate sees the embedding and every encoder's reading, so it can prefer one encoder per
    token (and learns which). The embedding path is not gated, which keeps x_i == es_i at
    initialization: warm-starting from a plain model is as safe as with one encoder.

    Parameters: `branches.<k>.{mix_logits,out_norm,null,proj}` per encoder, and `gate` shared.
    `gate` / `gate_hidden` come from the first config (the config validator requires all
    entries to agree). With `gate: vector` each encoder gets one gate per model channel.
    """

    def __init__(self, configs: Sequence[SemanticConditionerConfig], d_model: int):
        super().__init__()
        if len(configs) < 2:
            raise ValueError("CombinedSemanticFusion needs at least two encoders")
        self.configs = list(configs)
        self.names = branch_names(self.configs)
        self.branches = nn.ModuleList(_Branch(c, d_model) for c in self.configs)
        count = len(self.configs)
        self.gate_width = 1 if self.configs[0].gate == "scalar" else d_model
        self.gate = nn.Sequential(
            nn.Linear(
                d_model + sum(c.hidden_size for c in self.configs) + count,
                self.configs[0].gate_hidden,
            ),
            nn.SiLU(),
            nn.Linear(self.configs[0].gate_hidden, count * self.gate_width),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.__dict__["_encoders"] = [None] * count
        self.__dict__["_stats"] = []

    def identity(self) -> dict:
        return {"encoder": COMBINED, "encoders": [c.identity() for c in self.configs]}

    @property
    def encoders(self) -> list[SemanticEncoder]:
        self.load_encoder()
        return list(self._encoders)

    def set_encoders(self, encoders: Sequence[SemanticEncoder]) -> None:
        if len(encoders) != len(self.configs):
            raise ValueError(f"{len(self.configs)} encoders configured, got {len(encoders)}")
        for encoder, config in zip(encoders, self.configs, strict=True):
            self._check_fits(encoder, config)
        self.__dict__["_encoders"] = list(encoders)

    def load_encoder(self, source: Sequence[str | Path | None] | None = None) -> None:
        """Load the pinned encoders now (otherwise they load on first use).

        `source` is one local directory per encoder, in config order (None for an encoder that
        loads from the Hub); see SemanticFusion.load_encoder."""
        if all(e is not None for e in self._encoders):
            return
        if source is None:
            source = [None] * len(self.configs)
        elif isinstance(source, str | Path) or len(source) != len(self.configs):
            raise ValueError(
                f"the combined semantic fusion has {len(self.configs)} encoders: "
                "semantic_encoder_dir needs one directory per encoder, in config order"
            )
        self.set_encoders(
            [
                existing if existing is not None else self._build_encoder(config, directory)
                for existing, config, directory in zip(
                    self._encoders, self.configs, source, strict=True
                )
            ]
        )

    def forward(
        self,
        text: str,
        tokens: torch.Tensor,
        embeddings: torch.Tensor,
        sp: sentencepiece.SentencePieceProcessor,
    ) -> torch.Tensor:
        """As SemanticFusion.forward: [L, D] or [1, L, D] in, the same shape out."""
        squeeze = embeddings.dim() == 3
        es = embeddings[0] if squeeze else embeddings
        sp_spans = _check_tokens(text, tokens, es, sp)
        char_to_s, visible = char_owner(len(text), sp_spans), _visible(text)
        pooled, covers = [], []
        for branch, encoder in zip(self.branches, self.encoders, strict=True):
            p, c = _pooled_latents(
                encoder, branch.layers, branch.mix, branch.null, text, char_to_s, visible, es
            )
            pooled.append(p)
            covers.append(c)
        count = len(self.branches)
        gate_in = torch.cat([es, *pooled, torch.stack(covers, dim=-1)], dim=-1)
        gate = torch.sigmoid(self.gate(gate_in)).view(es.shape[0], count, self.gate_width)
        contributions = [
            gate[:, k] * branch.proj(p)
            for k, (branch, p) in enumerate(zip(self.branches, pooled, strict=True))
        ]
        fused = es + sum(contributions)
        if self.training:
            self._record(text, sp_spans, es, gate, covers, contributions)
        return fused[None] if squeeze else fused

    def _record(self, text, spans, es, gate, covers, contributions) -> None:
        groups = _token_groups(text, spans, gate.device)
        per_token = gate.detach().float().mean(dim=-1)  # [L, encoders]
        stats = _token_stats(per_token.mean(dim=-1), groups, "gate")
        stats["cover"] = torch.stack(covers).detach().float().mean()
        scale = es.detach().float().norm(dim=-1).clamp(min=1e-6)
        for k, name in enumerate(self.names):
            stats.update(_token_stats(per_token[:, k], groups, f"gate_{name}"))
            stats[f"cover_{name}"] = covers[k].detach().float().mean()
            # How big the encoder's addition is next to the embedding it is added to.
            stats[f"contrib_{name}"] = (
                contributions[k].detach().float().norm(dim=-1) / scale
            ).mean()
        self._stats.append(stats)

    def pop_stats(self) -> dict[str, torch.Tensor]:
        """`semantic_gate*` / `semantic_cover` as for one encoder (the gate is the mean over
        encoders), plus per encoder `semantic_gate_<name>` (and its `_latin`, `_other`,
        `_class_*` splits), `semantic_cover_<name>` and `semantic_contrib_<name>`: the norm of
        that encoder's addition relative to the embedding's. Training mode only."""
        return _pop_stats(self)


# ------------------------------------------------------------------------------------------
# Pieces both fusions share
# ------------------------------------------------------------------------------------------


def _mix_layers(
    hidden: torch.Tensor, mix_logits: torch.Tensor, out_norm: nn.LayerNorm
) -> torch.Tensor:
    """[layers, N, H] -> [N, H]: per-layer non-affine LayerNorm, softmax mix, LayerNorm."""
    normed = F.layer_norm(hidden.float(), (hidden.shape[-1],))
    weights = torch.softmax(mix_logits.float(), dim=0)
    return out_norm((weights[:, None, None] * normed).sum(dim=0))


def _check_tokens(
    text: str, tokens: torch.Tensor, es: torch.Tensor, sp: sentencepiece.SentencePieceProcessor
) -> list[tuple[int, int]]:
    """The character spans of the SentencePiece tokens of `text`, after checking that `tokens`
    is exactly its encoding and `es` has a row per token."""
    sp_ids, sp_spans = sentencepiece_spans(sp, text)
    given = tokens.reshape(-1).tolist()
    if sp_ids != given:
        raise ValueError(
            "semantic conditioning: the tokens are not sp.encode(text) "
            f"({len(given)} given, {len(sp_ids)} from the text {text[:80]!r}); the fusion "
            "must receive the exact string the SentencePiece tokenizer was given"
        )
    if es.shape[0] != len(sp_ids):
        raise ValueError(f"{es.shape[0]} embeddings for {len(sp_ids)} tokens")
    return sp_spans


def _visible(text: str) -> torch.Tensor:
    return torch.tensor([not ch.isspace() for ch in text], dtype=torch.bool)


def _pooled_latents(
    encoder: SemanticEncoder,
    layers: list[int],
    mix,
    null: torch.Tensor,
    text: str,
    char_to_s: torch.Tensor,
    visible: torch.Tensor,
    es: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One encoder's reading of `text` voted onto the SentencePiece tokens: ([L, H], [L]) in
    `es`'s dtype."""
    d_ids, d_spans = encoder.tokenize(text)
    latents = mix(encoder.encode(d_ids, layers, es.device))
    char_to_d = extend_over_marks(text, char_owner(len(text), d_spans))
    pooled, cover = pool_to_tokens(latents, char_to_d, char_to_s, es.shape[0], null, visible)
    return pooled.to(es.dtype), cover.to(es.dtype)


def _token_groups(
    text: str, spans: list[tuple[int, int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """(has a Latin letter [L] bool, character class index [L]) of each SentencePiece token."""
    latin = torch.tensor(
        [any(ch.isascii() and ch.isalpha() for ch in text[b:e]) for b, e in spans],
        dtype=torch.bool,
        device=device,
    )
    classes = torch.tensor(
        [CHAR_CLASSES.index(piece_class(text[b:e])) for b, e in spans], device=device
    )
    return latin, classes


def _token_stats(
    per_token: torch.Tensor, groups: tuple[torch.Tensor, torch.Tensor], name: str
) -> dict[str, torch.Tensor]:
    """Mean of `per_token` ([L]) overall (`name`), over Latin-letter tokens (`name_latin`), the
    rest (`name_other`) and per character class (`name_class_<class>`); empty groups omitted."""
    latin, classes = groups
    stats = {name: per_token.mean()}
    if bool(latin.any()):
        stats[f"{name}_latin"] = per_token[latin].mean()
    if bool((~latin).any()):
        stats[f"{name}_other"] = per_token[~latin].mean()
    for index, cls in enumerate(CHAR_CLASSES):
        members = classes == index
        if bool(members.any()):
            stats[f"{name}_class_{cls}"] = per_token[members].mean()
    return stats


def _pop_stats(fusion) -> dict[str, torch.Tensor]:
    records, fusion.__dict__["_stats"] = fusion._stats, []
    keys = dict.fromkeys(k for record in records for k in record)
    return {
        f"semantic_{key}": torch.stack([r[key] for r in records if key in r]).mean() for key in keys
    }


FUSION_TYPES = (SemanticFusion, CombinedSemanticFusion)


def build_semantic_fusion(
    configs: Sequence[SemanticConditionerConfig], d_model: int
) -> SemanticFusion | CombinedSemanticFusion | None:
    """No fusion without encoders, SemanticFusion for one (the legacy tensors), the combined
    fusion for several."""
    if not configs:
        return None
    if len(configs) == 1:
        return SemanticFusion(configs[0], d_model)
    return CombinedSemanticFusion(configs, d_model)


def fuse_rows(
    fusion: SemanticFusion | CombinedSemanticFusion,
    texts: list[str] | None,
    tokens: list[torch.Tensor],
    embeddings: torch.Tensor,
    sp: sentencepiece.SentencePieceProcessor,
) -> torch.Tensor:
    """Fuse a right-padded batch [B, Lmax, D] row by row (padding positions become zeros)."""
    if texts is None or len(texts) != len(tokens):
        raise ValueError(
            "semantic conditioning needs the text of every row (texts=...), "
            f"got {None if texts is None else len(texts)} for {len(tokens)} rows"
        )
    width = embeddings.shape[1]
    rows = []
    for b, (text, row_tokens) in enumerate(zip(texts, tokens, strict=True)):
        length = row_tokens.shape[0]
        fused = fusion(text, row_tokens, embeddings[b, :length], sp)
        rows.append(F.pad(fused, (0, 0, 0, width - length)))
    return torch.stack(rows)


# ------------------------------------------------------------------------------------------
# Checkpoint compatibility
# ------------------------------------------------------------------------------------------


# The key under which checkpoints and exports record which encoder their fusion was trained with
# (a key of the training .pt payload; the safetensors metadata key is this, prefixed).
ENCODER_RECORD_KEY = "semantic_encoder"
ENCODER_METADATA_KEY = "pocket_tts." + ENCODER_RECORD_KEY
# The `encoder` of a combined fusion's record, which lists its members under `encoders`.
COMBINED = "combined"

_NOT_CHECKED = object()


def encoder_record(module: nn.Module) -> dict | None:
    """The encoder identity to record next to `module`'s fusion tensors, None without a fusion."""
    for sub in module.modules():
        if isinstance(sub, FUSION_TYPES):
            return sub.identity()
    return None


def encoder_metadata(module: nn.Module) -> dict[str, str] | None:
    """`encoder_record` as safetensors metadata (string values only), None without a fusion."""
    record = encoder_record(module)
    return None if record is None else {ENCODER_METADATA_KEY: json.dumps(record)}


def read_encoder_record(path) -> dict | None:
    """The encoder identity recorded in a safetensors file's metadata. None for a file that
    carries none: every file written before the encoder was selectable (these are DictaBERT's)."""
    import safetensors

    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        raw = (f.metadata() or {}).get(ENCODER_METADATA_KEY)
    return None if raw is None else json.loads(raw)


def _members(record: dict) -> tuple[bool, list[dict]]:
    """(is it a combined record, the per-encoder identities it holds)."""
    if record.get("encoder") == COMBINED:
        return True, list(record["encoders"])
    return False, [record]


def _describe_record(record: dict | None) -> str:
    if record is None:
        return "dictabert (written before the encoder was recorded)"
    combined, members = _members(record)
    parts = [
        f"{m['encoder']} ({m['model_name']}@{m.get('revision')})"
        if "model_name" in m
        else m["encoder"]
        for m in members
    ]
    return "combined: " + " + ".join(parts) if combined else parts[0]


def check_encoder_record(expected: dict, recorded: dict | None, source: str = "checkpoint") -> None:
    """Raise if fusion tensors trained with another encoder would be loaded into a model
    configured with `expected` (`SemanticFusion.identity()`).

    A checkpoint with no record predates the choice of encoder and is DictaBERT's: only the
    encoder family can be compared then. Otherwise the encoder, model, revision and file must
    all match. A combined fusion's tensors belong to both encoders in their configured order:
    they load only into a combined model with the same encoders in the same order (a single
    encoder's tensors do not fit it, nor the reverse). A different layer band only warns: the
    mix weights then index other layers, but the tensors still fit.
    """
    expected_combined, expected_members = _members(expected)
    legacy = recorded is None
    if legacy:
        recorded_combined, recorded_members = False, [{"encoder": "dictabert"}]
    else:
        recorded_combined, recorded_members = _members(recorded)

    def mismatch(why: str) -> ValueError:
        return ValueError(
            f"{source} has semantic-conditioning tensors trained with encoder "
            f"{_describe_record(recorded)}, but the model is configured with "
            f"{_describe_record(expected)}: {why}. The fusion cannot be loaded across "
            "encoders; configure the encoder(s) the checkpoint was trained with, or start from "
            "a checkpoint without semantic tensors."
        )

    if recorded_combined != expected_combined or len(recorded_members) != len(expected_members):
        raise mismatch(
            "a combined fusion and a single-encoder fusion (or fusions of different sizes) "
            "do not share tensors"
        )
    for index, (member, wanted) in enumerate(zip(recorded_members, expected_members, strict=True)):
        keys = ("encoder",) if legacy else ("encoder", "model_name", "revision", "filename")
        differing = [k for k in keys if member.get(k) != wanted.get(k)]
        if differing:
            where = f"encoder #{index + 1} " if expected_combined else ""
            raise mismatch(f"{where}{', '.join(differing)} differ")
        if member.get("layers") not in (None, wanted.get("layers")):
            logger.warning(
                "%s was trained on encoder layers %s, the model mixes layers %s",
                source,
                member["layers"],
                wanted.get("layers"),
            )


def load_state_dict_allowing_semantic(
    module: nn.Module,
    state: dict,
    strict: bool = True,
    prefix: str = SEMANTIC_PREFIX,
    recorded_encoder: object = _NOT_CHECKED,
    source: str = "checkpoint",
) -> tuple[list[str], list[str]]:
    """`module.load_state_dict(state, strict)` that tolerates the optional fusion tensors.

    - Neither side has them: exactly the plain call.
    - The checkpoint has them, the model does not: they are dropped with a warning.
    - The model has them, the checkpoint does not: they keep their initialization (W = 0, so
      the model behaves as the checkpoint did).
    - Both have them: they must come from the same encoder. `recorded_encoder` is what the
      checkpoint recorded (`read_encoder_record`, or the .pt payload's `semantic_encoder`), None
      for a checkpoint that recorded nothing (DictaBERT's); a mismatch raises ValueError before
      anything is loaded. Left out, the check is skipped.

    Returns (missing, unexpected) without the fusion keys. `prefix` is the full key prefix
    in `module`'s naming, e.g. "flow_lm.semantic_fusion." for a TTSModel.
    """
    model_has = any(k.startswith(prefix) for k in module.state_dict())
    state_has = any(k.startswith(prefix) for k in state)
    if model_has and state_has and recorded_encoder is not _NOT_CHECKED:
        expected = encoder_record(module)
        if expected is not None:
            check_encoder_record(expected, recorded_encoder, source)
    if not model_has and not state_has:
        result = module.load_state_dict(state, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)
    if not model_has:
        dropped = [k for k in state if k.startswith(prefix)]
        logger.warning(
            "ignoring %d semantic-conditioning tensors (%s*): the model config has no "
            "semantic_conditioner section",
            len(dropped),
            prefix,
        )
        result = module.load_state_dict(
            {k: v for k, v in state.items() if not k.startswith(prefix)}, strict=strict
        )
        return list(result.missing_keys), list(result.unexpected_keys)
    result = module.load_state_dict(state, strict=False)
    fusion_missing = [k for k in result.missing_keys if k.startswith(prefix)]
    if fusion_missing:
        logger.info(
            "semantic conditioning starts from its initialization: %d tensors are not in the "
            "checkpoint",
            len(fusion_missing),
        )
    missing = [k for k in result.missing_keys if not k.startswith(prefix)]
    unexpected = list(result.unexpected_keys)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Error(s) in loading state_dict for {type(module).__name__}: "
            f"missing keys {missing}, unexpected keys {unexpected}"
        )
    return missing, unexpected
