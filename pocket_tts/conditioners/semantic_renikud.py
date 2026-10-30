"""RenikudPlus as a frozen semantic encoder (SEMANTIC_CONDITIONING.md), run in ONNX Runtime.

RenikudPlus (https://huggingface.co/notmax123/RenikudPlus, Apache-2.0) is a Hebrew
grapheme-to-phoneme model: a 24-layer, 1024-wide BERT over *characters* (a fine-tune of
dicta-il/dictabert-large-char) with FiLM speaker conditioning and three heads, published only
as ONNX. This module uses its encoder layers as a second `SemanticEncoder` next to DictaBERT:

- The graph is edited in memory at load time. The output of each layer in the configured band
  becomes a graph output, and everything above the last tapped layer (the top layers, FiLM, the
  final LayerNorm, the MLP and the heads) is pruned away.
- FiLM modulates only the output of the last encoder layer, so every encoder-layer hidden state
  is speaker-independent. The pruned graph does not contain the speaker inputs at all.
- Only the base model is imported at module level: `onnx` and `onnxruntime` are imported lazily
  by the loading functions, with an error that names the extra to install.

The text it reads is an *encoder view* of the string the SentencePiece tokenizer reads
(`build_encoder_view`): character-level, deterministic edits only, with a map back to the
original string, so the shared character voting in semantic.py applies unchanged.
"""

import gc
import json
import logging
import re
import tempfile
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pocket_tts.conditioners.semantic import SemanticEncoder
from pocket_tts.utils.config import SemanticConditionerConfig
from pocket_tts.utils.text_fold import FOLD

logger = logging.getLogger(__name__)

EXPECTED_LAYERS = 24

_NEEDS_ONNX = (
    "the renikud_plus semantic encoder needs `onnx` and `onnxruntime`: "
    "pip install 'pocket-tts[semantic-renikud]'"
)

# ------------------------------------------------------------------------------------------
# Encoder view: the text as the character model reads it
# ------------------------------------------------------------------------------------------

# Niqqud, dagesh, shin/sin dots and cantillation: the model is trained on unpointed text and has
# no tokens for them. The range of renikud_onnx.py's `_NIQQUD_RE`. Maqaf (05BE), paseq (05C0),
# sof pasuq (05C3) and nun hafukha (05C6) are punctuation and stay.
_MARKS = re.compile("[\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7]")

# One character for one character. The geresh and gershayim fold is renikud_onnx.py's
# `normalize_graphemes` and comes from text_fold.FOLD (the single definition of the fold); the
# acute accent and the backtick are renikud_onnx.py's other apostrophes. The ellipsis is not in
# the model's vocabulary. Other characters text_fold folds (curly quotes, Unicode dashes, maqaf)
# are in the vocabulary, so they are not rewritten here: with `fold_text` the string already
# has them folded, and without it the model reads them as they are.
_REPLACE: dict[str, str] = {
    "\u05f3": FOLD["\u05f3"],
    "\u05f4": FOLD["\u05f4"],
    "`": "'",
    "\u00b4": "'",
    "\u2026": ".",
}


@dataclass(frozen=True)
class EncoderView:
    """The text the encoder reads, and where each of its characters came from.

    chars  : the view string; one encoder token per character.
    source : for each view character, the index in the original string of the character it
             came from (a decomposed character's later pieces share their starter's index).
    spans  : for each view character, the span of the original string it votes for. The first
             view character of an original character has that character's own span; further
             pieces (an accent split off by NFD) have an empty span and do not vote. Deleted
             marks have no view character at all: they stay uncovered, and `extend_over_marks`
             gives them the token of the character before them.
    """

    chars: str
    source: list[int]
    spans: list[tuple[int, int]]


def _starter_runs(text: str) -> list[tuple[int, int]]:
    """Split `text` into runs that each start at a starter (canonical combining class 0).
    NFD never reorders or composes across such a boundary, so normalizing the runs one by one
    is the same as normalizing the whole string, and every run has a known original span."""
    runs, start = [], 0
    for i in range(1, len(text)):
        first = unicodedata.normalize("NFD", text[i])[0]
        if unicodedata.combining(first) == 0:
            runs.append((start, i))
            start = i
    if text:
        runs.append((start, len(text)))
    return runs


def build_encoder_view(text: str) -> EncoderView:
    """The encoder's reading of `text`, with no expansions:

    1. NFD, as renikud_onnx.py does (Hebrew presentation forms U+FB1D-FB4F become letter plus
       points; accented Latin becomes base plus combining accent). Done run by run on the
       original string so offsets stay exact however NFD changes the length.
    2. Niqqud and cantillation are deleted.
    3. 1:1 replacements (`_REPLACE`), every other whitespace character to a plain space, and
       lowercase (the vocabulary has no capitals but K, N and U).

    The same function runs in training and at inference, on the final string the SentencePiece
    tokenizer reads, so the two see the same view.
    """
    chars: list[str] = []
    source: list[int] = []
    spans: list[tuple[int, int]] = []
    for begin, end in _starter_runs(text):
        first = True
        for ch in unicodedata.normalize("NFD", text[begin:end]):
            if _MARKS.match(ch):
                continue
            ch = _REPLACE.get(ch, ch)
            if ch.isspace():
                ch = " "
            else:
                lowered = ch.lower()
                ch = lowered if len(lowered) == 1 else ch
            chars.append(ch)
            source.append(begin)
            spans.append((begin, begin + 1) if first else (begin, begin))
            first = False
    return EncoderView("".join(chars), source, spans)


class RenikudTokenizer:
    """One token per view character: `[CLS]`, the characters, `[SEP]`, as renikud_onnx.py's
    `_tokenize`. A character outside the vocabulary is `[UNK]` and, like DictaBERT's unknown
    tokens, does not vote (an empty span)."""

    def __init__(self, vocab: dict[str, int], cls_id: int, sep_id: int):
        self.vocab = vocab
        self.cls_id = cls_id
        self.sep_id = sep_id
        self.unk_id = vocab["[UNK]"]
        self.space_id = vocab.get(" ")

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        view = build_encoder_view(text)
        ids = [self.vocab.get(ch, self.unk_id) for ch in view.chars]
        spans = [
            (span[0], span[0]) if i == self.unk_id else span
            for i, span in zip(ids, view.spans, strict=True)
        ]
        return ids, spans


def split_windows(ids: Sequence[int], window: int, space_id: int | None) -> list[tuple[int, int]]:
    """Contiguous, non-overlapping (start, end) ranges of at most `window` tokens covering
    `ids`. A range that has to end inside the text ends after the last space token in its
    second half when there is one, so words are not cut, and at `window` otherwise."""
    ranges, start, n = [], 0, len(ids)
    while start < n:
        end = min(start + window, n)
        if end < n and space_id is not None:
            for j in range(end - 1, start + window // 2 - 1, -1):
                if ids[j] == space_id:
                    end = j + 1
                    break
        ranges.append((start, end))
        start = end
    return ranges


# ------------------------------------------------------------------------------------------
# Graph surgery
# ------------------------------------------------------------------------------------------


def _need_onnx():
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(_NEEDS_ONNX) from exc
    return onnx


def _need_onnxruntime():
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError(_NEEDS_ONNX) from exc
    return onnxruntime


def _nearest_layer_norms(node, producers: dict, layer_norms: set[int]) -> set[int]:
    """Ids of the LayerNormalization nodes `node` depends on with no other one in between."""
    found: set[int] = set()
    seen: set[int] = set()
    stack = [node]
    while stack:
        for name in stack.pop().input:
            producer = producers.get(name)
            if producer is None or id(producer) in seen:
                continue
            seen.add(id(producer))
            if id(producer) in layer_norms:
                found.add(id(producer))
            else:
                stack.append(producer)
    return found


def find_layer_outputs(graph, expected_layers: int = EXPECTED_LAYERS) -> list[str]:
    """Names of the tensors that are the output of encoder layers 1..N, found by structure.

    The encoder is a post-LayerNorm BERT, so the graph's LayerNormalization nodes, in order,
    are: the embedding LayerNorm, then two per layer (after attention, after the feed-forward,
    the second being the layer's output), then the final LayerNorm that follows FiLM. Tensor
    names are not relied on. Raises unless there are exactly 2N + 2 of them and each depends
    on the previous one with no other LayerNormalization in between (a chain).
    """
    layer_norms = [n for n in graph.node if n.op_type == "LayerNormalization"]
    if len(layer_norms) != 2 * expected_layers + 2:
        raise ValueError(
            f"expected {2 * expected_layers + 2} LayerNormalization nodes (embedding, "
            f"{expected_layers} layers x 2, final), found {len(layer_norms)}: this is not the "
            "RenikudPlus graph this code was written for"
        )
    producers = {out: n for n in graph.node for out in n.output}
    ids = {id(n) for n in layer_norms}
    for index, node in enumerate(layer_norms):
        found = _nearest_layer_norms(node, producers, ids)
        expected = set() if index == 0 else {id(layer_norms[index - 1])}
        if found != expected:
            raise ValueError(
                f"LayerNormalization #{index} ({node.name!r}) does not follow the expected "
                "embedding / 2 per layer / final chain"
            )
    outputs = [layer_norms[2 * k].output[0] for k in range(1, expected_layers + 1)]
    assert len(outputs) == expected_layers
    return outputs


def tap_layer_outputs(
    model, layers: Sequence[int], expected_layers: int = EXPECTED_LAYERS, prune: bool = True
) -> tuple[object, list[str]]:
    """Edit `model` (an onnx.ModelProto, in place) so the output of each of `layers` (1-based,
    hidden state k = output of layer k) is a graph output, in the order given.

    With `prune` (what inference uses) those are the only outputs and everything not needed to
    compute them is removed: the layers above the highest one, FiLM, the heads, and the graph
    inputs and weights only they use. Without it the original outputs are kept and the tapped
    tensors are added, so the full graph can be compared against the original.

    Returns (model, tapped tensor names).
    """
    onnx = _need_onnx()
    graph = model.graph
    layer_outputs = find_layer_outputs(graph, expected_layers)
    if not layers or not all(1 <= k <= expected_layers for k in layers):
        raise ValueError(f"layers must be in 1..{expected_layers}, got {list(layers)}")
    names = [layer_outputs[k - 1] for k in layers]
    hidden = None
    for info in list(graph.value_info) + list(graph.output):
        if info.name in names:
            dims = info.type.tensor_type.shape.dim
            if dims and dims[-1].dim_value:
                hidden = dims[-1].dim_value
    extra = [
        onnx.helper.make_tensor_value_info(
            name, onnx.TensorProto.FLOAT, ["batch", "seq_len", hidden]
        )
        for name in dict.fromkeys(names)
    ]
    if not prune:
        graph.output.extend(extra)
        return model, names

    producers = {out: i for i, n in enumerate(graph.node) for out in n.output}
    needed: set[int] = set()
    stack = [producers[name] for name in names]
    while stack:
        index = stack.pop()
        if index in needed:
            continue
        needed.add(index)
        for name in graph.node[index].input:
            if name in producers:
                stack.append(producers[name])
    for index in reversed(range(len(graph.node))):
        if index not in needed:
            del graph.node[index]
    used = {name for node in graph.node for name in node.input}
    for index in reversed(range(len(graph.initializer))):
        if graph.initializer[index].name not in used:
            del graph.initializer[index]
    for index in reversed(range(len(graph.input))):
        if graph.input[index].name not in used:
            del graph.input[index]
    del graph.output[:]
    graph.output.extend(extra)
    del graph.value_info[:]
    del model.metadata_props[:]
    return model, names


# ------------------------------------------------------------------------------------------
# The encoder
# ------------------------------------------------------------------------------------------


def _onnx_providers(provider: str) -> list:
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider != "cuda":
        raise ValueError(f"unknown RenikudPlus ONNX provider: {provider}")
    ort = _need_onnxruntime()
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "onnx_provider: cuda requires onnxruntime-gpu with CUDAExecutionProvider; "
            "the installed ONNX Runtime is CPU-only. Use a build compatible with your "
            "PyTorch CUDA/cuDNN and NVIDIA driver. Newer GPU wheels may require a "
            "newer CUDA major version."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("onnx_provider: cuda requires an available CUDA device")
    # The GPU wheel can reuse PyTorch's CUDA/cuDNN libraries. Do not do this on the CPU path.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    return [("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(), "use_tf32": 0})]


def _tapped_session(path: str, layers: Sequence[int], threads: int, provider: str = "cpu"):
    """(InferenceSession, output names, ONNX metadata) of the pruned graph tapping `layers`.

    The graph is read, edited and written to a temporary file that onnxruntime then loads, and
    the edited copy is freed first. Handing onnxruntime the serialized bytes instead keeps about
    2 GB more resident for the life of the process (measured: 1.75 GB against 3.75 GB), and
    loading the 1.2 GB file whole peaks at about 4.8 GB either way, for a few seconds. If the
    temporary directory cannot be written the bytes are used.
    """
    onnx = _need_onnx()
    ort = _need_onnxruntime()
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.log_severity_level = 3
    providers = _onnx_providers(provider)

    model = onnx.load(path)
    meta = {entry.key: entry.value for entry in model.metadata_props}
    model, names = tap_layer_outputs(model, layers)
    with tempfile.TemporaryDirectory(prefix="renikud_plus_") as tmp:
        tapped = Path(tmp) / "tapped.onnx"
        try:
            onnx.save(model, str(tapped))
        except OSError:
            logger.warning("cannot write a temporary file; loading the graph from memory")
            session = ort.InferenceSession(model.SerializeToString(), options, providers=providers)
        else:
            del model
            gc.collect()
            session = ort.InferenceSession(str(tapped), options, providers=providers)
    if provider == "cuda":
        if session.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(
                "RenikudPlus CUDAExecutionProvider failed to initialize; check the "
                "onnxruntime-gpu build and CUDA/cuDNN libraries. Refusing to silently "
                "run the encoder on CPU."
            )
        # Shape bookkeeping may run on CPU, but a CUDA EP failure must not restart on CPU.
        session.disable_fallback()
    return session, names, meta


class RenikudPlusEncoder(SemanticEncoder):
    """Frozen RenikudPlus encoder layers. `runner(ids)` maps one window of token ids (with
    [CLS] and [SEP]) to the tapped layers' states, [len(tapped), len(ids), hidden]."""

    num_layers = EXPECTED_LAYERS

    def __init__(
        self,
        tokenizer: RenikudTokenizer,
        runner: Callable[[list[int]], np.ndarray],
        tapped: Sequence[int],
        hidden_size: int,
        max_tokens: int,
        identity: dict[str, str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.runner = runner
        self.tapped = list(tapped)
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.identity = identity
        self._warned_long = False

    @classmethod
    def from_config(
        cls, config: SemanticConditionerConfig, source: str | Path | None = None
    ) -> "RenikudPlusEncoder":
        """Download (or find in the cache) the pinned ONNX file and build the pruned session.
        With `source`, a directory holding `config.filename`, that file is used instead of the
        Hugging Face Hub."""
        from huggingface_hub import hf_hub_download

        _need_onnx(), _need_onnxruntime()  # before downloading 1.2 GB
        logger.info(
            "loading semantic encoder %s@%s/%s (ONNX Runtime, %s, %d CPU thread(s))",
            config.model_name,
            config.revision,
            config.filename,
            config.onnx_provider,
            config.onnx_threads,
        )
        if source is not None:
            path = str(Path(source) / config.filename)
            if not Path(path).is_file():
                raise FileNotFoundError(f"{config.filename} not found in {source}")
        else:
            path = hf_hub_download(config.model_name, config.filename, revision=config.revision)
        layers = list(range(config.first_layer, config.last_layer + 1))
        session, names, meta = _tapped_session(
            path, layers, config.onnx_threads, provider=config.onnx_provider
        )
        vocab = {str(k): int(v) for k, v in json.loads(meta["vocab"]).items()}
        tokenizer = RenikudTokenizer(vocab, int(meta["cls_token_id"]), int(meta["sep_token_id"]))
        inputs = [i.name for i in session.get_inputs()]

        def runner(ids: list[int]) -> np.ndarray:
            feeds = {
                "input_ids": np.asarray([ids], dtype=np.int64),
                "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            }
            # Speaker inputs are not in the pruned graph (FiLM is above the band); if a graph
            # ever kept them, 0 is "unknown speaker".
            for name in inputs:
                feeds.setdefault(name, np.zeros(1, dtype=np.int64))
            outputs = session.run(names, feeds)
            return np.stack([o[0] for o in outputs])

        probe = runner([tokenizer.cls_id, tokenizer.unk_id, tokenizer.sep_id])
        return cls(
            tokenizer,
            runner,
            layers,
            hidden_size=int(probe.shape[-1]),
            max_tokens=config.max_tokens,
            identity=config.identity(),
        )

    def tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        return self.tokenizer.tokenize(text)

    @torch.no_grad()
    def encode(self, ids: list[int], layers: list[int], device: torch.device) -> torch.Tensor:
        missing = [k for k in layers if k not in self.tapped]
        if missing:
            raise ValueError(f"layers {missing} were not tapped at load (tapped: {self.tapped})")
        rows = [self.tapped.index(k) for k in layers]
        window = self.max_tokens - 2
        ranges = split_windows(ids, window, self.tokenizer.space_id)
        if len(ranges) > 1 and not self._warned_long:
            self._warned_long = True
            logger.warning(
                "semantic encoder input of %d characters exceeds its %d-token limit: encoding it "
                "in %d independent windows (no context across windows). Reported once.",
                len(ids),
                self.max_tokens,
                len(ranges),
            )
        outputs = []
        for start, end in ranges:
            chunk = [self.tokenizer.cls_id, *ids[start:end], self.tokenizer.sep_id]
            hidden = self.runner(chunk)
            outputs.append(torch.from_numpy(np.ascontiguousarray(hidden[rows, 1:-1])))
        if not outputs:
            return torch.zeros(len(layers), 0, self.hidden_size, device=device)
        return torch.cat(outputs, dim=1).float().to(device)
