# Optional semantic conditioning (DictaBERT)

An experimental, opt-in extension for Hebrew. It is not part of Kyutai's released models or
training recipe. It is off unless a config turns it on, and with it off, training and
inference run exactly as they did before.

Unvocalized Hebrew is ambiguous (רצה is "wanted" or "ran"), and the ~100M FlowLM cannot
resolve that from spelling alone. A frozen bidirectional encoder,
[DictaBERT-large](https://huggingface.co/dicta-il/dictabert-large), reads the whole sentence
in both directions. Its contextual states are added to the text prefix through a learned,
zero-initialized gate. The niqqud, number and `alt_words` augmentations are unchanged
(see [hebrew_training/NIQQUD.md](hebrew_training/NIQQUD.md)).

Code: `pocket_tts/conditioners/semantic.py`.

## Design

**The grid stays the SentencePiece sequence.** The backbone still reads one vector per
SentencePiece token, so `_estimate_max_gen_len` and text chunking are unchanged. The sequence
length and positions do not change either.

**Alignment by per-character voting.** Both tokenizers read the same string T:

- **SentencePiece pieces.** Character spans come from SentencePiece's own proto, read from its
  serialized form, so no `protobuf` package is needed. Its UTF-8 byte offsets are converted
  to character offsets.
- **DictaBERT tokens.** Spans come from the fast tokenizer's `return_offsets_mapping`.
- **Voting.** Every visible (non-whitespace) character covered by both votes with its DictaBERT
  token's latent for its SentencePiece token. A token's vector `p_i` is the mean of its votes:
  - One DictaBERT token over several pieces copies its vector to each.
  - A piece over several DictaBERT tokens gets their average.
  - Partial overlaps are weighted by the characters they share.
- **Null vector.** A piece with no vote gets a learned `null` vector. That covers a lone `▁`
  piece, empty byte-fallback pieces and `[UNK]`.
- **Cover.** `cover_i` is the share of the piece's visible characters that voted.
- **Same-string check.** The fusion re-encodes the string and raises if the result differs from
  the tokens it was given. A string mismatch cannot silently corrupt the alignment.
- **Same string everywhere.** In training this is the rendered manifest text, after the
  augmentation. In `TTSModel` it is the chunk after `prepare_text_prompt` (fold, strip,
  capitalize, final period). In the deploy servers it is `fold_characters(text)`.

**Niqqud.** DictaBERT strips the marks (`שָׁלוֹם` becomes `שלום`), and its offsets end at a
word's last letter. Marks inside a word are inside the span. Marks after the final letter (the
dagesh and sheva of `אַתְּ`) are not. `extend_over_marks` assigns any uncovered combining mark
to the token of the character before it. Measured on vocalized sentences, no visible character
is left uncovered.

**Features.**

- **Encoder setup.** DictaBERT runs frozen, in eval mode, under `no_grad`, once per row. Its
  weights are never saved: it loads from the pinned revision. It is kept out of
  `state_dict()`, the optimizer, EMA and DDP.
- **Layer band.** The default band is layers 8–20 of 24; the embedding output is never used.
  Layers above the band are dropped at load, which saves about a sixth of the encoder's compute.
- **Normalization and mix.** Each layer gets a non-affine LayerNorm per token. The layers are
  then mixed with learned softmax weights (uniform at init), followed by a final affine
  LayerNorm.
- **Long input.** Inputs longer than 510 tokens are encoded in independent 510-token windows,
  with a warning logged once.

**Fusion.**

```
g_i = sigmoid(MLP([es_i ; p_i ; cover_i]))      # MLP: Linear(d+1024+1 -> 256), SiLU, Linear -> 1 (or d)
x_i = es_i + g_i * W p_i                         # W: Linear(1024 -> d_model, no bias), zero-initialized
```

- `es_i` is the existing lookup-table embedding: what is written, including niqqud, case and
  English.
- `p_i` is what is meant in context. The gate lets English, acronyms and punctuation ignore it.
- With W = 0, step 0 equals the base model exactly; tests check this bit for bit. Warm-starting
  from the current Hebrew model is therefore safe.
- The gate's last layer also starts at zero (g = 0.5) and receives gradient once W moves.

## Configuration

### Model config (`flow_lm.semantic_conditioner`)

The section is absent from every shipped config except
[`pocket_tts/config/hebrew_semantic.yaml`](pocket_tts/config/hebrew_semantic.yaml)
(hebrew.yaml plus this section). `semantic_conditioner: {}` means all defaults.

| key | default | |
|---|---|---|
| `model_name` | `dicta-il/dictabert-large` | |
| `revision` | `a972ffb612c03bcfb128f10f5ef4462f7adb9844` | pinned |
| `hidden_size` | 1024 | checked against the encoder when it loads |
| `first_layer`, `last_layer` | 8, 20 | 1-based, inclusive |
| `gate` | `scalar` | `vector` = one gate per model channel |
| `gate_hidden` | 256 | |
| `max_tokens` | 512 | the encoder's limit, `[CLS]`/`[SEP]` included |

For inference with a model trained this way, add the same section to its inference YAML. For
example, a copy of `artifacts/hebrew_24l.yaml` with the section under `flow_lm`, and
`weights_path` pointing at the run's `model.safetensors` export. `TTSModel.load_model` (and so
`generate`, `serve` and the deploy servers) loads the encoder once, at load time.

### Training (`training/train.py`)

The trainer takes one YAML file, so the switches are keys rather than command-line flags:

| key | default | |
|---|---|---|
| `semantic_conditioner` | `null` (off) | a mapping (`{}` for all defaults), merged into the model config's section |
| `semantic_lr_scale` | `3.0` | the fusion's learning rate relative to `optim.lr` |

- **Optimizer groups.** With the feature on, the optimizer gets two parameter groups and the
  schedule multiplies the fusion group's learning rate by `lr_scale`. With it off, the
  optimizer gets `model.parameters()` as before.
- **Why 3x and not 10x.** Adam moves every element by about `lr` per step whatever its size, so
  the zero-initialized layer needs only a modest boost.
- **Example config.** [`training/configs/finetune_hebrew_semantic.yaml`](training/configs/finetune_hebrew_semantic.yaml)
  warm-starts from `artifacts/hebrew_24l.yaml` with `reset_text_embedding: false`. Data and
  augmentation are as in `finetune_hebrew.yaml`. The samples include homograph pairs, an English
  line and a vocalized line for listening.
- **Logged metrics.** Training logs `semantic_gate`, `semantic_gate_latin` (pieces containing
  Latin letters), `semantic_gate_other` and `semantic_cover`. If the gate stays at 0.5 and
  `proj` stays near zero, the semantics are not being used.
- **Dependency.** `transformers` is already in the `dev` group. Outside it, install
  `pip install 'pocket-tts[semantic]'`. Without it, only the semantic path fails, with an
  error that says so.

## Checkpoints

The fusion's tensors live under `flow_lm.semantic_fusion.*` in training checkpoints and in the
`model.safetensors` export, and nowhere else. DictaBERT is never written; it is referenced by
name and revision.

| from \ into | plain model | semantic model |
|---|---|---|
| plain checkpoint / export | unchanged | loads; the fusion keeps its zero init |
| semantic checkpoint / export | loads; fusion tensors ignored with a warning | loads |

This applies to all of these loaders:

- `TTSModel` (`weights_path`, `flow_lm.weights_path` and `--checkpoint`);
- warm-start in `training/modules/builders.py`;
- `training/checkpointing.load_checkpoint` (resume, `training/eval`).

When neither side has the keys, the call is the plain strict `load_state_dict`. On a resume
that switches the feature on:

- the optimizer state, which has a different number of groups, is dropped with a warning;
- the EMA shadow is seeded with the fusion's initial values.

## Smoke test

Before any long run, train a few steps on a tiny manifest with `max_steps: 4`, `batch_size: 1`,
`compile: false` and `ema_decay: 0`. Then confirm:

- the loss is finite;
- the `semantic_*` metrics appear;
- the fusion parameters have gradients and change;
- the encoder has no gradients and is unchanged;
- the checkpoint reloads.

Use batch size 1 on an 8 GB card. DictaBERT adds about 1.7 GB in fp32. Unit tests use a fake
encoder: `uv run pytest tests/test_semantic_conditioning.py training/tests/test_semantic_training.py`.

## Measured (pinned revision)

- **Loading.** A standard `BertModel`, loaded with `add_pooling_layer=False`. No
  `trust_remote_code`. The checkpoint is a masked-LM, and its head weights are ignored.
- **Size.** 435M parameters, 131M of them the 128k-entry embedding. Layers 1–20 are run.
- **Tokens per word** (3,000 rows of `data_debug/tmp-sample-normalized.jsonl`):
  - SentencePiece (`tokenizers/tokenizer.model`, 8000 pieces): 1.62;
  - DictaBERT: 1.20, with no `[UNK]`.
- **Coverage** (same rows):
  - mean cover is 1.0;
  - 1.9% of pieces get the null vector, mostly lone `▁` pieces;
  - no visible character is uncovered.
- **CPU cost.** About 35–50 ms per sentence (~33 tokens) on one thread (Apple Silicon,
  `torch.set_num_threads(1)` as in `tts_model.py`). This is in addition to the FlowLM, once
  per text chunk.

## Limitations and follow-ups

- **512-token cap.** Longer inputs lose context across windows. Training rows and inference
  chunks are far shorter.
- **Context is per chunk.** `TTSModel` fuses each chunk on its own, so no context crosses
  sentence chunks. Encoding the whole input once and slicing per chunk is the follow-up.
- **English semantics are a gamble.** DictaBERT lowercases (`NASA` becomes `nas ##a`) and has
  about 5.3k Latin entries. Case and spelling reach the model only through the lookup table.
  Watch `semantic_gate_latin`.
- **Cost.** The extra encoder is 435M parameters, about 1.7 GB in fp32.
- **No batching yet.** The encoder runs once per row in training.
- **Not wired in:** the older `hebrew_training/train.py` trainer, `training/eval/librispeech.py`
  and `hebrew_training/watch_eval.py`. With semantic weights they raise a clear "needs texts"
  error.
- **Evaluation.** This proof of concept is judged by listening. No homograph or CER tooling was
  added.
