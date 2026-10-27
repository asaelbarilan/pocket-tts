# Optional semantic conditioning (DictaBERT or RenikudPlus)

An experimental, opt-in extension for Hebrew. It is not part of Kyutai's released models or
training recipe. It is off unless a config turns it on, and with it off, training and
inference run exactly as they did before.

Unvocalized Hebrew is ambiguous (רצה is "wanted" or "ran"), and the ~100M FlowLM cannot
resolve that from spelling alone. A frozen bidirectional encoder reads the whole sentence in
both directions. Its contextual states are added to the text prefix through a learned,
zero-initialized gate. The niqqud, number and `alt_words` augmentations are unchanged (see
[hebrew_training/NIQQUD.md](hebrew_training/NIQQUD.md)).

Config selects one of three modes:

| mode | config | encoder |
|---|---|---|
| 1. off (the default) | no `semantic_conditioner` section | none. Nothing is imported or loaded; the path is bit-identical to before |
| 2. DictaBERT | `semantic_conditioner: {}` or `encoder: dictabert` | [DictaBERT-large](https://huggingface.co/dicta-il/dictabert-large), frozen, pinned revision. Unchanged in behavior |
| 3. RenikudPlus | `encoder: renikud_plus` | the encoder of [RenikudPlus](https://huggingface.co/notmax123/RenikudPlus) (Apache-2.0), frozen, fp32, in ONNX Runtime, hidden states tapped from inside |

A missing `encoder` key means DictaBERT, so every existing YAML and checkpoint keeps working
without edits.

Code: `pocket_tts/conditioners/semantic.py` (the shared fusion, alignment and checkpoint rules,
and the DictaBERT encoder) and `pocket_tts/conditioners/semantic_renikud.py` (RenikudPlus; only
imported when selected).

## The two encoders

| | DictaBERT-large | RenikudPlus |
|---|---|---|
| token unit | WordPiece (1.2 tokens per word) | one token per character |
| layers, width | 24, 1024 | 24, 1024 |
| positions | 512 | 2048 |
| runtime | PyTorch (`transformers`) | ONNX Runtime, CPU (`onnx`, `onnxruntime`) |
| download | ~1.6 GB (safetensors) | ~1.2 GB (`model.onnx`, fp32) |
| lowercases | yes (the tokenizer) | yes (the encoder view, see below) |
| niqqud | the tokenizer strips it | deleted by the encoder view |
| Latin | about 5.3k Latin pieces | every ASCII digit and a-z has its own token |
| what it was trained for | masked-LM on Hebrew text | Hebrew grapheme-to-phoneme (consonant, vowel, stress per letter), so its states carry pronunciation |

RenikudPlus is a fine-tune of `dicta-il/dictabert-large-char` (a character-level BERT), not of the
earlier `renikud/renikud` checkpoint. Which encoder gives better speech is not settled here; this
change only makes them interchangeable, and the comparison is a separate piece of work.

## Design

**The grid stays the SentencePiece sequence.** The backbone still reads one vector per
SentencePiece token, so `_estimate_max_gen_len` and text chunking are unchanged. The sequence
length and positions do not change either.

**Alignment by per-character voting, shared by both encoders.** Both tokenizers read the same
string T. Each encoder supplies two things: the hidden states of its layers, and the span of T
each of its tokens covers. Everything else is shared:

- **SentencePiece pieces.** Character spans come from SentencePiece's own proto, read from its
  serialized form, so no `protobuf` package is needed. Its UTF-8 byte offsets are converted
  to character offsets.
- **Encoder tokens.** DictaBERT: the fast tokenizer's `return_offsets_mapping`. RenikudPlus: the
  source map of the encoder view (below), one token per character.
- **Voting.** Every visible (non-whitespace) character covered by both votes with its encoder
  token's latent for its SentencePiece token. A token's vector `p_i` is the mean of its votes:
  - One encoder token over several pieces copies its vector to each.
  - A piece over several encoder tokens gets their average.
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
to the token of the character before it. RenikudPlus's encoder view deletes the marks, so they
have no token and take the same route. Measured on vocalized sentences, no visible character
is left uncovered by either.

**Features.**

- **Encoder setup.** Both encoders run frozen, in eval mode, under no gradient, once per row.
  Their weights are never saved: they load from the pinned revision. They are kept out of
  `state_dict()`, the optimizer, EMA and DDP.
- **Layer band.** The default band is layers 8–20 of 24 for both; the embedding output is never
  used. Layers above the band are dropped at load, which saves about a sixth of the encoder's
  compute (DictaBERT: the layers are cut; RenikudPlus: the graph is pruned, see below).
- **Normalization and mix.** Each layer gets a non-affine LayerNorm per token. The layers are
  then mixed with learned softmax weights (uniform at init), followed by a final affine
  LayerNorm.
- **Long input.** Inputs longer than the encoder's positions are encoded in independent windows,
  with a warning logged once (DictaBERT: 510 tokens; RenikudPlus: 2,046 characters).

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
- The fusion is identical for both encoders (`flow_lm.semantic_fusion.*`, hidden size 1024).
  Its tensors are tied to the encoder that trained them, see "Checkpoints".

## RenikudPlus

### The encoder

RenikudPlus is published only as ONNX (`model.onnx`, fp32, ~1.23 GB; `model_int8.onnx`, ~0.31
GB). The pinned revision is `679c56ca449d41873fb8ff7711ddf7563d28198f` and the file is the fp32
one. Training and inference both use fp32; the int8 weights are about 5% off, and the two must
never be mixed (the config's `filename` is part of the checkpoint's encoder identity).

- **Graph.** BERT, 24 post-LayerNorm layers, 1024 wide, a character vocabulary of 841 entries
  (in the ONNX metadata), 2048 positions. Inputs `input_ids`, `attention_mask`, `speaker`,
  `target_speaker`; outputs `consonant_logits`, `vowel_logits`, `stress_logits`.
- **Tapping.** At load, `onnx` reads the graph, and the output of each layer in the band becomes
  a graph output. The layers are found by structure, not by tensor name: the graph has exactly
  50 LayerNormalization nodes (embedding, two per layer, final), each depending on the previous
  one with no other in between, and layer k's output is the 2k-th. Anything else raises. (In
  the pinned fp32 file they are `layer_norm_2`, `layer_norm_4`, … `layer_norm_48`.)
- **Pruning.** Everything not needed for the band is removed in the same edit: the layers above
  it, FiLM, the final LayerNorm, the residual MLP and the three heads, with their weights and
  the speaker inputs. The pruned graph has two inputs (`input_ids`, `attention_mask`). The edited
  graph goes through a temporary file into an `onnxruntime` CPU session (one intra-op thread by
  default, as `torch.set_num_threads(1)` in `tts_model.py`; `onnx_threads` changes it).
- **FiLM is above the encoder.** The speaker embeddings modulate only the output of the last
  encoder layer (`h * (1 + gamma) + beta` on `layer_norm_48`, then a final LayerNorm, a residual
  MLP and the heads). Every encoder-layer hidden state is therefore speaker-independent, and
  nothing about speaker or gender is fed. A test checks this on a toy graph, and the integration
  test checks that the pruned real graph has no speaker input.
- **Frozen, outside the model.** Like DictaBERT it is a plain attribute of the fusion, not a
  module. The ONNX session lives on the CPU whatever device the model trains on; its output is
  moved to the model's device.

### The text it reads: no heavy normalization at inference

The TTS should speak any text given only simple Unicode replacements. There is no number or
abbreviation normalizer at inference, so the encoder reads the **same final string the
SentencePiece tokenizer reads**, never a normalized `words` form mapped back through
`alt_words`. (That alternative was rejected: it reads a different string in training and at
serving, which is train/serve skew by construction.) The model learns what `15` and `ק״ג` mean
from the audio, as it does today, and the encoder is just another reader of the same text.

To fit RenikudPlus's character vocabulary, the string goes through an **encoder view**
(`build_encoder_view`): character-level, deterministic edits, identical in training and at
inference, with a map from every view character back to the final string. No expansions.

1. **NFD**, as the model's wrapper does (`renikud_onnx.py`). It is applied run by run on the
   original string (a run starts at each starter character), which equals NFD of the whole
   string, so offsets stay exact when NFD changes the length: Hebrew presentation forms
   U+FB1D–FB4F become letter plus points, accented Latin becomes base plus combining accent.
2. **Delete niqqud and cantillation**: U+0591–05BD, 05BF, 05C1, 05C2, 05C4, 05C5, 05C7, the
   wrapper's own range. Maqaf, paseq, sof pasuq and nun hafukha stay. A deleted mark has no
   token; `extend_over_marks` gives it the previous character's token.
3. **1:1 replacements.** `…` becomes `.`, and the geresh U+05F3 and gershayim U+05F4 become `'`
   and `"` (the wrapper's fold, taken from `text_fold.FOLD`, which stays the single definition),
   as do the backtick and acute accent. Every other whitespace character (NBSP, thin space, tab,
   newline) becomes a plain space. Curly quotes, en/em dashes, the maqaf and the RTL/LTR marks
   are in the vocabulary and are not rewritten: `fold_text` already folds them for the shipped
   configs, and `text_fold.py` is not duplicated.
4. **Lowercase** (single-character results only). The vocabulary has no capitals except K, N
   and U, so `NASA` is read as `nasa`.

Each view character is one token (`[CLS]`, characters, `[SEP]`); a character the vocabulary lacks
is `[UNK]` and, like DictaBERT's unknown tokens, does not vote. The view's source map is the span
of the original string each token covers: a precomposed or decomposed character's first piece
has its own span, further pieces (the accent NFD split off) have an empty span.

Not in the vocabulary, so `[UNK]` if they reach it: niqqud and cantillation (deleted first),
`…`, NBSP (replaced first), zero-width characters other than the RTL/LTR marks, and the
uppercase letters other than K, N and U (lowercased first). In the vocabulary: all 27 Hebrew
letters with their finals, ASCII digits, a–z, ASCII punctuation, curly quotes, en/em dashes,
geresh/gershayim, maqaf, RTL/LTR marks, and many symbols.

**Long input.** The view's tokens are split into contiguous, non-overlapping windows of at most
2,046 tokens, each with its own `[CLS]`/`[SEP]`. A window that has to end inside the text ends
after the last space in its second half, so words are not cut. The model's wrapper windows at
the same 2,046.

## Configuration

### Model config (`flow_lm.semantic_conditioner`)

The section is absent from every shipped config except
[`pocket_tts/config/hebrew_semantic.yaml`](pocket_tts/config/hebrew_semantic.yaml) (hebrew.yaml
plus a DictaBERT section) and
[`pocket_tts/config/hebrew_semantic_renikud.yaml`](pocket_tts/config/hebrew_semantic_renikud.yaml)
(the same with `encoder: renikud_plus`). `semantic_conditioner: {}` means all defaults, which
is DictaBERT. Omitted keys take the selected encoder's defaults:

| key | `dictabert` | `renikud_plus` | |
|---|---|---|---|
| `encoder` | (default) | `renikud_plus` | `dictabert` or `renikud_plus` |
| `model_name` | `dicta-il/dictabert-large` | `notmax123/RenikudPlus` | |
| `revision` | `a972ffb612c03bcfb128f10f5ef4462f7adb9844` | `679c56ca449d41873fb8ff7711ddf7563d28198f` | pinned |
| `filename` | not allowed | `model.onnx` | fp32 ONNX file; not the int8 one |
| `hidden_size` | 1024 | 1024 | checked against the encoder when it loads |
| `first_layer`, `last_layer` | 8, 20 | 8, 20 | 1-based, inclusive; the encoders have 24 |
| `gate` | `scalar` | `scalar` | `vector` = one gate per model channel |
| `gate_hidden` | 256 | 256 | |
| `max_tokens` | 512 | 2048 | the encoder's positions, `[CLS]`/`[SEP]` included |
| `onnx_threads` | not allowed | 1 | ONNX Runtime intra-op threads |

For inference with a model trained this way, add the same section to its inference YAML. For
example, a copy of `artifacts/hebrew_24l.yaml` with the section under `flow_lm`, and
`weights_path` pointing at the run's `model.safetensors` export. `TTSModel.load_model` (and so
`generate`, `serve` and the deploy servers) loads the encoder once, at load time. A model must
be run with the encoder it was trained with (see "Checkpoints").

### Training (`training/train.py`)

The trainer takes one YAML file, so the switches are keys rather than command-line flags:

| key | default | |
|---|---|---|
| `semantic_conditioner` | `null` (off) | a mapping (`{}` for all defaults), merged into the model config's section |
| `semantic_lr_scale` | `3.0` | the fusion's learning rate relative to `optim.lr` |

- **Choosing the encoder.** `encoder` goes in this mapping like any other key. If it names a
  different encoder than the model config's own section, the mapping *replaces* that section
  (an info line says so) instead of merging into it, so DictaBERT's model name and limits never
  leak into a RenikudPlus run.
- **Optimizer groups.** With the feature on, the optimizer gets two parameter groups and the
  schedule multiplies the fusion group's learning rate by `lr_scale`. With it off, the
  optimizer gets `model.parameters()` as before.
- **Why 3x and not 10x.** Adam moves every element by about `lr` per step whatever its size, so
  the zero-initialized layer needs only a modest boost.
- **Example configs.**
  [`training/configs/finetune_hebrew_semantic.yaml`](training/configs/finetune_hebrew_semantic.yaml)
  (DictaBERT) and
  [`training/configs/finetune_hebrew_semantic_renikud.yaml`](training/configs/finetune_hebrew_semantic_renikud.yaml)
  (RenikudPlus) warm-start from a Hebrew model with `reset_text_embedding: false`. Data and
  augmentation are as in `finetune_hebrew.yaml`. The samples include homograph pairs, an English
  line and a vocalized line for listening. A RenikudPlus run must warm-start from a model whose
  weights carry no semantic tensors (or RenikudPlus ones): a DictaBERT-trained export is
  refused, see "Checkpoints".
- **Logged metrics.** Training logs `semantic_gate` (the mean gate), `semantic_gate_latin`
  (pieces containing Latin letters), `semantic_gate_other` and `semantic_cover`. It also logs the
  gate per character class of the piece: `semantic_gate_class_hebrew`, `_digit`, `_latin` and
  `_other`. A piece's class is that of the letters or digits most common in it (Hebrew letters,
  digits, Latin letters; ties in that order); a piece with none is `other`. A class that does
  not occur in the logged rows is omitted. If the gate stays at 0.5 and `proj` stays near zero,
  the semantics are not being used. With RenikudPlus, `_digit` is the one to watch: its encoder
  reads digits as digits, with no normalization.
- **Dependencies.** DictaBERT: `transformers` (in the `dev` group, or `pip install
  'pocket-tts[semantic]'`). RenikudPlus: `onnx` and `onnxruntime` (in the `dev` group, or
  `pip install 'pocket-tts[semantic-renikud]'`), imported only when `encoder: renikud_plus`.
  Without them, only that path fails, with an error that says so before anything is downloaded.

## Starting the 90K RenikudPlus curriculum

`training/configs/finetune_hebrew_semantic_renikud_90k.yaml` copies all seven phases,
dataset paths, augmentation, optimization and listening sentences from
`training/configs/finetune_hebrew_semantic_dicta_90k.yaml`. It starts from the released English
24-layer weights with a fresh Hebrew text embedding and semantic fusion, not from a DictaBERT
checkpoint. The only behavioral changes are the encoder and a separate run directory:
`runs/finetune_hebrew_semantic_renikud_90k`.

From the repository root, use the normal commands (no special launcher or extra flags):

```bash
uv sync --locked
NCCL_P2P_DISABLE=1 uv run torchrun --standalone --nproc-per-node 8 -m training.train \
    training/configs/finetune_hebrew_semantic_renikud_90k.yaml
```

The default `dev` group already includes `onnx` and `onnxruntime`. The first launch downloads
the pinned fp32 encoder (~1.2 GB) to the Hugging Face cache. Each rank runs its frozen encoder
on CPU with one ONNX thread; the TTS model trains on GPU. The per-GPU batch is 8 (effective
batch 64). All 90,000 optimizer steps are inferred from the curriculum and checked against
`max_steps`. As usual, log in to W&B first or set `WANDB_MODE=disabled`. Run a short smoke
test with a separate config and run directory before the full run; never shorten the production
config in place. The trainer automatically resumes if its run directory contains a checkpoint.

Validated on this Linux host with eight A40 GPUs: four optimizer steps at the configured
per-GPU batch of 8, finite losses and semantic metrics, validation on all five datasets,
checkpoint/export saving and clean process exit. The checkpoint, optimizer and EMA reload;
the fusion projection moves from zero, the encoder output stays unchanged, and the exported
Mimi weights match the frozen pretrained codec. This is a startup smoke test, not a quality
evaluation or a full 90K training run. Smoke artifacts live outside the production run directory.

## Checkpoints

The fusion's tensors live under `flow_lm.semantic_fusion.*` in training checkpoints and in the
`model.safetensors` export, and nowhere else. The encoders are never written; they are
referenced by name and revision.

| from \ into | plain model | semantic model |
|---|---|---|
| plain checkpoint / export | unchanged | loads; the fusion keeps its zero init |
| semantic checkpoint / export | loads; fusion tensors ignored with a warning | loads if the encoder matches |

This applies to all of these loaders:

- `TTSModel` (`weights_path`, `flow_lm.weights_path` and `--checkpoint`);
- warm-start in `training/modules/builders.py` (and the distillation teacher);
- `training/checkpointing.load_checkpoint` (resume, `training/eval`).

When neither side has the keys, the call is the plain strict `load_state_dict`. On a resume
that switches the feature on:

- the optimizer state, which has a different number of groups, is dropped with a warning;
- the EMA shadow is seeded with the fusion's initial values.

**Encoder identity.** A fusion is tied to the encoder it was trained with (`W` reads that
encoder's feature space). Checkpoints and exports therefore record the identity of the encoder:
`encoder`, `model_name`, `revision`, `filename` (RenikudPlus) and the layer band. It is the
`semantic_encoder` entry of a training checkpoint and the `pocket_tts.semantic_encoder`
metadata of a `model.safetensors` export. When fusion tensors would be loaded into a model
configured with another encoder, loading raises `ValueError` before anything is loaded, naming
both. A DictaBERT run on another revision is a mismatch too. A different layer band only warns.
Files written before the choice existed carry no record and count as DictaBERT's (the
family only can be compared), so existing checkpoints load into DictaBERT models as before and
are refused by RenikudPlus ones. Plain files and plain models are unaffected: no record is
written without a fusion.

**Caches.** No path caches encoder outputs (the encoder runs per row, per chunk), so there is
no latent cache to key. The latent caches of `hebrew_training/` are Mimi codec latents and do not
involve the semantic encoder.

## Deployment (Cloud Run)

One image serves models with either encoder or none. The encoders are not baked in: like the
model they are artifacts of the model store, keyed by the sha256 of their main file
(`encoders/<sha256>/`, published once per encoder revision with `deploy/publish_encoder.sh`). A
model's own config names the encoder it needs, so swapping the model hash is all a new model takes
(`deploy/swap_model.sh`); the server finds the matching published encoder, copies it to local disk
verified byte for byte and loads it from there (`TTSModel.load_model(semantic_encoder_dir=...)`,
`SemanticFusion.load_encoder(source)`). Details and the environment overrides are in
[deploy/README.md](deploy/README.md).

## Smoke test

Before any long run, train a few steps on a tiny manifest with `max_steps: 4`, `batch_size: 1`,
`compile: false` and `ema_decay: 0`, with the encoder under test (`encoder: renikud_plus` for
the new one). Then confirm:

- the loss is finite;
- the `semantic_*` metrics appear;
- the fusion parameters have gradients and change;
- the encoder has no gradients and is unchanged;
- the checkpoint reloads.

Use batch size 1 on an 8 GB card. DictaBERT adds about 1.7 GB in fp32 on the training device;
RenikudPlus runs on the CPU (about 1.9 GB of host memory). Unit tests use a fake encoder:
`uv run pytest tests/test_semantic_conditioning.py tests/test_semantic_renikud.py
training/tests/test_semantic_training.py`. `tests/test_semantic_renikud.py` also has one
integration test, skipped unless the pinned fp32 ONNX file is in the Hugging Face cache (or
`RENIKUD_ONNX` points at it): it checks that the 24 layer tensors are found, that adding them as
outputs leaves the model's own logits unchanged and that the pruned graph agrees with the full one.

## Measured

### DictaBERT (pinned revision)

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

### CPU cost per sentence (both encoders)

One thread (Apple Silicon, `torch.set_num_threads(1)`, ONNX Runtime with one intra-op thread),
layers 1–20 run, per sentence with tokenization included, on the first 400 rows of
`data_debug/tmp-sample-normalized.jsonl` after warm-up. This is in addition to the FlowLM, once
per text chunk.

| encoder | tokens per sentence (mean) | mean | p50 | p95 |
|---|---|---|---|---|
| DictaBERT | 29.9 | 38 ms | 34 ms | 46 ms |
| RenikudPlus | 133 (one per character) | 105 ms | 107 ms | 164 ms |

The mean sentence is 133 characters. DictaBERT's cost is nearly flat in this range (31 ms for
8 to 23 tokens); RenikudPlus's grows with the characters (about 37 ms for 34 characters, 66 ms
for 90, 105 ms on average, roughly 0.8 ms per character). It is about 2.8x DictaBERT on this
text, and a vocalized sentence costs what its unvocalized characters cost, since the marks are
deleted.

### RenikudPlus

- **Load.** 1–4 s with the file cached. Reading the 1.2 GB graph peaks at about 4.8 GB of host
  memory for a few seconds; steady state afterwards is about 1.9 GB (layers 1–20 of the graph,
  one thread). Handing `onnxruntime` the serialized bytes instead of a temporary file keeps
  about 3.7 GB resident for the life of the process, which is why the file route is used.
- **Equivalence.** The pruned graph's hidden states match the full graph's to 1e-3 on every
  tapped layer, and adding the 24 layer tensors as extra outputs leaves the three logits
  unchanged (same tolerance). The tapped tensors are `[1, seq, 1024]`.
- **Smoke test** (CPU, 24-layer Hebrew model warm-started from a plain export, 4 steps, batch
  size 1, `encoder: renikud_plus`): finite loss; `semantic_*` metrics and the four
  `semantic_gate_class_*` metrics logged; the fusion gets gradients and moves from its zero
  init; the encoder's output is bit-identical before and after; the checkpoint reloads with the
  optimizer state; the export carries the encoder record; a `TTSModel` loaded from the checkpoint generates finite
  audio; loading it into a DictaBERT-configured model, or a legacy DictaBERT export into a
  RenikudPlus-configured one, raises the encoder-mismatch error, and the legacy export still
  loads into its DictaBERT config.
  Cover was 0.90–1.00 per step (the rest is the null vector).

## Limitations and follow-ups

- **Window cap.** Longer inputs lose context across windows (DictaBERT 510 tokens, RenikudPlus
  2,046 characters). Training rows and inference chunks are far shorter.
- **Context is per chunk.** `TTSModel` fuses each chunk on its own, so no context crosses
  sentence chunks. Encoding the whole input once and slicing per chunk is the follow-up.
- **English semantics are a gamble.** DictaBERT lowercases (`NASA` becomes `nas ##a`) and has
  about 5.3k Latin entries. RenikudPlus reads Latin letters one at a time, lowercased, and was
  trained for Hebrew. Case and spelling reach the model only through the lookup table. Watch
  `semantic_gate_latin` and `semantic_gate_class_latin`.
- **Digits and Latin in RenikudPlus.** Digits and Latin letters have tokens but no
  normalization: the encoder sees `15` as the characters 1 and 5. Whether a null latent for
  them would help is a possible ablation, not done here.
- **K, N and U are lowercased too**, although the vocabulary has the capitals. This follows the
  rule "lowercase Latin"; the effect on those three letters is not measured.
- **Cost.** DictaBERT is 435M parameters (about 1.7 GB in fp32); RenikudPlus's ONNX file is
  1.2 GB and costs about 2.8x DictaBERT's CPU time per sentence.
- **Peak memory when loading RenikudPlus** is about 4.8 GB for a few seconds (see above).
- **No batching yet.** The encoder runs once per row in training.
- **Not wired in:** the older `hebrew_training/train.py` trainer, `training/eval/librispeech.py`
  and `hebrew_training/watch_eval.py`. With semantic weights they raise a clear "needs texts"
  error.
- **Evaluation.** This proof of concept is judged by listening. No homograph or CER tooling was
  added, and the comparison of the two encoders (for example a linear probe on homograph and
  stress sentences) is not part of this change.
- **Not done here:** speaker or gender inputs of RenikudPlus, converting the ONNX model to
  PyTorch, a full-length TTS training run with it (only short smoke tests have run).
