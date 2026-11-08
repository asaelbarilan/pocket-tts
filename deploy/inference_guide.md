# Hebrew TTS: local inference guide (for coding agents)

This explains what must be in place to run the 24-layer Hebrew pocket-tts model locally and get
the same audio as the preview server. Text + a short voice prompt in, a 24 kHz mono WAV out.

The model is an experimental Hebrew finetune of a 24-layer pocket-tts teacher (flow language
model + the frozen Mimi audio codec). It is not Kyutai's original release.

Model weights SHA-256: `{{MODEL_SHA256}}`

{{SEMANTIC_ENCODERS}}

## 1. What you need

Download these from the server's start page. The page marks which is which.
When served from Cloud Run, `/download/model.safetensors` redirects (`302`) to a 30-minute
V4-signed Google Cloud Storage link so the weights download directly from GCS (use `curl -L`
when fetching from the command line).

| File | Goes to | Notes |
|---|---|---|
| `model.safetensors` | `artifacts/model.safetensors` | {{WEIGHTS_SIZE}}. FlowLM, Mimi, and semantic-fusion weights in one file. Verify the SHA-256 above. |
| `hebrew_24l.yaml` | `artifacts/hebrew_24l.yaml` | Model architecture and semantic-encoder config. Its `weights_path` and `tokenizer_path` are relative to the project root, so keep the layout below. |
| `tokenizer.model` | `tokenizers/tokenizer.model` | SentencePiece tokenizer (Hebrew, 8000 pieces). |
| `hebrew_tts_inference_code.zip` | unzip at the project root | `pocket_tts/` (model + semantic conditioners), the needed parts of `training/` (sampler), `hebrew_tts_infer.py` (working example), `requirements.txt`, and this guide. |
| `eval_sentences_he.jsonl` | anywhere | Optional: 40 test sentences (`id`, `category`, `text`). |
| a voice prompt | anywhere | You supply it: any speech recording (WAV, or FLAC/MP3/OGG/M4A via soundfile). At least 0.5 s; only the first 5 s are used. The output imitates this speaker. |

Project layout (run everything from the root, because the config's paths are relative):

```
project/
  artifacts/hebrew_24l.yaml
  artifacts/model.safetensors
  tokenizers/tokenizer.model
  pocket_tts/                 # from the zip
  training/                   # from the zip (args.py, curriculum.py, text_augment.py, modules/)
  hebrew_tts_infer.py         # from the zip
  requirements.txt            # from the zip
```

### Semantic conditioning (`dictabert`, `renikud_plus`, or combined)

Unvocalized Hebrew is ambiguous, so semantic models fuse hidden states from one or two frozen
bidirectional encoders into the SentencePiece lookup embeddings via a learned gate stored in
`model.safetensors` (`flow_lm.semantic_fusion.*`). The frozen encoders themselves are **not**
stored inside `model.safetensors`; their exact identity (encoder type, Hugging Face repo,
pinned commit `revision`, `filename`, and layer band) is declared in `artifacts/hebrew_24l.yaml`
and recorded in the `model.safetensors` metadata header (`TTSModel.load_model` verifies they
match and refuses mismatched weights).

Depending on the model, `artifacts/hebrew_24l.yaml` configures one of three semantic modes
(see the required encoder list at the top of this guide for what *this* model uses):

1. **DictaBERT (`flow_lm.semantic_conditioner` with `encoder: dictabert`)** —
   `dicta-il/dictabert-large` at its pinned revision (~1.6 GB, layers 8–20). Runs in PyTorch via
   `transformers` and moves to your chosen `--device` (`cuda`, `mps`, or `cpu`). Local directory
   files: `config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`.
2. **RenikudPlus (`flow_lm.semantic_conditioner` with `encoder: renikud_plus`)** —
   `notmax123/RenikudPlus` at its pinned revision, file `model.onnx` (~1.2 GB fp32 ONNX; never
   `model_int8.onnx`). Needs `onnx` and `onnxruntime`. At load time `pocket_tts` prunes the ONNX
   graph in memory to keep only layers `first_layer..last_layer` (8–20) and drops FiLM and the
   heads. `artifacts/hebrew_24l.yaml` sets `onnx_provider: cpu` (`onnx_threads: 1`) so the
   standard CPU `onnxruntime` package in `requirements.txt` works out of the box on CPU, Apple
   Silicon MPS, and CUDA. (If you run on a CUDA GPU and install a compatible `onnxruntime-gpu`
   instead of `onnxruntime`, you can set `onnx_provider: cuda`.)
3. **Combined mode (`flow_lm.semantic_conditioners`, a list of two entries: `dictabert` + `renikud_plus`)** —
   Runs both frozen encoders on the input text, projects each branch (`flow_lm.semantic_fusion.branches.{0,1}.*`),
   and combines them with a shared learned gate (`flow_lm.semantic_fusion.gate.*`). Branch order
   is part of the checkpoint identity and must not be reordered.

**How the frozen encoders are loaded locally:**
- **Automatic download (default):** `HebrewTTS()` / `TTSModel.load_model()` downloads the exact
  pinned revision(s) from Hugging Face on first run and caches them under `~/.cache/huggingface`.
  After the first run, you can set `HF_HUB_OFFLINE=1` to run completely offline.
- **From local directories (`--semantic-encoder-dir`):** You can pre-download the pinned encoder
  files into local folders and pass `--semantic-encoder-dir` (one directory for a single-encoder
  model, or two directories in the config's branch order for a combined model):
  ```bash
  python -c "
  from huggingface_hub import snapshot_download, hf_hub_download
  snapshot_download('dicta-il/dictabert-large', revision='a972ffb612c03bcfb128f10f5ef4462f7adb9844',
                    allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt'],
                    local_dir='semantic-encoders/dictabert')
  hf_hub_download('notmax123/RenikudPlus', 'model.onnx', revision='679c56ca449d41873fb8ff7711ddf7563d28198f',
                  local_dir='semantic-encoders/renikud_plus')
  "
  HF_HUB_OFFLINE=1 python hebrew_tts_infer.py \
      --semantic-encoder-dir semantic-encoders/dictabert semantic-encoders/renikud_plus \
      --text "שלום, זהו מבחן קצר של המודל." --voice my_voice.wav --out out.wav --seed 1
  ```
- Do not change `revision`, `encoder`, `filename`, `first_layer`/`last_layer`, or branch order in
  `artifacts/hebrew_24l.yaml`, and do not remove the `semantic_conditioner(s)` section: without
  it the trained fusion weights are ignored and speech quality degrades.

## 2. Environment

- Python 3.10 to 3.14 (tested with 3.12).
- `torch >= 2.5` (tested with 2.13). torch 2.4.0 produces wrong audio.
  Install it for your platform from https://pytorch.org/get-started/locally/ (CPU, CUDA or MPS).
- The rest: `pip install -r requirements.txt` (includes `transformers`, `tokenizers`, `onnx`,
  and `onnxruntime` for both semantic encoders).
  `soundfile` is required: it writes the WAV and reads non-WAV voice prompts.
- No GPU is needed. Measured at the default settings (below): about 1.8x realtime on an Apple
  M5 CPU, about 1.2x on a Cloud Run NVIDIA L4, about 0.25x on a 4 vCPU cloud CPU. Batch size is
  always 1, so a bigger GPU does not help.

## 3. Quick start

```bash
python hebrew_tts_infer.py --text "שלום, זהו מבחן קצר של המודל." --voice my_voice.wav --out out.wav --seed 1
```

Or from Python:

```python
from hebrew_tts_infer import HebrewTTS, write_wav

tts = HebrewTTS()                                # device auto: cuda > mps > cpu
# Or with pre-downloaded local semantic encoder folder(s):
# tts = HebrewTTS(semantic_encoder_dir=["semantic-encoders/dictabert", "semantic-encoders/renikud_plus"])
voice = tts.encode_voice("my_voice.wav")         # reuse it for many sentences
wave = tts.synthesize("שלום, זהו מבחן קצר של המודל.", voice, seed=1)
if wave is not None:                             # float32 tensor [samples] at tts.sample_rate (24000)
    write_wav("out.wav", wave, tts.sample_rate)
```

`hebrew_tts_infer.py` is short. Read it first; sections 4 and 5 explain each step.

## 4. The pipeline, step by step

1. **Load.** `TTSModel.load_model(config=Path("artifacts/hebrew_24l.yaml"), semantic_encoder_dir=...)`
   builds FlowLM, Mimi, and `flow_lm.semantic_fusion` (`SemanticFusion` or `CombinedSemanticFusion`),
   loads `model.safetensors` strictly (verifying the recorded semantic encoder identity), and
   loads the frozen semantic encoder(s). Move `tts.mimi` and `tts.flow_lm` (plus any PyTorch
   semantic encoder `enc.model` in `flow_lm.semantic_fusion._encoders`) to your device in `eval()` mode.
2. **Wrap for sampling.** `TrainableTTS(flow_lm, LSD(normalize=False), TrainArgs()).eval()`, from
   `training.modules.model`, `training.modules.samplers` and `training.args`. This wrapper is the
   only source of classifier-free guidance (see "Pitfalls"). The sampler has no weights of its own
   at decode time, so `TrainArgs()` defaults are fine.
3. **Fold and tokenize.**
   ```python
   folded = fold_characters(text)
   tokens = torch.tensor(flow_lm.conditioner.tokenizer.sp.encode(folded), dtype=torch.long)
   ```
   Pass the text as is: no normalization, no number expansion and no chunking, with one exception:
   first run it through `pocket_tts.utils.text_fold.fold_characters`. Training text had curly
   quotes (`’ “ ”`), Unicode dashes (`– —`), the Hebrew geresh `׳` and invisible marks folded to
   plain characters, so without the fold they reach the model as byte tokens it never saw.
   Keep `folded`: semantic conditioning needs the exact string `tokens` came from so it can
   align the semantic encoder's character spans to SentencePiece tokens (and it re-encodes
   `folded` to verify `tokens` match).
4. **Encode the voice prompt.**
   - Read the audio (`pocket_tts.data.audio.audio_read`) and keep the first `prompt_seconds` (5).
   - Resample to 24 kHz mono (`pocket_tts.data.audio_utils.convert_audio(audio, sr, 24000, 1)`).
   - `mimi.encode_to_latent(mono[None])` gives `[1, frames, channels]`. Take `[0].float()`, a
     `[frames, channels]` tensor. This is the `voice` condition. Cache it per voice.
5. **Generate latents.**
   ```python
   latents = model.generate(
       [tokens],
       [voice],
       texts=[folded],  # required by semantic conditioning; must match `tokens`
       max_frames=int(max_seconds * mimi.frame_rate),
       temp=0.3,
       n_steps=1,
       cfg_coef=2.0,
       eos_threshold=-1.0,
       eos_countdown=3,
   )[0]
   ```
   The result is `[frames, channels]` at 12.5 frames per second (80 ms each). Seed with
   `torch.manual_seed(seed)` right before the call if you want repeatable output.
6. **Check the result.**
   - Fewer than 8 frames means the model emitted EOS at once. That is a failure, not speech.
     Retry with another voice prompt or a higher `eos_threshold`.
   - `frames >= max_frames` means EOS never fired and the audio was cut at `max_seconds`.
7. **Decode to audio.**
   ```python
   ratio = round(mimi.encoder_frame_rate / mimi.frame_rate)
   state = init_states(mimi, 1, (latents.shape[0] + 8) * ratio)    # pocket_tts.modules.stateful_module
   wave = mimi.decode_from_latent(latents[None].to(device), state)[0, 0].float().cpu()
   ```
8. **Write** `wave.clamp(-1, 1)` as 16-bit PCM WAV at `tts.sample_rate` (24000).

## 5. Sampling settings

The defaults are what training and the training evals use. They are also the server's defaults.

{{DEFAULTS_TABLE}}

Notes:
- `cfg` is the guidance scale. `1.0` turns guidance off, which is about 2x faster and sounds
  worse. The model was trained with 20% text/voice dropout so it works with cfg 2.0.
- `eos_threshold`: EOS fires when its logit exceeds the threshold. Higher stops later and cuts
  off less. Lower stops earlier.
- `eos_countdown`: once EOS triggers at frame `t`, generation ticks down by 1 each frame
  (including frame `t`) and stops when it hits 0 (`3` keeps 2 extra 80 ms frames so the final
  word releases naturally).
- `n_steps` is the number of LSD decode steps. 1 is the fastest.
- `max_seconds` is the maximum length of one generation. The server accepts at most 60 s and at
  most 1500 characters per request. For longer text, split it into sentences yourself and
  generate each one with the same encoded voice; the model has no built-in chunking.

## 6. Pitfalls

- **Do not use `pocket_tts`'s own generation (`TTSModel.generate_audio`, `pocket-tts generate` or
  `pocket-tts serve`).** The package has no classifier-free guidance, so it always samples at
  cfg 1.0 and sounds worse than this model's intended output. Use `TrainableTTS.generate`, as
  above.
- **Always pass `texts=[folded]` to `TrainableTTS.generate`.** Semantic conditioning aligns
  encoder states to the SentencePiece tokens of `texts` and raises `ValueError` if `texts` is
  missing or was not folded the same way as `tokens`.
- **Run from the project root.** The config's `weights_path` and `tokenizer_path` are relative.
  Either `chdir` there first (the example script does) or edit the paths in the yaml.
- **CPU and GPU audio differ.** The same seed gives different, equally valid audio across
  devices and torch versions. With the same device type, torch version and seed, it was bit-for-bit repeatable in our checks. Do
  not compare audio across devices bit by bit.
- **Mimi and the semantic encoders stay frozen.** Never train or re-export them as part of inference.
- **Runtime type checking.** `pocket_tts` enables beartype on import. Set
  `POCKET_TTS_NO_BEARTYPE=1` before importing it for a little less overhead.
- **Threads.** `pocket_tts` pins `torch.set_num_threads(1)` at import. On CPU, call
  `torch.set_num_threads(4)` after importing it; more than 4 does not help at batch size 1.
- **Not thread-safe.** Use one generation at a time per model instance.
- **Quality is not guaranteed.** This is an experiment. The server's validation set (the
  "הצג סט ולידציה" button) shows what to expect across plain text, nikud, numbers, English and
  Latin words, and homographs. Judge by listening.

## 7. Sanity check

Run the quick start with any clear voice prompt. You should get a few seconds of intelligible
Hebrew speech in that speaker's voice, roughly 0.3 s per word, a 24000 Hz mono WAV, no
`hit max length` warning, and no "ended speech immediately" error. If the output is noise or
garbage, check, in order: the weights SHA-256, that the semantic encoder section in
`artifacts/hebrew_24l.yaml` was not removed, `torch >= 2.5`, that you used
`TrainableTTS.generate` with `texts=[fold_characters(text)]` and not `TTSModel.generate_audio`,
and that the voice prompt is real speech.
