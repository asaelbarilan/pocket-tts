# Hebrew TTS: local inference guide (for coding agents)

This explains what must be in place to run the 24-layer Hebrew pocket-tts model locally and get
the same audio as the preview server. Text + a short voice prompt in, a 24 kHz mono WAV out.

The model is an experimental Hebrew finetune of a 24-layer pocket-tts teacher (flow language
model + the frozen Mimi audio codec). It is not Kyutai's original release.

Model weights SHA-256: `{{MODEL_SHA256}}`

## 1. What you need

Download these from the server's start page. The page marks which is which.

| File | Goes to | Notes |
|---|---|---|
| `model.safetensors` | `artifacts/model.safetensors` | {{WEIGHTS_SIZE}}. FlowLM *and* Mimi weights in one file. Verify the SHA-256 above. |
| `hebrew_24l.yaml` | `artifacts/hebrew_24l.yaml` | Model architecture config. Its `weights_path` and `tokenizer_path` are relative to the project root, so keep the layout below. |
| `tokenizer.model` | `tokenizers/tokenizer.model` | SentencePiece tokenizer (Hebrew, 8000 pieces). |
| `hebrew_tts_inference_code.zip` | unzip at the project root | `pocket_tts/` (model code), the needed parts of `training/` (sampler), `hebrew_tts_infer.py` (working example), `requirements.txt`, and this guide. |
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

**Semantic conditioning.** The config has a `flow_lm.semantic_conditioner` section: a frozen
DictaBERT encoder (`dicta-il/dictabert-large`, a pinned revision, about 1.6 GB) reads the text and
its states are fused into the text embeddings by the small trained module that is inside
`model.safetensors`. DictaBERT itself is not in that file. It is downloaded from Hugging Face on the
first load (needs `transformers`, in `requirements.txt`) and cached under `~/.cache/huggingface`.
To work offline, fetch it once, then set `HF_HUB_OFFLINE=1`. Do not change the `revision` or delete
the section: the model was trained with exactly that encoder, and without the section the fusion
weights are silently ignored and the output is worse.

A model trained with the other encoder has `encoder: renikud_plus` in that section (with
`notmax123/RenikudPlus`, `model.onnx` and its pinned revision). It needs `onnx` and `onnxruntime`
(also in `requirements.txt`, so the same environment runs either kind), downloads the ~1.2 GB fp32 ONNX file the same way (cached, then
`HF_HUB_OFFLINE=1`), and runs on the CPU in ONNX Runtime with one thread. The two are not
interchangeable: the loader refuses weights trained with a different encoder than the config names.

Apart from DictaBERT (or RenikudPlus), no network access is needed at run time. (`pocket_tts` can
fetch `hf://` paths, but this config has none.)

## 2. Environment

- Python 3.10 to 3.14 (tested with 3.12).
- `torch >= 2.5` (tested with 2.13). torch 2.4.0 produces wrong audio.
  Install it for your platform from https://pytorch.org/get-started/locally/ (CPU, CUDA or MPS).
- The rest: `pip install -r requirements.txt`.
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
voice = tts.encode_voice("my_voice.wav")         # reuse it for many sentences
wave = tts.synthesize("שלום, זהו מבחן קצר של המודל.", voice, seed=1)
if wave is not None:                             # float32 tensor [samples] at tts.sample_rate (24000)
    write_wav("out.wav", wave, tts.sample_rate)
```

`hebrew_tts_infer.py` is short. Read it first; sections 4 and 5 explain each step.

## 4. The pipeline, step by step

1. **Load.** `TTSModel.load_model(config=Path("artifacts/hebrew_24l.yaml"))` builds FlowLM and Mimi
   from the config and loads `model.safetensors` strictly. Keep only `tts.mimi` and
   `tts.flow_lm`, and put them in `eval()` mode on your device.
2. **Wrap for sampling.** `TrainableTTS(flow_lm, LSD(normalize=False), TrainArgs()).eval()`, from
   `training.modules.model`, `training.modules.samplers` and `training.args`. This wrapper is the
   only source of classifier-free guidance (see "Pitfalls"). The sampler has no weights of its own
   at decode time, so `TrainArgs()` defaults are fine.
3. **Tokenize.** `flow_lm.conditioner.tokenizer.sp.encode(text)` gives token ids; make them a
   `torch.long` tensor. Pass the text as is: no normalization, no number expansion and no
   chunking, with one exception: first run it through
   `pocket_tts.utils.text_fold.fold_characters`. Training text had curly quotes (`’ “ ”`),
   Unicode dashes (`– —`), the Hebrew geresh `׳` and invisible marks folded to plain
   characters, so without the fold they reach the model as byte tokens it never saw.
   Everything else (digits, abbreviations, punctuation) goes as written: the model was trained
   on both forms. Plain text and text with nikud both work.
4. **Encode the voice prompt.**
   - Read the audio (`pocket_tts.data.audio.audio_read`) and keep the first `prompt_seconds` (5).
   - Resample to 24 kHz mono (`pocket_tts.data.audio_utils.convert_audio(audio, sr, 24000, 1)`).
   - `mimi.encode_to_latent(mono[None])` gives `[1, frames, channels]`. Take `[0].float()`, a
     `[frames, channels]` tensor. This is the `voice` condition. Cache it per voice.
5. **Generate latents.**
   ```python
   latents = model.generate([tokens], [voice], max_frames=int(max_seconds * mimi.frame_rate),
                            temp=0.3, n_steps=1, cfg_coef=2.0, eos_threshold=-1.0)[0]
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
- `n_steps` is the number of LSD decode steps. 1 is the fastest.
- `max_seconds` is the maximum length of one generation. The server accepts at most 60 s and at
  most 1500 characters per request. For longer text, split it into sentences yourself and
  generate each one with the same encoded voice; the model has no built-in chunking.

## 6. Pitfalls

- **Do not use `pocket_tts`'s own generation (`TTSModel.generate_audio`, `pocket-tts generate` or
  `pocket-tts serve`).** The package has no classifier-free guidance, so it always samples at
  cfg 1.0 and sounds worse than this model's intended output. Use `TrainableTTS.generate`, as
  above.
- **Run from the project root.** The config's `weights_path` and `tokenizer_path` are relative.
  Either `chdir` there first (the example script does) or edit the paths in the yaml.
- **CPU and GPU audio differ.** The same seed gives different, equally valid audio across
  devices and torch versions. With the same device type, torch version and seed, it was bit-for-bit repeatable in our checks. Do
  not compare audio across devices bit by bit.
- **Mimi stays frozen.** Never train or re-export it as part of inference.
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
garbage, check, in order: the weights SHA-256, `torch >= 2.5`, that you used
`TrainableTTS.generate` and not `TTSModel.generate_audio`, and that the voice prompt is real
speech.
