# Hebrew TTS test API

Text + voice prompt -> WAV, for trying the exported 24-layer Hebrew model
(`artifacts/model.safetensors`, config `artifacts/hebrew_24l.yaml`).

It samples with the training code (`TrainableTTS.generate`), so it uses **CFG 2.0**, EOS
threshold -1.0 and temperature 0.3 like the training evals. `pocket-tts serve` has no CFG.

```bash
# from the repo root
mkdir -p deploy/voices && cp some_clean_hebrew_prompt.wav deploy/voices/   # gitignored
uv run python -m deploy.hebrew_tts_server            # device: cuda > mps > cpu
uv run python -m deploy.hebrew_tts_server --device cpu --host 0.0.0.0 --port 8002
```

Open http://127.0.0.1:8002/ for a small test page, or use the API:

```bash
curl -o out.wav -F "text=שלום, זהו מבחן" -F voice=my_voice.wav localhost:8002/tts
curl -o out.wav -F "text=שלום, זהו מבחן" -F voice_wav=@/path/to/prompt.wav -F cfg=1 -F seed=7 localhost:8002/tts
```

| Endpoint | |
|---|---|
| `POST /tts` | form fields below, returns `audio/wav` (24 kHz, mono, 16-bit) |
| `GET /voices` | files in `deploy/voices/` (`.wav .flac .mp3 .ogg .m4a`) |
| `GET /health` | device, layer count, defaults |
| `GET /` | test page |

`/tts` fields: `text` (required), one of `voice` (name from `/voices`) or `voice_wav`
(upload, wins), then optional:

| Field | Default | Notes |
|---|---|---|
| `temp` | 0.3 | sampling temperature |
| `cfg` | 2.0 | guidance; 1.0 = off (one backbone pass instead of two) |
| `n_steps` | 1 | flow decode steps |
| `eos_threshold` | -1.0 | EOS fires when its logit > threshold: higher stops later |
| `max_seconds` | 30 | generation cap; training clips were <= 30 s |
| `prompt_seconds` | 5.0 | only the first N s of the prompt are used (training used <= 5 s) |
| `seed` | random | fixed seed = reproducible output |

Response headers: `X-Audio-Seconds`, `X-Generation-Seconds`, `X-Realtime-Factor`,
`X-Hit-Max-Length` (true = EOS never fired), `X-Device`.

Notes:
- One request generates at a time (a lock); others wait.
- Text goes to the tokenizer as is: no chunking, no normalization. Keep it to what fits in
  `max_seconds`. Use the same text style as the training data (niqqud or not).
- Use a clean Hebrew prompt of ~3-5 s; a noisy prompt makes every output noisy.
- Voice latents are cached per voice/prompt length, so repeat requests skip the encoding.
- Needs the repo's environment (`uv sync`); Python 3.12 here because of the scipy issue on
  macOS 27.
