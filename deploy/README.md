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

## Cloud Run (CPU)

```bash
deploy/deploy.sh                      # Cloud Build -> Artifact Registry -> Cloud Run
gcloud run services proxy hebrew-tts --region europe-west1 --port 8080   # then open localhost:8080
```

`deploy/Dockerfile` is a CPU-only image (4 torch threads). Defaults in `deploy.sh`: project
`project-71583fab-5cd0-4bba-a6f`, region `europe-west1`, 4 vCPU / 16 GiB, min 0 / max 1
instances, IAM-protected (no auth in the app itself).

- The model loads in a background thread: the port opens at once, `/health` says
  `"warming up..."` and `/tts` answers `503 warming up...` until it is ready (~8 s on Cloud Run).
- The service uses `--no-cpu-throttling` for that reason: with request-based CPU the background
  load would be throttled between requests. The cost is that an instance is billed while alive
  (including its idle linger), not just during requests.
- Measured on Cloud Run (4 vCPU): ~0.2-0.3x realtime at cfg 2.0, ~0.46x at cfg 1.0
  (the same text runs at ~1.8x / ~3.4x on an Apple M5).

`GPU=1 deploy/deploy.sh` deploys the same image built with CUDA torch (`cu130`) as a separate
service `hebrew-tts-l4` (1x NVIDIA L4, 4 vCPU, 16 GiB, no zonal redundancy; `--device auto`
picks cuda). Measured: ~1.2x realtime at cfg 2.0 (~68 ms per 80 ms frame, launch-bound at
batch 1), ~2x at cfg 1.0, i.e. about 4-5x faster than the Cloud Run CPU service.

## Validation set

The test page has a **הצג סט ולידציה** button. It renders the 40 fixed sentences of
`deploy/eval_sentences_he.jsonl` (plain, nikud, numbers, English / Latin / loanwords in Hebrew,
homographs) with one randomly chosen voice prompt and the default sampling settings (seed 1234),
and shows them with players. The page polls `GET /validation` every 5 s while it runs;
`POST /validation/start` starts (or resumes) a run.

- Audio and a `manifest.json` are cached under `--cache-dir` (env `VALIDATION_CACHE_DIR`;
  default `deploy/validation_cache/`, gitignored), in a folder named `<weights sha256[:16]>-<device>`.
  A different model, device kind, sentence file or sampling default never reuses another's audio.
- A finished cache is shown at once. A run cut short resumes with the same voice.
- The weights hash is the `MODEL_SHA256` env var baked into the image (`deploy.sh` computes it,
  and the Dockerfile verifies it against the weights at build time); without it the server hashes
  the weights file at startup.
- On Cloud Run the cache is a Cloud Storage bucket (`<project>-hebrew-tts-cache`) mounted at
  `/cache`; `deploy.sh` creates it and grants the service account access. Cloud Run cannot attach
  disks.

## MILIM-Bench sample picker

The page has a **בחר דוגמה מ MILIM** button (hidden when the dataset is not installed). It opens a
dialog over the 1,653 sentences of [renikud/MILIM-Bench](https://huggingface.co/datasets/renikud/MILIM-Bench):
pick a category or all of them, tick up to 5 samples, and **צור** generates them in one request
with the page's current voice and sampling settings. Each sample's labelled word is highlighted;
hover it to see the expected IPA. Results appear below the player as text + audio.

- The dataset repo is downloaded into the image at build time (`MILIM_REVISION` build arg, pinned
  to a commit; the server reads `$MILIM_DIR/data/gold.tsv`). The server never contacts Hugging Face.
- Locally, fetch it once: `uv run python -m deploy.milim_bench` (into the gitignored
  `deploy/milim_bench_data/`; override with `--milim-dir` / `MILIM_DIR`).
- `GET /milim` returns `{categories, samples: [{id, category, text, targets: [{index, ipa}]}]}`;
  `index` is the whitespace-split word position. Labels that point past the end of their text
  (two rows in v5) are dropped.
- `POST /milim/generate` takes the `/tts` voice and sampling fields plus repeated `ids` (1-5) and
  streams NDJSON, one line per sample as it finishes: `{id, text, status: ok|eos|error,
  audio_seconds, generation_seconds, hit_max_length, wav (base64)}`. Samples run one after another,
  each under the generation lock like the validation set.

## Local inference downloads

The top of the page links the files needed to run the model locally: `model.safetensors`,
`tokenizer.model`, `hebrew_24l.yaml`, `hebrew_tts_inference_code.zip` (`pocket_tts`, the sampler
part of `training/`, `hebrew_tts_infer.py`, `requirements.txt`), the example sentences, and
`inference_guide.md` (written for coding agents; the server fills in the weights hash and the
sampling defaults). They are served at `/download/<name>`; the weights support Range requests.
`hebrew_tts_infer.py` is a standalone copy of the server's pipeline; on the same CPU, torch and seed
it reproduces the validation-set audio byte for byte. Edit `deploy/inference_guide.md`
and `deploy/hebrew_tts_infer.py` together with the server's sampling code.

Notes:
- One request generates at a time (a lock); others wait. Validation generation takes the lock per
  sentence, so `/tts` requests interleave with it.
- Text goes to the tokenizer as is, except that curly quotes, Unicode dashes and invisible marks
  are folded to the plain characters training used (`pocket_tts/utils/text_fold.py`): no
  chunking, no other normalization. Keep it to what fits in
  `max_seconds`. Use the same text style as the training data (niqqud or not).
- Use a clean Hebrew prompt of ~3-5 s; a noisy prompt makes every output noisy.
- Voice latents are cached per voice/prompt length, so repeat requests skip the encoding.
- Needs the repo's environment (`uv sync`); Python 3.12 here because of the scipy issue on
  macOS 27.
