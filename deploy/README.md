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

The image holds code only. The model (weights, config, tokenizer) lives in a **model store**, a
Cloud Storage bucket, and a revision is pinned to one model by its weights hash.

```bash
deploy/publish_model.sh               # upload artifacts/model.safetensors (+ config, tokenizer); prints the sha256
deploy/deploy.sh                      # first deploy / code change: Cloud Build -> Artifact Registry -> Cloud Run
deploy/swap_model.sh [sha256]         # a new model only: no build, a new revision in about a minute
gcloud run services proxy hebrew-tts --region europe-west1 --port 8080   # then open localhost:8080
```

`GPU=1` on any of the three scripts targets the L4 service. `deploy/Dockerfile` is a CPU-only
image (4 torch threads). Defaults in `deploy/_common.sh`: project `project-71583fab-5cd0-4bba-a6f`,
region `europe-west1`, 4 vCPU / 16 GiB, min 0 / max 1 instances, IAM-protected (no auth in the
app itself).

### Model store

- Layout: `gs://<project>-hebrew-tts-models/<weights sha256>/{manifest.json, model.safetensors,
  hebrew_24l.yaml, tokenizer.model}`. The manifest holds each file's sha256 and size and is
  uploaded last: a model without one is an interrupted upload and is never served. A published
  directory is immutable (`publish_model.sh` is a no-op for the same weights and refuses the same
  weights with a different config/tokenizer), so a hash always means the same model.
- The service mounts the bucket read-only at `/models` and is pinned by env `MODEL_SHA256`
  (`MODEL_STORE=/models` is set in the image). At startup the server copies config and tokenizer
  (before the port opens) and the weights (in the background) to the paths the config expects,
  hashing every byte on the way; anything that does not match the manifest, or a pin that is not
  in the store, is an error. Copying to local disk, not reading through the FUSE mount, is
  deliberate: the model is then loaded from a fast local file. (Cloud Run's disk is in-memory, so
  the copy counts against the instance memory: 1.4 GB of 16 GiB.)
- **Swapping the model** = `swap_model.sh <sha256>`, which only updates `MODEL_SHA256` and so makes
  a new revision. **Rollback** = `swap_model.sh <old sha256>` (or shift traffic to the old revision;
  `gcloud run revisions list` shows which hash each carries).
- The revision has a startup probe on `GET /ready`, which answers 200 only once the model is
  loaded and verified. A revision that cannot load its model (bad hash, corrupt copy, weights the
  code cannot run) never passes it, the deploy fails, and the previous revision keeps serving.
- A model needs a new image only when the *code* has to change (new architecture, new config keys,
  sampler changes). `config`/weights that the image's code runs are swapped freely, whatever
  semantic encoder they use.
- **Semantic encoders (DictaBERT, RenikudPlus).** A model whose config has
  `flow_lm.semantic_conditioner` also needs a frozen encoder, which is not in the weights. It is a
  store artifact too, keyed by the sha256 of its main file, so **one image serves models with
  either encoder or none** (the dependencies of both are in `requirements.txt`; the image has no
  encoder baked in and runs with `HF_HUB_OFFLINE=1`):

  ```bash
  deploy/publish_encoder.sh dictabert        # once per encoder revision, not per model
  deploy/publish_encoder.sh renikud_plus
  ```

  Layout, next to the models in the same bucket: `encoders/<sha256>/{manifest.json, <files>}`
  (DictaBERT: `config.json model.safetensors tokenizer.json tokenizer_config.json vocab.txt`;
  RenikudPlus: the fp32 `model.onnx`, never the int8 one). The script downloads them from Hugging
  Face at the revision the shipped configs pin, hashes them and uploads the manifest last. Like a
  model, a published encoder is immutable.

  The *model's own config* names the encoder it needs (`encoder`, `model_name`, `revision`, and
  `filename` for RenikudPlus), because its fusion weights are tied to exactly that encoder. So
  swapping the model hash is all it takes: the server finds the published encoder whose manifest
  matches the config, copies its files to local disk hashing every byte (the same as the weights,
  and freed again once loaded), and loads it from there. A model without the section needs none; a
  config written before RenikudPlus existed (no `encoder` key) means DictaBERT. Both pin
  the same way as before: a model that names an encoder that is not published stops the server at
  startup with a message that says what to publish, so its revision never passes `/ready` and the
  old one keeps serving. `GET /health` and `/ready` report the encoder and its hash.

  Environment (both optional): `SEMANTIC_ENCODER_STORE` (where to copy encoders from; default
  `$MODEL_STORE/encoders`) and `SEMANTIC_ENCODER_SHA256` (pin one by hash, for the case of several
  published encoders with the same identity; it cannot select an encoder the config does not name).

  Memory: loading RenikudPlus peaks at about 4.8 GB for a few seconds (about 1.9 GB afterwards),
  and its 1.2 GB (DictaBERT: 1.6 GB) file is copied to Cloud Run's in-memory disk first, so budget
  for that on top of the 1.4 GB weights copy.
- To tell which model answered: `GET /health` (`model_sha256`) and the `X-Model-Sha256` header of
  `/tts`. The validation-set cache is keyed on the hash, so each model has its own.
- Locally nothing changes: without `MODEL_STORE` the server uses `artifacts/` and hashes the
  weights at startup. To try the store locally, `gcloud storage cp -r gs://<bucket>/<sha256> store/`
  and `MODEL_STORE=$PWD/store MODEL_SHA256=<sha256> uv run python -m deploy.hebrew_tts_server`
  (it writes the verified copies to `artifacts/` and `tokenizers/`).

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
- **צור מחדש עם הקול הנבחר** ("regenerate with the selected voice") next to the button deletes the
  loaded model's cached set and renders all sentences again with the voice selected in the **Voice
  prompt** list (not a random one; an uploaded file is refused, the set needs a named voice). It
  asks for confirmation, is refused while a run is in progress, and the new set replaces the old
  one as that model's cache for everyone. API: `POST /validation/start` with form fields
  `force=true` and `voice=<name from /voices>` (400 for a missing or unknown voice, 409 while a run
  is in progress; without the fields it behaves as before).
- The weights hash is the model store's `MODEL_SHA256` pin (verified while the weights are
  copied); without a store the server hashes the weights file at startup.
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
