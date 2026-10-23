# Dataloader preview

A web page to look at what the training dataloader produces: the text the model is trained on
and the voice prompt / target audio of every sample, with the real loader code
(`training.dataloader.DataLoader`, the curriculum mixing, cuts, digit/niqqud text draws).

```bash
uv run python -m hebrew_training.data_preview.server training/configs/finetune_hebrew_curriculum.yaml
# --host 0.0.0.0 --port 8765   to serve a remote box
# --tokenizer path/to.model    when the config's tokenizer is not on this machine
```

Open http://127.0.0.1:8765. Alpine.js is loaded from a CDN, nothing to build.

- **Set / phase / dataset**: phases are `data.curriculum`'s (0-based); a config with plain
  `train_jsonl` / `valid_jsonl` has the single phase 0. Train follows the phase mix. Eval uses
  the validation sets of the phase's datasets, weighted by the mix (training's own validation
  ignores phases). Pick one dataset to see it alone.
- **Loader parameters**: seed, batch size, max durations, digit/niqqud probabilities, shuffle,
  fixed draws. Defaults are the config's for train and training's validation settings for eval
  (seed 0, no shuffle, fixed draws). Changing anything rebuilds the loader from its start.
- **Rows per manifest**: the unmodified `DataLoader` reads its whole manifest into memory when
  built (and again on each reload), which is slow and huge for millions of rows. With a row
  count set (or `--max-rows N` as the default), each loader is given a random subset of that
  many rows instead: picked by random byte offsets, no pass over the file, kept in order and
  cached under `~/.cache/pocket_tts/data_preview` (`--cache-dir`). Shuffle, epochs and "skip"
  then cover the subset. The page warns when the selected manifests are over 256 MB.
- **Skip batches first**: draws and drops N batches, then shows the next. **Restart** rebuilds
  the loader.

Config paths are relative to the repo root; the server changes to it.

## Mock data

```bash
uv run python -m hebrew_training.data_preview.make_mock_data   # uses ./my_voice.wav
uv run python -m hebrew_training.data_preview.server hebrew_training/data_preview/mock/curriculum.yaml
```

`mock/` (git-ignored) holds a small Hebrew + English manifest set pointing at one wav, a
two-phase `curriculum.yaml` and a plain `single.yaml`.

## Static export to S3

A fixed set of batches as a static site (no server, no parameters to edit): `index.html` with
Prev / Next (and ← → keys, `#3` in the URL opens the third batch), `style.css`, and one audio
file per prompt and target. Drawn by the same loader as the live page.

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...   # AWS_SESSION_TOKEN, AWS_DEFAULT_REGION if needed
uv run --with boto3 python -m hebrew_training.data_preview.export_s3 \
    training/configs/finetune_hebrew_curriculum.yaml \
    --s3 s3://my-bucket/previews/hebrew-phase1 --phase 1 --batches 20
```

- `--split train|eval`, `--dataset NAME`, `--skip N`, and every loader parameter
  (`--seed`, `--batch-size`, `--nikud-word-prob`, `--max-rows`, ...; defaults are the config's
  for that split). `--audio-format flac` roughly halves the size (lossless).
- Credentials are read from the environment and checked before any batch is built. boto3 is
  not a project dependency: `uv run --with boto3` supplies it.
- No ACLs are set. Make the prefix readable through the bucket policy, static website hosting or
  CloudFront; the page itself only needs Range requests for audio, which S3 supports.
- `index.html` is uploaded last, with `Cache-Control: no-cache`. Files of an earlier export
  under the same prefix are overwritten, not deleted.
- `--out-dir DIR` keeps a local copy (alone, without `--s3`, it needs no credentials);
  `--endpoint-url` targets S3-compatible storage.
