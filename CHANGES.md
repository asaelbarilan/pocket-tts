
## Tagging tool: hostable for several annotators

`align_tag_server.py` can now be put on the internet so more than one person marks.

- `--multi` gives each annotator their own file under `--out`, keyed by a name they enter
  once. Saves lock and re-read, so simultaneous saves cannot drop each other.
- `--token` / `TAG_TOKEN` gates every request. Not authentication — no HTTPS in this
  server, so it relies on the host terminating TLS.
- `--clips-root` rebases clips by file name. The manifests carry Windows paths that resolve
  nowhere on a Linux host; the server now refuses to start rather than 404 every clip.
- Audio no longer imports `training.dataloader`, so no torch: the clips are already their
  own wav files and soundfile + numpy is the whole dependency. Image ~200 MB.
- `HOST`, `PORT`, `TAG_TOKEN` read from the environment, which is what managed hosts inject.
- Keyboard shortcuts no longer fire while typing — every letter of a name was also a
  transport shortcut.

`deploy/tagging/Dockerfile` and `docs/hosting-the-tagging-tool.md`. Build locally and push
the image: `/data*/` is gitignored, so a host building from GitHub gets no audio.

**Overlap the first ~20 clips between annotators on purpose.** Human-to-human disagreement
is the floor for the whole alignment experiment — an aligner that lands inside it is already
as good as the reference.

**The volume is not optional.** Without one, a redeploy discards every hour of marking.
