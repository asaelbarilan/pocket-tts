
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

## Why the gold set is not sized in hours

Asked about hand-tagging 6 hours of audio. Measured cost: ~6,092 clips, ~44,600 boundaries,
34-150 hours of human work depending on care. The existing 134-clip set is 7.9 min of audio,
981 boundaries, 0.7-3.4 hours of work.

**The gold set is a ruler, not training data.** Its only job is to say which aligner to trust.
981 boundaries already separates aligners at tens of milliseconds; more hours buys precision
nobody uses. Once the best aligner is picked it runs over all 2,000+ hours unattended, which
is the only way the corpus ever gets aligned — hand-tagging 6 hours would clean 0.3% of it.

Word timings matter to training in two specific places (`training/dataloader.py:119-160`):
the random cut between two words that splits voice prompt from target, and the trim of
trailing silence. A wrong boundary puts the cut mid-word, so the audio target starts mid-word
while the text starts at a whole word. That is a real corruption — and the fix is a better
aligner over everything, not hand-marks over a slice.

Fine-tuning an aligner on hand-marks is also not the escape hatch: CTC learns alignment from
transcripts alone and never consumes word boundaries as labels.
