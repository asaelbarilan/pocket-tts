# Hosting the tagging tool

The point of putting it on the internet is to get more than one person marking, so the two
things that change from the localhost version are both about that: several people writing
at once, and knowing whose marks are whose.

## What changed for multiple annotators

`--multi` turns `--out` from a file into a directory and gives each annotator their own
file inside it. They are asked for a name once, and the name is remembered in the browser.

    /srv/marks/asael.jsonl
    /srv/marks/yoad.jsonl

This is not only bookkeeping. **Two people marking the same clip is the measurement that
tells us whether the gold set is worth anything.** Human-to-human disagreement is the floor
for the whole experiment: if two annotators differ by 80 ms on a boundary, then an aligner
that lands within 80 ms is already as good as the reference, and a benchmark that claims to
resolve finer than that is measuring noise. So let the first 20 or so clips overlap between
annotators on purpose, before splitting the rest up.

Each file has the same schema as the single-file output, so `alignment_disagreement.py`
scores an aligner against any one of them, or scores two annotators against each other,
with no conversion.

Saves take a lock and re-read before writing, so two people saving at the same moment cannot
drop each other's work.

## Access

`--token abc123` requires `?t=abc123` on every request, and the page carries it forward.
It stops a crawler finding the tool and writing to the gold set. It is **not** authentication:
anyone with the link has the link, and there is no HTTPS in this server, so the token
crosses the wire in the clear unless the host terminates TLS in front of it (Render,
Railway, and Fly all do). Adequate for a link handed to named colleagues; not for anything
that needs to stay private.

## Deploying

Build locally and push the built image, rather than pointing a host at the GitHub repo:

    docker build -f deploy/tagging/Dockerfile -t heb-tag .
    docker run -p 8080:8080 -e TAG_TOKEN=pick-something -v heb-marks:/srv/marks heb-tag

    docker tag heb-tag ghcr.io/<you>/heb-tag:1 && docker push ghcr.io/<you>/heb-tag:1

**Building from the repo will not work.** `.gitignore` excludes `/data*/`, so the clips and
the manifests are not in git; a host that clones and builds gets an image whose every clip
404s. Building locally is what puts the 35 MB of audio inside the image, and every host
that can deploy a Dockerfile can also deploy a pushed image. The alternative -- committing
the audio -- means 35 MB of wav in the repo to work around an ignore rule that is there on
purpose.

`HOST`, `PORT` and `TAG_TOKEN` are read from the environment, which is what the managed
hosts inject, so the same image runs on Render, Railway, Fly.io or a plain VPS with no
argument changes.

**The volume is not optional.** Without `-v` the marks live in the container's writable
layer, and the next redeploy or restart throws away every hour of hand-marking. On Render
that means a Disk mounted at `/srv/marks`; on Fly, a volume; on Railway, a persistent
volume. Check it survives a restart *before* asking anyone to start marking.

## The path trap

The manifests carry the absolute Windows paths they were built with:

    C:\Users\Asael\PycharmProjects\F5-TTS\pocket-tts\data\gold_set\clips\clip000.wav

which resolves nowhere on a Linux host. `--clips-root` looks each clip up by file name in a
directory instead, and the server refuses to start if the files are not where it expects,
rather than serving a page whose every clip fails to load.

The clips are their own wav files, so serving them needs soundfile and numpy alone. That is
why the image is ~200 MB rather than the several GB a torch base would cost, and why the
whole corpus ships inside it: 134 clips is 35 MB.
