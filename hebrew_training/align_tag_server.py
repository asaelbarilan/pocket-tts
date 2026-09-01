"""A browser tool for hand-marking Hebrew word boundaries, to build the alignment gold set.

There is no Hebrew corpus with human word timings, so the gold set has to be made by hand,
and hand-marking is the expensive step. Praat can do it but costs a lot of friction per
boundary. This is built around the observation that makes the job cheap:

    most boundaries are already right in one of the aligners

So each word shows both proposals, and the common case is one keypress to accept the better
one. Dragging is the fallback, not the default.

    python -m hebrew_training.align_tag_server \\
        --a A_hebrew.jsonl --b B_mms.jsonl --out gold.jsonl

then open http://localhost:8080. Clips are served worst-disagreement-first, because that is
where a human judgement is worth the most; agreement regions teach nothing.

Saves after every clip, so it can be closed and reopened. Standard library plus soundfile —
no web framework, no CDN, works offline.

The output has the same schema as the input manifests, with `words` carrying the human
times, so `alignment_disagreement.py` can score any aligner against it directly.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import unicodedata
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

_WS = re.compile(r"\s+")
PAD = 0.25  # seconds of context served either side, so boundaries at the edges are judgeable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", type=Path, required=True, help="First aligned manifest.")
    parser.add_argument("--b", type=Path, help="Second aligned manifest, shown for comparison.")
    parser.add_argument("--out", type=Path, required=True, help="Gold manifest, written as you go.")
    parser.add_argument("--name-a", default="A")
    parser.add_argument("--name-b", default="B")
    parser.add_argument("--limit", type=int, default=100, help="How many clips to serve.")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def clean(text: str) -> str:
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def load(path: Path) -> dict[tuple[str, float], dict]:
    rows = {}
    if path is None or not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[(row["path"], round(float(row.get("start", 0.0)), 3))] = row
    return rows


def timed(row: dict) -> list[dict]:
    return [
        w
        for w in (row.get("words") or [])
        if w.get("start") is not None and w.get("end") is not None
    ]


def disagreement(a: dict, b: dict | None) -> float:
    """How far apart the two aligners are on this clip, used to order the work."""
    if not b:
        return 0.0
    wa, wb = timed(a), timed(b)
    if len(wa) != len(wb) or not wa:
        return 0.0
    ends = sorted(abs(float(x["end"]) - float(y["end"])) for x, y in zip(wa, wb))
    return ends[int(0.9 * (len(ends) - 1))]


def build_clips(args) -> list[dict]:
    rows_a, rows_b = load(args.a), load(args.b)
    clips = []
    for key, row in rows_a.items():
        words = timed(row)
        if len(words) < 2:
            continue
        other = rows_b.get(key)
        clips.append(
            {
                "path": row["path"],
                "start": key[1],
                "duration": float(row["duration"]),
                "transcript": row.get("transcript", ""),
                "a": [
                    {"word": clean(w["word"]), "start": float(w["start"]), "end": float(w["end"])}
                    for w in words
                ],
                "b": (
                    [
                        {
                            "word": clean(w["word"]),
                            "start": float(w["start"]),
                            "end": float(w["end"]),
                        }
                        for w in timed(other)
                    ]
                    if other and len(timed(other)) == len(words)
                    else None
                ),
                "score": disagreement(row, other),
            }
        )
    clips.sort(key=lambda c: c["score"], reverse=True)
    return clips[: args.limit]


def clip_wav(clip: dict) -> bytes:
    """The clip's audio as wav bytes, with PAD seconds of context either side."""
    import soundfile

    from training.dataloader import _load_window

    begin = max(0.0, clip["start"] - PAD)
    lead = clip["start"] - begin
    wav = _load_window(clip["path"], begin, clip["duration"] + lead + PAD, 24000)
    buffer = io.BytesIO()
    soundfile.write(buffer, wav, 24000, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), lead


PAGE_FILE = Path(__file__).with_name("align_tag_page.html")


def page() -> bytes:
    """Read the UI from disk on every request, so editing the page needs no restart."""
    return PAGE_FILE.read_bytes()


def make_handler(args, clips, saved):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: A003 -- quiet; progress is in the page
            pass

        def send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/":
                return self.send(200, page(), "text/html; charset=utf-8")
            if route == "/api/clips":
                payload = []
                for index, clip in enumerate(clips):
                    entry = dict(clip)
                    entry["saved"] = saved.get(index)
                    payload.append(entry)
                return self.send(
                    200,
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            if route.startswith("/api/audio/"):
                index = int(route.rsplit("/", 1)[1])
                try:
                    data, lead = clip_wav(clips[index])
                except Exception as exc:  # noqa: BLE001 -- a bad clip must not kill the server
                    return self.send(500, str(exc).encode(), "text/plain")
                return self.send(200, data, "audio/wav", {"X-Lead": f"{lead:.4f}"})
            return self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/api/gold":
                return self.send(404, b"not found", "text/plain")
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            saved[int(body["index"])] = body["words"]
            write_gold(args.out, clips, saved)
            return self.send(200, b'{"ok":true}', "application/json")

    return Handler


def write_gold(out: Path, clips: list[dict], saved: dict[int, list]) -> None:
    """Rewritten in full after each clip: 100 rows is nothing, and a partial append that
    crashed mid-write would be worse than the rewrite cost."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for index in sorted(saved):
            clip = clips[index]
            handle.write(
                json.dumps(
                    {
                        "path": clip["path"],
                        "start": clip["start"],
                        "duration": clip["duration"],
                        "transcript": clip["transcript"],
                        "words": [
                            {
                                "word": w["word"],
                                "start": round(float(w["start"]), 4),
                                "end": round(float(w["end"]), 4),
                            }
                            for w in saved[index]
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    clips = build_clips(args)
    if not clips:
        raise SystemExit(f"no usable clips in {args.a}")

    saved: dict[int, list] = {}
    if args.out.exists():
        by_key = {
            (r["path"], round(float(r["start"]), 3)): r["words"]
            for r in (
                json.loads(line)
                for line in args.out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        for index, clip in enumerate(clips):
            hit = by_key.get((clip["path"], round(clip["start"], 3)))
            if hit:
                saved[index] = hit
        print(f"resuming: {len(saved)} clips already marked")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args, clips, saved))
    words = sum(len(c["a"]) for c in clips)
    print(f"{len(clips)} clips, {words} boundaries to check")
    print(f"open http://localhost:{args.port}   (ctrl-c to stop; progress is saved per clip)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nstopped. {len(saved)} clips marked -> {args.out}")


if __name__ == "__main__":
    main()
