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


PAGE = """<!doctype html><meta charset="utf-8"><title>Hebrew alignment tagging</title>
<style>
 body{font:14px system-ui,sans-serif;margin:0;padding:16px;background:#14161a;color:#e8e8ea}
 h1{font-size:15px;margin:0 0 10px;font-weight:600}
 #bar{color:#8a8f98;margin-bottom:10px}
 #wrap{position:relative;background:#1c1f25;border-radius:6px;padding:0;overflow:hidden}
 canvas{display:block;width:100%;height:190px}
 #words{margin:14px 0;direction:rtl;text-align:right;line-height:2.4}
 .w{padding:4px 9px;margin:0 3px;border-radius:5px;background:#252932;cursor:pointer}
 .w.cur{background:#3d6fd6;color:#fff}
 .w.done{background:#2b6b46}
 #hint{color:#8a8f98;margin-top:14px;line-height:1.9;font-size:13px}
 kbd{background:#2a2e37;border:1px solid #3a3f4a;border-radius:4px;padding:1px 6px;font-size:12px}
 #cmp{margin-top:10px;font-variant-numeric:tabular-nums}
 .pill{display:inline-block;padding:3px 10px;margin-right:8px;border-radius:5px;background:#252932}
 #done{color:#4ec98a}
</style>
<h1>Hebrew alignment tagging</h1>
<div id="bar"></div>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="words"></div>
<div id="cmp"></div>
<div id="hint">
<kbd>space</kbd> play word &nbsp; <kbd>shift+space</kbd> play clip &nbsp;
<kbd>&larr;</kbd><kbd>&rarr;</kbd> word &nbsp;
<kbd>a</kbd>/<kbd>b</kbd> accept that aligner's END &nbsp;
<kbd>,</kbd><kbd>.</kbd> nudge 10ms &nbsp; <kbd>&lt;</kbd><kbd>&gt;</kbd> nudge 2ms &nbsp;
<kbd>enter</kbd> save clip &amp; next &nbsp; <kbd>s</kbd> skip
<br>Drag on the waveform to move the current word's end. The line you are setting is the
<b>end</b> of the highlighted word.
</div>
<script>
let clips=[],ci=0,wi=0,gold=[],buf=null,lead=0,ctx=null,peaks=null;
const cv=document.getElementById('cv'),g=cv.getContext('2d');
const fmt=t=>t.toFixed(3);
async function boot(){
  clips=await (await fetch('/api/clips')).json();
  ci=clips.findIndex(c=>!c.saved); if(ci<0)ci=0;
  await loadClip();
}
async function loadClip(){
  const c=clips[ci];
  gold=(c.saved||c.a).map(w=>({word:w.word,start:w.start,end:w.end}));
  wi=0;
  const r=await fetch('/api/audio/'+ci); lead=parseFloat(r.headers.get('X-Lead')||'0');
  const ab=await r.arrayBuffer();
  ctx=ctx||new (window.AudioContext||window.webkitAudioContext)();
  buf=await ctx.decodeAudioData(ab);
  peaks=null; draw(); render();
}
function draw(){
  const w=cv.clientWidth,h=190; cv.width=w*devicePixelRatio; cv.height=h*devicePixelRatio;
  g.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  g.clearRect(0,0,w,h);
  if(!buf)return;
  const d=buf.getChannelData(0),n=d.length;
  if(!peaks||peaks.length!==w*2){peaks=new Float32Array(w*2);
    for(let x=0;x<w;x++){let lo=1,hi=-1;const s=Math.floor(x*n/w),e=Math.floor((x+1)*n/w);
      for(let i=s;i<e;i++){const v=d[i];if(v<lo)lo=v;if(v>hi)hi=v;}
      peaks[x*2]=lo;peaks[x*2+1]=hi;}}
  g.fillStyle='#2f3540';
  for(let x=0;x<w;x++){const lo=peaks[x*2],hi=peaks[x*2+1];
    g.fillRect(x,h/2+lo*h/2.2,1,Math.max(1,(hi-lo)*h/2.2));}
  const dur=buf.duration;
  gold.forEach((wd,i)=>{
    const x0=(wd.start+lead)/dur*w, x1=(wd.end+lead)/dur*w;
    g.fillStyle=i===wi?'rgba(61,111,214,.22)':'rgba(255,255,255,.04)';
    g.fillRect(x0,0,Math.max(1,x1-x0),h);
    g.fillStyle=i===wi?'#6aa0ff':'#48505e';
    g.fillRect(x1,0,i===wi?2:1,h);
  });
  const c=clips[ci];
  if(c.b){const x=(c.b[wi].end+lead)/dur*w; g.fillStyle='#d98a3d'; g.fillRect(x,0,1,h);}
  const xa=(c.a[wi].end+lead)/dur*w; g.fillStyle='#4ec98a'; g.fillRect(xa,0,1,h);
}
function render(){
  const c=clips[ci];
  document.getElementById('bar').textContent=
    `clip ${ci+1}/${clips.length}  ·  word ${wi+1}/${gold.length}  ·  saved ${clips.filter(x=>x.saved).length}`;
  const ws=document.getElementById('words'); ws.innerHTML='';
  gold.forEach((wd,i)=>{const s=document.createElement('span');
    s.className='w'+(i===wi?' cur':''); s.textContent=wd.word;
    s.onclick=()=>{wi=i;draw();render();}; ws.appendChild(s);});
  const parts=[`<span class="pill" style="color:#4ec98a">A end ${fmt(c.a[wi].end)}</span>`];
  if(c.b)parts.push(`<span class="pill" style="color:#d98a3d">B end ${fmt(c.b[wi].end)}</span>`);
  parts.push(`<span class="pill" style="color:#6aa0ff">gold ${fmt(gold[wi].end)}</span>`);
  if(c.b)parts.push(`<span class="pill">A-B ${Math.round(Math.abs(c.a[wi].end-c.b[wi].end)*1000)} ms</span>`);
  document.getElementById('cmp').innerHTML=parts.join('');
}
let src=null;
function play(from,to){ if(src){try{src.stop()}catch(e){}}
  src=ctx.createBufferSource(); src.buffer=buf; src.connect(ctx.destination);
  src.start(0,Math.max(0,from+lead),Math.max(.05,to-from)); }
function setEnd(t){
  // Only the previous boundary and the clip end constrain this one. Clamping against the
  // NEXT word's end would silently refuse an 'accept aligner B' keypress whenever B places
  // this word later than A placed the following one -- which is exactly the disagreement
  // the annotator is here to resolve. Later boundaries are pushed instead.
  const lo=wi>0?gold[wi-1].end+.005:0;
  gold[wi].end=Math.min(Math.max(t,lo),buf.duration-lead);
  for(let j=wi+1;j<gold.length;j++){
    if(gold[j].end<gold[j-1].end+.005) gold[j].end=gold[j-1].end+.005;
  }
  for(let j=1;j<gold.length;j++) gold[j].start=gold[j-1].end;
  draw(); render();
}
cv.onmousedown=e=>{const r=cv.getBoundingClientRect();
  const t=(e.clientX-r.left)/r.width*buf.duration-lead; setEnd(t);
  const mv=ev=>setEnd((ev.clientX-r.left)/r.width*buf.duration-lead);
  const up=()=>{removeEventListener('mousemove',mv);removeEventListener('mouseup',up)};
  addEventListener('mousemove',mv); addEventListener('mouseup',up);};
addEventListener('keydown',async e=>{
  const c=clips[ci];
  if(e.code==='Space'){e.preventDefault();
    if(e.shiftKey)play(0,buf.duration-2*0); else play(gold[wi].start,gold[wi].end); return;}
  if(e.key==='ArrowLeft'){wi=Math.max(0,wi-1);draw();render();return;}
  if(e.key==='ArrowRight'){wi=Math.min(gold.length-1,wi+1);draw();render();return;}
  if(e.key==='a'){setEnd(c.a[wi].end);return;}
  if(e.key==='b'&&c.b){setEnd(c.b[wi].end);return;}
  if(e.key===','){setEnd(gold[wi].end-.01);return;}
  if(e.key==='.'){setEnd(gold[wi].end+.01);return;}
  if(e.key==='<'){setEnd(gold[wi].end-.002);return;}
  if(e.key==='>'){setEnd(gold[wi].end+.002);return;}
  if(e.key==='Enter'){
    await fetch('/api/gold',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({index:ci,words:gold})});
    clips[ci].saved=gold.map(w=>({...w}));
    if(ci<clips.length-1){ci++;await loadClip();}else{render();}
    return;}
  if(e.key==='s'){if(ci<clips.length-1){ci++;await loadClip();}return;}
});
addEventListener('resize',()=>{peaks=null;draw()});
boot();
</script>
"""


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
                return self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
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
