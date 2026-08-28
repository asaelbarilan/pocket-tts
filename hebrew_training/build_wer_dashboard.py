"""Render hebrew_eval.jsonl as a WER/CER-over-time chart with the audio beside it.

    python -m hebrew_training.build_wer_dashboard --run-dir runs/finetune_hebrew

Writes `<run-dir>/hebrew_eval.html`. Serve the run directory and open it:

    python -m http.server 8000 --directory runs/finetune_hebrew

The chart is inline SVG and the page has no external assets, so it works over a plain
static server on a remote box with no network access from the browser.

Pass --run-dir more than once to overlay several runs on the same axes.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

W, H = 900, 340
PAD_L, PAD_R, PAD_T, PAD_B = 62, 20, 20, 46
SERIES = [("wer", "#d94f4f", "WER"), ("cer", "#3b7dd8", "CER")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, action="append", required=True,
                        help="Repeat to overlay runs. The page is written beside the first.")
    parser.add_argument("--output", type=Path,
                        help="Defaults to <first run-dir>/hebrew_eval.html")
    return parser.parse_args()


def load(run_dir: Path) -> list[dict]:
    path = run_dir / "hebrew_eval.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # A checkpoint can be scored twice across restarts; the last write wins.
    return sorted({row["step"]: row for row in rows}.values(), key=lambda r: r["step"])


def chart(runs: list[tuple[str, list[dict]]]) -> str:
    points = [(r["step"], r[key]) for _, rows in runs for r in rows
              for key, _, _ in SERIES if r.get(key) is not None]
    if not points:
        return '<p class="empty">No scored checkpoints yet.</p>'
    max_step = max(p[0] for p in points)
    max_value = max(0.05, max(p[1] for p in points)) * 1.15

    def x(step: float) -> float:
        return PAD_L + (step / max_step if max_step else 0) * (W - PAD_L - PAD_R)

    def y(value: float) -> float:
        return H - PAD_B - (value / max_value) * (H - PAD_T - PAD_B)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="WER and CER by step">']
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        value = max_value * fraction
        parts.append(f'<line class="grid" x1="{PAD_L}" y1="{y(value):.1f}" '
                     f'x2="{W - PAD_R}" y2="{y(value):.1f}"/>')
        parts.append(f'<text class="tick" x="{PAD_L - 8}" y="{y(value) + 4:.1f}" '
                     f'text-anchor="end">{value:.2f}</text>')
    for fraction in (0, 0.25, 0.5, 0.75, 1.0):
        step = max_step * fraction
        parts.append(f'<text class="tick" x="{x(step):.1f}" y="{H - PAD_B + 20}" '
                     f'text-anchor="middle">{int(step):,}</text>')
    parts.append(f'<text class="axis" x="{(W) / 2:.0f}" y="{H - 6}" '
                 f'text-anchor="middle">training step</text>')

    dash = ""
    for run_index, (name, rows) in enumerate(runs):
        if run_index:
            dash = f' stroke-dasharray="{2 + run_index * 3},3"'
        for key, colour, _ in SERIES:
            pts = [(r["step"], r[key]) for r in rows if r.get(key) is not None]
            if not pts:
                continue
            d = " ".join(f"{'M' if i == 0 else 'L'}{x(s):.1f},{y(v):.1f}"
                         for i, (s, v) in enumerate(pts))
            parts.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2"{dash}/>')
            for s, v in pts:
                parts.append(f'<circle cx="{x(s):.1f}" cy="{y(v):.1f}" r="3" fill="{colour}">'
                             f'<title>{html.escape(name)} step {s:,}: {v:.3f}</title></circle>')
            best = min(pts, key=lambda p: p[1])
            parts.append(f'<circle cx="{x(best[0]):.1f}" cy="{y(best[1]):.1f}" r="6" '
                         f'fill="none" stroke="{colour}" stroke-width="2"/>')
    parts.append("</svg>")
    return "".join(parts)


def clip_cell(run_dir_name: str, clip: dict) -> str:
    if clip.get("empty"):
        return '<td class="empty-gen">empty generation</td>'
    src = html.escape(f"{run_dir_name}{clip['file']}") if clip.get("file") else ""
    wer = clip.get("wer")
    badge = f'<span class="w">{wer:.2f}</span>' if wer is not None else ""
    return (f'<td><audio controls preload="none" src="{src}"></audio>{badge}'
            f'<div class="hyp">{html.escape(clip.get("hypothesis") or "—")}</div></td>')


def render(run_dirs: list[Path], output: Path | None = None) -> tuple[Path, int]:
    """Write the dashboard for `run_dirs`; returns (path, checkpoints charted).

    Importable so watch_eval.py can refresh the page after each checkpoint without
    shelling out.
    """
    runs = [(d.name, load(d)) for d in run_dirs]
    scored = [(n, r) for n, r in runs if r]
    output = output or (run_dirs[0] / "hebrew_eval.html")

    summary_rows = []
    for name, rows in scored:
        best = min(rows, key=lambda r: r["wer"])
        latest = rows[-1]
        summary_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{len(rows)}</td>"
            f"<td><b>{best['wer']:.3f}</b> @ {best['step']:,}</td>"
            f"<td>{latest['wer']:.3f} @ {latest['step']:,}</td>"
            f"<td>{latest['cer']:.3f}</td></tr>"
        )

    # Audio lives under each run dir; the page sits in the first one, so only that run's
    # clips are playable from here. Others still chart.
    primary = run_dirs[0]
    detail = []
    for name, rows in scored:
        prefix = "" if Path(name) == Path(primary.name) else None
        if prefix is None:
            continue
        sentences = [c["reference"] for c in rows[-1]["clips"]]
        head = "".join(f"<th>{html.escape(s)}</th>" for s in sentences)
        body = []
        for row in reversed(rows):
            cells = "".join(clip_cell("", c) for c in row["clips"])
            body.append(
                f'<tr><th class="step">{row["step"]:,}<span>WER {row["wer"]:.3f} · '
                f'CER {row["cer"]:.3f}</span></th>{cells}</tr>'
            )
        detail.append(
            f"<h2>{html.escape(name)}</h2>"
            f'<div class="scroll"><table class="clips"><thead><tr><th></th>{head}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table></div>"
        )

    page = f"""<!doctype html>
<meta charset="utf-8">
<title>Hebrew TTS — WER by checkpoint</title>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; padding: 24px;
        max-width: 1100px; color: #1c1c1e; background: #fff; }}
 h1 {{ font-size: 20px; margin: 0 0 4px; }}
 h2 {{ font-size: 16px; margin: 28px 0 8px; }}
 p.sub {{ color: #666; margin: 0 0 20px; }}
 svg {{ width: 100%; height: auto; }}
 .grid {{ stroke: #e6e6e9; stroke-width: 1; }}
 .tick {{ font-size: 11px; fill: #888; }}
 .axis {{ font-size: 12px; fill: #666; }}
 .legend span {{ margin-right: 16px; font-size: 13px; }}
 .swatch {{ display: inline-block; width: 22px; height: 3px; vertical-align: middle;
            margin-right: 6px; }}
 table {{ border-collapse: collapse; font-size: 13px; }}
 table.sum td, table.sum th {{ padding: 5px 14px 5px 0; text-align: left;
                               border-bottom: 1px solid #eee; }}
 .scroll {{ overflow-x: auto; }}
 table.clips th, table.clips td {{ border: 1px solid #e6e6e9; padding: 7px 9px;
                                   vertical-align: top; text-align: right; }}
 table.clips thead th {{ background: #fafafa; font-weight: 500; max-width: 230px;
                         direction: rtl; }}
 th.step {{ text-align: left; white-space: nowrap; font-weight: 600; }}
 th.step span {{ display: block; font-weight: 400; color: #777; font-size: 11px; }}
 audio {{ width: 190px; height: 30px; vertical-align: middle; }}
 .w {{ font-size: 11px; color: #666; margin-right: 6px; }}
 .hyp {{ direction: rtl; color: #444; margin-top: 4px; max-width: 230px; }}
 .empty-gen {{ color: #b00; font-size: 12px; }}
 .empty {{ color: #666; }}
</style>
<h1>Hebrew TTS — WER and CER by checkpoint</h1>
<p class="sub">Circled point is the best checkpoint per series. Five sentences per
checkpoint: a progress signal, not a measurement — confirm the winner with
<code>score_wer.py</code> over a few hundred clips.</p>
{chart(scored)}
<p class="legend">
 <span><i class="swatch" style="background:#d94f4f"></i>WER</span>
 <span><i class="swatch" style="background:#3b7dd8"></i>CER</span>
</p>
<table class="sum"><thead><tr><th>run</th><th>checkpoints</th><th>best WER</th>
<th>latest WER</th><th>latest CER</th></tr></thead>
<tbody>{''.join(summary_rows) or '<tr><td colspan="5">nothing scored yet</td></tr>'}</tbody>
</table>
{''.join(detail)}
"""
    output.write_text(page, encoding="utf-8")
    return output, sum(len(r) for _, r in scored)


def main() -> None:
    args = parse_args()
    output, total = render(args.run_dir, args.output)
    print(f"wrote {output} ({total} scored checkpoints)")
    print(f"serve it: python -m http.server 8000 --directory {args.run_dir[0]}")


if __name__ == "__main__":
    main()
