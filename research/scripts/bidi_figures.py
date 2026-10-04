"""Two figures for hobson-bidi: latency by model (box plots) and JevBench accuracy against Brier.

    python research/scripts/bidi_figures.py --reports ~/hobson-bidi/reports --out research/figures

Latency is per request, measured on one RTX 3090 under WSL2 in one session:
- JevBench, from `evaluation/jevbench/jevbench.sh` (each task an HTTP request to the served
  checkpoint, timed by JevBench). b1, e1a and e1b are from their runs; v19 and b2 were rerun
  locally for this figure, in reports/latency/.
- JF100, from `research/scripts/jf100_latency.py` (300 requests through the engine, in
  process).

Accuracy and Brier are each run's recorded JevBench result (the preregistrations' outcomes).

Static SVG, no script: light and dark from CSS custom properties under
prefers-color-scheme; one series colour (palette slot 1, validated on both surfaces);
model identity on the axis or as a direct label; a native <title> tooltip on every
mark; a CSV of the plotted numbers beside each figure.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from html import escape

MODELS = [  # (name, torso, subtitle)
    ("v19", "Qwen3.5-2B", "decoder"),
    ("b1", "T5Gemma 2 1B+1B", "encoder-decoder"),
    ("b2", "T5Gemma 2 4B+4B", "encoder-decoder"),
    ("e1a", "T5Gemma 2B-it", "encoder"),
    ("e1b", "T5Gemma 2B-it", "encoder + state cache"),
]
REFERENCE = "e1b"
# Recorded JevBench outcomes (PREREGISTRATION-b1, -b2, -e1; v19 from the upstream record).
JEVBENCH = {"v19": (168, 0.342), "b1": (153, 0.411), "b2": (178, 0.287),
            "e1a": (153, 0.373), "e1b": (171, 0.372)}

STYLE = """
<style>
  .viz { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
         --axis:#c3c2b7; --s1:#2a78d6; --s1-fill:rgba(42,120,214,0.16);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  @media (prefers-color-scheme: dark) {
    .viz { --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
           --axis:#383835; --s1:#3987e5; --s1-fill:rgba(57,135,229,0.22); }
  }
  .bg { fill: var(--surface); }
  .title { fill: var(--ink); font-size: 16px; font-weight: 600; }
  .sub { fill: var(--ink2); font-size: 12px; }
  .panel { fill: var(--ink); font-size: 13px; font-weight: 600; }
  .tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
  .name { fill: var(--ink); font-size: 12px; font-weight: 600; }
  .name.ref { font-weight: 700; }
  .torso { fill: var(--muted); font-size: 10px; }
  .val { fill: var(--ink2); font-size: 10.5px; font-variant-numeric: tabular-nums; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--axis); stroke-width: 1; }
  .box { fill: var(--s1-fill); stroke: var(--s1); stroke-width: 1.5; }
  .median { stroke: var(--ink); stroke-width: 2; }
  .whisker { stroke: var(--s1); stroke-width: 1.5; }
  .outlier { fill: var(--s1); fill-opacity: 0.45; }
  .dot { fill: var(--s1); stroke: var(--surface); stroke-width: 2; }
  .ring { fill: none; stroke: var(--s1); stroke-width: 1.5; }
  .note { fill: var(--muted); font-size: 11px; }
</style>
"""


def five(values: list[float]) -> dict:
    v = sorted(values)
    q = statistics.quantiles(v, n=4, method="inclusive")
    iqr = q[2] - q[0]
    lo_fence, hi_fence = q[0] - 1.5 * iqr, q[2] + 1.5 * iqr
    inside = [x for x in v if lo_fence <= x <= hi_fence]
    return {"n": len(v), "min": v[0], "q1": q[0], "median": q[1], "q3": q[2], "max": v[-1],
            "p95": v[int(0.95 * (len(v) - 1))], "lo": inside[0], "hi": inside[-1],
            "outliers": [x for x in v if x < lo_fence or x > hi_fence]}


def load_latency(reports: str) -> dict[str, dict[str, list[float]]]:
    def jev(path: str) -> list[float]:
        return [r["latency_s"] * 1000 for r in map(json.loads, open(path, encoding="utf-8"))]

    def jf(path: str) -> list[float]:
        return [r["latency_ms"] for r in map(json.loads, open(path, encoding="utf-8"))]

    jevbench_paths = {"b1": "b1/jevbench", "e1a": "e1a/jevbench", "e1b": "e1b/jevbench",
                      "v19": "latency/jevbench_v19", "b2": "latency/jevbench_b2"}
    out: dict[str, dict[str, list[float]]] = {"JevBench": {}, "JF100": {}}
    for m, _, _ in MODELS:
        out["JevBench"][m] = jev(os.path.join(reports, jevbench_paths[m], "results.jsonl"))
        out["JF100"][m] = jf(os.path.join(reports, "latency", f"jf100_{m}.jsonl"))
    return out


def fmt_ms(x: float) -> str:
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"


def latency_svg(lat: dict, path: str) -> list[dict]:
    W, H = 920, 470
    top, bottom, left = 96, 92, 64
    panel_w, gap = 400, 56
    stats = {b: {m: five(lat[b][m]) for m, _, _ in MODELS} for b in lat}
    lo = min(s["min"] for b in stats for s in stats[b].values())
    hi = max(s["max"] for b in stats for s in stats[b].values())
    ylo = 10 ** math.floor(math.log10(lo * 0.9) * 2) / 10 ** 0 if False else lo * 0.85
    yhi = hi * 1.15
    ticks = [t for t in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if ylo <= t <= yhi]

    def y(v: float) -> float:
        frac = (math.log10(v) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo))
        return top + (H - top - bottom) * (1 - frac)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'class="viz" role="img" aria-labelledby="t d">', STYLE,
             f'<title id="t">Request latency by model, JevBench and JF100</title>',
             '<desc id="d">Box plots of per-request latency in milliseconds on one RTX 3090, log scale, '
             'for v19, b1, b2, e1a and e1b. ' + escape("; ".join(
                 f"{b} {m} median {stats[b][m]['median']:.0f} ms" for b in stats for m, _, _ in MODELS))
             + '</desc>',
             f'<rect class="bg" width="{W}" height="{H}"/>',
             f'<text class="title" x="{left - 40}" y="30">Request latency by model</text>',
             f'<text class="sub" x="{left - 40}" y="50">Per request, one RTX 3090 under WSL2, log scale. '
             'Box: quartiles · line: median · whiskers: 1.5 × IQR · dots: beyond.</text>']
    rows = []
    for p, bench in enumerate(("JevBench", "JF100")):
        x0 = left + p * (panel_w + gap)
        n = stats[bench][MODELS[0][0]]["n"]
        how = "231 tasks, served over HTTP" if bench == "JevBench" else "300 requests, in process"
        parts.append(f'<text class="panel" x="{x0}" y="{top - 22}">{bench}</text>'
                     f'<text class="sub" x="{x0 + (78 if bench == "JevBench" else 52)}" y="{top - 22}">'
                     f'· {how}</text>')
        for t in ticks:
            parts.append(f'<line class="grid" x1="{x0}" x2="{x0 + panel_w}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                         f'<text class="tick" x="{x0 - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t:,}</text>')
        parts.append(f'<text class="tick" x="{x0 - 8}" y="{top - 6}" text-anchor="end">ms</text>')
        parts.append(f'<line class="axis" x1="{x0}" x2="{x0 + panel_w}" y1="{H - bottom}" y2="{H - bottom}"/>')
        slot = panel_w / len(MODELS)
        for i, (m, torso, kind) in enumerate(MODELS):
            s = stats[bench][m]
            cx = x0 + slot * (i + 0.5)
            bw = 30
            tip = escape(f"{m} on {bench} (n={s['n']}): min {fmt_ms(s['min'])}, Q1 {fmt_ms(s['q1'])}, "
                         f"median {fmt_ms(s['median'])}, Q3 {fmt_ms(s['q3'])}, p95 {fmt_ms(s['p95'])}, "
                         f"max {fmt_ms(s['max'])} ms")
            g = [f'<g><title>{tip}</title>',
                 f'<rect x="{cx - bw}" y="{y(s["hi"]) - 4:.1f}" width="{2 * bw}" '
                 f'height="{y(s["lo"]) - y(s["hi"]) + 8:.1f}" fill="transparent"/>',
                 f'<line class="whisker" x1="{cx}" x2="{cx}" y1="{y(s["hi"]):.1f}" y2="{y(s["q3"]):.1f}"/>',
                 f'<line class="whisker" x1="{cx}" x2="{cx}" y1="{y(s["q1"]):.1f}" y2="{y(s["lo"]):.1f}"/>',
                 f'<line class="whisker" x1="{cx - 7}" x2="{cx + 7}" y1="{y(s["hi"]):.1f}" y2="{y(s["hi"]):.1f}"/>',
                 f'<line class="whisker" x1="{cx - 7}" x2="{cx + 7}" y1="{y(s["lo"]):.1f}" y2="{y(s["lo"]):.1f}"/>',
                 f'<rect class="box" x="{cx - bw / 2}" y="{y(s["q3"]):.1f}" width="{bw}" '
                 f'height="{max(1.5, y(s["q1"]) - y(s["q3"])):.1f}" rx="3"/>',
                 f'<line class="median" x1="{cx - bw / 2}" x2="{cx + bw / 2}" '
                 f'y1="{y(s["median"]):.1f}" y2="{y(s["median"]):.1f}"/>']
            for o in s["outliers"]:
                g.append(f'<circle class="outlier" cx="{cx}" cy="{y(o):.1f}" r="2.4"/>')
            g.append(f'<text class="val" x="{cx + bw / 2 + 4}" y="{y(s["median"]) + 3.5:.1f}">'
                     f'{fmt_ms(s["median"])}</text></g>')
            parts += g
            ref = " ref" if m == REFERENCE else ""
            parts.append(f'<text class="name{ref}" x="{cx}" y="{H - bottom + 18}" text-anchor="middle">{m}</text>'
                         f'<text class="torso" x="{cx}" y="{H - bottom + 32}" text-anchor="middle">{escape(torso)}</text>'
                         f'<text class="torso" x="{cx}" y="{H - bottom + 44}" text-anchor="middle">{escape(kind)}</text>')
            rows.append({"benchmark": bench, "model": m, **{k: round(v, 1) for k, v in s.items()
                                                           if k not in ("outliers",)},
                         "outliers": len(s["outliers"])})
    parts.append(f'<text class="note" x="{left - 40}" y="{H - 12}">e1b is hobson-bidi\'s reference. '
                 'Numbers beside each box are medians, in ms.</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(parts) + "\n")
    return rows


def scatter_svg(path: str) -> None:
    W, H = 600, 430
    left, right, top, bottom = 72, 28, 84, 64
    xs = [b for _, b in JEVBENCH.values()]
    ys = [a for a, _ in JEVBENCH.values()]
    xlo, xhi = 0.27, 0.43
    ylo, yhi = 148, 182

    def x(v: float) -> float:
        return left + (W - left - right) * (v - xlo) / (xhi - xlo)

    def y(v: float) -> float:
        return top + (H - top - bottom) * (1 - (v - ylo) / (yhi - ylo))

    assert xlo < min(xs) and max(xs) < xhi and ylo < min(ys) and max(ys) < yhi
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
             f'class="viz" role="img" aria-labelledby="t d">', STYLE,
             '<title id="t">JevBench accuracy against Brier score</title>',
             '<desc id="d">' + escape("; ".join(
                 f"{m}: {a} of 231 tasks ({a / 231:.1%}), Brier {b:.3f}" for m, (a, b) in JEVBENCH.items()))
             + '</desc>',
             f'<rect class="bg" width="{W}" height="{H}"/>',
             f'<text class="title" x="24" y="30">JevBench: accuracy against Brier score</text>',
             '<text class="sub" x="24" y="50">231 public tasks. Better is up and to the left '
             '(more correct, lower Brier).</text>']
    for t in range(150, 183, 5):
        parts.append(f'<line class="grid" x1="{left}" x2="{W - right}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>'
                     f'<text class="tick" x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t}</text>')
    for t in (0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42):
        parts.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{H - bottom}"/>'
                     f'<text class="tick" x="{x(t):.1f}" y="{H - bottom + 16}" text-anchor="middle">{t:.2f}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{W - right}" y1="{H - bottom}" y2="{H - bottom}"/>'
                 f'<line class="axis" x1="{left}" x2="{left}" y1="{top}" y2="{H - bottom}"/>'
                 f'<text class="sub" x="{(left + W - right) / 2}" y="{H - 22}" text-anchor="middle">'
                 'Brier score (mean squared error of the predicted distribution; lower is better)</text>'
                 f'<text class="sub" transform="translate(20 {(top + H - bottom) / 2}) rotate(-90)" '
                 'text-anchor="middle">Tasks correct, of 231</text>')
    offsets = {"v19": (10, 4, "start"), "b1": (-10, 4, "end"), "b2": (10, 4, "start"),
               "e1a": (10, 4, "start"), "e1b": (10, -8, "start")}
    torso = {m: t for m, t, _ in MODELS}
    for m, (a, b) in JEVBENCH.items():
        cx, cy = x(b), y(a)
        dx, dy, anchor = offsets[m]
        tip = escape(f"{m} ({torso[m]}): {a}/231 correct ({a / 231:.1%}), Brier {b:.3f}")
        ref = m == REFERENCE
        parts.append(f'<g><title>{tip}</title><circle cx="{cx:.1f}" cy="{cy:.1f}" r="14" fill="transparent"/>'
                     + (f'<circle class="ring" cx="{cx:.1f}" cy="{cy:.1f}" r="10"/>' if ref else "")
                     + f'<circle class="dot" cx="{cx:.1f}" cy="{cy:.1f}" r="6"/>'
                     f'<text class="name{" ref" if ref else ""}" x="{cx + dx:.1f}" y="{cy + dy:.1f}" '
                     f'text-anchor="{anchor}">{m}</text>'
                     f'<text class="val" x="{cx + dx:.1f}" y="{cy + dy + 13:.1f}" text-anchor="{anchor}">'
                     f'{a} · {b:.3f}</text></g>')
    parts.append(f'<text class="note" x="24" y="{H - 4}">Each run\'s recorded result. e1a and b1 tie at 153; '
                 'e1b, ringed, is hobson-bidi\'s reference.</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(parts) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reports", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = latency_svg(load_latency(args.reports), os.path.join(args.out, "bidi_latency_box.svg"))
    with open(os.path.join(args.out, "bidi_latency_box.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    scatter_svg(os.path.join(args.out, "bidi_jevbench_accuracy_brier.svg"))
    with open(os.path.join(args.out, "bidi_jevbench_accuracy_brier.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "torso", "correct_of_231", "accuracy", "brier"])
        for m, t, _ in MODELS:
            a, b = JEVBENCH[m]
            w.writerow([m, t, a, round(a / 231, 4), b])
    for r in rows:
        print(f"{r['benchmark']:9} {r['model']:4} median {r['median']:7.1f}  p95 {r['p95']:7.1f}  n {r['n']}")


if __name__ == "__main__":
    main()
