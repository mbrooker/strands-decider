"""Per-request latency on JF100 (100 items x 3 option rotations), in process.

Builds each request exactly as JF100's strands-decider runner does (`~/sd_eval/run/eval_jf100.py`:
option letters A-D as choice names, one question per request, rotation by trial) and times
`engine.ask` end to end: tokenisation, the forward, the readout and the answer pulled back to
Python, which syncs the device. A few untimed requests warm up first. Writes one JSON line
per request: item, trial, latency_ms, correct.

    python research/scripts/jf100_latency.py CHECKPOINT ~/sd_eval/jf100 --out lat.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from strands_decider.infer import load_engine
from strands_decider.schema import ChoiceQuestion

TRIALS = 3


def presented(item: dict, trial: int) -> tuple[dict, str]:
    """JF100's own rotation (jf100.core.presented): letters A-D, shifted by trial."""
    pairs = list(item["options"].items())
    shift = trial % 4
    pairs = pairs[shift:] + pairs[:shift]
    options = {label: text for label, (_, text) in zip("ABCD", pairs, strict=False)}
    gold = next(label for label, (orig, _) in zip("ABCD", pairs, strict=False) if orig == item["answer"])
    return options, gold


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("jf100")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.jf100)
    raw = (root / "data/items.jsonl").read_bytes()
    want = json.loads((root / "data/manifest.json").read_text())["sha256"]
    assert hashlib.sha256(raw).hexdigest() == want, "JF100 items differ from the frozen manifest"
    items = [json.loads(line) for line in raw.decode().splitlines()]

    engine = load_engine(args.checkpoint, device=args.device)

    def ask(item: dict, trial: int) -> tuple[str, str]:
        options, gold = presented(item, trial)
        a = engine.ask(item["state"], {"answer": ChoiceQuestion(
            instructions=item["question"], criteria=options)}).answers["answer"]
        return a.choice, gold

    for item in items[: args.warmup]:
        ask(item, 0)
    rows = []
    for item in items:
        for t in range(TRIALS):
            t0 = time.perf_counter()
            choice, gold = ask(item, t)
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"item_id": item["id"], "trial": t, "latency_ms": round(ms, 2),
                         "correct": choice == gold})
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    lat = sorted(r["latency_ms"] for r in rows)
    print(json.dumps({"checkpoint": args.checkpoint, "n": len(rows),
                      "correct": sum(r["correct"] for r in rows),
                      "p50_ms": round(statistics.median(lat), 1),
                      "p95_ms": round(lat[int(0.95 * (len(lat) - 1))], 1)}))


if __name__ == "__main__":
    main()
