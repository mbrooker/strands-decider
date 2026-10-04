"""JevBench accuracy by prompt length, for runs on the same 231 tasks.

Buckets every task by the input-token count a reference run's results recorded (its own
tokenizer), then reports each run's accuracy per bucket, and per family for the families
named. Runs are `jevbench.sh` output directories (results.jsonl) or a run name in
research/data/jevbench_results.csv.

    python research/scripts/jevbench_by_length.py --ref reports/e1a/jevbench \
        --runs v19 reports/b1/jevbench reports/e1a/jevbench
"""
from __future__ import annotations

import argparse
import csv
import json
import os

BUCKETS = [(0, 500), (500, 1500), (1500, 3000), (3000, 10**9)]
FAMILIES = ["long_policy", "temporal_numeric", "multi_hop", "probability", "judge_hard",
            "ambiguous", "adequacy"]


def correctness(run: str) -> dict[str, bool]:
    path = os.path.join(run, "results.jsonl")
    if os.path.exists(path):
        return {r["task_id"]: bool(r["correct"]) for r in map(json.loads, open(path, encoding="utf-8"))}
    rows = csv.DictReader(open("research/data/jevbench_results.csv", encoding="utf-8"))
    out = {r["task_id"]: r["correct"] == "1" for r in rows if r["run"] == run}
    if not out:
        raise SystemExit(f"{run}: neither a jevbench.sh output nor a recorded run")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, help="the run whose input-token counts bucket the tasks")
    ap.add_argument("--runs", nargs="+", required=True)
    args = ap.parse_args()
    ref = {r["task_id"]: r for r in map(json.loads, open(os.path.join(args.ref, "results.jsonl"), encoding="utf-8"))}
    runs = {os.path.basename(os.path.dirname(r.rstrip("/"))) or r: correctness(r) for r in args.runs}
    names = list(runs)
    print(f"{'input tokens':>16} {'n':>4} " + " ".join(f"{n:>8}" for n in names))
    for lo, hi in BUCKETS:
        ids = [t for t, r in ref.items() if lo <= r["usage"]["input_tokens"] < hi]
        if ids:
            cells = " ".join(f"{sum(runs[n][t] for t in ids) / len(ids):8.3f}" for n in names)
            print(f"{lo:>7}-{hi if hi < 10**9 else '':<8} {len(ids):>4} {cells}")
    for fam in FAMILIES:
        ids = [t for t, r in ref.items() if r["family"] == fam]
        if ids:
            toks = sorted(ref[t]["usage"]["input_tokens"] for t in ids)
            cells = " ".join(f"{sum(runs[n][t] for t in ids):>8}" for n in names)
            print(f"{fam:>16} {len(ids):>4} {cells}   tokens {toks[0]}-{toks[-1]}, median {toks[len(toks) // 2]}")


if __name__ == "__main__":
    main()
