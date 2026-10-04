"""Several runs' per-row correctness on the eval sets, by prompt-length quartile and by task.

Reads `evaluation/multistep_eval.py --out` files (one per run and set) and buckets the rows
every run answered by prompt length in characters, so all runs are compared on the same
rows whatever their tokenizers. The multi-step file is split by task, since its tasks
differ in length and kind.

    python research/scripts/sets_by_length.py \
        --run v19=reports/b1/rows/v19_{}.json --run e1a=reports/e1a/rows/e1a_{}.json \
        --run e1b=reports/e1b/rows/e1b_{}.json

`{}` in each pattern is the set's name (multistep_v14_eval, generated_v16_eval, ...). The
last two columns are the last run minus each of the others.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from strands_decider.data.format import load_examples
from strands_decider.prompting import build_prompt

SETS = ["multistep_v14_eval", "generated_v16_eval", "generated_v18_eval",
        "adequacy_hs2_eval", "adequacy_gen_eval"]


def table(title: str, idx: list[int], res: dict, length: dict) -> None:
    names = list(res)
    last = names[-1]
    order = sorted(idx, key=length.__getitem__)
    q = len(order) // 4
    parts = [order[k * q:(k + 1) * q] if k < 3 else order[3 * q:] for k in range(4)]
    print(f"\n{title}  (n={len(idx)})")
    print(f"  {'prompt chars':>17} {'n':>5} " + " ".join(f"{n:>6}" for n in names)
          + "".join(f"  {last}-{n:>4}" for n in names[:-1]))
    for p in [*parts, order]:
        acc = {n: sum(res[n][i] for i in p) / len(p) for n in names}
        label = "all" if p is order else f"{length[p[0]]:,}-{length[p[-1]]:,}"
        print(f"  {label:>17} {len(p):>5} " + " ".join(f"{acc[n]:6.3f}" for n in names)
              + "".join(f"  {acc[last] - acc[n]:+8.3f}" for n in names[:-1]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="NAME=PATTERN, {} = set name")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    runs = dict(r.split("=", 1) for r in args.run)
    for s in SETS:
        rows = load_examples([f"{args.data}/{s}.jsonl"])
        res = {n: json.load(open(p.format(s), encoding="utf-8")) for n, p in runs.items()}
        keep = [i for i in range(len(rows)) if all(res[n][i] is not None for n in runs)]
        length = {i: len(build_prompt(rows[i].state, rows[i].to_question())[0]) for i in keep}
        if s == "multistep_v14_eval":
            by: dict[str, list[int]] = defaultdict(list)
            for i in keep:
                by[rows[i].task].append(i)
            for task in sorted(by):
                table(f"{s} / {task}", by[task], res, length)
        else:
            table(s, keep, res, length)


if __name__ == "__main__":
    main()
