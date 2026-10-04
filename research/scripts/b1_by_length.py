"""b1 against v19 by input length, and class-balanced accuracy (PREREGISTRATION-b1.md, outcome).

Reads the per-row correctness that `evaluation/multistep_eval.py --out` writes for two
checkpoints on the same eval file, and reports, per file:

    - accuracy of each model in quartiles of prompt length (characters, so both models are
      bucketed on the same rows whatever their tokenizers), and the difference;
    - class-balanced accuracy: the mean over gold labels of each label's accuracy (g4's
      "generated adequacy, balanced");
    - rows both answered (rows over either model's window are left out of both).

If b1's deficit against v19 grows with length, the long-context reading is the suspect
(T5Gemma 2's encoder attends locally, +-256 tokens, in 22 of its 26 layers); if it is
flat, the cause is elsewhere.

    python research/scripts/b1_by_length.py --data data/x_eval.jsonl --a v19.json --b b1.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from strands_decider.data.format import load_examples
from strands_decider.prompting import build_prompt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True)
    ap.add_argument("--a", required=True, help="per-row correctness of model A (multistep_eval --out)")
    ap.add_argument("--b", required=True, help="per-row correctness of model B")
    ap.add_argument("--names", default="v19,b1")
    args = ap.parse_args()
    na, nb = args.names.split(",")

    rows = load_examples([args.data])
    a, b = json.load(open(args.a)), json.load(open(args.b))
    assert len(a) == len(b) == len(rows), (len(a), len(b), len(rows))
    keep = [i for i in range(len(rows)) if a[i] is not None and b[i] is not None]
    length = {i: len(build_prompt(rows[i].state, rows[i].to_question())[0]) for i in keep}

    order = sorted(keep, key=length.__getitem__)
    q = len(order) // 4
    quartiles = [order[k * q:(k + 1) * q] if k < 3 else order[3 * q:] for k in range(4)]
    print(f"{args.data}: {len(keep)} rows answered by both ({len(rows) - len(keep)} left out)")
    print(f"  {'prompt chars':>17}  {'n':>4}  {na:>6}  {nb:>6}  {nb + '-' + na:>7}")
    for idx in quartiles:
        lo, hi = length[idx[0]], length[idx[-1]]
        acc_a = sum(a[i] for i in idx) / len(idx)
        acc_b = sum(b[i] for i in idx) / len(idx)
        print(f"  {lo:>7,}-{hi:<9,}  {len(idx):>4}  {acc_a:6.3f}  {acc_b:6.3f}  {acc_b - acc_a:+7.3f}")
    acc_a = sum(a[i] for i in keep) / len(keep)
    acc_b = sum(b[i] for i in keep) / len(keep)
    print(f"  {'all':>17}  {len(keep):>4}  {acc_a:6.3f}  {acc_b:6.3f}  {acc_b - acc_a:+7.3f}")

    by_label: dict[str, list[int]] = defaultdict(list)
    for i in keep:
        by_label[rows[i].options[rows[i].label][0]].append(i)
    if 1 < len(by_label) <= 6:
        bal = {m: sum(sum(c[i] for i in v) / len(v) for v in by_label.values()) / len(by_label)
               for m, c in ((na, a), (nb, b))}
        per = ", ".join(f"{k}: n={len(v)} {na} {sum(a[i] for i in v) / len(v):.3f} "
                        f"{nb} {sum(b[i] for i in v) / len(v):.3f}" for k, v in sorted(by_label.items()))
        print(f"  class-balanced: {na} {bal[na]:.3f}  {nb} {bal[nb]:.3f}   ({per})")


if __name__ == "__main__":
    main()
