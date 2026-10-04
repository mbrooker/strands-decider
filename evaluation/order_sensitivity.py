"""How much does a checkpoint's answer depend on the order its options are listed in?

Each row is read twice: options in canonical order, then reversed (for a score rubric too:
reversal is meaning-preserving, as in training). The reversed distribution is mapped back
to canonical order and compared with the first:

    tv        total variation distance, 0.5 * sum |p - p_reversed|, averaged over rows
    flip      share of rows whose argmax changes
    acc       accuracy in canonical order, and reversed (a check that both are read right)

A causal torso reads each option only after the ones before it, so reversal changes what
every option has seen; a bidirectional encoder sees them all either way and is left with
position embeddings alone (docs/bidi-design.md). Upstream recorded ~0.016 for an untrained
pointer head (research/history.md); this measures trained checkpoints, on the same rows.

    python evaluation/order_sensitivity.py CHECKPOINT [--data data/holdout_v5_norule.jsonl] [--limit 3000]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import torch

from strands_decider.data.collate import CollatorConfig, SystemOneCollator
from strands_decider.data.format import load_examples
from strands_decider.modeling import StrandsDeciderModel


class _Reversed(SystemOneCollator):
    """The evaluation collator, with every row's options in reverse order."""

    def _option_order(self, ex):  # type: ignore[no-untyped-def]
        return list(reversed(range(ex.n_options)))


def _probs(model, coll, rows, device):  # type: ignore[no-untyped-def]
    b = {k: v.to(device) for k, v in coll(rows).items()}
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                n_slots=b["n_slots"], opt_idx=b.get("opt_idx"), state_len=b.get("state_len"))
    return out["log_probs"].float().exp().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data/holdout_v5_norule.jsonl")
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=None, help="write the summary as JSON here")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StrandsDeciderModel.load(args.checkpoint).to(device).eval()
    rows = load_examples([args.data])
    random.Random(0).shuffle(rows)
    rows = rows[: args.limit]
    cfg = CollatorConfig(max_length=model.config.max_length, num_slots=model.config.num_slots,
                         head_type=model.config.head_type,
                         state_mask=getattr(model.config, "state_mask", False))
    canonical = SystemOneCollator(model.tokenizer, cfg, train=False)
    reverse = _Reversed(model.tokenizer, cfg, train=False)

    stats: dict[str, list[tuple[float, int, int, int]]] = defaultdict(list)
    with torch.inference_mode():
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            p, q = _probs(model, canonical, chunk, device), _probs(model, reverse, chunk, device)
            for j, ex in enumerate(chunk):
                n = ex.n_options
                a, b = p[j, :n], q[j, :n].flip(0)  # slot k of the reversal shows option n-1-k
                row = (0.5 * float((a - b).abs().sum()), int(a.argmax() != b.argmax()),
                       int(a.argmax() == ex.label), int(b.argmax() == ex.label))
                for key in ("all", ex.kind, f"task:{ex.task}"):
                    stats[key].append(row)

    summary = {}
    for key in sorted(stats, key=lambda k: (k.startswith("task:"), k)):
        v = stats[key]
        m = len(v)
        summary[key] = {"n": m, "tv": sum(x[0] for x in v) / m, "flip": sum(x[1] for x in v) / m,
                        "acc": sum(x[2] for x in v) / m, "acc_reversed": sum(x[3] for x in v) / m}
        s = summary[key]
        print(f"{key:32} n={m:5}  tv {s['tv']:.4f}  flip {s['flip']:.3f}  "
              f"acc {s['acc']:.3f} / reversed {s['acc_reversed']:.3f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"checkpoint": args.checkpoint, "data": args.data, **summary}, fh, indent=2)


if __name__ == "__main__":
    main()
