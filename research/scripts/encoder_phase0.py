"""Phase 0 for an encoder-only torso: frozen features with a reasoning slice, and latency.

b1's Phase 0 fitted heads on frozen features of short held-out classification only, the
distribution b1 then won; it lost on reasoning and adequacy, which the probe never
sampled (PREREGISTRATION-b1.md). This probe trains its heads on a mix that includes the
multi-step, generated-document and adequacy training rows, and evaluates every slice
separately:

    heldout    holdout_v5_norule (short tasks never trained on)
    hotpotqa   multistep_v14_eval, HotpotQA rows (never trained on)
    boardgame  multistep_v14_eval, BoardgameQA rows
    gen        generated_v16_eval + generated_v18_eval
    adequacy   adequacy_hs2_eval + adequacy_gen_eval, reported class-balanced too

Every model sees the same rows: a row is used only if its prompt is under --max-chars,
whatever each tokenizer makes of it. The query candidates are the state at the
`<answer>` marker's last token, at position 0 ([CLS] / <bos>), and the mean over the
prompt; option keys are each option line's last token. For T5Gemma 2 (b1's torso) the
decoder's `<bos>` state is also kept, as the encoder-decoder anchor; for a causal torso
(v19's Qwen3.5-2B) only the last token is a query.

Frozen MLM encoders (ModernBERT, Ettin) are known to need fine-tuning before a linear
readout uses their features well, so their figures here are floors.

    PYTHONPATH=src python research/scripts/encoder_phase0.py features --model M --out /path/prefix
    PYTHONPATH=src python research/scripts/encoder_phase0.py fit --feats /path/prefix
    PYTHONPATH=src python research/scripts/encoder_phase0.py latency --model M
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from strands_decider.data.format import Example, load_examples
from strands_decider.prompting import build_prompt

DATA = "data"
TRAIN_MIX = [("train_v5", 16000), ("multistep_v14", 2500), ("generated_v16", 600),
             ("generated_v18", 600), ("adequacy_hs2", 1500), ("adequacy_gen", 800)]


# ---- data ---------------------------------------------------------------------------

def _fits(ex: Example, max_chars: int) -> bool:
    return len(build_prompt(ex.state, ex.to_question())[0]) <= max_chars


def _take(path: str, n: int, seed: int, max_chars: int, task: str | None = None) -> list[Example]:
    rows = [e for e in load_examples([path]) if (task is None or e.task == task)]
    random.Random(seed).shuffle(rows)
    return [e for e in rows if _fits(e, max_chars)][:n]


def train_rows(max_chars: int, scale: float = 1.0) -> list[Example]:
    out: list[Example] = []
    for name, n in TRAIN_MIX:
        out += _take(f"{DATA}/{name}.jsonl", max(1, int(n * scale)), seed=2, max_chars=max_chars)
    return out


def eval_slices(max_chars: int, n_heldout: int) -> dict[str, list[Example]]:
    ms = f"{DATA}/multistep_v14_eval.jsonl"
    return {
        "heldout": _take(f"{DATA}/holdout_v5_norule.jsonl", n_heldout, 1, max_chars),
        "hotpotqa": _take(ms, 10**6, 1, max_chars, task="hotpotqa"),
        "boardgame": _take(ms, 10**6, 1, max_chars, task="boardgame"),
        "gen": _take(f"{DATA}/generated_v16_eval.jsonl", 10**6, 1, max_chars)
        + _take(f"{DATA}/generated_v18_eval.jsonl", 10**6, 1, max_chars),
        "adequacy": _take(f"{DATA}/adequacy_hs2_eval.jsonl", 10**6, 1, max_chars)
        + _take(f"{DATA}/adequacy_gen_eval.jsonl", 10**6, 1, max_chars),
    }


def render(ex: Example, shuffle: random.Random | None) -> tuple[str, Any, int]:
    order = None
    if shuffle is not None and ex.kind != "score":
        order = list(range(ex.n_options))
        shuffle.shuffle(order)
    prompt, rq = build_prompt(ex.state, ex.to_question(), option_order=order)
    return prompt, rq, ex.label if order is None else order.index(ex.label)


# ---- models -------------------------------------------------------------------------

def load(model_id: str, cloze: bool = False, encoder_only: bool = False) -> tuple[Any, Any, str]:
    """(model, tokenizer, kind): kind is "encoder", "encdec" (T5Gemma 2), "causal", or "mlm"
    (an MLM encoder loaded with its head, for the cloze reading). `encoder_only` keeps only
    an encoder-decoder's encoder (T5Gemma 2 and T5Gemma, which load encoder-only anyway)."""
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    cfg = transformers.AutoConfig.from_pretrained(model_id)
    kw: dict[str, Any] = {"dtype": torch.bfloat16}
    if cloze:
        m = transformers.AutoModelForMaskedLM.from_pretrained(model_id, attn_implementation="sdpa", **kw)
        kind = "mlm"
    elif cfg.model_type == "t5gemma2":
        m = transformers.T5Gemma2Model.from_pretrained(model_id, attn_implementation="sdpa", **kw)
        del m.encoder.vision_tower, m.encoder.multi_modal_projector
        kind = "encdec"
        if encoder_only:  # the text encoder alone: b1's torso without its decoder
            m, kind = m.encoder, "encoder"
    elif cfg.model_type == "t5gemma":
        # T5GemmaEncoderModel refuses an encoder-decoder config (transformers 5.17), so load
        # the pair and keep the encoder.
        m = transformers.T5GemmaModel.from_pretrained(model_id, attn_implementation="sdpa", **kw).encoder
        kind = "encoder"
    elif cfg.model_type in {"qwen3_5", "qwen3_5_text"}:
        m = transformers.Qwen3_5ForCausalLM.from_pretrained(model_id, config=cfg.get_text_config(), **kw).model
        kind = "causal"
    else:  # ModernBERT, Ettin
        m = transformers.AutoModel.from_pretrained(model_id, attn_implementation="sdpa", **kw)
        kind = "encoder"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n = sum(p.numel() for p in m.parameters())
    print(f"{model_id}: {kind}, {n / 1e6:.0f}M params")
    return m.cuda().eval(), tok, kind


def forward(model: Any, kind: str, tok: Any, ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(last-layer states over the prompt, decoder query or None)."""
    if kind == "encdec":
        start = torch.full((ids.size(0), 1), model.config.decoder.bos_token_id, device=ids.device)
        out = model(input_ids=ids, attention_mask=mask, decoder_input_ids=start, use_cache=False)
        return out.encoder_last_hidden_state, out.last_hidden_state[:, -1]
    if kind == "mlm":  # the encoder's states; the MLM head is applied at [MASK] only
        return model.model(input_ids=ids, attention_mask=mask).last_hidden_state, None
    return model(input_ids=ids, attention_mask=mask).last_hidden_state, None


def digit_forms(tok: Any) -> list[list[int]]:
    """For k = 1..9, the single-token spellings of k ("1" and " 1"): the token at [MASK]
    may carry the space the mask token absorbed, so both count as answering k."""
    out = []
    for k in range(1, 10):
        forms = [tok.encode(s, add_special_tokens=False) for s in (str(k), f" {k}")]
        out.append(sorted({f[0] for f in forms if len(f) == 1}))
    return out


def encode(tok: Any, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor, list]:
    enc = tok(prompts, add_special_tokens=True, return_offsets_mapping=True, padding=True,
              return_tensors="pt", padding_side="right")
    return enc["input_ids"].cuda(), enc["attention_mask"].cuda(), enc["offset_mapping"].tolist()


def batches(lengths: list[int], max_tokens: int) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, cur, longest = [], [], 0
    for i in order:
        if cur and max(longest, lengths[i]) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, lengths[i])
    return out + ([cur] if cur else [])


def _last_text_token(offsets: list) -> int:
    """Index of the last token carrying prompt text (skips [SEP], <eos> and padding)."""
    return max(j for j, (lo, hi) in enumerate(offsets) if hi > lo)


def _option_last(offsets: list, spans, base: int) -> list[int]:
    out = []
    for s, e in spans:
        a, b = base + s, base + e
        out.append(max(j for j, (lo, hi) in enumerate(offsets) if hi > lo and lo >= a and hi <= b))
    return out


@torch.no_grad()
def extract(model: Any, tok: Any, kind: str, rows: list[Example], shuffle: bool, max_tokens: int) -> list[dict]:
    rng = random.Random(3) if shuffle else None
    rendered = [render(ex, rng) for ex in rows]
    cloze = kind == "mlm"
    texts = [p + " " + tok.mask_token if cloze else p for p, _, _ in rendered]
    digits = digit_forms(tok) if cloze else []
    lengths = [len(tok(t)["input_ids"]) for t in texts]
    feats: list[dict] = [None] * len(rows)  # type: ignore[list-item]
    for b in batches(lengths, max_tokens):
        ids, mask, offs = encode(tok, [texts[i] for i in b])
        h, q_dec = forward(model, kind, tok, ids, mask)
        for j, i in enumerate(b):
            prompt, rq, label = rendered[i]
            hj = h[j]
            n = int(mask[j].sum())
            if cloze:  # [MASK] follows `<answer>`; it carries text offsets, so find it by id
                at = int((ids[j] == tok.mask_token_id).nonzero()[-1])
                answer = at - 1
            else:
                answer = _last_text_token(offs[j])
            f = {"label": label, "kind": rows[i].kind, "task": rows[i].task,
                 "gold": rows[i].options[rows[i].label][0],
                 "q_answer": hj[answer].float().cpu().half(),
                 "keys": hj[_option_last(offs[j], rq.option_spans, len(prompt) - len(rq.text))].float().cpu().half()}
            if cloze:
                f["q_mask"] = hj[at].float().cpu().half()
                k = rows[i].n_options
                if k <= len(digits):  # the untrained cloze readout over options 1..k
                    logits = model.decoder(model.head(hj[at][None]))[0].float()
                    per = torch.stack([torch.logsumexp(logits[d], 0) for d in digits[:k]])
                    f["readout"] = torch.log_softmax(per, 0).cpu()
            if kind != "causal":
                f["q_first"] = hj[0].float().cpu().half()
                f["q_mean"] = hj[:n].float().mean(0).cpu().half()
            if q_dec is not None:
                f["q_dec"] = q_dec[j].float().cpu().half()
            feats[i] = f
    return feats


def cmd_features(a: argparse.Namespace) -> None:
    model, tok, kind = load(a.model, cloze=a.cloze, encoder_only=a.encoder_only)
    t0 = time.time()
    out = {"model": a.model, "kind": kind,
           "train": extract(model, tok, kind, train_rows(a.max_chars, a.train_scale), True, a.max_tokens)}
    for name, rows in eval_slices(a.max_chars, a.n_heldout).items():
        out[name] = extract(model, tok, kind, rows, False, a.max_tokens)
    torch.save(out, a.out)
    sizes = {k: len(v) for k, v in out.items() if isinstance(v, list)}
    print(f"features {sizes} in {time.time() - t0:.0f} s -> {a.out}")


# ---- fit ----------------------------------------------------------------------------

class TwoNormPointer(nn.Module):
    def __init__(self, d: int, dim: int = 256):
        super().__init__()
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k = nn.Linear(d, dim), nn.Linear(d, dim)
        self.scale = dim ** -0.5

    def forward(self, q: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        return (self.k(self.nk(o)) @ self.q(self.nq(q)).unsqueeze(-1)).squeeze(-1) * self.scale


def _stack(feats: list[dict], qk: str):
    width = max(f["keys"].shape[0] for f in feats)
    d = feats[0][qk].shape[-1]
    queries = torch.stack([f[qk] for f in feats]).float()
    keys = torch.zeros(len(feats), width, d)
    for i, f in enumerate(feats):
        keys[i, : f["keys"].shape[0]] = f["keys"].float()
    n = torch.tensor([f["keys"].shape[0] for f in feats])
    y = torch.tensor([f["label"] for f in feats])
    return queries.cuda(), keys.cuda(), n.cuda(), y.cuda()


def _logp(head: nn.Module, queries, keys, n) -> torch.Tensor:
    valid = torch.arange(keys.shape[1], device=keys.device)[None] < n[:, None]
    return F.log_softmax(head(queries, keys).masked_fill(~valid, -1e4), -1)


def _score(feats: list[dict], correct: torch.Tensor) -> tuple[float, float | None]:
    """Accuracy, and class-balanced accuracy for yes/no slices (mean over gold labels)."""
    acc = float(correct.float().mean())
    if all(f["kind"] == "noul" for f in feats):
        by: dict[str, list[float]] = defaultdict(list)
        for f, c in zip(feats, correct.tolist(), strict=True):
            by[f["gold"]].append(c)
        return acc, statistics.mean(sum(v) / len(v) for v in by.values())
    return acc, None


def cmd_fit(a: argparse.Namespace) -> None:
    data = torch.load(a.feats)
    slices = [k for k in ("heldout", "hotpotqa", "boardgame", "gen", "adequacy") if k in data]
    queries = [q for q in ("q_dec", "q_mask", "q_answer", "q_first", "q_mean") if q in data["train"][0]]
    results = {}
    if any("readout" in f for f in data[slices[0]]):
        # The untrained cloze readout, on the rows with at most 9 options (coverage shown).
        row = {}
        for name in slices:
            covered = [f for f in data[name] if "readout" in f]
            if not covered:
                continue
            correct = torch.tensor([int(f["readout"].argmax()) == f["label"] for f in covered])
            acc, bal = _score(covered, correct)
            row[name] = bal if bal is not None else acc
            row[name + "_coverage"] = len(covered) / len(data[name])
        results["untrained_readout"] = row
        cells = "  ".join(f"{s} {row[s]:.3f} ({row[s + '_coverage']:.0%})" for s in slices if s in row)
        print(f"{data['model']:36} {'readout':9} {cells}")
    for qk in queries:
        torch.manual_seed(0)
        Qt, Kt, nt, yt = _stack(data["train"], qk)
        head = TwoNormPointer(Qt.shape[-1]).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        steps = a.epochs * math.ceil(len(yt) / 256)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
        for _ in range(a.epochs):
            perm = torch.randperm(len(yt), device="cuda")
            for s in range(0, len(yt), 256):
                idx = perm[s:s + 256]
                loss = F.nll_loss(_logp(head, Qt[idx], Kt[idx], nt[idx]), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        row = {}
        with torch.no_grad():
            for name in slices:
                Qe, Ke, ne, ye = _stack(data[name], qk)
                acc, bal = _score(data[name], _logp(head, Qe, Ke, ne).argmax(-1) == ye)
                row[name] = bal if bal is not None else acc
                if bal is not None:
                    row[name + "_plain"] = acc
        reasoning = statistics.mean(row[s] for s in slices if s != "heldout")
        row["reasoning_mean"] = reasoning
        results[qk] = row
        cells = "  ".join(f"{s} {row[s]:.3f}" for s in slices)
        print(f"{data['model']:36} {qk:9} {cells}  | reasoning mean {reasoning:.3f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"model": data["model"], "kind": data["kind"], "sizes": {
                k: len(data[k]) for k in ["train", *slices]}, "results": results}, fh, indent=2)


# ---- latency ------------------------------------------------------------------------

@torch.no_grad()
def cmd_latency(a: argparse.Namespace) -> None:
    """Torso forward time, bf16, eager: what the prompt costs before the (tiny) head.

    One question is batch 1 at the state length. Eight questions against one state are
    eight rows of that length: a bidirectional torso re-reads the state for each
    question. For a causal torso that is an upper bound; its prefix cache would read the
    state once (v19's engine measures that path in evaluation/bench_local.py).
    """
    model, tok, kind = load(a.model, encoder_only=a.encoder_only)
    out = []
    for length in (256, 1024, 2048, 4000):
        for nq in (1, 8):
            ids = torch.randint(1000, 30000, (nq, length), device="cuda")
            mask = torch.ones_like(ids)
            for _ in range(3):
                forward(model, kind, tok, ids, mask)
            torch.cuda.synchronize()
            ts = []
            for _ in range(7):
                t0 = time.perf_counter()
                forward(model, kind, tok, ids, mask)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            med = statistics.median(ts)
            out.append({"tokens": length, "questions": nq, "median_ms": med})
            print(f"{a.model:36} {length:5} tok x {nq} q: {med:7.1f} ms")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"model": a.model, "kind": kind, "latency": out}, fh, indent=2)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("features")
    s.add_argument("--model", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-chars", type=int, default=14000)
    s.add_argument("--n-heldout", type=int, default=3000)
    s.add_argument("--max-tokens", type=int, default=16384)
    s.add_argument("--encoder-only", action="store_true", help="T5Gemma 2: the encoder alone, no decoder query")
    s.add_argument("--train-scale", type=float, default=1.0, help="shrink the training mix (smoke tests)")
    s.add_argument("--cloze", action="store_true",
                   help="MLM encoders: end the prompt with [MASK], keep its state as a query and "
                        "score the MLM head's untrained readout over the option numbers there")
    s = sub.add_parser("fit")
    s.add_argument("--feats", required=True)
    s.add_argument("--epochs", type=int, default=8)
    s.add_argument("--out")
    s = sub.add_parser("latency")
    s.add_argument("--model", required=True)
    s.add_argument("--encoder-only", action="store_true", help="T5Gemma 2: time the encoder alone")
    s.add_argument("--out")
    a = p.parse_args()
    {"features": cmd_features, "fit": cmd_fit, "latency": cmd_latency}[a.cmd](a)


if __name__ == "__main__":
    main()
