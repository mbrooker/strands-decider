"""Phase 0 probes for the T5Gemma 2 torso (docs/bidi-design.md#phase-0-probes-before-any-training).

Nothing here trains the torso. Each subcommand answers one design question:

    smoke     load, drop the vision tower, attach LoRA, fwd+bwd memory and time per shape,
              and right-padding invariance of the bidirectional encoder
    tokens    BOS/EOS behaviour and whether the option numbers 1..9 are single tokens
    readout   the untrained model's own answer: softmax over "1".."k" at the decoder's
              first position (T5Gemma 2) or the last prompt token (a causal torso)
    features  frozen hidden states for head-only fits: query candidates and pooled options
    fit       fit a two-norm pointer head on frozen features, report held-out accuracy

    PYTHONPATH=src python research/scripts/bidi_phase0.py readout --model google/t5gemma-2-1b-1b
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from strands_decider.data.format import Example, load_examples
from strands_decider.prompting import build_prompt

T5 = "google/t5gemma-2-1b-1b"
DATA = "/home/marc/hobson-gemma4/data"


# ---- data ---------------------------------------------------------------------------

def sample(path: str, n: int, seed: int, max_options: int = 99) -> list[Example]:
    rows = [e for e in load_examples([path]) if e.n_options <= max_options]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def render(ex: Example, shuffle: random.Random | None) -> tuple[str, Any, int, list[int] | None]:
    """Prompt, rendered question, label under this rendering, and the option order."""
    order = None
    if shuffle is not None and ex.kind != "score":
        order = list(range(ex.n_options))
        shuffle.shuffle(order)
    prompt, rq = build_prompt(ex.state, ex.to_question(), option_order=order)
    label = ex.label if order is None else order.index(ex.label)
    return prompt, rq, label, order


def option_token_spans(offsets: list[tuple[int, int]], spans, base: int) -> list[tuple[int, int]]:
    """(first, last) token index of each option line; character spans shifted by `base`."""
    out = []
    for s, e in spans:
        a, b = base + s, base + e
        idx = [j for j, (lo, hi) in enumerate(offsets) if hi > lo and lo >= a and hi <= b]
        if not idx:
            raise ValueError("option has no tokens")
        out.append((idx[0], idx[-1]))
    return out


# ---- models -------------------------------------------------------------------------

def is_t5(model_id: str) -> bool:
    return "t5gemma" in model_id.lower()


def load(model_id: str, *, lm: bool) -> tuple[Any, Any]:
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    kw: dict[str, Any] = {"dtype": torch.bfloat16}
    if is_t5(model_id):
        cls = transformers.T5Gemma2ForConditionalGeneration if lm else transformers.T5Gemma2Model
        model = cls.from_pretrained(model_id, attn_implementation="sdpa", **kw)
        core = model.model if lm else model
        enc = core.encoder
        n_vis = sum(p.numel() for p in enc.vision_tower.parameters()) + sum(
            p.numel() for p in enc.multi_modal_projector.parameters())
        del enc.vision_tower, enc.multi_modal_projector
        print(f"dropped vision tower + projector: {n_vis / 1e6:.0f}M params")
    else:
        cfg = transformers.AutoConfig.from_pretrained(model_id)
        if cfg.model_type in {"qwen3_5", "qwen3_5_text"}:
            m = transformers.Qwen3_5ForCausalLM.from_pretrained(model_id, config=cfg.get_text_config(), **kw)
        else:
            m = transformers.AutoModelForCausalLM.from_pretrained(model_id, **kw)
        model = m if lm else m.model
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n = sum(p.numel() for p in model.parameters())
    print(f"{model_id}: {n / 1e6:.0f}M params loaded")
    return model.cuda().eval(), tok


def encode_batch(tok, prompts: list[str], *, add_eos: bool) -> tuple[torch.Tensor, torch.Tensor, list]:
    enc = tok(prompts, add_special_tokens=True, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    if add_eos:
        ids = [x + [tok.eos_token_id] for x in ids]
        offs = [o + [(0, 0)] for o in offs]
    width = max(map(len, ids))
    pad = tok.pad_token_id
    t = torch.tensor([x + [pad] * (width - len(x)) for x in ids])
    m = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids])
    return t.cuda(), m.cuda(), offs


def batches(items: list, max_tokens: int, length) -> list[list[int]]:
    order = sorted(range(len(items)), key=lambda i: -length(items[i]))
    out, cur, longest = [], [], 0
    for i in order:
        L = length(items[i])
        if cur and max(longest, L) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, L)
    if cur:
        out.append(cur)
    return out


def digit_ids(tok) -> list[int]:
    ids = []
    for d in "123456789":
        e = tok.encode(d, add_special_tokens=False)
        assert len(e) == 1 and tok.decode(e) == d, (d, e)
        ids.append(e[0])
    return ids


# ---- smoke --------------------------------------------------------------------------

def cmd_smoke(a: argparse.Namespace) -> None:
    from peft import LoraConfig, get_peft_model

    model, tok = load(T5, lm=False)
    cfg = model.config
    dec = cfg.decoder
    etc = cfg.encoder.text_config
    print(json.dumps({
        "hidden": etc.hidden_size, "enc_layers": etc.num_hidden_layers, "dec_layers": dec.num_hidden_layers,
        "heads": etc.num_attention_heads, "kv_heads": etc.num_key_value_heads, "head_dim": etc.head_dim,
        "sliding_window": etc.sliding_window, "layer_types_enc": etc.layer_types,
        "final_logit_softcapping": dec.final_logit_softcapping, "attn_softcap": etc.attn_logit_softcapping,
        "dropout_rate": getattr(etc, "dropout_rate", None), "dec_dropout": getattr(dec, "dropout_rate", None),
        "bos": dec.bos_token_id, "eos": dec.eos_token_id, "pad": dec.pad_token_id,
        "tie": cfg.tie_word_embeddings,
    }, indent=1, default=str))

    # Padding invariance: one prompt alone vs. right-padded beside a longer one.
    short = "<state>\nThe invoice was paid twice.\n</state>\n<answer>"
    long = "<state>\n" + "Lorem ipsum dolor sit amet. " * 60 + "\n</state>\n<answer>"
    with torch.no_grad():
        ids1, m1, _ = encode_batch(tok, [short], add_eos=False)
        ids2, m2, _ = encode_batch(tok, [short, long], add_eos=False)
        bos = torch.full((1, 1), dec.bos_token_id, device="cuda")
        h1 = model(input_ids=ids1, attention_mask=m1, decoder_input_ids=bos)
        h2 = model(input_ids=ids2, attention_mask=m2, decoder_input_ids=bos.expand(2, 1))
        n = ids1.shape[1]
        de = (h1.encoder_last_hidden_state[0] - h2.encoder_last_hidden_state[0, :n]).abs().max().item()
        dd = (h1.last_hidden_state[0] - h2.last_hidden_state[0]).abs().max().item()
        scale = h1.encoder_last_hidden_state.abs().mean().item()
        print(f"padding invariance (bf16): encoder max|diff| {de:.4f}, decoder {dd:.4f} "
              f"(mean |h| {scale:.3f}); padded to {ids2.shape[1]} from {n}")

    targets = r".*(encoder\.text_model|decoder)\.layers\.\d+\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=targets, task_type="FEATURE_EXTRACTION"))
    names = [n for n, _ in model.named_modules() if n.endswith(".lora_A")]
    where = defaultdict(int)
    for nm in names:
        where["encoder" if ".encoder." in nm else "decoder" if ".decoder." in nm else "OTHER"] += 1
    print(f"LoRA modules: {dict(where)}; trainable {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
    model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for L in a.lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(1000, 200000, (a.batch, L), device="cuda")
        mask = torch.ones_like(ids)
        bos = torch.full((a.batch, 1), dec.bos_token_id, device="cuda")
        times = []
        try:
            for _ in range(3):
                torch.cuda.synchronize()
                t0 = time.time()
                out = model(input_ids=ids, attention_mask=mask, decoder_input_ids=bos)
                loss = out.last_hidden_state.float().pow(2).mean() + out.encoder_last_hidden_state[:, :64].float().pow(2).mean()
                loss.backward()
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            model.zero_grad(set_to_none=True)
            print(f"batch {a.batch} x {L}: fwd+bwd {min(times):.3f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        except torch.OutOfMemoryError:
            print(f"batch {a.batch} x {L}: OOM")
            model.zero_grad(set_to_none=True)


# ---- tokens -------------------------------------------------------------------------

def cmd_tokens(a: argparse.Namespace) -> None:
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    ids = tok("hello world")["input_ids"]
    print("special:", {k: getattr(tok, k) for k in ("bos_token", "eos_token", "pad_token", "unk_token")})
    print("'hello world' ->", ids, tok.convert_ids_to_tokens(ids))
    for d in ["1", "9", "10", "<answer>", "\n1. billing"]:
        e = tok.encode(d, add_special_tokens=False)
        print(f"{d!r:16} -> {e} {tok.convert_ids_to_tokens(e)}")


# ---- readout ------------------------------------------------------------------------

def cmd_readout(a: argparse.Namespace) -> None:
    model, tok = load(a.model, lm=True)
    rows = sample(a.data, a.n, seed=1, max_options=9)
    digits = torch.tensor(digit_ids(tok), device="cuda")
    rendered = [render(ex, None) for ex in rows]
    t5 = is_t5(a.model)
    variants = [("bos", []), ("bos+Answer:", tok.encode("Answer:", add_special_tokens=False))] if t5 else [("last", None)]
    for eos in ([False, True] if t5 else [False]):
        for vname, lead in variants:
            stats: dict[str, list] = defaultdict(list)
            bl = batches(rendered, a.max_tokens, lambda r: len(r[0]) // 3 + 8)
            with torch.no_grad():
                for b in bl:
                    ids, mask, _ = encode_batch(tok, [rendered[i][0] for i in b], add_eos=eos)
                    if t5:
                        dec = torch.tensor([[tok.bos_token_id] + lead] * len(b), device="cuda")
                        logits = model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec).logits[:, -1]
                    else:
                        out = model(input_ids=ids, attention_mask=mask).logits
                        last = mask.sum(1) - 1
                        logits = out[torch.arange(len(b)), last]
                    sel = logits.float()[:, digits]
                    for j, i in enumerate(b):
                        ex = rows[i]
                        lp = F.log_softmax(sel[j, : ex.n_options], -1)
                        lab = rendered[i][2]
                        for key in ("all", ex.kind, f"task:{ex.task}"):
                            stats[key].append((int(lp.argmax() == lab), -lp[lab].item(),
                                               1.0 / ex.n_options))
            name = f"{a.model} enc_eos={eos} dec={vname}"
            print(f"\n== {name}")
            for key in sorted(stats, key=lambda k: (k.startswith("task:"), k)):
                v = stats[key]
                acc = sum(x[0] for x in v) / len(v)
                nll = sum(x[1] for x in v) / len(v)
                chance = sum(x[2] for x in v) / len(v)
                print(f"  {key:40} n={len(v):5} acc {acc:.3f} (chance {chance:.3f}) nll {nll:.3f}")


# ---- features -----------------------------------------------------------------------

def cmd_features(a: argparse.Namespace) -> None:
    model, tok = load(a.model, lm=False)
    t5 = is_t5(a.model)
    for split, path, n, seed in (("train", a.train, a.n_train, 2), ("eval", a.data, a.n_eval, 1)):
        rows = sample(path, n, seed=seed, max_options=24)
        rng = random.Random(seed) if split == "train" else None
        rendered = [render(ex, rng) for ex in rows]
        feats: list[dict] = [None] * len(rows)  # type: ignore[list-item]
        bl = batches(rendered, a.max_tokens, lambda r: len(r[0]) // 3 + 8)
        t0 = time.time()
        with torch.no_grad():
            for b in bl:
                prompts = [rendered[i][0] for i in b]
                ids, mask, offs = encode_batch(tok, prompts, add_eos=a.eos)
                if t5:
                    dec = torch.full((len(b), 1), tok.bos_token_id, device="cuda")
                    out = model(input_ids=ids, attention_mask=mask, decoder_input_ids=dec)
                    h, q_dec = out.encoder_last_hidden_state, out.last_hidden_state[:, -1]
                else:
                    h = model(input_ids=ids, attention_mask=mask).last_hidden_state
                    q_dec = None
                lengths = mask.sum(1)
                for j, i in enumerate(b):
                    prompt, rq, label, _ = rendered[i]
                    spans = option_token_spans(offs[j], rq.option_spans, len(prompt) - len(rq.text))
                    ans = int(lengths[j]) - 1 - (1 if a.eos else 0)  # the `<answer>` marker's last token
                    hj = h[j]
                    f = {
                        "q_answer": hj[ans].half().cpu(),
                        "opt_last": torch.stack([hj[e] for _, e in spans]).half().cpu(),
                        "opt_mean": torch.stack([hj[s:e + 1].mean(0) for s, e in spans]).half().cpu(),
                        "label": label, "kind": rows[i].kind, "task": rows[i].task,
                    }
                    if q_dec is not None:
                        f["q_dec"] = q_dec[j].half().cpu()
                    feats[i] = f
        out_path = f"{a.out}.{split}.pt"
        torch.save(feats, out_path)
        print(f"{split}: {len(feats)} rows in {time.time() - t0:.0f} s -> {out_path}")


# ---- fit ----------------------------------------------------------------------------

class TwoNormPointer(nn.Module):
    def __init__(self, d: int, dim: int = 256):
        super().__init__()
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k = nn.Linear(d, dim), nn.Linear(d, dim)
        self.scale = dim ** -0.5

    def forward(self, q: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        return (self.k(self.nk(o)) @ self.q(self.nq(q)).unsqueeze(-1)).squeeze(-1) * self.scale


def _stack(feats: list[dict], qk: str, ok: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    K = max(f[ok].shape[0] for f in feats)
    d = feats[0][qk].shape[-1]
    Q = torch.stack([f[qk] for f in feats]).float()
    O = torch.zeros(len(feats), K, d)
    n = torch.tensor([f[ok].shape[0] for f in feats])
    for i, f in enumerate(feats):
        O[i, : f[ok].shape[0]] = f[ok].float()
    y = torch.tensor([f["label"] for f in feats])
    return Q.cuda(), O.cuda(), n.cuda(), y.cuda()


def _logp(head: nn.Module, Q, O, n) -> torch.Tensor:
    logits = head(Q, O)
    valid = torch.arange(O.shape[1], device=O.device)[None] < n[:, None]
    return F.log_softmax(logits.masked_fill(~valid, -1e4), -1)


def cmd_fit(a: argparse.Namespace) -> None:
    tr, ev = torch.load(f"{a.feats}.train.pt"), torch.load(f"{a.feats}.eval.pt")
    combos = [(q, o) for q in ("q_dec", "q_answer") for o in ("opt_mean", "opt_last") if q in tr[0]]
    for qk, ok in combos:
        torch.manual_seed(0)
        Qt, Ot, nt, yt = _stack(tr, qk, ok)
        Qe, Oe, ne, ye = _stack(ev, qk, ok)
        head = TwoNormPointer(Qt.shape[-1]).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
        steps = a.epochs * math.ceil(len(yt) / 256)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.05)
        for _ in range(a.epochs):
            perm = torch.randperm(len(yt), device="cuda")
            for s in range(0, len(yt), 256):
                idx = perm[s:s + 256]
                loss = F.nll_loss(_logp(head, Qt[idx], Ot[idx], nt[idx]), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        with torch.no_grad():
            lp = _logp(head, Qe, Oe, ne)
            acc = (lp.argmax(-1) == ye).float()
            nll = -lp[torch.arange(len(ye)), ye]
        by: dict[str, list] = defaultdict(list)
        for i, f in enumerate(ev):
            by[f["kind"]].append(i)
        parts = " ".join(f"{k} {acc[v].mean():.3f}" for k, v in sorted(by.items()))
        print(f"{a.feats} {qk:9} x {ok:8}: acc {acc.mean():.3f} nll {nll.mean():.3f} | {parts}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--lengths", type=int, nargs="+", default=[256, 1024, 2048, 3072, 4096])
    s = sub.add_parser("tokens")
    s.add_argument("--model", default=T5)
    s = sub.add_parser("readout")
    s.add_argument("--model", default=T5)
    s.add_argument("--data", default=f"{DATA}/holdout_v5_norule.jsonl")
    s.add_argument("--n", type=int, default=3000)
    s.add_argument("--max-tokens", type=int, default=16384)
    s = sub.add_parser("features")
    s.add_argument("--model", default=T5)
    s.add_argument("--train", default=f"{DATA}/train_v5.jsonl")
    s.add_argument("--data", default=f"{DATA}/holdout_v5_norule.jsonl")
    s.add_argument("--n-train", type=int, default=20000)
    s.add_argument("--n-eval", type=int, default=3000)
    s.add_argument("--eos", action="store_true")
    s.add_argument("--max-tokens", type=int, default=16384)
    s.add_argument("--out", required=True)
    s = sub.add_parser("fit")
    s.add_argument("--feats", required=True)
    s.add_argument("--epochs", type=int, default=8)
    a = p.parse_args()
    {"smoke": cmd_smoke, "tokens": cmd_tokens, "readout": cmd_readout,
     "features": cmd_features, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    main()
