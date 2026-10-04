"""An encoder-only torso (T5Gemma's encoder, e1) on CPU, with a tiny random model.

Pins the plumbing of docs/encoder-design.md: the decoder is dropped and LoRA reaches only the
encoder, the torso counts as bidirectional so serving never uses a prefix cache, the mean
query is the masked mean, right padding and batching change nothing, a checkpoint
round-trips, and there is no LM readout to anchor to. The tokenizer is the real one; the
tests skip where it is not downloadable (the model is gated on the Hub).
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from strands_decider.data.collate import CollatorConfig, SystemOneCollator  # noqa: E402
from strands_decider.data.format import Example  # noqa: E402
from strands_decider.infer import EngineConfig, SystemOneEngine  # noqa: E402
from strands_decider.modeling import (  # noqa: E402
    StrandsDeciderConfig,
    StrandsDeciderModel,
    mean_pool,
)
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion  # noqa: E402

REPO = "google/t5gemma-2b-2b-ul2-it"


def _module(**kw):
    return dict(vocab_size=256_000, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16, query_pre_attn_scalar=16,
                sliding_window=8, layer_types=["sliding_attention", "full_attention"],
                attn_logit_softcapping=50.0, final_logit_softcapping=30.0,
                pad_token_id=0, eos_token_id=1, bos_token_id=2, **kw)


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    try:
        tok = transformers.AutoTokenizer.from_pretrained(REPO)
    except Exception as e:  # gated, offline, or no token
        pytest.skip(f"T5Gemma tokenizer unavailable: {type(e).__name__}")
    cfg = transformers.T5GemmaConfig(encoder=_module(), decoder=_module(is_decoder=True))
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("t5gemma-tiny")
    transformers.T5GemmaModel(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def _model(base_dir, *, query_pool="mean"):
    cfg = StrandsDeciderConfig(base_model=base_dir, head_type="xpointer", pointer_dim=16,
                               query_pool=query_pool, max_length=512, use_lora=True,
                               torch_dtype="float32", lora_r=4)
    torch.manual_seed(1)
    m = StrandsDeciderModel.from_pretrained_base(cfg)
    with torch.no_grad():  # a fresh adapter is a no-op; perturb it, as training would
        for n, p in m.torso.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    return m.eval()


def _examples():
    return [
        Example(kind="choice", state="Help! My payouts have been failing for 3 days.",
                instructions="Which team should handle this?",
                options=[["billing", "payments"], ["technical", "bugs"], ["sales", "pricing"]],
                label=0, task="t"),
        Example(kind="noul", state="The meeting moved to Thursday. " * 12,
                instructions="Is the meeting on Thursday?",
                options=[["false", ""], ["true", ""]], label=1, task="t"),
        Example(kind="score", state="ok", instructions="How positive?",
                options=[["0", "negative"], ["1", "neutral"], ["2", "positive"]], label=2, task="t"),
    ]


def _batch(model, examples):
    coll = SystemOneCollator(model.tokenizer, CollatorConfig(head_type="xpointer", max_length=512),
                             train=False)
    return coll(examples)


def _forward(model, b):
    return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                 n_slots=b["n_slots"], opt_idx=b["opt_idx"], labels=b["labels"])


def test_encoder_only_torso_with_lora_on_the_encoder(base_dir):
    m = _model(base_dir)
    names = [n for n, _ in m.torso.named_modules()]
    assert not any("decoder" in n or "cross_attn" in n for n in names)
    assert any(n.endswith(".lora_A") for n in names)
    assert StrandsDeciderModel.is_bidirectional(m.torso)
    assert not StrandsDeciderModel.is_encoder_decoder(m.torso)


def test_mean_query_is_the_masked_mean():
    h = torch.randn(2, 5, 3)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    out = mean_pool(h, mask)
    torch.testing.assert_close(out[0], h[0, :3].mean(0))
    torch.testing.assert_close(out[1], h[1].mean(0))


def test_forward_is_a_masked_distribution_and_trains(base_dir):
    m = _model(base_dir).train()
    b = _batch(m, _examples())
    out = _forward(m, b)
    probs = out["log_probs"].exp()
    for i, n in enumerate(b["n_slots"].tolist()):
        assert probs[i, :n].sum().item() == pytest.approx(1.0, abs=1e-5)
        assert probs[i, n:].sum().item() == 0.0
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in m.torso.named_parameters() if "lora_A" in n)


@pytest.mark.parametrize("query_pool", ["mean", "last"])
def test_right_padding_and_batching_change_nothing(base_dir, query_pool):
    m = _model(base_dir, query_pool=query_pool)
    exs = _examples()
    with torch.no_grad():
        together = _forward(m, _batch(m, exs))["log_probs"]
        for i, ex in enumerate(exs):
            alone = _forward(m, _batch(m, [ex]))["log_probs"]
            n = ex.n_options
            torch.testing.assert_close(together[i, :n], alone[0, :n], atol=1e-4, rtol=1e-4)


def test_engine_never_takes_the_prefix_path(base_dir):
    m = _model(base_dir)
    eng = SystemOneEngine(m, EngineConfig(device="cpu", use_prefix_cache=True))
    assert eng.cfg.use_prefix_cache is False
    state = "Help! My payouts have been failing for 3 days!"
    qs = {
        "team": ChoiceQuestion(instructions="Which team?", criteria={"billing": None, "sales": None}),
        "urgent": NoulQuestion(instructions="Does this convey urgency?"),
        "mood": ScoreQuestion(instructions="How frustrated?", criteria=["calm", "frustrated", "furious"]),
    }
    together = eng.ask(state, qs).answers
    eng.cfg = replace(eng.cfg, use_prefix_cache=True)  # as evaluation/bench_local.py does
    assert eng.ask(state, qs).answers == together


def test_checkpoint_round_trip(base_dir, tmp_path):
    m = _model(base_dir)
    state = "Payouts failing."
    qs = {"urgent": NoulQuestion(instructions="Does this convey urgency?")}
    before = SystemOneEngine(m, EngineConfig(device="cpu")).ask(state, qs).answers
    m.save_pretrained(str(tmp_path / "ckpt"))
    again = StrandsDeciderModel.load(str(tmp_path / "ckpt"))
    assert again.config.query_pool == "mean"
    assert SystemOneEngine(again, EngineConfig(device="cpu")).ask(state, qs).answers == before


def test_no_lm_readout_to_anchor_to(base_dir):
    m = _model(base_dir)
    b = _batch(m, _examples())
    with pytest.raises(ValueError, match="kl_frozen_weight"):
        m.frozen_slot_log_probs(b["input_ids"], b["attention_mask"], b["n_slots"])
