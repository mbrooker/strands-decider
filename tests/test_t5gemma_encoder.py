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


def _model(base_dir, *, query_pool="mean", state_mask=False):
    cfg = StrandsDeciderConfig(base_model=base_dir, head_type="xpointer", pointer_dim=16,
                               query_pool=query_pool, state_mask=state_mask, max_length=512,
                               use_lora=True, torch_dtype="float32", lora_r=4)
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
    coll = SystemOneCollator(model.tokenizer, CollatorConfig(
        head_type="xpointer", max_length=512, state_mask=model.config.state_mask), train=False)
    return coll(examples)


def _forward(model, b):
    return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                 n_slots=b["n_slots"], opt_idx=b["opt_idx"], labels=b["labels"],
                 state_len=b.get("state_len"))


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


# ---- e1b: the masked state cache ------------------------------------------------------


def test_masks_match_transformers(base_dir):
    """Without a state length, bidirectional_masks is exactly what transformers builds for
    SDPA, in both layer types, on a right-padded batch."""
    from transformers.masking_utils import (
        create_bidirectional_mask,
        create_bidirectional_sliding_window_mask,
    )

    from strands_decider.modeling import bidirectional_masks

    m = _model(base_dir)
    cfg = m.torso.config
    am = torch.ones(2, 20, dtype=torch.long)
    am[1, 13:] = 0
    emb = torch.empty(2, 20, 1)
    ours = bidirectional_masks(am, cfg.sliding_window)
    full = create_bidirectional_mask(config=cfg, inputs_embeds=emb, attention_mask=am)
    sliding = create_bidirectional_sliding_window_mask(config=cfg, inputs_embeds=emb, attention_mask=am)
    assert torch.equal(ours["full_attention"], full)
    assert torch.equal(ours["sliding_attention"], sliding)


def test_state_len_is_the_state_alone(base_dir):
    """The collator's state length is the state rendered and tokenized alone: the boundary
    tokenizes cleanly, so serving (state tokenized on its own) sees training's split."""
    from strands_decider.prompting import render_state

    m = _model(base_dir, state_mask=True)
    exs = _examples()
    b = _batch(m, exs)
    for i, ex in enumerate(exs):
        alone = m.tokenizer(render_state(ex.state), add_special_tokens=True)["input_ids"]
        assert int(b["state_len"][i]) == len(alone)
        assert b["input_ids"][i, : len(alone)].tolist() == alone


def test_state_never_sees_the_question(base_dir):
    """Under the state mask, two prompts with one state and different questions give the
    state's tokens the same encoding, which is what makes caching it exact."""
    m = _model(base_dir, state_mask=True)
    a, b = _examples()[0], _examples()[0]
    b = Example(kind="noul", state=a.state, instructions="Is anyone angry?",
                options=[["false", ""], ["true", ""]], label=0, task="t")
    batch = _batch(m, [a, b])
    n = int(batch["state_len"][0])
    assert n == int(batch["state_len"][1])
    from strands_decider.modeling import bidirectional_masks

    with torch.no_grad():
        masks = bidirectional_masks(batch["attention_mask"], m.torso.config.sliding_window,
                                    batch["state_len"])
        h = m.torso(input_ids=batch["input_ids"], attention_mask=masks).last_hidden_state
    torch.testing.assert_close(h[0, :n], h[1, :n], atol=1e-5, rtol=1e-5)


def test_state_masked_model_trains(base_dir):
    m = _model(base_dir, state_mask=True).train()
    out = _forward(m, _batch(m, _examples()))
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for n, p in m.torso.named_parameters() if "lora_A" in n)


def test_state_cache_is_exact(base_dir):
    """The serving path (state once, questions against its cached keys and values) gives the
    answers of the full masked forward, in fp32."""
    m = _model(base_dir, state_mask=True)
    state = "Help! My payouts have been failing for 3 days! " * 6
    qs = {
        "team": ChoiceQuestion(instructions="Which team should handle this?",
                               criteria={"billing": "payments", "technical": "bugs", "sales": None}),
        "urgent": NoulQuestion(instructions="Does this convey urgency?"),
        "mood": ScoreQuestion(instructions="How frustrated is the writer?",
                              criteria=["calm", "frustrated", "furious", "livid"]),
    }
    cached = SystemOneEngine(m, EngineConfig(device="cpu", use_prefix_cache=True))
    assert cached.cfg.use_prefix_cache and cached._state_cache
    batched = SystemOneEngine(m, EngineConfig(device="cpu", use_prefix_cache=False))
    a, b = cached.ask(state, qs).answers, batched.ask(state, qs).answers
    for name in qs:
        da, db = a[name].model_dump(), b[name].model_dump()
        for key in ("noul", "score", "confidence"):
            if key in da:
                assert da[key] == pytest.approx(db[key], abs=2e-4), (name, key)
        if "probabilities" in da:
            for k in da["probabilities"]:
                assert da["probabilities"][k] == pytest.approx(db["probabilities"][k], abs=2e-4)
    # The attention implementation is restored after the cached call.
    assert m.torso.config._attn_implementation == "sdpa"
