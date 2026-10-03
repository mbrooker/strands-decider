"""The encoder-decoder torso (T5Gemma 2) on CPU, with a tiny random model.

The weights are random, so these pin plumbing, not quality: the vision tower is dropped
before LoRA attaches, the readout comes out a masked distribution, right padding and
batching change nothing, a checkpoint round-trips, the frozen readout is the LM's own
logits, and serving takes the batched path. The tokenizer is the real one (its ids are
what the prompt format depends on); the tests skip where it is not downloadable, as the
model is gated on the Hub.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from strands_decider.data.collate import CollatorConfig, SystemOneCollator  # noqa: E402
from strands_decider.data.format import Example  # noqa: E402
from strands_decider.infer import EngineConfig, SystemOneEngine, load_engine  # noqa: E402
from strands_decider.modeling import (  # noqa: E402
    CrossPointerHead,
    StrandsDeciderConfig,
    StrandsDeciderModel,
)
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion  # noqa: E402

REPO = "google/t5gemma-2-1b-1b"


def _tokenizer():
    try:
        return transformers.AutoTokenizer.from_pretrained(REPO)
    except Exception as e:  # gated, offline, or no token
        pytest.skip(f"T5Gemma 2 tokenizer unavailable: {type(e).__name__}")


def _text(**kw):
    return dict(vocab_size=262_208, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                query_pre_attn_scalar=16, sliding_window=8,
                layer_types=["sliding_attention", "full_attention"],
                bos_token_id=2, eos_token_id=1, pad_token_id=0, **kw)


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    """A tiny random T5Gemma 2 saved as a Hub-style folder: config, weights, tokenizer."""
    tok = _tokenizer()
    cfg = transformers.T5Gemma2Config(
        encoder=dict(
            text_config=_text(),
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, image_size=28, patch_size=14),
            mm_tokens_per_image=4,
        ),
        decoder=_text(),
    )
    torch.manual_seed(0)
    lm = transformers.T5Gemma2ForConditionalGeneration(cfg)
    path = tmp_path_factory.mktemp("t5g2-tiny")
    lm.save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def _model(base_dir, *, lora=True, dtype="float32"):
    cfg = StrandsDeciderConfig(base_model=base_dir, head_type="xpointer", pointer_dim=16,
                               max_length=512, use_lora=lora, torch_dtype=dtype, lora_r=4)
    torch.manual_seed(1)
    m = StrandsDeciderModel.from_pretrained_base(cfg)
    if lora:
        # LoRA's B starts at zero, so a fresh adapter is a no-op; perturb it so the tests
        # see an adapter that changes the output, as a trained one does.
        with torch.no_grad():
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
                options=[["0", "negative"], ["1", "neutral"], ["2", "positive"], ["3", "glowing"]],
                label=2, task="t"),
    ]


def _batch(model, examples):
    coll = SystemOneCollator(model.tokenizer,
                             CollatorConfig(head_type="xpointer", max_length=512), train=False)
    return coll(examples)


def _forward(model, b):
    return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                 n_slots=b["n_slots"], opt_idx=b["opt_idx"], labels=b["labels"])


def test_vision_tower_dropped_and_lora_on_text_stacks_only(base_dir):
    m = _model(base_dir)
    names = [n for n, _ in m.torso.named_modules()]
    assert not any("vision_tower" in n or "multi_modal_projector" in n for n in names)
    lora = [n for n in names if n.endswith(".lora_A")]
    assert lora and all(".encoder.text_model.layers." in n or ".decoder.layers." in n for n in lora)
    assert any(".encoder." in n for n in lora) and any(".decoder." in n for n in lora)
    assert isinstance(m.head, CrossPointerHead)
    assert StrandsDeciderModel.is_encoder_decoder(m.torso)
    assert m.decoder_start_id() == 2


def test_forward_is_a_masked_distribution_and_trains(base_dir):
    m = _model(base_dir).train()
    b = _batch(m, _examples())
    out = _forward(m, b)
    probs = out["log_probs"].exp()
    for i, n in enumerate(b["n_slots"].tolist()):
        assert probs[i, :n].sum().item() == pytest.approx(1.0, abs=1e-5)
        assert probs[i, n:].sum().item() == 0.0
    out["loss"].backward()
    lora_a = [p for n, p in m.torso.named_parameters() if "lora_A" in n and ".encoder." in n]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora_a)
    assert m.head.norm_k.weight.grad is not None and m.head.norm_q.weight.grad is not None


def test_right_padding_and_batching_change_nothing(base_dir):
    m = _model(base_dir)
    exs = _examples()
    with torch.no_grad():
        together = _forward(m, _batch(m, exs))["log_probs"]
        for i, ex in enumerate(exs):
            alone = _forward(m, _batch(m, [ex]))["log_probs"]
            n = ex.n_options
            torch.testing.assert_close(together[i, :n], alone[0, :n], atol=1e-4, rtol=1e-4)


def _request_questions():
    return {
        "team": ChoiceQuestion(instructions="Which team should handle this?",
                               criteria={"billing": "payments", "technical": "bugs", "sales": None}),
        "urgent": NoulQuestion(instructions="Does this convey urgency?"),
        "mood": ScoreQuestion(instructions="How frustrated is the writer?",
                              criteria=["calm", "frustrated", "furious"]),
    }


def test_engine_batches_questions_without_a_prefix_cache(base_dir):
    m = _model(base_dir)
    eng = SystemOneEngine(m, EngineConfig(device="cpu", use_prefix_cache=True))
    assert eng.cfg.use_prefix_cache is False
    state = "Help! My payouts have been failing for 3 days!"
    qs = _request_questions()
    together = eng.ask(state, qs).answers
    for name, q in qs.items():
        alone = eng.ask(state, {name: q}).answers[name]
        a, b = together[name].model_dump(), alone.model_dump()
        for key in ("noul", "score", "confidence"):
            if key in a:
                assert a[key] == pytest.approx(b[key], abs=2e-4)


def test_checkpoint_round_trip(base_dir, tmp_path):
    m = _model(base_dir)
    state, qs = "Payouts failing.", _request_questions()
    before = SystemOneEngine(m, EngineConfig(device="cpu")).ask(state, qs).answers
    m.save_pretrained(str(tmp_path / "ckpt"))
    again = StrandsDeciderModel.load(str(tmp_path / "ckpt"))
    after = SystemOneEngine(again, EngineConfig(device="cpu")).ask(state, qs).answers
    assert after == before


def test_frozen_readout_is_the_lm_logits(base_dir):
    """With the adapter disabled, the frozen readout is the pretrained model's next-token
    distribution over "1".."k" at the decoder's first position: the tied embedding is the
    LM head."""
    m = _model(base_dir)
    exs = _examples()
    b = _batch(m, exs)
    lp, eligible = m.frozen_slot_log_probs(b["input_ids"], b["attention_mask"], b["n_slots"])
    assert eligible.all()
    lm = transformers.T5Gemma2ForConditionalGeneration.from_pretrained(base_dir, dtype=torch.float32).eval()
    start = torch.full((len(exs), 1), 2)
    with torch.no_grad():
        logits = lm(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    decoder_input_ids=start).logits[:, -1]
    digits = [m.tokenizer.encode(str(k), add_special_tokens=False)[0] for k in range(1, 10)]
    for i, ex in enumerate(exs):
        n = ex.n_options
        want = torch.log_softmax(logits[i, digits[:n]], -1)
        torch.testing.assert_close(lp[i, :n], want, atol=1e-4, rtol=1e-4)


def _lora_grads(base_dir, *, compiled, checkpointing):
    m = _model(base_dir).to("cuda").train()  # lora_dropout 0.05, as trained
    if checkpointing:
        m.torso.base_model.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    if compiled:
        m.compile_layers()
    b = {k: v.to("cuda") for k, v in _batch(m, _examples()).items()}
    torch.manual_seed(123)
    _forward(m, b)["loss"].backward()
    return {n: p.grad.detach().clone() for n, p in m.torso.named_parameters() if "lora_" in n}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_checkpointing_replays_the_lora_dropout_masks(base_dir, compiled):
    """Checkpointing recomputes each layer in backward. If the recompute drew new adapter
    dropout masks, the gradient would belong to neither forward, silently. With the RNG
    replayed, the gradient equals the one without checkpointing."""
    with_ckpt = _lora_grads(base_dir, compiled=compiled, checkpointing=True)
    without = _lora_grads(base_dir, compiled=compiled, checkpointing=False)
    assert with_ckpt.keys() == without.keys()
    for n in with_ckpt:
        torch.testing.assert_close(with_ckpt[n], without[n], atol=1e-5, rtol=1e-4, msg=n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compiled_layers_compute_the_same_function(base_dir):
    m = _model(base_dir).to("cuda").eval()
    b = {k: v.to("cuda") for k, v in _batch(m, _examples()).items()}
    with torch.no_grad():
        eager = _forward(m, b)["log_probs"]
        m.compile_layers()
        compiled = _forward(m, b)["log_probs"]
    finite = torch.isfinite(eager)
    torch.testing.assert_close(compiled[finite], eager[finite], atol=1e-4, rtol=1e-4)


def test_mlx_refuses_this_torso(base_dir, tmp_path, monkeypatch):
    import strands_decider.infer as infer

    _model(base_dir, lora=False).save_pretrained(str(tmp_path / "ckpt"))
    monkeypatch.setattr(infer, "mlx_available", lambda: True)  # reach the torso check anywhere
    with pytest.raises(RuntimeError, match="encoder-decoder"):
        load_engine(str(tmp_path / "ckpt"), device="mlx")
