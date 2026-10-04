# hobson-bidi: design for a T5Gemma 2 (1B+1B) torso

This fork replaces the decoder-only torso of strands-decider (Qwen3.5-2B-Base in v19,
Gemma 4 E2B in the `hobson-gemma4` fork's g4) with the encoder-decoder
`google/t5gemma-2-1b-1b`. This document designs the training and inference processes,
and was written before the code. Figures marked *estimate* are projections, not
measurements.

> **Status, 4 October 2026: b1 has run and failed its preregistration.** JevBench 153/231
> (v19 168, g4 183). Order sensitivity fell to 0.035 from v19's 0.088, and JevBench ECE
> (0.044) was the best recorded. Reasoning, adequacy and latency regressed. See
> [The first run, b1](#the-first-run-b1) and
> [PREREGISTRATION-b1.md](../research/preregistrations/PREREGISTRATION-b1.md). Several
> estimates below were wrong, and the sections below say which.

## Why an encoder-decoder

Upstream names two limits of a causal torso in [architecture.md](architecture.md):

- "a bidirectional encoder would pool better in principle", and was rejected only
  because converting a causal LLM to bidirectional attention needs retraining;
- option order still moves the distribution by about 0.016, because each option
  attends to the options before it but not to the ones after.

T5Gemma 2 has a pretrained bidirectional encoder, so neither limit requires
retraining the torso. The fork tests whether that is worth the cost to multi-question
serving that it brings (see [Inference](#inference)).

## The torso

| | `google/t5gemma-2-1b-1b` |
| --- | --- |
| Parameters | 2.116B in the checkpoint: ~0.30B shared embedding (262k vocabulary x 1152), ~0.7B encoder, ~0.7B decoder, ~0.4B SigLIP vision tower (*estimate* of the split from Gemma 3 1B's shapes) |
| Layers | 26 encoder, 26 decoder; every 6th layer global attention, others local (sliding window) |
| Encoder attention | bidirectional; local layers see a symmetric window |
| Decoder attention | "merged": one attention over [decoder self-attention keys (causal); encoder outputs], with the self-attention weights reused for cross-attention |
| Embeddings | input embeddings are shared by encoder and decoder and tied to the LM head |
| Licence | Gemma Terms of Use, gated on Hugging Face (see [Licence](#licence)) |
| transformers | `T5Gemma2Model` / `T5Gemma2ForConditionalGeneration`, present in the 5.17.0 already installed in WSL |

Implementation facts from `transformers/models/t5gemma2/modeling_t5gemma2.py` (5.17.0)
that the design depends on:

- `_supports_sdpa = True`, `_supports_flash_attn = False`. The torso has no Gated
  DeltaNet layers, so `flash-linear-attention` is not needed. Native Windows would work
  too, but we train in WSL2 as the upstream recipe does.
- The encoder and the decoder each apply a final RMSNorm, so both `last_hidden_state`s
  are normalised.
- `final_logit_softcapping` exists on the decoder config but is `None` in this
  checkpoint (Phase 0, from `config.json`), so the frozen readout is a plain dot product
  with the tied embedding. The code still applies the cap when a config sets one, as
  the Gemma 4 fork learned to (`teacher: read a soft-capped LM's letters through its cap`).
- From `config.json`: hidden 1152, 26 layers per stack, 4 query heads and 1 KV head of
  dim 256, MLP 6912, sliding window 512, `dropout_rate` 0.0.
- The tokenizer prepends `<bos>` and appends no `<eos>`. `1`…`9` are single tokens and
  `10` is two, so the option-number readout covers up to 9 options, as upstream's does.
- The encoder builds its vision tower and projector whether or not images are used.
  SigLIP's attention modules are also named `q_proj`/`k_proj`/`v_proj`, so a LoRA
  `target_modules` list of bare names would wrap the vision tower too. We delete the
  tower and projector after loading, before LoRA attaches.

## Architecture: how to read an encoder-decoder

```
            <bos> <state>…</state> <question>…<options> 1. a … 2. b … </options> … <answer>
                                          │
                              ┌───────────┴───────────┐
                              │  encoder (bidirectional)  │  LoRA
                              └───────────┬───────────┘
                     h_enc: every token has read the whole prompt
                    ┌──────────┬──────────┼───────────────────────┐
              pool(option 1) pool(option 2) …                    cross-attention
                    │          │                                  │
                    │          │               decoder input: <bos>   (one token)
                    │          │                     ┌────────────┴───────────┐
                    │          │                     │  decoder (merged attn) │  LoRA
                    │          │                     └────────────┬───────────┘
                    │          │                            h_dec[<bos>]  = query
                    ▼          ▼                                  ▼
               k(·), LayerNorm_k                          q(·), LayerNorm_q
                    └──────────────── logit_k = <q, k_k> / sqrt(256) ────────┘
```

**Decision: one joint encoder pass, a one-token decoder query.** The whole prompt
(state, question, options, `<answer>`) goes through the encoder. The decoder sees only
`<bos>`, so its hidden state is what the pretrained model holds just before it would
emit the answer. That state is the pointer query. The keys are pooled from encoder
states over each option's line. The prompt text is unchanged from upstream
(`prompting.py`).

Why this design:

- Every option key has read the state, the question and **every** option, in both
  directions. This is the property the fork exists to test.
- The decoder costs one position per question. Its cross-attention projections over
  the encoder outputs add a small fraction of the encoder's cost (*estimate*: under 5%).
- The decoder at `<bos>` is the position whose LM logits are the untrained model's own
  answer. That keeps the frozen-readout probe and an optional frozen-KL anchor
  available, with the same meaning they have upstream: softmax over the option-number
  tokens `1`…`9`, read through the tied embedding (and the soft cap, if one is set).

**Option keys: the last token of the option's line.** Under bidirectional attention no
token is privileged, so the design first defaulted to a mean over the line. Phase 0's
frozen-feature fits reversed that: the last token beat the mean by 0.04 to 0.05
([Phase 0 results](#phase-0-results-3-october-2026)). The mean stays as a config option.

**The head gets two norms.** Upstream's `PointerHead` applies one `LayerNorm` to both
the query and the keys. Here they come from different stacks (decoder and encoder) with
different statistics, so the new head (`head_type: "xpointer"`) has `norm_q` and
`norm_k`. The rest is unchanged: `dim=256`, fp32, no per-option parameters,
`masked_log_softmax`, and the three primitives read back as in `schema.py`.

Alternatives considered and not chosen for the first run:

| Design | Why not first |
| --- | --- |
| **Encoder only** (drop the decoder, query from the encoder's `<answer>` state) | Halves the parameters and is the cheapest to serve, but discards the decoder and with it the untrained-readout probe and the frozen-KL anchor. Kept as a Phase 0 frozen-feature arm; promote it if it matches the joint design there. |
| **Split: state in the encoder, question in the decoder** | Makes the state encoding independent of the question, so it can be shared across questions exactly, which upstream's prefix cache gets from causality. But the state is then read without knowing the question, and options are read causally in the decoder, so the order-sensitivity limit comes back. This is the natural second experiment if multi-question throughput matters more than accuracy. |
| **Full seq2seq** (decoder generates the option number) | Generation is what the architecture exists to avoid. |

## Training

### Recipe: g4's targets, the new torso

The only change from the Gemma 4 fork's g4 recipe (JevBench public 183/231, the
strongest model across these forks) is the torso. That is the cleanest
comparison available, and it removes the two most expensive stages:

| Stage | v19 recipe | b1 (this fork) |
| --- | --- | --- |
| Corpus | built from downloads | **copied** from `~/hobson-gemma4/data` (see below) |
| Teacher labelling | Qwen3.5-4B, ~1 h | none: the committed gemma-4-31B-it labels |
| Parent + replay | ~5 h + ~40 min | none: v14's committed replay for the multi-step rows |
| Train | ~6 h | *estimate* 3 to 4 h |
| Calibrate, eval | as upstream | as upstream, plus the order-sensitivity measure |

Training rows: `train_v5`, `multistep_v14`, `generated_v16`, `generated_v18`,
`adequacy_hs2`, `adequacy_gen`, the same six files as v19 and g4. Targets: gold
labels, plus `teacher_weight: 1.0` KL toward `data/teacher_g4.jsonl` (31B distributions
on the short-task, generated and adequacy rows; v14's replay on the multi-step rows;
rating-scale rows on gold alone). `kl_frozen_weight: 0`, as in g4.

**The corpus must be the local build, not a fresh one.** Teacher labels attach to rows
by position. The `train_v5.jsonl` in `~/hobson-gemma4/data` and `~/hobson/data` has
sha256 `e63263d8…`, while upstream's `data/SHA256SUMS` records `3d17d148…` for a fresh
build. The 31B labels were made against the local file. So `recipe.sh` gets a
`corpus` stage that copies the six training files, `holdout_v5_norule.jsonl` and the
eval sets from `~/hobson-gemma4/data`. It also copies the five
`teacher_gemma4-31b-it_*.jsonl` files and `scripts/merge_teacher_g4.py` from that fork.
`data/SHA256SUMS` is updated to the local hashes, so `verify` keeps guarding positional
alignment. The merge script's output is also checked by sha against g4's
`data/teacher_g4.jsonl`.

### Loss and optimisation

The training loop (`train.py`) is unchanged apart from the forward call: cross-entropy
on the gold label with 0.1 ordinal smoothing on scores, teacher KL, and options
re-shuffled per example per epoch (scores are reversed, not permuted). Hyperparameters
are kept equal to g4 so that the torso is the variable:

The run's config is `configs/experiments/b1.yaml`. Its torso-specific keys:

```yaml
base_model: "google/t5gemma-2-1b-1b"
head_type: "xpointer"        # query: decoder <bos>; keys: encoder, option line's last token
pointer_dim: 256
head_dropout: 0.05
max_length: 4096
lora_r: 16
lora_alpha: 32
lora_targets: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
                             # encoder and decoder; vision tower deleted before attach
micro_batch_size: 16         # g4: 8 x 4; same 32 rows per step (speed, see Phase 0)
grad_accum: 2
lr: 0.0001
head_lr: 0.001
epochs: 1
gradient_checkpointing: true
compile_layers: true         # speed only (Phase 0)
group_by_length: true
teacher_file: "data/teacher_g4.jsonl"
teacher_weight: 1.0
kl_frozen_weight: 0.0
```

LoRA reaches both stacks. In the decoder it adapts both the one-token query and,
through the shared projections of merged attention, how the decoder reads the
encoder.

### Memory and time on the 3090 (*estimates*)

- Weights: ~1.7B text parameters in bf16 = 3.4 GiB once the vision tower is deleted.
- Compute per token: the encoder is ~0.7B non-embedding parameters over the full
  prompt. The decoder runs over one position. v19 ran ~1.5B over the full prompt. So
  roughly half of v19's FLOPs per training token, and with ~3,450 optimizer steps,
  about 3 to 4 h against v19's 6 h. The first 40 steps measure this. Record it in
  `training/hardware.md`.
- Attention: the encoder's local layers need an explicit bidirectional window mask, so
  SDPA uses the memory-efficient kernel rather than flash. Head dim is 256. Measure
  peak memory at the longest batch: `group_by_length` runs it first, so an
  out-of-memory error appears at step 1.
- The 262k-row embedding is touched only as an input lookup. No LM-head projection
  runs in training while `kl_frozen_weight` is 0.

## Inference

### Single question

One encoder forward over `<bos>` + state + question, then a one-token decoder forward.
Output, calibration (temperature per primitive) and the HTTP API do not change.
*Estimate*: about half of v19's per-question cost. v19 measured a median 115 ms per
JevBench question on this 3090.

### Many questions: the shared-prefix cache does not carry over

Upstream encodes the state once and forks its KV cache across questions. That relies
on causality: the state's representation does not depend on what follows it. Under a
bidirectional encoder it does, so the cache would compute a different function from
the one trained. The engine therefore:

- for `config.is_encoder_decoder`, always takes the batched path: N rows of state +
  question_i in one forward, chunked by `max_batch`;
- ignores `use_prefix_cache` for this torso and says so once in the log, rather than
  raising `UnforkableCache` per request.

Cost, in encoder-token x parameter units (*estimate*), for a 2,000-token state:

| | v19 (prefix cache) | b1 (joint encoder) |
| --- | --- | --- |
| 1 question (40 tokens) | 2,040 x 1.5B | 2,040 x 0.7B, **~2x cheaper** |
| 5 questions | 2,200 x 1.5B | 10,200 x 0.7B, **~2x dearer** |

Upstream's claim that more questions are "nearly free" therefore holds only for short
states. The README and `docs/inference.md` have to say so. If multi-question cost
matters, the split design above restores exact sharing.

### Truncation

`_fit` is unchanged: the question is reserved first, and the state takes what is
left. Under bidirectional attention a truncated state degrades evenly instead of
removing context from the options. The rule "drop, never truncate" for training rows
also stays.

### Devices

CUDA and CPU through torch. MPS should work (no custom kernels are needed). MLX is not
supported: `mlx-lm` has no T5Gemma 2, so `--device mlx` refuses this torso.
`mps_kernels.py` does not apply. `hf_export.py` is out of scope for the first run.

## Phase 0: probes before any training (~1 to 2 h)

Each probe is cheap and decides one setting above. Results go into the preregistration.

1. **Load and smoke.** Load in bf16 with `attn_implementation="sdpa"`, delete the vision
   tower, attach LoRA, and check that every LoRA module is under `encoder.text_model`
   or `decoder`. Then run one forward and one backward at batch 8 x 4,096 tokens.
   Record peak memory and step time.
2. **Tokenisation.** Does the tokenizer add `<bos>` (the g2 to g3 lesson: running
   without it cost 4+ JevBench tasks)? Does the encoder input want a trailing `<eos>`?
   Are `1`…`9` single, round-trip tokens?
3. **What the torso knows untrained.** Frozen readout on the held-out short tasks:
   softmax over `1`…`9` at the decoder's `<bos>` through the soft cap. Variants: decoder
   input `<bos>` alone or `<bos>` + a short fixed lead such as `Answer:`. Pick the best
   variant. It is the query position for training too.
4. **Frozen-feature head fits.** As upstream did for the pointer head on frozen v6
   features: freeze the torso, fit the head alone on ~20k rows, and compare on
   held-out tasks: mean against last-token option pooling, and decoder against encoder
   (`<answer>`) query. Also fit v19's torso under the same protocol as the reference.
   The winner goes into b1's config. Ties go to the defaults above.
5. **Padding invariance.** Right padding with the encoder's bidirectional mask must
   leave real tokens' states unchanged. Pin it with a CPU test on a tiny random config,
   as `tests/test_prefix_cache.py` pins the cache.

### Phase 0 results (3 October 2026)

All runs are on the RTX 3090 under WSL2, with `research/scripts/bidi_phase0.py` and
transformers 5.17.0. Raw outputs are in `~/hobson-bidi/reports/phase0/`. The held-out
file is 3,000 rows of `holdout_v5_norule.jsonl` (emotion, hate_severity,
massive_intent, sarcasm: tasks never trained on). The head-only fits train a two-norm
pointer head on frozen features of 20,000 shuffled `train_v5` rows.

**Load and memory.** Deleting the vision tower and projector removes 418M parameters,
leaving 1,698M. LoRA (r=16) on the seven projections reaches 182 modules in each stack,
26.1M trainable, and none in the vision tower. Forward plus backward, with gradient
checkpointing, at micro-batch 8:

| tokens | 256 | 1,024 | 2,048 | 3,072 | 4,096 |
| --- | --- | --- | --- | --- | --- |
| time | 1.06 s | 2.29 s | 4.31 s | 6.65 s | 9.40 s |
| peak memory | 4.0 GiB | 5.5 GiB | 7.6 GiB | 9.7 GiB | 11.8 GiB |

Memory is not a constraint. Short batches are about 3x slower than their FLOP count
predicts. **The 1 s floor is CPU-side kernel launches.** The profiler puts GPU time at
481 ms of a 1,034 ms step at 8 x 256, across 26,829 launches: LoRA (364 wrappers, each
with dropout and a bf16/fp32 cast) and gradient checkpointing (which reruns every
forward) contribute about 400 ms each. Two fixes, neither changing the objective:

| rows/s at 256 tokens | micro-batch 8 | 16 | 32 |
| --- | --- | --- | --- |
| eager | 7.7 | 11.3 | 14.7 |
| `torch.compile` per layer (`dynamic=True`, compiled once) | 15.8 | **22.0** | 26.9 |

Compilation alone is 1.6x (at 2,048 tokens) to 2x (at 256). b1 uses compiled layers
and micro-batch 16 x accumulation 2, the same 32 rows per step: 2.9x on short rows, and
12.3 GiB at 16 x 2,048. GPU tests pin that compiled layers compute the same function
and that checkpointing replays LoRA's dropout masks under compilation.

**Padding invariance holds.** In fp32, right-padding a row, alone or beside a longer
row, changes its encoder states by at most 1.2e-5 relative (L2), with sdpa or eager
attention. In bf16 the maximum absolute difference is 12, but encoder states have
outlier dimensions with |h| up to 982, where one bf16 step is 4 to 8. The head's
LayerNorms absorb this.

**Untrained readout: weak.** Softmax over `1`…`k` at the decoder's first position:

| | all | choice | noul | score |
| --- | --- | --- | --- | --- |
| chance | 0.274 | 0.171 | 0.500 | 0.250 |
| Qwen3.5-2B-Base, last token (v19's torso) | **0.482** | 0.577 | 0.550 | 0.272 |
| T5Gemma 2, decoder `<bos>` | 0.319 | 0.238 | 0.543 | 0.260 |
| T5Gemma 2, decoder `<bos>Answer:` | 0.334 | 0.269 | 0.543 | 0.264 |
| T5Gemma 2, encoder `+<eos>`, decoder `<bos>` | 0.323 | 0.268 | 0.539 | 0.231 |

The pretrained T5Gemma 2 barely follows the option-number format without training. A
frozen-KL anchor toward this readout would pull toward a poor target, which confirms
g4's `kl_frozen_weight: 0`. Open question 3 is closed: no anchor.

**Frozen features: T5Gemma 2 is stronger, and last-token pooling wins.** Head-only
fits:

| torso | query | option key | acc | NLL |
| --- | --- | --- | --- | --- |
| Qwen3.5-2B | `<answer>` (last token) | last token | 0.472 | 1.454 |
| Qwen3.5-2B | `<answer>` | mean of line | 0.438 | 1.684 |
| T5Gemma 2 | decoder `<bos>` | **last token** | **0.523** | **1.223** |
| T5Gemma 2 | decoder `<bos>` | mean of line | 0.472 | 1.526 |
| T5Gemma 2 | encoder `<answer>` | last token | 0.511 | 1.322 |
| T5Gemma 2 | encoder `<answer>` | mean of line | 0.484 | 1.601 |

Decisions for b1:

- **Option key: last token of the line, not the mean.** The design's default was wrong.
  Even bidirectionally, the line-final token beats the mean by 0.04 to 0.05 for either
  query. Mean pooling dilutes the option with its `k.` number prefix and its rubric
  text.
- **Query: the decoder's `<bos>`.** +0.012 over the encoder's `<answer>`, and lower NLL.
  The encoder-only arm is not promoted (open question 2).
- **Encoder input: the tokenizer's default** (`<bos>`, no `<eos>`). Decoder input:
  `<bos>` alone. The `Answer:` lead helped the untrained readout by 0.015, which is
  irrelevant once no anchor reads it.
- T5Gemma 2's frozen features beat Qwen3.5-2B's by +0.051 with the same head, despite
  the much weaker untrained readout. This is the first evidence for the torso swap.

## The first run, b1

Preregistered in
[PREREGISTRATION-b1.md](../research/preregistrations/PREREGISTRATION-b1.md), which holds
the predictions as fixed, the run and the full outcome. In brief:

| | v19 | g4 | b1 | prediction |
| --- | --- | --- | --- | --- |
| JevBench (231) | 168 | 183 | **153** | >= 179, failed |
| order sensitivity, TV / argmax flips | 0.088 / 15.6% | | **0.035 / 5.2%** | <= 0.044, held |
| JevBench ECE; held-out ECE | 0.051; 0.054 | 0.048; 0.065 | **0.044**; 0.065 | held |
| floors within 0.02 of g4 (nine sets) | | | three held, six below | failed |
| single-question latency, median | 115 ms | | 216 ms | < 115 ms, failed |

What held is what the architecture argument predicted: order invariance, calibration,
and short classification (held-out 0.671, the best recorded). What failed is judgement
and reasoning (multi-hop, temporal and numeric, adequacy, HotpotQA 0.605), at every
prompt length. So the encoder's local window is not the cause. Adequacy fails one-sidedly:
adequate answers are called inadequate. The leading reading, not measured, is torso
capacity: Gemma 3 1B is weaker at reasoning than Qwen3.5-2B or Gemma 4 E2B.

Where this document's estimates were wrong:

- *Latency.* b1 is slower than v19 at every state length, not twice as fast. At short
  lengths the cost looks like per-layer overhead across 52 uncompiled layers, not FLOPs.
- *Training time.* It took 3 h, inside the first estimate, but only after compiling
  the layers and switching to `expandable_segments` (the first launch spilled under WSL2).
- *Phase 0 as a predictor.* The frozen-feature fits used short held-out classification,
  where b1 did win. They never sampled the reasoning families, where it lost. A future
  probe needs a reasoning slice.

## Code changes

As built. Phase 0 chose last-token option keys, so the collator's existing `opt_idx` is
all the head needs: no span pooling, and the collator, evaluator and training loop are
unchanged apart from recognising the new head type. The tokenizer adds `<bos>` itself,
so the Gemma 4 fork's `force_bos` was not ported. Mean pooling and an encoder-only query
were measured in Phase 0 and not implemented.

| File | Change |
| --- | --- |
| `modeling.py` | `_load_torso`: for `model_type == "t5gemma2"`, load `T5Gemma2Model`, delete the vision tower and projector, cast after load. `readout_states()` returns (option-key states, query): the last-layer states and the last real token for a causal torso, the encoder states and the decoder's `<bos>` state for an encoder-decoder. `CrossPointerHead` (`head_type: "xpointer"`). `frozen_slot_log_probs` reads through `readout_states` and applies a logit soft cap when the config sets one. `is_encoder_decoder()`, `decoder_start_id()`, `compile_layers()`. |
| `data/collate.py` | `POINTER_HEADS = ("pointer", "xpointer")`, used wherever the code asked `== "pointer"`. |
| `infer.py` | An encoder-decoder torso turns the shared-prefix cache off once, at engine start, and takes the batched path. `--device mlx` refuses an encoder-decoder base model, after the availability check. |
| `train.py` | `compile_layers` (default off, so saved configs reproduce). Gradient checkpointing reaches both stacks unchanged. |
| `configs/experiments/b1.yaml`, `configs/train-bidi.yaml` | b1, and the recipe default (b1 except `output_dir`). `configs/train.yaml` stays v19's, as upstream's tests pin it. |
| `training/recipe.sh` | `corpus` (copy from `$CORPUS_SRC` and verify) and `teacher31b` (merge the committed labels, verify against g4's hash). `all` runs corpus, teacher31b, train, calibrate, eval. `all-v19` keeps upstream's route. |
| `data/SHA256SUMS`, `tests/test_data_identity.py` | The local `train_v5` and holdout builds replace upstream's entries. Entries for the copied files, `teacher_g4.jsonl` and the five label files. |
| `scripts/merge_teacher_g4.py`, `data/synthetic/teacher_gemma4-31b-it_*` | Ported from `hobson-gemma4`. The merge reproduces `teacher_g4.jsonl` byte for byte. |
| `evaluation/order_sensitivity.py` | New: total-variation shift and argmax flips under option reversal. |
| `tests/test_t5gemma2.py` | Tiny random T5Gemma 2, real tokenizer, skipped without access. CPU: vision tower gone and LoRA on text stacks only, masked distribution and gradients, padding and batching invariance, engine batching, checkpoint round-trip, frozen readout equal to the LM's own logits, MLX refusal. GPU: compiled layers compute the same function, and checkpointing replays dropout masks, compiled and eager. |
| `research/scripts/bidi_phase0.py` | The Phase 0 probes. |

## Environment and workflow

- **Source of truth:** this folder (`C:\Users\marcb\Documents\projects\hobson-bidi`),
  a fork of `strands-labs/strands-decider` (remote `upstream`).
- **Training copy:** `~/hobson-bidi` in WSL, a clone of this folder via
  `/mnt/c/...`, updated with `git pull`. Data, checkpoints and the HF cache stay on the
  Linux filesystem, as upstream advises: reads across `/mnt/c` bottleneck loading.
- **Venv:** `~/venvs/hobson-bidi`, the same pins as `~/venvs/hobson` (torch 2.7.1+cu126,
  transformers 5.17.0, peft 0.21.0), without `flash-linear-attention`.
- **Access:** the model is gated. Accept the Gemma terms on its Hugging Face page, then
  run `hf auth login` inside WSL. No token is stored there now (Gemma 4 is not gated,
  which is why its downloads worked).

## Licence

T5Gemma 2 is released under the Gemma Terms of Use, not Apache-2.0 (upstream's Qwen
torso and Gemma 4 are Apache-2.0). The code in this fork stays Apache-2.0. Trained
adapters and heads are derivatives of a Gemma model and carry its terms, including the
prohibited-use policy. This matters only when publishing a checkpoint.

## Open questions

1. *Joint-encoder accuracy against multi-question cost.* Answered against: b1 lost
   accuracy and paid the multi-question cost anyway (eight questions on a
   4,000-token state, 2.3 s against v19's 0.5 s).
2. *Encoder-only or encoder-decoder.* Phase 0 put the decoder query +0.012 ahead on
   frozen features, and b1 kept it. Still open, and now the next direction, chosen for
   latency and multi-question cost: an encoder-only torso, with candidates compared
   first in a Phase 0 that includes a reasoning slice.
3. *A frozen-KL anchor.* Not needed: the untrained readout was 0.33, near chance, and b1
   trained without one, as g4 did.
