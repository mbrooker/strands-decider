# hobson-bidi: encoder-only torsos

The second direction of this fork. [bidi-design.md](bidi-design.md) covers the first, an
encoder-decoder, and its run b1. This document covers why the fork moved to encoder-only
torsos, the Phase 0 that chose between them, and the design of the runs it led to: e1 in two
arms, then e2. Figures marked *estimate* are projections, not measurements.

## Why

b1 (T5Gemma 2 1B+1B) failed its preregistration: JevBench 153/231, against v19's 168
([PREREGISTRATION-b1.md](../research/preregistrations/PREREGISTRATION-b1.md)). What held was
what bidirectional reading promised: order sensitivity fell to 0.035 from v19's 0.088, and
calibration was the best recorded. What failed was reasoning and judgement at every prompt
length, plus latency. A request cost 216 ms against v19's 115, and eight questions on a
4,000-token state took 2.3 s against v19's 0.5 s with its prefix cache.

The goals for this direction, in order: **latency and multi-question cost, with JevBench no
worse than b1's 153.** An encoder-only torso drops b1's decoder, which on its own cost about
70 ms per call (b1's torso: 130 ms at 256 tokens, its encoder alone 60 ms). It keeps the
bidirectional reading that produced b1's gains. Whether it recovers reasoning depends on
which encoder, and that is what Phase 0 measured.

## Phase 0 (4 October 2026)

`research/scripts/encoder_phase0.py`, RTX 3090, transformers 5.17.0. Raw outputs are in
`~/hobson-bidi/reports/enc0/`.

**The protocol fixes b1's blind spot.** b1's Phase 0 evaluated only short held-out
classification, where b1 then won. Here, a two-norm pointer head is fitted on frozen
features of 22,000 rows: 16,000 short-task rows, 2,500 multi-step, 1,200 generated
documents and 2,300 adequacy. Every slice is then scored separately:
- held-out short tasks (3,000 rows)
- HotpotQA (959, never trained on)
- BoardgameQA (900)
- generated documents, v16's and v18's eval sets (597)
- adequacy, HelpSteer2 and generated (536, class-balanced)

The "reasoning mean" averages the four non-short slices. Every model sees the same rows: a
row is used only if its prompt is under 14,000 characters, whatever each tokenizer makes of
it. Option keys are each option line's last token. Query candidates are `<answer>`'s last
token, position 0, the mean over the prompt, and, for T5Gemma 2, the decoder's `<bos>`.

**The protocol reproduces b1's outcome.** On frozen features, b1's torso beats v19's on
short held-out tasks (0.523 against 0.482) and loses on reasoning (0.512 against 0.570),
mostly on generated documents (0.479 against 0.693). That is the shape b1 showed after
training, which the old probe could not see.

| frozen, best query | params | held-out | HotpotQA | BoardgameQA | generated | adequacy | reasoning mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **T5Gemma 2B-it UL2, encoder** (mean) | 2.6B | **0.594** | **0.620** | **0.583** | 0.670 | 0.722 | **0.649** |
| T5Gemma 2B-it PrefixLM, encoder (mean) | 2.6B | 0.560 | 0.560 | 0.564 | **0.700** | **0.731** | 0.639 |
| Qwen3.5-2B, v19's torso (last token) | 1.9B | 0.482 | 0.544 | 0.482 | 0.693 | 0.561 | 0.570 |
| T5Gemma 2 1B, b1's torso (mean) | 1.0B enc | 0.523 | 0.507 | 0.443 | 0.479 | 0.617 | 0.512 |
| Ettin encoder 1B (mean) | 1.0B | 0.464 | 0.519 | 0.509 | 0.357 | 0.624 | 0.502 |
| ModernBERT-large (position 0) | 0.4B | 0.292 | 0.537 | 0.392 | 0.360 | 0.546 | 0.459 |

T5Gemma v1 has only Gemma 2-based sizes. 2B+2B is the smallest that is instruction-tuned, and
it is the only candidate clearly above both b1's and v19's torsos. That matches what the
hobson-gemma4 fork saw moving to an instruction-tuned torso: g1 156 to g2 172 on JevBench.

**Cloze framing for the MLM encoders.** The prompt ends `<answer> [MASK]`; the `[MASK]`
state is a query, and the MLM head's distribution over option numbers is an untrained
readout. It does not improve the head fits: the reasoning mean is 0.448 to 0.453 for
ModernBERT and 0.492 to 0.500 for Ettin. But Ettin's untrained readout answers above chance
(held-out 0.421, generated 0.454), and ModernBERT's does not (0.332, 0.330). Ettin has a
pretrained answer position, which ModernBERT lacks. Frozen MLM features are known to
understate these models after fine-tuning, so their rows are floors.

**Torso latency** (bf16, eager, median ms, torso forward only; one question, then eight
against one state):

| torso | 256 x 1 | 1,024 x 1 | 4,000 x 1 | 1,024 x 8 | 4,000 x 8 |
| --- | --- | --- | --- | --- | --- |
| ModernBERT-large | 26 | 30 | 121 | 164 | 888 |
| Ettin encoder 1B | 28 | 52 | 223 | 337 | 1,712 |
| T5Gemma 2B-it encoder | 43 | 92 | 370 | 636 | 2,684 |
| T5Gemma 2 1B, encoder only | 60 | 61 | 206 | 304 | 1,450 |
| T5Gemma 2 1B+1B (b1) | 130 | 131 | 264 | 359 | 1,449 |
| Qwen3.5-2B, no prefix cache | 65 | 75 | 261 | 485 | 1,866 |

A bidirectional torso re-reads the state for every question, so the eight-question column
is the cost the masked arm below exists to remove.

**Decisions.**
- **e1:** the T5Gemma 2B-it UL2 encoder, the accuracy lead. UL2 and PrefixLM tie on the
  reasoning mean (0.649 against 0.639). UL2 leads on held-out (0.594 against 0.560) and its
  objective trains the encoder through span corruption, so UL2.
- **e2:** the Ettin encoder 1B, the speed option, level with b1's torso on reasoning.
- **ModernBERT-large is dropped:** lowest on every reasoning slice.

## e1: the T5Gemma 2B-it encoder

**Torso.** The encoder of `google/t5gemma-2b-2b-ul2-it`, loaded through `T5GemmaModel` with
the decoder discarded. `T5GemmaEncoderModel` refuses an encoder-decoder config in
transformers 5.17. 26 layers, hidden 2304, 8 query heads and 4 KV heads of dim 256,
alternating sliding (window 4,096, bidirectional, about +-2,048 tokens) and full attention,
about 2.6B parameters. Its tokenizer adds no `<bos>`; Phase 0 used that default, and so do
the runs.

**Attention implementation: SDPA, which drops Gemma 2's attention soft-capping (50.0).**
Measured on 32 real prompts against eager attention, which applies the cap: mean-pooled
states differ by a median 1.1% (at most 2.6%), and the last token's state keeps cosine
>= 0.997. Single tokens can differ much more: the median of each prompt's worst token is
0.28 relative L2, at most 1.93. SDPA is 1.7x faster, and Phase 0's results were measured
with it. Training and serving both use SDPA. Mixing implementations between training and
serving would be wrong, and the checkpoint records `attn_implementation`.

**Readout.** Query: the mean of the encoder's last-layer states over the prompt's tokens.
That is Phase 0's best query for both T5Gemma encoders, +0.035 over `<answer>` on the
reasoning mean. Option keys: each option line's last token, as in b1 and v19. Head:
`xpointer`, two LayerNorms, since a mean and a token state differ in scale.

**Recipe.** g4's rows and targets, as b1: v19's six files, the gemma-4-31B-it teacher file,
no frozen anchor. LoRA r=16 on the encoder's seven projections. Compiled layers,
`expandable_segments`. The micro-batch is set by the smoke test: the encoder is about 3x
b1's non-embedding size, so 8 x 4 may not fit beside Windows' own use of the card, and the
fallback is 4 x 8. Either way, a step is 32 rows.

**Serving.** Batched, as b1: no shared-prefix cache, because a bidirectional state depends
on the question. The engine's check is "is the torso bidirectional", which covers
encoder-only torsos as well as encoder-decoders.

### e1b: the masked state cache

Same torso, readout and recipe as e1a, with one change to attention:

- **State tokens attend only to state tokens.**
- **Question tokens attend to the state and to their own question.**

Within the question, options still read each other in both directions, so b1's
order-invariance mechanism is intact. What is given up is reading the state with the
question in mind. Phase 0 cannot measure that cost; the run does.

Because the state never sees a question, its states at every layer are the same for every
question in a request. Serving encodes the state once (batch 1, positions 0 to S-1),
keeping each layer's keys and values. It then runs all N questions as one batch
(positions from S), each attending to the cached state keys and values plus its own.
That is exact, not an approximation: the model computes what it was trained to compute.
Eight questions on a 4,000-token state cost one state pass plus eight question passes:
*about 400 to 500 ms* on this card, against about 2.7 s batched (*estimate* from the
latency table).

Implementation:
- **Masks.** The collator emits each row's state length in tokens. The model builds both of
  the encoder's per-layer masks from transformers' own `create_bidirectional_mask` and
  `create_bidirectional_sliding_window_mask`, so padding and the 4,096 window are exactly
  e1a's. It ANDs in the state/question rule and passes the encoder the
  `{"full_attention", "sliding_attention"}` dict it accepts.
- **The query.** The mean over the prompt now mixes question-independent state tokens with
  the question's tokens, which is the same definition as e1a's.
- **Serving.** A layer loop that reuses each layer's own projections, norms and rotary
  embedding, with the state's keys and values concatenated ahead of the questions'. A CPU
  test pins it equal to the full masked forward in fp32. If that test fails, e1b serves
  through the exact batched path and the speed prediction is scored as failed.

## e2: the Ettin encoder 1B

`jhu-clsp/ettin-encoder-1b` (ModernBERT architecture): 28 layers, hidden 1792, local
attention (window 128) with global attention every third layer, about 1.0B parameters.

- **Readout:** as e1, mean query and last-token keys.
- **LoRA:** on ModernBERT's projections (`Wqkv`, `Wo`, `Wi`, `Wo` of the MLP; names to
  be checked against the loaded module).
- **Head initialisation:** random, as e1, for comparability. A head seeded from Ettin's own
  cloze readout is a follow-up if e2 is close.
- **The masked state cache:** applies to e2 as well. Whichever of e1a and e1b wins sets
  e2's arm.

e2 is written up so it is not lost. It is not scheduled until e1 reports.
