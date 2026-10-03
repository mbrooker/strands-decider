# Pre-registration: b1, g4's recipe on a T5Gemma 2 (1B+1B) torso

Committed before training. Same discipline as upstream and the hobson-gemma4 fork:
predictions fixed here, the outcome appended below without editing anything above it.
The design is [docs/bidi-design.md](../../docs/bidi-design.md). Phase 0 (no training)
is recorded there.

## Why

Every model in this series reads its prompt causally. Upstream names two costs of that
in [architecture.md](../../docs/architecture.md): a bidirectional encoder "would pool
better in principle", and option order moves the answer, because each option has read
only the options before it. Converting a causal LLM to bidirectional attention needs
retraining that a 3090 cannot afford. T5Gemma 2 ships a pretrained bidirectional
encoder: Gemma 3 adapted to an encoder-decoder.

Phase 0 (3 October 2026, RTX 3090) read both torsos frozen on 3,000 held-out rows:

| frozen | Qwen3.5-2B-Base (v19's torso) | T5Gemma 2 1B+1B |
| --- | --- | --- |
| untrained readout (option-number logits) | **0.482** | 0.334 (best variant) |
| head-only fit, best query x option key | 0.472 | **0.523** (decoder `<bos>` x option last token) |

The pretrained T5Gemma 2 does not follow the option-number format: its untrained readout
is near chance on choice questions. But a pointer head fitted on its frozen states is
+0.051 ahead of the same head on v19's torso, on tasks the head never saw. g4 has no
frozen-torso anchor, so it does not read the weak untrained readout. The representation
is what training adapts.

## What is being changed

The torso, and two speed settings it requires (`configs/experiments/b1.yaml`):

| | g4 (hobson-gemma4) | b1 |
| --- | --- | --- |
| torso | gemma-4-E2B-it, causal | **google/t5gemma-2-1b-1b**, encoder-decoder; vision tower dropped |
| query | last token (`<answer>`) | **decoder state at `<bos>`**, cross-attending the encoder |
| option keys | last token of each option line | last token of each option line, **encoder** state |
| head | pointer, one LayerNorm | pointer, **separate query and key LayerNorms** (`xpointer`) |
| LoRA r=16 on q/k/v/o/gate/up/down | torso | encoder and decoder (26.1M trainable) |
| micro-batch x accumulation | 8 x 4 | **16 x 2** (32 rows per step either way) |
| compiled torso layers | no | **yes** |
| rows, teacher file, losses, lr, epochs, seed | | unchanged: v19's six files, `data/teacher_g4.jsonl` (rebuilt byte for byte from the committed 31B labels), no frozen anchor |

The two speed settings change bf16 rounding and how rows group into micro-batches, not
the objective. Phase 0 found a step bound by CPU kernel launches. Compiling each layer
and doubling the micro-batch gives 2.9x the rows per second at 256 tokens. A GPU test
pins that the compiled layers compute the same function, and that gradient
checkpointing replays the LoRA dropout masks under compilation
(`tests/test_t5gemma2.py`).

Training on the local RTX 3090 under WSL2 (torch 2.7.1+cu126, transformers 5.17.0, peft
0.21.0). Evaluation on the same card, with the same scripts as g4's fixed evaluation
(upstream's `collect_logits` and `multistep_eval` already carry the fork's two masking
fixes).

## Baselines

g4's figures are from PREREGISTRATION-g4.md (the six sets re-run on this 3090). v19's
order sensitivity was measured for this preregistration by
`evaluation/order_sensitivity.py`.

| evaluation | v19 | g4 |
| --- | --- | --- |
| JevBench (231) at 4096: tasks / Brier / ECE | 168 / 0.342 / 0.051 | 183 / 0.292 / 0.048 |
| held-out short tasks: accuracy / ECE | 0.647 / 0.054 | 0.655 / 0.065 |
| MuSiQue / ContractNLI / BoardgameQA / HotpotQA (held out) | 0.879 / 0.862 / 0.810 / 0.726 | 0.902 / 0.861 / 0.781 / 0.759 |
| generated documents, v16's / v18's | 0.846 / 0.757 | 0.863 / 0.794 |
| adequacy: HelpSteer2 / generated (balanced) | 0.739 / 0.788 | 0.722 / 0.801 |
| order sensitivity on held-out rows, options reversed: mean TV / argmax flips | 0.088 / 0.156 (3,000 rows; choice 0.097 / 0.105, yes/no 0.066 / 0.113, score 0.083 / 0.308) | not measurable here (hobson package) |
| JevBench latency, RTX 3090: median / p95 | 115 ms / 299 ms (README) | |

Noise: two runs of one recipe differ by about 4.5 JevBench tasks (hobson-gemma4); six
v17 retrains had a standard deviation of 3.2 (upstream).

## Predictions

1. **JevBench at least 179** of 231 at the 4096 window: no worse than g4 beyond noise.
2. **Bidirectional reading cuts order dependence:** mean total-variation shift under
   option reversal on the held-out rows at most half of v19's (at most 0.044), and
   fewer argmax flips than v19.
3. **Nothing is lost against g4**, each within 0.02: MuSiQue at least 0.882, ContractNLI
   at least 0.841, BoardgameQA at least 0.761, HotpotQA at least 0.739; generated sets at
   least 0.843 and 0.774; adequacy at least 0.702 (HelpSteer2) and 0.781 (generated,
   balanced); held-out short-task accuracy at least 0.635.
4. **Calibration holds:** JevBench ECE at most 0.07 and held-out ECE at most 0.074.
5. **Serving is faster for one question:** JevBench median latency on the 3090 below v19's
   115 ms, and p95 at most 600 ms.

Exploratory, no numeric prediction: JevBench by tier and by family against g4 and v19
(paired); BoardgameQA, where g4 lost to v19 and joint reading of the rules might help;
the multi-question cost (five questions on a 2,000-token state, against v19's prefix
cache), measured and reported; validation loss and accuracy.

## Which model is recommended afterwards

b1 becomes hobson-bidi's reference model if predictions 1, 3 and 4 hold. If 2 holds as
well, the README states the order result as a property of the architecture. If 1 holds by
a margin beyond noise (188 or more), the torso is the gain, and the next step is the split
design or encoder-only serving, to win back multi-question cost.

## What would count as failure

- **JevBench under 179.** The frozen-feature advantage did not survive training, or the
  weak untrained readout matters more than g4's recipe allows for. The next step would be
  the encoder-only arm or a longer schedule, not a frozen anchor: the readout it would
  pull toward is near chance.
- **Prediction 2 fails.** Then position embeddings, not causal attention, carry most of
  the order effect, and the bidirectional argument for this torso rests on accuracy alone.

## What this test cannot show

- **Torso and speed settings together.** The micro-batch and compilation change rounding
  and grouping. They are not expected to matter, and are not separated.
- **Which half of the torso matters.** Phase 0 put the decoder query +0.012 ahead of an
  encoder-only query on frozen features. Training both does not show the decoder is
  needed.
- **One run**, one seed.
- **Licence.** T5Gemma 2 is under the Gemma Terms of Use. A published b1 checkpoint would
  carry them; g4 and v19 are Apache-2.0.

## Smoke test (added before training)

3 October 2026, RTX 3090: `configs/experiments/b1.yaml` with `max_steps: 40` (the cosine
schedule compressed into 40 steps), then the checkpoint through `strands-decider ask`,
`evaluation/order_sensitivity.py` (300 rows) and `evaluation/multistep_eval.py`
(generated adequacy). Everything ran. Outputs are in `~/hobson-bidi/reports/smoke/`.

- **Data:** 93,298 teacher rows on 123,339, train 119,623 / val 3,716, as g4. Grouped
  padding 6.3% of positions, against 80.8% ungrouped.
- **Memory:** peak 15.4 GiB. The longest batch runs first, so this is the run's peak.
- **Speed:** 52 layers compiled. 40 steps in 210 s, including compilation and the longest
  batch; about 3.2 s per step by the last interval. The full run is 3,738 steps, so
  *about 3.5 to 4.5 h* (v19: about 6 h on this card).
- **Loss:** 2.23 at step 5, 1.47 at step 40; teacher KL 0.90 to 0.68. Validation
  0.902 / 0.535 after 40 steps.
- **Serving:** the engine logs that it turned off the shared-prefix cache, and answers all
  three question types.
- **Measures run:** order sensitivity tv 0.161, flips 0.347. Score rows flip 0.929: after
  40 steps the head still reads rubric position, not rubric text. Generated adequacy
  0.460. These are plumbing checks, not results.

## Outcome (added after the run)
