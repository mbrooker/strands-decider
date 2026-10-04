# Pre-registration: e1, an encoder-only torso (T5Gemma 2B-it), unmasked and masked

Committed before training either arm. Predictions are fixed here; each arm's outcome is
appended below without editing anything above it. The design and the Phase 0 that chose
this torso are in [docs/encoder-design.md](../../docs/encoder-design.md).

## Why

b1 (T5Gemma 2 1B+1B) kept what bidirectional reading promised (order sensitivity 0.035, the
best calibration recorded) and lost on reasoning, adequacy and latency: JevBench 153 against
v19's 168 ([PREREGISTRATION-b1.md](PREREGISTRATION-b1.md)). The goals now are **latency and
multi-question cost, with JevBench no worse than b1.**

The encoder-only Phase 0 (4 October 2026, frozen features, a reasoning slice included,
same rows for every model) reproduced b1's outcome, then ranked the candidates:

| frozen, best query | held-out | reasoning mean | torso latency, 256 tokens x 1 / 4,000 x 8 |
| --- | --- | --- | --- |
| **T5Gemma 2B-it UL2, encoder** | **0.594** | **0.649** | 43 / 2,684 ms |
| Qwen3.5-2B (v19's torso) | 0.482 | 0.570 | 65 / 1,866 ms (no cache) |
| T5Gemma 2 1B (b1's torso) | 0.523 | 0.512 | 130 / 1,449 ms (with decoder) |

This is the only candidate clearly above both earlier torsos on the slices where b1 lost.
Its multi-question cost on long states is the highest of them. That is what the second arm
addresses.

## What is being run

Two arms, the same in everything but attention. Both are on the local RTX 3090 under WSL2.

| | e1a (unmasked) | e1b (masked state cache) |
| --- | --- | --- |
| torso | encoder of `google/t5gemma-2b-2b-ul2-it`, SDPA (no attention soft-cap; measured, see design) | same |
| attention | bidirectional over the whole prompt | state attends to state only; question attends to state and itself |
| serving | batched: each question re-reads the state | state encoded once per request, its keys and values reused by every question |
| query / keys / head | mean over the prompt / option line's last token / `xpointer` | same |
| rows, teacher, losses | g4's (as b1): six files, `data/teacher_g4.jsonl`, no frozen anchor | same |
| LoRA, schedule | r=16 on the seven projections; 1 epoch; lr 1e-4, head 1e-3; 32 rows per step | same |
| config | `configs/experiments/e1a.yaml` | `configs/experiments/e1b.yaml` |

e1a trains first. e1b's serving path needs code that is written while e1a trains. e1b's
code and config are committed before e1b trains, and noted in its outcome. The micro-batch
(8 x 4, or 4 x 8 if 8 does not fit) is set by e1a's smoke test, recorded in the Smoke
section, and used for both arms.

## Baselines

v19's sets are the rerun on this card (PREREGISTRATION-b1.md, Outcome). JF100 and Typed
Decisions for v19 and g4 are from hobson-gemma4's README. b1's are measured in e1a's
pipeline, with the same scripts.

| evaluation | v19 | g4 | b1 |
| --- | --- | --- | --- |
| JevBench (231) / Brier / ECE | 168 / 0.342 / 0.051 | 183 / 0.292 / 0.048 | 153 / 0.411 / 0.044 |
| held-out short tasks: accuracy / ECE | 0.647 / 0.054 | 0.655 / 0.065 | 0.671 / 0.065 |
| MuSiQue / ContractNLI / BoardgameQA / HotpotQA | 0.882 / 0.873 / 0.820 / 0.719 | 0.902 / 0.861 / 0.781 / 0.759 | 0.923 / 0.843 / 0.748 / 0.605 |
| generated, v16's / v18's | 0.857 / 0.777 | 0.863 / 0.794 | 0.797 / 0.741 |
| adequacy: HelpSteer2 / generated, balanced | 0.722 / 0.788 | 0.722 / 0.801 | 0.590 / 0.693 |
| order sensitivity: TV / flips | 0.088 / 0.156 | | 0.035 / 0.052 |
| JevBench latency p50 / p95 (3090) | 115 / 299 ms | | 216 / 348 ms |
| 8 questions on a 4,000-token state (`bench_local`) | 500 ms (prefix cache) | | 2,317 ms |
| JF100 (300) / Typed Decisions accuracy | 162 / 0.614 | 172 / 0.669 | measured with e1a |

Noise: about 4.5 JevBench tasks between two runs of one recipe.

## Predictions: e1a

1. **JevBench at least 168** (v19). Below 153 (b1) is the failure named below.
2. **Reasoning recovers to v19's level**, each within 0.02 of v19's rerun: HotpotQA at least
   0.699, BoardgameQA at least 0.800, generated at least 0.837 and 0.757, adequacy at least
   0.702 (HelpSteer2) and 0.768 (generated, balanced). Also MuSiQue at least 0.862,
   ContractNLI at least 0.853, held-out at least 0.627.
3. **Order sensitivity stays low:** TV at most 0.044, argmax flips below v19's 0.156.
4. **Calibration holds:** JevBench ECE at most 0.07, held-out ECE at most 0.074.
5. **Faster than b1:** JevBench median latency at most 150 ms (b1 216), p95 at most 600 ms.

Exploratory, no numeric prediction: JF100 and Typed Decisions against v19, g4 and b1;
JevBench by tier and family, paired against v19 and b1; `bench_local` at 1, 5 and 8
questions.

## Predictions: e1b, against e1a on the same card

1. **The cache is exact:** the serving path equals the full masked forward. Pinned by a CPU
   test in fp32 before e1b trains, and checked on the trained checkpoint at bf16 tolerance.
2. **Multi-question cost falls:** eight questions on a 4,000-token state at most 700 ms
   (`bench_local`, engine end to end), and five at 1,024 tokens below e1a's.
3. **Accuracy holds:** JevBench at least e1a's minus 5. No eval set more than 0.03 below
   e1a's.
4. **Single-question latency unchanged:** JevBench median within 10% of e1a's.

## Which model is recommended afterwards

- **e1a becomes hobson-bidi's reference** if predictions 1, 3 and 4 hold.
- **e1b replaces e1a** if e1b's 1, 2 and 3 hold. It would then be the first bidirectional
  model here with a shared-state cache.
- **e2** (PREREGISTRATION-e2.md, written alongside this file) inherits the winning arm.

## What would count as failure

- **e1a's JevBench under 153.** A stronger, instruction-tuned encoder did not beat b1. Then
  Phase 0's frozen ranking does not survive training, and the encoder-only direction rests
  on latency alone.
- **e1b more than 5 tasks below e1a.** Reading the state without the question costs more
  than a cache saves, and multi-question serving needs another route.

## What this test cannot show

- **Instruction tuning and architecture together.** The torso differs from b1's in
  generation (Gemma 2 against Gemma 3), size (2.6B against 1.0B encoder), instruction
  tuning, and the absent decoder. A gain cannot be assigned to one of them.
- **SDPA's dropped soft-cap.** Its cost against eager attention is not measured after
  training.
- **One run per arm.**
- **Licence.** T5Gemma is under the Gemma Terms of Use. A published checkpoint would carry
  them.

## Smoke test (added before e1a trains)

4 October 2026, RTX 3090, `configs/experiments/e1a.yaml` with `max_steps: 30` (the schedule
compressed into 30 steps), micro-batch 8 x 4. It loaded the encoder alone: 26 layers
compiled, 21.96M of 2.636B parameters trainable. The longest batch runs first, so its
peak, **14.9 GiB allocated (at most 17.6 GB in use on the card)**, is the run's peak. There
is room beside Windows' own use, so **micro-batch 8 x 4 is kept for both arms.** Speed:
0.21 steps/s by step 30, compilation included. b1's smoke ran 0.19 at step 40, so the full
run is *about 4 h*. Loss 2.59 to 1.78 over 30 steps; validation 1.081 / 0.573. A plumbing
check, not a result.

Code at this preregistration's commit: `query_pool` (mean query), encoder-only loading for
T5Gemma, and `is_bidirectional`, which keeps every bidirectional torso off the prefix
cache. Pinned by `tests/test_t5gemma_encoder.py`; the full suite passes (265).

## Outcome: e1a (added after the run)

Nothing above this section was edited after e1a trained. **By the rule fixed above, e1a does
not become hobson-bidi's reference.** Predictions 1 (JevBench) and 4 (JevBench calibration)
failed. Predictions 2, 3 and 5 held. The named failure, "JevBench under 153", is not met:
e1a scored exactly 153.

The run: commit `3f09448`, RTX 3090 under WSL2, 4 October 2026. Training took 5 h 13 m
(03:41 to 08:54 UTC), 3,738 steps at 0.21 to 0.22 steps/s. Peak allocation 15.6 GiB, at
most 18.3 GB in use on the card, no spill. Final validation loss / accuracy 0.393 / 0.820:
the lowest loss in the series (b1 0.407, g4 0.427, v19 0.421). Calibration temperatures:
choice 1.192, yes/no 1.328, score 0.911. Outputs are in `~/hobson-bidi/reports/e1a/`.

| | prediction | v19 | b1 | e1a | |
| --- | --- | --- | --- | --- | --- |
| 1 | JevBench >= 168 | 168 | 153 | **153** | FAIL |
| 2 | nine sets within 0.02 of v19 | | six below | **all nine held, most above v19** | pass |
| 3 | order TV <= 0.044, flips < 0.156 | 0.088 / 0.156 | 0.035 / 0.052 | 0.038 / 0.055 | pass |
| 4 | JevBench ECE <= 0.07; held-out ECE <= 0.074 | 0.051; 0.054 | 0.044; 0.065 | **0.100**; 0.057 | FAIL |
| 5 | JevBench p50 <= 150 ms, p95 <= 600 ms | 115 / 299 | 216 / 348 | **80** / 348 | pass |

**Sets** (floor in brackets; v19 from its rerun on this card):

| set | v19 | b1 | e1a | |
| --- | --- | --- | --- | --- |
| held-out short tasks [0.627] | 0.647 | 0.671 | **0.695** (ECE 0.057) | pass |
| MuSiQue [0.862] | 0.882 | 0.923 | **0.948** | pass |
| ContractNLI [0.853] | 0.873 | 0.843 | 0.864 | pass |
| BoardgameQA [0.800] | 0.820 | 0.748 | **0.884** | pass |
| HotpotQA, held out [0.699] | 0.719 | 0.605 | **0.748** | pass |
| generated, v16's [0.837] | 0.857 | 0.797 | 0.851 | pass |
| generated, v18's [0.757] | 0.777 | 0.741 | 0.786 | pass |
| adequacy, HelpSteer2 [0.702] | 0.722 | 0.590 | **0.761** | pass |
| adequacy, generated, balanced [0.768] | 0.788 | 0.693 | **0.821** | pass |

b1's adequacy failure is gone. Adequate answers are recognised at 0.718 on HelpSteer2
(b1 0.444, v19 0.684) and 0.802 on the generated set (b1 0.562, v19 0.719).

**JevBench: 153/231** (easy 48, standard 59, hard 46). Paired against v19's recorded run
(167): 11 gained, 25 lost, p = 0.029. Brier 0.373 (v19 0.342, b1 0.411), ECE 0.100,
paraphrase consistency 0.972, the highest recorded. Families at 1.0: fact, intent,
ordinal, routing, routing_hard, tool_selection; extraction 0.958. The weakest:
temporal_numeric 0.133, long_policy 0.211, probability 0.30, multi_hop 0.389.

**External benchmarks** (exploratory; b1 measured with the same scripts):

| | v19 | g4 | b1 | e1a |
| --- | --- | --- | --- | --- |
| JF100 (300) | 162 | 172 | 135 | 158 |
| Typed Decisions: accuracy / KL | 0.614 / | 0.669 / 0.262 | 0.487 / 0.363 | 0.525 / 0.309 |

**Latency** (`bench_local`, engine end to end, median): one question at 256 / 1,024 /
2,048 / 4,000 tokens: 74 / 136 / 260 / 516 ms (b1 197 / 207 / 271 / 406; v19 102 / 133 /
209 / 395). Eight questions on a 4,000-token state: 3,803 ms (b1 2,317, v19 500 with its
cache). Short requests are faster than v19's. Long and multi-question requests are the
slowest in the series, which is e1b's target.

**Diagnostics (exploratory, after the predictions were scored).**
`research/scripts/b1_by_length.py` and `research/scripts/jevbench_by_length.py`:

- *The eval sets gain at every length.* Multi-step by prompt-length quartile against v19:
  +0.061, +0.019, +0.040, +0.031.
- *JevBench's loss is mostly short tasks.* Bucketed by e1a's token counts:

  | input tokens | n | v19 | b1 | e1a |
  | --- | --- | --- | --- | --- |
  | under 500 | 166 | 0.861 | 0.771 | 0.795 |
  | 500 to 1,500 | 28 | 0.321 | 0.321 | 0.357 |
  | 1,500 to 3,000 | 29 | 0.414 | 0.414 | 0.345 |
  | 3,000 and over | 8 | 0.375 | 0.500 | 0.125 |

  About 11 of the 14 tasks e1a is behind v19 are under 500 tokens, in judgement families
  (judge_hard, temporal_numeric). long_policy, 2,296 to 3,874 tokens, adds 3 against v19
  and 6 against b1 (v19 7 of 19, b1 10, e1a 4).

**Reading.** On everything drawn from the training families, e1a is the strongest model
in the series: every set at or above v19, most well above, the held-out tasks the best
recorded, and b1's adequacy failure repaired. Short requests are the fastest yet. On
JevBench, an external benchmark of judgements phrased unlike the training rows, it lands
exactly where b1 did, below v19, and its calibration there is the worst of the three
(ECE 0.100 against 0.057 on held-out tasks). JF100 and Typed Decisions, also external,
show the same: e1a well above b1, below v19 and g4. Two different bidirectional torsos now
improve the in-distribution sets and stop at 153 on JevBench. The pattern points at
something they share: transfer out of the training distribution, with the
instruction-tuned decoders of v19 and g4 generalising further. Torso size alone does not
explain it. Encoder Phase 0's reasoning slices were drawn from the training families, so,
like b1's probe, it could not see this. A probe that predicts JevBench needs out-of-
distribution items: JevBench's own, JF100 or Typed Decisions.

## Outcome: e1b (added after the run)

Nothing above this section was edited after e1b trained. e1b's code and config were
committed at `3cecf40`, before it trained, as the "What is being run" section requires.
**By the rule fixed above, e1b does not replace e1a.** Prediction 1 held. Predictions 2,
3 and 4 each failed on one clause, two of them narrowly. The named failure, "e1b more than
5 tasks below e1a", is not met: e1b is 18 tasks above it.

The run: commit `3cecf40`, RTX 3090 under WSL2, 4 October 2026. Training took about 5 h
(09:22 to about 14:25 UTC), 3,738 steps at 0.21 to 0.22 steps/s, the same rate as e1a.
Peak allocation 15.6 GiB, at most 18.6 GB in use on the card, no spill. Final validation
loss / accuracy 0.406 / 0.813 (e1a 0.393 / 0.820). Calibration temperatures: choice 1.130,
yes/no 1.192, score 2.401. Outputs are in `~/hobson-bidi/reports/e1b/`.

| | prediction (against e1a) | e1a | e1b | |
| --- | --- | --- | --- | --- |
| 1 | the cache is exact: CPU test in fp32; trained checkpoint at bf16 tolerance | | fp32 identical (max difference 0.0000, 0 of 32 answers differ); bf16 max difference 0.005, one near-tie flip in 32 | pass |
| 2 | 8 questions x 4,000 tokens <= 700 ms; 5 x 1,024 below e1a's | 3,803; 617 ms | **785** ms; 227 ms | FAIL (first clause) |
| 3 | JevBench >= e1a - 5 (148); no set more than 0.03 below e1a | 153 | **171**; HotpotQA 0.677 (e1a 0.748) | FAIL (HotpotQA) |
| 4 | JevBench median within 10% of e1a's | 80 ms | 89 ms (+11%) | FAIL |

**Exactness on the trained checkpoint**: eight `bench_local` questions at each of 256,
1,024, 2,048 and 4,000 state tokens, through the cached path and the batched masked path.
In fp32 the two give identical distributions. In bf16 the largest difference in any
probability is 0.005. The one changed answer was a tie: billing 0.459 against technical
0.454 through the cache, 0.456 against 0.457 batched.

**JevBench: 171/231** (easy 48, standard 64, hard 59). Paired against v19's recorded run
(167): 20 gained, 16 lost, p = 0.62, so level with v19, not ahead. Against e1a: +18 with
only the attention mask changed, beyond the 4.5-task noise between retrains. Brier 0.372,
**ECE 0.063**, paraphrase consistency 0.889. Families: adequacy 0.833 (e1a 0.50),
multi_hop 0.667 (e1a 0.389), probability 0.60 (0.30), long_policy 0.368 (0.211), trap
1.0. judge_hard 0.353 and temporal_numeric 0.267 remain the weakest.

| JevBench, by e1a's input tokens | n | v19 | b1 | e1a | e1b |
| --- | --- | --- | --- | --- | --- |
| under 500 | 166 | 0.861 | 0.771 | 0.795 | **0.843** |
| 500 to 1,500 | 28 | 0.321 | 0.321 | 0.357 | **0.429** |
| 1,500 to 3,000 | 29 | 0.414 | 0.414 | 0.345 | **0.552** |
| 3,000 and over | 8 | 0.375 | 0.500 | 0.125 | 0.375 |

**Sets** (e1a's in brackets): held-out 0.669 (0.695, ECE 0.068); MuSiQue 0.942 (0.948);
ContractNLI 0.862 (0.864); BoardgameQA 0.856 (0.884); HotpotQA 0.677 (0.748); generated
0.829 / 0.818 (0.851 / 0.786); adequacy, HelpSteer2 0.731 (0.761); adequacy, generated,
balanced 0.823 (0.821). Every v19-level floor of e1a's prediction 2 still holds except
HotpotQA (0.699).

**External benchmarks:** JF100 156/300 (e1a 158, v19 162, g4 172); Typed Decisions 0.549 /
KL 0.305 (e1a 0.525 / 0.309, v19 0.614, g4 0.669).

**Latency** (`bench_local`, median ms; cache / batched):

| state tokens | 1 question | 5 questions | 8 questions |
| --- | --- | --- | --- |
| 256 | 84 | 166 / 235 | 182 / 337 |
| 1,024 | 146 | 227 / 665 | 253 / 1,041 |
| 2,048 | 285 | 369 / 1,334 | 417 / 2,159 |
| 4,000 | 605 | 731 / 2,861 | 785 / 4,514 |

The cache makes the eighth question nearly free, as the prefix cache does for v19. Eight
questions on 4,000 tokens cost 1.3x one question (v19: 500 ms with its cache). The cost
that remains is the state pass itself, about 0.6 s at 4,000 tokens on this card.

**Diagnostics (exploratory, after the predictions were scored).**
`research/scripts/sets_by_length.py`, v19, e1a and e1b on the same rows, by prompt-length
quartile:

| e1b minus e1a | shortest quarter | 2nd | 3rd | longest | all |
| --- | --- | --- | --- | --- | --- |
| HotpotQA | -0.092 | -0.054 | -0.042 | -0.095 | -0.071 |
| BoardgameQA | -0.036 | 0.000 | +0.009 | -0.089 | -0.029 |
| generated, v16's | -0.046 | 0.000 | -0.011 | -0.034 | -0.023 |
| generated, v18's | 0.000 | +0.066 | +0.049 | +0.016 | +0.032 |
| adequacy, HelpSteer2 | -0.034 | +0.034 | -0.086 | -0.033 | -0.030 |

MuSiQue (-0.006), ContractNLI (-0.002) and generated adequacy (-0.007) are flat. The mask's
cost does not grow with length. It falls mostly on HotpotQA, at every length. HotpotQA's
rows are distractor paragraphs, where finding the relevant passage depends on the
question. Under the mask only the question's own tokens can do that selection; the state's
tokens never see the question. Against v19, e1b is ahead on MuSiQue (+0.060), BoardgameQA
(+0.036), generated v18 (+0.040) and both adequacy sets, and behind on HotpotQA (-0.043),
generated v16 (-0.029) and ContractNLI (-0.012).

**Reading.** The mask was introduced for speed and did more than that: e1b is the first
bidirectional model in this fork to reach v19 on JevBench (171 against 168), and the only
one with an exact shared-state cache. Against e1a, the in-distribution sets dropped a
little (HotpotQA most), while JevBench rose 18, at every prompt length and most in the
middle lengths and the multi-hop, adequacy and probability families. A reading, not
tested: a state encoded without sight of the question is encoded generically, and that
transfers to JevBench's unfamiliar questions better than e1a's question-shaped encoding,
which the training families had rewarded. The rule's three misses are each a clause of an
efficiency or tolerance prediction: 785 against 700 ms, +11% against 10%, and one set
outside 0.03. None is in the direction the preregistration feared (accuracy lost to the
mask). For context, recorded after this run: b2, b1's encoder-decoder recipe on
T5Gemma 2 4B+4B, scored 178 (PREREGISTRATION-b2.md). That is the best bidirectional result,
at about four times e1b's torso size.
