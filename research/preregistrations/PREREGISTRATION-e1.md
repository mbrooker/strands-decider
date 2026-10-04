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

## Outcome: e1b (added after the run)
