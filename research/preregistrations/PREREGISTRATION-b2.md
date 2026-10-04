# Pre-registration: b2, b1's recipe on T5Gemma 2 4B+4B

Committed before training. Same discipline as b1 and e1: predictions fixed here, the
outcome appended below without editing anything above it.

## Why

b1 (T5Gemma 2 1B+1B on g4's recipe) scored **153/231** on JevBench against v19's 168 and
g4's 183. The architecture's own claims held: order sensitivity fell to two-fifths of
v19's (TV 0.035, argmax flips 5.2% against 15.6%), and JevBench ECE was the best in the
series (0.044). Judgement and reasoning failed: HotpotQA 0.605, adequacy 0.590 / 0.693,
both generated sets below their floors. The deficit did not grow with prompt length, and
on adequacy b1 called adequate answers inadequate (HelpSteer2 adequate-class accuracy
0.444 against v19's 0.684). b1's outcome reads this as **torso capacity**: 1B+1B is
adapted from Gemma 3 1B, weaker at reasoning than Qwen3.5-2B or Gemma 4 E2B, and a
bidirectional encoder adds no reasoning the base lacks. That is a reading, not a
measurement. b2 tests it by changing only the torso's size.

e1a (an encoder-only T5Gemma 2B-it UL2) also scored 153, with every evaluation set above
its floor. In-distribution sets did not predict JevBench there, so b2's primary measure
is JevBench, as b1's was.

## What is being changed

Only `base_model` (`configs/experiments/b2.yaml` against `b1.yaml`):

| | b1 | b2 |
| --- | --- | --- |
| torso | T5Gemma 2 1B+1B (Gemma 3 1B) | **T5Gemma 2 4B+4B** (Gemma 3 4B), revision `487d4ac` |
| text parameters after dropping vision | 1.70B | about 7.1B (estimate: 0.67B shared embedding, about 3.2B per stack) |
| head, query, keys, LoRA r=16 targets, rows, teacher file, schedule, micro-batch 16 x 2, compiled layers | | unchanged |
| hardware | RTX 3090, WSL2 | **one g7e.2xlarge** (RTX PRO 6000, 96 GB), us-east-2, the DLAMI and `training/aws/image/setup-host.sh` |

The model is staged from the local HF cache through S3 and loaded offline on the host
(`HF_HUB_OFFLINE=1`). Training, calibration, the evaluation sets, order sensitivity and
JevBench all run on the host. The checkpoint and reports are copied back before the host
terminates.

Estimates, not measurements: peak memory 50 to 55 GiB at micro-batch 16; about 3 h for
3,738 steps.

## Baselines

The v19 figures are its reruns on the 3090 (b1's outcome), b1's are its own, and g4's are
from PREREGISTRATION-g4.md.

| evaluation | v19 | g4 | b1 |
| --- | --- | --- | --- |
| JevBench (231) at 4096: tasks / Brier / ECE | 168 / 0.342 / 0.051 | 183 / 0.292 / 0.048 | 153 / 0.411 / 0.044 |
| held-out short tasks: accuracy / ECE | 0.647 / 0.054 | 0.655 / 0.065 | 0.671 / 0.065 |
| MuSiQue / ContractNLI / BoardgameQA / HotpotQA (held out) | 0.882 / 0.873 / 0.820 / 0.719 | 0.902 / 0.861 / 0.781 / 0.759 | 0.923 / 0.843 / 0.748 / 0.605 |
| generated documents, v16's / v18's | 0.857 / 0.777 | 0.863 / 0.794 | 0.797 / 0.741 |
| adequacy: HelpSteer2 / generated (balanced) | 0.722 / 0.788 | 0.722 / 0.801 | 0.590 / 0.693 |
| HelpSteer2 adequacy, adequate-class accuracy | 0.684 | | 0.444 |
| order sensitivity, held-out, reversed: TV / flips | 0.088 / 0.156 | | 0.035 / 0.052 |

Noise: two runs of one recipe differ by about 4.5 JevBench tasks.

## Predictions

1. **JevBench at least 168** (v19) at the 4096 window.
2. **Reasoning recovers to v19's level**, each within 0.02 of v19's rerun (e1a's floors):
   HotpotQA at least 0.699, BoardgameQA at least 0.800, generated at least 0.837 and
   0.757, adequacy at least 0.702 (HelpSteer2) and 0.768 (generated, balanced). Also
   MuSiQue at least 0.862, ContractNLI at least 0.853, held-out at least 0.627.
3. **The adequacy bias closes:** adequate-class accuracy on HelpSteer2 at least 0.60
   (b1 0.444, v19 0.684).
4. **The architecture's gains hold at size:** order TV at most 0.044, argmax flips below
   v19's 0.156.
5. **Calibration holds:** JevBench ECE at most 0.07, held-out ECE at most 0.074.

Exploratory, no numeric prediction: JevBench by tier and family, paired against v19 and
b1; latency on the host and on the 3090 once the checkpoint is home (b1's latency is
from the 3090, so the host's numbers are not comparable to it); training memory and
time against the estimates above.

## Which model is recommended afterwards

If 1, 2 and 5 hold, b2 becomes hobson-bidi's reference model, and capacity, not the
encoder-decoder design, explains b1. If 1 also clears g4's 183 beyond noise (188 or
more), the bidirectional torso at this size is the strongest decider here, and serving
cost becomes the question: 4B+4B with no prefix cache, against e1b's cached encoder.

## What would count as failure

- **JevBench at most 158** (b1 plus noise). Four times the torso did not move the
  benchmark, so capacity is not the explanation. The limit is in T5Gemma 2 itself, or in
  how this recipe reads it, and the bidirectional direction should stop at the
  encoder-only speed options.
- **Prediction 2 fails broadly** (more than three floors) while 1 holds. Then JevBench
  rose for other reasons, and the reasoning sets show the same gap at both sizes.

## What this test cannot show

- **Size and base quality together.** Gemma 3 4B is larger and was trained differently
  from Gemma 3 1B. b2 cannot say which matters.
- **Hardware.** b1 trained and evaluated on a 3090, b2 on an RTX PRO 6000. bf16 kernels
  differ, and accuracy differences under about 0.01 on a set are not interpretable as the
  torso's.
- **One run**, one seed.
- **Licence.** As b1: Gemma Terms of Use.

## Outcome (added after the run)

Nothing above this section was edited after training. **By the rule fixed above, b2 does
not become hobson-bidi's reference model.** Predictions 1, 3, 4 and 5 held. Prediction 2
missed one floor of nine, generated v16's, by 0.006. Neither failure condition is met.
**The capacity reading of b1 is supported:** the same recipe on the 4B+4B torso moved
JevBench from 153 to 178.

The run: commit `10f163c`, one g7e.2xlarge (RTX PRO 6000 Blackwell, 96 GB) in us-east-2,
4 October 2026. `training/aws/image/setup-host.sh` installs torch 2.7.1+cu126, which has no
sm_120 kernels, so every CUDA call failed. It was replaced by the same version's cu128
build before anything ran; transformers 5.17.0 and peft 0.21.0 were unchanged. Training
took 2 h 43 m (11:59 to 14:42 UTC), 3,738 steps at 0.38 to 0.40 steps/s. Peak allocation
39.0 GiB (estimate 50 to 55), at most 41.3 GiB in use. Final validation loss / accuracy
0.373 / 0.839, the lowest loss recorded on these rows (b1 0.407, v19 0.421, g4 0.427).
Calibration temperatures: choice 1.192, yes/no 1.479, score 1.328. The checkpoint and
reports are in `~/hobson-bidi/checkpoints/bidi-b2/` and `~/hobson-bidi/reports/b2/`, and in
`s3://hobson-v17-195880352761-us-west-2/runs/b2/`.

| | prediction | v19 | g4 | b1 | b2 | |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | JevBench >= 168 | 168 | 183 | 153 | **178** | pass |
| 2 | nine floors within 0.02 of v19's rerun | | | | eight held; generated v16 0.831 against 0.837 | FAIL (one floor) |
| 3 | HelpSteer2 adequate-class accuracy >= 0.60 | 0.684 | | 0.444 | **0.675** | pass |
| 4 | order TV <= 0.044, flips below 0.156 | 0.088 / 0.156 | | 0.035 / 0.052 | **0.035 / 0.060** | pass |
| 5 | JevBench ECE <= 0.07; held-out ECE <= 0.074 | 0.051; 0.054 | 0.048; 0.065 | 0.044; 0.065 | **0.062; 0.048** | pass |

**JevBench: 178/231** (easy 48, standard 65, hard 65). Paired against b1: 38 gained, 13
lost, **p = 0.0006**. Against the recorded v19 run (167 at 3,072): 25 gained, 14 lost,
p = 0.11. Brier **0.287**, the best recorded here (g4 0.292, v19 0.342, b1 0.411).
Paraphrase consistency 0.917. Latency on the RTX PRO 6000: median 95 ms, p95 214 ms,
not comparable with the 3090 figures. Families at 1.0: adversarial, extraction, fact,
intent, ordinal, routing, tool_selection. Weakest: temporal_numeric 0.33 (b1 0.20),
multi_hop 0.44 (b1 0.33), adequacy 0.50 (b1 0.42), probability 0.50 (b1 0.40).

**Sets** (floor in brackets):

| set | v19 (rerun) | g4 | b1 | b2 | |
| --- | --- | --- | --- | --- | --- |
| held-out short tasks [0.627] | 0.647 | 0.655 | 0.671 | **0.685** | pass |
| MuSiQue [0.862] | 0.882 | 0.902 | 0.923 | **0.950** | pass |
| ContractNLI [0.853] | 0.873 | 0.861 | 0.843 | 0.860 | pass |
| BoardgameQA [0.800] | 0.820 | 0.781 | 0.748 | **0.880** | pass |
| HotpotQA, held out [0.699] | 0.719 | 0.759 | 0.605 | 0.748 | pass |
| generated, v16's [0.837] | 0.857 | 0.863 | 0.797 | 0.831 | FAIL |
| generated, v18's [0.757] | 0.777 | 0.794 | 0.741 | **0.818** | pass |
| adequacy, HelpSteer2 [0.702] | 0.722 | 0.722 | 0.590 | **0.731** | pass |
| adequacy, generated, balanced [0.768] | 0.788 | 0.801 | 0.693 | **0.816** | pass |

The adequacy bias is gone. HelpSteer2 by gold label, inadequate / adequate: v19 0.761 /
0.684, b1 0.735 / 0.444, b2 0.786 / 0.675. Generated: b2 0.773 / 0.860.

**Order sensitivity** (3,000 held-out rows, reversed): TV 0.035, argmax flips 0.060 (choice
0.039, yes/no 0.024). The bidirectional torso's order result holds at four times the size.

**Reading.** b1's failure was capacity. With nothing changed but the torso, every
reasoning and adequacy set that b1 lost came back to or above v19, and BoardgameQA (0.880),
MuSiQue (0.950), HotpotQA (0.748) and both adequacy sets are the best recorded in any of
these forks. JevBench rose 25 tasks over b1 and sits 11 above v19 and 5 below g4, inside
the noise of g4 (about 4.5 tasks between runs). The miss on generated v16's set is 0.006,
two rows of 350. Against g4, b2 trades JevBench count for calibration (Brier 0.287 against
0.292) and order stability, with a larger torso (about 7.1B text parameters) and no
prefix cache. That trade, and serving cost at 7B, are the next question, not torso capacity.

Not tested here: latency on the 3090 (it fits for inference); a seed replicate; whether
the generated-v16 miss is noise.
