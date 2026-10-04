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
