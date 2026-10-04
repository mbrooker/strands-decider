# Pre-registration: e2, the Ettin encoder 1B (speed option)

**Status: written, not scheduled.** It runs after e1 reports
([PREREGISTRATION-e1.md](PREREGISTRATION-e1.md)), with e1's winning arm (unmasked or masked
state cache). The arm, and any change to the numbers below that e1's outcome makes
necessary, are fixed in a dated amendment before e2 trains. Nothing below the amendment is
edited after training. The design is in [docs/encoder-design.md](../../docs/encoder-design.md#e2-the-ettin-encoder-1b).

## Why

The goals of the encoder direction are latency and multi-question cost, with JevBench no
worse than b1's 153. In the encoder Phase 0, `jhu-clsp/ettin-encoder-1b` is level with b1's
torso on the reasoning slices, frozen: reasoning mean 0.502 against 0.507 to 0.512. It is
the fastest candidate that is: its torso reads a 256-token question in 28 ms, against 130 ms
for b1's torso and 43 ms for e1's. It is also the one MLM encoder whose untrained cloze
readout answers above chance (held-out 0.421, generated documents 0.454), so it has a
pretrained answer position. If e1 shows the recipe can lift an encoder well past its frozen
ranking, e2 tests how far a 1B, natively bidirectional encoder gets for a third of e1's
latency.

## What is being run

e1's recipe and readout, with the torso changed:

| | e1 | e2 |
| --- | --- | --- |
| torso | T5Gemma 2B-it UL2 encoder, 2.6B | Ettin encoder 1B, ModernBERT architecture, 1.0B |
| attention | as e1's winning arm | same arm; local window 128 with global every third layer, as pretrained |
| LoRA r=16 | the seven Gemma projections | ModernBERT's projections (`Wqkv`, `Wo`, the MLP's `Wi`, `Wo`), checked against the loaded module |
| query / keys / head | mean / option line's last token / `xpointer` | same; head initialised at random |
| rows, teacher, schedule | g4's, as b1 and e1 | same |

## Predictions (to be confirmed or amended once e1 has reported)

1. **JevBench at least 150:** b1's 153 within noise.
2. **Fast:** JevBench median latency at most 80 ms on this card (v19 115, b1 216), and, with
   e1's winning arm, eight questions on a 4,000-token state below e1's.
3. **Order sensitivity:** TV at most 0.044.
4. **Calibration:** JevBench ECE at most 0.07, held-out ECE at most 0.074.

Exploratory: JF100, Typed Decisions, the sets e1 reports, and the gap to e1 per family.

## What would count as failure

- **JevBench under 145.** At 1B, a natively bidirectional MLM encoder does not carry this
  recipe's reasoning families, whatever its speed.

## What this test cannot show

- **MLM against LM-derived encoders in general.** One model of each kind, at different sizes.
- **Whether a cloze-initialised head would do better.** Not part of e2. It is the named
  follow-up if e2 is close to its bar.
