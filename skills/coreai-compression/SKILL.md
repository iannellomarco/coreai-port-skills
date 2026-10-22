---
name: coreai-compression
description: Evaluate Core AI weight compression with palettization, int8, mixed precision, and explicit size math while preserving accuracy and placement evidence. Use whenever an agent considers four-bit or lower weights, pal4g8s, per-channel scales, mixed-bit recipes, or a multilingual compression claim.
---

# Purpose

Find the smallest model that still meets the quality, placement, latency, and memory gates. Compression is an evaluation program, not a single export flag.

## Use this when

- A working fp16 Core AI baseline exists.
- The bundle is too large or load/memory cost matters.
- A model serves multiple languages, domains, or input lengths.

## Do not use this when

- The fp16 graph has not passed parity or placement.
- The requested target is below four bits but there is no QAT plan. Use [`palettization-aware-finetune`](../palettization-aware-finetune/SKILL.md).

## Prerequisites

- A frozen fp16 reference and placement-verified graph.
- A corpus-level evaluator with per-language/domain rows.
- Weight names grouped by sensitivity: FFN, attention, convolution, depthwise, projection, embeddings, and small layers.
- A size-report script that distinguishes MiB from decimal MB.

## Procedure

1. Record the fp16 baseline.

   Save package size, model-part sizes, warm latency by input bucket, first load, process `phys_footprint`, neural footprint, placement, and task metrics. This is the comparison point for every ablation.

2. Establish compression semantics.

   - Palettization stores a lookup table and indices. `pal4g8s` means a 4-bit palette, groups of 8 output channels, with a per-channel scale in the tested recipe.
   - Int8 or int4 uses a linear quantizer. `g32` commonly means a per-block group of 32 input values; verify the actual exporter configuration.
   - Mixed precision assigns bits by module class, not by a single headline bit width.

3. Keep sensitive small layers in higher precision.

   Do not compress depthwise convolutions or tiny layers merely to make a table look uniform. Their byte savings are small, and the graph or parity cost can be disproportionate. Test attention/convolution at six or eight bits before forcing four bits.

4. Run the ladder.

   Use the same data, seeds, host runtime, and placement check for:

   ```text
   fp16 -> int8/per-channel -> 6-bit palette -> mixed 4/6/8-bit -> pal4g8s -> lower-bit candidate
   ```

   For an ANE-authored graph, keep compression after the re-authoring step and re-run the region census. A quantized graph that falls to GPU is a different product.

5. Compute size correctly.

   ```python
   bytes_total = sum(p.stat().st_size for p in asset.rglob("*") if p.is_file())
   print(bytes_total / 2**20, "MiB", bytes_total / 1_000_000, "MB")
   ```

   Report encoder, decoder/joint, tokenizer/processor, runtime/tests, total package, and compiled-cache disk separately. Never compare a decimal download number with a binary model number without labeling units.

6. Evaluate every claimed language and domain.

   A small Italian/English set is not evidence for 25 languages. The measured Parakeet ablation found post-training four-bit regressions on less-represented FLEURS languages while Italian/English looked acceptable. The available rows showed roughly +1.0 to +1.8 WER points for Bulgarian, Czech, and Danish, smaller change for Dutch, and little change for English. The broader release rule is to measure Bulgarian, Hungarian, Greek, Czech, Danish, Dutch, Finnish, and every other claimed language when the data exists.

7. Use ablations to locate loss.

   Compare fp16, six-bit, mixed precision, and four-bit on the same manifest. If six-bit recovers the gap, leave sensitive classes at six bits or use QAT. If only one module class causes loss, isolate that class instead of raising every layer.

8. Compare with confidence, not a single mean.

   Report corpus WER, language/domain rows, reference word counts, bootstrap intervals or paired tests where appropriate, and the number of runs. A one-run result may be deterministic but still have sampling uncertainty.

9. Re-test host and Swift runtimes.

   Compression can change encoder hidden states enough to change CPU decoding, window joins, or text identity. Score the actual shipped runtime, not only a Python graph call.

10. Publish the largest regression.

   State the worst language/domain and the bit recipe that caused it. If the result is not acceptable, keep the fp16 or six-bit candidate and stop the sweep.

## Decision points

- If fp16 is within the quality target and already fits, prefer fp16 on ANE over a smaller GPU fallback.
- If four-bit works on high-resource languages but not the long tail, use mixed precision or QAT; do not average the loss away.
- If shared static functions remove duplicate weights, do that before lower-bit compression.
- If the decoder/joint dominates the remaining package, compress it separately and re-run the runtime parity gate.

## Verification gates

- fp16 placement and parity are recorded.
- Every compression candidate has exact size units and a manifest hash.
- The candidate's compiled region census matches the product claim.
- Per-language/domain quality and intervals are published.
- Warm, cold, process, neural, and cache measurements use the same protocol.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Four-bit is small but slower | It fell to GPU or added host work | Prove placement and separate graph time from decode time. |
| Mean WER is fine; one language regresses | Aggregate hides long-tail loss | Add language rows and stop gates. |
| Size numbers disagree | MiB and MB were mixed | Report both, with formulas. |
| A quantized asset is not ANE | Quantized op pattern is unsupported | Re-author the op or choose higher precision. |
| Lower-bit candidate is all blank/garbage | Calibration or range failure | Gate hidden-state PSNR and finite values before WER. |

## References

- [`references/compression-matrix.md`](references/compression-matrix.md) gives recipes and the measured ASR interpretation.
- Continue with [`palettization-aware-finetune`](../palettization-aware-finetune/SKILL.md) for below-four-bit work.

## Evidence notes

The available multilingual ablation gives a concrete warning. On its FLEURS rows, `pal4g8s` was about +1.42 WER points on Bulgarian, +1.04 on Czech, +1.58 on Danish, +0.46 on Dutch, and +0.03 on English against its fp16 control. The six-bit `pal6g16` control brought those same rows back to within roughly 0.2 points of fp16. These are the measured rows in the source snapshot, not a license to extrapolate to Hungarian, Greek, or Finnish.

Keep the full language table even when the release chooses a high-resource headline. Compression can preserve an average while changing which users are affected.

Use the same seed and data ordering for a compression ladder, but do not confuse deterministic decoding with statistical certainty. The harness can reproduce a run exactly while the held-out sample is still too small for a strong claim.

Keep rejected candidates in the experiment table with a reason such as placement failure, quality regression, or size not worth the cost. A missing row makes later decisions look more arbitrary than they were.

If a compression library changes, rerun at least one fp16 and one compressed control. A new exporter can change placement or scale semantics even when the recipe name is unchanged.

Never claim that a palette is “lossless” from a cosine or token identity result alone. Weight reconstruction, hidden states, and task metrics answer different questions.

## Done when

- The selected recipe is the smallest one that passes all declared gates.
- The public result names the worst regression and unmeasured claims.
- Placement, package units, and runtime measurements are reproducible.

