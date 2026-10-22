---
name: palettization-aware-finetune
description: Plan a quantization-aware fine-tune for Core AI palettization below four bits, including straight-through LUT fake quantization, k-means initialization, zero LUT weight decay, teacher distillation, multilingual replay, automatic stop gates, checkpoint/resume, and palette export. Use when post-training compression misses the quality or size target.
---

# Purpose

Give a below-four-bit experiment a falsifiable plan before renting compute. This skill describes the current plan/in-progress direction; it does not claim that the planned run has succeeded.

## Use this when

- Post-training four-bit compression loses quality on important languages/domains.
- A 2-bit FFN or similar recipe is required for the package budget.
- The team can train on a CUDA machine and re-export the learned palettes.

## Do not use this when

- The fp16 baseline is not verified.
- The training data mix, license, or compute budget is unknown.
- A six-bit or mixed recipe already meets the declared gates.

## Prerequisites

- A frozen original checkpoint and a validated ANE-authored export path.
- Public, licensed train/dev manifests and a held-out evaluation harness.
- A CUDA environment with resumable checkpoints.
- A projected runtime and cost estimate from a pilot.

## Status

This is our current plan and is in progress. Results below the four-bit target are not available in the supplied evidence. Keep all future numbers labeled as pilot, projected, or measured.

## Procedure

1. Freeze the baseline and the target.

   Record fp16, six-bit, mixed, and post-training four-bit quality by language/domain, asset size, placement, latency, and memory. Define the target size and the maximum allowed regression before training.

2. Choose the fake-quantized parameterization.

   Use grouped LUT fake quantization with a straight-through estimator (STE). Initialize each one-dimensional LUT with deterministic k-means over the group's weights, not a min/max linspace. Keep LUT parameters at zero weight decay so optimization does not shrink the learned palette.

   ```text
   q(w) = scale[group] * LUT[index(w)]
   forward: use q(w)
   backward: pass the gradient to w with STE; update LUT/scale normally
   ```

3. Assign precision by sensitivity.

   The current plan uses 2-bit grouped LUTs for FFN linears, 4-bit for attention and convolution, 4-bit for the first and last two encoder layers, and leaves depthwise convolution out because it is small and folded with normalization during export. Re-measure before treating this assignment as final.

4. Add distillation from the frozen original.

   Keep the original model frozen. Combine logit KL on the joint/decoder output with encoder feature loss at selected layers. Use the same masks and valid-frame boundaries in teacher and student.

   ```text
   loss = task_loss + lambda_logits * KL(student || frozen_teacher)
        + lambda_features * feature_loss(student, frozen_teacher)
   ```

   Record the exact temperature, layers, weights, and logits mask.

5. Build a replay-balanced data mix.

   Replay every language/domain that the release claims. Include meetings, code-switching, multilingual speech, and clean speech in proportions justified by the product. Keep evaluation splits disjoint. Verify each dataset license for commercial use before training.

6. Add automatic stop gates.

   Validate every fixed interval on meeting/dev, code-switch/dev, and a multilingual replay slice. Stop if the replay WER regresses beyond the declared gate or if the real target domain does not improve. Do not continue a rented run because a training loss is falling.

7. Run a pilot before committing GPU time.

   Measure steps per second, audio-hours per hour, data-build time, projected wall time, and projected cost. Stop if the projection exceeds the approved cap or the smoothed loss check fails. The supplied plan used a short pilot, a capped full run, and a projected-cost stop; preserve that structure without copying private budgets into a public release.

8. Checkpoint and resume.

   Write a wall-clock checkpoint, a resumable `last.ckpt`, optimizer state, RNG state, config, manifest fingerprints, and the current LUTs. Test resume from a copied checkpoint before starting the long run.

9. Export learned palettes.

   At clean termination, export plain weights and learned LUTs. Reconstruct palette weights in a separate export step, then run the existing ANE re-authoring and compression path. The export must be deterministic from checkpoint plus manifest/config metadata.

10. Re-run the complete ladder.

   Compare original fp16, post-training four-bit, QAT candidate, six-bit, and mixed controls on the same full harness. Re-check placement, layer parity, per-language WER, long-form joins, first load, warm latency, process footprint, neural footprint, and package size.

11. Publish only measured outcomes.

   If QAT does not beat post-training compression, say so. If only a subset of languages improves, restrict the claim. The plan itself is not a result.

## Decision points

- If the QAT pilot cannot project under the size/quality target, stop and ship six-bit or mixed precision.
- If feature loss improves encoder parity but not WER, tune the distillation target or host decoder, not only the palette.
- If a layer is highly sensitive, raise its bit width rather than applying a global precision increase.
- If a dataset license cannot be carried into training, remove it and adjust the claim.

## Verification gates

- The frozen teacher matches the existing reference.
- Fake quantization matches exported palette reconstruction on a toy layer.
- LUT initialization, gradients, and zero weight decay are tested.
- Resume produces the same next-step result for a fixed seed.
- Pilot projection is recorded before the full run.
- Full post-training/QAT ablation and all placement gates pass.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| LUTs collapse to one value | Weight decay or bad initialization | Zero LUT decay and use grouped k-means initialization. |
| Training loss improves; language WER worsens | Missing replay or overfit domain | Add balanced replay and automatic per-language gates. |
| Export differs from training | Palette reconstruction or scale mismatch | Round-trip one layer and compare weights/outputs. |
| Resume drifts | RNG/optimizer/manifest state not saved | Save and test the complete checkpoint state. |
| GPU budget is exceeded | No pilot projection | Stop before renting more time; use a higher-bit fallback. |

## References

- [`references/qat-plan.md`](references/qat-plan.md) contains the public plan template and ablation table.
- Continue with [`coreai-model-release`](../coreai-model-release/SKILL.md) only after a measured candidate exists.

## Evidence notes

Do not put a projected runtime, projected cost, or pilot curve in the final model card as if it were a measured result. Store the training config, manifest fingerprints, checkpoint metadata, LUT histograms, validation rows, and export checksum with every candidate. Keep a higher-bit fallback ready so a failed QAT run does not force a lower-quality release.

The final comparison must use the same host front end, decoder, masks, window router, and scoring normalizer as the post-training controls. Otherwise a QAT gain may be an evaluation-path change rather than a learned-palette gain.

Required candidate record:

```text
checkpoint and parent checksum
palette configuration
training data fingerprint
validation gates
export checksum
placement result
full-harness result
```

The planned run is not a substitute for a completed measurement. Keep the words “planned”, “pilot”, “projected”, and “measured” distinct in notes, charts, and release metadata.

If the QAT plan changes the data mix, optimizer, or palette grouping, start a new candidate record rather than comparing it to the old plan as if only the bit width changed.

## Done when

- A measured QAT candidate either passes the full gates or is rejected with evidence.
- Learned palettes round-trip into the ANE export deterministically.
- The release card separates current plan, pilot data, and final results.

