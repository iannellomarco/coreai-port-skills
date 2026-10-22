---
name: ane-reauthoring
description: Re-author a PyTorch model for full Apple Neural Engine placement and prove it with compiled-region counts, GPU-time controls, neural footprint, and layer-by-layer parity. Use when an ordinary Core AI export falls back to GPU, when ANE eligibility matters, or when masks, attention, layout, or fp16 range need surgery.
---

# Purpose

Change the graph to match the ANE's useful tensor patterns, then verify the change independently. Treat “the export succeeded” and “the ANE ran” as separate gates.

## Use this when

- A direct export uses GPU regions or has no placement evidence.
- A model contains linear projections, dynamic masks, relative attention, or fp16 range problems.
- A fixed-window host router can replace unsupported dynamic control flow.

## Do not use this when

- The existing graph is already all-ANE and parity is complete. Use the export and runtime skills.
- The only problem is model size. Use [`coreai-compression`](../coreai-compression/SKILL.md) after the placement baseline.

## Prerequisites

- A frozen PyTorch/Hugging Face reference and representative inputs.
- A layer-output capture path or hooks in the reference.
- A Core AI export canary, a GPU positive control, and a CPU numerical control.
- macOS tooling that exposes compiled graph artifacts or region diagnostics.

## Procedure

1. Freeze the reference.

   Save inputs, masks, intermediate tensors, final outputs, and the exact dtype. Run the reference twice and assert determinism before rewriting.

2. Adopt BC1S for convolution-friendly graphs.

   Use `(B, C, 1, S)` for channel-first sequence data. Move time to `S`, keep frequency or feature axes explicit, and make masks broadcastable over `(B, C, 1, S)`.

   ```python
   # sequence linear: [B, C_in, 1, S] -> [B, C_out, 1, S]
   projection = torch.nn.Conv2d(C_in, C_out, kernel_size=1, bias=True)
   y = projection(x)
   ```

   Copy weights from a linear layer into the `1x1` convolution without changing output order. Compare the module alone before composing the block.

3. Replace unsupported dynamic operations with inputs or constants.

   Pass validity masks as graph inputs. Precompute relative-position tables, sine/cosine tables, and fixed window metadata. Use finite additive mask values rather than relying on an infinite sentinel when the reference and target disagree about softmax edge cases.

4. Re-author relative attention without dynamic skew/gather.

   Prefer fixed per-window tables and a form made from `1x1` projections, elementwise operations, and per-head matmul. For a relative term, express `sin(q-k)` and `cos(q-k)` through angle addition so the graph does not need a dynamic relative shift.

5. Make fp16 range safe.

   Audit residual magnitudes, variance squares, softmax inputs, reciprocal operations, and output projections. If a norm squares values outside fp16 range, rescale by a per-row or power-of-two factor before the square and undo it analytically. Validate the transformed equation in fp32 first.

   ```python
   m = x.abs().amax(dim=channel_axis, keepdim=True).clamp_min(eps)
   u = x / m
   norm = u * torch.rsqrt((u * u).mean(dim=channel_axis, keepdim=True) + (eps / m) ** 2)
   ```

   Do not keep the division if the ANE rejects it; use an equivalent reciprocal construction only after numeric comparison.

6. Use fixed windows and host routing for long sequences.

   Export functions for the shortest useful windows. Route each input to the smallest function that fits. For long audio, cut at a quiet point inside a bounded look-back region, carry valid-frame masks, and decode each piece on the host. State clearly that this is host chunking, not streaming, unless a cache-aware graph has been implemented.

7. Export one fp16 ANE candidate before compression.

   Keep the authoring and compression changes separate. A compressed graph that fails placement is not an ANE result.

8. Prove placement in three ways.

   - Count ANE and GPU regions in the compiled graph. Use [`scripts/region_census.py`](scripts/region_census.py) as a starting point and inspect raw names when patterns do not match.
   - Measure accumulated GPU time with `ioreg -c AGXDeviceUserClient` around repeated calls. A positive GPU-forced control must move the counter; the ANE candidate should remain at or near zero.
   - Read `proc_pid_rusage` with `RUSAGE_INFO_V6` and record `ri_neural_footprint`. A positive neural footprint is supporting evidence, not a region count.

   ```bash
   python scripts/region_census.py <specialized-mpsgraph> --expect-ane 1 --expect-gpu 0
   ioreg -c AGXDeviceUserClient | rg 'accumulatedGPUTime'
   ```

9. Compare parity layer by layer.

   Use identical inputs and compare after preprocessing, subsampling, each normalization, each attention/conv/FFN block, the projection, and the final output. Report max absolute difference, cosine, PSNR, and any NaN/Inf. Then compare greedy decoded identity or task metric.

10. Re-run placement after every material graph or compression change.

   A later change can split a region or silently move the whole short bucket to GPU. Keep the old fp16 graph and a forced-GPU graph as controls.

## Decision points

- If one op prevents ANE placement, rewrite the op or isolate it on the host only when the boundary cost is measured.
- If a graph is all-ANE but slower than GPU for short inputs, keep the ANE route only if its CPU/GPU power or concurrency benefit is real.
- If quantization breaks ANE conversion, do not claim “quantized ANE”; first re-author the quantized op pattern or ship fp16.
- If long-form accuracy depends on a hand-tuned cut, hold out a separate long-form set before publishing the cut policy.

## Verification gates

- Module-level Linear→Conv2d parity passes.
- Full fp16 graph has expected ANE/GPU region counts.
- GPU counter moves in the positive control and not in the ANE candidate.
- Neural footprint is positive for the ANE candidate.
- Every captured layer has finite values and a written threshold.
- Host window routing and long-audio joins pass the end-to-end harness.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Zero ANE regions | Unsupported op, shape, or dtype | Inspect the compile diagnostic, rewrite the op, and recompile. |
| GPU counter rises on the ANE run | Mixed or silently fallen-back graph | Count regions and split the graph until the boundary is understood. |
| PSNR collapses after a norm | fp16 square overflow or changed reduction order | Rescale analytically and compare the norm in isolation. |
| Short clips work; long clips fail | Padding/mask or fixed-window join error | Use explicit validity masks and quiet cuts; score long-form separately. |
| Census says zero but runtime is busy | Brittle function/region-name matcher | Inspect raw compiled names; do not trust a regex-only zero. |

## References

- [`references/authoring-patterns.md`](references/authoring-patterns.md) gives the tensor layouts, parity ladder, and placement evidence table.
- [`scripts/region_census.py`](scripts/region_census.py) is a read-only census helper.

## Evidence notes

Keep four artifacts for each candidate:

1. A graph census with the exact function names matched.
2. A GPU-counter log for the candidate and a forced-GPU control.
3. A neural-footprint reading with process and cache state.
4. A parity JSON containing stage names, shapes, thresholds, and failures.

If a region name does not match the helper pattern, preserve the raw directory listing and revise the pattern before counting. If the cache was not proven cold, write `cache: unknown` rather than calling the compile measurement cold.

Use a new candidate tag for every graph rewrite. Never overwrite the fp16 control before its placement and parity artifacts are copied into the run directory. That control is the fastest way to tell an ANE authoring regression from a compression regression.

Required run output:

```text
candidate asset and function
reference fixture identifier
compiled region counts
GPU-control counter delta
neural footprint
stage parity summary
end-to-end metric
```

## Done when

- The graph is proven all-ANE for the tested functions, or the mixed placement is explicitly documented.
- GPU and neural controls agree with the region census.
- Layer parity and the end-to-end metric pass on held-out inputs.

