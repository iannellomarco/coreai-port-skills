---
name: coreai-export-basics
description: Export a PyTorch or Hugging Face model to Apple Core AI using coreai-torch recipes, static or dynamic shapes, shared multi-function assets, specialization options, and Python or Swift runtime calls. Use whenever an agent needs a first `.aimodel` export or a correct load/call path.
---

# Purpose

Produce a small, inspectable `.aimodel` bundle before doing architecture surgery. Keep shape decisions, function names, metadata, specialization choices, and cache behavior explicit.

## Use this when

- An upstream Core AI recipe exists or the model is a mostly stateless PyTorch graph.
- You need fixed buckets, a dynamic axis, or several functions that share weights.
- You need Python and Swift examples for loading and calling functions.

## Do not use this when

- The graph already failed ANE conversion and you need to rewrite it. Use [`ane-reauthoring`](../ane-reauthoring/SKILL.md).
- The model is an autoregressive LLM whose cache and host loop are not designed yet. Start with [`coreai-llm-port`](../coreai-llm-port/SKILL.md).

## Prerequisites

- macOS 27 / Xcode 27 beta-era tooling as applicable; APIs may move.
- A pinned `coreai-torch`, `coreai-core`, PyTorch, and transformers environment.
- A reference module with deterministic example inputs.
- The model card and processor/tokenizer files.

## Procedure

1. Create an isolated export environment and record versions.

   ```bash
   uv venv --python 3.12 .venv
   uv pip install --python .venv/bin/python \
     torch==<validated> coreai-torch==<validated> coreai-core==<validated> \
     transformers==<validated> safetensors
   .venv/bin/python - <<'PY'
   import coreai_torch, torch
   print(torch.__version__, coreai_torch.__version__)
   PY
   ```

   Do not copy a pin blindly from another release. Install it in a fresh environment and run the recipe canary.

2. Make the module exportable.

   Keep `forward` tensor-only. Replace Python-side mutation with explicit inputs and outputs. Use `torch.export.export`, then the recipe's decomposition table and `TorchConverter`.

   ```python
   exported = torch.export.export(module.eval(), args=(), kwargs=example_inputs,
                                  dynamic_shapes=dynamic_shapes)
   exported = exported.run_decompositions(coreai_torch.get_decomp_table())
   program = (coreai_torch.TorchConverter()
              .add_exported_program(exported, input_names=list(example_inputs),
                                    output_names=["output"], entrypoint_name="main")
              .to_coreai())
   program.save_asset(out / "model.aimodel", metadata)
   ```

3. Choose static or dynamic shape intentionally.

   Use static shapes when the target has a small bounded distribution, deterministic latency matters, and padding can be masked. Use dynamic shapes when the runtime and graph have been verified across the actual lengths, not merely because export accepts a dynamic axis. For retrieval or audio, bucketed static functions are often easier to test and can share weights.

4. For shared functions, export once and specialize after conversion.

   The pattern is one dynamic `main` graph followed by static shape configuration for buckets. Verify the resulting functions and measure the exact bucket set because adding functions changes compile time, cache size, and sometimes latency.

   ```python
   program.set_static_shape_config("main", {
       '"32"': {"input_ids": [1, 32]},
       '"64"': {"input_ids": [1, 64]},
   })
   ```

   Use the recipe's exact key quoting and function naming convention; inspect the asset descriptor rather than assuming names.

5. Include masks for padded inputs.

   A fixed window must receive a validity mask if the reference excludes padding from attention, subsampling, convolution, pooling, or normalization. Compare a short input run with a manually padded reference.

6. Save metadata with the asset.

   Include source model ID/revision, license, shape map, input/output names, precision, compression, exporter versions, and the expected compute request. Metadata is part of the evidence, not decoration.

7. Load once and cache functions.

   Python:

   ```python
   import asyncio, coreai.runtime as rt
   opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(
       rt.ComputeUnitKind.neural_engine())
   model = asyncio.run(rt.AIModel.load("model.aimodel", opts))
   main = model.load_function("main_64")
   out = asyncio.run(main(inputs={"input_ids": ids, "attention_mask": mask}))
   ```

   Swift:

   ```swift
   let options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
   let model = try await AIModel(contentsOf: assetURL, options: options)
   guard let fn = try model.loadFunction(named: "main_64") else { throw Error.missing }
   let output = try await fn.run(inputs: inputs)
   ```

   Load the model and functions once. Do not call `loadFunction` inside a hot loop.

8. Separate cold, warm, and cached measurements.

   “First load” may include compilation and specialization. A new process can hit a per-process cache; a renamed executable or new OS build can miss it. Measure at least cold first load, same-executable warm load, and a new-process load.

9. Add a placement negative control.

   Request ANE, GPU, and CPU separately. The preferred compute unit is a scheduling request, not a placement proof. Count compiled regions and record GPU time and neural footprint.

## Decision points

- If the input-length distribution has a long tail, use multiple fixed functions plus host routing.
- If dynamic shapes pass only on CPU, keep the dynamic graph as a CPU reference and export bounded ANE buckets.
- If several functions repeat the same weights, use one shared asset and verify storage is shared.
- If adding a bucket increases first load or cache disk beyond budget, remove the bucket or move the long tail to host chunking.

## Verification gates

- Export a reduced canary and full graph separately.
- Compare tensor outputs before and after conversion at several shapes.
- Verify function names and descriptor shapes from the asset.
- Compare padded and unpadded outputs.
- Record cold and warm load separately.
- Prove the requested compute unit with an independent placement check.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `torch.export` fails on mutation | Stateful Python object or cache | Make state an explicit tensor input/output. |
| Static bucket output differs | Padding is visible to the graph | Add validity masks and compare frame/token counts. |
| New process is slow again | Cache is per process/executable/build | Publish the cold cost and prewarm in the background. |
| `default()` appears fast but placement is unknown | Default is a scheduling decision | Pin ANE/GPU/CPU for the test and inspect evidence. |
| Bundle loads but call fails | Input names/dtypes/shapes differ | Read descriptors and construct `NDArray` values from them. |

## References

- [`references/export-checklist.md`](references/export-checklist.md) contains a bundle checklist and a static/dynamic comparison.
- Continue with [`coreai-compression`](../coreai-compression/SKILL.md) after the fp16 baseline is green.

## Done when

- A fresh environment exports a canary and the full asset.
- Descriptor shapes, function names, metadata, and host calls are recorded.
- The fp16 reference parity and placement gates pass before compression.

