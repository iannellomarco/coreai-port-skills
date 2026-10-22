---
name: coreai-port-triage
description: Assess a PyTorch or Hugging Face model before exporting it to Apple Core AI. Use whenever an agent must decide whether a model belongs on the Neural Engine, GPU, or CPU, define host/runtime boundaries, check licensing, budget memory, or set a defensible completion gate.
---

# Purpose

Turn a model request into a bounded porting decision. Record what is licensed, what graph can be captured, what must remain host code, which compute unit is the target, and the measurements that can falsify the plan.

## Use this when

- A team has a PyTorch/Hugging Face checkpoint and wants a `.aimodel`.
- A model mixes an exportable tensor graph with autoregressive control flow, caches, tokenization, or audio/image preprocessing.
- A claimed ANE result needs a memory, accuracy, latency, or placement budget.

## Do not use this when

- The task is only to run an already validated bundle.
- The user wants a generic Core ML conversion with no Core AI target.
- Licensing or evaluation data cannot be inspected. Stop and report the missing evidence.

## Prerequisites

- A pinned checkpoint revision and a PyTorch reference command.
- The upstream model card, license, tokenizer/processor files, and any adapted source notices.
- A target device class and OS/SDK build.
- A representative evaluation set, including every language or domain that will be claimed.

## Procedure

1. Capture the identity before changing code.

   ```bash
   python - <<'PY'
   import platform, sys
   print(platform.platform())
   print(sys.version)
   PY
   sha256sum <checkpoint-file>  # use shasum -a 256 on macOS
   ```

   Write the model ID, revision, checksum, exporter versions, OS build, Xcode version, device, and target compute unit to a run manifest. A port without this manifest cannot be reproduced.

2. Read licenses transitively.

   Check the base weights, tokenizer, processor, adapted files, evaluation sets, and third-party runtime code. Separate commercial use from non-commercial use. Preserve attribution, NOTICE files, and original terms for adapted files. Do not assume that the license of the Python package covers the weights.

3. Draw the architecture boundary.

   Classify each stage as:

   - stateless tensor graph: a direct Core AI candidate;
   - autoregressive graph: export the step graph and keep sampling/cache control on the host;
   - encoder plus host loop: export encoder/decoder/joint pieces and keep routing or duration logic on the host;
   - preprocessing/postprocessing: keep it on the host until numeric parity is proven.

4. Estimate the shape and memory envelope.

   Record parameter bytes, state bytes per token or frame, temporary activations, number of static functions, and compiled-cache cost. Measure process `phys_footprint`; do not substitute RSS. For ANE work also record `ri_neural_footprint` or the runtime's equivalent.

5. Decide the compute target explicitly.

   Use the decision table below. A request for “ANE” is not evidence that the graph is ANE-eligible. Plan a GPU or CPU negative control and a placement measurement before claiming ANE execution.

6. Define the reference and the “done” gate.

   The reference is the original PyTorch/Hugging Face path, with its exact tokenizer, processor, normalizer, and decode policy. Define acceptable deltas for quality, latency, first load, warm load, process footprint, neural footprint, and package size. For multilingual ASR, use corpus WER per language and domain, not a single aggregate.

7. Run a small canary before a full export.

   Export a reduced model or short input first. Compare one layer or one output against the reference. Only then scale the export and compression sweep.

8. Record the decision as one of:

   - **Port now:** graph shape, license, budget, and gates are clear.
   - **Port with host loop:** the graph is useful but control flow stays in Swift/Python.
   - **Prototype only:** a key placement, memory, or parity gate is unverified.
   - **Do not port:** the host runtime already wins, or the Core AI copy/wiring cost defeats the use case.

## Decision table

| Model shape | Core AI boundary | Preferred target | Main risk | First gate |
|---|---|---|---|---|
| Fixed-shape encoder | One or more static functions | ANE if the compiled graph is all ANE | Padding and bucket routing | Padded-vs-unpadded parity |
| Encoder with dynamic length | Dynamic graph or fixed buckets | ANE only after placement proof; GPU/CPU control | Shape specialization and cache cost | Several lengths, cold and warm |
| Autoregressive LLM | Prefill and one-token/decode functions with explicit state | GPU or ANE by measured partition; host sampling | KV state, tokenizer, memory | Teacher-forced logits and chunk consistency |
| Transducer / TDT | Encoder, decoder step, joint | ANE encoder plus host CPU decode | Window joins and front-end drift | Layer parity plus corpus WER |
| Embedding encoder | Static buckets, mask input, one embedding output | ANE when CPU/GPU work is bounded | Norm overflow, gathers, bucket padding | Cosine and retrieval rank parity |
| Large mmap-friendly LLM | Often no port | llama.cpp/Metal or another mmap runtime | Core AI copies/wires weights per load | Like-for-like footprint and reload test |

## Verification gates

- [ ] Checkpoint, code, tokenizer, and data licenses are recorded.
- [ ] Every claimed language/domain has an evaluation row or is marked unmeasured.
- [ ] Host versus Core AI responsibilities are explicit.
- [ ] ANE, GPU, and CPU are separate test targets.
- [ ] Memory uses `phys_footprint` and, where relevant, neural footprint.
- [ ] Cold first load, warm load, and package/cache size are measured.
- [ ] The acceptance thresholds are written before tuning.

## Common failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| ANE is requested but GPU time is non-zero | Unsupported op or a mixed graph | Count compiled regions and run a GPU positive control. |
| Text looks right on a smoke set but fails on a language | Compression or normalizer regression | Add per-language corpus scoring and publish the regression. |
| Runtime is fast but memory is too high | Core AI weights/state are copied or wired per load | Measure `phys_footprint`, state bytes, and a mmap-based alternative. |
| Export succeeds but output is wrong | Re-authoring changed layout, masks, or reductions | Add layer-by-layer parity and negative controls. |
| First query stalls | Specialization is lazy | Load once, prewarm deliberately, and show a preparing state. |

## References and scripts

- Read [`references/triage-template.md`](references/triage-template.md) for the run-manifest template and a worked boundary analysis.
- Continue with [`coreai-export-basics`](../coreai-export-basics/SKILL.md) for a direct export, or [`ane-reauthoring`](../ane-reauthoring/SKILL.md) when the graph is not ANE-friendly.

## Evidence notes

Record negative results beside positive ones:

```text
reference command:
reference output checksum:
candidate command:
candidate output checksum:
quality metric and normalizer:
placement evidence:
memory method:
cache state:
unmeasured claims:
reason to proceed or stop:
```

If a target OS or SDK is beta, write that in every result table. A result on one Apple silicon machine is a sample, not a hardware-independent guarantee. If a product requirement cannot be measured yet, label the decision “prototype only” and list the missing experiment.

Before starting a compression sweep, put the triage sheet in the run archive. Compression without a frozen baseline makes quality loss impossible to attribute.

The triage sheet is also the handoff between an export engineer and a runtime engineer. It should be possible to reject a port from this sheet without reading an implementation diff.

If the answer depends on a proprietary runtime or unavailable device, mark that dependency explicitly instead of filling the gap with a guess.

Do not use a model's parameter count as a proxy for Core AI feasibility. State tensors, gathers, static buckets, and host control flow can dominate a smaller model's cost.

## Done when

- The decision is written with evidence, not optimism.
- A reference command and acceptance gates exist.
- Unverified compiler/runtime observations are isolated from public claims.

