---
name: coreai-llm-port
description: Port an autoregressive Hugging Face LLM to Core AI with explicit dynamic KV-cache state, prefill and decode functions, host-side sampling, tokenizer/chat-template parity, and honest memory comparisons. Use whenever an agent considers Gemma, Qwen, or another causal decoder for `.aimodel`.
---

# Purpose

Make the Core AI graph responsible for tensor math and cache updates while the host owns tokenization, sampling, stop rules, and user-visible generation state. Reject a port when Core AI's memory model loses to mmap-based GPU runtimes.

## Use this when

- The model is causal and has a growing KV cache.
- Prefill, one-token decode, chunking, or sampling must be measured separately.
- A model card or runtime claims Core AI LLM support but correctness is not established.

## Do not use this when

- The model is an encoder or embedding model. Use export basics or ANE re-authoring.
- A validated mmap/Metal runtime already meets the product target and Core AI would copy or wire a full multi-gigabyte model per load. Run the “do not port” check below.

## Prerequisites

- A teacher-forced PyTorch oracle with known tokenizer and chat template.
- Model config: layer count, KV heads, head dimensions, context limit, sliding-window behavior, special tokens.
- A target Core AI runtime and a way to measure `phys_footprint`, cache state, and output determinism.

## Procedure

1. Lock tokenizer behavior before graph work.

   Record tokenizer revision, added BOS/EOS/EOT tokens, `chat_template.jinja`, `clean_up_tokenization_spaces`, stop tokens, and whether thinking/reasoning content is enabled. Save token IDs and rendered prompts for representative conversations.

2. Build a teacher-forced oracle.

   For each prompt, save prefill logits, selected continuation logits, top-1 IDs, margins, hidden-state taps where available, and a token-level reference. Distinguish knife-edge top-1 flips from real flips using a recorded margin rule.

3. Author explicit cache state.

   Do not hide a mutable cache in the module. Pass cache tensors as state inputs and return updated state. If the model has sliding attention, make the window and mask semantics explicit. If a ring is used, its state kind must force a full reset and re-prefill after a prefix rewind; a stale ring is a correctness bug.

4. Export dynamic query length only after static probes pass.

   Use dynamic `input_ids [1,S]` and `position_ids`, but first compare fixed small `S` and `P` cases. Check single-shot prefill versus chunked prefill in PyTorch before involving Core AI.

   The tested Gemma route made KV state explicit and used dynamic query length rather than a host-fixed single-token signature. That removed a fixed-shape export limitation in the validated route, but it did not make every JIT chunked-prefill path safe. Keep single-shot, AOT, and chunked controls in the matrix.

The supplied Qwen example records 4-bit symmetric per-block-32 weights and an 8192-token runtime context in its metadata, while its attribution records a larger export cap. Treat the runtime metadata as the shipped contract and test the exact context limit; do not infer a safe context from the exporter flag alone.

5. Separate prefill from decode.

   Prefill may accept a chunk and write KV state. Decode consumes one token or a small fixed step and returns logits plus the next state. A headless prefill graph can avoid materializing a full `[S,vocab]` logits tensor when the host needs only the final prefill row, but verify the runtime actually uses that entrypoint.

6. Use a safe KV layout.

   The tested Gemma work used a ring layout for sliding layers: a growing cache for full-attention producers plus a fixed-size ring for sliding producers. It was numerically checked against a unified layout in PyTorch. Record the layout in `EXPORT-FACTS.json` and allocate every state by descriptor, not by a guessed name.

7. Keep sampling on the host.

   The host applies temperature, top-k/top-p or greedy selection, repetition rules, stop tokens, and chat message assembly. The graph returns logits. Never compare a Core AI greedy result to a sampled reference and call it a parity failure.

8. Measure memory correctly.

   Sample `phys_footprint` during export, first load, prefill, and decode. Include KV state and logits buffers. Do not use RSS as the product budget. Core AI can copy or wire weights per load; compare against a llama.cpp/Metal path that uses mmap on the same prompt and context lengths.

9. Test the failure classes deliberately.

   - NaN/Inf in logits or KV state.
   - `<pad>` or repeated-token collapse after long prefill.
   - Chunked versus single-shot prefill.
   - Prefix rewind with ring state.
   - First call versus later call in the same process and in a new process.
   - Long-context memory and decode throughput.

10. Publish only the measured route.

   If Python `coreai.runtime` is only an instrument and the stock Swift sidecar is the target, publish speed from Swift. If a prefill path is correct only in AOT or only with a particular runtime revision, state that scope.

## Decision points

- If dynamic KV export is correct but chunking is not, use single-shot prefill or a bounded host chunk policy while keeping the failure visible.
- If a fixed ring is used, reset and prefill on rewind; do not merely decrement a position counter.
- If the graph is correct but Core AI `phys_footprint` or reload time is much worse than mmap-based Metal, stop the port and ship the alternative.
- If tokenizer rendering changes whitespace or special tokens, fix the bundle before tuning weights.

## Verification gates

- Token IDs and rendered chat prompts match the reference.
- Prefill logits and teacher-forced decode pass finite-value, top-1, cosine/PSNR, and margin-aware gates.
- Single-shot and chunked prefill agree on the chosen route.
- Ring/unified or cache-layout equivalence is proven in PyTorch.
- Host sampler and stop conditions are tested independently.
- `phys_footprint` and reload are compared with the chosen alternative runtime.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| NaN or `<pad>` collapse after long prompts | Cache mask, shape specialization, or runtime path failure | Bisect `S,P`, compare single-shot/chunked, and keep the safe route. |
| First token differs only at one position | Quantization or a knife-edge margin | Report top-2 margin and compare logits, not just text. |
| Rewound prefix produces different output | Ring state retained stale rows | Full reset and re-prefill; mark state fixed. |
| Chat output has missing spaces or extra turn markers | Template or cleanup defaults differ | Ship tokenizer config and template; set cleanup explicitly. |
| Core AI load exhausts memory | Weights/states are resident or copied | Measure footprint and choose mmap/Metal if it wins. |

## References

- [`references/llm-checklist.md`](references/llm-checklist.md) contains cache, tokenizer, and memory gates.
- The Gemma and Qwen lessons in this skill are deliberately scoped to tested behavior, not universal compiler rules.

## Evidence notes

For each prompt, save:

```text
rendered prompt and token IDs
prefill top-1 agreement and margin-aware flips
teacher-forced decode agreement
chunked-vs-single logits
NaN/Inf counts
KV bytes and peak phys_footprint
first-token and steady-state timing
```

A coherent sample is not enough. Repeat the same prompt in one process and a new process, then compare the output bytes and token IDs before investigating performance.

For a public report, include at least one short prompt, one long prompt, one prompt that crosses a sliding-window boundary, one prefix-rewind case, and one chat-template case. State which calls are instrument-only and which use the production Swift runtime.

Keep a failing prompt fixture permanently. A later runtime update that makes it pass is useful evidence, but it should not erase the original regression from the history.

Required result fields:

```text
prompt token count
prefill mode
cache layout
decode tokens per second
first-token latency
peak phys_footprint
output identity
```

Use the same prompt bytes and sampling seed for every engine. Record whether logits were copied to host memory; that copy can dominate a graph-only benchmark.

## Done when

- The exact tokenizer, cache layout, runtime path, and sampler are documented.
- Correctness is shown on prefill, decode, chunking, rewind, and long context.
- The memory comparison justifies Core AI over a mmap-based alternative, or the port is rejected.

