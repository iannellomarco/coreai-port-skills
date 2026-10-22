---
name: swift-coreai-runtime
description: Build and verify a native Swift runtime around a Core AI model, including actor isolation, load-once function reuse, mel/front-end parity, tokenizer detokenization, CPU decode loops, audio loading, and first-run UX. Use whenever a `.aimodel` must become a safe macOS or iOS Swift API.
---

# Purpose

Port the validated inference path line by line into Swift and prove every boundary. Do not let a code generator “simplify” a working decoder or front end before parity is measured.

## Use this when

- A Python or Core AI graph needs a native Swift package or CLI.
- Multiple tasks may call one model instance.
- Audio preprocessing, tokenization, host decoding, or first-run specialization affects correctness.

## Do not use this when

- The task is only exporting a graph.
- The model has no host runtime and no Swift target.

## Prerequisites

- Xcode 27 / Swift toolchain matching the target SDK.
- A working Python reference and saved stage fixtures.
- Core AI asset descriptors and tokenizer/processor files.
- A test CLI that can emit JSONL for the evaluation harness.

## Procedure

1. Define the concurrency boundary.

   Put the model, loaded functions, reusable descriptors, and mutable decoder state in an `actor`. One actor instance serializes calls; use one instance per independent stream if parallelism is required.

   ```swift
   public actor SpeechModel {
       private let encoder: InferenceFunction
       private var decoderState: DecoderState
       public func transcribe(_ pcm: [Float]) async throws -> String { /* ... */ }
   }
   ```

   Prove thread safety with concurrent callers. A real data race in a runtime is a correctness failure even when single-threaded WER passes.

2. Load once and reuse functions.

   Load `AIModel` once, resolve descriptors and functions once, and precompute mel bases or static metadata once. Do not reload assets or resolve functions per utterance.

3. Port the front end line by line.

   Match sample rate conversion, mono policy, frame count, window, hop, FFT, mel filters, log floor, reduction order, dtype casts, and padding. `AVAudioFile.read(into:)` can return short reads; loop until the requested samples are filled or EOF is reached. Convert NaN and infinite input samples to a documented finite policy.

4. Build a mel parity fixture.

   Save reference and Swift mel tensors for clips from every input class, including near-silence, short audio, resampled audio, long audio, and multi-channel input. Run:

   ```bash
   python skills/swift-coreai-runtime/scripts/mel_parity.py \
     reference.npy candidate.npy --texts text_pairs.jsonl
   ```

   Report frame-count equality, max absolute difference, mean difference, fp16 one-step differences, and decoded-text identity.

5. Match Core AI shapes and masks.

   Resolve dynamic dimensions from the concrete mel shape. For static windows, zero-pad the tail and pass a valid mask if the graph expects one. Copy outputs before starting another graph call if the runtime may recycle output storage.

6. Port the host decode loop exactly.

   Keep state initialization, blank handling, duration advancement, max-symbol rules, cache resets, and stop conditions identical to the validated reference. A small NEON fp16-weight/fp32-accumulate kernel can reduce CPU time, but compare it against an Accelerate or fp32 reference before choosing it.

7. Port detokenization with explicit tests.

   For SentencePiece-like tokenizers, convert the `▁` marker to a word boundary according to the reference, collapse repeated IDs only when the model's blank/repetition rule says so, skip the exact special-token set, and drop out-of-vocabulary sentinels. Test random IDs, repeated IDs, blank IDs, special tokens, punctuation, and Unicode.

8. Add a user-facing CLI and batch mode.

   Support `transcribe`, `smoke`, and `--list <tsv> --out <jsonl>`. The batch output should include per-clip text and milliseconds plus a final load metadata row. Print a clear “preparing” state while the first ANE specialization runs.

9. Measure decoder alternatives.

   Compare Core AI CPU/GPU/ANE graph calls, Accelerate, and a small C/NEON kernel in microseconds per call. Keep accuracy separate: fp16 recurrent state can be faster but change WER. Choose the fastest method that passes the full reference gate.

10. Run the full matrix.

   Compare Python and Swift stage tensors, decoded tokens, text identity, corpus metric, warm latency, first load, process footprint, neural footprint, and placement. The Python runner is the reference only if the Swift front end and host loop are proven equivalent.

## Decision points

- If mel values differ but text is identical, keep the difference recorded and investigate reduction/FFT order before declaring parity.
- If a fp16 CPU recurrent state changes WER, use fp32 accumulation or the NEON path with fp32 accumulation.
- If Core AI graph decode is slower than CPU, keep it as a correctness control and use the faster host path only after parity.
- If the first specialization is long, prewarm in a background task and surface progress.

## Verification gates

- Concurrent calls do not race or corrupt state.
- Audio loading handles short reads, resampling, NaN, mono conversion, and empty input.
- Mel shape and numeric thresholds pass on all fixture classes.
- Encoder hidden state and decoded token parity pass.
- Swift text matches Python within the declared identity/delta gate.
- Batch JSONL protocol and full harness pass.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Swift is one or two words different | FFT/reduction/fp16 round-off | Capture mel and hidden states; match operation order. |
| Sporadic wrong output with concurrent calls | Shared mutable model/decoder state | Isolate it in an actor or one stream instance. |
| First query blocks the UI | Lazy specialization | Load and prewarm off the main actor; show progress. |
| Output spacing differs | SentencePiece boundary or cleanup rule | Test token IDs and preserve reference detokenization. |
| Audio is truncated or padded oddly | Short `AVAudioFile` read | Read in a loop and validate sample count. |

## References and scripts

- [`references/swift-parity.md`](references/swift-parity.md) lists stage fixtures and runtime measurements.
- [`scripts/mel_parity.py`](scripts/mel_parity.py) is a generic numeric/text comparison helper.

## Evidence notes

Expose a verification-only API that returns mel values, encoder hidden states, emitted token IDs, and decoded text. Keep it out of the normal user path if it increases memory or work. Test the actor with overlapping calls, cancellation, model close during an active call, and a second independent actor. Record whether each failure is a numeric mismatch, a state race, an audio I/O issue, or a Core AI runtime error.

Prefer fixtures with a stable binary format such as NumPy arrays or JSONL. Include the dtype and shape in fixture metadata so a later Swift or SDK change cannot silently compare transposed data.

Keep the normal path small. Verification hooks may copy arrays and add work, but production code should reuse loaded functions and avoid a second front-end pass.

Keep a small text-only tokenizer test separate from model loading. It should run on a machine without Core AI so detokenization failures are isolated from specialization failures.

When a runtime error is returned, include the function name, concrete input shape, requested compute unit, and whether the call was the first specialization. This makes SDK-specific failures actionable without exposing private machine details.

## Done when

- The Swift runtime follows the validated inference path and passes full parity.
- The user-facing CLI, first-run state, and batch protocol are tested.
- Memory, placement, and load claims are measured on the target device.

