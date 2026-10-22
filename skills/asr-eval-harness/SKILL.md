---
name: asr-eval-harness
description: Build a trustworthy ASR evaluation harness with public manifests, per-language normalization, corpus WER, bootstrap intervals, RTF, first-load, peak-memory, and a batch JSONL engine protocol. Use whenever an agent compares speech models, Core AI exports, Swift runtimes, or published WER numbers.
---

# Purpose

Make an ASR comparison reproducible and difficult to game. Keep data preparation, normalization, engine execution, scoring, and reporting separate.

## Use this when

- A port claims WER, throughput, memory, or realtime performance.
- Python and Swift runtimes need a like-for-like comparison.
- A result will be compared with a leaderboard or model card.

## Do not use this when

- You only need a one-file smoke test. Add the smoke test, but do not call it a benchmark.
- The reference transcripts, normalizer, or public data license cannot be identified.

## Prerequisites

- Public or otherwise authorized audio and transcripts.
- A manifest with stable IDs, paths, language, raw reference, and duration.
- The normalizer required by the target benchmark.
- An engine command that can load once and process a list.

## Procedure

1. Build a manifest.

   Use JSONL rows such as:

   ```json
   {"id":"utt-0001","wav":"wav/utt-0001.wav","text":"...","lang":"it","dur":2.34}
   ```

   Convert to the model sample rate and mono with a documented resampler. Keep source IDs and dataset revisions. Do not include private paths in a public manifest.

2. Use the right public sets.

   A useful public mix can include FLEURS for per-language speech, MLS for read speech, VoxPopuli for spontaneous/oratory speech, AMI SDM for meetings, synthetic code-switching as a diagnostic, and long-form held-out audio. Keep test clips out of training and label synthetic speech as synthetic.

3. Match the normalizer.

   For English leaderboard comparisons, use the Whisper English normalizer when that is what the leaderboard uses. For other languages use the specified basic normalizer, preserving diacritics when required. For Italian, convert digits to Italian words before the basic normalizer if the reference protocol does so. When comparing to a leaderboard, vendor its normalizer and record its revision.

4. Score corpus WER.

   Compute total substitutions, deletions, and insertions over total reference words. Do not average utterance WER unless the published metric does. Keep normalized references and hypotheses for audit.

   ```python
   wer = (substitutions + deletions + insertions) / reference_words
   ```

5. Define the engine protocol.

   Prefer batch mode:

   ```text
   input TSV: id<TAB>absolute_wav_path
   output JSONL: {"id":"...","text":"...","ms":12.3}
   final JSONL: {"_meta":{"load_ms":123.4}}
   ```

   Load the model once per set process. A single-file command pays load on every utterance and must mark RTF as including load.

6. Measure the process.

   Poll `proc_pid_rusage` and report the maximum of current and lifetime-max `phys_footprint`, not RSS. Record load time, wall time, per-utterance decode time, audio duration, and missing/failed rows.

7. Separate warm and first-load paths.

   Report cold first load, new-process load with a warm system cache, same-process warm load, and the first inference call if it triggers specialization. Never hide a two-minute first specialization behind a warm median.

8. Add confidence intervals.

   Bootstrap utterances for ordinary sets. For long-form sets with very few files, say that the interval is not informative and avoid precise claims. For paired engine comparisons, resample the same utterance IDs.

9. Compare with published numbers only when scoring matches.

   Confirm dataset split, language, transcript policy, normalization, decoding policy, and unit. If one differs, call the comparison approximate or omit it.

10. Keep outputs auditable.

   Store the manifest hash, command line, engine revision, normalized references/hypotheses, summary JSON, and a markdown table. Keep each run immutable; use a new tag for a new candidate.

11. Run the skeleton.

   ```bash
   python skills/asr-eval-harness/scripts/wer_harness.py results.jsonl --bootstrap 1000
   ```

   Replace its fallback normalizer with the benchmark-matched normalizer before publication.

## Decision points

- If a language has too few words, report it as diagnostic rather than a stable estimate.
- If an engine cannot batch, keep it in the harness but mark load-inclusive RTF.
- If a Swift runtime differs from Python because of mel round-off, compare stage tensors and decoded identity before changing inference.
- If a long-form cut policy was tuned on the test set, add a holdout and mark the original result exploratory.

## Verification gates

- Manifest rows are stable and licensed.
- Normalization is named and versioned.
- Corpus WER reproduces a known reference result.
- Engine output is complete or missing rows are counted.
- Memory is `phys_footprint`; load and RTF conventions are explicit.
- Every public language/domain claim has a row and a caveat.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| WER is much better than the leaderboard | Different normalizer or split | Match the leaderboard protocol exactly. |
| RTF changes between batch and single mode | Model load is counted differently | Separate load from decode and label inclusion. |
| Peak memory looks tiny | RSS or child process was measured | Poll the engine PID's `phys_footprint`. |
| One bad file disappears | Missing output is skipped | Count every manifest ID and fail or report missing. |
| Mean masks a language loss | Aggregate over words or sets | Publish per-language/domain rows and ablations. |

## References and scripts

- [`references/harness-protocol.md`](references/harness-protocol.md) has manifest, JSONL, and report schemas.
- [`scripts/wer_harness.py`](scripts/wer_harness.py) is the dependency-light scoring skeleton.

## Evidence notes

Keep these files per run:

```text
manifest.jsonl and its checksum
engine.json and resolved command
normalizer revision
raw output JSONL
per-utterance scored rows
summary.json
stderr tail for failures
```

The included script uses a conservative fallback normalizer so it can run without a large dependency stack. Replace it before comparing to a published leaderboard. Do not mix its output with a leaderboard-normalized table.

Keep audio duration in seconds in machine-readable output and use minutes only in display tables.

Required report footer:

```text
dataset revisions:
normalizer:
engine revision:
hardware and OS:
load convention:
memory convention:
unmeasured sets:
```

## Done when

- Another engineer can run the same engine command against the same manifest and get the same protocol.
- The report includes WER, confidence, RTF, first load, warm latency, and memory with units.

