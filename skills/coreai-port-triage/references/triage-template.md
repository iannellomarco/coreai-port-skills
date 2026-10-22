# Port triage worksheet

Copy this into a run directory and fill every field.

## Identity

| Field | Value |
|---|---|
| Upstream model ID | |
| Revision / commit | |
| Weight checksum | |
| Parameter count | |
| Base license | |
| Tokenizer / processor license | |
| Adapted-code notices | |
| Target OS / SDK | |
| Device | |
| Exporter versions | |

## Graph boundary

| Component | Reference inputs/outputs | Core AI function? | Host responsibility |
|---|---|---|---|
| Preprocess | | | |
| Main encoder / transformer | | | |
| Cache or recurrent state | | | |
| Decoder / sampler | | | |
| Postprocess | | | |

## Budgets

Record units separately. MiB is `bytes / 2**20`; decimal MB is `bytes / 1_000_000`.

| Metric | Budget | Measurement method | Result | Gate |
|---|---:|---|---:|---|
| Package bytes | | file sizes | | |
| Process peak | | `proc_pid_rusage` | | |
| Neural footprint | | `RUSAGE_INFO_V6` or runtime API | | |
| First load | | cold cache | | |
| Warm load | | same process / new process | | |
| Warm call | | p50 and p95 | | |
| Quality | | reference-matched metric | | |

## Architecture decision

Write one sentence for each:

1. The graph is / is not a good Core AI candidate because …
2. The host owns … because …
3. The target unit is …; GPU and CPU controls are …
4. We will call this complete only when …

