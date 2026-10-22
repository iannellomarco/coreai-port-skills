# Swift parity matrix

| Stage | Fixture | Metric | Gate |
|---|---|---|---|
| Audio load | short, resampled, multi-channel, NaN | sample count / finite values | exact policy |
| Mel | silence, speech, long, boundary | max abs / mean abs / frame count | reference-specific |
| Encoder | same mel tensor | max abs / cosine / PSNR | reference-specific |
| Decoder state | blank, emitted token, duration | state tensor diff | exact or declared tolerance |
| Tokens | random IDs and real utterances | identity | exact where possible |
| Text | punctuation, Unicode, repeated IDs | string identity | exact unless float drift is explained |

## Measurement record

Record warm p50/p95, first load, new-process load, process `phys_footprint`, neural footprint, GPU counter, and whether the runtime was called from the main actor. Keep graph and host decode times separate.

