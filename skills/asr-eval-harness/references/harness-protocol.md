# ASR harness protocol

## Manifest schema

Required: `id`, `wav`, `text`, `lang`, `dur`. Optional fields may identify dataset split, speaker, domain, or meeting, but do not expose private data.

## Engine schema

```json
{
  "name": "engine-name",
  "mode": "batch",
  "cmd": ["./run.sh", "--list", "{list}", "--out", "{out}"]
}
```

Each output row has `id`, `text`, and `ms`. The final metadata row has `load_ms`. A failed process must still leave a stderr tail and a missing-ID list.

## Report columns

```text
set, n, audio_min, wer, sub, del, ins, ref_words,
rtf, x_realtime, peak_phys_footprint_mb, load_ms, wall_s,
rc, missing, rtf_includes_load
```

Keep the evaluator's normalizer and dataset revisions beside the report. An accuracy number without those fields is not comparable.

