# Export checklist

## Bundle manifest

```text
bundle/
  metadata.json
  model.aimodel/
  tokenizer/                 # when needed
  processor/                 # when needed
  LICENSE
  NOTICE
  MODEL-ATTRIBUTION.json
  EXPORT-FACTS.json
```

`EXPORT-FACTS.json` should record the source revision, exporter versions, input names, output names, shape buckets, precision, compression, compute request, and whether the numbers are reference-only or full-pipeline numbers.

## Static versus dynamic

| Question | Static buckets | Dynamic axis |
|---|---|---|
| Shape coverage | Explicit, easy to audit | Broad, but must be tested at many lengths |
| First-load cost | One specialization per bucket | Can specialize per observed shape |
| Latency | Predictable within a bucket | May vary with shape and cache history |
| Padding | Required; mask it | Often less padding, but dynamic masks still need testing |
| ANE confidence | Higher when operations are simple | Lower until every target length has placement evidence |
| Host design | Route to smallest fitting function | Pass concrete dimensions and manage failures |

## Runtime record

```json
{
  "os_build": "<macOS 27 build>",
  "xcode": "<Xcode 27 build>",
  "device": "<Apple silicon device>",
  "compute_request": "neural-engine",
  "functions_loaded": ["main_32", "main_64"],
  "cold_load_ms": 0,
  "warm_load_ms": 0,
  "peak_phys_footprint_mb": 0,
  "neural_footprint_mb": 0
}
```

