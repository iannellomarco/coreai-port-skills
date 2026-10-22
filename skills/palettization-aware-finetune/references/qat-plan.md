# QAT plan template

## Precision ablation

| Candidate | FFN | Attention | Conv | First/last layers | Result |
|---|---:|---:|---:|---:|---|
| fp16 | 16 | 16 | 16 | 16 | baseline |
| post-training | 4 | 4 | 4 | 4 | measured control |
| mixed | 4 | 6 | 6 | 6 | measured control |
| QAT pilot | 2 | 4 | 4 | 4 | pending |

## Stop-gate record

```text
manifest fingerprint:
teacher revision:
student checkpoint:
validation step:
language/domain:
teacher WER:
student WER:
delta:
gate:
action: continue | stop | raise precision
```

## Round trip

1. Save a tiny layer and its LUT/index/scale representation.
2. Reconstruct the fake-quantized weight outside training.
3. Compare weight bytes, layer output, and exported Core AI output.
4. Only then apply the exporter to the full checkpoint.

