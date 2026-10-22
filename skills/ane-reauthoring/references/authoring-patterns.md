# ANE authoring patterns

## Shape conventions

| Operation | Useful shape | Notes |
|---|---|---|
| Sequence channel tensor | `[B,C,1,S]` | Time is the last axis; broadcast masks over it. |
| Log-mel input | `[1,1,n_mels,8S]` | Pad only the tail and pass valid lengths. |
| Validity masks | `[1,1,1,4S]`, `[1,1,1,2S]`, `[1,1,1,S]` | Build from the same stride rule as the reference. |
| Attention head | `[B,head,1,S]` | Keep head loops or fixed head groups explicit if the matcher needs it. |

## Parity ladder

1. Weight surgery: Linear versus `Conv2d(1,1)`.
2. Activation and normalization in fp32.
3. Same graph in fp16 on CPU.
4. Core AI CPU specialization.
5. Core AI GPU specialization.
6. Core AI ANE specialization.
7. Host runtime and final task metric.

At every step store max absolute difference, cosine, PSNR, finite-value count, and shape. A task score can remain unchanged while a hidden state is already wrong, so do not skip the intermediate ladder.

## Placement record

```text
asset: <asset name>
function: <function name>
device: <device class>
compiled ANE regions: <count>
compiled GPU regions: <count>
GPU counter delta, candidate: <time>
GPU counter delta, forced-GPU control: <time>
ri_neural_footprint: <bytes>
cache state: cold | warm | unknown
```

Do not call a zero GPU-counter result proof if the graph census is unavailable, and do not call a positive neural footprint proof of all-ANE placement. Keep both pieces of evidence.

