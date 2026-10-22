# Case study: Parakeet TDT 0.6B v3 to an ANE encoder

Public artifact: [EverParakeet TDT 0.6B v3 Core AI](https://huggingface.co/EverPaper/everparakeet-tdt-0.6b-v3-coreai).

## Starting point

The source was NVIDIA Parakeet TDT 0.6B v3, a FastConformer encoder paired with a Token-and-Duration Transducer decoder. The export boundary was not one graph: the encoder is a tensor graph, while duration-aware greedy decoding advances an encoder-frame pointer and conditionally runs the prediction network. The final bundle therefore contains an encoder asset and host-owned decoder/joint control.

The baseline was a Core ML build run through FluidAudio on one M5 Mac with macOS 27. The full evaluation used public FLEURS, MLS, VoxPopuli, AMI SDM, synthetic Italian/English code-switching, and five long-form files. The harness used corpus WER and reported RTF, first load, peak process `phys_footprint`, and per-utterance output.

## What changed

The first direct Core AI export followed Apple's Parakeet recipe. It used fixed windows and the original encoder shape. The fp16 fixed-window path was accurate and ran on the ANE, but ordinary quantized/palettized variants did not meet the placement goal. Four-bit variants were smaller but had worse WER on code-switching and long audio, and the dynamic-length route did not solve long-form ANE execution.

The successful re-authoring moved the encoder to BC1S `(B,C,1,S)`, converted linear projections to `1x1` convolutions, folded the convolution normalization, passed validity masks as inputs, and expressed relative attention with fixed sine/cosine tables and per-head operations. Norm calculations were kept inside a safe fp16 range. The encoder was then compressed with `pal4g8s`: a 4-bit palette, groups of eight output channels, with per-channel scale.

Three fixed functions were retained in one shared asset: approximately 5.12, 15.36, and 30.72 seconds. The host routes an utterance to the smallest fitting window. Long audio is cut at the quietest 200 ms in the last four seconds of a 30-second piece. This is bounded host chunking, not streaming.

## Measured result

The recommended shared encoder was 335 MiB. The measured shipped-set estimate was about 369 MiB: 335 MiB encoder, 33 MiB fp16 decoder/joint, 0.15 MiB tokenizer, and 0.5 MiB preprocessor. The public download card reports about 390 MB in decimal units. The Core ML baseline was about 463 MiB in the lab report, with 426 MiB for its encoder; the report also counted the complete baseline at about 471 MB depending on the package boundary.

On the full eight-set harness, the shared `pal4g8s` result was mean WER 13.67 versus 13.54 for the baseline. Per-set values were:

| Set | Baseline | Shared ANE encoder |
|---|---:|---:|
| FLEURS Italian | 5.70 | 6.13 |
| FLEURS English | 6.90 | 7.05 |
| MLS Italian | 12.20 | 12.11 |
| VoxPopuli Italian | 13.00 | 13.14 |
| VoxPopuli English | 5.70 | 5.94 |
| AMI SDM | 19.10 | 19.23 |
| Synthetic code-switch | 24.10 | 23.36 |
| Long-form | 21.70 | 22.40 |

The encoder had one ANE region and zero GPU regions in the verified shared-asset check. GPU time was zero for the candidate and 0.94 seconds after 30 clips for the forced-GPU decoder control in the Swift report. Warm encoder p50 was 7.5, 30.7, and 54.2 ms for the three windows. The Swift runtime later measured 13.2 ms for clips up to three seconds and 50.2 ms for 10–16 second clips, with a warm cache. First ANE specialization was about 160 seconds in the Swift release report; the earlier shared-asset report measured 118 seconds for a fresh compile. These are different runs and should not be collapsed into one number.

## Mistakes and fixes

- The first direct export treated “Core AI export succeeded” as close to “ANE product.” Placement was later checked by compiled-region census, GPU-time control, and neural footprint.
- Four-bit post-training compression looked acceptable on high-resource Italian/English samples but exposed long-form and code-switch weaknesses. The full multilingual claim was narrowed: only Italian, English, and synthetic code-switching were measured after compression; the other base-model languages were not re-measured.
- A single fixed window was not enough. The 15-second-only candidate measured mean WER 14.07; routing across three windows measured 13.67.
- The initial end-to-end timing used a Python torch decoder while the baseline used a Swift decoder. The release separated graph, host decoder, and full-pipeline measurements.
- Swift differed from Python on about 1.4% of utterances because FFT/log and reduction order changed tiny mel values. The fix was a line-by-line front-end port, an actor-isolated runtime, and a NEON decoder with fp32 accumulation. Swift text matched Python on 1658 of 1681 utterances in the full harness.
- A first-run specialization lasting minutes is user-visible. The runtime needs a background preparing state, not a lazy stall on the first transcription.

## What remains limited

There is no streaming encoder in this port. The 25-language base model is retained in the package, but the compressed result was not re-evaluated on all 25 languages. The long-form set has five files. The mean comparison is a useful engineering result, not proof of equal accuracy on every domain or hardware generation.

