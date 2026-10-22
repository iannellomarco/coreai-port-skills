# Case study: EmbeddingGemma 2 text encoder to Core AI

Public artifact: [EmbeddingGemma 2 Core AI text port](https://huggingface.co/EverPaper/embeddinggemma-2-coreai).

## Starting point

The source was the text encoder of `google/embeddinggemma-2`, an Apache-2.0 model of about 270 million parameters producing 768-dimensional Matryoshka embeddings. The reference was Sentence Transformers on the original checkpoint. The target was one Core AI graph that performs token embedding, 24 bidirectional encoder layers, final normalization, projection, masked mean pooling, and L2 normalization.

The measurements were made on one M5 Mac with 24 GB RAM, macOS 27.0, and an Xcode 27 beta-era toolchain. Seven static sequence buckets were exported: 32, 64, 128, 256, 512, 1024, and 2048 tokens. The functions share the weights. The host selects the smallest bucket and right-pads with a mask.

## Re-authoring decisions

The Neural Engine path used fp16 throughout. The model's activations can exceed fp16 range, so every RMSNorm and the final L2 normalization rescales each row by its maximum absolute value before squaring, then applies the algebraically equivalent correction. A naive fp16 control reached cosine 0.94–0.99 and about 27 dB PSNR; the rescaled path reached at least 68 dB in the torch ANE mode.

The attention authoring kept GQA/MQA broadcast explicit without `repeat_interleave` or an equivalent dynamic expansion. Masks were finite additive values and were built from `attention_mask` plus the sliding-window band. The graph avoids host-side embedding lookup: token IDs gather the in-graph embedding table. The static buckets made padding exact and bounded the runtime's specialization work.

## Measured result

The text asset was 631.5 MB decimal, or 602 MiB, with tokenizer and configuration adding to a 664 MB model directory. On the ANE route, the compiled graph had one ANE region plus the token lookup partition. The text embedding matched the fp32 reference with minimum cosine 0.99992 and median 0.99997 on the 357-text release set. The independent 100-text set had minimum cosine 0.99985 and median 0.99997. The CPU-only route was less precise because of fp16 CPU kernels; it was still above the release gate on the tested inputs.

Retrieval used 1712 fictional passages and 53 queries, including 16 Italian queries. At 768 dimensions, the ANE route had the same first-relevant rank for 53 of 53 queries and MRR 0.766, equal to the fp32 reference. At 256 dimensions, MRR was 0.785 versus 0.786. Italian MRR was reported separately and stayed aligned across the tested sizes.

Swift and Python tokenizer IDs matched on 18 parity texts, including English, Italian, code, long input, and Unicode cases. Padding to a bucket did not change the embedding on the tested set.

The Neural Engine latency table measured p50 per call: 4.8 ms at 32 tokens, 5.3 at 64, 6.6 at 128, 11.6 at 256, 27.8 at 512, 69 at 1024, and 195 at 2048. These are one-machine measurements with 50 trials per bucket. First load by a new executable was about 140 seconds; cached load was 0.4–1.5 seconds. Warm process footprint was about 230 MB, with roughly 600–770 MB of neural footprint. The GPU route became faster above 512 tokens but used more CPU and memory.

## Mistakes and fixes

- The naive fp16 normalization was numerically unsafe. The fix was max-absolute rescaling before variance and L2 reductions, with a negative-control test retained.
- An initial grouped MQA formulation was not accepted by the tested ANE matcher. The graph was rewritten with a 4-D matmul and explicit key/value broadcast.
- A dynamic sequence graph was replaced by one dynamic export followed by static bucket specializations. This made the padding, latency, and parity gates inspectable.
- Tokenizer correctness was treated as a model concern. Swift used the bundled tokenizer file, explicit BOS/EOS handling, task prefixes, and a token-ID parity suite.
- `coreai-build inspect --compute` was not used as the sole placement proof. Compilation warnings and runtime neural footprint were recorded as separate evidence.

## What remains limited

The figures are from one machine and an embedding benchmark built from fictional texts and queries. Long input is truncated to 2048 tokens by the release runner. The ANE route has a per-call floor and is not automatically the fastest route for long sequences. Users who need a different tokenizer revision, bucket set, or OS build must rerun smoke and parity.

