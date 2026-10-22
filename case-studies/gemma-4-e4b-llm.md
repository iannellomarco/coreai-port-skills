# Case study: Gemma 4 E4B LLM port lessons

This is a lessons-only case study. It does not present a product claim or a complete public release.

## Starting point

The port targeted the text decoder of a Gemma 4 E4B checkpoint with a 32K context and a stock Core AI language-model runtime. The graph had a large per-layer-embedding table, sliding and full attention with different head dimensions, KV-shared layers, a tied large vocabulary head, and an autoregressive host loop.

## Lessons that survived the experiments

1. Export the cache, not a mutable Python object. The successful design used explicit cache state and dynamic `input_ids [1,S]` for prefill/decode. The host still owned tokenization, sampling, stop tokens, and chat messages.
2. Use a ring layout when sliding attention needs a bounded window. The tested design kept a growing cache for full-attention producers and a fixed 512-position ring for sliding producers. PyTorch equivalence was checked against a unified layout. Rewind requires a full reset and re-prefill when the runtime treats the ring as fixed state.
3. Separate prefill and decode. A headless prefill entrypoint can avoid a full vocabulary logits tensor for every prefill row, but the runtime must actually bind and use that entrypoint. Measure time-to-first-token and steady decode separately.
4. Split large gathered constants. The measured graph needed its int8 per-layer-embedding table split into four chunks so the tested runtime could gather correct rows. This is a build-specific observation and belongs in a versioned export fact, not a universal Core AI promise.
5. Treat tokenizer configuration as part of the graph contract. The bundle carried the tokenizer, chat template, generation config, extra stop token, and explicit `clean_up_tokenization_spaces: false`. Otherwise the same IDs can detokenize with changed spaces or stop behavior.
6. Quantization needs placement and numerical gates. The tested recipe used int4 attention/head and int8 MLP weights from a QAT grid, while fp32 islands could improve numerics at the cost of ANE eligibility. Record which modules actually run on which unit.
7. Measure `phys_footprint`, not RSS. KV state, logits buffers, compiled resources, and Core AI weight wiring can dominate. A mmap-based llama.cpp/Metal runtime can be the better choice when reload and resident memory matter more than a single `.aimodel` bundle.

## Honest stopping rule

The port is not done when one prompt produces coherent text. It is done when prefill logits, teacher-forced decode, chunked versus single-shot prefill, cache rewind, tokenizer rendering, first load, long-context memory, and the stock Swift runtime all pass their declared gates. If a compiler/runtime failure remains version-specific or unreproduced, it stays in internal review rather than becoming public guidance.

