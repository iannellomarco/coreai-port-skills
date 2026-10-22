# Core AI porting skills

This repository is a public, shareable set of Claude Code / agent skills for porting PyTorch and Hugging Face models to Apple's Core AI runtime on macOS 27 and iOS 27. It covers the work that tends to decide whether a port is useful: graph re-authoring for the Apple Neural Engine (ANE), compression, placement evidence, numerical parity, Swift runtimes, and release hygiene.

It is for engineers who already have a working PyTorch reference and need a reproducible `.aimodel` bundle and an honest runtime claim. The files are also plain Markdown documentation; an agent can follow them without installing Claude Code.

## Skill map

| Skill | Use it for |
|---|---|
| `coreai-port-triage` | Decide whether a model is a sensible port, what belongs in Core AI, and what “done” means. |
| `coreai-export-basics` | Export static or dynamic graphs, shared multi-function assets, and load them from Python or Swift. |
| `ane-reauthoring` | Re-author graphs for full ANE placement and prove placement and layer parity. |
| `coreai-compression` | Compare palettization, integer quantization, and mixed precision without hiding multilingual regressions. |
| `coreai-llm-port` | Port autoregressive LLMs with dynamic KV state, prefill/decode separation, tokenizers, and host sampling. |
| `asr-eval-harness` | Build a corpus-level WER, latency, first-load, and memory harness with reproducible engine I/O. |
| `swift-coreai-runtime` | Build a native Swift runtime with actor isolation, front-end parity, decoding, and user-facing behavior. |
| `coreai-model-release` | Package and verify a public Hugging Face release with licenses, checksums, fresh-install smoke tests, and honest cards. |
| `palettization-aware-finetune` | Plan QAT below four bits with STE palettes, distillation, replay, and stop gates. This is marked as in progress. |

Start with `coreai-port-triage`, then use the narrowest skills that match the architecture. The case studies show how the procedures were applied to an ASR encoder, an embedding encoder, and an LLM.

## Install

For Claude Code, copy the `skills/` directory into the user skill directory or into a project:

```bash
cp -R skills/* ~/.claude/skills/
# or, from a project root:
cp -R skills/* .claude/skills/
```

Each folder contains a `SKILL.md` with Claude Code front matter. The `references/` and `scripts/` folders stay beside it. You can instead read or link the Markdown files directly from another agent.

## Requirements

- Apple silicon and macOS 27 or iOS 27 for Core AI runtime verification.
- Xcode 27 for Swift and the Core AI framework.
- Python 3.12 for the supplied helper scripts and the validated export environments.
- A PyTorch/Hugging Face reference, a pinned exporter environment, and a way to record exact model revisions and licenses.
- Public evaluation data appropriate to every language or domain claim. Do not treat a smoke set as a benchmark.

Core AI APIs and toolchains described here are beta-era APIs. Names, placement diagnostics, cache layout, and compiler behavior can change with the OS or SDK. Record the OS build, Xcode version, model revision, exporter versions, compute request, and whether the measurement is warm or cold.

## License

The skills and helper scripts in this repository are licensed under Apache-2.0. Ported model weights, tokenizer files, adapted source files, and evaluation data retain their upstream terms. A public release must carry the upstream notices that apply to those files.

## Credits

The workflow is informed by Apple's Core AI model recipes and runtime utilities, public Hugging Face model cards, and the engineering evidence in the case studies. Related community porting material is useful context, but these skills add ANE re-authoring, placement verification, multilingual compression evaluation, Swift front-end parity, release hygiene, and QAT planning.

