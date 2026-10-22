# Compression matrix

## Recipe comparison

| Recipe | Typical use | Keep uncompressed or higher precision | Required checks |
|---|---|---|---|
| fp16 | ANE quality baseline | None | Full parity, placement, load, memory |
| int8 per channel | Fidelity and size | Sensitive residual/norm paths as needed | Per-layer PSNR and task metric |
| six-bit palette | Long-tail recovery | Attention or convolution if sensitive | Same manifest as four-bit |
| mixed 4/6/8 | Targeted size reduction | Depthwise and first/last or attention layers | Class ablation and placement |
| pal4g8s | Four-bit ANE candidate | Tiny/depthwise layers | Region census, all languages, long audio |
| two-bit palette | Research only without QAT | Most sensitive layers | Do not publish from smoke data |

## Size worksheet

```text
asset bytes = encoder + decoder/joint + tokenizer + processor + runtime/tests
MiB = asset bytes / 1,048,576
decimal MB = asset bytes / 1,000,000
```

Include the specialization cache as a separate operational cost. It is not part of the download, but it affects first-use disk and cache behavior.

## ASR interpretation

The supplied full-harness report found fp16 and int8 close to the Core ML baseline, while un-authored four-bit candidates were smaller but lost more on code-switch and long-form audio. After BC1S ANE re-authoring and shared static windows, `pal4g8s` measured 13.67 mean WER against 13.54 for the baseline on the tested eight-set harness, with a +0.7 point long-form gap and only Italian and English plus synthetic code-switch measured in that full run. The release card therefore must not imply all 25 languages were re-evaluated.

