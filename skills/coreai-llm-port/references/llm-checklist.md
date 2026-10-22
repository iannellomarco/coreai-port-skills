# LLM port checklist

## Bundle

```text
metadata.json
<name>.aimodel/
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
tokenizer/chat_template.jinja
tokenizer/generation_config.json
LICENSE
MODEL-ATTRIBUTION.json
EXPORT-FACTS.json
```

Set `clean_up_tokenization_spaces` explicitly when the host tokenizer defaults differ from the reference. Include EOT and other extra stop tokens in the metadata path the runtime actually reads.

## Cache record

| Test | Context | Query length | Cache layout | Result |
|---|---:|---:|---|---|
| Single-shot prefill | | | | |
| Chunked prefill | | | | |
| One-token decode | | | | |
| Prefix rewind | | | | |
| New process first call | | | | |

## Memory record

Record peak process `phys_footprint`, KV bytes, logits buffer bytes, neural footprint, load time, and disk cache. Repeat for a mmap-based control. The comparison is not meaningful if one path measures only the model file and the other includes the runtime state.

