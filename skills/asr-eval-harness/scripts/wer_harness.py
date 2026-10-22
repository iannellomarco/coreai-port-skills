#!/usr/bin/env python3
"""Small, dependency-light corpus WER harness skeleton.

Input is JSONL with one row per utterance: {"id", "reference", "hypothesis",
"lang", "audio_seconds", "decode_ms"}. The normalizer is intentionally
conservative; replace it with the scorer used by the benchmark you compare to.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


def normalize(text: str, lang: str) -> list[str]:
    # Use a benchmark-matched normalizer in production. This fallback preserves
    # accents and only lowercases, strips punctuation, and collapses whitespace.
    text = re.sub(r"[^\w\s]", " ", (text or "").lower(), flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().split()


def edits(ref: list[str], hyp: list[str]) -> tuple[int, int, int, int]:
    row = [(j, 0, j, 0) for j in range(len(hyp) + 1)]
    for i, token in enumerate(ref, 1):
        nxt = [(i, 0, 0, 0)]
        for j, candidate in enumerate(hyp, 1):
            ins = nxt[j - 1][0] + 1, 0, 0, 1
            dele = row[j][0] + 1, 0, 1, 0
            sub = row[j - 1][0] + (token != candidate), int(token != candidate), 0, 0
            nxt.append(min((ins, dele, sub), key=lambda x: x[0]))
        row = nxt
    cost, sub, dele, ins = row[-1]
    return sub, dele, ins, cost


def corpus(rows: list[dict], indices: list[int] | None = None) -> dict:
    chosen = range(len(rows)) if indices is None else indices
    s = d = ins = words = 0
    for i in chosen:
        r = normalize(rows[i].get("reference", ""), rows[i].get("lang", ""))
        h = normalize(rows[i].get("hypothesis", ""), rows[i].get("lang", ""))
        a, b, c, _ = edits(r, h)
        s, d, ins, words = s + a, d + b, ins + c, words + len(r)
    return {"wer": (s + d + ins) / words if words else None, "sub": s,
            "del": d, "ins": ins, "ref_words": words}


def bootstrap(rows: list[dict], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    vals = []
    for _ in range(samples):
        idx = [rng.randrange(len(rows)) for _ in rows]
        value = corpus(rows, idx)["wer"]
        if value is not None:
            vals.append(value)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    result = corpus(rows)
    result["rtf"] = sum(float(r.get("decode_ms", 0)) for r in rows) / 1000 / max(
        1e-12, sum(float(r.get("audio_seconds", 0)) for r in rows))
    if rows and args.bootstrap:
        result["wer_bootstrap_95"] = bootstrap(rows, args.bootstrap, args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

