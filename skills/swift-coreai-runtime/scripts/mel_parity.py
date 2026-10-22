#!/usr/bin/env python3
"""Compare reference and candidate front-end tensors saved as NumPy arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--texts", type=Path,
                    help="Optional JSONL with reference_text and candidate_text")
    args = ap.parse_args()
    ref, got = np.load(args.reference), np.load(args.candidate)
    if ref.shape != got.shape:
        print(json.dumps({"shape_equal": False, "reference": ref.shape, "candidate": got.shape}))
        return 2
    diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
    result = {"shape_equal": True, "shape": list(ref.shape),
              "max_abs_diff": float(diff.max()) if diff.size else 0.0,
              "mean_abs_diff": float(diff.mean()),
              "exact_elements": int(np.count_nonzero(diff == 0)),
              "elements": int(diff.size)}
    if args.texts:
        pairs = [json.loads(line) for line in args.texts.read_text().splitlines() if line.strip()]
        result["text_identical"] = sum(p.get("reference_text") == p.get("candidate_text") for p in pairs)
        result["text_total"] = len(pairs)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

