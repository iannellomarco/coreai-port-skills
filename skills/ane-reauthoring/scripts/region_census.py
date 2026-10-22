#!/usr/bin/env python3
"""Count ANE/GPU region names in a compiled MPSGraph package.

The script is intentionally read-only. Point it at a specialized graph
directory, not at an uncompiled `.aimodel` bundle. Region names vary by SDK,
so the patterns are configurable and the raw matching names are printed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--ane-pattern", default=r"ANE_region_[0-9]+")
    ap.add_argument("--gpu-pattern", default=r"GPU_region_[0-9]+")
    ap.add_argument("--expect-ane", type=int)
    ap.add_argument("--expect-gpu", type=int)
    args = ap.parse_args()
    names = sorted({p.name for p in args.root.rglob("*") if p.name})
    ane = sorted(n for n in names if re.search(args.ane_pattern, n))
    gpu = sorted(n for n in names if re.search(args.gpu_pattern, n))
    result = {"root": str(args.root), "ane": ane, "gpu": gpu,
              "ane_count": len(ane), "gpu_count": len(gpu)}
    print(json.dumps(result, indent=2))
    if args.expect_ane is not None and len(ane) != args.expect_ane:
        return 2
    if args.expect_gpu is not None and len(gpu) != args.expect_gpu:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

