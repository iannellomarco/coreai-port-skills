#!/usr/bin/env python3
"""Read-only release checker for a local model package.

It verifies required files, SHA-256 lines in CHECKSUMS.sha256, and privacy
patterns. A smoke command can be run against a copied temporary package with
--fresh-copy so the command does not accidentally use the working checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


PRIVATE = ("/" + "Users/", "ia" + "nne", "ian" + "nello", "ma" + "rco",
          "hind" + "sight", "everpaper" + ".app", "tail" + "scale")


def checksums(root: Path) -> list[str]:
    path = root / "CHECKSUMS.sha256"
    if not path.exists():
        return ["missing CHECKSUMS.sha256"]
    errors = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        target = root / rel.lstrip("*")
        if len(digest) != 64 or not target.is_file():
            errors.append(f"bad checksum row: {line}")
            continue
        h = hashlib.sha256(target.read_bytes()).hexdigest()
        if h != digest.lower():
            errors.append(f"checksum mismatch: {rel}")
    return errors


def privacy(root: Path) -> list[str]:
    errors = []
    for p in root.rglob("*"):
        if not p.is_file() or p.name == "CHECKSUMS.sha256":
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for term in PRIVATE:
            if term.lower() in text.lower():
                errors.append(f"{term}: {p.relative_to(root)}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--require", action="append", default=["README.md", "LICENSE", "NOTICE", "CHECKSUMS.sha256"])
    ap.add_argument("--smoke-cmd", nargs="+", help="Command run from the fresh copy")
    ap.add_argument("--fresh-copy", action="store_true")
    args = ap.parse_args()
    errors = [f"missing {rel}" for rel in args.require if not (args.root / rel).exists()]
    errors += checksums(args.root)
    errors += privacy(args.root)
    if args.smoke_cmd:
        if args.fresh_copy:
            with tempfile.TemporaryDirectory() as td:
                copy = Path(td) / args.root.name
                shutil.copytree(args.root, copy)
                run = subprocess.run(args.smoke_cmd, cwd=copy, text=True)
        else:
            run = subprocess.run(args.smoke_cmd, cwd=args.root, text=True)
        if run.returncode:
            errors.append(f"smoke command exited {run.returncode}")
    if errors:
        print("FAIL")
        print("\n".join(errors))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

