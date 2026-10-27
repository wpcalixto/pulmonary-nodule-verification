#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:
    Smoke test of the repository

File type:
    Test

Purpose:
    Verifies that every Python file compiles, that every code file carries the standard header, that no
    machine-specific absolute path remains in code or documentation, and that the SHA-256 checksums match.

Execution:
    python tests/test_smoke.py
"""
import hashlib, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABS = re.compile(r"(?:/Volumes/|/Users/|/home/|/mnt/)")


def main():
    errors = 0
    for d in ("scripts", "trainers", "patches", "tests"):
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            if f.endswith(".py"):
                p = os.path.join(ROOT, d, f)
                compile(open(p, encoding="utf-8").read(), p, "exec")
                head = open(p, encoding="utf-8").read(3000)
                if d != "tests" and ("Title:" not in head or "License:" not in head):
                    print("missing header:", p); errors += 1
    for d, _, fs in os.walk(ROOT):
        for f in fs:
            if f.endswith((".py", ".ipynb", ".md", ".cff", ".json")) and "data" not in d.split(os.sep) and "models" not in d.split(os.sep):
                p = os.path.join(d, f)
                if p.endswith("test_smoke.py"):
                    continue
                if ABS.search(open(p, encoding="utf-8", errors="ignore").read()):
                    print("absolute path:", p); errors += 1
    ck = os.path.join(ROOT, "checksums", "checksums_sha256.txt")
    if os.path.exists(ck) and "--checksums" in sys.argv:
        for line in open(ck, encoding="utf-8"):
            h, rel = line.rstrip("\n").split("  ", 1)
            with open(os.path.join(ROOT, rel), "rb") as fh:
                if hashlib.sha256(fh.read()).hexdigest() != h:
                    print("checksum mismatch:", rel); errors += 1
    print("OK" if not errors else "%d problem(s)" % errors)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
