#!/usr/bin/env python3
"""Extract local PDF text with Poppler's pdftotext and optionally search queries."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path


def safe_name(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "paper"
    return f"{stem}.txt"


def extract(pdf: Path, out_dir: Path, force: bool) -> Path:
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None:
        raise RuntimeError("pdftotext not found. Install poppler-utils or use another PDF extractor.")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / safe_name(pdf)
    if force or not out.is_file() or out.stat().st_mtime < pdf.stat().st_mtime:
        subprocess.run([pdftotext, "-layout", str(pdf), str(out)], check=True)
    return out


def search(path: Path, queries: list[str], context: int) -> None:
    if not queries:
        return
    lines = path.read_text(errors="replace").splitlines()
    lowered = [(q, q.lower()) for q in queries]
    for i, line in enumerate(lines, start=1):
        lower = line.lower()
        hits = [q for q, q_lower in lowered if q_lower in lower]
        if not hits:
            continue
        start = max(1, i - context)
        end = min(len(lines), i + context)
        print(f"\n{path}:{i}: matches {', '.join(hits)}")
        for j in range(start, end + 1):
            print(f"{j}: {lines[j - 1]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="+", type=Path, help="PDF path(s) to extract")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/cosmos3-paper-text-cache"))
    parser.add_argument("--force", action="store_true", help="Re-extract even if cached text is fresh")
    parser.add_argument("--query", action="append", default=[], help="Case-insensitive query to search in extracted text")
    parser.add_argument("--context", type=int, default=2, help="Context lines around query hits")
    args = parser.parse_args()

    for pdf in args.pdf:
        out = extract(pdf, args.out_dir, args.force)
        print(out)
        search(out, args.query, args.context)


if __name__ == "__main__":
    main()
