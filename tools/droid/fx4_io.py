# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Small immutable-output helpers shared by DROID Fx4 stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(text)
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def write_checksums(path: Path, files: Iterable[Path]) -> None:
    lines = [f"{sha256_file(file)}  {file.name}\n" for file in files]
    atomic_write_text(path, "".join(lines))


def validate_checksums(path: Path) -> None:
    for line in path.read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        target = path.parent / filename.strip()
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"Checksum mismatch: {target}")


def refuse_source_output(success_root: Path, output_dir: Path) -> None:
    success_root = success_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir == success_root or success_root in output_dir.parents:
        raise ValueError(f"Refusing to write derived output inside source dataset: {output_dir}")
