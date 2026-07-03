---
name: cosmos3-paper-reference
description: Use when Codex needs to verify Cosmos3, Cosmos3-Nano/Super, RoboMIND, RoboMIND 2.0, UR5(e) robot-action post-training, dataset schema, model architecture, training, inference, or implementation claims against the local PDF papers under tmps/. Provides a workflow for extracting and searching paper text with local PDF tools and citing file/page evidence before answering or editing.
---

# Cosmos3 Paper Reference

## Purpose

Use this skill to ground Cosmos3 and RoboMIND work in the local papers instead of memory. The key paper `tmps/cosmos3_main.pdf` is an NVIDIA-provided Cosmos3 paper that maps to this repository and runtime environment.

## Required Workflow

1. Read `references/papers.md` first to select the relevant PDF(s).
2. Prefer local evidence over recall for claims about architecture, training recipes, action-policy behavior, dataset schema, evaluation, or terminology.
3. Extract searchable text with `scripts/extract_pdf_text.py` or directly with `pdftotext -layout`.
4. Search extracted text with `rg`; inspect enough surrounding lines to understand the claim.
5. When answering, cite the paper by PDF filename and page when available, and cite repo code/docs in `file:line` format when connecting paper claims to implementation.
6. If extraction quality is poor, use `pdftoppm` to render relevant pages and inspect images/tables manually.

## PDF Tools

This device has Poppler tools available: `pdfinfo`, `pdftotext`, and `pdftoppm`. A 17.2 MiB / 66-page paper (`tmps/RoboMIND2.pdf`) was fully extracted with `pdftotext -layout` in under one second, so no apt installation is normally needed for these papers.

Use this command for repeatable extraction:

```bash
python .agents/skills/cosmos3-paper-reference/scripts/extract_pdf_text.py tmps/cosmos3_main.pdf --query action --query post-training
```

The script writes text files under `/tmp/cosmos3-paper-text-cache` by default and prints matching line numbers for supplied queries.

## Evidence Standards

- For Cosmos3 implementation questions, consult `tmps/cosmos3_main.pdf` and the local code together.
- For RoboMIND dataset questions, consult `tmps/RoboMind.pdf`, `tmps/RoboMIND2.pdf`, and local dataset/converter code.
- Do not treat paper text as a replacement for code behavior. If the code disagrees with the paper, report both and call out the mismatch.
- Do not quote long paper passages. Use short excerpts only when necessary and otherwise summarize.

## Resources

- `references/papers.md`: local paper inventory and search guidance.
- `scripts/extract_pdf_text.py`: deterministic wrapper around `pdftotext` with optional query search.
