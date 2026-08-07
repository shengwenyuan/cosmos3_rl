# Local Paper Inventory

All paths are relative to the repository root `/root/code/cosmos-framework`.

## Papers

| PDF | Size | Role | Use for |
| --- | ---: | --- | --- |
| `tmps/cosmos3_main.pdf` | 9.97 MiB | NVIDIA official Cosmos3 paper mapped to this repo/runtime | Cosmos3 model architecture, post-training, inference, action-policy design, terminology, paper-to-code validation |
| `tmps/RoboMind.pdf` | 3.50 MiB | RoboMIND paper | RoboMIND v1 dataset background, robot embodiments, data organization, task/language/action conventions |
| `tmps/RoboMIND2.pdf` | 17.20 MiB | RoboMIND 2.0 paper | bimanual/mobile manipulation dataset details, RoboMIND 2.0 schema/context, UR5-related dataset claims |

## Tool Status On This Device

Detected command-line tools:

- `/usr/bin/pdfinfo`
- `/usr/bin/pdftotext`
- `/usr/bin/pdftoppm`
- `/usr/bin/python3`

Not detected at skill creation time: `mutool`, `qpdf`, `pdfgrep`. Poppler is sufficient for the current paper sizes.

`tmps/RoboMIND2.pdf` was validated with `pdfinfo` and `pdftotext -layout`: 66 pages, 18,033,189 bytes, text extraction completed in about 0.27 seconds and produced about 149 KB of text.

## Search Workflow

1. Extract one or more PDFs:

   ```bash
   python .agents/skills/cosmos3-paper-reference/scripts/extract_pdf_text.py tmps/cosmos3_main.pdf tmps/RoboMIND2.pdf
   ```

2. Search the generated text files:

   ```bash
   rg -n "post-training|action|UR5|RoboMIND" /tmp/cosmos3-paper-text-cache
   ```

3. For page-specific checks, use `pdfinfo` for page count and render suspect pages with `pdftoppm`:

   ```bash
   pdftoppm -f 3 -l 3 -png tmps/cosmos3_main.pdf /tmp/cosmos3_page
   ```

4. Cite evidence as `PDF:page` in prose and local repo files as `file:line`.

## Query Seeds

- Cosmos3/action-policy: `action`, `policy`, `post-training`, `robot`, `DROID`, `chunk`, `state`, `embodiment`.
- RoboMIND/UR5: `UR5`, `bimanual`, `joint`, `gripper`, `camera`, `language instruction`, `HDF5`, `LeRobot`.
- Validation: `dataset`, `schema`, `evaluation`, `benchmark`, `ablation`, `training`.
