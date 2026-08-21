# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""F4 deterministic QC package and human-review gate."""

from __future__ import annotations

import collections
import csv
import io
import json
import random
from pathlib import Path
from typing import Any

import matplotlib
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.droid.fx4_f2 import (
    CAMERA_KEYS,
    F2Config,
    decode_video_samples,
    load_accepted_trajectories,
    load_episode_metadata,
)
from tools.droid.fx4_io import (
    atomic_write_text,
    json_fingerprint,
    load_jsonl,
    sha256_file,
    write_checksums,
    write_json,
)

POSITIVE_FAMILIES = ("pick_place_relocate", "push_pull_slide", "open_close", "stack_arrange")
REJECTION_BUCKETS = (
    "hard_exclude:liquid_pouring",
    "hard_exclude:wipe_clean",
    "hard_exclude:cloth_bag_fold",
    "hard_exclude:appliance",
    "hard_exclude:insertion_fastener",
    "complex_multistage",
)


def deterministic_qc_selection(
    *,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    seed: int = 42,
    positive_per_family: int = 50,
    rejected_per_bucket: int = 15,
    extra_random: int = 30,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()

    def choose(candidates: list[dict[str, Any]], count: int, bucket: str, expected_keep: bool) -> None:
        available = [record for record in candidates if int(record["episode_index"]) not in selected_indices]
        if len(available) < count:
            raise ValueError(f"F4 bucket {bucket} has {len(available)} candidates, requires {count}")
        for record in rng.sample(sorted(available, key=lambda item: int(item["episode_index"])), count):
            episode_index = int(record["episode_index"])
            selected_indices.add(episode_index)
            selected.append(
                {
                    "episode_index": episode_index,
                    "bucket": bucket,
                    "expected_keep": expected_keep,
                    "task_family": record.get("task_family"),
                    "reason_codes": record.get("f1_reason_codes", record.get("reason_codes", [])),
                    "tasks_raw": record.get("tasks_raw", []),
                }
            )

    for family in POSITIVE_FAMILIES:
        choose(
            [record for record in accepted if record["task_family"] == family],
            positive_per_family,
            f"accepted:{family}",
            True,
        )
    for reason in REJECTION_BUCKETS:
        choose(
            [record for record in rejected if reason in record.get("f1_reason_codes", [])],
            rejected_per_bucket,
            reason,
            False,
        )
    remaining = [record for record in [*accepted, *rejected] if int(record["episode_index"]) not in selected_indices]
    if len(remaining) < extra_random:
        raise ValueError(f"F4 random bucket has {len(remaining)} candidates, requires {extra_random}")
    accepted_indices = {int(record["episode_index"]) for record in accepted}
    for record in rng.sample(sorted(remaining, key=lambda item: int(item["episode_index"])), extra_random):
        episode_index = int(record["episode_index"])
        selected_indices.add(episode_index)
        selected.append(
            {
                "episode_index": episode_index,
                "bucket": "random",
                "expected_keep": episode_index in accepted_indices,
                "task_family": record.get("task_family"),
                "reason_codes": record.get("f1_reason_codes", record.get("reason_codes", [])),
                "tasks_raw": record.get("tasks_raw", []),
            }
        )
    return selected


def evaluate_review_gate(rows: list[dict[str, str]]) -> dict[str, Any]:
    labels = [row.get("review_label", "").strip().lower() for row in rows]
    pending = sum(label not in {"correct", "incorrect"} for label in labels)
    if pending:
        return {"status": "pending", "pending_rows": pending, "reviewed_rows": len(rows) - pending}
    incorrect = sum(label == "incorrect" for label in labels)
    family_rates: dict[str, float] = {}
    for family in POSITIVE_FAMILIES:
        family_rows = [row for row in rows if row["bucket"] == f"accepted:{family}"]
        family_rates[family] = sum(row["review_label"].strip().lower() == "incorrect" for row in family_rows) / len(
            family_rows
        )
    overall_rate = incorrect / len(rows)
    passed = overall_rate <= 0.05 and all(rate <= 0.05 for rate in family_rates.values())
    return {
        "status": "pass" if passed else "fail",
        "pending_rows": 0,
        "reviewed_rows": len(rows),
        "overall_error_rate": overall_rate,
        "family_error_rates": family_rates,
    }


def _decode_contact_sheet(
    *, success_root: Path, metadata: dict[str, Any], length: int, title: str
) -> tuple[Image.Image, list[str]]:
    target_frames = [0, max(0, (length - 1) // 2), max(0, length - 1)]
    cell_width, cell_height = 640, 360
    sheet = Image.new("RGB", (cell_width * 3, cell_height * 3 + 90), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title[:240], fill="black")
    failures: list[str] = []
    for row_index, camera in enumerate(CAMERA_KEYS):
        video_meta = {
            "chunk_index": int(metadata[f"videos/{camera}/chunk_index"]),
            "file_index": int(metadata[f"videos/{camera}/file_index"]),
            "from_timestamp": float(metadata[f"videos/{camera}/from_timestamp"]),
        }
        path = success_root / (
            f"videos/{camera}/chunk-{video_meta['chunk_index']:03d}/file-{video_meta['file_index']:03d}.mp4"
        )
        timestamps = [video_meta["from_timestamp"] + frame / 15.0 for frame in target_frames]
        samples = decode_video_samples(path, timestamps, F2Config(), downsample_stride=1)
        for column_index, sample in enumerate(samples):
            if sample["status"] == "ok":
                image = Image.fromarray(sample["thumbnail"]).resize((cell_width, cell_height))
            else:
                image = Image.new("RGB", (cell_width, cell_height), "red")
                ImageDraw.Draw(image).text((10, 10), sample["status"], fill="white")
                failures.append(f"{camera}:{sample['status']}")
            sheet.paste(image, (column_index * cell_width, 90 + row_index * cell_height))
    return sheet, failures


def _save_trajectory_plot(path: Path, trajectory, title: str) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    axes[0].plot(trajectory.raw_xyz[:, 0], label="raw x", alpha=0.45)
    axes[0].plot(trajectory.raw_xyz[:, 1], label="raw y", alpha=0.45)
    axes[0].plot(trajectory.raw_xyz[:, 2], label="raw z", alpha=0.45)
    axes[0].plot(trajectory.smooth_xyz[:, 0], label="smooth x")
    axes[0].plot(trajectory.smooth_xyz[:, 1], label="smooth y")
    axes[0].plot(trajectory.smooth_xyz[:, 2], label="smooth z")
    axes[0].set_ylabel("EEF xyz (m)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].plot(trajectory.gripper_close_fraction, label="source close_fraction")
    axes[1].plot(1.0 - trajectory.gripper_close_fraction, label="model open_fraction", alpha=0.7)
    axes[1].set_xlabel("original 15 Hz frame")
    axes[1].set_ylabel("gripper")
    axes[1].legend(fontsize=8)
    figure.suptitle(title[:160])
    figure.savefig(path, dpi=120)
    plt.close(figure)


def build_f4(
    *, success_root: Path, manifest_dir: Path, taxonomy_path: Path, qc_root: Path, seed: int = 42
) -> dict[str, Any]:
    f1_path = manifest_dir / "f1_episodes.jsonl"
    train_path = manifest_dir / "manifest_train.jsonl"
    val_path = manifest_dir / "manifest_val.jsonl"
    f1_records = load_jsonl(f1_path)
    accepted = [*load_jsonl(train_path), *load_jsonl(val_path)]
    rejected = [record for record in f1_records if record["f1_status"] == "rejected"]
    selection = deterministic_qc_selection(accepted=accepted, rejected=rejected, seed=seed)
    metadata = load_episode_metadata(success_root)
    selected_map = {item["episode_index"]: item for item in selection}
    trajectories = load_accepted_trajectories(success_root, selected_map, metadata, F2Config())

    images_root = qc_root / "images"
    trajectories_root = qc_root / "trajectories"
    images_root.mkdir(parents=True, exist_ok=True)
    trajectories_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    decode_failures: collections.Counter[str] = collections.Counter()
    for index, item in enumerate(selection, start=1):
        episode_index = item["episode_index"]
        raw_tasks = item["tasks_raw"]
        if isinstance(raw_tasks, list):
            title = " | ".join(str(value) for value in raw_tasks)
        else:
            title = str(raw_tasks)
        sheet, failures = _decode_contact_sheet(
            success_root=success_root,
            metadata=metadata[episode_index],
            length=int(metadata[episode_index]["length"]),
            title=f"episode={episode_index} bucket={item['bucket']} {title}",
        )
        sheet_path = images_root / f"episode_{episode_index:06d}.jpg"
        sheet.save(sheet_path, quality=88)
        plot_path = trajectories_root / f"episode_{episode_index:06d}.png"
        _save_trajectory_plot(plot_path, trajectories[episode_index], title)
        decode_failures.update(failures)
        rows.append(
            {
                "episode_index": str(episode_index),
                "bucket": item["bucket"],
                "expected_keep": str(item["expected_keep"]).lower(),
                "task_family": item["task_family"] or "",
                "reason_codes": "|".join(item["reason_codes"]),
                "tasks_raw": json.dumps(raw_tasks, ensure_ascii=False),
                "contact_sheet": str(sheet_path),
                "trajectory_plot": str(plot_path),
                "review_label": "",
                "reviewer": "",
                "review_notes": "",
            }
        )
        if index % 25 == 0:
            print(f"F4 render progress: {index}/{len(selection)} episodes", flush=True)

    qc_csv = manifest_dir / "qc_review.csv"
    with io.StringIO(newline="") as buffer:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_write_text(qc_csv, buffer.getvalue())
    taxonomy_output = manifest_dir / "taxonomy.yaml"
    if taxonomy_output.exists():
        if sha256_file(taxonomy_output) != sha256_file(taxonomy_path):
            raise ValueError("Existing taxonomy.yaml differs from frozen F1 taxonomy")
    else:
        atomic_write_text(taxonomy_output, taxonomy_path.read_text())

    rejection_counts = collections.Counter(
        reason
        for record in f1_records
        for reason in record.get("f1_reason_codes", [])
        if record["f1_status"] == "rejected"
    )
    rejected_summary_path = manifest_dir / "rejected_summary.json"
    write_json(rejected_summary_path, {"f1_reason_counts": dict(sorted(rejection_counts.items()))})
    bucket_counts = collections.Counter(item["bucket"] for item in selection)
    coverage_pass = (
        len(selection) == 320
        and all(bucket_counts[f"accepted:{family}"] == 50 for family in POSITIVE_FAMILIES)
        and all(bucket_counts[reason] == 15 for reason in REJECTION_BUCKETS)
        and bucket_counts["random"] == 30
    )
    review_gate = evaluate_review_gate(rows)
    summary = {
        "stage": "f4",
        "counts": {
            "review_rows": len(rows),
            "buckets": dict(sorted(bucket_counts.items())),
            "render_failures": dict(sorted(decode_failures.items())),
        },
        "gates": {"coverage_pass": coverage_pass, "semantic_review": review_gate},
        "inputs": {
            "manifest_train": {"path": str(train_path), "sha256": sha256_file(train_path)},
            "manifest_val": {"path": str(val_path), "sha256": sha256_file(val_path)},
            "f1_episodes": {"path": str(f1_path), "sha256": sha256_file(f1_path)},
            "taxonomy": {"path": str(taxonomy_path), "sha256": sha256_file(taxonomy_path)},
        },
    }
    summary["input_fingerprint"] = json_fingerprint({"inputs": summary["inputs"], "seed": seed})
    f4_summary_path = manifest_dir / "f4_summary.json"
    write_json(f4_summary_path, summary)
    write_checksums(manifest_dir / "F4_SHA256SUMS", (f4_summary_path, qc_csv, taxonomy_output, rejected_summary_path))
    return summary


def read_review_gate(qc_csv: Path) -> dict[str, Any]:
    with qc_csv.open(newline="") as handle:
        return evaluate_review_gate(list(csv.DictReader(handle)))
