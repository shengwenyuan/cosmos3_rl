#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Batch runner for Cosmos3 RoboLab atomic_pick and atomic_pnp tasks."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("/mlp_vepfs/share/swy/cosmos3-framework/client_outputs")
DEFAULT_WRAPPER = DEFAULT_OUTPUT_ROOT / "_tools" / "cosmos3_run_variant.py"
DEFAULT_TASK_ROOTS = ("robolab/tasks/atomic_pnp", "robolab/tasks/atomic_pick")
DEFAULT_REMOTE_HOST = "10.174.241.114"
DEFAULT_REMOTE_PORT = 8000
PI_OVER_2 = "1.5707963267948966"
ROW_TIMEOUT_S = 3 * 60 * 60

STATUS_FIELDS = [
    "task",
    "trial",
    "condition",
    "rc",
    "asset_errors",
    "status",
    "duration_s",
    "log_path",
    "output_dir",
    "cmd",
    "first_error",
]

SERIOUS_PATTERNS = [
    re.compile(
        r"\[Error\].*(References an asset that can not be found|Couldn't process file|Failed to load image|STB Failed|asset|texture|usd|render)",
        re.IGNORECASE,
    ),
    re.compile(r"Terminated with error", re.IGNORECASE),
    re.compile(r"Traceback", re.IGNORECASE),
    re.compile(r"Segmentation fault", re.IGNORECASE),
    re.compile(r"core dumped", re.IGNORECASE),
    re.compile(r"Could not open|Failed to open|Failed to load|Unable to open|Could not load", re.IGNORECASE),
]

WHITELIST_PATTERNS = [
    re.compile(r"GLFW initialization failed", re.IGNORECASE),
    re.compile(r"Failed to startup plugin carb\.windowing-glfw\.plugin", re.IGNORECASE),
    re.compile(r"failed to open the default display", re.IGNORECASE),
    re.compile(r"Source: omni\.hydra was already registered", re.IGNORECASE),
    re.compile(r"omni\.isaac\.dynamic_control is deprecated", re.IGNORECASE),
    re.compile(r"No material configuration file", re.IGNORECASE),
    re.compile(r"Parameter .* not available in the MDL representation", re.IGNORECASE),
    re.compile(r"DLSS increasing input dimensions", re.IGNORECASE),
    re.compile(r"using high frequency span with attrs is disabled", re.IGNORECASE),
    re.compile(r"ImplicitActuatorCfg effort_limit/velocity_limit warnings", re.IGNORECASE),
    re.compile(r"Not all actuators are configured", re.IGNORECASE),
]


def beijing_timestamp() -> str:
    return (dt.datetime.utcnow() + dt.timedelta(hours=8)).strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("/root/code/RoboLab"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wrapper", type=Path, default=DEFAULT_WRAPPER)
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-port", type=int, default=DEFAULT_REMOTE_PORT)
    parser.add_argument("--task-roots", nargs="+", default=list(DEFAULT_TASK_ROOTS))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--row-timeout-s", type=int, default=ROW_TIMEOUT_S)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def task_files(repo_root: Path, task_roots: list[str]) -> list[Path]:
    paths: list[Path] = []
    for root in task_roots:
        paths.extend(sorted((repo_root / root).glob("*.py")))
    return paths


def task_slug(task: Path) -> str:
    parent = task.parent.name
    return f"{parent}/{task.stem}"


def row_id(task: Path, condition: str, trial: int) -> tuple[str, str, str]:
    return (task_slug(task), condition, str(trial))


def command_for_row(
    *,
    args: argparse.Namespace,
    task: Path,
    condition: str,
    trial: int,
    run_dir: Path,
    row_index: int,
) -> list[str]:
    rel_task = task.relative_to(args.repo_root).as_posix()
    out_dir = run_dir / task_slug(task) / condition / f"trial_{trial:02d}"
    env_seed = 1
    lighting_profile = "base"
    lighting_seed = args.seed + row_index
    extra: list[str] = []

    if condition == "pose_xy40_yawpi2":
        env_seed = args.seed + 1000 + row_index
        extra.extend(
            [
                "--randomize-contact-pose",
                "--contact-pose-xy-range",
                "0.4",
                "--contact-pose-yaw-range",
                PI_OVER_2,
            ]
        )
    elif condition == "lighting_random":
        env_seed = args.seed + 2000 + row_index
        lighting_profile = "random"
    elif condition != "baseline":
        raise ValueError(f"unknown condition: {condition}")

    return [
        "/workspace/isaaclab/isaaclab.sh",
        "-p",
        str(args.wrapper),
        "--task",
        rel_task,
        "--remote-host",
        args.remote_host,
        "--remote-port",
        str(args.remote_port),
        "--headless",
        "--video-mode",
        "all",
        "--num-envs",
        "1",
        "--num-runs",
        "1",
        "--output-folder-name",
        str(out_dir),
        "--lighting-profile",
        lighting_profile,
        "--lighting-seed",
        str(lighting_seed),
        "--env-seed",
        str(env_seed),
        *extra,
    ]


def scan_log(log_path: Path) -> tuple[int, str]:
    count = 0
    first = ""
    if not log_path.exists():
        return 0, ""
    with log_path.open("r", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if any(pattern.search(line) for pattern in WHITELIST_PATTERNS):
                continue
            if any(pattern.search(line) for pattern in SERIOUS_PATTERNS):
                count += 1
                if not first:
                    first = line[:500]
    return count, first


def load_completed(status_path: Path) -> tuple[set[tuple[str, str, str]], set[str]]:
    completed: set[tuple[str, str, str]] = set()
    asset_blocked_tasks: set[str] = set()
    if not status_path.exists():
        return completed, asset_blocked_tasks
    with status_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            key = (row["task"], row["condition"], row["trial"])
            if row["status"] == "ok" and row["rc"] == "0" and row["asset_errors"] == "0":
                completed.add(key)
            if row["status"] in {"asset_render_error", "skipped_asset_render_error"}:
                asset_blocked_tasks.add(row["task"])
    return completed, asset_blocked_tasks


def append_status(status_path: Path, row: dict[str, object]) -> None:
    exists = status_path.exists()
    with status_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATUS_FIELDS, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in STATUS_FIELDS})
        fh.flush()
        os.fsync(fh.fileno())


def write_manifest(manifest_path: Path, rows: list[dict[str, object]]) -> None:
    with manifest_path.open("w", newline="") as fh:
        fieldnames = ["task", "condition", "trial", "cmd"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resource_snapshot(path: Path) -> dict[str, object]:
    output_usage = shutil.disk_usage(path)
    root_usage = shutil.disk_usage("/")
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return {
        "output_free_gb": round(output_usage.free / (1024**3), 2),
        "root_free_gb": round(root_usage.free / (1024**3), 2),
        "gpu": gpu.stdout.strip(),
    }


def resource_ok(snapshot: dict[str, object]) -> tuple[bool, str]:
    if float(snapshot["output_free_gb"]) < 100.0:
        return False, f"low output disk free: {snapshot['output_free_gb']} GB"
    if float(snapshot["root_free_gb"]) < 3.0:
        return False, f"low root disk free: {snapshot['root_free_gb']} GB"
    return True, ""


def run_command(cmd: list[str], log_path: Path, timeout_s: int, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_file:
        log_file.write(("CMD " + shlex.join(cmd) + "\n").encode())
        log_file.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        try:
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                return proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                return 124


def mp4_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.mp4"))


def main() -> int:
    args = parse_args()
    if args.run_name is None:
        args.run_name = f"atomic_pick_pnp_{beijing_timestamp()}"

    run_dir = args.output_root / args.run_name
    logs_dir = run_dir / "_batch_logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    tasks = task_files(args.repo_root, args.task_roots)
    if not tasks:
        print("[batch] no tasks discovered", file=sys.stderr)
        return 2

    rows: list[dict[str, object]] = []
    row_index = 0
    for task in tasks:
        for condition in ("baseline", "pose_xy40_yawpi2", "lighting_random"):
            for trial in range(args.trials):
                cmd = command_for_row(
                    args=args,
                    task=task,
                    condition=condition,
                    trial=trial,
                    run_dir=run_dir,
                    row_index=row_index,
                )
                rows.append(
                    {
                        "task": task_slug(task),
                        "condition": condition,
                        "trial": str(trial),
                        "cmd": shlex.join(cmd),
                        "_task_path": task,
                        "_row_index": row_index,
                    }
                )
                row_index += 1

    write_manifest(run_dir / "manifest.tsv", rows)
    with (run_dir / "batch_config.json").open("w") as fh:
        json.dump(
            {
                "repo_root": str(args.repo_root),
                "output_root": str(args.output_root),
                "run_name": args.run_name,
                "remote_host": args.remote_host,
                "remote_port": args.remote_port,
                "task_roots": args.task_roots,
                "trials": args.trials,
                "seed": args.seed,
                "row_timeout_s": args.row_timeout_s,
                "total_rows": len(rows),
            },
            fh,
            indent=2,
            sort_keys=True,
        )

    status_path = run_dir / "batch_status.tsv"
    completed, asset_blocked_tasks = load_completed(status_path)
    print(f"[batch] run_dir={run_dir}")
    print(f"[batch] rows={len(rows)} completed={len(completed)} asset_blocked={len(asset_blocked_tasks)}")
    if args.dry_run:
        print("[batch] dry run complete")
        return 0

    for row in rows:
        task = str(row["task"])
        condition = str(row["condition"])
        trial = str(row["trial"])
        key = (task, condition, trial)
        cmd = shlex.split(str(row["cmd"]))
        output_dir = cmd[cmd.index("--output-folder-name") + 1]
        log_path = logs_dir / task.replace("/", "__") / condition / f"trial_{int(trial):02d}.log"

        if key in completed:
            print(f"[batch] skip completed task={task} condition={condition} trial={trial}")
            continue

        if task in asset_blocked_tasks:
            append_status(
                status_path,
                {
                    "task": task,
                    "trial": trial,
                    "condition": condition,
                    "rc": 999,
                    "asset_errors": 0,
                    "status": "skipped_asset_render_error",
                    "duration_s": 0,
                    "log_path": str(log_path),
                    "output_dir": output_dir,
                    "cmd": row["cmd"],
                    "first_error": "prior asset/render/import error for task",
                },
            )
            print(f"[batch] skip asset-blocked task={task} condition={condition} trial={trial}")
            continue

        snapshot = resource_snapshot(args.output_root)
        ok, reason = resource_ok(snapshot)
        if not ok:
            append_status(
                status_path,
                {
                    "task": task,
                    "trial": trial,
                    "condition": condition,
                    "rc": 998,
                    "asset_errors": 0,
                    "status": "skipped_resource",
                    "duration_s": 0,
                    "log_path": str(log_path),
                    "output_dir": output_dir,
                    "cmd": row["cmd"],
                    "first_error": reason,
                },
            )
            print(f"[batch] resource skip task={task} condition={condition} trial={trial}: {reason}")
            continue

        print(
            f"[batch] start task={task} condition={condition} trial={trial} "
            f"root_free_gb={snapshot['root_free_gb']} output_free_gb={snapshot['output_free_gb']} gpu={snapshot['gpu']}"
        )
        start = time.monotonic()
        rc = run_command(cmd, log_path=log_path, timeout_s=args.row_timeout_s, cwd=args.repo_root)
        duration_s = round(time.monotonic() - start, 2)
        asset_errors, first_error = scan_log(log_path)
        videos = mp4_count(Path(output_dir))

        if asset_errors:
            status = "asset_render_error"
            asset_blocked_tasks.add(task)
        elif rc == 0:
            status = "ok"
        else:
            status = "timeout" if rc == 124 else "error"

        append_status(
            status_path,
            {
                "task": task,
                "trial": trial,
                "condition": condition,
                "rc": rc,
                "asset_errors": asset_errors,
                "status": status,
                "duration_s": duration_s,
                "log_path": str(log_path),
                "output_dir": output_dir,
                "cmd": row["cmd"],
                "first_error": first_error,
            },
        )
        print(
            f"[batch] done task={task} condition={condition} trial={trial} "
            f"rc={rc} asset_errors={asset_errors} status={status} duration_s={duration_s} videos={videos}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
