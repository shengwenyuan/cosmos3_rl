# UR5e post-training - local addition, not part of upstream Cosmos3.

"""Convert RoboMIND2.0 UR5e HDF5 episodes to LeRobot v3.

This converter is intentionally separate from
``tools/convert_robomind_hdf5_to_lerobot.py``.  RoboMIND2.0-UR is large, and the
source tree must never be scanned with a recursive ``rglob("*.hdf5")`` style
fallback.  The only supported source layout is:

    data/ur/<task>/success_episodes/<episode>/data/trajectory.hdf5

The LeRobot feature names match ``RoboMINDUR5Dataset``:

    action.arm_left_joint, action.gripper_left, action.arm_right_joint, action.gripper_right
    observation.state.arm_left_joint, observation.state.gripper_left, ...
    observation.images.camera_front, observation.images.camera_wrist_left, observation.images.camera_wrist_right

Use this in three phases:

1. ``--dry-run-paths --save-episode-manifest ...`` to create a fixed manifest.
2. ``--limit 2 --max-frames-per-episode 8 --out /tmp/... --min-free-gb 0`` for smoke validation.
3. ``--episode-manifest ... --resume --large-run-ack`` for the guarded full conversion.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Literal

import h5py
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
import numpy as np
import tyro
from lerobot.datasets.compute_stats import compute_episode_stats as _lerobot_compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

ActionSource = Literal["puppet", "master"]
OnError = Literal["abort", "skip"]

DEFAULT_SRC = Path("/dexmal-datainfra-swy/modelscope/datasets/X-Humanoid/RoboMIND2.0-UR5")
DEFAULT_OUT = Path("/dexmal-datainfra-swy/modelscope/datasets/X-Humanoid/RoboMIND2.0-UR5-LeRobot/success")
DEFAULT_CAMS = ("camera_front", "camera_wrist_left", "camera_wrist_right")
ALL_CAMS = ("camera_front", "camera_left", "camera_right", "camera_top", "camera_wrist_left", "camera_wrist_right")

ARM_DOF = 6
SIDES = ("left", "right")

F_JOINT_ACTION = {"left": "action.arm_left_joint", "right": "action.arm_right_joint"}
F_GRIPPER_ACTION = {"left": "action.gripper_left", "right": "action.gripper_right"}
F_JOINT_STATE = {"left": "observation.state.arm_left_joint", "right": "observation.state.arm_right_joint"}
F_GRIPPER_STATE = {"left": "observation.state.gripper_left", "right": "observation.state.gripper_right"}


@dataclasses.dataclass
class Args:
    src: Path = DEFAULT_SRC
    """RoboMIND2.0-UR5 dataset root, data/ur root, task dir, success_episodes dir, or one trajectory.hdf5."""

    out: Path = DEFAULT_OUT
    """LeRobot output root. Full conversion defaults to a sibling under /dexmal-datainfra-swy/modelscope/datasets."""

    repo_id: str = "local/robomind2_ur5_dual"
    """LeRobot repo id stored in metadata."""

    cameras: tuple[str, ...] = DEFAULT_CAMS
    """Color cameras to convert. Default matches the RoboMINDUR5Dataset overview+wrist canvas preference."""

    action_source: ActionSource = "puppet"
    """Imitation target stream. observation.state.* is always puppet; action.* comes from this stream."""

    store_hw: tuple[int, int] = (360, 640)
    """(H, W) each camera frame is resized to before LeRobot video encoding."""

    fps: float = 7.0
    """Fallback fps when timestamp-derived fps is unavailable or implausible."""

    estimate_fps: bool = True
    """Estimate fps from the first valid episode timestamps when possible."""

    episode_manifest: Path | None = None
    """Optional newline-delimited trajectory manifest. When set, path scanning is skipped."""

    save_episode_manifest: Path | None = None
    """Optional path to write collected trajectory paths before conversion."""

    dry_run_paths: bool = False
    """Collect/read episode paths, optionally write the manifest and report, then exit before LeRobot output."""

    task_allowlist: tuple[str, ...] | None = None
    """Optional exact task directory names to include."""

    task_denylist: tuple[str, ...] = ()
    """Optional exact task directory names to exclude."""

    path_progress_every: int = 25
    """Print path-scan progress every N task directories. Set 0 to disable."""

    task_override: str | None = None
    """Force one task string for all episodes."""

    limit: int | None = None
    """Optional cap on episode count, required for smoke conversions."""

    max_frames_per_episode: int | None = None
    """Optional cap on frames per episode, useful for writer/API validation."""

    overwrite: bool = False
    """Delete an existing output directory before conversion. Mutually exclusive with resume."""

    resume: bool = False
    """Append to an existing output root and skip source paths already marked saved in the ledger."""

    large_run_ack: bool = False
    """Required for an uncapped conversion to the default large output path."""

    on_error: OnError = "abort"
    """Per-episode error policy. abort logs then raises; skip logs and continues."""

    max_errors: int | None = 20
    """Abort skipped conversion after this many total errors. None disables the guard."""

    max_consecutive_errors: int | None = 5
    """Abort skipped conversion after this many consecutive errors. None disables the guard."""

    min_free_gb: float = 500.0
    """Abort before/after each episode if the filesystem containing out has less free space than this."""

    error_log: Path | None = None
    """JSONL conversion error log. Defaults to <run_root>/logs/conversion_errors.jsonl."""

    ledger_path: Path | None = None
    """JSONL ledger of saved/skipped episodes. Defaults to <run_root>/logs/conversion_ledger.jsonl."""

    heartbeat_path: Path | None = None
    """JSON heartbeat updated before each episode and after progress events."""

    report_path: Path | None = None
    """Markdown report written on dry-run, success, or abort. Defaults to <run_root>/logs/conversion_report.md."""

    progress_every: int = 10
    """Print progress every N saved episodes."""

    image_writer_processes: int = 0
    image_writer_threads: int = 4
    batch_encoding_size: int = 1
    video_codec: str = "libsvtav1"
    video_files_size_mb: int | None = 1
    """LeRobot video shard size in MB. Small shards avoid repeated huge concat work during guarded conversion."""

    data_files_size_mb: int | None = None
    """Optional LeRobot parquet shard size in MB. None keeps the LeRobot default."""

    skip_image_stats: bool = True
    """Skip LeRobot image/video stats. The Cosmos UR5 adapter does not use image stats for training."""

    parallel_encoding: bool = True


@dataclasses.dataclass
class RunStats:
    started_at: float = dataclasses.field(default_factory=time.time)
    status: str = "initializing"
    total: int = 0
    attempted: int = 0
    saved: int = 0
    skipped: int = 0
    total_frames: int = 0
    consecutive_errors: int = 0
    last_path: str | None = None
    last_error: str | None = None
    free_gb_min: float | None = None


class ResourceDanger(RuntimeError):
    """Raised when local resource checks say conversion should stop."""


def _run_root(args: Args) -> Path:
    return args.out.parent if args.out.name == "success" else args.out


def _default_logs(args: Args) -> tuple[Path, Path, Path, Path]:
    log_dir = _run_root(args) / "logs"
    return (
        args.error_log or log_dir / "conversion_errors.jsonl",
        args.ledger_path or log_dir / "conversion_ledger.jsonl",
        args.heartbeat_path or log_dir / "heartbeat.json",
        args.report_path or log_dir / "conversion_report.md",
    )


def _is_traj(path: Path) -> bool:
    return path.is_file() and path.name in {"trajectory.hdf5", "trajectory.h5"}


def _find_ur_root(src: Path) -> Path:
    if src.name == "ur" and src.is_dir():
        return src
    if (src / "data" / "ur").is_dir():
        return src / "data" / "ur"
    if (src / "ur").is_dir():
        return src / "ur"
    raise FileNotFoundError(
        f"Unsupported RoboMIND2-UR source root: {src}. Expected dataset root, data/ur, task dir, "
        "success_episodes dir, or one trajectory.hdf5."
    )


def _iter_episode_paths(args: Args) -> list[Path]:
    src = args.src
    if _is_traj(src):
        return [src]

    if src.name == "success_episodes" and src.is_dir():
        task_dirs = [src.parent]
    elif (src / "success_episodes").is_dir():
        task_dirs = [src]
    else:
        ur_root = _find_ur_root(src)
        allow = set(args.task_allowlist or ())
        deny = set(args.task_denylist or ())
        task_dirs = []
        scanned = 0
        for entry in os.scandir(ur_root):
            if not entry.is_dir():
                continue
            task = entry.name
            if allow and task not in allow:
                continue
            if task in deny:
                continue
            task_dir = Path(entry.path)
            if (task_dir / "success_episodes").is_dir():
                task_dirs.append(task_dir)
            scanned += 1
            if args.path_progress_every > 0 and scanned % args.path_progress_every == 0:
                print(f"Path scan: tasks_seen={scanned} task_dirs={len(task_dirs)} last_task={task}", flush=True)

    paths: list[Path] = []
    for task_dir in sorted(task_dirs):
        success_root = task_dir / "success_episodes"
        if not success_root.is_dir():
            continue
        for ep_entry in sorted(os.scandir(success_root), key=lambda e: e.name):
            if not ep_entry.is_dir():
                continue
            traj = Path(ep_entry.path) / "data" / "trajectory.hdf5"
            if traj.is_file():
                paths.append(traj)
    return paths


def _read_episode_manifest(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_episode_manifest(path: Path, episode_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{p}\n" for p in episode_paths), encoding="utf-8")


def _collect_episode_paths(args: Args) -> list[Path]:
    paths = _read_episode_manifest(args.episode_manifest) if args.episode_manifest else _iter_episode_paths(args)
    return paths[: args.limit] if args.limit is not None else paths


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_text(value.item())
        if value.size:
            return _decode_text(value.reshape(-1)[0])
    return str(value)


def _task_from_path(ep_path: Path) -> str:
    parts = ep_path.parts
    raw = parts[parts.index("success_episodes") - 1] if "success_episodes" in parts else ep_path.parent.name
    return raw.replace("_", " ").strip() or "Perform the manipulation task."


def _task_for_episode(f: h5py.File, ep_path: Path, args: Args) -> str:
    if args.task_override:
        return args.task_override
    default = _task_from_path(ep_path)
    if "metadata" in f and "language_instruction" in f["metadata"].attrs:
        return _decode_text(f["metadata"].attrs["language_instruction"]).strip() or default
    return default


def _require_dataset(f: h5py.File, key: str):
    if key not in f:
        raise KeyError(f"Missing required dataset {key!r}")
    return f[key]


def _validate_cameras(cameras: tuple[str, ...]) -> None:
    unknown = sorted(set(cameras) - set(ALL_CAMS))
    if unknown:
        raise ValueError(f"Unsupported camera names: {unknown}. Supported: {list(ALL_CAMS)}")
    if not cameras:
        raise ValueError("At least one camera is required.")


def _validate_episode_schema(f: h5py.File, cameras: tuple[str, ...], action_source: ActionSource) -> int:
    if "metadata" in f:
        trajectory_length = f["metadata"].attrs.get("trajectory_length")
        if trajectory_length is not None and int(trajectory_length) <= 0:
            raise ValueError(f"metadata trajectory_length is non-positive: {trajectory_length}")

    required = []
    for who in ("puppet", action_source):
        for side in SIDES:
            required.append(f"{who}/arm_{side}_position_align/data")
            required.append(f"{who}/end_effector_{side}_position_align/data")
    required.extend(f"camera_observations/color_images/{cam}" for cam in cameras)
    missing = [key for key in required if key not in f]
    if missing:
        raise KeyError(f"Missing required dataset(s): {missing}")

    lengths = [len(_require_dataset(f, key)) for key in required]
    n = min(lengths)
    if n <= 0:
        raise ValueError(f"Episode produced zero aligned frames; lengths={dict(zip(required, lengths, strict=False))}")

    for side in SIDES:
        for who in ("puppet", action_source):
            arm = _require_dataset(f, f"{who}/arm_{side}_position_align/data")
            grip = _require_dataset(f, f"{who}/end_effector_{side}_position_align/data")
            if arm.shape[-1] != ARM_DOF:
                raise ValueError(f"{who}/arm_{side}_position_align/data width {arm.shape[-1]} != {ARM_DOF}")
            if grip.shape[-1] != 1:
                raise ValueError(f"{who}/end_effector_{side}_position_align/data width {grip.shape[-1]} != 1")
    return n


def _decode_image(elem: Any, store_hw: tuple[int, int]) -> np.ndarray:
    if isinstance(elem, (bytes, bytearray, memoryview, np.bytes_)):
        encoded_bytes = bytes(elem)
    else:
        arr = np.asarray(elem)
        encoded_bytes = arr.astype(np.uint8).tobytes() if arr.ndim == 1 else None
    if encoded_bytes is not None:
        frame = np.asarray(Image.open(io.BytesIO(encoded_bytes)).convert("RGB"), dtype=np.uint8)
    else:
        frame = arr.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
    h, w = store_hw
    if frame.shape[:2] != (h, w):
        frame = np.asarray(Image.fromarray(frame).resize((w, h), Image.BILINEAR), dtype=np.uint8)
    return np.ascontiguousarray(frame)


def _estimate_fps(f: h5py.File, fallback: float) -> float:
    if "camera_observations/timestamp" not in f:
        return fallback
    ts = np.asarray(f["camera_observations/timestamp"][:], dtype=np.float64).ravel()
    if ts.size < 2:
        return fallback
    dur = float(ts[-1] - ts[0])
    if dur <= 0:
        return fallback
    fps = (len(ts) - 1) / dur
    if not np.isfinite(fps) or fps < 1.0 or fps > 120.0:
        return fallback
    return round(fps * 2.0) / 2.0


def _video_feat(store_hw: tuple[int, int]) -> dict:
    h, w = store_hw
    return {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]}


def _f32(n: int) -> dict:
    return {"dtype": "float32", "shape": (n,), "names": None}


def _build_features(cameras: tuple[str, ...], store_hw: tuple[int, int]) -> dict:
    features: dict[str, dict] = {}
    for side in SIDES:
        features[F_JOINT_ACTION[side]] = _f32(ARM_DOF)
        features[F_GRIPPER_ACTION[side]] = _f32(1)
        features[F_JOINT_STATE[side]] = _f32(ARM_DOF)
        features[F_GRIPPER_STATE[side]] = _f32(1)
    for cam in cameras:
        features[f"observation.images.{cam}"] = _video_feat(store_hw)
    return features


def _limit_n(n: int, args: Args) -> int:
    return min(n, int(args.max_frames_per_episode)) if args.max_frames_per_episode is not None else n


def _read_episode(f: h5py.File, ep_path: Path, args: Args):
    n = _limit_n(_validate_episode_schema(f, args.cameras, args.action_source), args)
    if n <= 0:
        raise ValueError("Episode produced zero frames after max_frames_per_episode.")

    state = {
        side: (
            np.asarray(_require_dataset(f, f"puppet/arm_{side}_position_align/data")[:n], dtype=np.float32),
            np.asarray(_require_dataset(f, f"puppet/end_effector_{side}_position_align/data")[:n], dtype=np.float32),
        )
        for side in SIDES
    }
    action = {
        side: (
            np.asarray(_require_dataset(f, f"{args.action_source}/arm_{side}_position_align/data")[:n], dtype=np.float32),
            np.asarray(
                _require_dataset(f, f"{args.action_source}/end_effector_{side}_position_align/data")[:n],
                dtype=np.float32,
            ),
        )
        for side in SIDES
    }
    cams = {cam: _require_dataset(f, f"camera_observations/color_images/{cam}") for cam in args.cameras}
    task = _task_for_episode(f, ep_path, args)

    for t in range(n):
        frame = {
            "task": task,
            **{f"observation.images.{cam}": _decode_image(cams[cam][t], args.store_hw) for cam in args.cameras},
        }
        for side in SIDES:
            frame[F_JOINT_ACTION[side]] = np.asarray(action[side][0][t], dtype=np.float32)
            frame[F_GRIPPER_ACTION[side]] = np.asarray(action[side][1][t], dtype=np.float32).reshape(1)
            frame[F_JOINT_STATE[side]] = np.asarray(state[side][0][t], dtype=np.float32)
            frame[F_GRIPPER_STATE[side]] = np.asarray(state[side][1][t], dtype=np.float32).reshape(1)
        yield frame


def _compute_episode_stats_without_images(episode_data: dict, features: dict) -> dict:
    numeric_features = {
        key: feature for key, feature in features.items() if feature.get("dtype") not in ("image", "video", "string")
    }
    numeric_episode_data = {key: episode_data[key] for key in numeric_features}
    return _lerobot_compute_episode_stats(numeric_episode_data, numeric_features)


def _free_gb(path: Path) -> float:
    target = path if path.exists() else path.parent
    while not target.exists() and target != target.parent:
        target = target.parent
    usage = shutil.disk_usage(target)
    return usage.free / 1024**3


def _check_resources(args: Args, stats: RunStats, when: str) -> None:
    free = _free_gb(args.out)
    stats.free_gb_min = free if stats.free_gb_min is None else min(stats.free_gb_min, free)
    if free < float(args.min_free_gb):
        raise ResourceDanger(f"Free space at {args.out} is {free:.1f} GiB during {when}; min_free_gb={args.min_free_gb}")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_heartbeat(path: Path, args: Args, stats: RunStats, extra: dict | None = None) -> None:
    rec = {
        "status": stats.status,
        "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "elapsed_s": round(time.time() - stats.started_at, 1),
        "total": stats.total,
        "attempted": stats.attempted,
        "saved": stats.saved,
        "skipped": stats.skipped,
        "frames": stats.total_frames,
        "consecutive_errors": stats.consecutive_errors,
        "last_path": stats.last_path,
        "last_error": stats.last_error,
        "free_gb_min": stats.free_gb_min,
        "out": str(args.out),
    }
    if extra:
        rec.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _log_error(error_log: Path, ep_path: Path, exc: BaseException, index: int) -> None:
    _append_jsonl(
        error_log,
        {
            "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "index": index,
            "path": str(ep_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )


def _read_saved_ledger(path: Path) -> set[str]:
    saved: set[str] = set()
    if not path.is_file():
        return saved
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") == "saved" and rec.get("path"):
            saved.add(str(rec["path"]))
    return saved


def _write_report(path: Path, args: Args, stats: RunStats, manifest_count: int, error_log: Path, ledger: Path) -> None:
    args_dict = {k: _json_ready(v) for k, v in dataclasses.asdict(args).items()}
    lines = [
        "# RoboMIND2-UR LeRobot Conversion Report",
        "",
        f"- Status: `{stats.status}`",
        f"- Started UTC: `{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(stats.started_at))}`",
        f"- Elapsed seconds: `{time.time() - stats.started_at:.1f}`",
        f"- Source: `{args.src}`",
        f"- Output: `{args.out}`",
        f"- Manifest entries: `{manifest_count}`",
        f"- Attempted: `{stats.attempted}`",
        f"- Saved: `{stats.saved}`",
        f"- Skipped/errors: `{stats.skipped}`",
        f"- Frames saved: `{stats.total_frames}`",
        f"- Minimum observed free GiB: `{stats.free_gb_min}`",
        f"- Last path: `{stats.last_path}`",
        f"- Last error: `{stats.last_error}`",
        f"- Error log: `{error_log}`",
        f"- Ledger: `{ledger}`",
        f"- LeRobot info: `{args.out / 'meta' / 'info.json'}`",
        "",
        "## Args",
        "",
        "```json",
        json.dumps(args_dict, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _prepare_output(args: Args, ledger: Path, error_log: Path) -> None:
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive.")
    if args.out == DEFAULT_OUT and args.limit is None and not args.dry_run_paths and not args.large_run_ack:
        raise ValueError("Uncapped conversion to the default large output path requires --large-run-ack.")
    if args.out.exists():
        if args.overwrite:
            shutil.rmtree(args.out)
        elif not args.resume:
            raise FileExistsError(f"Output path already exists: {args.out}. Use --resume or --overwrite.")
    elif args.resume:
        raise FileNotFoundError(f"--resume requested but output path does not exist: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and args.out.exists() and not ledger.is_file():
        raise FileNotFoundError(f"--resume requires an existing ledger: {ledger}")
    if not args.resume:
        for path in (ledger, error_log):
            if path is not None and path.exists():
                path.unlink()


def _make_dataset(args: Args, fps: int, features: dict) -> LeRobotDataset:
    if args.resume:
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=args.out,
            batch_encoding_size=args.batch_encoding_size,
            vcodec=args.video_codec,
        )
        got_features = dataset.meta.info.get("features", {})
        missing = [key for key in features if key not in got_features]
        if missing:
            raise ValueError(f"Existing LeRobot dataset is missing expected feature(s): {missing}")
        got_fps = int(round(float(dataset.meta.info.get("fps", fps))))
        if got_fps != int(fps):
            raise ValueError(f"Existing LeRobot fps={got_fps} does not match requested fps={fps}")
        return dataset

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=args.out,
        features=features,
        robot_type="ur5",
        use_videos=True,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
        vcodec=args.video_codec,
    )
    if args.video_files_size_mb is not None or args.data_files_size_mb is not None:
        dataset.meta.update_chunk_settings(
            data_files_size_in_mb=args.data_files_size_mb,
            video_files_size_in_mb=args.video_files_size_mb,
        )
    return dataset


def _find_first_valid(paths: list[Path], args: Args) -> tuple[Path, float]:
    last_error: Exception | None = None
    for path in paths:
        try:
            with h5py.File(path, "r") as f:
                _validate_episode_schema(f, args.cameras, args.action_source)
                fps = _estimate_fps(f, args.fps) if args.estimate_fps else args.fps
                return path, fps
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"No valid episode found in {len(paths)} path(s). Last error: {last_error}") from last_error


def convert(args: Args) -> None:
    _validate_cameras(args.cameras)
    error_log, ledger, heartbeat, report = _default_logs(args)
    stats = RunStats()
    manifest_count = 0
    dataset: LeRobotDataset | None = None
    finalized = False

    try:
        paths = _collect_episode_paths(args)
        if not paths:
            raise FileNotFoundError(f"No RoboMIND2-UR trajectory.hdf5 files found from {args.src}")
        manifest_count = len(paths)
        stats.total = len(paths)

        if args.save_episode_manifest is not None:
            _write_episode_manifest(args.save_episode_manifest, paths)
            print(f"Wrote episode manifest: {args.save_episode_manifest} entries={len(paths)}", flush=True)

        if args.dry_run_paths:
            stats.status = "dry_run_paths_finished"
            print(f"Dry-run path collection: episodes={len(paths)}", flush=True)
            for ep_path in paths[:5]:
                print(f"  sample: {ep_path}", flush=True)
            _write_heartbeat(heartbeat, args, stats)
            return

        _prepare_output(args, ledger, error_log)
        _check_resources(args, stats, "startup")

        if args.skip_image_stats:
            lerobot_dataset_module.compute_episode_stats = _compute_episode_stats_without_images

        first_valid, fps_float = _find_first_valid(paths, args)
        fps = int(round(fps_float))
        features = _build_features(args.cameras, args.store_hw)
        dataset = _make_dataset(args, fps=fps, features=features)
        saved_ledger = _read_saved_ledger(ledger) if args.resume else set()

        print(
            f"Starting RoboMIND2-UR conversion: episodes={len(paths)} fps={fps_float} "
            f"cameras={args.cameras} action_source={args.action_source} first_valid={first_valid} out={args.out}",
            flush=True,
        )
        stats.status = "running"
        _write_heartbeat(heartbeat, args, stats, {"first_valid": str(first_valid), "fps": fps})

        for i, ep_path in enumerate(paths, start=1):
            stats.last_path = str(ep_path)
            if str(ep_path) in saved_ledger:
                continue
            _check_resources(args, stats, "before_episode")
            stats.status = "starting_episode"
            _write_heartbeat(heartbeat, args, stats, {"index": i})

            frame_count = 0
            try:
                stats.attempted += 1
                with h5py.File(ep_path, "r") as f:
                    for frame in _read_episode(f, ep_path, args):
                        dataset.add_frame(frame)
                        frame_count += 1
                    if frame_count <= 0:
                        raise ValueError("Episode produced zero frames.")
                dataset.save_episode(parallel_encoding=args.parallel_encoding)
                stats.saved += 1
                stats.total_frames += frame_count
                stats.consecutive_errors = 0
                stats.status = "progress"
                _append_jsonl(
                    ledger,
                    {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                        "status": "saved",
                        "index": i,
                        "path": str(ep_path),
                        "frames": frame_count,
                        "episode_index": int(dataset.meta.total_episodes) - 1,
                    },
                )
                _check_resources(args, stats, "after_episode")
                if stats.saved == 1 or stats.saved % max(1, args.progress_every) == 0:
                    print(
                        f"[{i}/{len(paths)}] saved={stats.saved} skipped={stats.skipped} "
                        f"frames={stats.total_frames} free_min_gb={stats.free_gb_min:.1f} last={ep_path}",
                        flush=True,
                    )
                    _write_heartbeat(heartbeat, args, stats, {"index": i})
            except Exception as exc:
                stats.skipped += 1
                stats.consecutive_errors += 1
                stats.last_error = f"{type(exc).__name__}: {exc}"
                _log_error(error_log, ep_path, exc, i)
                _append_jsonl(
                    ledger,
                    {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                        "status": "skipped",
                        "index": i,
                        "path": str(ep_path),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                try:
                    if getattr(dataset, "episode_buffer", None) is not None:
                        dataset.clear_episode_buffer(delete_images=True)
                except Exception as cleanup_exc:
                    stats.last_error = f"{stats.last_error}; cleanup={type(cleanup_exc).__name__}: {cleanup_exc}"
                print(f"ERROR episode={ep_path}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

                if args.on_error == "abort":
                    raise
                if args.max_errors is not None and stats.skipped >= args.max_errors:
                    raise RuntimeError(f"Reached max_errors={args.max_errors}; aborting guarded conversion") from exc
                if args.max_consecutive_errors is not None and stats.consecutive_errors >= args.max_consecutive_errors:
                    raise RuntimeError(
                        f"Reached max_consecutive_errors={args.max_consecutive_errors}; aborting guarded conversion"
                    ) from exc

        dataset.finalize()
        finalized = True
        stats.status = "finished"
        _write_heartbeat(heartbeat, args, stats)
        print(
            f"Finished conversion: saved={stats.saved} skipped={stats.skipped} frames={stats.total_frames} "
            f"out={args.out} error_log={error_log if stats.skipped else 'none'} report={report}",
            flush=True,
        )
    except Exception:
        stats.status = "aborted"
        _write_heartbeat(heartbeat, args, stats)
        raise
    finally:
        if dataset is not None and not finalized:
            try:
                dataset.finalize()
            except Exception as finalize_exc:
                stats.last_error = f"{stats.last_error}; finalize={type(finalize_exc).__name__}: {finalize_exc}"
        _write_report(report, args, stats, manifest_count, error_log, ledger)


if __name__ == "__main__":
    try:
        convert(tyro.cli(Args))
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
