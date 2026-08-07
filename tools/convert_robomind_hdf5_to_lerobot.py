# UR5 post-training - local addition, not part of upstream Cosmos3.

"""Convert RoboMIND UR HDF5 episodes to LeRobot for Cosmos action-policy SFT.

The primary target is RoboMIND1-UR ``h5_ur_1rgb`` data stored as one
``trajectory.hdf5`` per episode::

    benchmark*_compressed/h5_ur_1rgb/<task>/success_episodes/<train|val>/<episode>/data/trajectory.hdf5
    benchmark*_compressed/h5_ur_1rgb/<task>/success_episodes/<episode>/data/trajectory.hdf5

Observed RoboMIND1-UR schema:

    puppet/joint_position              (T, 7) float64  # [6 UR joints, gripper]
    master/joint_position              (T, 7) float64  # optional leader command stream
    puppet/end_effector                (T, 6) float64  # present, ignored for joint-space conversion
    observations/rgb_images/camera_top (T,) object     # JPEG bytes
    language_raw                       (1,) object     # natural-language task, bytes

The output feature names match ``RoboMINDUR5Dataset`` and the
``action_policy_robomind_ur5_single_nano`` recipe.

This script still keeps the older RoboMIND 2 dual-arm path, but the guarded full
conversion flow should be run separately for each source format.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Iterable, Literal

import h5py
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
import numpy as np
import tyro
from lerobot.datasets.compute_stats import compute_episode_stats as _lerobot_compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

Fmt = Literal["auto", "ur_1rgb", "mind2_dual"]
ActionSource = Literal["puppet", "master"]
BgrMode = Literal["auto", "true", "false"]
OnError = Literal["abort", "skip"]

_ARM_DOF = 6
_MIND2_CAMS = ("camera_top", "camera_front", "camera_wrist_left", "camera_wrist_right")

# LeRobot destination feature names (must match robomind_ur5_dataset.py).
_F_JOINT_ACTION = {"left": "action.arm_left_joint", "right": "action.arm_right_joint"}
_F_GRIPPER_ACTION = {"left": "action.gripper_left", "right": "action.gripper_right"}
_F_JOINT_STATE = {"left": "observation.state.arm_left_joint", "right": "observation.state.arm_right_joint"}
_F_GRIPPER_STATE = {"left": "observation.state.gripper_left", "right": "observation.state.gripper_right"}


@dataclasses.dataclass
class Args:
    src: Path
    """Source file or directory. For RoboMIND1-UR this can be the dataset root, benchmark root, h5_ur_1rgb root, or one trajectory.hdf5."""

    out: Path
    """Output LeRobot root, e.g. /mlp_vepfs/share/swy/cosmos3-framework/lerobot/RoboMIND1-ur5."""

    repo_id: str = "local/robomind1_ur5"
    """LeRobot repo id stored in metadata for the local dataset."""

    format: Fmt = "auto"
    """Source format. Use ur_1rgb for RoboMIND1-UR; auto detects from the first episode."""

    action_source: ActionSource = "puppet"
    """Imitation target stream. observation.state.* is always puppet; action.* comes from this stream."""

    store_hw: tuple[int, int] = (360, 640)
    """(H, W) every camera is resized to before video encoding."""

    fps: float = 15.0
    """fps for ur_1rgb, which has no timestamps in-file. mind2_dual estimates from timestamps."""

    splits: tuple[str, ...] = ("train", "val")
    """RoboMIND1-UR success_episodes splits to include. Splitless episode dirs are always included."""

    episode_manifest: Path | None = None
    """Optional newline-delimited trajectory manifest. When set, path scanning is skipped."""

    save_episode_manifest: Path | None = None
    """Optional path to write the collected trajectory list before conversion."""

    dry_run_paths: bool = False
    """Collect/read episode paths, optionally write the manifest, then exit before creating LeRobot output."""

    path_progress_every: int = 25
    """Print path-scan progress every N task directories. Set 0 to disable."""

    task_override: str | None = None
    """Force the task string for all episodes."""

    bgr: BgrMode = "auto"
    """Raw image channel order. JPEG bytes are decoded by PIL as RGB and are not swapped."""

    limit: int | None = None
    """Optional cap on the number of episodes, useful for smoke conversions."""

    max_frames_per_episode: int | None = None
    """Optional cap on frames per episode, useful for fast writer/API validation."""

    overwrite: bool = False
    """Delete an existing output directory before conversion. Refuses to overwrite by default."""

    on_error: OnError = "abort"
    """Per-episode error policy. abort logs then raises; skip logs and continues."""

    max_errors: int | None = 20
    """Abort skipped conversion after this many per-episode errors. None disables the guard."""

    error_log: Path | None = None
    """JSONL file for conversion errors. Defaults to <out>/conversion_errors.jsonl."""

    heartbeat_path: Path | None = None
    """Optional JSON heartbeat updated before each episode and after progress events for external supervision."""

    progress_every: int = 25
    """Print progress every N successfully saved episodes."""

    image_writer_processes: int = 0
    image_writer_threads: int = 4
    batch_encoding_size: int = 1
    video_codec: str = "libsvtav1"
    video_files_size_mb: int | None = None
    """LeRobot video shard size in MB. Use 1 for many small per-episode videos and no repeated concat."""

    data_files_size_mb: int | None = None
    """Optional LeRobot parquet shard size in MB. None keeps the LeRobot default."""

    skip_image_stats: bool = False
    """Skip LeRobot per-episode image/video stats. The Cosmos UR5 adapter does not use them for training."""

    parallel_encoding: bool = True


# ----------------------------- path collection -----------------------------------------------
def _is_traj(path: Path) -> bool:
    return path.is_file() and path.name.endswith((".hdf5", ".h5"))


def _iter_ur_1rgb_paths(src: Path, splits: Iterable[str], progress_every: int = 0) -> list[Path]:
    """Collect RoboMIND1-UR trajectories without recursively walking archives or unrelated folders."""
    if _is_traj(src):
        return [src]

    h5_roots: list[Path] = []
    if src.name == "h5_ur_1rgb" and src.is_dir():
        h5_roots.append(src)
    if (src / "h5_ur_1rgb").is_dir():
        h5_roots.append(src / "h5_ur_1rgb")
    h5_roots.extend(sorted(p for p in src.glob("benchmark*_compressed/h5_ur_1rgb") if p.is_dir()))

    seen: set[Path] = set()
    paths: list[Path] = []
    split_set = tuple(splits)
    split_names = set(split_set)
    task_count = 0
    for h5root in h5_roots:
        for task_entry in os.scandir(h5root):
            if not task_entry.is_dir():
                continue
            success_root = Path(task_entry.path) / "success_episodes"
            if not success_root.is_dir():
                continue
            task_count += 1
            for split in split_set:
                split_dir = success_root / split
                if not split_dir.is_dir():
                    continue
                for ep_entry in os.scandir(split_dir):
                    if not ep_entry.is_dir():
                        continue
                    traj = Path(ep_entry.path) / "data" / "trajectory.hdf5"
                    if traj.is_file() and traj not in seen:
                        seen.add(traj)
                        paths.append(traj)
            # Some RoboMIND1-UR dumps place episodes directly under success_episodes
            # without a train/val split directory. Include those alongside the requested splits.
            for ep_entry in os.scandir(success_root):
                if not ep_entry.is_dir() or ep_entry.name in split_names:
                    continue
                traj = Path(ep_entry.path) / "data" / "trajectory.hdf5"
                if traj.is_file() and traj not in seen:
                    seen.add(traj)
                    paths.append(traj)
            if progress_every > 0 and task_count % progress_every == 0:
                print(
                    f"Path scan: tasks={task_count} trajectories={len(paths)} last_task={Path(task_entry.path).name}",
                    flush=True,
                )
    return sorted(paths)


def _read_episode_manifest(path: Path) -> list[Path]:
    paths = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return paths


def _write_episode_manifest(path: Path, episode_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{p}\n" for p in episode_paths), encoding="utf-8")


def _collect_episode_paths(args: Args) -> list[Path]:
    if args.episode_manifest is not None:
        paths = _read_episode_manifest(args.episode_manifest)
        return paths[: args.limit] if args.limit is not None else paths
    if args.format in ("auto", "ur_1rgb"):
        paths = _iter_ur_1rgb_paths(args.src, args.splits, args.path_progress_every)
        if paths:
            return paths[: args.limit] if args.limit is not None else paths
    if _is_traj(args.src):
        paths = [args.src]
    else:
        paths = sorted(args.src.rglob("*.hdf5"))
    return paths[: args.limit] if args.limit is not None else paths


# ----------------------------- format detection -----------------------------------------------
def _detect_format(f: h5py.File) -> str:
    if "metadata" in f and "puppet/arm_left_position_align/data" in f:
        return "mind2_dual"
    if "puppet/joint_position" in f:
        w = int(np.asarray(f["puppet/joint_position"].shape)[-1])
        if w == 7:
            return "ur_1rgb"
        raise ValueError(f"puppet/joint_position width {w} != 7; not a single-arm UR h5_ur_1rgb file")
    raise ValueError("Unrecognized RoboMIND HDF5 layout: no mind2 *_position_align and no puppet/joint_position")


# ----------------------------- image decoding --------------------------------------------------
def _decode_image(elem, store_hw: tuple[int, int], bgr: bool) -> np.ndarray:
    """Decode one stored frame to uint8 RGB [H, W, 3] at store_hw."""
    if isinstance(elem, (bytes, bytearray, memoryview, np.bytes_)):
        encoded_bytes = bytes(elem)
    else:
        arr = np.asarray(elem)
        encoded_bytes = arr.astype(np.uint8).tobytes() if arr.ndim == 1 else None
    if encoded_bytes is not None:
        # PIL returns RGB for JPEG/PNG bytes. Do not BGR-swap encoded frames.
        img = Image.open(io.BytesIO(encoded_bytes)).convert("RGB")
        frame = np.asarray(img, dtype=np.uint8)
    else:
        frame = arr.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        if bgr:
            frame = frame[..., ::-1]
    h, w = store_hw
    if frame.shape[:2] != (h, w):
        frame = np.asarray(Image.fromarray(frame).resize((w, h), Image.BILINEAR), dtype=np.uint8)
    return np.ascontiguousarray(frame)


# ----------------------------- task / fps ------------------------------------------------------
def _decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_text(value.item())
        if value.size:
            return _decode_text(value.reshape(-1)[0])
    return str(value)


def _task_from_metadata(f: h5py.File, default: str) -> str:
    if "metadata" in f and "language_instruction" in f["metadata"].attrs:
        return _decode_text(f["metadata"].attrs["language_instruction"]).strip() or default
    if "language_raw" in f and len(f["language_raw"]):
        return _decode_text(f["language_raw"][0]).strip() or default
    return default


def _task_from_path(ep_path: Path, override: str | None) -> str:
    if override:
        return override
    parts = ep_path.parts
    raw = parts[parts.index("success_episodes") - 1] if "success_episodes" in parts else ep_path.parent.name
    raw = re.sub(r"_copy_\d+$", "", raw)
    raw = re.sub(r"_\d+$", "", raw)
    return raw.replace("_", " ").strip() or "Perform the manipulation task."


def _task_for_episode(f: h5py.File, ep_path: Path, args: Args) -> str:
    return (
        _task_from_metadata(f, _task_from_path(ep_path, args.task_override))
        if args.task_override is None
        else args.task_override
    )


def _estimate_fps(f: h5py.File, override: float) -> float:
    if "camera_observations/timestamp" not in f:
        return override
    ts = np.asarray(f["camera_observations/timestamp"][:]).astype(np.float64).ravel()
    dur = float(ts[-1] - ts[0])
    return round((len(ts) - 1) / dur * 2) / 2 if dur > 0 else override


# ----------------------------- feature schemas -------------------------------------------------
def _video_feat(store_hw: tuple[int, int]) -> dict:
    h, w = store_hw
    return {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]}


def _f32(n: int) -> dict:
    return {"dtype": "float32", "shape": (n,), "names": None}


def _build_features(fmt: str, store_hw: tuple[int, int]) -> dict:
    feats: dict = {}
    sides = ["left", "right"] if fmt == "mind2_dual" else ["left"]
    for side in sides:
        feats[_F_JOINT_ACTION[side]] = _f32(_ARM_DOF)
        feats[_F_GRIPPER_ACTION[side]] = _f32(1)
        feats[_F_JOINT_STATE[side]] = _f32(_ARM_DOF)
        feats[_F_GRIPPER_STATE[side]] = _f32(1)
    cams = _MIND2_CAMS if fmt == "mind2_dual" else ("camera_top",)
    for cam in cams:
        feats[f"observation.images.{cam}"] = _video_feat(store_hw)
    return feats


# ----------------------------- per-episode readers ---------------------------------------------
def _limit_n(n: int, args: Args) -> int:
    return min(n, int(args.max_frames_per_episode)) if args.max_frames_per_episode is not None else n


def _require_dataset(f: h5py.File, key: str):
    if key not in f:
        raise KeyError(f"Missing required dataset {key!r}")
    return f[key]


def _read_ur_1rgb(f: h5py.File, ep_path: Path, args: Args):
    """Yield frame dicts for RoboMIND1-UR, writing the single arm into the left slot."""
    jp_state = np.asarray(_require_dataset(f, "puppet/joint_position"), dtype=np.float32)
    action_key = f"{args.action_source}/joint_position"
    jp_action = np.asarray(_require_dataset(f, action_key), dtype=np.float32)
    cam = _require_dataset(f, "observations/rgb_images/camera_top")
    task = _task_for_episode(f, ep_path, args)
    n = _limit_n(min(len(jp_state), len(jp_action), len(cam)), args)
    bgr = _use_bgr(args, "ur_1rgb")
    for t in range(n):
        yield {
            "task": task,
            "observation.images.camera_top": _decode_image(cam[t], args.store_hw, bgr=bgr),
            _F_JOINT_ACTION["left"]: jp_action[t, :_ARM_DOF],
            _F_GRIPPER_ACTION["left"]: jp_action[t, _ARM_DOF : _ARM_DOF + 1],
            _F_JOINT_STATE["left"]: jp_state[t, :_ARM_DOF],
            _F_GRIPPER_STATE["left"]: jp_state[t, _ARM_DOF : _ARM_DOF + 1],
        }


def _read_mind2_dual(f: h5py.File, ep_path: Path, args: Args):
    """Yield frame dicts for RoboMIND 2.0 dual-arm UR."""

    def arm(who, side):
        return np.asarray(_require_dataset(f, f"{who}/arm_{side}_position_align/data"), dtype=np.float32)

    def grip(who, side):
        return np.asarray(
            _require_dataset(f, f"{who}/end_effector_{side}_position_align/data"), dtype=np.float32
        ).reshape(-1, 1)

    state = {s: (arm("puppet", s), grip("puppet", s)) for s in ("left", "right")}
    action = {s: (arm(args.action_source, s), grip(args.action_source, s)) for s in ("left", "right")}
    cams = {c: _require_dataset(f, f"camera_observations/color_images/{c}") for c in _MIND2_CAMS}
    task = _task_for_episode(f, ep_path, args)
    bgr = _use_bgr(args, "mind2_dual")
    n = min(min(len(v[0]) for v in state.values()), min(len(cams[c]) for c in _MIND2_CAMS))
    n = _limit_n(n, args)
    for t in range(n):
        frame = {
            "task": task,
            **{f"observation.images.{c}": _decode_image(cams[c][t], args.store_hw, bgr=bgr) for c in _MIND2_CAMS},
        }
        for side in ("left", "right"):
            frame[_F_JOINT_ACTION[side]] = action[side][0][t]
            frame[_F_GRIPPER_ACTION[side]] = action[side][1][t]
            frame[_F_JOINT_STATE[side]] = state[side][0][t]
            frame[_F_GRIPPER_STATE[side]] = state[side][1][t]
        yield frame


def _use_bgr(args: Args, fmt: str) -> bool:
    if args.bgr == "true":
        return True
    if args.bgr == "false":
        return False
    # Current RoboMIND1-UR samples store camera_top as JPEG bytes; those decode to RGB through PIL.
    # Keep auto=false for encoded streams. Raw-array legacy files can opt in with --bgr true.
    return False if fmt == "ur_1rgb" else False


# ----------------------------- conversion ------------------------------------------------------
def _prepare_output(args: Args) -> None:
    if args.out.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output path already exists: {args.out}. Pass --overwrite to delete it first.")
        shutil.rmtree(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)


def _log_error(log_path: Path, ep_path: Path, exc: BaseException) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "path": str(ep_path),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_heartbeat(path: Path | None, rec: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _compute_episode_stats_without_images(episode_data: dict, features: dict) -> dict:
    numeric_features = {
        key: feature for key, feature in features.items() if feature.get("dtype") not in ("image", "video", "string")
    }
    numeric_episode_data = {key: episode_data[key] for key in numeric_features}
    return _lerobot_compute_episode_stats(numeric_episode_data, numeric_features)


def _read_episode(fmt: str, f: h5py.File, ep_path: Path, args: Args):
    return _read_ur_1rgb(f, ep_path, args) if fmt == "ur_1rgb" else _read_mind2_dual(f, ep_path, args)


def convert(args: Args) -> None:
    episode_paths = _collect_episode_paths(args)
    if not episode_paths:
        raise FileNotFoundError(f"No HDF5 trajectory files found under {args.src}")
    if args.save_episode_manifest is not None:
        _write_episode_manifest(args.save_episode_manifest, episode_paths)
        print(f"Wrote episode manifest: {args.save_episode_manifest} entries={len(episode_paths)}", flush=True)

    if args.dry_run_paths:
        print(f"Dry-run path collection: episodes={len(episode_paths)}", flush=True)
        for ep_path in episode_paths[:5]:
            print(f"  sample: {ep_path}", flush=True)
        return

    if args.skip_image_stats:
        lerobot_dataset_module.compute_episode_stats = _compute_episode_stats_without_images

    with h5py.File(episode_paths[0], "r") as f0:
        fmt = _detect_format(f0) if args.format == "auto" else args.format
        fps = _estimate_fps(f0, args.fps)

    _prepare_output(args)
    error_log = args.error_log or (args.out / "conversion_errors.jsonl")
    if error_log.exists():
        error_log.unlink()

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(round(fps)),
        root=args.out,
        features=_build_features(fmt, args.store_hw),
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

    saved = 0
    skipped = 0
    total_frames = 0
    print(
        f"Starting conversion: episodes={len(episode_paths)} format={fmt} fps={fps} "
        f"action_source={args.action_source} out={args.out}",
        flush=True,
    )
    for i, ep_path in enumerate(episode_paths, start=1):
        _write_heartbeat(
            args.heartbeat_path,
            {
                "status": "starting_episode",
                "index": i,
                "total": len(episode_paths),
                "saved": saved,
                "skipped": skipped,
                "frames": total_frames,
                "path": str(ep_path),
            },
        )
        try:
            with h5py.File(ep_path, "r") as f:
                got = _detect_format(f)
                if got != fmt:
                    raise ValueError(f"Mixed formats: {ep_path} is {got!r}, run format is {fmt!r}")
                frame_count = 0
                for frame in _read_episode(fmt, f, ep_path, args):
                    dataset.add_frame(frame)
                    frame_count += 1
                if frame_count <= 0:
                    raise ValueError("Episode produced zero frames")
            dataset.save_episode(parallel_encoding=args.parallel_encoding)
            saved += 1
            total_frames += frame_count
            if saved == 1 or saved % max(1, args.progress_every) == 0:
                print(
                    f"[{i}/{len(episode_paths)}] saved={saved} skipped={skipped} frames={total_frames} last={ep_path}",
                    flush=True,
                )
                _write_heartbeat(
                    args.heartbeat_path,
                    {
                        "status": "progress",
                        "index": i,
                        "total": len(episode_paths),
                        "saved": saved,
                        "skipped": skipped,
                        "frames": total_frames,
                        "path": str(ep_path),
                    },
                )
        except Exception as exc:
            skipped += 1
            _log_error(error_log, ep_path, exc)
            print(f"ERROR episode={ep_path}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.on_error == "abort":
                raise
            if args.max_errors is not None and skipped >= args.max_errors:
                raise RuntimeError(f"Reached max_errors={args.max_errors}; aborting guarded conversion") from exc

    _write_heartbeat(
        args.heartbeat_path,
        {
            "status": "finished",
            "index": len(episode_paths),
            "total": len(episode_paths),
            "saved": saved,
            "skipped": skipped,
            "frames": total_frames,
        },
    )
    print(
        f"Finished conversion: saved={saved} skipped={skipped} frames={total_frames} "
        f"out={args.out} error_log={error_log if skipped else 'none'}",
        flush=True,
    )


if __name__ == "__main__":
    convert(tyro.cli(Args))
