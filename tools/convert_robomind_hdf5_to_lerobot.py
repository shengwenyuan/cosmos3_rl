# UR5e post-training — local addition, not part of upstream Cosmos3.

"""Convert RoboMIND UR5(e) HDF5 to LeRobot v3 for Cosmos Action policy SFT (two formats).

RoboMIND ships one HDF5 per episode; the Cosmos action datasets read LeRobot v3. This tool feeds
``RoboMINDUR5Dataset`` / the ``action_policy_robomind_ur5_{single,dual}_nano`` recipes. Two source
formats are supported (auto-detected; one format per run — point separate downloads at separate runs):

* ``ur_1rgb`` — **RoboMIND 1.2 single-arm UR5e** (`h5_ur_1rgb`), schema from
  ``tmps/mind1_all_robot_h5_info_v1.2.md``:
    puppet/joint_position (T,7) = [6 UR joints, gripper]   # executed; master/… is the leader
    puppet/end_effector  (T,6)  = [x,y,z,r,p,y]            # cartesian (ignored for joint_pos)
    observations/rgb_images/camera_top (T,) encoded frames # SINGLE camera, **BGR**
  No `/metadata` and no timestamps → task comes from the directory path, fps from --fps.
  Written as the single-arm **"left"** slot so the dataset's which_arm="left" reads it.

* ``mind2_dual`` — **RoboMIND 2.0 dual-arm UR5e** (verified on a real `trajectory.hdf5`):
    puppet/arm_{left,right}_position_align (T,6); puppet/end_effector_{L,R}_position_align (T,1)
    camera_observations/color_images/camera_{top,front,left,right,wrist_left,wrist_right} (JPEG, RGB)
    /metadata attr language_instruction; camera_observations/timestamp for fps.

Verified on hardware-free bytes for the mind2 path (JPEG decode, joint/gripper assembly). The
LeRobot **writer** calls carry `TODO(verify)` for the installed lerobot version; the ur_1rgb read
(BGR swap, task-from-path) is `TODO(verify)` until a real `h5_ur_1rgb` file is on hand.

Usage::

    # single-arm (RoboMIND 1.2)
    python tools/convert_robomind_hdf5_to_lerobot.py --src <h5_ur_1rgb> --out <…/single/success> --fps 15
    # dual-arm (RoboMIND 2.0)
    python tools/convert_robomind_hdf5_to_lerobot.py --src <mind2_ur5> --out <…/dual/success>
"""

import dataclasses
import io
import re
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import tyro
from PIL import Image

# LeRobot v3 dataset writer. TODO(verify): confirm the writer API for the installed lerobot version.
from lerobot.datasets.lerobot_dataset import LeRobotDataset

Fmt = Literal["auto", "ur_1rgb", "mind2_dual"]
ActionSource = Literal["puppet", "master"]
BgrMode = Literal["auto", "true", "false"]

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
    """Directory of RoboMIND per-episode .hdf5 files (searched recursively)."""
    out: Path
    """Output LeRobot v3 root (point UR5_SINGLE_ROOT / UR5_DUAL_ROOT here, e.g. …/success)."""
    repo_id: str = "local/robomind_ur5"
    """LeRobot repo id (metadata only for a local dataset)."""
    format: Fmt = "auto"
    """Source format; 'auto' detects from the first file and asserts the rest match."""
    action_source: ActionSource = "puppet"
    """Which stream is the imitation target: 'puppet' (executed follower) or 'master' (leader).
    observation.state.* is always puppet."""
    store_hw: tuple[int, int] = (360, 640)
    """(H, W) every camera is resized to before mp4 encoding (matches the DROID 480p canvas)."""
    fps: float = 15.0
    """fps for ur_1rgb (no timestamps in-file — SET THIS to your data's rate). mind2 estimates from timestamps."""
    task_override: str | None = None
    """Force the task string (ur_1rgb falls back to this if the path parse fails)."""
    bgr: BgrMode = "auto"
    """Stored channel order. 'auto' = BGR for ur_1rgb (h5_ur_1rgb is BGR), RGB for mind2_dual."""
    limit: int | None = None
    """Optional cap on the number of episodes (smoke conversions)."""


# ----------------------------- format detection ------------------------------------------------
def _detect_format(f: h5py.File) -> str:
    if "metadata" in f and "puppet/arm_left_position_align/data" in f:
        return "mind2_dual"
    if "puppet/joint_position" in f:
        w = int(np.asarray(f["puppet/joint_position"].shape)[-1])
        if w == 7:
            return "ur_1rgb"
        raise ValueError(
            f"puppet/joint_position width {w} != 7 — not a single-arm UR (h5_ur_1rgb). "
            "This converter only handles UR5e single-arm (7) and RoboMIND 2.0 dual-arm."
        )
    raise ValueError("Unrecognized RoboMIND HDF5 layout (no mind2 *_position_align, no puppet/joint_position).")


# ----------------------------- image decoding --------------------------------------------------
def _decode_image(elem, store_hw: tuple[int, int], bgr: bool) -> np.ndarray:
    """Decode one stored frame (encoded bytes or a raw HWC array) -> uint8 [H, W, 3] RGB at store_hw."""
    arr = np.asarray(elem)
    if arr.ndim == 1:  # encoded (JPEG/PNG) bytes
        img = Image.open(io.BytesIO(arr.astype(np.uint8).tobytes())).convert("RGB")
        frame = np.asarray(img, dtype=np.uint8)
    else:  # already an image array
        frame = arr.astype(np.uint8)
        if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
            frame = np.transpose(frame, (1, 2, 0))  # CHW -> HWC
    if bgr:  # stored channel order is BGR -> swap to RGB
        frame = frame[..., ::-1]
    h, w = store_hw
    if frame.shape[:2] != (h, w):
        frame = np.asarray(Image.fromarray(frame).resize((w, h), Image.BILINEAR), dtype=np.uint8)
    return np.ascontiguousarray(frame)


# ----------------------------- task / fps ------------------------------------------------------
def _task_from_metadata(f: h5py.File, default: str) -> str:
    if "metadata" in f and "language_instruction" in f["metadata"].attrs:
        val = f["metadata"].attrs["language_instruction"]
        return val.decode() if isinstance(val, bytes) else str(val)
    return default


def _task_from_path(ep_path: Path, override: str | None) -> str:
    if override:
        return override
    parts = ep_path.parts
    raw = parts[parts.index("success_episodes") - 1] if "success_episodes" in parts else ep_path.parent.name
    raw = re.sub(r"_\d+$", "", raw)  # strip a trailing "_<id>"
    return raw.replace("_", " ").strip() or "Perform the manipulation task."


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
def _read_ur_1rgb(f: h5py.File, ep_path: Path, args: Args):
    """Yield (frame_dict, task) for a single-arm UR (RoboMIND 1.2), single arm -> 'left' slot."""
    jp_state = np.asarray(f["puppet/joint_position"], dtype=np.float32)  # (T,7)=[6 joints, gripper]
    jp_action = np.asarray(f[f"{args.action_source}/joint_position"], dtype=np.float32)  # (T,7)
    cam = f["observations/rgb_images/camera_top"]
    task = _task_from_path(ep_path, args.task_override)
    n = min(len(jp_state), len(jp_action), len(cam))
    for t in range(n):
        yield (
            {
                "observation.images.camera_top": _decode_image(cam[t], args.store_hw, bgr=_use_bgr(args, "ur_1rgb")),
                _F_JOINT_ACTION["left"]: jp_action[t, :_ARM_DOF],
                _F_GRIPPER_ACTION["left"]: jp_action[t, _ARM_DOF:_ARM_DOF + 1],
                _F_JOINT_STATE["left"]: jp_state[t, :_ARM_DOF],
                _F_GRIPPER_STATE["left"]: jp_state[t, _ARM_DOF:_ARM_DOF + 1],
            },
            task,
        )


def _read_mind2_dual(f: h5py.File, args: Args):
    """Yield (frame_dict, task) for RoboMIND 2.0 dual-arm UR5e (both arms)."""
    def arm(who, side):
        return np.asarray(f[f"{who}/arm_{side}_position_align/data"], dtype=np.float32)  # (T,6)

    def grip(who, side):
        return np.asarray(f[f"{who}/end_effector_{side}_position_align/data"], dtype=np.float32).reshape(-1, 1)

    state = {s: (arm("puppet", s), grip("puppet", s)) for s in ("left", "right")}
    action = {s: (arm(args.action_source, s), grip(args.action_source, s)) for s in ("left", "right")}
    cams = {c: f[f"camera_observations/color_images/{c}"] for c in _MIND2_CAMS}
    task = _task_from_metadata(f, args.task_override or "Perform the manipulation task.")
    bgr = _use_bgr(args, "mind2_dual")
    n = min(min(len(v[0]) for v in state.values()), min(len(cams[c]) for c in _MIND2_CAMS))
    for t in range(n):
        frame = {f"observation.images.{c}": _decode_image(cams[c][t], args.store_hw, bgr=bgr) for c in _MIND2_CAMS}
        for s in ("left", "right"):
            frame[_F_JOINT_ACTION[s]] = action[s][0][t]
            frame[_F_GRIPPER_ACTION[s]] = action[s][1][t]
            frame[_F_JOINT_STATE[s]] = state[s][0][t]
            frame[_F_GRIPPER_STATE[s]] = state[s][1][t]
        yield frame, task


def _use_bgr(args: Args, fmt: str) -> bool:
    # RoboMIND 1.2 h5_ur_1rgb (and franka *rgb / fr3_dual) store BGR; RoboMIND 2.0 JPEGs are RGB.
    if args.bgr == "true":
        return True
    if args.bgr == "false":
        return False
    return fmt == "ur_1rgb"


def convert(args: Args) -> None:
    episode_paths = sorted(args.src.rglob("*.hdf5"))
    if args.limit is not None:
        episode_paths = episode_paths[: args.limit]
    if not episode_paths:
        raise FileNotFoundError(f"No .hdf5 files under {args.src}")

    with h5py.File(episode_paths[0], "r") as f0:
        fmt = _detect_format(f0) if args.format == "auto" else args.format
        fps = _estimate_fps(f0, args.fps)

    dataset = LeRobotDataset.create(  # TODO(verify): signature for the installed lerobot version.
        repo_id=args.repo_id,
        fps=int(round(fps)),
        root=args.out,
        features=_build_features(fmt, args.store_hw),
        use_videos=True,
    )

    for ep_path in episode_paths:
        with h5py.File(ep_path, "r") as f:
            got = _detect_format(f)
            if got != fmt:
                raise ValueError(f"Mixed formats: {ep_path} is {got!r} but the run is {fmt!r}. Convert each format separately.")
            reader = _read_ur_1rgb(f, ep_path, args) if fmt == "ur_1rgb" else _read_mind2_dual(f, args)
            for frame, task in reader:
                dataset.add_frame(frame, task=task)  # TODO(verify): some versions take task= on save_episode.
        dataset.save_episode()

    print(f"Wrote {len(episode_paths)} episodes to {args.out} (format={fmt}, fps={fps}, action_source={args.action_source})")


if __name__ == "__main__":
    _args = tyro.cli(Args)
    convert(_args)
