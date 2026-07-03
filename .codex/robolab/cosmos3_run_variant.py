# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 RoboLab runner with seedable lighting-profile registration.

This mirrors policies/cosmos3/run.py but keeps registration local so batch jobs
can vary lighting and environment seed without editing RoboLab's source tree.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import traceback
from typing import Any

import cv2  # Must import this before isaaclab.
from isaaclab.app import AppLauncher

POLICY = "cosmos3"
logger = logging.getLogger(__name__)


def _add_variant_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--remote-host",
        default="localhost",
        help="Remote host for policy server (default: localhost).",
    )
    parser.add_argument(
        "--remote-port",
        default=8000,
        type=int,
        help="Remote port for policy server (default: 8000).",
    )
    parser.add_argument(
        "--lighting-profile",
        default="base",
        choices=[
            "base",
            "random",
            "red",
            "blue",
            "green",
            "dim",
            "front",
            "behind",
            "top",
            "left",
            "right",
        ],
        help="Lighting profile to register for this process.",
    )
    parser.add_argument(
        "--lighting-seed",
        type=int,
        default=None,
        help="Seed used when --lighting-profile=random.",
    )
    parser.add_argument(
        "--env-seed",
        type=int,
        default=1,
        help="Seed forwarded into the generated RoboLab environment config.",
    )


parser = argparse.ArgumentParser(description="Evaluate the Cosmos3 policy backend with run variants.")
_add_variant_args(parser)

from robolab.eval.runner import add_common_eval_args, clear_task_filter_for_explicit_paths, run_evaluation

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from policies.cosmos3.client import Cosmos3Client
from robolab.constants import DEFAULT_TASK_SUBFOLDERS, TASK_DIR
from robolab.core.environments.factory import auto_discover_and_create_cfgs
from robolab.core.observations.observation_utils import generate_image_obs_from_cameras, generate_obs_cfg
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
from robolab.robots.droid import (
    DroidCfg,
    DroidJointPositionActionCfg,
    ProprioceptionObservationCfg,
    WristCameraCfg,
    contact_gripper,
)
from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
from robolab.variations.camera import EgocentricMirroredCameraCfg
from robolab.variations.lighting import (
    BehindDirectionalLightCfg,
    BlueSphereLightCfg,
    ExtremelyDimSphereLightCfg,
    FrontDirectionalLightCfg,
    GreenSphereLightCfg,
    LeftDirectionalLightCfg,
    RedSphereLightCfg,
    RightDirectionalLightCfg,
    SphereLightCfg,
    TopDownDirectionalLightCfg,
)


def _lighting_profiles() -> dict[str, Any]:
    return {
        "base": SphereLightCfg,
        "red": RedSphereLightCfg,
        "blue": BlueSphereLightCfg,
        "green": GreenSphereLightCfg,
        "dim": ExtremelyDimSphereLightCfg,
        "front": FrontDirectionalLightCfg,
        "behind": BehindDirectionalLightCfg,
        "top": TopDownDirectionalLightCfg,
        "left": LeftDirectionalLightCfg,
        "right": RightDirectionalLightCfg,
    }


def _select_lighting(profile: str, seed: int | None) -> tuple[str, Any]:
    profiles = _lighting_profiles()
    if profile == "random":
        rng = random.Random(seed)
        resolved = rng.choice([name for name in profiles if name != "base"])
    else:
        resolved = profile
    return resolved, profiles[resolved]


def register_droid_envs_for_variant(args: argparse.Namespace) -> None:
    resolved_lighting, lighting_cfg = _select_lighting(args.lighting_profile, args.lighting_seed)
    print(
        "[RoboLab] variant_registration "
        f"lighting_profile={args.lighting_profile} resolved_lighting={resolved_lighting} "
        f"lighting_seed={args.lighting_seed} env_seed={args.env_seed}"
    )

    cameras = WRIST_LEFT_RIGHT_HEAD
    image_obs_cfg = generate_image_obs_from_cameras(cameras)
    viewport_camera_cfg = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
    observation_cfg = generate_obs_cfg(
        {
            "image_obs": image_obs_cfg(),
            "proprio_obs": ProprioceptionObservationCfg(),
            "viewport_cam": viewport_camera_cfg(),
        }
    )

    scene_cameras = [camera for camera in cameras if camera is not WristCameraCfg]
    env_postfix = "" if resolved_lighting == "base" else f"_{resolved_lighting.title()}Light"

    auto_discover_and_create_cfgs(
        task_dir=TASK_DIR,
        task_subdirs=args.task_dirs or DEFAULT_TASK_SUBFOLDERS,
        tasks=args.task,
        pattern="*.py",
        env_prefix="",
        env_postfix=env_postfix,
        observations_cfg=observation_cfg(),
        actions_cfg=DroidJointPositionActionCfg(),
        robot_cfg=DroidCfg,
        camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
        lighting_cfg=lighting_cfg,
        background_cfg=HomeOfficeBackgroundCfg,
        contact_gripper=contact_gripper,
        dt=1 / (60 * 2),
        render_interval=8,
        decimation=8,
        seed=args.env_seed,
    )

    if clear_task_filter_for_explicit_paths(args):
        logger.debug("Registered explicit task path(s); cleared eval task filter.")


def make_client(args: argparse.Namespace) -> Cosmos3Client:
    return Cosmos3Client(remote_host=args.remote_host, remote_port=args.remote_port)


def main() -> None:
    register_droid_envs_for_variant(args_cli)
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\033[96m[RoboLab] Terminated with error: {exc}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
