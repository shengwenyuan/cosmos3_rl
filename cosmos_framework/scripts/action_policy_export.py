# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Lightweight action-policy sidecar helpers for checkpoint export."""

from pathlib import Path
from typing import Any

from cosmos_framework.data.generator.action.policy_schema import (
    ActionPolicyManifest,
    find_action_policy_manifest,
    load_action_policy_manifest,
)


def resolve_action_policy_manifest(checkpoint_path: str, explicit: Path | None) -> ActionPolicyManifest | None:
    discovered_path = find_action_policy_manifest(checkpoint_path)
    discovered = load_action_policy_manifest(discovered_path) if discovered_path is not None else None
    requested = load_action_policy_manifest(explicit) if explicit is not None else None
    if discovered is not None and requested is not None and discovered != requested:
        raise ValueError(
            f"Explicit policy config {explicit} conflicts with the checkpoint owner's canonical {discovered_path}"
        )
    return discovered or requested


def validate_action_policy_destination(manifest: ActionPolicyManifest | None, output_dir: Path) -> None:
    """Reject stale export semantics before any model files are written."""
    destination = output_dir / "action_policy.yaml"
    if manifest is None:
        if destination.exists():
            raise ValueError(
                f"Export source has no action-policy manifest, but destination already contains {destination}. "
                "Use a clean output directory."
            )
        return
    if destination.exists() and load_action_policy_manifest(destination) != manifest:
        raise ValueError(f"Refusing to replace a different exported action-policy manifest: {destination}")


def validate_edge_policy_metadata(
    manifest: ActionPolicyManifest | None,
    edge_policy_metadata: dict[str, Any] | None,
) -> None:
    """Keep the official Edge checkpoint policy block and richer sidecar consistent."""
    if manifest is None or edge_policy_metadata is None:
        return
    expected = {
        "action_chunk_size": manifest.chunk_size,
        "conditioning_fps": float(manifest.policy_fps),
        "domain_name": manifest.domain_name,
    }
    mismatches = {
        key: (edge_policy_metadata.get(key), value)
        for key, value in expected.items()
        if edge_policy_metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Action-policy manifest conflicts with official Edge checkpoint metadata "
            f"(checkpoint, manifest): {mismatches}"
        )
