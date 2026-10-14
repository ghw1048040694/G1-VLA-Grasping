#!/usr/bin/env python3
"""Convert strict-expert data to a Cartesian contact-aware action contract.

The 35D action is ``waist(3), palm positions(6), wrist orientation 6D(12),
and hand joints(14)``.  A separate low-level conversion turns predicted palm
poses into the robot's 31 joint targets at inference time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from run_g1_language_pick_place import OBJECT_NAMES, PALM_NAMES, controlled_joints
from validate_g1_bimanual_actuation import apply_regularized_dynamics


CONTACT_ACTION_DIM = 35


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recovery-cache", type=Path)
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument("--recovery-repeat", type=int, default=20)
    return parser.parse_args()


def orientation6(data: mujoco.MjData, site_ids: list[int]) -> np.ndarray:
    return np.concatenate(
        [data.site_xmat[site].reshape(3, 3)[:, :2].reshape(-1) for site in site_ids]
    ).astype(np.float32)


def future_indices(cache: dict, horizon: int) -> list[list[int]]:
    episodes = cache["episode_index"].long().tolist()
    frames = cache["frame_index"].long().tolist()
    lookup = {
        (int(episode), int(frame)): index
        for index, (episode, frame) in enumerate(zip(episodes, frames))
    }
    result = []
    for episode, frame in zip(episodes, frames):
        result.append(
            [
                lookup.get((int(episode), int(frame) + offset),
                           lookup.get((int(episode), int(frame))))
                for offset in range(horizon)
            ]
        )
    return result


def base_contact_actions(cache: dict) -> torch.Tensor:
    horizon = int(cache["horizon"])
    indices = future_indices(cache, horizon)
    palm = cache["palm_position"].float()
    orientation = cache["palm_orientation"].float()
    action = cache["action"].float()
    rows = []
    for row, future in enumerate(indices):
        future_index = torch.tensor(future, dtype=torch.long)
        waist = action[row, :, :3]
        palms = palm[future_index]
        poses = orientation[future_index]
        hands = torch.cat((action[row, :, 10:17], action[row, :, 24:31]), dim=-1)
        rows.append(torch.cat((waist, palms, poses, hands), dim=-1))
    return torch.stack(rows)


def target_relative_contact_actions(
    contact_action: torch.Tensor,
    object_position: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Express both palm positions relative to the language-selected object."""
    if contact_action.shape[-1] != CONTACT_ACTION_DIM:
        raise ValueError("Unexpected contact action dimension")
    objects = object_position.float().reshape(-1, 3, 3)
    target_position = objects[
        torch.arange(len(objects), dtype=torch.long), target.long()
    ]
    relative = contact_action.clone()
    relative[:, :, 3:9] -= target_position[:, None, :].repeat(1, 1, 2)
    return relative


def palm_residual_contact_actions(
    contact_action: torch.Tensor,
    current_palm_position: torch.Tensor,
) -> torch.Tensor:
    """Express future palm waypoints as residuals from the measured current palms.

    Unlike ``contact_relative_action``, this does not anchor trajectories to a
    static object coordinate.  Every policy replan reconstructs an absolute
    Cartesian target from the current robot forward kinematics, so a small
    closed-loop position error is corrected instead of being carried through
    the rest of an absolute action chunk.
    """
    if contact_action.shape[-1] != CONTACT_ACTION_DIM:
        raise ValueError("Unexpected contact action dimension")
    if current_palm_position.shape != (len(contact_action), 6):
        raise ValueError(
            "current_palm_position must have shape "
            f"[{len(contact_action)},6], got {tuple(current_palm_position.shape)}"
        )
    residual = contact_action.clone()
    residual[:, :, 3:9] -= current_palm_position[:, None, :]
    return residual


def palm_lookahead_residual_contact_actions(
    palm_residual_action: torch.Tensor,
) -> torch.Tensor:
    """Keep the expert chunk endpoint as a one-step visual-servo target.

    Expert chunks interpolate from the observed state to the selected waypoint.
    Their first sample is intentionally tiny, which is correct while executing
    a whole chunk but stalls a replan-every-frame controller.  The final
    waypoint carries the learned lookahead displacement needed for a true
    closed-loop residual action.
    """
    if palm_residual_action.ndim != 3 or palm_residual_action.shape[-1] != CONTACT_ACTION_DIM:
        raise ValueError("Unexpected palm residual contact action shape")
    return palm_residual_action[:, -1:, :].clone()


def set_rollout_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    state: np.ndarray,
    objects: np.ndarray,
    baseline_qpos: np.ndarray,
):
    upper_names = list(controlled)[12:]
    data.qpos[:] = baseline_qpos
    data.qvel[:] = 0.0
    for name, value in zip(upper_names, state):
        data.qpos[controlled[name]["qpos_id"]] = float(value)
    for object_index, name in enumerate(OBJECT_NAMES):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint = int(model.body_jntadr[body])
        address = int(model.jnt_qposadr[joint])
        data.qpos[address : address + 3] = objects[object_index]
        data.qpos[address + 3 : address + 7] = baseline_qpos[address + 3 : address + 7]
    mujoco.mj_forward(model, data)


def recovery_contact_actions(
    cache: dict,
    rollout_root: Path,
    recovery_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    horizon = int(cache["horizon"])
    episode_ids = cache["episode_index"].long().tolist()
    frames = cache["frame_index"].long().tolist()
    rollout_dirs = sorted(rollout_root.glob("episode_*/trajectory.npz"))
    recovery_episodes = sorted(set(episode_ids))
    if len(recovery_episodes) != len(rollout_dirs):
        raise ValueError(
            f"Found {len(recovery_episodes)} recovery episodes but "
            f"{len(rollout_dirs)} rollout scenes"
        )
    episode_to_offset = {
        episode: offset for offset, episode in enumerate(recovery_episodes)
    }
    models = {}
    data_by_episode = {}
    controlled_by_episode = {}
    baseline_by_episode = {}
    site_by_episode = {}
    for episode_dir in rollout_dirs:
        offset = int(episode_dir.parent.name.split("_")[-1])
        scene = episode_dir.parent / "scene.xml"
        model = mujoco.MjModel.from_xml_path(str(scene))
        apply_regularized_dynamics(model)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        models[offset] = model
        data_by_episode[offset] = data
        controlled_by_episode[offset] = controlled_joints(model)
        baseline_by_episode[offset] = data.qpos.copy()
        site_by_episode[offset] = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in PALM_NAMES
        ]
    trajectories = {}
    for offset, path in enumerate(rollout_dirs):
        with np.load(path, allow_pickle=True) as arrays:
            trajectories[offset] = {
                "state": arrays["observation_joint_position_rad"].astype(np.float32),
                "objects": arrays["object_position_m"].astype(np.float32),
            }
    rows = []
    current_palms = []
    current_orientations = []
    for row, (episode_id, frame) in enumerate(zip(episode_ids, frames)):
        offset = episode_to_offset[int(episode_id)]
        if offset not in models:
            raise ValueError(f"Recovery episode id {episode_id} is not mapped")
        model = models[offset]
        data = data_by_episode[offset]
        controlled = controlled_by_episode[offset]
        set_rollout_state(
            model,
            data,
            controlled,
            trajectories[offset]["state"][frame],
            trajectories[offset]["objects"][frame],
            baseline_by_episode[offset],
        )
        current_palms.append(
            torch.from_numpy(
                np.concatenate(
                    [data.site_xpos[site].copy() for site in site_by_episode[offset]]
                ).astype(np.float32)
            )
        )
        current_orientations.append(
            torch.from_numpy(orientation6(data, site_by_episode[offset]))
        )
        action_chunk = cache["action"][row].numpy()
        target_rows = []
        for action in action_chunk:
            for name, value in zip(list(controlled)[12:], action):
                data.qpos[controlled[name]["qpos_id"]] = float(value)
            mujoco.mj_forward(model, data)
            palms = np.concatenate(
                [data.site_xpos[site].copy() for site in site_by_episode[offset]]
            ).astype(np.float32)
            pose = orientation6(data, site_by_episode[offset])
            target_rows.append(
                np.concatenate(
                    (action[:3], palms, pose, action[10:17], action[24:31])
                ).astype(np.float32)
            )
        rows.append(torch.from_numpy(np.stack(target_rows)))
    result = torch.stack(rows)
    palms = torch.stack(current_palms)
    orientations = torch.stack(current_orientations)
    if recovery_repeat > 1:
        result = result.repeat((recovery_repeat, 1, 1))
        palms = palms.repeat((recovery_repeat, 1))
        orientations = orientations.repeat((recovery_repeat, 1))
    return result, palms, orientations


def main() -> None:
    args = parse_args()
    if args.recovery_repeat < 1:
        raise ValueError("recovery repeat must be positive")
    base = torch.load(args.pose_cache / "train.pt", map_location="cpu", weights_only=False)
    if "palm_position" not in base or "palm_orientation" not in base:
        raise ValueError("pose cache must contain palm_position and palm_orientation")
    base_contact = base_contact_actions(base)
    merged = dict(base)
    merged["contact_action"] = base_contact
    recovery_count = 0
    if args.recovery_cache is not None:
        if args.rollout_root is None:
            raise ValueError("--rollout-root is required with --recovery-cache")
        recovery = torch.load(args.recovery_cache / "train.pt", map_location="cpu", weights_only=False)
        # The recovery cache is built from one copy of each rollout frame. Its
        # length depends on the rollout duration and sample stride, so derive
        # the suffix length from the unmodified pose cache rather than assuming
        # the old 12-second, 135-sample collection contract.
        base_count = len(base["state"])
        recovery_unique_count = len(recovery["state"]) - base_count
        if recovery_unique_count <= 0:
            raise ValueError(
                "recovery cache must append at least one unique rollout sample"
            )
        for key in (
            "vision", "state", "action", "object_position", "target", "phase",
            "active_arm", "color_features", "task_index", "episode_index", "frame_index",
        ):
            if len(recovery[key]) != len(recovery["state"]):
                raise ValueError(f"Recovery cache field {key} has inconsistent length")
        recovery_unique = {
            key: (
                value[-recovery_unique_count:]
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in recovery.items()
        }
        recovery_contact, recovery_palms, recovery_orientations = recovery_contact_actions(
            recovery_unique, args.rollout_root, args.recovery_repeat
        )
        recovery_count = len(recovery_contact)
        for key in ("vision", "state", "action", "object_position", "target", "phase", "active_arm", "color_features", "task_index", "episode_index", "frame_index"):
            merged[key] = torch.cat((merged[key], recovery[key][-recovery_unique_count:].repeat((args.recovery_repeat,) + (1,) * (recovery[key].ndim - 1))), dim=0)
        merged["contact_action"] = torch.cat((base_contact, recovery_contact), dim=0)
        merged["palm_position"] = torch.cat(
            (base["palm_position"], recovery_palms), dim=0
        )
        merged["palm_orientation"] = torch.cat(
            (base["palm_orientation"], recovery_orientations), dim=0
        )
    merged["contact_relative_action"] = target_relative_contact_actions(
        merged["contact_action"], merged["object_position"], merged["target"]
    )
    merged["contact_palm_residual_action"] = palm_residual_contact_actions(
        merged["contact_action"], merged["palm_position"]
    )
    merged["contact_palm_lookahead_residual_action"] = (
        palm_lookahead_residual_contact_actions(
            merged["contact_palm_residual_action"]
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output_dir / "train.pt")
    val = torch.load(args.pose_cache / "val.pt", map_location="cpu", weights_only=False)
    val_contact = base_contact_actions(val)
    val["contact_action"] = val_contact
    val["contact_relative_action"] = target_relative_contact_actions(
        val_contact, val["object_position"], val["target"]
    )
    if "palm_position" not in val:
        raise ValueError("pose cache validation split must contain palm_position")
    val["contact_palm_residual_action"] = palm_residual_contact_actions(
        val_contact, val["palm_position"]
    )
    val["contact_palm_lookahead_residual_action"] = (
        palm_lookahead_residual_contact_actions(
            val["contact_palm_residual_action"]
        )
    )
    torch.save(val, args.output_dir / "val.pt")
    summary = {
        "experiment": "G1LANG-207_contact_aware_action_cache",
        "contact_action_dim": CONTACT_ACTION_DIM,
        "target_relative_contact_action_dim": CONTACT_ACTION_DIM,
        "palm_residual_contact_action_dim": CONTACT_ACTION_DIM,
        "palm_lookahead_residual_contact_action_dim": CONTACT_ACTION_DIM,
        "base_samples": len(base_contact),
        "recovery_weighted_samples": recovery_count,
        "recovery_unique_samples": (
            recovery_unique_count if args.recovery_cache is not None else 0
        ),
        "merged_samples": len(merged["contact_action"]),
        "contract": "waist3 + palm_position6 + wrist_orientation6D12 + hand_joints14",
        "relative_contract": "palm positions are relative to language-selected object",
        "palm_residual_contract": (
            "future palm Cartesian waypoints are residuals from the current "
            "measured palm positions"
        ),
        "palm_lookahead_residual_contract": (
            "the final expert chunk waypoint is a one-step visual-servo "
            "residual target"
        ),
        "privileged_positions_at_inference": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
