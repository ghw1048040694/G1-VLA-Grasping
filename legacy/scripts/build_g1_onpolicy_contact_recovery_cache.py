#!/usr/bin/env python3
"""Build a contact-recovery cache from states actually visited by a policy.

The old DAgger cache aligned every rollout frame to a canonical expert frame by
episode time.  This builder instead restores the observed MuJoCo state, solves
the strict contact-only wrist targets from that state, and records a short
recovering action chunk.  Privileged object positions are used only to create
expert labels; the resulting cache still contains the RGB/language/state inputs
used by the VLA at inference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import av
import mujoco
import numpy as np
import torch

from g1_compact_vla import SpatialFrozenResNet18
from run_g1_language_pick_place import (
    CONTACT_GRASP_COMPENSATION_M,
    HOME_POSITIONS_M,
    LEFT_PALM_INDEX,
    OBJECT_ASSIST_OFFSET_M,
    OBJECT_NAMES,
    OBJECT_SLOTS_XY_M,
    PALM_NAMES,
    OBJECT_SPAWN_Z_M,
    arm_ik_contract,
    controlled_joints,
    object_hand_target,
    solve_pose,
    smoothstep,
    PALM_ORIENTATION_TARGETS,
)
from collect_g1_language_pick_place_dataset import contact_profile_for_target_slot
from train_g1_compact_vla import encode_color_features
from validate_g1_bimanual_actuation import apply_regularized_dynamics
from validate_g1_bimanual_actuation import HAND_TARGETS
from scipy.spatial.transform import Rotation


CAMERAS = {
    "observation.images.head": "head_camera.mp4",
    "observation.images.left_wrist": "left_wrist_camera.mp4",
    "observation.images.right_wrist": "right_wrist_camera.mp4",
}
TASKS = (
    "put the red triangular prism into the blue box",
    "put the yellow rod into the blue box",
    "put the green cube into the blue box",
    "place the red triangle inside the blue container",
    "place the yellow stick inside the blue container",
    "place the green block inside the blue container",
    "move the red triangular object into the blue bin",
    "move the yellow cylindrical object into the blue bin",
    "move the green cubic object into the blue bin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-horizon", type=int, default=20)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument(
        "--recovery-repeat",
        type=int,
        default=1,
        help="Repeat each on-policy recovery sample during cache aggregation.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spatial-vision", action="store_true")
    return parser.parse_args()


def encode_video(path: Path, encoder: SpatialFrozenResNet18, device: str) -> torch.Tensor:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
    if not frames:
        raise RuntimeError(f"Video contains no frames: {path}")
    encoded = []
    for start in range(0, len(frames), 64):
        batch = torch.stack(frames[start : start + 64]).permute(0, 3, 1, 2)
        encoded.append(encoder(batch.to(device)).cpu().half())
    return torch.cat(encoded)


def strict_profile(expert_root: Path, target: str, initial_xy: np.ndarray) -> dict:
    """Choose the calibrated profile for the target's nearest nominal slot."""
    slot = int(np.argmin([np.linalg.norm(initial_xy - slot_xy) for slot_xy in OBJECT_SLOTS_XY_M]))
    profile = contact_profile_for_target_slot(target, slot)
    # The production profile stores this as a string in collection specs.
    compensation = np.asarray(
        [float(value) for value in profile["contact_grasp_compensation"].split(",")],
        dtype=np.float64,
    )
    return {
        "slot": slot,
        "grasp_yaw_deg": float(profile["grasp_yaw_deg"]),
        "hand_closure_multiplier": float(profile["hand_closure_multiplier"]),
        "contact_compensation": compensation,
    }


def open_hand_value(model: mujoco.MjModel, item: dict) -> float:
    low, high = model.jnt_range[item["joint_id"]]
    if abs(float(low)) < 1e-9:
        return 0.01
    if abs(float(high)) < 1e-9:
        return -0.01
    return 0.5 * float(low + high)


def action_chunk_from_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ik_data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    upper_names: list[str],
    palm_ids: list[int],
    object_body_ids: dict[str, int],
    object_site_ids: dict[str, int],
    target: str,
    profile: dict,
    horizon: int,
) -> tuple[np.ndarray, int, int, dict]:
    target_body = object_body_ids[target]
    target_site = object_site_ids[target]
    target_position = data.site_xpos[target_site].copy()
    active_arm = LEFT_PALM_INDEX if target_position[1] >= 0.05 else 1
    side_lateral = 0.030 if active_arm == LEFT_PALM_INDEX else -0.030
    object_assist = target_position.copy()
    object_assist[1] += side_lateral

    PALM_ORIENTATION_TARGETS[:] = np.eye(3)
    PALM_ORIENTATION_TARGETS[active_arm] = Rotation.from_euler(
        "z", profile["grasp_yaw_deg"], degrees=True
    ).as_matrix()
    target_sets = HOME_POSITIONS_M.copy()
    target_sets[active_arm] = object_assist + np.array((0.0, 0.0, 0.28))
    high_pose, high_report = solve_pose(
        model,
        ik_data,
        palm_ids,
        *arm_ik_contract(model)[1:],
        target_sets,
        data.qpos[arm_ik_contract(model)[1]],
        orientation_weight=0.06,
    )
    pre_targets = HOME_POSITIONS_M.copy()
    pre_targets[active_arm] = object_assist + np.array((0.0, side_lateral * 0.5, 0.12))
    pre_pose, pre_report = solve_pose(
        model,
        ik_data,
        palm_ids,
        *arm_ik_contract(model)[1:],
        pre_targets,
        high_pose,
        orientation_weight=0.06,
    )
    grasp_targets = HOME_POSITIONS_M.copy()
    grasp_targets[active_arm] = object_assist + profile["contact_compensation"]
    if active_arm != LEFT_PALM_INDEX:
        grasp_targets[active_arm, 1] -= 2.0 * profile["contact_compensation"][1]
    grasp_pose, grasp_report = solve_pose(
        model,
        ik_data,
        palm_ids,
        *arm_ik_contract(model)[1:],
        grasp_targets,
        pre_pose,
        orientation_weight=0.06,
    )

    current_palm = data.site_xpos[palm_ids[active_arm]]
    high_distance = float(np.linalg.norm(current_palm - target_sets[active_arm]))
    pre_distance = float(np.linalg.norm(current_palm - pre_targets[active_arm]))
    palm_distance = float(np.linalg.norm(current_palm - grasp_targets[active_arm]))

    # Select the next waypoint from the state, rather than comparing only to
    # the final grasp pose.  The high waypoint is deliberately 28 cm above
    # the object, so final-grasp distance alone would keep a recovered policy
    # in phase 1 forever after it reached the high waypoint.
    nearest_waypoint = int(np.argmin((high_distance, pre_distance, palm_distance)))
    waypoint_tolerance_m = 0.075
    approach_gate_m = 0.12
    if min(high_distance, pre_distance) > approach_gate_m:
        # The robot is still at home or otherwise far from both approach
        # waypoints. Preserve the high approach rather than cutting directly
        # toward the object from an arbitrary recovery state.
        selected_pose, phase = high_pose, 1
    elif pre_distance <= waypoint_tolerance_m:
        selected_pose, phase = grasp_pose, 3
    elif high_distance <= waypoint_tolerance_m or pre_distance < high_distance:
        selected_pose, phase = pre_pose, 2
    else:
        selected_pose, phase = high_pose, 1

    current = np.asarray([data.qpos[controlled[name]["qpos_id"]] for name in upper_names], dtype=np.float32)
    desired = current.copy()
    selected_names, selected_qpos_ids, _, _ = arm_ik_contract(model)
    selected_by_name = dict(zip(selected_names, selected_pose))
    for index, name in enumerate(upper_names):
        if name in selected_by_name:
            desired[index] = float(selected_by_name[name])
        elif "hand_" in name:
            desired[index] = open_hand_value(model, controlled[name])

    # The physical expert closes while approaching the contact pose, not only
    # after the palm is already touching.  Starting around 16 cm gives the
    # fingers time to enter the object geometry as the wrist reaches grasp.
    close_hand = palm_distance <= 0.16 or phase == 3
    for index, name in enumerate(upper_names):
        if not name.startswith(("left_hand_", "right_hand_")):
            continue
        if not name.startswith(("left_hand_" if active_arm == LEFT_PALM_INDEX else "right_hand_")):
            continue
        if close_hand:
            desired[index] = float(
                profile["hand_closure_multiplier"]
                * object_hand_target(target, name, HAND_TARGETS[name])
            )

    chunk = np.empty((horizon, len(upper_names)), dtype=np.float32)
    for step in range(horizon):
        alpha = smoothstep((step + 1) / max(horizon, 1))
        chunk[step] = current + alpha * (desired - current)
    diagnostics = {
        "active_arm": active_arm,
        "high_distance_m": high_distance,
        "palm_distance_m": palm_distance,
        "pre_distance_m": pre_distance,
        "nearest_waypoint": ("high", "pre", "grasp")[nearest_waypoint],
        "selected_phase": phase,
        "close_hand": close_hand,
        "high_solver_error_m": high_report["maximum_position_error_m"],
        "pre_solver_error_m": pre_report["maximum_position_error_m"],
        "grasp_solver_error_m": grasp_report["maximum_position_error_m"],
    }
    return chunk, active_arm, phase, diagnostics


def append_recovery_episode(args: argparse.Namespace, episode_dir: Path, encoder, device, base: dict, episode_offset: int):
    rollout = np.load(episode_dir / "trajectory.npz", allow_pickle=True)
    summary = json.loads((episode_dir / "summary.json").read_text())
    target = summary["target_object"]
    target_index = OBJECT_NAMES.index(target)
    scene = episode_dir / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    ik_data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    controlled_names = list(controlled)
    upper_names = controlled_names[12:]
    palm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_NAMES]
    object_body_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in OBJECT_NAMES}
    object_site_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_assist_site") for name in OBJECT_NAMES}
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    baseline_qpos = data.qpos.copy()
    baseline_quat = {name: data.qpos[model.jnt_qposadr[model.body_jntadr[body]]] + 3 for name, body in object_body_ids.items()}
    initial_objects = {name: data.xpos[body].copy() for name, body in object_body_ids.items()}
    profile = strict_profile(args.expert_root, target, initial_objects[target][:2])

    states = rollout["observation_joint_position_rad"].astype(np.float32)
    objects = rollout["object_position_m"].astype(np.float32)
    frame_indices = list(range(0, len(states), max(1, args.sample_stride)))
    action_parts = []
    phase_parts = []
    arm_parts = []
    diagnostics = []
    for frame in frame_indices:
        data.qpos[:] = baseline_qpos
        data.qvel[:] = 0.0
        for name, value in zip(upper_names, states[frame]):
            data.qpos[controlled[name]["qpos_id"]] = value
        for object_index, name in enumerate(OBJECT_NAMES):
            body = object_body_ids[name]
            joint = model.body_jntadr[body]
            if joint < 0:
                raise RuntimeError(f"Object body {name} has no free joint")
            address = int(model.jnt_qposadr[joint])
            data.qpos[address : address + 3] = objects[frame, object_index]
            data.qpos[address + 3 : address + 7] = baseline_qpos[address + 3 : address + 7]
        mujoco.mj_forward(model, data)
        ik_data.qpos[:] = data.qpos
        ik_data.qvel[:] = data.qvel
        mujoco.mj_forward(model, ik_data)
        chunk, active_arm, phase, item = action_chunk_from_state(
            model, data, ik_data, controlled, upper_names, palm_ids,
            object_body_ids, object_site_ids, target, profile, args.chunk_horizon
        )
        action_parts.append(torch.from_numpy(chunk))
        phase_parts.append(phase)
        arm_parts.append(active_arm)
        diagnostics.append({"frame": frame, **item})
    vision_parts = [encode_video(episode_dir / filename, encoder, device) for filename in CAMERAS.values()]
    vision = torch.stack([part[frame_indices] for part in vision_parts], dim=1)
    color = encode_color_features(episode_dir / CAMERAS["observation.images.head"])[frame_indices]
    task = summary.get("language_instruction", TASKS[target_index])
    try:
        task_index = list(base["tasks"]).index(task)
    except ValueError:
        task_index = target_index
    result = {
        "vision": vision,
        "state": torch.from_numpy(states[frame_indices]),
        "action": torch.stack(action_parts),
        "object_position": torch.from_numpy(objects[frame_indices].reshape(len(frame_indices), -1)),
        "target": torch.full((len(frame_indices),), target_index, dtype=torch.long),
        "phase": torch.tensor(phase_parts, dtype=torch.long),
        "active_arm": torch.tensor(arm_parts, dtype=torch.long),
        "color_features": color.float(),
        "task_index": torch.full((len(frame_indices),), task_index, dtype=torch.long),
        "episode_index": torch.full((len(frame_indices),), int(base["episode_index"].max().item()) + 1 + episode_offset, dtype=torch.long),
        "frame_index": torch.tensor(frame_indices, dtype=torch.long),
    }
    rollout.close()
    return result, diagnostics


def main() -> None:
    args = parse_args()
    if args.chunk_horizon < 1 or args.sample_stride < 1 or args.recovery_repeat < 1:
        raise ValueError("chunk horizon, sample stride, and recovery repeat must be positive")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    base = torch.load(args.base_cache / "train.pt", map_location="cpu", weights_only=False)
    if "color_features" not in base or "active_arm" not in base:
        raise ValueError("base cache must contain color_features and active_arm")
    encoder = (SpatialFrozenResNet18() if args.spatial_vision else SpatialFrozenResNet18()).to(device).eval()
    recovery_parts = []
    all_diagnostics = {}
    episode_dirs = sorted(args.rollout_root.glob("episode_*/trajectory.npz"))
    if not episode_dirs:
        raise FileNotFoundError(args.rollout_root)
    for offset, trajectory in enumerate(episode_dirs):
        part, diagnostics = append_recovery_episode(args, trajectory.parent, encoder, device, base, offset)
        recovery_parts.append(part)
        all_diagnostics[trajectory.parent.name] = diagnostics
        print(f"RECOVERY_EPISODE={trajectory.parent.name} samples={len(part['state'])}", flush=True)
    merged = {}
    for key in ("vision", "state", "action", "object_position", "target", "phase", "active_arm", "color_features", "task_index", "episode_index", "frame_index"):
        repeated = [part[key] for part in recovery_parts for _ in range(args.recovery_repeat)]
        merged[key] = torch.cat((base[key], *repeated), dim=0)
    for key in ("vocab", "tasks", "horizon", "action_offset", "spatial_vision"):
        if key in base:
            merged[key] = base[key]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output_dir / "train.pt")
    shutil.copy2(args.base_cache / "val.pt", args.output_dir / "val.pt")
    summary = {
        "experiment": "G1LANG-205_true_onpolicy_contact_recovery_cache",
        "base_cache": str(args.base_cache),
        "rollout_root": str(args.rollout_root),
        "expert_root": str(args.expert_root),
        "base_samples": len(base["state"]),
        "recovery_samples": sum(len(part["state"]) for part in recovery_parts),
        "weighted_recovery_samples": sum(len(part["state"]) for part in recovery_parts) * args.recovery_repeat,
        "merged_samples": len(merged["state"]),
        "sample_stride": args.sample_stride,
        "recovery_repeat": args.recovery_repeat,
        "chunk_horizon": args.chunk_horizon,
        "labels": "state-restored MuJoCo contact-only pose recovery; no canonical time alignment",
        "privileged_positions_at_inference": False,
        "diagnostics": all_diagnostics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "diagnostics"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
