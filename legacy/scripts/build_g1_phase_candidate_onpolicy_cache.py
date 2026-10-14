#!/usr/bin/env python3
"""Add all learned phase candidates at states visited by a strict VLA rollout.

Each sampled MuJoCo state receives four offline expert labels: home, high
approach, pregrasp, and contact grasp.  This is a data-labeling operation only.
At deployment the VLA still uses RGB, language and 31-D pose, while the learned
World Model selects among its learned candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from collect_g1_language_pick_place_dataset import contact_profile_for_target_slot
from g1_geometric_contact_vla import calibrated_object_positions
from run_g1_language_pick_place import (
    HOME_POSITIONS_M,
    LEFT_PALM_INDEX,
    OBJECT_NAMES,
    OBJECT_SLOTS_XY_M,
    PALM_NAMES,
    PALM_ORIENTATION_TARGETS,
    arm_ik_contract,
    controlled_joints,
    object_hand_target,
    solve_pose,
)
from validate_g1_bimanual_actuation import HAND_TARGETS, apply_regularized_dynamics


def strict_profile(target: str, initial_xy: np.ndarray) -> dict:
    slot = int(
        np.argmin([np.linalg.norm(initial_xy - slot_xy) for slot_xy in OBJECT_SLOTS_XY_M])
    )
    profile = contact_profile_for_target_slot(target, slot)
    return {
        "slot": slot,
        "grasp_yaw_deg": float(profile["grasp_yaw_deg"]),
        "hand_closure_multiplier": float(profile["hand_closure_multiplier"]),
        "contact_compensation": np.asarray(
            [float(value) for value in profile["contact_grasp_compensation"].split(",")],
            dtype=np.float64,
        ),
    }


def open_hand_value(model: mujoco.MjModel, item: dict) -> float:
    low, high = model.jnt_range[item["joint_id"]]
    if abs(float(low)) < 1e-9:
        return 0.01
    if abs(float(high)) < 1e-9:
        return -0.01
    return 0.5 * float(low + high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--recovery-repeat", type=int, default=20)
    parser.add_argument("--max-solver-error-m", type=float, default=0.025)
    parser.add_argument(
        "--grasp-nominal-error-m",
        type=float,
        default=0.060,
        help="Maximum phase-3 nominal solve error before rejecting its FK-consistent label.",
    )
    parser.add_argument(
        "--onpolicy-phases",
        default="1,2,3",
        help="Comma-separated learned candidates to supervise at policy-visited states.",
    )
    return parser.parse_args()


def parse_phases(value: str) -> tuple[int, ...]:
    try:
        phases = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise ValueError("--onpolicy-phases must be comma-separated integers") from error
    if not phases or any(phase < 0 or phase > 3 for phase in phases):
        raise ValueError("--onpolicy-phases must be a nonempty subset of 0,1,2,3")
    return phases


def color_features(image: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    red = (value[0] > 0.47) & (value[0] > value[1] * 1.25) & (value[0] > value[2] * 1.25)
    yellow = (value[0] > 0.47) & (value[1] > 0.35) & (value[2] < 0.47)
    green = (value[1] > 0.31) & (value[1] > value[0] * 1.25) & (value[1] > value[2] * 1.15)
    height, width = image.shape[:2]
    x_grid = torch.arange(width, dtype=torch.float32).view(1, width)
    y_grid = torch.arange(height, dtype=torch.float32).view(height, 1)
    result = []
    for mask in (red, yellow, green):
        count = mask.sum().float().clamp_min(1.0)
        result.extend(
            (
                (mask.float() * x_grid).sum() / count / width,
                (mask.float() * y_grid).sum() / count / height,
                count / float(height * width),
            )
        )
    return torch.stack(result)


def restore_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    upper_names: list[str],
    state: np.ndarray,
    objects: np.ndarray,
    baseline_qpos: np.ndarray,
) -> None:
    data.qpos[:] = baseline_qpos
    data.qvel[:] = 0.0
    for name, value in zip(upper_names, state):
        data.qpos[controlled[name]["qpos_id"]] = float(value)
    for index, name in enumerate(OBJECT_NAMES):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint = int(model.body_jntadr[body])
        address = int(model.jnt_qposadr[joint])
        data.qpos[address : address + 3] = objects[index]
        data.qpos[address + 3 : address + 7] = baseline_qpos[address + 3 : address + 7]
    mujoco.mj_forward(model, data)


def orientation6(data: mujoco.MjData, palm_ids: list[int]) -> np.ndarray:
    return np.concatenate(
        [data.site_xmat[site].reshape(3, 3)[:, :2].reshape(-1) for site in palm_ids]
    ).astype(np.float32)


def pose_candidates(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    upper_names: list[str],
    palm_ids: list[int],
    object_site_ids: dict[str, int],
    target: str,
    profile: dict,
) -> tuple[np.ndarray, int, list[dict]]:
    names, qpos_ids, lower, upper = arm_ik_contract(model)
    target_position = data.site_xpos[object_site_ids[target]].copy()
    active_arm = LEFT_PALM_INDEX if target_position[1] >= 0.05 else 1
    side_lateral = 0.030 if active_arm == LEFT_PALM_INDEX else -0.030
    object_assist = target_position.copy()
    object_assist[1] += side_lateral
    PALM_ORIENTATION_TARGETS[:] = np.eye(3)
    PALM_ORIENTATION_TARGETS[active_arm] = Rotation.from_euler(
        "z", profile["grasp_yaw_deg"], degrees=True
    ).as_matrix()
    current_arm = data.qpos[qpos_ids].copy()
    current_palm_targets = np.stack(
        [data.site_xpos[site].copy() for site in palm_ids]
    )
    home_pose, home_report = solve_pose(
        model, data, palm_ids, qpos_ids, lower, upper, HOME_POSITIONS_M, current_arm,
        orientation_weight=0.06,
    )
    # The policy's passive-palm output is a local residual. Its offline IK
    # labels must therefore preserve the actual passive palm, rather than
    # forcing it back to the canonical home pose from a different state.
    high_targets = current_palm_targets.copy()
    high_targets[active_arm] = object_assist + np.array((0.0, 0.0, 0.28))
    high_pose, high_report = solve_pose(
        model, data, palm_ids, qpos_ids, lower, upper, high_targets, current_arm,
        orientation_weight=0.06,
    )
    pre_targets = current_palm_targets.copy()
    pre_targets[active_arm] = object_assist + np.array((0.0, side_lateral * 0.5, 0.12))
    pre_pose, pre_report = solve_pose(
        model, data, palm_ids, qpos_ids, lower, upper, pre_targets, high_pose,
        orientation_weight=0.06,
    )
    grasp_targets = current_palm_targets.copy()
    grasp_targets[active_arm] = object_assist + profile["contact_compensation"]
    if active_arm != LEFT_PALM_INDEX:
        grasp_targets[active_arm, 1] -= 2.0 * profile["contact_compensation"][1]
    grasp_pose, grasp_report = solve_pose(
        model, data, palm_ids, qpos_ids, lower, upper, grasp_targets, pre_pose,
        orientation_weight=0.06,
    )
    selected = (home_pose, high_pose, pre_pose, grasp_pose)
    reports = (home_report, high_report, pre_report, grasp_report)
    actions = []
    arm_index = {name: index for index, name in enumerate(names)}
    current = np.asarray(
        [data.qpos[controlled[name]["qpos_id"]] for name in upper_names], dtype=np.float32
    )
    for phase, pose in enumerate(selected):
        desired = current.copy()
        for index, name in enumerate(upper_names):
            if name in arm_index:
                desired[index] = pose[arm_index[name]]
            elif "hand_" in name:
                desired[index] = open_hand_value(model, controlled[name])
        if phase == 3:
            prefix = "left_hand_" if active_arm == LEFT_PALM_INDEX else "right_hand_"
            for index, name in enumerate(upper_names):
                if name.startswith(prefix):
                    desired[index] = float(
                        profile["hand_closure_multiplier"]
                        * object_hand_target(target, name, HAND_TARGETS[name])
                    )
        rollout = mujoco.MjData(model)
        rollout.qpos[:] = data.qpos
        rollout.qvel[:] = data.qvel
        for name, value in zip(upper_names, desired):
            rollout.qpos[controlled[name]["qpos_id"]] = value
        mujoco.mj_forward(model, rollout)
        actions.append(
            np.concatenate(
                (
                    desired[:3],
                    np.concatenate([rollout.site_xpos[site].copy() for site in palm_ids]),
                    orientation6(rollout, palm_ids),
                    desired[10:17],
                    desired[24:31],
                )
            ).astype(np.float32)
        )
    return np.stack(actions), active_arm, list(reports)


def phase_candidate_rows(
    episode_dir: Path,
    base: dict,
    calibration: torch.Tensor,
    stride: int,
    max_solver_error_m: float,
    grasp_nominal_error_m: float,
    supervised_phases: tuple[int, ...],
) -> tuple[dict[str, torch.Tensor], dict]:
    summary = json.loads((episode_dir / "summary.json").read_text())
    target_name = summary["target_object"]
    target_index = OBJECT_NAMES.index(target_name)
    with np.load(episode_dir / "trajectory.npz") as trajectory:
        states = trajectory["observation_joint_position_rad"].astype(np.float32)
        objects = trajectory["object_position_m"].astype(np.float32)
    model = mujoco.MjModel.from_xml_path(str(episode_dir / "scene.xml"))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    upper_names = list(controlled)[12:]
    palm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_NAMES]
    object_site_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_assist_site")
        for name in OBJECT_NAMES
    }
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    baseline_qpos = data.qpos.copy()
    profile = strict_profile(target_name, objects[0, target_index, :2])
    renderer = mujoco.Renderer(model, height=120, width=160)
    fields = {key: [] for key in (
        "state", "color_features", "visual_object_position", "palm_position", "action",
        "target", "phase", "active_arm", "task_index",
    )}
    maximum_solver_error = 0.0
    rejected_states = 0
    retained_states = 0
    task = summary["language_instruction"]
    task_index = list(base["tasks"]).index(task)
    for frame in range(0, len(states), stride):
        restore_state(model, data, controlled, upper_names, states[frame], objects[frame], baseline_qpos)
        candidates, active_arm, reports = pose_candidates(
            model, data, controlled, upper_names, palm_ids, object_site_ids, target_name, profile
        )
        errors = [float(report["maximum_position_error_m"]) for report in reports]
        maximum_solver_error = max(maximum_solver_error, *errors)
        thresholds = {
            phase: (grasp_nominal_error_m if phase == 3 else max_solver_error_m)
            for phase in supervised_phases
        }
        if any(errors[phase] > thresholds[phase] for phase in supervised_phases):
            rejected_states += 1
            continue
        # solve_pose mutates its working MjData. Inputs must describe the state
        # the policy actually visited, not the final offline grasp solution.
        restore_state(model, data, controlled, upper_names, states[frame], objects[frame], baseline_qpos)
        renderer.update_scene(data, camera="head_camera")
        feature = color_features(renderer.render().copy())
        visual = calibrated_object_positions(feature[None], calibration)[0].reshape(9)
        palms = np.concatenate([data.site_xpos[site].copy() for site in palm_ids]).astype(np.float32)
        retained_states += 1
        for phase in supervised_phases:
            action = candidates[phase]
            mixed = action.copy()
            selected = objects[frame, target_index]
            for arm in range(2):
                columns = slice(3 + arm * 3, 6 + arm * 3)
                if arm == active_arm:
                    mixed[columns] -= selected
                else:
                    mixed[columns] -= palms[arm * 3 : arm * 3 + 3]
            fields["state"].append(torch.from_numpy(states[frame]))
            fields["color_features"].append(feature)
            fields["visual_object_position"].append(visual)
            fields["palm_position"].append(torch.from_numpy(palms))
            fields["action"].append(torch.from_numpy(mixed))
            fields["target"].append(torch.tensor(target_index, dtype=torch.long))
            fields["phase"].append(torch.tensor(phase, dtype=torch.long))
            fields["active_arm"].append(torch.tensor(active_arm, dtype=torch.long))
            fields["task_index"].append(torch.tensor(task_index, dtype=torch.long))
    close = getattr(renderer, "close", None)
    if close is not None:
        close()
    return {key: torch.stack(value) for key, value in fields.items()}, {
        "sampled_states": len(range(0, len(states), stride)),
        "retained_states": retained_states,
        "rejected_unreachable_states": rejected_states,
        "labels": len(fields["state"]),
        "maximum_solver_error_m": maximum_solver_error,
    }


def merge(base: dict, generated: dict[str, torch.Tensor], repeat: int) -> dict:
    result = {}
    for key, value in base.items():
        if key in generated:
            rows = generated[key].repeat((repeat,) + (1,) * (generated[key].ndim - 1))
            result[key] = torch.cat((value, rows), dim=0)
        else:
            result[key] = value
    return result


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1 or args.recovery_repeat < 1:
        raise ValueError("sample stride and recovery repeat must be positive")
    if args.max_solver_error_m <= 0 or args.grasp_nominal_error_m <= 0:
        raise ValueError("solver-error thresholds must be positive")
    supervised_phases = parse_phases(args.onpolicy_phases)
    calibration = torch.from_numpy(np.load(args.calibration)["coefficients"].astype(np.float32))
    base_train = torch.load(args.base_cache / "train.pt", map_location="cpu", weights_only=False)
    base_val = torch.load(args.base_cache / "val.pt", map_location="cpu", weights_only=False)
    required = {"state", "color_features", "visual_object_position", "palm_position", "action", "target", "phase", "active_arm", "task_index", "tasks", "vocab"}
    missing = sorted(required - set(base_train))
    if missing:
        raise ValueError(f"Base cache is missing: {missing}")
    parts = []
    diagnostics = {}
    for directory in sorted(args.rollout_root.glob("episode_*")):
        if not (directory / "trajectory.npz").is_file():
            continue
        part, detail = phase_candidate_rows(
            directory,
            base_train,
            calibration,
            args.sample_stride,
            args.max_solver_error_m,
            args.grasp_nominal_error_m,
            supervised_phases,
        )
        if not detail["retained_states"]:
            raise RuntimeError(f"No reachable requested phase labels in {directory}")
        parts.append(part)
        diagnostics[directory.name] = detail
        print(f"PHASE_CANDIDATE_ONPOLICY_EPISODE={directory.name} labels={detail['labels']}", flush=True)
    if not parts:
        raise FileNotFoundError(f"No rollout trajectories under {args.rollout_root}")
    generated = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
    merged = merge(base_train, generated, args.recovery_repeat)
    merged["action_contract"] = "active_target_passive_palm_residual"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output_dir / "train.pt")
    torch.save(base_val, args.output_dir / "val.pt")
    summary = {
        "experiment": "G1LANG-phase-candidate-allphase-onpolicy-cache",
        "base_cache": str(args.base_cache),
        "rollout_root": str(args.rollout_root),
        "base_samples": len(base_train["state"]),
        "unique_onpolicy_states": len(generated["state"]) // len(supervised_phases),
        "unique_onpolicy_phase_labels": len(generated["state"]),
        "recovery_repeat": args.recovery_repeat,
        "max_solver_error_m": args.max_solver_error_m,
        "grasp_nominal_error_m": args.grasp_nominal_error_m,
        "onpolicy_phases": list(supervised_phases),
        "merged_samples": len(merged["state"]),
        "label_contract": "offline strict candidate labels at policy-visited states; active target relative and passive palm residual",
        "privileged_positions_at_inference": False,
        "diagnostics": diagnostics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
