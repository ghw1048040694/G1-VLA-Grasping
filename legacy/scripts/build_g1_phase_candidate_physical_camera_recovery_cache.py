#!/usr/bin/env python3
"""Build camera-frame, physically verified phase-3 recovery labels.

This is an offline labeling tool.  MuJoCo object poses are used only to find
and verify a real contact pose.  The stored active-palm action is expressed
relative to the RGB-calibrated object position, which is the coordinate frame
available to the deployed VLA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from audit_g1_contact_pad_poses import hand_object_contacts
from build_g1_phase_candidate_onpolicy_cache import (
    color_features,
    open_hand_value,
    orientation6,
    restore_state,
)
from g1_geometric_contact_vla import calibrated_object_positions
from run_g1_language_pick_place import (
    HOME_POSITIONS_M,
    LEFT_PALM_INDEX,
    OBJECT_NAMES,
    PALM_NAMES,
    PALM_ORIENTATION_TARGETS,
    arm_ik_contract,
    controlled_joints,
    object_hand_target,
    solve_pose,
)
from validate_g1_bimanual_actuation import HAND_TARGETS, apply_regularized_dynamics


ACTION_FIELDS = (
    "state",
    "color_features",
    "visual_object_position",
    "palm_position",
    "action",
    "target",
    "phase",
    "active_arm",
    "task_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--contact-audit", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--recovery-repeat", type=int, default=100)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--max-ik-error-m", type=float, default=0.015)
    parser.add_argument(
        "--max-target-drift-m",
        type=float,
        default=0.008,
        help="Discard rollout states whose target XY has moved beyond the latched camera observation.",
    )
    return parser.parse_args()


def audited_contact_targets(path: Path) -> dict[str, dict]:
    audit = json.loads(path.read_text())
    result = {}
    for name in OBJECT_NAMES:
        entry = audit["results"][name]
        pose = entry.get("best_contact_pose")
        if pose is None or not pose.get("contact_geoms"):
            raise ValueError(f"No audited physical contact pose for {name}")
        center = np.asarray(entry["object_center_m"], dtype=np.float64)
        site = np.asarray(pose["palm_site_target_m"], dtype=np.float64)
        result[name] = {
            "audited_arm": LEFT_PALM_INDEX if entry["active_arm"] == "left" else 1,
            "site_offset": site - center,
            "yaw_deg": float(pose["yaw_deg"]),
        }
    return result


def mirrored_contact_target(values: dict, active_arm: int) -> tuple[np.ndarray, float]:
    if active_arm == values["audited_arm"]:
        return values["site_offset"].copy(), float(values["yaw_deg"])
    return values["site_offset"] * np.asarray((1.0, -1.0, 1.0)), -float(values["yaw_deg"])


def contact_action(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    upper_names: list[str],
    palm_ids: list[int],
    target_name: str,
    target_center: np.ndarray,
    active_arm: int,
    target_values: dict,
    object_geoms: set[int],
    max_ik_error_m: float,
) -> tuple[np.ndarray, list[str], float] | None:
    """Solve and verify one static physical-contact action from a rollout state."""
    names, qpos_ids, lower, upper = arm_ik_contract(model)
    original_qpos = data.qpos.copy()
    current_targets = np.stack([data.site_xpos[site].copy() for site in palm_ids])
    current_orientations = np.stack(
        [data.site_xmat[site].reshape(3, 3).copy() for site in palm_ids]
    )
    offset, yaw = mirrored_contact_target(target_values, active_arm)
    desired_targets = current_targets.copy()
    desired_targets[active_arm] = target_center + offset
    desired_orientations = current_orientations.copy()
    desired_orientations[active_arm] = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
    PALM_ORIENTATION_TARGETS[:] = desired_orientations
    seed = data.qpos[qpos_ids].copy()
    pose, report = solve_pose(
        model,
        data,
        palm_ids,
        qpos_ids,
        lower,
        upper,
        desired_targets,
        seed,
        orientation_weight=0.06,
    )
    if float(report["maximum_position_error_m"]) > max_ik_error_m:
        return None

    desired = np.asarray(
        [data.qpos[controlled[name]["qpos_id"]] for name in upper_names], dtype=np.float32
    )
    arm_index = {name: index for index, name in enumerate(names)}
    for index, name in enumerate(upper_names):
        if name in arm_index:
            desired[index] = pose[arm_index[name]]
        elif "hand_" in name:
            desired[index] = open_hand_value(model, controlled[name])
    prefix = "left_hand_" if active_arm == LEFT_PALM_INDEX else "right_hand_"
    for index, name in enumerate(upper_names):
        if name.startswith(prefix):
            desired[index] = object_hand_target(target_name, name, HAND_TARGETS[name])

    rollout = mujoco.MjData(model)
    rollout.qpos[:] = original_qpos
    for name, value in zip(upper_names, desired):
        rollout.qpos[controlled[name]["qpos_id"]] = value
    mujoco.mj_forward(model, rollout)
    side = "left" if active_arm == LEFT_PALM_INDEX else "right"
    contacts = hand_object_contacts(model, rollout, object_geoms, side)
    if not contacts:
        return None
    action = np.concatenate(
        (
            desired[:3],
            np.concatenate([rollout.site_xpos[site].copy() for site in palm_ids]),
            orientation6(rollout, palm_ids),
            desired[10:17],
            desired[24:31],
        )
    ).astype(np.float32)
    return action, contacts, float(report["maximum_position_error_m"])


def initialize_fields() -> dict[str, list[torch.Tensor]]:
    return {key: [] for key in ACTION_FIELDS}


def add_row(
    fields: dict[str, list[torch.Tensor]],
    state: np.ndarray,
    feature: torch.Tensor,
    visual: torch.Tensor,
    palms: np.ndarray,
    action: np.ndarray,
    target: int,
    active_arm: int,
    target_center: np.ndarray,
    task_index: int,
) -> None:
    mixed = action.copy()
    for arm in range(2):
        columns = slice(3 + arm * 3, 6 + arm * 3)
        if arm == active_arm:
            # Supervise the MuJoCo-audited contact offset itself.  The deployed
            # policy adds its RGB-localized target estimate back to this value;
            # subtracting that estimate here would bake camera noise into the
            # label and make identical physical contacts contradictory.
            mixed[columns] -= target_center
        else:
            mixed[columns] -= palms[arm * 3 : arm * 3 + 3]
    fields["state"].append(torch.from_numpy(state.copy()))
    fields["color_features"].append(feature)
    fields["visual_object_position"].append(visual)
    fields["palm_position"].append(torch.from_numpy(palms))
    fields["action"].append(torch.from_numpy(mixed))
    fields["target"].append(torch.tensor(target, dtype=torch.long))
    fields["phase"].append(torch.tensor(3, dtype=torch.long))
    fields["active_arm"].append(torch.tensor(active_arm, dtype=torch.long))
    fields["task_index"].append(torch.tensor(task_index, dtype=torch.long))


def stack_fields(fields: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not fields["state"]:
        raise RuntimeError("No physical contact labels were retained")
    return {key: torch.stack(values) for key, values in fields.items()}


def merge_training(base: dict, generated: dict[str, torch.Tensor], repeat: int) -> dict:
    result = dict(base)
    for key, values in generated.items():
        repeated = values.repeat((repeat,) + (1,) * (values.ndim - 1))
        result[key] = torch.cat((base[key], repeated), dim=0)
    result["action_contract"] = "active_target_passive_palm_residual"
    result["active_target_coordinate_frame"] = "camera_localized_object_position"
    result["phase3_contact_labels"] = "MuJoCo-verified physical hand-object contact"
    return result


def validation_cache(base: dict, generated: dict[str, torch.Tensor]) -> dict:
    result = dict(base)
    result.update(generated)
    result["action_contract"] = "active_target_passive_palm_residual"
    result["active_target_coordinate_frame"] = "camera_localized_object_position"
    result["phase3_contact_labels"] = "MuJoCo-verified physical hand-object contact"
    return result


def main() -> None:
    args = parse_args()
    if args.sample_stride < 1 or args.recovery_repeat < 1 or args.validation_modulus < 2:
        raise ValueError("sample stride, recovery repeat, and validation modulus must be positive")
    if args.max_ik_error_m <= 0.0 or args.max_target_drift_m < 0.0:
        raise ValueError("contact thresholds must be non-negative and max IK error positive")
    calibration = torch.from_numpy(np.load(args.calibration)["coefficients"].astype(np.float32))
    targets = audited_contact_targets(args.contact_audit)
    base_train = torch.load(args.base_cache / "train.pt", map_location="cpu", weights_only=False)
    base_val = torch.load(args.base_cache / "val.pt", map_location="cpu", weights_only=False)
    missing = set(ACTION_FIELDS) - set(base_train)
    if missing:
        raise ValueError(f"Base cache is missing: {sorted(missing)}")
    # A checkpoint consumes this calibration at deployment.  Recompute every
    # cached RGB geometry input with it so phase-0 through phase-3 examples do
    # not mix incompatible camera coordinate frames.
    for cache in (base_train, base_val):
        cache["visual_object_position"] = calibrated_object_positions(
            cache["color_features"].float(), calibration
        ).reshape(-1, 9).to(dtype=torch.float32)
        cache["visual_object_position_frame"] = "current_rgb_calibration"
    train_fields = initialize_fields()
    val_fields = initialize_fields()
    diagnostics = {}
    retained_by_target = {name: 0 for name in OBJECT_NAMES}

    for episode_dir in sorted(args.rollout_root.glob("episode_*")):
        trajectory_path = episode_dir / "trajectory.npz"
        summary_path = episode_dir / "summary.json"
        if not trajectory_path.is_file() or not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        target_name = summary["target_object"]
        target = OBJECT_NAMES.index(target_name)
        task_index = list(base_train["tasks"]).index(summary["language_instruction"])
        with np.load(trajectory_path) as trajectory:
            states = trajectory["observation_joint_position_rad"].astype(np.float32)
            objects = trajectory["object_position_m"].astype(np.float32)
        model = mujoco.MjModel.from_xml_path(str(episode_dir / "scene.xml"))
        apply_regularized_dynamics(model)
        data = mujoco.MjData(model)
        controlled = controlled_joints(model)
        upper_names = list(controlled)[12:]
        palm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_NAMES]
        object_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        object_geoms = {
            geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) == object_body
        }
        # Match evaluate_g1_language_pick_place.run_episode exactly: the RGB
        # observation is taken after the home IK pose is installed and before
        # the first physics step.  Trajectory row k is then the observation
        # state for control frame k + 1.
        names, qpos_ids, lower, upper = arm_ik_contract(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        home_pose, home_report = solve_pose(
            model,
            data,
            palm_ids,
            qpos_ids,
            lower,
            upper,
            HOME_POSITIONS_M,
            np.zeros(len(names)),
            orientation_weight=0.06,
        )
        if not home_report["solver_success"]:
            raise RuntimeError(f"Could not reproduce evaluation home pose for {episode_dir}")
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_ids] = home_pose
        for name, item in controlled.items():
            if "hand_" in name:
                data.qpos[item["qpos_id"]] = open_hand_value(model, item)
        mujoco.mj_forward(model, data)
        baseline_qpos = data.qpos.copy()
        renderer = mujoco.Renderer(model, height=120, width=160)
        # The deployed policy latches this first unobstructed observation.
        renderer.update_scene(data, camera="head_camera")
        latched_feature = color_features(renderer.render().copy())
        latched_visual = calibrated_object_positions(latched_feature[None], calibration)[0].reshape(9)
        initial_target = data.xpos[object_body].copy()
        retained = rejected = rejected_drift = contact_count = 0
        residuals = []
        for frame in range(0, len(states), args.sample_stride):
            restore_state(model, data, controlled, upper_names, states[frame], objects[frame], baseline_qpos)
            center = objects[frame, target]
            # Free objects settle vertically onto the table in the first
            # physics step.  The latched RGB estimate is still valid for that
            # expected Z settling; only unobserved XY motion is disqualifying.
            if float(np.linalg.norm(center[:2] - initial_target[:2])) > args.max_target_drift_m:
                rejected_drift += 1
                continue
            active_arm = LEFT_PALM_INDEX if center[1] >= 0.05 else 1
            result = contact_action(
                model,
                data,
                controlled,
                upper_names,
                palm_ids,
                target_name,
                center,
                active_arm,
                targets[target_name],
                object_geoms,
                args.max_ik_error_m,
            )
            if result is None:
                rejected += 1
                continue
            action, contacts, _ = result
            restore_state(model, data, controlled, upper_names, states[frame], objects[frame], baseline_qpos)
            palms = np.concatenate([data.site_xpos[site].copy() for site in palm_ids]).astype(np.float32)
            destination = (
                val_fields
                if retained_by_target[target_name] % args.validation_modulus == 0
                else train_fields
            )
            add_row(
                destination,
                states[frame],
                latched_feature,
                latched_visual,
                palms,
                action,
                target,
                active_arm,
                center,
                task_index,
            )
            retained_by_target[target_name] += 1
            retained += 1
            contact_count += int(bool(contacts))
            columns = slice(3 + active_arm * 3, 6 + active_arm * 3)
            residuals.append((action[columns] - center).tolist())
        close = getattr(renderer, "close", None)
        if close is not None:
            close()
        diagnostics[episode_dir.name] = {
            "target_object": target_name,
            "sampled_states": len(range(0, len(states), args.sample_stride)),
            "retained_physical_contact_labels": retained,
            "rejected_states": rejected,
            "rejected_target_drift_states": rejected_drift,
            "verified_contact_labels": contact_count,
            "latched_camera_minus_initial_true_target_m": (
                latched_visual.reshape(3, 3)[target].numpy() - initial_target
            ).tolist(),
            "mean_camera_frame_active_site_residual_m": np.asarray(residuals).mean(0).tolist() if residuals else None,
        }
        print("PHYSICAL_CAMERA_RECOVERY_EPISODE=" + json.dumps(diagnostics[episode_dir.name]), flush=True)

    generated_train = stack_fields(train_fields)
    generated_val = stack_fields(val_fields)
    output_train = merge_training(base_train, generated_train, args.recovery_repeat)
    output_val = validation_cache(base_val, generated_val)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(output_train, args.output_dir / "train.pt")
    torch.save(output_val, args.output_dir / "val.pt")
    summary = {
        "experiment": "G1LANG-physical-camera-frame-contact-recovery-cache",
        "base_cache": str(args.base_cache),
        "rollout_root": str(args.rollout_root),
        "contact_audit": str(args.contact_audit),
        "calibration": str(args.calibration),
        "action_contract": "audited active-palm offset applied to camera-localized target; passive palm residual",
        "physical_contact_verification": "MuJoCo hand-object geom contact required for every generated label",
        "base_visual_positions_recalibrated": True,
        "privileged_positions_at_inference": False,
        "base_train_samples": len(base_train["state"]),
        "generated_train_states": len(generated_train["state"]),
        "generated_validation_states": len(generated_val["state"]),
        "recovery_repeat": args.recovery_repeat,
        "merged_train_samples": len(output_train["state"]),
        "diagnostics": diagnostics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
