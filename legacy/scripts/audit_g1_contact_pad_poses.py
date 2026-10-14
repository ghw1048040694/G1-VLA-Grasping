#!/usr/bin/env python3
"""Find IK-reachable G1 poses that create real hand-object contacts in MuJoCo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

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
from collect_g1_language_pick_place_dataset import contact_profile_for_target_slot


PAD_HALF_EXTENT_X = {"red_triangle": 0.045, "yellow_rod": 0.060, "green_cube": 0.025}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def open_hand_value(model: mujoco.MjModel, item: dict) -> float:
    low, high = model.jnt_range[item["joint_id"]]
    if abs(float(low)) < 1e-9:
        return 0.01
    if abs(float(high)) < 1e-9:
        return -0.01
    return 0.5 * float(low + high)


def hand_object_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_geom_ids: set[int],
    side: str,
) -> list[str]:
    result = []
    for index in range(data.ncon):
        contact = data.contact[index]
        first, second = int(contact.geom1), int(contact.geom2)
        if first not in object_geom_ids and second not in object_geom_ids:
            continue
        other = second if first in object_geom_ids else first
        body = int(model.geom_bodyid[other])
        while body > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
            if name.startswith(f"{side}_"):
                result.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or "")
                break
            body = int(model.body_parentid[body])
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(args.asset))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    selected_names, qpos_ids, lower, upper = arm_ik_contract(model)
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in PALM_NAMES
    ]
    body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in OBJECT_NAMES
    }
    object_geoms = {
        name: {
            geom
            for geom in range(model.ngeom)
            if int(model.geom_bodyid[geom]) == body_id
        }
        for name, body_id in body_ids.items()
    }
    mujoco.mj_forward(model, data)
    home_pose, home_report = solve_pose(
        model,
        data,
        palm_ids,
        qpos_ids,
        lower,
        upper,
        HOME_POSITIONS_M,
        np.zeros(len(selected_names)),
        orientation_weight=0.06,
    )
    base_qpos = data.qpos.copy()
    base_qpos[qpos_ids] = home_pose
    for name, item in controlled.items():
        if "hand_" in name:
            base_qpos[item["qpos_id"]] = open_hand_value(model, item)

    summaries = {}
    slots = np.asarray(((0.36, -0.36), (0.33, -0.08), (0.35, 0.19)))
    for target in OBJECT_NAMES:
        mujoco.mj_resetData(model, data)
        data.qpos[:] = base_qpos
        mujoco.mj_forward(model, data)
        center = data.xpos[body_ids[target]].copy()
        active_arm = LEFT_PALM_INDEX if center[1] >= 0.05 else 1
        side = "left" if active_arm == LEFT_PALM_INDEX else "right"
        slot = int(np.argmin(np.linalg.norm(slots - center[:2], axis=1)))
        profile = contact_profile_for_target_slot(target, slot)
        yaw = float(profile["grasp_yaw_deg"])
        orientation = Rotation.from_euler("z", yaw, degrees=True).as_matrix()
        # The palm site lies ahead of the contact pad in the hand frame.
        site_from_pad = np.asarray(
            (0.070, -0.003 if active_arm == 0 else 0.003, 0.0)
        )
        expected_pad_x = -(PAD_HALF_EXTENT_X[target] + 0.025)
        candidates = []
        for dx in (expected_pad_x - 0.02, expected_pad_x, expected_pad_x + 0.02):
            for dy in (-0.02, 0.0, 0.02):
                for dz in (-0.02, 0.0, 0.02):
                    mujoco.mj_resetData(model, data)
                    data.qpos[:] = base_qpos
                    mujoco.mj_forward(model, data)
                    pad_goal = center + np.asarray((dx, dy, dz))
                    targets = HOME_POSITIONS_M.copy()
                    targets[active_arm] = pad_goal + orientation @ site_from_pad
                    PALM_ORIENTATION_TARGETS[:] = np.eye(3)
                    PALM_ORIENTATION_TARGETS[active_arm] = orientation
                    pose, report = solve_pose(
                        model,
                        data,
                        palm_ids,
                        qpos_ids,
                        lower,
                        upper,
                        targets,
                        home_pose,
                        orientation_weight=0.06,
                    )
                    if float(report["maximum_position_error_m"]) > 0.015:
                        continue
                    data.qpos[qpos_ids] = pose
                    for name, item in controlled.items():
                        if name.startswith(f"{side}_hand_"):
                            data.qpos[item["qpos_id"]] = object_hand_target(
                                target, name, HAND_TARGETS[name]
                            )
                    mujoco.mj_forward(model, data)
                    contacts = hand_object_contacts(
                        model, data, object_geoms[target], side
                    )
                    pad_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_palm_contact_pad"
                    )
                    candidates.append(
                        {
                            "pad_goal_offset_m": [float(dx), float(dy), float(dz)],
                            "palm_site_target_m": targets[active_arm].tolist(),
                            "yaw_deg": yaw,
                            "ik_position_error_m": float(report["maximum_position_error_m"]),
                            "pad_center_distance_m": float(
                                np.linalg.norm(data.geom_xpos[pad_id] - center)
                            ),
                            "contact_geoms": contacts,
                            "arm_pose_rad": pose.tolist(),
                        }
                    )
        contacts = [row for row in candidates if row["contact_geoms"]]
        contacts.sort(key=lambda row: row["pad_center_distance_m"])
        summaries[target] = {
            "active_arm": side,
            "object_center_m": center.tolist(),
            "searched_reachable_poses": len(candidates),
            "physical_contact_poses": len(contacts),
            "best_contact_pose": contacts[0] if contacts else None,
        }
        print(
            "CONTACT_PAD_AUDIT="
            + json.dumps({target: summaries[target]}, separators=(",", ":")),
            flush=True,
        )
    result = {
        "experiment": "G1LANG-physical-contact-pad-pose-audit",
        "asset": str(args.asset),
        "home_ik_position_error_m": float(home_report["maximum_position_error_m"]),
        "results": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
