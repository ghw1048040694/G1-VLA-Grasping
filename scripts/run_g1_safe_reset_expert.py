#!/usr/bin/env python3
"""Solve a collision-free reset pose and record a pre-grasp expert episode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


PALM_NAMES = ("left_palm_center", "right_palm_center")
SAFE_HOME_POSITIONS_M = np.array(((0.25, 0.38, 1.0), (0.25, -0.38, 1.0)))
SAFE_PREGRASP_CLEARANCE_M = 0.14


def name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"unnamed_{index}"


def unexpected_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[list[str]]:
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        body_names = sorted(
            {
                name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[int(contact.geom1)])),
                name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[int(contact.geom2)])),
            }
        )
        if "world" in body_names and any("foot" in item for item in body_names):
            continue
        if "warehouse_tote" in body_names and "world" in body_names:
            continue
        contacts.append(body_names)
    return contacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.asset.resolve()))
    data = mujoco.MjData(model)
    selected_joint_ids = []
    selected_names = []
    for joint_id in range(model.njnt):
        joint_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name.startswith("waist_") or (
            (joint_name.startswith("left_") or joint_name.startswith("right_"))
            and any(part in joint_name for part in ("shoulder_", "elbow_", "wrist_"))
        ):
            selected_joint_ids.append(joint_id)
            selected_names.append(joint_name)
    qpos_ids = np.asarray([model.jnt_qposadr[item] for item in selected_joint_ids])
    lower = np.asarray([model.jnt_range[item, 0] for item in selected_joint_ids])
    upper = np.asarray([model.jnt_range[item, 1] for item in selected_joint_ids])
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, palm_name)
        for palm_name in PALM_NAMES
    ]

    def residual(values: np.ndarray) -> np.ndarray:
        data.qpos[qpos_ids] = values
        mujoco.mj_forward(model, data)
        position_error = np.concatenate(
            [data.site_xpos[site] - target for site, target in zip(palm_ids, SAFE_HOME_POSITIONS_M)]
        )
        orientation_error = np.concatenate(
            [
                Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_rotvec()
                for site in palm_ids
            ]
        )
        return np.concatenate((position_error, 0.08 * orientation_error, 0.005 * values))

    solution = least_squares(
        residual,
        np.zeros(len(selected_joint_ids)),
        bounds=(lower + 0.05, upper - 0.05),
        max_nfev=1000,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    data.qpos[qpos_ids] = solution.x
    for joint_id in range(model.njnt):
        joint_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if "hand_" not in joint_name:
            continue
        low, high = model.jnt_range[joint_id]
        qpos_id = model.jnt_qposadr[joint_id]
        if abs(float(low)) < 1e-9:
            data.qpos[qpos_id] = 0.01
        elif abs(float(high)) < 1e-9:
            data.qpos[qpos_id] = -0.01
    mujoco.mj_forward(model, data)
    position_error = max(
        np.linalg.norm(data.site_xpos[site] - target)
        for site, target in zip(palm_ids, SAFE_HOME_POSITIONS_M)
    )
    orientation_error = max(
        np.linalg.norm(Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_rotvec())
        for site in palm_ids
    )
    contacts = unexpected_contacts(model, data)
    home_report = {
        "experiment": "G1WH-20-safe-reset-expert-demonstration",
        "joint_positions_rad": {
            joint_name: float(value) for joint_name, value in zip(selected_names, solution.x)
        },
        "maximum_palm_position_error_m": float(position_error),
        "maximum_palm_orientation_error_rad": float(orientation_error),
        "unexpected_contacts": contacts,
        "passed": bool(solution.success and position_error < 0.01 and not contacts),
    }
    home_report_path = args.output_dir / "safe_reset_pose.json"
    home_report_path.write_text(json.dumps(home_report, indent=2) + "\n", encoding="utf-8")
    if not home_report["passed"]:
        raise RuntimeError("Safe reset pose did not pass its contract")

    reachability = json.loads(args.reachability_report.read_text())
    seed_candidate = reachability["candidates"][0]
    seed = np.asarray([seed_candidate["joint_solution_rad"][item] for item in selected_names])
    tote_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, item)
        for item in ("tote_left_grasp_site", "tote_right_grasp_site")
    ]
    mujoco.mj_forward(model, data)
    pregrasp_positions = np.stack([data.site_xpos[item].copy() for item in tote_ids])
    pregrasp_positions[0, 1] += SAFE_PREGRASP_CLEARANCE_M
    pregrasp_positions[1, 1] -= SAFE_PREGRASP_CLEARANCE_M

    def pregrasp_residual(values: np.ndarray) -> np.ndarray:
        data.qpos[qpos_ids] = values
        mujoco.mj_forward(model, data)
        position_error_vector = np.concatenate(
            [data.site_xpos[site] - target for site, target in zip(palm_ids, pregrasp_positions)]
        )
        orientation_error_vector = np.concatenate(
            [
                Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_rotvec()
                for site in palm_ids
            ]
        )
        return np.concatenate(
            (position_error_vector, 0.08 * orientation_error_vector, 0.005 * values)
        )

    pregrasp_solution = least_squares(
        pregrasp_residual,
        seed,
        bounds=(lower + 0.05, upper - 0.05),
        max_nfev=1000,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    data.qpos[qpos_ids] = pregrasp_solution.x
    mujoco.mj_forward(model, data)
    pregrasp_position_error = max(
        np.linalg.norm(data.site_xpos[site] - target)
        for site, target in zip(palm_ids, pregrasp_positions)
    )
    pregrasp_orientation_error = max(
        np.linalg.norm(Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_rotvec())
        for site in palm_ids
    )
    pregrasp_contacts = unexpected_contacts(model, data)
    pregrasp_report = {
        "selected_tote_x_m": 0.55,
        "pregrasp_clearance_m": SAFE_PREGRASP_CLEARANCE_M,
        "candidates": [
            {
                "tote_x_m": 0.55,
                "passed": bool(
                    pregrasp_solution.success
                    and pregrasp_position_error < 0.03
                    and pregrasp_orientation_error < 0.35
                    and not pregrasp_contacts
                ),
                "joint_solution_rad": {
                    joint_name: float(value)
                    for joint_name, value in zip(selected_names, pregrasp_solution.x)
                },
            }
        ],
    }
    pregrasp_report_path = args.output_dir / "safe_pregrasp_target.json"
    pregrasp_report_path.write_text(
        json.dumps(pregrasp_report, indent=2) + "\n", encoding="utf-8"
    )
    if not pregrasp_report["candidates"][0]["passed"]:
        raise RuntimeError("Safe pre-grasp target did not pass its contract")

    subprocess.run(
        [
            str(args.python),
            str(args.runner),
            "--asset",
            str(args.asset.resolve()),
            "--reachability-report",
            str(pregrasp_report_path),
            "--initial-pose-report",
            str(home_report_path),
            "--output-dir",
            str(args.output_dir),
            "--tote-x",
            "0.55",
            "--shoulder-lead-seconds",
            "0.0",
            "--duration",
            "6.0",
            "--gravity-compensation",
            "--kp-scale",
            "1.5",
            "--kd-scale",
            str(np.sqrt(1.5)),
            "--record-demonstration",
            "--pregrasp-clearance-m",
            str(SAFE_PREGRASP_CLEARANCE_M),
        ],
        check=True,
        env={**os.environ, "MUJOCO_GL": "egl"},
    )
    trajectory = json.loads((args.output_dir / "pregrasp_trajectory_audit.json").read_text())
    summary = {
        "experiment": "G1WH-20-safe-reset-expert-demonstration",
        "safe_reset": home_report,
        "trajectory": trajectory,
        "dataset_recorded": (args.output_dir / "expert_episode.npz").exists(),
        "experiment_passed": bool(home_report["passed"] and trajectory["passed"]),
    }
    summary_path = args.output_dir / "safe_reset_expert_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
