#!/usr/bin/env python3
"""Sweep IK orientation weights and validate each solution under compensated gravity."""

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


ORIENTATION_WEIGHTS_M_PER_RAD = (0.08, 0.06, 0.04, 0.02)
PALM_SITES = ("left_palm_center", "right_palm_center")
TOTE_SITES = ("tote_left_grasp_site", "tote_right_grasp_site")


def object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or f"unnamed_{object_id}"


def unexpected_contact_count(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    count = 0
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom_names = {
            object_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
            object_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
        }
        if "floor" in geom_names:
            continue
        if "calibration_table" in geom_names and any(
            name.startswith("tote_") for name in geom_names
        ):
            continue
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--hold-runner", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(args.asset.resolve()))
    data = mujoco.MjData(model)
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_SITES
    ]
    tote_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in TOTE_SITES
    ]
    selected_names = []
    selected_joint_ids = []
    for joint_id in range(model.njnt):
        name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name.startswith("waist_") or (
            (name.startswith("left_") or name.startswith("right_"))
            and any(part in name for part in ("shoulder_", "elbow_", "wrist_"))
        ):
            selected_names.append(name)
            selected_joint_ids.append(joint_id)
    qpos_ids = np.array([model.jnt_qposadr[joint_id] for joint_id in selected_joint_ids])
    lower = np.array([model.jnt_range[joint_id, 0] for joint_id in selected_joint_ids])
    upper = np.array([model.jnt_range[joint_id, 1] for joint_id in selected_joint_ids])
    nominal = np.zeros(len(selected_joint_ids), dtype=np.float64)
    regularization = 0.005
    groups = []

    for weight in ORIENTATION_WEIGHTS_M_PER_RAD:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        target_positions = np.stack([data.site_xpos[site_id].copy() for site_id in tote_ids])
        target_positions[0, 1] += 0.12
        target_positions[1, 1] -= 0.12

        def residual(values: np.ndarray) -> np.ndarray:
            data.qpos[qpos_ids] = values
            mujoco.mj_forward(model, data)
            position_error = np.concatenate(
                [data.site_xpos[site_id] - target for site_id, target in zip(palm_ids, target_positions)]
            )
            orientation_error = np.concatenate(
                [
                    Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_rotvec()
                    for site_id in palm_ids
                ]
            )
            return np.concatenate(
                (
                    position_error,
                    weight * orientation_error,
                    regularization * (values - nominal),
                )
            )

        solution = least_squares(
            residual,
            nominal.copy(),
            bounds=(lower + 0.05, upper - 0.05),
            max_nfev=500,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        data.qpos[qpos_ids] = solution.x
        mujoco.mj_forward(model, data)
        position_error = max(
            np.linalg.norm(data.site_xpos[site_id] - target)
            for site_id, target in zip(palm_ids, target_positions)
        )
        orientation_error = max(
            np.linalg.norm(
                Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_rotvec()
            )
            for site_id in palm_ids
        )
        margin = float(np.min(np.minimum(solution.x - lower, upper - solution.x)))
        static_contacts = unexpected_contact_count(model, data)
        group_name = f"orientation_weight_{int(weight * 100):03d}"
        group_dir = args.output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        target_report = {
            "selected_tote_x_m": 0.55,
            "candidates": [
                {
                    "tote_x_m": 0.55,
                    "passed": bool(
                        solution.success
                        and position_error < 0.03
                        and orientation_error < 0.35
                        and margin >= 0.0499
                        and static_contacts == 0
                    ),
                    "joint_solution_rad": {
                        name: float(value) for name, value in zip(selected_names, solution.x)
                    },
                }
            ],
        }
        target_report_path = group_dir / "ik_target_report.json"
        target_report_path.write_text(
            json.dumps(target_report, indent=2) + "\n", encoding="utf-8"
        )
        subprocess.run(
            [
                str(args.python),
                str(args.hold_runner),
                "--asset",
                str(args.asset.resolve()),
                "--reachability-report",
                str(target_report_path),
                "--output-dir",
                str(group_dir),
                "--enable-gravity",
                "--gravity-compensation",
            ],
            check=True,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        dynamic = json.loads((group_dir / "pregrasp_hold_audit.json").read_text())
        groups.append(
            {
                "group": group_name,
                "orientation_weight_m_per_rad": weight,
                "static_position_error_m": float(position_error),
                "static_orientation_error_rad": float(orientation_error),
                "static_minimum_joint_margin_rad": margin,
                "static_unexpected_contact_count": static_contacts,
                "ik_solver_success": bool(solution.success),
                "dynamic_hold": dynamic,
                "passed": bool(target_report["candidates"][0]["passed"] and dynamic["passed"]),
            }
        )

    passing = [group for group in groups if group["passed"]]
    selected = max(passing, key=lambda item: item["orientation_weight_m_per_rad"]) if passing else None
    summary = {
        "experiment": "G1WH-17-ik-task-weight-dynamic-sweep",
        "independent_variable": "orientation_weight_m_per_rad",
        "selection_rule": "Highest orientation weight among dynamically passing groups",
        "groups": groups,
        "selected_orientation_weight_m_per_rad": (
            selected["orientation_weight_m_per_rad"] if selected else None
        ),
        "experiment_passed": selected is not None,
    }
    summary_path = args.output_dir / "ik_weight_dynamic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
