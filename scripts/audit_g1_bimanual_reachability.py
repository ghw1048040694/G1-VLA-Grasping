#!/usr/bin/env python3
"""Audit bimanual pre-grasp reachability across tote work distances."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


TOTE_X_POSITIONS_M = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
PALM_SITES = ("left_palm_center", "right_palm_center")
TOTE_SITES = ("tote_left_grasp_site", "tote_right_grasp_site")
PREGRASP_CLEARANCE_M = 0.12


def add_palm_sites(root: ET.Element) -> None:
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        if body is None:
            raise RuntimeError(f"Missing {side} wrist-yaw body")
        ET.SubElement(
            body,
            "site",
            name=f"{side}_palm_center",
            type="sphere",
            pos="0.14 0 0",
            size="0.012",
            rgba="1 0.85 0.1 1",
        )


def remove_calibration_markers(root: ET.Element) -> None:
    for worldbody in root.findall("worldbody"):
        for geom in list(worldbody.findall("geom")):
            if geom.get("name") in {"left_marker", "right_marker"}:
                worldbody.remove(geom)


def object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or f"unnamed_{object_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(args.source_asset.resolve())
    root = tree.getroot()
    remove_calibration_markers(root)
    add_palm_sites(root)
    asset_path = args.output_dir / "g1_warehouse_bimanual_reachability.xml"
    tree.write(asset_path, encoding="unicode", xml_declaration=False)

    model = mujoco.MjModel.from_xml_path(str(asset_path.resolve()))
    data = mujoco.MjData(model)
    tote_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "warehouse_tote_freejoint"
    )
    tote_qpos_adr = int(model.jnt_qposadr[tote_joint_id])
    table_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "calibration_table"
    )
    palm_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_SITES
    ]
    tote_site_ids = [
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
    orientation_weight_m_per_rad = 0.08

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.55, 0.0, 0.9)
    camera.distance = 2.4
    camera.azimuth = 145
    camera.elevation = -10
    candidates = []

    for tote_x in TOTE_X_POSITIONS_M:
        mujoco.mj_resetData(model, data)
        data.qpos[tote_qpos_adr] = tote_x
        model.geom_pos[table_geom_id, 0] = tote_x + 0.15
        mujoco.mj_forward(model, data)
        target_positions = np.stack([data.site_xpos[site_id].copy() for site_id in tote_site_ids])
        target_positions[0, 1] += PREGRASP_CLEARANCE_M
        target_positions[1, 1] -= PREGRASP_CLEARANCE_M

        def set_configuration(values: np.ndarray) -> None:
            data.qpos[qpos_ids] = values
            mujoco.mj_forward(model, data)

        def residual(values: np.ndarray) -> np.ndarray:
            set_configuration(values)
            position_error = np.concatenate(
                [data.site_xpos[site_id] - target for site_id, target in zip(palm_site_ids, target_positions)]
            )
            orientation_error = np.concatenate(
                [
                    Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_rotvec()
                    for site_id in palm_site_ids
                ]
            )
            return np.concatenate(
                (
                    position_error,
                    orientation_weight_m_per_rad * orientation_error,
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
        set_configuration(solution.x)
        palm_positions = np.stack([data.site_xpos[site_id].copy() for site_id in palm_site_ids])
        palm_errors = np.linalg.norm(palm_positions - target_positions, axis=1)
        palm_orientation_errors = np.array(
            [
                np.linalg.norm(
                    Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_rotvec()
                )
                for site_id in palm_site_ids
            ]
        )
        joint_margin = float(np.min(np.minimum(solution.x - lower, upper - solution.x)))

        unexpected_contacts = []
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
            body_names = {
                object_name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[int(contact.geom1)]),
                ),
                object_name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[int(contact.geom2)]),
                ),
            }
            unexpected_contacts.append(
                {"geoms": sorted(geom_names), "bodies": sorted(body_names)}
            )

        renderer.update_scene(data, camera=camera)
        image_path = args.output_dir / f"reachability_x_{int(tote_x * 100):03d}cm.png"
        Image.fromarray(renderer.render()).save(image_path)
        passed = bool(
            float(np.max(palm_errors)) < 0.03
            and float(np.max(palm_orientation_errors)) < 0.35
            and joint_margin >= 0.0499
            and not unexpected_contacts
            and solution.success
        )
        candidates.append(
            {
                "tote_x_m": tote_x,
                "left_palm_error_m": float(palm_errors[0]),
                "right_palm_error_m": float(palm_errors[1]),
                "maximum_palm_error_m": float(np.max(palm_errors)),
                "left_palm_orientation_error_rad": float(palm_orientation_errors[0]),
                "right_palm_orientation_error_rad": float(palm_orientation_errors[1]),
                "maximum_palm_orientation_error_rad": float(
                    np.max(palm_orientation_errors)
                ),
                "minimum_selected_joint_margin_rad": joint_margin,
                "unexpected_contact_count": len(unexpected_contacts),
                "unexpected_contacts": unexpected_contacts,
                "ik_solver_success": bool(solution.success),
                "ik_function_evaluations": int(solution.nfev),
                "joint_solution_rad": {
                    name: float(value) for name, value in zip(selected_names, solution.x)
                },
                "image": str(image_path),
                "passed": passed,
            }
        )

    renderer.close()
    passing = [candidate for candidate in candidates if candidate["passed"]]
    selected = max(passing, key=lambda item: item["tote_x_m"]) if passing else None
    report = {
        "experiment": "G1WH-14-bimanual-reachability-scan",
        "source_asset": str(args.source_asset.resolve()),
        "derived_asset": str(asset_path.resolve()),
        "target_definition": (
            "Palm centers 0.12 m outside the tote left/right grasp sites at tote center height"
        ),
        "pass_thresholds": {
            "maximum_palm_error_m": 0.03,
            "maximum_palm_orientation_error_rad": 0.35,
            "minimum_selected_joint_margin_rad": 0.05,
            "unexpected_contact_count": 0,
        },
        "selection_rule": "Farthest tote x position among passing candidates",
        "candidates": candidates,
        "selected_tote_x_m": selected["tote_x_m"] if selected else None,
        "scan_passed": selected is not None,
    }
    report_path = args.output_dir / "bimanual_reachability_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
