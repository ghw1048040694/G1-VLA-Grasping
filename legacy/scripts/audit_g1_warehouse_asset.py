#!/usr/bin/env python3
"""Audit the G1 body-and-hands asset before warehouse task integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
from PIL import Image


def names(model: mujoco.MjModel, object_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, object_type, index) or "" for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.asset.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_names = names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    actuator_names = names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
    camera_names = names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
    hand_joints = [name for name in joint_names if "hand_" in name]
    body_joints = [name for name in joint_names if name and name != "floating_base_joint" and "hand_" not in name]
    left_hand_joints = [name for name in hand_joints if name.startswith("left_")]
    right_hand_joints = [name for name in hand_joints if name.startswith("right_")]
    actuator_joint_ids = [int(model.actuator_trnid[index, 0]) for index in range(model.nu)]
    actuated_joint_names = [joint_names[index] for index in actuator_joint_ids]
    body_names = names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    hand_geom_ids = [
        index for index in range(model.ngeom) if "hand_" in body_names[int(model.geom_bodyid[index])]
    ]

    checks = {
        "29_body_joints": len(body_joints) == 29,
        "14_hand_joints": len(hand_joints) == 14,
        "7_joints_per_hand": len(left_hand_joints) == 7 and len(right_hand_joints) == 7,
        "43_unique_actuators": model.nu == 43 and len(set(actuator_joint_ids)) == 43,
        "all_non_base_joints_actuated": set(actuated_joint_names) == set(body_joints + hand_joints),
        "hand_collision_geometry_present": bool(hand_geom_ids),
        "task_camera_present": model.ncam > 0,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.0, 0.0, 0.85)
    camera.distance = 2.8
    camera.azimuth = 145
    camera.elevation = -12
    renderer.update_scene(data, camera=camera)
    image_path = args.output_dir / "initial_pose.png"
    Image.fromarray(renderer.render()).save(image_path)
    renderer.close()

    report = {
        "experiment": "G1WH-01-asset-contract-audit",
        "asset": str(args.asset.resolve()),
        "dimensions": {
            "qpos": model.nq,
            "qvel": model.nv,
            "actuators": model.nu,
            "body_joints": len(body_joints),
            "hand_joints": len(hand_joints),
            "bodies": model.nbody,
            "geometries": model.ngeom,
            "cameras": model.ncam,
            "sensors": model.nsensor,
        },
        "total_mass_kg": float(model.body_mass.sum()),
        "body_joints": body_joints,
        "left_hand_joints": left_hand_joints,
        "right_hand_joints": right_hand_joints,
        "actuators": actuator_names,
        "cameras": camera_names,
        "checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "next_action": (
            "Add head and wrist task cameras before collecting warehouse demonstrations."
            if not checks["task_camera_present"]
            else "Proceed to contact and action-mapping validation."
        ),
    }
    report_path = args.output_dir / "asset_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")
    print(f"Saved {image_path}")


if __name__ == "__main__":
    main()
