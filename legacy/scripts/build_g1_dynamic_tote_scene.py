#!/usr/bin/env python3
"""Build and validate the first dynamic tote scene for the G1 warehouse task."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

from validate_g1_bimanual_actuation import apply_regularized_dynamics, unitree_gains


CAMERAS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
HAND_OPEN_MARGIN_RAD = 0.01


def remove_static_tote(root: ET.Element) -> None:
    for worldbody in root.findall("worldbody"):
        for geom in list(worldbody.findall("geom")):
            if geom.get("name") == "blue_tote":
                worldbody.remove(geom)


def add_dynamic_tote(root: ET.Element) -> None:
    scene = root.findall("worldbody")[-1]
    tote = ET.SubElement(scene, "body", name="warehouse_tote", pos="0.95 0 0.88")
    ET.SubElement(
        tote,
        "inertial",
        pos="0 0 0",
        mass="2.0",
        diaginertia="0.0435333 0.0283333 0.0493333",
    )
    ET.SubElement(tote, "freejoint", name="warehouse_tote_freejoint")
    wall_specs = {
        "tote_bottom": ("0 0 -0.12", "0.16 0.22 0.01"),
        "tote_front_wall": ("-0.15 0 0", "0.01 0.22 0.12"),
        "tote_back_wall": ("0.15 0 0", "0.01 0.22 0.12"),
        "tote_left_wall": ("0 0.21 0", "0.15 0.01 0.12"),
        "tote_right_wall": ("0 -0.21 0", "0.15 0.01 0.12"),
    }
    for name, (pos, size) in wall_specs.items():
        ET.SubElement(
            tote,
            "geom",
            name=name,
            type="box",
            pos=pos,
            size=size,
            friction="0.8 0.02 0.002",
            rgba="0.08 0.35 0.85 1",
        )
    ET.SubElement(
        tote,
        "site",
        name="tote_left_grasp_site",
        type="sphere",
        pos="0 0.235 0",
        size="0.015",
        rgba="0.95 0.2 0.15 1",
    )
    ET.SubElement(
        tote,
        "site",
        name="tote_right_grasp_site",
        type="sphere",
        pos="0 -0.235 0",
        size="0.015",
        rgba="0.15 0.9 0.25 1",
    )


def object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or f"unnamed_{object_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(args.source_asset.resolve())
    root = tree.getroot()
    remove_static_tote(root)
    add_dynamic_tote(root)
    asset_path = args.output_dir / "g1_warehouse_dynamic_tote.xml"
    tree.write(asset_path, encoding="unicode", xml_declaration=False)

    model = mujoco.MjModel.from_xml_path(str(asset_path.resolve()))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)

    tote_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "warehouse_tote")
    tote_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "warehouse_tote_freejoint"
    )
    tote_qpos_adr = int(model.jnt_qposadr[tote_joint_id])
    initial_tote_qpos = data.qpos[tote_qpos_adr : tote_qpos_adr + 7].copy()

    controlled_joints = {}
    for actuator_id in range(model.nu):
        actuator_name = object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, actuator_name)
        if joint_id >= 0:
            controlled_joints[actuator_name] = (
                actuator_id,
                int(model.jnt_qposadr[joint_id]),
                int(model.jnt_dofadr[joint_id]),
            )
    target_qpos = {
        name: float(data.qpos[qpos_id])
        for name, (_, qpos_id, _) in controlled_joints.items()
    }
    for name, (_, qpos_id, _) in controlled_joints.items():
        if "hand_" not in name:
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[joint_id]
        if abs(float(low)) < 1e-9:
            target_qpos[name] = HAND_OPEN_MARGIN_RAD
        elif abs(float(high)) < 1e-9:
            target_qpos[name] = -HAND_OPEN_MARGIN_RAD
        else:
            continue
        data.qpos[qpos_id] = target_qpos[name]
    mujoco.mj_forward(model, data)

    initial_penetration = max(
        (max(0.0, -float(data.contact[index].dist)) for index in range(data.ncon)),
        default=0.0,
    )
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.75, 0.0, 0.9)
    camera.distance = 2.5
    camera.azimuth = 145
    camera.elevation = -12
    video_path = args.output_dir / "dynamic_tote_settling.mp4"
    writer = imageio.get_writer(video_path, fps=args.video_fps, codec="libx264", quality=8)

    dt = float(model.opt.timestep)
    total_steps = int(args.duration / dt)
    render_interval = max(1, round(1.0 / (args.video_fps * dt)))
    table_contact_steps = 0
    maximum_tote_speed = 0.0
    for step in range(total_steps):
        for name, (actuator_id, qpos_id, qvel_id) in controlled_joints.items():
            kp, kd = unitree_gains(name)
            data.ctrl[actuator_id] = (
                kp * (target_qpos[name] - data.qpos[qpos_id]) - kd * data.qvel[qvel_id]
            )
        mujoco.mj_step(model, data)
        maximum_tote_speed = max(
            maximum_tote_speed, float(np.linalg.norm(data.cvel[tote_body_id, 3:]))
        )
        touching_table = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            names = {
                object_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)),
                object_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)),
            }
            if "calibration_table" in names and any(name.startswith("tote_") for name in names):
                touching_table = True
        table_contact_steps += int(touching_table)
        if step % render_interval == 0:
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    writer.close()

    final_tote_qpos = data.qpos[tote_qpos_adr : tote_qpos_adr + 7].copy()
    horizontal_drift = float(np.linalg.norm(final_tote_qpos[:2] - initial_tote_qpos[:2]))
    vertical_settle = float(final_tote_qpos[2] - initial_tote_qpos[2])
    orientation_change = float(
        2.0 * math.acos(np.clip(abs(final_tote_qpos[3]), 0.0, 1.0))
    )
    final_speed = float(np.linalg.norm(data.cvel[tote_body_id, 3:]))

    image_checks = {}
    for camera_name in CAMERAS:
        renderer.update_scene(data, camera=camera_name)
        frame = renderer.render()
        image_path = args.output_dir / f"{camera_name}.png"
        Image.fromarray(frame).save(image_path)
        blue_pixels = int(
            np.count_nonzero(
                (frame[:, :, 2] > 100)
                & (frame[:, :, 2] > frame[:, :, 0] * 1.3)
                & (frame[:, :, 2] > frame[:, :, 1] * 1.15)
            )
        )
        image_checks[camera_name] = {
            "path": str(image_path),
            "blue_tote_pixels": blue_pixels,
            "tote_visible": blue_pixels > 500,
        }
    renderer.close()

    report = {
        "experiment": "G1WH-13-dynamic-tote-scene-contract",
        "source_asset": str(args.source_asset.resolve()),
        "derived_asset": str(asset_path.resolve()),
        "tote": {
            "mass_kg": float(model.body_mass[tote_body_id]),
            "joint_type": "free",
            "collision_wall_count": 5,
            "grasp_sites": ["tote_left_grasp_site", "tote_right_grasp_site"],
        },
        "initial_maximum_penetration_m": initial_penetration,
        "table_contact_step_fraction": table_contact_steps / total_steps,
        "horizontal_drift_m": horizontal_drift,
        "vertical_settle_m": vertical_settle,
        "orientation_change_rad": orientation_change,
        "maximum_linear_speed_m_s": maximum_tote_speed,
        "final_linear_speed_m_s": final_speed,
        "image_checks": image_checks,
        "tote_visible_in_all_cameras": all(
            item["tote_visible"] for item in image_checks.values()
        ),
        "video": str(video_path),
    }
    report["contract_passed"] = bool(
        report["tote"]["mass_kg"] == 2.0
        and initial_penetration < 0.001
        and report["table_contact_step_fraction"] > 0.95
        and horizontal_drift < 0.005
        and abs(vertical_settle) < 0.005
        and orientation_change < 0.02
        and final_speed < 0.01
        and report["tote_visible_in_all_cameras"]
    )
    report_path = args.output_dir / "dynamic_tote_scene_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
