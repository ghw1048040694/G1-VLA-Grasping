#!/usr/bin/env python3
"""Audit whether the G1 controller can hold a static pre-grasp configuration."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from validate_g1_bimanual_actuation import apply_regularized_dynamics, unitree_gains


PALM_SITES = ("left_palm_center", "right_palm_center")
TOTE_SITES = ("tote_left_grasp_site", "tote_right_grasp_site")
PREGRASP_CLEARANCE_M = 0.12


def object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or f"unnamed_{object_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--enable-gravity", action="store_true")
    parser.add_argument("--gravity-compensation", action="store_true")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.gravity_compensation and not args.enable_gravity:
        raise ValueError("Gravity compensation requires gravity to be enabled")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reachability = json.loads(args.reachability_report.read_text())
    selected_x = reachability["selected_tote_x_m"]
    candidate = next(
        item for item in reachability["candidates"] if item["tote_x_m"] == selected_x
    )
    selected_targets = candidate["joint_solution_rad"]

    model = mujoco.MjModel.from_xml_path(str(args.asset.resolve()))
    if not args.enable_gravity:
        model.opt.gravity[:] = 0.0
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    tote_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "warehouse_tote_freejoint"
    )
    tote_qpos_adr = int(model.jnt_qposadr[tote_joint_id])
    tote_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "warehouse_tote")
    palm_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_SITES
    ]
    tote_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in TOTE_SITES
    ]

    controlled = {}
    for actuator_id in range(model.nu):
        name = object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            controlled[name] = {
                "actuator_id": actuator_id,
                "joint_id": joint_id,
                "qpos_id": int(model.jnt_qposadr[joint_id]),
                "qvel_id": int(model.jnt_dofadr[joint_id]),
            }
    targets = {
        name: float(data.qpos[item["qpos_id"]]) for name, item in controlled.items()
    }
    for name, value in selected_targets.items():
        targets[name] = value
        data.qpos[controlled[name]["qpos_id"]] = value
    for name, item in controlled.items():
        if "hand_" not in name:
            continue
        low, high = model.jnt_range[item["joint_id"]]
        if abs(float(low)) < 1e-9:
            targets[name] = 0.01
        elif abs(float(high)) < 1e-9:
            targets[name] = -0.01
        else:
            continue
        data.qpos[item["qpos_id"]] = targets[name]
    mujoco.mj_forward(model, data)
    initial_tote_qpos = data.qpos[tote_qpos_adr : tote_qpos_adr + 7].copy()
    target_positions = np.stack([data.site_xpos[site_id].copy() for site_id in tote_site_ids])
    target_positions[0, 1] += PREGRASP_CLEARANCE_M
    target_positions[1, 1] -= PREGRASP_CLEARANCE_M

    initial_unexpected_contacts = 0
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
        initial_unexpected_contacts += 1

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.50, 0.0, 0.9)
    camera.distance = 2.4
    camera.azimuth = 145
    camera.elevation = -10
    video_path = args.output_dir / "pregrasp_hold.mp4"
    writer = imageio.get_writer(video_path, fps=args.video_fps, codec="libx264", quality=8)

    dt = float(model.opt.timestep)
    total_steps = int(args.duration / dt)
    render_interval = max(1, round(1.0 / (args.video_fps * dt)))
    selected_errors = []
    selected_minimum_margin = math.inf
    limit_violations = 0
    joint_samples = 0
    saturated_samples = 0
    actuator_samples = 0
    contact_stats = defaultdict(lambda: {"samples": 0, "maximum_normal_force_n": 0.0})

    for step in range(total_steps):
        for name, item in controlled.items():
            kp, kd = unitree_gains(name)
            kp *= args.kp_scale
            kd *= args.kd_scale
            qpos = data.qpos[item["qpos_id"]]
            qvel = data.qvel[item["qvel_id"]]
            torque = kp * (targets[name] - qpos) - kd * qvel
            if args.gravity_compensation:
                torque += float(data.qfrc_bias[item["qvel_id"]])
            data.ctrl[item["actuator_id"]] = torque
        mujoco.mj_step(model, data)

        for name, item in controlled.items():
            joint_id = item["joint_id"]
            position = float(data.qpos[item["qpos_id"]])
            low, high = model.jnt_range[joint_id]
            limit_violations += int(position < low - 1e-6 or position > high + 1e-6)
            joint_samples += 1
            force_limit = max(abs(float(value)) for value in model.jnt_actfrcrange[joint_id])
            if force_limit > 0 and abs(float(data.actuator_force[item["actuator_id"]])) >= 0.98 * force_limit:
                saturated_samples += 1
            actuator_samples += 1
            if name in selected_targets:
                selected_errors.append(position - targets[name])
                selected_minimum_margin = min(
                    selected_minimum_margin, position - float(low), float(high) - position
                )

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
            bodies = tuple(
                sorted(
                    {
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
                )
            )
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, force)
            stats = contact_stats[bodies]
            stats["samples"] += 1
            stats["maximum_normal_force_n"] = max(
                stats["maximum_normal_force_n"], abs(float(force[0]))
            )
        if step % render_interval == 0:
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())

    writer.close()
    renderer.close()
    mujoco.mj_forward(model, data)
    palm_positions = np.stack([data.site_xpos[site_id].copy() for site_id in palm_site_ids])
    palm_errors = np.linalg.norm(palm_positions - target_positions, axis=1)
    orientation_errors = [
        np.linalg.norm(Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_rotvec())
        for site_id in palm_site_ids
    ]
    final_tote_qpos = data.qpos[tote_qpos_adr : tote_qpos_adr + 7].copy()
    contacts = [
        {
            "bodies": list(bodies),
            "contact_samples": stats["samples"],
            "maximum_normal_force_n": stats["maximum_normal_force_n"],
        }
        for bodies, stats in contact_stats.items()
    ]
    report = {
        "experiment": "G1WH-16-pregrasp-hold-control-ablation",
        "gravity_enabled": args.enable_gravity,
        "gravity_compensation_enabled": args.gravity_compensation,
        "kp_scale": args.kp_scale,
        "kd_scale": args.kd_scale,
        "initial_unexpected_contact_count": initial_unexpected_contacts,
        "maximum_palm_position_error_m": float(np.max(palm_errors)),
        "maximum_palm_orientation_error_rad": float(np.max(orientation_errors)),
        "selected_joint_rmse_rad": float(np.sqrt(np.mean(np.square(selected_errors)))),
        "minimum_selected_joint_margin_rad": selected_minimum_margin,
        "joint_limit_violation_fraction": limit_violations / joint_samples,
        "actuator_saturation_fraction": saturated_samples / actuator_samples,
        "unexpected_contact_pair_count": len(contacts),
        "unexpected_contacts": contacts,
        "tote_horizontal_drift_m": float(
            np.linalg.norm(final_tote_qpos[:2] - initial_tote_qpos[:2])
        ),
        "tote_orientation_change_rad": float(
            2.0 * math.acos(np.clip(abs(final_tote_qpos[3]), 0.0, 1.0))
        ),
        "tote_final_linear_speed_m_s": float(np.linalg.norm(data.cvel[tote_body_id, 3:])),
        "video": str(video_path),
    }
    report["passed"] = bool(
        initial_unexpected_contacts == 0
        and report["maximum_palm_position_error_m"] < 0.03
        and report["maximum_palm_orientation_error_rad"] < 0.35
        and report["selected_joint_rmse_rad"] < 0.05
        and selected_minimum_margin > 0.02
        and report["joint_limit_violation_fraction"] == 0.0
        and report["actuator_saturation_fraction"] < 0.05
        and not contacts
        and report["tote_horizontal_drift_m"] < 0.005
        and report["tote_orientation_change_rad"] < 0.02
        and report["tote_final_linear_speed_m_s"] < 0.01
    )
    report_path = args.output_dir / "pregrasp_hold_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
