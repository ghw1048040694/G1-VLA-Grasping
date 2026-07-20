#!/usr/bin/env python3
"""Execute and audit one continuous G1 bimanual pre-grasp trajectory."""

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


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or f"unnamed_{object_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tote-x", type=float, required=True)
    parser.add_argument("--shoulder-lead-seconds", type=float, required=True)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--gravity-compensation", action="store_true")
    parser.add_argument("--kp-scale", type=float, default=1.0)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reachability = json.loads(args.reachability_report.read_text())
    selected_x = args.tote_x
    selected_candidate = next(
        item
        for item in reachability["candidates"]
        if abs(item["tote_x_m"] - selected_x) < 1e-9
    )
    if not selected_candidate["passed"]:
        raise RuntimeError(f"Requested tote x={selected_x} did not pass G1WH-14")
    selected_targets = selected_candidate["joint_solution_rad"]

    model = mujoco.MjModel.from_xml_path(str(args.asset.resolve()))
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
    initial_targets = {
        name: float(data.qpos[item["qpos_id"]]) for name, item in controlled.items()
    }
    for name, item in controlled.items():
        if "hand_" not in name:
            continue
        low, high = model.jnt_range[item["joint_id"]]
        if abs(float(low)) < 1e-9:
            initial_targets[name] = 0.01
        elif abs(float(high)) < 1e-9:
            initial_targets[name] = -0.01
        else:
            continue
        data.qpos[item["qpos_id"]] = initial_targets[name]
    mujoco.mj_forward(model, data)
    initial_tote_qpos = data.qpos[tote_qpos_adr : tote_qpos_adr + 7].copy()
    target_positions = np.stack([data.site_xpos[site_id].copy() for site_id in tote_site_ids])
    target_positions[0, 1] += PREGRASP_CLEARANCE_M
    target_positions[1, 1] -= PREGRASP_CLEARANCE_M

    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.50, 0.0, 0.9)
    camera.distance = 2.4
    camera.azimuth = 145
    camera.elevation = -10
    video_path = args.output_dir / "pregrasp_trajectory.mp4"
    writer = imageio.get_writer(video_path, fps=args.video_fps, codec="libx264", quality=8)

    dt = float(model.opt.timestep)
    total_steps = int(args.duration / dt)
    render_interval = max(1, round(1.0 / (args.video_fps * dt)))
    unexpected_contact_stats = defaultdict(
        lambda: {"samples": 0, "maximum_normal_force_n": 0.0}
    )
    selected_hold_errors = []
    saturated_samples = 0
    actuator_samples = 0
    selected_minimum_margin = math.inf
    joint_limit_violations = 0
    joint_samples = 0

    for step in range(total_steps):
        time_s = step * dt
        shoulder_scale = smoothstep((time_s - 1.0) / 1.0)
        other_scale = smoothstep((time_s - 1.0 - args.shoulder_lead_seconds) / 1.0)
        commanded_targets = dict(initial_targets)
        for name, final_target in selected_targets.items():
            scale = shoulder_scale if "shoulder_" in name else other_scale
            commanded_targets[name] = initial_targets[name] + scale * (
                final_target - initial_targets[name]
            )
        for name, item in controlled.items():
            kp, kd = unitree_gains(name)
            kp *= args.kp_scale
            kd *= args.kd_scale
            qpos = data.qpos[item["qpos_id"]]
            qvel = data.qvel[item["qvel_id"]]
            torque = kp * (commanded_targets[name] - qpos) - kd * qvel
            if args.gravity_compensation:
                torque += float(data.qfrc_bias[item["qvel_id"]])
            data.ctrl[item["actuator_id"]] = torque
        mujoco.mj_step(model, data)

        for name, item in controlled.items():
            joint_id = item["joint_id"]
            force_limit = max(abs(float(value)) for value in model.jnt_actfrcrange[joint_id])
            if force_limit > 0 and abs(float(data.actuator_force[item["actuator_id"]])) >= 0.98 * force_limit:
                saturated_samples += 1
            actuator_samples += 1
            low, high = model.jnt_range[joint_id]
            position = float(data.qpos[item["qpos_id"]])
            joint_limit_violations += int(position < low - 1e-6 or position > high + 1e-6)
            joint_samples += 1
            if name in selected_targets:
                selected_minimum_margin = min(
                    selected_minimum_margin, position - float(low), float(high) - position
                )
                if time_s >= 4.5:
                    selected_hold_errors.append(position - selected_targets[name])

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
            body_names = tuple(
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
            stats = unexpected_contact_stats[body_names]
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
    palm_orientation_errors = np.array(
        [
            np.linalg.norm(
                Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_rotvec()
            )
            for site_id in palm_site_ids
        ]
    )
    final_tote_qpos = data.qpos[tote_qpos_adr : tote_qpos_adr + 7].copy()
    tote_horizontal_drift = float(
        np.linalg.norm(final_tote_qpos[:2] - initial_tote_qpos[:2])
    )
    tote_orientation_change = float(
        2.0 * math.acos(np.clip(abs(final_tote_qpos[3]), 0.0, 1.0))
    )
    tote_final_speed = float(np.linalg.norm(data.cvel[tote_body_id, 3:]))
    selected_rmse = float(np.sqrt(np.mean(np.square(selected_hold_errors))))
    contacts = [
        {
            "bodies": list(pair),
            "contact_samples": stats["samples"],
            "maximum_normal_force_n": stats["maximum_normal_force_n"],
        }
        for pair, stats in unexpected_contact_stats.items()
    ]
    report = {
        "experiment": "G1WH-15-continuous-pregrasp-trajectory",
        "shoulder_lead_seconds": args.shoulder_lead_seconds,
        "selected_tote_x_m": selected_x,
        "gravity_compensation_enabled": args.gravity_compensation,
        "kp_scale": args.kp_scale,
        "kd_scale": args.kd_scale,
        "maximum_palm_position_error_m": float(np.max(palm_errors)),
        "maximum_palm_orientation_error_rad": float(np.max(palm_orientation_errors)),
        "selected_joint_hold_rmse_rad": selected_rmse,
        "minimum_selected_joint_margin_rad": selected_minimum_margin,
        "joint_limit_violation_fraction": joint_limit_violations / joint_samples,
        "actuator_saturation_fraction": saturated_samples / actuator_samples,
        "unexpected_contact_pair_count": len(contacts),
        "unexpected_contacts": contacts,
        "tote_horizontal_drift_m": tote_horizontal_drift,
        "tote_orientation_change_rad": tote_orientation_change,
        "tote_final_linear_speed_m_s": tote_final_speed,
        "video": str(video_path),
    }
    report["passed"] = bool(
        report["maximum_palm_position_error_m"] < 0.03
        and report["maximum_palm_orientation_error_rad"] < 0.35
        and selected_rmse < 0.05
        and selected_minimum_margin > 0.02
        and report["joint_limit_violation_fraction"] == 0.0
        and report["actuator_saturation_fraction"] < 0.05
        and not contacts
        and tote_horizontal_drift < 0.005
        and tote_orientation_change < 0.02
        and tote_final_speed < 0.01
    )
    report_path = args.output_dir / "pregrasp_trajectory_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
