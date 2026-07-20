#!/usr/bin/env python3
"""Drive the G1 waist, arms, and hands and record an actuation smoke test."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image


BODY_TARGETS = {
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.0,
    "left_ankle_pitch_joint": 0.0,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.0,
    "right_ankle_pitch_joint": 0.0,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.18,
    "waist_roll_joint": 0.08,
    "waist_pitch_joint": 0.12,
    "left_shoulder_pitch_joint": 0.45,
    "left_shoulder_roll_joint": 0.35,
    "left_shoulder_yaw_joint": 0.15,
    "left_elbow_joint": 0.80,
    "left_wrist_roll_joint": 0.10,
    "left_wrist_pitch_joint": 0.20,
    "left_wrist_yaw_joint": 0.10,
    "right_shoulder_pitch_joint": 0.45,
    "right_shoulder_roll_joint": -0.35,
    "right_shoulder_yaw_joint": -0.15,
    "right_elbow_joint": 0.80,
    "right_wrist_roll_joint": -0.10,
    "right_wrist_pitch_joint": 0.20,
    "right_wrist_yaw_joint": -0.10,
}

HAND_TARGETS = {
    "left_hand_thumb_0_joint": 0.40,
    "left_hand_thumb_1_joint": 0.60,
    "left_hand_thumb_2_joint": 1.20,
    "left_hand_middle_0_joint": -1.00,
    "left_hand_middle_1_joint": -1.20,
    "left_hand_index_0_joint": -1.00,
    "left_hand_index_1_joint": -1.20,
    "right_hand_thumb_0_joint": -0.40,
    "right_hand_thumb_1_joint": -0.60,
    "right_hand_thumb_2_joint": -1.20,
    "right_hand_middle_0_joint": 1.00,
    "right_hand_middle_1_joint": 1.20,
    "right_hand_index_0_joint": 1.00,
    "right_hand_index_1_joint": 1.20,
}


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def unitree_gains(name: str) -> tuple[float, float]:
    if "hand_" in name:
        return 1.5, 0.2
    if "wrist_" in name:
        return 40.0, 1.5
    weak_body = "ankle_pitch" in name or "shoulder_" in name or "elbow_" in name
    return (80.0, 3.0) if weak_body else (300.0, 3.0)


def apply_regularized_dynamics(model: mujoco.MjModel) -> None:
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not name or name == "floating_base_joint":
            continue
        dof_id = int(model.jnt_dofadr[joint_id])
        if "hand_" in name:
            damping, armature = 0.02, 0.0005
        elif "wrist_" in name:
            damping, armature = 0.1, 0.005
        else:
            damping, armature = 0.2, 0.01
        model.dof_damping[dof_id] = damping
        model.dof_armature[dof_id] = armature


def object_name(model: mujoco.MjModel, object_type: mujoco.mjtObj, object_id: int) -> str:
    return mujoco.mj_id2name(model, object_type, object_id) or f"unnamed_{object_id}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--body-kp", type=float, default=80.0)
    parser.add_argument("--body-kd", type=float, default=4.0)
    parser.add_argument("--hand-kp", type=float, default=8.0)
    parser.add_argument("--hand-kd", type=float, default=0.30)
    parser.add_argument("--hand-target-scale", type=float, default=1.0)
    parser.add_argument("--hand-open-margin", type=float, default=0.0)
    parser.add_argument("--shoulder-lead-seconds", type=float, default=0.0)
    parser.add_argument("--gain-profile", choices=("custom", "unitree"), default="custom")
    parser.add_argument("--dynamics-profile", choices=("raw", "both"), default="raw")
    parser.add_argument("--enable-gravity", action="store_true")
    parser.add_argument("--enable-contacts", action="store_true")
    parser.add_argument("--base-mode", choices=("clamp", "model"), default="clamp")
    parser.add_argument("--experiment-name", default="G1WH-03-bimanual-actuation-smoke-test")
    args = parser.parse_args()
    if not 0.0 <= args.hand_target_scale <= 1.0:
        raise ValueError("--hand-target-scale must be between 0 and 1")
    if args.hand_open_margin < 0.0:
        raise ValueError("--hand-open-margin must be non-negative")
    if args.shoulder_lead_seconds < 0.0:
        raise ValueError("--shoulder-lead-seconds must be non-negative")

    model = mujoco.MjModel.from_xml_path(str(args.asset.resolve()))
    if not args.enable_gravity:
        model.opt.gravity[:] = 0.0
    if not args.enable_contacts:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if args.dynamics_profile == "both":
        apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base_qpos = data.qpos[:7].copy()

    joint_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in BODY_TARGETS | HAND_TARGETS
    }
    if any(joint_id < 0 for joint_id in joint_ids.values()):
        raise RuntimeError("A commanded joint is missing from the asset")
    qpos_ids = {name: int(model.jnt_qposadr[joint_id]) for name, joint_id in joint_ids.items()}
    qvel_ids = {name: int(model.jnt_dofadr[joint_id]) for name, joint_id in joint_ids.items()}
    actuator_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in joint_ids
    }
    if any(actuator_id < 0 for actuator_id in actuator_ids.values()):
        raise RuntimeError("A commanded joint has no same-named actuator")

    open_hand_targets = {
        name: float(np.sign(target) * args.hand_open_margin)
        for name, target in HAND_TARGETS.items()
    }
    target_ranges_valid = {}
    scaled_hand_targets = {
        name: target * args.hand_target_scale for name, target in HAND_TARGETS.items()
    }
    for name, target in BODY_TARGETS.items():
        joint_id = joint_ids[name]
        low, high = model.jnt_range[joint_id]
        target_ranges_valid[name] = bool(low <= target <= high)
    for name, target in scaled_hand_targets.items():
        joint_id = joint_ids[name]
        low, high = model.jnt_range[joint_id]
        target_ranges_valid[name] = bool(
            low <= open_hand_targets[name] <= high and low <= target <= high
        )
    for name, target in open_hand_targets.items():
        data.qpos[qpos_ids[name]] = target
    mujoco.mj_forward(model, data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.45, 0.0, 0.9)
    camera.distance = 2.8
    camera.azimuth = 145
    camera.elevation = -10
    video_path = args.output_dir / "bimanual_actuation.mp4"
    writer = imageio.get_writer(video_path, fps=args.video_fps, codec="libx264", quality=8)

    dt = float(model.opt.timestep)
    total_steps = int(args.duration / dt)
    render_interval = max(1, round(1.0 / (args.video_fps * dt)))
    rows = []
    hold_errors = {name: [] for name in joint_ids}
    contact_counts = []
    saturated_actuators = 0
    actuator_saturation_counts = defaultdict(int)
    actuator_samples = 0
    joint_limit_violations = 0
    joint_limit_violation_counts = defaultdict(int)
    joint_samples = 0
    samples_per_joint = 0
    contact_pair_stats = defaultdict(
        lambda: {
            "contact_samples": 0,
            "active_steps": set(),
            "normal_force_sum_n": 0.0,
            "maximum_normal_force_n": 0.0,
            "maximum_penetration_m": 0.0,
            "first_contact_time_s": None,
            "last_contact_time_s": None,
        }
    )
    final_frame = None

    for step in range(total_steps):
        time_s = step * dt
        shoulder_scale = smoothstep((time_s - 1.0) / 1.0)
        body_scale = smoothstep((time_s - 1.0 - args.shoulder_lead_seconds) / 1.0)
        hand_scale = smoothstep((time_s - 3.0) / 1.0)
        if time_s > 5.0:
            hand_scale = 1.0 - smoothstep((time_s - 5.0) / 1.0)

        targets = {
            **{
                name: value * (shoulder_scale if "shoulder_" in name else body_scale)
                for name, value in BODY_TARGETS.items()
            },
            **{
                name: open_hand_targets[name]
                + hand_scale * (value - open_hand_targets[name])
                for name, value in scaled_hand_targets.items()
            },
        }
        for name, target in targets.items():
            qpos = data.qpos[qpos_ids[name]]
            qvel = data.qvel[qvel_ids[name]]
            is_hand = name in HAND_TARGETS
            if args.gain_profile == "unitree":
                kp, kd = unitree_gains(name)
            else:
                kp, kd = (
                    (args.hand_kp, args.hand_kd)
                    if is_hand
                    else (args.body_kp, args.body_kd)
                )
            data.ctrl[actuator_ids[name]] = kp * (target - qpos) - kd * qvel

        mujoco.mj_step(model, data)
        if args.base_mode == "clamp":
            data.qpos[:7] = base_qpos
            data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)
        contact_counts.append(int(data.ncon))
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)
            geom1_name = object_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1_id)
            geom2_name = object_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2_id)
            body1_name = object_name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1_id])
            )
            body2_name = object_name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2_id])
            )
            pair = tuple(sorted(((geom1_name, body1_name), (geom2_name, body2_name))))
            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, contact_force)
            normal_force = abs(float(contact_force[0]))
            stats = contact_pair_stats[pair]
            stats["contact_samples"] += 1
            stats["active_steps"].add(step)
            stats["normal_force_sum_n"] += normal_force
            stats["maximum_normal_force_n"] = max(
                stats["maximum_normal_force_n"], normal_force
            )
            stats["maximum_penetration_m"] = max(
                stats["maximum_penetration_m"], max(0.0, -float(contact.dist))
            )
            if stats["first_contact_time_s"] is None:
                stats["first_contact_time_s"] = time_s
            stats["last_contact_time_s"] = time_s
        for name, actuator_id in actuator_ids.items():
            joint_id = joint_ids[name]
            force_limit = max(abs(float(value)) for value in model.jnt_actfrcrange[joint_id])
            if force_limit > 0 and abs(float(data.actuator_force[actuator_id])) >= 0.98 * force_limit:
                saturated_actuators += 1
                actuator_saturation_counts[name] += 1
            actuator_samples += 1
            low, high = model.jnt_range[joint_id]
            position = float(data.qpos[qpos_ids[name]])
            if position < low - 1e-6 or position > high + 1e-6:
                joint_limit_violations += 1
                joint_limit_violation_counts[name] += 1
            joint_samples += 1
        samples_per_joint += 1

        if 4.5 <= time_s < 5.0:
            for name, target in targets.items():
                hold_errors[name].append(float(data.qpos[qpos_ids[name]] - target))

        if step % render_interval == 0:
            renderer.update_scene(data, camera=camera)
            final_frame = renderer.render().copy()
            writer.append_data(final_frame)
            rows.append(
                {
                    "time_s": time_s,
                    "body_target_scale": body_scale,
                    "hand_target_scale": hand_scale,
                    "mean_body_abs_error_rad": float(
                        np.mean(
                            [abs(data.qpos[qpos_ids[name]] - targets[name]) for name in BODY_TARGETS]
                        )
                    ),
                    "mean_hand_abs_error_rad": float(
                        np.mean(
                            [abs(data.qpos[qpos_ids[name]] - targets[name]) for name in HAND_TARGETS]
                        )
                    ),
                }
            )

    writer.close()
    renderer.close()
    if final_frame is None:
        raise RuntimeError("No video frame was rendered")
    final_image_path = args.output_dir / "final_pose.png"
    Image.fromarray(final_frame).save(final_image_path)

    trajectory_path = args.output_dir / "trajectory_metrics.csv"
    with trajectory_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    body_errors = [error for name in BODY_TARGETS for error in hold_errors[name]]
    hand_errors = [error for name in HAND_TARGETS for error in hold_errors[name]]
    joint_hold_rmse = {
        name: float(np.sqrt(np.mean(np.square(errors)))) for name, errors in hold_errors.items()
    }
    ranked_contact_pairs = []
    for pair, stats in contact_pair_stats.items():
        ranked_contact_pairs.append(
            {
                "geom1": pair[0][0],
                "body1": pair[0][1],
                "geom2": pair[1][0],
                "body2": pair[1][1],
                "contact_samples": stats["contact_samples"],
                "active_step_fraction": len(stats["active_steps"]) / total_steps,
                "mean_normal_force_n": (
                    stats["normal_force_sum_n"] / stats["contact_samples"]
                ),
                "maximum_normal_force_n": stats["maximum_normal_force_n"],
                "maximum_penetration_m": stats["maximum_penetration_m"],
                "first_contact_time_s": stats["first_contact_time_s"],
                "last_contact_time_s": stats["last_contact_time_s"],
            }
        )
    ranked_contact_pairs.sort(
        key=lambda item: (item["active_step_fraction"], item["maximum_normal_force_n"]),
        reverse=True,
    )
    report = {
        "experiment": args.experiment_name,
        "asset": str(args.asset.resolve()),
        "commanded_body_joints": len(BODY_TARGETS),
        "commanded_hand_joints": len(HAND_TARGETS),
        "all_targets_within_joint_limits": all(target_ranges_valid.values()),
        "gain_profile": args.gain_profile,
        "dynamics_profile": args.dynamics_profile,
        "hand_target_scale": args.hand_target_scale,
        "hand_open_margin_rad": args.hand_open_margin,
        "shoulder_lead_seconds": args.shoulder_lead_seconds,
        "gravity_enabled": args.enable_gravity,
        "contacts_enabled": args.enable_contacts,
        "base_mode": args.base_mode,
        "controller": (
            {
                "strong_body_kp_kd": [300.0, 3.0],
                "weak_body_kp_kd": [80.0, 3.0],
                "wrist_kp_kd": [40.0, 1.5],
                "hand_kp_kd": [1.5, 0.2],
            }
            if args.gain_profile == "unitree"
            else {
                "body_kp": args.body_kp,
                "body_kd": args.body_kd,
                "hand_kp": args.hand_kp,
                "hand_kd": args.hand_kd,
            }
        ),
        "body_hold_rmse_rad": float(np.sqrt(np.mean(np.square(body_errors)))),
        "hand_hold_rmse_rad": float(np.sqrt(np.mean(np.square(hand_errors)))),
        "joint_hold_rmse_rad": joint_hold_rmse,
        "mean_contact_count": float(np.mean(contact_counts)),
        "maximum_contact_count": max(contact_counts),
        "actuator_saturation_fraction": saturated_actuators / actuator_samples,
        "actuator_saturation_fraction_by_joint": {
            name: actuator_saturation_counts[name] / samples_per_joint for name in joint_ids
        },
        "joint_limit_violation_fraction": joint_limit_violations / joint_samples,
        "joint_limit_violation_fraction_by_joint": {
            name: joint_limit_violation_counts[name] / samples_per_joint for name in joint_ids
        },
        "contact_pair_count": len(ranked_contact_pairs),
        "contact_pairs_ranked": ranked_contact_pairs,
        "video": str(video_path),
        "final_pose": str(final_image_path),
        "trajectory_metrics": str(trajectory_path),
    }
    report_path = args.output_dir / "actuation_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
