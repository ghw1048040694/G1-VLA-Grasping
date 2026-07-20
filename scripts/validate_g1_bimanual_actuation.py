#!/usr/bin/env python3
"""Drive the G1 waist, arms, and hands and record an actuation smoke test."""

from __future__ import annotations

import argparse
import csv
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--video-fps", type=int, default=30)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.asset.resolve()))
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
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

    target_ranges_valid = {}
    for name, target in (BODY_TARGETS | HAND_TARGETS).items():
        joint_id = joint_ids[name]
        low, high = model.jnt_range[joint_id]
        target_ranges_valid[name] = bool(low <= target <= high)

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
    final_frame = None

    for step in range(total_steps):
        time_s = step * dt
        body_scale = smoothstep((time_s - 1.0) / 1.0)
        hand_scale = smoothstep((time_s - 3.0) / 1.0)
        if time_s > 5.0:
            hand_scale = 1.0 - smoothstep((time_s - 5.0) / 1.0)

        targets = {
            **{name: value * body_scale for name, value in BODY_TARGETS.items()},
            **{name: value * hand_scale for name, value in HAND_TARGETS.items()},
        }
        for name, target in targets.items():
            qpos = data.qpos[qpos_ids[name]]
            qvel = data.qvel[qvel_ids[name]]
            is_hand = name in HAND_TARGETS
            kp, kd = (8.0, 0.30) if is_hand else (80.0, 4.0)
            data.ctrl[actuator_ids[name]] = kp * (target - qpos) - kd * qvel

        mujoco.mj_step(model, data)
        data.qpos[:7] = base_qpos
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)

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
    report = {
        "experiment": "G1WH-03-bimanual-actuation-smoke-test",
        "asset": str(args.asset.resolve()),
        "commanded_body_joints": len(BODY_TARGETS),
        "commanded_hand_joints": len(HAND_TARGETS),
        "all_targets_within_joint_limits": all(target_ranges_valid.values()),
        "gravity_disabled": True,
        "contacts_disabled": True,
        "body_hold_rmse_rad": float(np.sqrt(np.mean(np.square(body_errors)))),
        "hand_hold_rmse_rad": float(np.sqrt(np.mean(np.square(hand_errors)))),
        "joint_hold_rmse_rad": joint_hold_rmse,
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
