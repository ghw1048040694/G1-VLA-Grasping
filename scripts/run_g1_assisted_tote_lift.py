#!/usr/bin/env python3
"""Execute a waist-constrained, assisted bimanual tote lift task."""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from validate_g1_bimanual_actuation import (
    HAND_TARGETS,
    apply_regularized_dynamics,
    unitree_gains,
)


PALM_NAMES = ("left_palm_center", "right_palm_center")
TOTE_SITE_NAMES = ("tote_left_assist_site", "tote_right_assist_site")
HOME_POSITIONS_M = np.array(((0.22, 0.32, 1.00), (0.22, -0.32, 1.00)))
PREGRASP_CLEARANCE_M = 0.06
LIFT_HEIGHT_M = 0.18
WAIST_PITCH_LIMIT_RAD = 0.10


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def object_name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"unnamed_{index}"


def build_scene(source: Path, destination: Path, tote_x_m: float) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    tote = root.find(".//body[@name='warehouse_tote']")
    table = root.find(".//geom[@name='calibration_table']")
    equality = root.find("equality")
    if tote is None or table is None or equality is None:
        raise RuntimeError("Source scene is missing tote, table, or equality section")
    tote_pos = [float(value) for value in tote.get("pos", "").split()]
    table_pos = [float(value) for value in table.get("pos", "").split()]
    tote_pos[0] = tote_x_m
    table_pos[0] = tote_x_m + 0.15
    tote.set("pos", " ".join(str(value) for value in tote_pos))
    table.set("pos", " ".join(str(value) for value in table_pos))
    ET.SubElement(
        tote,
        "site",
        name="tote_left_assist_site",
        type="sphere",
        pos="0 0.295 0",
        size="0.008",
        rgba="1 1 0 0.35",
    )
    ET.SubElement(
        tote,
        "site",
        name="tote_right_assist_site",
        type="sphere",
        pos="0 -0.295 0",
        size="0.008",
        rgba="1 1 0 0.35",
    )
    ET.SubElement(
        equality,
        "connect",
        name="left_assisted_grasp",
        site1="left_palm_center",
        site2="tote_left_assist_site",
        active="false",
        solref="0.02 1",
    )
    ET.SubElement(
        equality,
        "connect",
        name="right_assisted_grasp",
        site1="right_palm_center",
        site2="tote_right_assist_site",
        active="false",
        solref="0.02 1",
    )
    tree.write(destination, encoding="unicode", xml_declaration=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--tote-x", type=float, default=0.45)
    parser.add_argument("--record-demonstration", action="store_true")
    parser.add_argument(
        "--language-instruction", default="lift the blue tote with both hands"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = args.output_dir / "g1_assisted_tote_lift.xml"
    build_scene(args.asset.resolve(), scene_path, args.tote_x)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = {}
    for actuator_id in range(model.nu):
        actuator_name = object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, actuator_name)
        if joint_id >= 0:
            controlled[actuator_name] = {
                "actuator_id": actuator_id,
                "joint_id": joint_id,
                "qpos_id": int(model.jnt_qposadr[joint_id]),
                "qvel_id": int(model.jnt_dofadr[joint_id]),
            }

    selected_names = []
    selected_joint_ids = []
    for joint_id in range(model.njnt):
        joint_name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name.startswith("waist_") or (
            (joint_name.startswith("left_") or joint_name.startswith("right_"))
            and any(part in joint_name for part in ("shoulder_", "elbow_", "wrist_"))
        ):
            selected_names.append(joint_name)
            selected_joint_ids.append(joint_id)
    selected_qpos_ids = np.asarray(
        [model.jnt_qposadr[joint_id] for joint_id in selected_joint_ids]
    )
    lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in selected_joint_ids])
    upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in selected_joint_ids])
    for index, joint_name in enumerate(selected_names):
        if joint_name == "waist_pitch_joint":
            lower[index], upper[index] = -WAIST_PITCH_LIMIT_RAD, WAIST_PITCH_LIMIT_RAD
        elif joint_name in ("waist_yaw_joint", "waist_roll_joint"):
            lower[index], upper[index] = -0.05, 0.05
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, item) for item in PALM_NAMES
    ]
    tote_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, item)
        for item in TOTE_SITE_NAMES
    ]

    def solve_pose(
        target_positions: np.ndarray, seed: np.ndarray, orientation_weight: float = 0.06
    ) -> tuple[np.ndarray, dict]:
        def residual(values: np.ndarray) -> np.ndarray:
            data.qpos[selected_qpos_ids] = values
            mujoco.mj_forward(model, data)
            position_error = np.concatenate(
                [
                    data.site_xpos[site_id] - target
                    for site_id, target in zip(palm_ids, target_positions)
                ]
            )
            orientation_error = np.concatenate(
                [
                    Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_rotvec()
                    for site_id in palm_ids
                ]
            )
            return np.concatenate(
                (position_error, orientation_weight * orientation_error, 0.003 * values)
            )

        solution = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=1200,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        data.qpos[selected_qpos_ids] = solution.x
        mujoco.mj_forward(model, data)
        report = {
            "solver_success": bool(solution.success),
            "maximum_position_error_m": float(
                max(
                    np.linalg.norm(data.site_xpos[site_id] - target)
                    for site_id, target in zip(palm_ids, target_positions)
                )
            ),
            "maximum_orientation_error_rad": float(
                max(
                    np.linalg.norm(
                        Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_rotvec()
                    )
                    for site_id in palm_ids
                )
            ),
            "waist_pitch_rad": float(solution.x[selected_names.index("waist_pitch_joint")]),
        }
        return solution.x.copy(), report

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    zero_seed = np.zeros(len(selected_names), dtype=np.float64)
    home_pose, home_report = solve_pose(HOME_POSITIONS_M, zero_seed)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    tote_positions = np.stack([data.site_xpos[item].copy() for item in tote_site_ids])
    pregrasp_positions = tote_positions.copy()
    pregrasp_positions[0, 1] += PREGRASP_CLEARANCE_M
    pregrasp_positions[1, 1] -= PREGRASP_CLEARANCE_M
    pregrasp_pose, pregrasp_report = solve_pose(pregrasp_positions, home_pose)
    grasp_pose, grasp_report = solve_pose(tote_positions, pregrasp_pose, orientation_weight=0.04)
    lift_positions = tote_positions.copy()
    lift_positions[:, 2] += LIFT_HEIGHT_M
    lift_pose, lift_report = solve_pose(lift_positions, grasp_pose, orientation_weight=0.04)
    pose_reports = {
        "home": home_report,
        "pregrasp": pregrasp_report,
        "grasp": grasp_report,
        "lift": lift_report,
    }
    if any(not item["solver_success"] for item in pose_reports.values()):
        raise RuntimeError("At least one task pose failed IK")

    mujoco.mj_resetData(model, data)
    data.qpos[selected_qpos_ids] = home_pose
    open_hand_targets = {}
    for joint_name, item in controlled.items():
        if "hand_" not in joint_name:
            continue
        low, high = model.jnt_range[item["joint_id"]]
        if abs(float(low)) < 1e-9:
            open_hand_targets[joint_name] = 0.01
        elif abs(float(high)) < 1e-9:
            open_hand_targets[joint_name] = -0.01
        else:
            open_hand_targets[joint_name] = float(data.qpos[item["qpos_id"]])
        data.qpos[item["qpos_id"]] = open_hand_targets[joint_name]
    mujoco.mj_forward(model, data)
    initial_tote_z = float(data.site_xpos[tote_site_ids[0], 2])
    initial_targets = {
        joint_name: float(data.qpos[item["qpos_id"]])
        for joint_name, item in controlled.items()
    }
    selected_index = {joint_name: index for index, joint_name in enumerate(selected_names)}
    assisted_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, item)
        for item in ("left_assisted_grasp", "right_assisted_grasp")
    ]

    renderer = mujoco.Renderer(
        model, height=args.render_height, width=args.render_width
    )
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.42, 0.0, 0.95)
    camera.distance = 2.25
    camera.azimuth = 145
    camera.elevation = -8
    video_path = args.output_dir / "assisted_tote_lift.mp4"
    writer = imageio.get_writer(video_path, fps=args.video_fps, codec="libx264", quality=8)
    task_camera_names = ("head_camera", "left_wrist_camera", "right_wrist_camera")
    task_camera_writers = {}
    if args.record_demonstration:
        for camera_name in task_camera_names:
            task_camera_writers[camera_name] = imageio.get_writer(
                args.output_dir / f"{camera_name}.mp4",
                fps=args.video_fps,
                codec="libx264",
                quality=8,
            )
    controlled_names = tuple(controlled)
    demonstration_times = []
    demonstration_qpos = []
    demonstration_qvel = []
    demonstration_actions = []
    demonstration_phases = []
    demonstration_assist_active = []

    dt = float(model.opt.timestep)
    total_time = 10.0
    total_steps = int(total_time / dt)
    render_interval = max(1, round(1.0 / (args.video_fps * dt)))
    joint_limit_violations = 0
    joint_samples = 0
    joint_limit_violation_counts = defaultdict(int)
    saturated_samples = 0
    actuator_samples = 0
    actuator_saturation_counts = defaultdict(int)
    maximum_abs_waist_pitch = 0.0
    contact_stats = defaultdict(int)
    bilateral_contact_seen = {"left": False, "right": False}
    final_table_contact = False

    for step in range(total_steps):
        time_s = step * dt
        if time_s < 0.5:
            selected_target = home_pose
            task_phase = 0
        elif time_s < 2.5:
            alpha = smoothstep((time_s - 0.5) / 2.0)
            selected_target = (1.0 - alpha) * home_pose + alpha * pregrasp_pose
            task_phase = 1
        elif time_s < 4.0:
            alpha = smoothstep((time_s - 2.5) / 1.5)
            selected_target = (1.0 - alpha) * pregrasp_pose + alpha * grasp_pose
            task_phase = 2
        elif time_s < 4.8:
            selected_target = grasp_pose
            task_phase = 3
        elif time_s < 7.8:
            alpha = smoothstep((time_s - 4.8) / 3.0)
            selected_target = (1.0 - alpha) * grasp_pose + alpha * lift_pose
            task_phase = 4
        else:
            selected_target = lift_pose
            task_phase = 5

        if time_s >= 4.0:
            for equality_id in assisted_ids:
                data.eq_active[equality_id] = 1
        hand_alpha = smoothstep((time_s - 4.0) / 0.8)
        commanded_targets = dict(initial_targets)
        for joint_name, index in selected_index.items():
            commanded_targets[joint_name] = float(selected_target[index])
        for joint_name, closed_target in HAND_TARGETS.items():
            commanded_targets[joint_name] = (
                (1.0 - hand_alpha) * open_hand_targets[joint_name]
                + hand_alpha * 0.4 * closed_target
            )

        for joint_name, item in controlled.items():
            kp, kd = unitree_gains(joint_name)
            kp *= 1.5
            kd *= math.sqrt(1.5)
            qpos = float(data.qpos[item["qpos_id"]])
            qvel = float(data.qvel[item["qvel_id"]])
            torque = kp * (commanded_targets[joint_name] - qpos) - kd * qvel
            torque += float(data.qfrc_bias[item["qvel_id"]])
            data.ctrl[item["actuator_id"]] = torque
        mujoco.mj_step(model, data)

        waist_item = controlled["waist_pitch_joint"]
        maximum_abs_waist_pitch = max(
            maximum_abs_waist_pitch, abs(float(data.qpos[waist_item["qpos_id"]]))
        )
        for joint_name, item in controlled.items():
            joint_id = item["joint_id"]
            position = float(data.qpos[item["qpos_id"]])
            low, high = model.jnt_range[joint_id]
            violated = position < low - 1e-6 or position > high + 1e-6
            joint_limit_violations += int(violated)
            joint_limit_violation_counts[joint_name] += int(violated)
            joint_samples += 1
            force_limit = max(abs(float(value)) for value in model.jnt_actfrcrange[joint_id])
            saturated = (
                force_limit > 0
                and abs(float(data.actuator_force[item["actuator_id"]])) >= 0.98 * force_limit
            )
            saturated_samples += int(saturated)
            actuator_saturation_counts[joint_name] += int(saturated)
            actuator_samples += 1

        table_contact_now = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            bodies = {
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
            if "warehouse_tote" in bodies:
                if any(item.startswith("left_hand_") for item in bodies):
                    bilateral_contact_seen["left"] = True
                if any(item.startswith("right_hand_") for item in bodies):
                    bilateral_contact_seen["right"] = True
                if "world" in bodies:
                    table_contact_now = True
            contact_stats["|".join(sorted(bodies))] += 1
        if time_s >= 9.5:
            final_table_contact = final_table_contact or table_contact_now

        if step % render_interval == 0:
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
            if args.record_demonstration:
                demonstration_times.append(time_s)
                demonstration_qpos.append(
                    [float(data.qpos[controlled[name]["qpos_id"]]) for name in controlled_names]
                )
                demonstration_qvel.append(
                    [float(data.qvel[controlled[name]["qvel_id"]]) for name in controlled_names]
                )
                demonstration_actions.append(
                    [float(commanded_targets[name]) for name in controlled_names]
                )
                demonstration_phases.append(task_phase)
                demonstration_assist_active.append(int(time_s >= 4.0))
                for camera_name, camera_writer in task_camera_writers.items():
                    renderer.update_scene(data, camera=camera_name)
                    camera_writer.append_data(renderer.render())

    writer.close()
    for camera_writer in task_camera_writers.values():
        camera_writer.close()
    renderer.close()
    mujoco.mj_forward(model, data)
    final_tote_z = float(data.site_xpos[tote_site_ids[0], 2])
    tote_lift_height = final_tote_z - initial_tote_z
    tote_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "warehouse_tote")
    final_tote_speed = float(np.linalg.norm(data.cvel[tote_body_id, 3:]))
    report = {
        "experiment": "G1WH-22-assisted-bimanual-tote-lift",
        "task_semantics": "approach, bilateral contact, close hands, lift tote, hold",
        "assisted_grasp_constraint": True,
        "tote_x_m": args.tote_x,
        "waist_pitch_limit_rad": WAIST_PITCH_LIMIT_RAD,
        "pose_reports": pose_reports,
        "tote_lift_height_m": tote_lift_height,
        "required_lift_height_m": 0.10,
        "final_tote_linear_speed_m_s": final_tote_speed,
        "maximum_abs_waist_pitch_rad": maximum_abs_waist_pitch,
        "bilateral_hand_contact_seen": bilateral_contact_seen,
        "final_table_contact": final_table_contact,
        "joint_limit_violation_fraction": joint_limit_violations / joint_samples,
        "joint_limit_violation_counts": {
            name: count
            for name, count in sorted(
                joint_limit_violation_counts.items(), key=lambda item: item[1], reverse=True
            )
            if count
        },
        "actuator_saturation_fraction": saturated_samples / actuator_samples,
        "actuator_saturation_counts": {
            name: count
            for name, count in sorted(
                actuator_saturation_counts.items(), key=lambda item: item[1], reverse=True
            )
            if count
        },
        "assisted_constraints_active": bool(all(data.eq_active[item] for item in assisted_ids)),
        "contact_sample_counts": dict(contact_stats),
        "video": str(video_path),
    }
    report["passed"] = bool(
        tote_lift_height >= 0.10
        and final_tote_speed < 0.05
        and maximum_abs_waist_pitch <= 0.11
        and all(bilateral_contact_seen.values())
        and not final_table_contact
        and report["joint_limit_violation_fraction"] == 0.0
        and report["actuator_saturation_fraction"] < 0.05
        and report["assisted_constraints_active"]
    )
    if args.record_demonstration:
        dataset_path = args.output_dir / "expert_lift_episode.npz"
        np.savez_compressed(
            dataset_path,
            timestamp_s=np.asarray(demonstration_times, dtype=np.float32),
            joint_names=np.asarray(controlled_names),
            observation_joint_position_rad=np.asarray(demonstration_qpos, dtype=np.float32),
            observation_joint_velocity_rad_s=np.asarray(demonstration_qvel, dtype=np.float32),
            action_joint_position_rad=np.asarray(demonstration_actions, dtype=np.float32),
            task_phase=np.asarray(demonstration_phases, dtype=np.int64),
            assisted_grasp_active=np.asarray(demonstration_assist_active, dtype=np.int8),
        )
        metadata = {
            "task": "assisted_bimanual_tote_lift",
            "language_instruction": args.language_instruction,
            "episode_success": report["passed"],
            "assisted_grasp_constraint": True,
            "frames": len(demonstration_times),
            "fps": args.video_fps,
            "image_width": args.render_width,
            "image_height": args.render_height,
            "joint_count": len(controlled_names),
            "task_cameras": list(task_camera_names),
            "dataset": str(dataset_path),
        }
        (args.output_dir / "episode_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    report_path = args.output_dir / "assisted_tote_lift_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
