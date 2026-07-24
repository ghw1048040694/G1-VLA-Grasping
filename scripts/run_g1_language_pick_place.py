#!/usr/bin/env python3
"""Generate one language-conditioned G1 tabletop pick-and-place episode."""

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

OBJECT_NAMES = ("red_triangle", "yellow_rod", "green_cube")
OBJECT_LABELS = {
    "red_triangle": "red triangular prism",
    "yellow_rod": "yellow rod",
    "green_cube": "green cube",
}
INSTRUCTIONS = {
    name: f"put the {label} into the blue box" for name, label in OBJECT_LABELS.items()
}
PALM_NAMES = ("left_palm_center", "right_palm_center")
RIGHT_PALM_INDEX = 1
HOME_POSITIONS_M = np.array(((0.22, 0.32, 1.00), (0.22, -0.32, 1.00)))
WAIST_PITCH_LIMIT_RAD = 0.10
TABLE_TOP_Z_M = 0.75
BIN_POSITION_M = np.array((0.59, -0.23, TABLE_TOP_Z_M + 0.01))
OBJECT_ASSIST_Z_M = {
    "red_triangle": 0.055,
    "yellow_rod": 0.050,
    "green_cube": 0.055,
}
OBJECT_SPAWN_Z_M = {
    "red_triangle": TABLE_TOP_Z_M + 0.030,
    "yellow_rod": TABLE_TOP_Z_M + 0.025,
    "green_cube": TABLE_TOP_Z_M + 0.045,
}


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def object_name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"unnamed_{index}"


def remove_old_task(root: ET.Element) -> None:
    for worldbody in root.findall("worldbody"):
        for body in list(worldbody.findall("body")):
            if body.get("name") in {"warehouse_tote", "blue_receptacle", *OBJECT_NAMES}:
                worldbody.remove(body)
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for constraint in list(equality):
        name = constraint.get("name", "")
        if "assisted_grasp" in name:
            equality.remove(constraint)


def ensure_triangle_mesh(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") == "red_triangle_prism_mesh":
            asset.remove(mesh)
    vertices = (
        (-0.045, -0.040, -0.030),
        (-0.045, 0.040, -0.030),
        (-0.045, 0.000, 0.045),
        (0.045, -0.040, -0.030),
        (0.045, 0.040, -0.030),
        (0.045, 0.000, 0.045),
    )
    faces = (
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4),
        (0, 4, 3),
        (1, 2, 5),
        (1, 5, 4),
        (2, 0, 3),
        (2, 3, 5),
    )
    ET.SubElement(
        asset,
        "mesh",
        name="red_triangle_prism_mesh",
        vertex=" ".join(str(value) for vertex in vertices for value in vertex),
        face=" ".join(str(value) for face in faces for value in face),
    )


def add_free_object(
    scene: ET.Element,
    equality: ET.Element,
    name: str,
    position: np.ndarray,
) -> None:
    assist_z = OBJECT_ASSIST_Z_M[name]
    body = ET.SubElement(
        scene,
        "body",
        name=name,
        pos=" ".join(f"{value:.8f}" for value in position),
    )
    ET.SubElement(body, "freejoint", name=f"{name}_freejoint")
    if name == "red_triangle":
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_visual",
            type="mesh",
            mesh="red_triangle_prism_mesh",
            contype="0",
            conaffinity="0",
            density="0",
            rgba="0.90 0.08 0.06 1",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_collision",
            type="box",
            size="0.045 0.040 0.030",
            mass="0.25",
            friction="0.9 0.02 0.002",
            rgba="0.90 0.08 0.06 0",
        )
    elif name == "yellow_rod":
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_geom",
            type="cylinder",
            fromto="-0.09 0 0 0.09 0 0",
            size="0.025",
            mass="0.20",
            friction="0.9 0.02 0.002",
            rgba="0.95 0.78 0.05 1",
        )
    else:
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_geom",
            type="box",
            size="0.045 0.045 0.045",
            mass="0.25",
            friction="0.9 0.02 0.002",
            rgba="0.08 0.70 0.20 1",
        )
    ET.SubElement(
        body,
        "site",
        name=f"{name}_center",
        type="sphere",
        pos="0 0 0",
        size="0.006",
        rgba="1 1 1 0.15",
    )
    ET.SubElement(
        body,
        "site",
        name=f"{name}_assist_site",
        type="sphere",
        pos=f"0 0 {assist_z}",
        size="0.009",
        rgba="1 1 0 0.35",
    )
    ET.SubElement(
        equality,
        "connect",
        name=f"{name}_assisted_grasp",
        site1="right_palm_center",
        site2=f"{name}_assist_site",
        active="false",
        solref="0.02 1",
    )


def add_blue_box(scene: ET.Element) -> None:
    box = ET.SubElement(
        scene,
        "body",
        name="blue_receptacle",
        pos=" ".join(f"{value:.8f}" for value in BIN_POSITION_M),
    )
    specs = {
        "bottom": ("0 0 0", "0.13 0.15 0.01"),
        "front": ("-0.12 0 0.065", "0.01 0.15 0.065"),
        "back": ("0.12 0 0.065", "0.01 0.15 0.065"),
        "left": ("0 0.14 0.065", "0.13 0.01 0.065"),
        "right": ("0 -0.14 0.065", "0.13 0.01 0.065"),
    }
    for part, (pos, size) in specs.items():
        ET.SubElement(
            box,
            "geom",
            name=f"blue_box_{part}",
            type="box",
            pos=pos,
            size=size,
            friction="0.9 0.02 0.002",
            rgba="0.05 0.25 0.90 1",
        )
    ET.SubElement(
        box,
        "site",
        name="blue_box_drop_site",
        type="sphere",
        pos="0 0 0.19",
        size="0.012",
        rgba="0.10 0.85 1 0.35",
    )


def build_scene(
    source: Path,
    destination: Path,
    positions: dict[str, np.ndarray],
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    remove_old_task(root)
    ensure_triangle_mesh(root)
    scene = root.findall("worldbody")[-1]
    equality = root.find("equality")
    if equality is None:
        raise RuntimeError("Failed to create equality section")
    table = root.find(".//geom[@name='calibration_table']")
    if table is None:
        raise RuntimeError("Source scene has no calibration table")
    table.set("pos", "0.52 0 0.72")
    table.set("size", "0.42 0.55 0.03")
    add_blue_box(scene)
    for name in OBJECT_NAMES:
        add_free_object(scene, equality, name, positions[name])
    tree.write(destination, encoding="unicode", xml_declaration=False)


def controlled_joints(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
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
    return controlled


def arm_ik_contract(
    model: mujoco.MjModel,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = []
    joint_ids = []
    for joint_id in range(model.njnt):
        name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name.startswith("waist_") or (
            (name.startswith("left_") or name.startswith("right_"))
            and any(part in name for part in ("shoulder_", "elbow_", "wrist_"))
        ):
            names.append(name)
            joint_ids.append(joint_id)
    qpos_ids = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in joint_ids])
    upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in joint_ids])
    for index, name in enumerate(names):
        if name == "waist_pitch_joint":
            lower[index], upper[index] = -WAIST_PITCH_LIMIT_RAD, WAIST_PITCH_LIMIT_RAD
        elif name in ("waist_yaw_joint", "waist_roll_joint"):
            lower[index], upper[index] = -0.05, 0.05
    return names, qpos_ids, lower, upper


def solve_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    palm_ids: list[int],
    qpos_ids: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_positions: np.ndarray,
    seed: np.ndarray,
    orientation_weight: float = 0.04,
) -> tuple[np.ndarray, dict]:
    def residual(values: np.ndarray) -> np.ndarray:
        data.qpos[qpos_ids] = values
        mujoco.mj_forward(model, data)
        position = np.concatenate(
            [
                data.site_xpos[site] - target
                for site, target in zip(palm_ids, target_positions)
            ]
        )
        orientation = np.concatenate(
            [
                Rotation.from_matrix(data.site_xmat[site].reshape(3, 3)).as_rotvec()
                for site in palm_ids
            ]
        )
        return np.concatenate(
            (position, orientation_weight * orientation, 0.003 * values)
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
    data.qpos[qpos_ids] = solution.x
    mujoco.mj_forward(model, data)
    errors = [
        float(np.linalg.norm(data.site_xpos[site] - target))
        for site, target in zip(palm_ids, target_positions)
    ]
    return solution.x.copy(), {
        "solver_success": bool(solution.success),
        "maximum_position_error_m": max(errors),
        "cost": float(solution.cost),
    }


def interpolate(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    alpha = smoothstep(alpha)
    return (1.0 - alpha) * start + alpha * end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-object", choices=OBJECT_NAMES, required=True)
    parser.add_argument("--slot-permutation", default="0,1,2")
    parser.add_argument("--slot-offsets", default="0,0,0,0,0,0")
    parser.add_argument("--language-instruction")
    parser.add_argument("--record-demonstration", action="store_true")
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    permutation = tuple(int(value) for value in args.slot_permutation.split(","))
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError("--slot-permutation must contain 0,1,2 exactly once")
    offset_values = tuple(float(value) for value in args.slot_offsets.split(","))
    if len(offset_values) != 6:
        raise ValueError("--slot-offsets must contain six comma-separated x,y values")
    offsets = tuple(
        np.asarray(offset_values[index : index + 2]) for index in range(0, 6, 2)
    )
    slots_xy = (
        np.array((0.36, -0.34)),
        np.array((0.34, -0.10)),
        np.array((0.36, 0.14)),
    )
    positions = {
        name: np.array(
            (*(slots_xy[permutation[index]] + offsets[index]), OBJECT_SPAWN_Z_M[name])
        )
        for index, name in enumerate(OBJECT_NAMES)
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = args.output_dir / "g1_language_pick_place.xml"
    build_scene(args.asset.resolve(), scene_path, positions)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    controlled_names = tuple(controlled)
    selected_names, selected_qpos_ids, lower, upper = arm_ik_contract(model)
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_NAMES
    ]
    object_body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in OBJECT_NAMES
    }
    object_site_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_assist_site")
        for name in OBJECT_NAMES
    }
    equality_ids = {
        name: mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{name}_assisted_grasp"
        )
        for name in OBJECT_NAMES
    }
    bin_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "blue_box_drop_site"
    )

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    home_pose, home_report = solve_pose(
        model,
        data,
        palm_ids,
        selected_qpos_ids,
        lower,
        upper,
        HOME_POSITIONS_M,
        np.zeros(len(selected_names)),
        orientation_weight=0.06,
    )
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    object_assist = data.site_xpos[object_site_ids[args.target_object]].copy()
    pregrasp_targets = HOME_POSITIONS_M.copy()
    pregrasp_targets[RIGHT_PALM_INDEX] = object_assist + np.array((0.0, -0.015, 0.12))
    grasp_targets = HOME_POSITIONS_M.copy()
    grasp_targets[RIGHT_PALM_INDEX] = object_assist
    lift_targets = grasp_targets.copy()
    lift_targets[RIGHT_PALM_INDEX, 2] += 0.18
    place_targets = HOME_POSITIONS_M.copy()
    place_targets[RIGHT_PALM_INDEX] = data.site_xpos[bin_site_id].copy()
    retreat_targets = place_targets.copy()
    retreat_targets[RIGHT_PALM_INDEX, 2] += 0.14

    pregrasp_pose, pregrasp_report = solve_pose(
        model,
        data,
        palm_ids,
        selected_qpos_ids,
        lower,
        upper,
        pregrasp_targets,
        home_pose,
    )
    grasp_pose, grasp_report = solve_pose(
        model,
        data,
        palm_ids,
        selected_qpos_ids,
        lower,
        upper,
        grasp_targets,
        pregrasp_pose,
    )
    lift_pose, lift_report = solve_pose(
        model, data, palm_ids, selected_qpos_ids, lower, upper, lift_targets, grasp_pose
    )
    place_pose, place_report = solve_pose(
        model, data, palm_ids, selected_qpos_ids, lower, upper, place_targets, lift_pose
    )
    retreat_pose, retreat_report = solve_pose(
        model,
        data,
        palm_ids,
        selected_qpos_ids,
        lower,
        upper,
        retreat_targets,
        place_pose,
    )
    pose_reports = {
        "home": home_report,
        "pregrasp": pregrasp_report,
        "grasp": grasp_report,
        "lift": lift_report,
        "place": place_report,
        "retreat": retreat_report,
    }
    if any(not report["solver_success"] for report in pose_reports.values()):
        raise RuntimeError(f"IK failed: {pose_reports}")
    if (
        max(report["maximum_position_error_m"] for report in pose_reports.values())
        > 0.08
    ):
        raise RuntimeError(f"IK position error is too large: {pose_reports}")

    mujoco.mj_resetData(model, data)
    data.qpos[selected_qpos_ids] = home_pose
    open_hand_targets = {}
    for name, item in controlled.items():
        if "hand_" not in name:
            continue
        low, high = model.jnt_range[item["joint_id"]]
        if abs(float(low)) < 1e-9:
            value = 0.01
        elif abs(float(high)) < 1e-9:
            value = -0.01
        else:
            value = float(data.qpos[item["qpos_id"]])
        open_hand_targets[name] = value
        data.qpos[item["qpos_id"]] = value
    mujoco.mj_forward(model, data)
    initial_targets = {
        name: float(data.qpos[item["qpos_id"]]) for name, item in controlled.items()
    }
    initial_object_positions = {
        name: data.xpos[body_id].copy() for name, body_id in object_body_ids.items()
    }
    selected_index = {name: index for index, name in enumerate(selected_names)}

    renderer = mujoco.Renderer(
        model, height=args.render_height, width=args.render_width
    )
    overview_camera = mujoco.MjvCamera()
    overview_camera.lookat[:] = (0.45, -0.05, 0.92)
    overview_camera.distance = 2.15
    overview_camera.azimuth = 145
    overview_camera.elevation = -12
    video_path = args.output_dir / "language_pick_place.mp4"
    writer = imageio.get_writer(
        video_path, fps=args.video_fps, codec="libx264", quality=8
    )
    task_camera_names = ("head_camera", "left_wrist_camera", "right_wrist_camera")
    camera_writers = {}
    if args.record_demonstration:
        for camera_name in task_camera_names:
            camera_writers[camera_name] = imageio.get_writer(
                args.output_dir / f"{camera_name}.mp4",
                fps=args.video_fps,
                codec="libx264",
                quality=8,
            )

    records = defaultdict(list)
    dt = float(model.opt.timestep)
    total_time = 12.0
    render_interval = max(1, round(1.0 / (args.video_fps * dt)))
    total_steps = round(total_time / dt)
    joint_limit_violations = 0
    joint_samples = 0
    wrong_object_max_displacement = 0.0

    for step in range(total_steps):
        time_s = step * dt
        if time_s < 0.5:
            selected_pose, phase = home_pose, 0
        elif time_s < 2.5:
            selected_pose, phase = (
                interpolate(home_pose, pregrasp_pose, (time_s - 0.5) / 2.0),
                1,
            )
        elif time_s < 4.0:
            selected_pose, phase = (
                interpolate(pregrasp_pose, grasp_pose, (time_s - 2.5) / 1.5),
                2,
            )
        elif time_s < 4.8:
            selected_pose, phase = grasp_pose, 3
        elif time_s < 6.8:
            selected_pose, phase = (
                interpolate(grasp_pose, lift_pose, (time_s - 4.8) / 2.0),
                4,
            )
        elif time_s < 9.3:
            selected_pose, phase = (
                interpolate(lift_pose, place_pose, (time_s - 6.8) / 2.5),
                5,
            )
        elif time_s < 9.8:
            selected_pose, phase = place_pose, 6
        elif time_s < 11.3:
            selected_pose, phase = (
                interpolate(place_pose, retreat_pose, (time_s - 9.8) / 1.5),
                7,
            )
        else:
            selected_pose, phase = retreat_pose, 7

        grasp_active = 4.0 <= time_s < 9.35
        for name, equality_id in equality_ids.items():
            data.eq_active[equality_id] = int(
                grasp_active and name == args.target_object
            )
        close_alpha = smoothstep((time_s - 4.0) / 0.6)
        open_alpha = smoothstep((time_s - 9.25) / 0.45)
        hand_alpha = close_alpha * (1.0 - open_alpha)
        commanded = dict(initial_targets)
        for name, index in selected_index.items():
            commanded[name] = float(selected_pose[index])
        for name, closed in HAND_TARGETS.items():
            if name.startswith("right_hand_"):
                commanded[name] = (1.0 - hand_alpha) * open_hand_targets[
                    name
                ] + hand_alpha * 0.4 * closed
            else:
                commanded[name] = open_hand_targets[name]

        for name, item in controlled.items():
            kp, kd = unitree_gains(name)
            kp *= 1.5
            kd *= math.sqrt(1.5)
            qpos = float(data.qpos[item["qpos_id"]])
            qvel = float(data.qvel[item["qvel_id"]])
            torque = kp * (commanded[name] - qpos) - kd * qvel
            torque += float(data.qfrc_bias[item["qvel_id"]])
            data.ctrl[item["actuator_id"]] = torque
        mujoco.mj_step(model, data)

        for name, item in controlled.items():
            low, high = model.jnt_range[item["joint_id"]]
            qpos = float(data.qpos[item["qpos_id"]])
            joint_limit_violations += int(qpos < low - 1e-6 or qpos > high + 1e-6)
            joint_samples += 1
        for name in OBJECT_NAMES:
            if name == args.target_object:
                continue
            displacement = np.linalg.norm(
                data.xpos[object_body_ids[name]] - initial_object_positions[name]
            )
            wrong_object_max_displacement = max(
                wrong_object_max_displacement, float(displacement)
            )

        if step % render_interval == 0:
            renderer.update_scene(data, camera=overview_camera)
            writer.append_data(renderer.render())
            if args.record_demonstration:
                records["time_s"].append(time_s)
                records["joint_position_rad"].append(
                    [
                        float(data.qpos[controlled[name]["qpos_id"]])
                        for name in controlled_names
                    ]
                )
                records["joint_velocity_rad_s"].append(
                    [
                        float(data.qvel[controlled[name]["qvel_id"]])
                        for name in controlled_names
                    ]
                )
                records["action_joint_position_rad"].append(
                    [commanded[name] for name in controlled_names]
                )
                records["task_phase"].append(phase)
                records["assist_active"].append(int(grasp_active))
                records["object_position_m"].append(
                    np.stack(
                        [
                            data.xpos[object_body_ids[name]].copy()
                            for name in OBJECT_NAMES
                        ]
                    )
                )
                records["object_quaternion_wxyz"].append(
                    np.stack(
                        [
                            data.xquat[object_body_ids[name]].copy()
                            for name in OBJECT_NAMES
                        ]
                    )
                )
                records["palm_position_m"].append(
                    np.stack([data.site_xpos[site].copy() for site in palm_ids])
                )
                records["target_object_index"].append(
                    OBJECT_NAMES.index(args.target_object)
                )
                for camera_name, camera_writer in camera_writers.items():
                    renderer.update_scene(data, camera=camera_name)
                    camera_writer.append_data(renderer.render())

    writer.close()
    for camera_writer in camera_writers.values():
        camera_writer.close()
    renderer.close()
    mujoco.mj_forward(model, data)

    final_positions = {
        name: data.xpos[body_id].copy() for name, body_id in object_body_ids.items()
    }
    target_position = final_positions[args.target_object]
    relative_to_bin = target_position - BIN_POSITION_M
    target_in_box = bool(
        abs(relative_to_bin[0]) <= 0.105
        and abs(relative_to_bin[1]) <= 0.125
        and TABLE_TOP_Z_M <= target_position[2] <= TABLE_TOP_Z_M + 0.18
    )
    wrong_objects_in_box = []
    for name in OBJECT_NAMES:
        if name == args.target_object:
            continue
        relative = final_positions[name] - BIN_POSITION_M
        if abs(relative[0]) <= 0.105 and abs(relative[1]) <= 0.125:
            wrong_objects_in_box.append(name)
    violation_fraction = joint_limit_violations / max(joint_samples, 1)
    passed = bool(
        target_in_box
        and not wrong_objects_in_box
        and wrong_object_max_displacement <= 0.05
        and violation_fraction <= 0.01
    )
    instruction = args.language_instruction or INSTRUCTIONS[args.target_object]
    summary = {
        "experiment": "G1-Language-Grounded-Manipulation",
        "task": "language_conditioned_object_to_box",
        "language_instruction": instruction,
        "target_object": args.target_object,
        "object_order": list(OBJECT_NAMES),
        "slot_permutation": list(permutation),
        "slot_offsets_xy_m": [offset.tolist() for offset in offsets],
        "object_initial_position_m": {
            name: value.tolist() for name, value in initial_object_positions.items()
        },
        "object_final_position_m": {
            name: value.tolist() for name, value in final_positions.items()
        },
        "blue_box_position_m": BIN_POSITION_M.tolist(),
        "target_in_box": target_in_box,
        "wrong_objects_in_box": wrong_objects_in_box,
        "wrong_object_max_displacement_m": wrong_object_max_displacement,
        "joint_limit_violation_fraction": violation_fraction,
        "ik": pose_reports,
        "passed": passed,
        "scene": str(scene_path),
        "video": str(video_path),
    }
    summary_path = args.output_dir / "language_pick_place_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.record_demonstration:
        np.savez_compressed(
            args.output_dir / "expert_pick_place_episode.npz",
            joint_names=np.asarray(controlled_names),
            time_s=np.asarray(records["time_s"], dtype=np.float32),
            joint_position_rad=np.asarray(
                records["joint_position_rad"], dtype=np.float32
            ),
            joint_velocity_rad_s=np.asarray(
                records["joint_velocity_rad_s"], dtype=np.float32
            ),
            action_joint_position_rad=np.asarray(
                records["action_joint_position_rad"], dtype=np.float32
            ),
            task_phase=np.asarray(records["task_phase"], dtype=np.int64),
            assist_active=np.asarray(records["assist_active"], dtype=np.int64),
            object_position_m=np.asarray(
                records["object_position_m"], dtype=np.float32
            ),
            object_quaternion_wxyz=np.asarray(
                records["object_quaternion_wxyz"], dtype=np.float32
            ),
            palm_position_m=np.asarray(records["palm_position_m"], dtype=np.float32),
            target_object_index=np.asarray(
                records["target_object_index"], dtype=np.int64
            ),
            metadata_json=np.asarray(json.dumps(summary)),
        )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {summary_path}", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
