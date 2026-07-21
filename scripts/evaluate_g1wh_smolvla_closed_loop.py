#!/usr/bin/env python3
"""Run step-500 and step-1000 G1 SmolVLA checkpoints in MuJoCo closed loop."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

# Keep MuJoCo and PyTorch from g1ub_genesis, then expose packages that only
# exist in the LeRobot environment, such as transformers and safetensors.
if lerobot_site_packages := os.environ.get("LEROBOT_SITE_PACKAGES"):
    sys.path.append(lerobot_site_packages)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from run_g1_assisted_tote_lift import (
    HOME_POSITIONS_M,
    PALM_NAMES,
    TOTE_SITE_NAMES,
    WAIST_PITCH_LIMIT_RAD,
    build_scene,
    object_name,
)
from validate_g1_bimanual_actuation import apply_regularized_dynamics, unitree_gains


TASK_CAMERAS = {
    "observation.images.head": "head_camera",
    "observation.images.left_wrist": "left_wrist_camera",
    "observation.images.right_wrist": "right_wrist_camera",
}
LOWER_BODY_JOINTS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--checkpoint-500", type=Path, required=True)
    parser.add_argument("--checkpoint-1000", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_assisted_lift_train")
    parser.add_argument("--validation-start", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--assist-distance-m", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2707)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_policy(path: Path, train_meta: LeRobotDatasetMetadata, device: str):
    config = PreTrainedConfig.from_pretrained(path)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig for {path}")
    config.device = device
    config.pretrained_path = path
    policy = make_policy(config, ds_meta=train_meta).eval()
    if policy.config.action_feature.shape != (31,):
        raise ValueError(f"Expected 31 actions, found {policy.config.action_feature.shape}")
    return policy


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


def solve_home_pose(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = []
    joint_ids = []
    for joint_id in range(model.njnt):
        name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name.startswith("waist_") or (
            name.startswith(("left_", "right_"))
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
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in PALM_NAMES
    ]

    def residual(values: np.ndarray) -> np.ndarray:
        data.qpos[qpos_ids] = values
        mujoco.mj_forward(model, data)
        position = np.concatenate(
            [
                data.site_xpos[site_id] - target
                for site_id, target in zip(palm_ids, HOME_POSITIONS_M, strict=True)
            ]
        )
        orientation = np.concatenate(
            [
                Rotation.from_matrix(data.site_xmat[site_id].reshape(3, 3)).as_rotvec()
                for site_id in palm_ids
            ]
        )
        return np.concatenate((position, 0.06 * orientation, 0.003 * values))

    solution = least_squares(
        residual,
        np.zeros(len(names), dtype=np.float64),
        bounds=(lower, upper),
        max_nfev=1200,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    if not solution.success:
        raise RuntimeError("Failed to solve the collision-free home pose")
    return names, qpos_ids, solution.x.copy()


def render_observation(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    upper_state: np.ndarray,
    task: str,
    device: str,
) -> dict:
    batch: dict[str, object] = {
        "observation.state": torch.from_numpy(upper_state.astype(np.float32))
        .unsqueeze(0)
        .to(device),
        "task": [task],
    }
    for key, camera in TASK_CAMERAS.items():
        renderer.update_scene(data, camera=camera)
        image = np.asarray(renderer.render()).copy()
        batch[key] = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device, dtype=torch.float32)
            / 255.0
        )
    return batch


def seeded_noise(policy, seed: int, device: str) -> torch.Tensor:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    return torch.randn(
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        device=device,
    )


def contact_state(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[dict[str, bool], bool]:
    bilateral = {"left": False, "right": False}
    table_contact = False
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
        if "warehouse_tote" not in bodies:
            continue
        bilateral["left"] = any(name.startswith("left_hand_") for name in bodies)
        bilateral["right"] = any(name.startswith("right_hand_") for name in bodies)
        table_contact = "world" in bodies
    return bilateral, table_contact


@torch.no_grad()
def run_episode(
    policy,
    policy_name: str,
    episode: dict,
    args: argparse.Namespace,
    expected_upper_names: list[str],
    device: str,
) -> dict:
    episode_index = int(episode["episode_index"])
    task = episode["language_instruction"]
    output_dir = args.output_dir / policy_name / f"episode_{episode_index:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "scene.xml"
    build_scene(args.asset.resolve(), scene_path, float(episode["tote_x_m"]))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    controlled_names = list(controlled)
    upper_names = controlled_names[LOWER_BODY_JOINTS:]
    if upper_names != expected_upper_names:
        raise ValueError("Simulation and LeRobot upper-body joint order differ")

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    _, selected_qpos_ids, home_pose = solve_home_pose(model, data)
    mujoco.mj_resetData(model, data)
    data.qpos[selected_qpos_ids] = home_pose
    for name, item in controlled.items():
        if "hand_" not in name:
            continue
        low, high = model.jnt_range[item["joint_id"]]
        if abs(float(low)) < 1e-9:
            data.qpos[item["qpos_id"]] = 0.01
        elif abs(float(high)) < 1e-9:
            data.qpos[item["qpos_id"]] = -0.01
    mujoco.mj_forward(model, data)
    initial_targets = np.asarray(
        [data.qpos[controlled[name]["qpos_id"]] for name in controlled_names],
        dtype=np.float64,
    )
    upper_lower = np.asarray(
        [model.jnt_range[controlled[name]["joint_id"], 0] for name in upper_names]
    )
    upper_upper = np.asarray(
        [model.jnt_range[controlled[name]["joint_id"], 1] for name in upper_names]
    )
    for index, name in enumerate(upper_names):
        if name == "waist_pitch_joint":
            upper_lower[index], upper_upper[index] = (
                -WAIST_PITCH_LIMIT_RAD,
                WAIST_PITCH_LIMIT_RAD,
            )
        elif name in ("waist_yaw_joint", "waist_roll_joint"):
            upper_lower[index], upper_upper[index] = -0.05, 0.05

    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in PALM_NAMES
    ]
    tote_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in TOTE_SITE_NAMES
    ]
    assisted_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        for name in ("left_assisted_grasp", "right_assisted_grasp")
    ]
    initial_tote_z = float(data.site_xpos[tote_site_ids[0], 2])

    task_renderer = mujoco.Renderer(model, height=240, width=320)
    video_renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.42, 0.0, 0.95)
    camera.distance = 2.25
    camera.azimuth = 145
    camera.elevation = -8
    video_path = output_dir / "closed_loop.mp4"
    writer = imageio.get_writer(video_path, fps=args.control_fps, codec="libx264", quality=8)

    policy.reset()
    dt = float(model.opt.timestep)
    physics_per_control = max(1, round(1.0 / (args.control_fps * dt)))
    control_frames = round(args.duration_s * args.control_fps)
    current_chunk = None
    action_clip_values = 0
    action_values = 0
    joint_limit_violations = 0
    joint_samples = 0
    saturated_samples = 0
    actuator_samples = 0
    maximum_abs_waist_pitch = 0.0
    bilateral_seen = {"left": False, "right": False}
    assist_activation_s = None
    final_table_contact = False
    states = []
    commands = []
    tote_heights = []
    inference_times = []
    started = time.perf_counter()

    for frame in range(control_frames):
        upper_state = np.asarray(
            [data.qpos[controlled[name]["qpos_id"]] for name in upper_names]
        )
        chunk_index = frame % args.replan_steps
        if chunk_index == 0:
            batch = render_observation(task_renderer, data, upper_state, task, device)
            inference_started = time.perf_counter()
            current_chunk = (
                policy.predict_action_chunk(
                    batch,
                    noise=seeded_noise(
                        policy,
                        args.seed + episode_index * 10_000 + frame,
                        device,
                    ),
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            inference_times.append(time.perf_counter() - inference_started)
        if current_chunk is None or chunk_index >= len(current_chunk):
            raise RuntimeError("Policy did not produce enough actions for replanning")
        raw_action = current_chunk[chunk_index].astype(np.float64)
        action = np.clip(raw_action, upper_lower, upper_upper)
        action_clip_values += int(np.count_nonzero(np.abs(action - raw_action) > 1e-8))
        action_values += action.size
        targets = initial_targets.copy()
        targets[LOWER_BODY_JOINTS:] = action

        for _ in range(physics_per_control):
            if assist_activation_s is None:
                distances = [
                    np.linalg.norm(data.site_xpos[palm] - data.site_xpos[tote])
                    for palm, tote in zip(palm_ids, tote_site_ids, strict=True)
                ]
                if max(distances) <= args.assist_distance_m:
                    for equality_id in assisted_ids:
                        data.eq_active[equality_id] = 1
                    assist_activation_s = float(data.time)
            for target, name in zip(targets, controlled_names, strict=True):
                item = controlled[name]
                kp, kd = unitree_gains(name)
                kp *= 1.5
                kd *= math.sqrt(1.5)
                qpos = float(data.qpos[item["qpos_id"]])
                qvel = float(data.qvel[item["qvel_id"]])
                torque = kp * (target - qpos) - kd * qvel
                torque += float(data.qfrc_bias[item["qvel_id"]])
                data.ctrl[item["actuator_id"]] = torque
            mujoco.mj_step(model, data)

            for name, item in controlled.items():
                joint_id = item["joint_id"]
                position = float(data.qpos[item["qpos_id"]])
                low, high = model.jnt_range[joint_id]
                joint_limit_violations += int(
                    position < low - 1e-6 or position > high + 1e-6
                )
                joint_samples += 1
                force_limit = max(
                    abs(float(value)) for value in model.jnt_actfrcrange[joint_id]
                )
                saturated_samples += int(
                    force_limit > 0
                    and abs(float(data.actuator_force[item["actuator_id"]]))
                    >= 0.98 * force_limit
                )
                actuator_samples += 1
            waist = controlled["waist_pitch_joint"]
            maximum_abs_waist_pitch = max(
                maximum_abs_waist_pitch,
                abs(float(data.qpos[waist["qpos_id"]])),
            )
            bilateral, table_contact = contact_state(model, data)
            for side in bilateral_seen:
                bilateral_seen[side] |= bilateral[side]
            if data.time >= args.duration_s - 0.5:
                final_table_contact |= table_contact

        video_renderer.update_scene(data, camera=camera)
        writer.append_data(video_renderer.render())
        states.append(upper_state.astype(np.float32))
        commands.append(action.astype(np.float32))
        tote_heights.append(float(data.site_xpos[tote_site_ids[0], 2]) - initial_tote_z)

    writer.close()
    task_renderer.close()
    video_renderer.close()
    mujoco.mj_forward(model, data)
    tote_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "warehouse_tote"
    )
    final_tote_speed = float(np.linalg.norm(data.cvel[tote_body_id, 3:]))
    final_lift = float(data.site_xpos[tote_site_ids[0], 2]) - initial_tote_z
    report = {
        "policy": policy_name,
        "checkpoint": str(policy.config.pretrained_path),
        "source_episode": episode_index,
        "task": task,
        "tote_x_m": float(episode["tote_x_m"]),
        "control_fps": args.control_fps,
        "replan_steps": args.replan_steps,
        "action_chunk_size": int(policy.config.chunk_size),
        "assisted_grasp_activation_distance_m": args.assist_distance_m,
        "assisted_grasp_activated": assist_activation_s is not None,
        "assisted_grasp_activation_s": assist_activation_s,
        "tote_lift_height_m": final_lift,
        "maximum_tote_lift_height_m": max(tote_heights),
        "required_lift_height_m": 0.10,
        "final_tote_linear_speed_m_s": final_tote_speed,
        "maximum_abs_waist_pitch_rad": maximum_abs_waist_pitch,
        "bilateral_hand_contact_seen": bilateral_seen,
        "final_table_contact": final_table_contact,
        "action_clip_fraction": action_clip_values / action_values,
        "joint_limit_violation_fraction": joint_limit_violations / joint_samples,
        "actuator_saturation_fraction": saturated_samples / actuator_samples,
        "mean_inference_s": float(np.mean(inference_times)),
        "p95_inference_s": float(np.percentile(inference_times, 95)),
        "wall_time_s": time.perf_counter() - started,
        "video": str(video_path),
    }
    report["passed"] = bool(
        final_lift >= 0.10
        and final_tote_speed < 0.05
        and maximum_abs_waist_pitch <= 0.11
        and all(bilateral_seen.values())
        and not final_table_contact
        and report["joint_limit_violation_fraction"] == 0.0
        and report["actuator_saturation_fraction"] < 0.05
        and report["assisted_grasp_activated"]
    )
    np.savez_compressed(
        output_dir / "trajectory.npz",
        observation_joint_position_rad=np.asarray(states),
        action_joint_position_rad=np.asarray(commands),
        tote_lift_height_m=np.asarray(tote_heights, dtype=np.float32),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{policy_name} episode {episode_index}: passed={report['passed']} "
        f"lift={final_lift:.3f}m max_lift={max(tote_heights):.3f}m",
        flush=True,
    )
    return report


def aggregate(reports: list[dict]) -> dict:
    return {
        "episodes": len(reports),
        "successes": sum(report["passed"] for report in reports),
        "success_rate": sum(report["passed"] for report in reports) / len(reports),
        "mean_final_lift_height_m": float(
            np.mean([report["tote_lift_height_m"] for report in reports])
        ),
        "mean_maximum_lift_height_m": float(
            np.mean([report["maximum_tote_lift_height_m"] for report in reports])
        ),
        "assisted_grasp_activation_rate": sum(
            report["assisted_grasp_activated"] for report in reports
        )
        / len(reports),
        "bilateral_contact_rate": sum(
            all(report["bilateral_hand_contact_seen"].values()) for report in reports
        )
        / len(reports),
        "mean_action_clip_fraction": float(
            np.mean([report["action_clip_fraction"] for report in reports])
        ),
        "mean_inference_s": float(
            np.mean([report["mean_inference_s"] for report in reports])
        ),
    }


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.control_fps < 1 or args.replan_steps < 1:
        raise ValueError("episodes, control-fps, and replan-steps must be positive")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    episodes = source["episodes"][
        args.validation_start : args.validation_start + args.episodes
    ]
    if len(episodes) != args.episodes or not all(item["success"] for item in episodes):
        raise ValueError("Requested validation episodes are missing or failed")
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)
    expected_upper_names = train_meta.features["action"]["names"]
    specs = {
        "step500": args.checkpoint_500,
        "step1000": args.checkpoint_1000,
    }
    all_reports = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for policy_name, checkpoint in specs.items():
        print(f"Loading {policy_name}: {checkpoint}", flush=True)
        policy = load_policy(checkpoint, train_meta, device)
        all_reports[policy_name] = [
            run_episode(
                policy,
                policy_name,
                episode,
                args,
                expected_upper_names,
                device,
            )
            for episode in episodes
        ]
        del policy
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    report = {
        "experiment": "G1WH-27-smolvla-closed-loop-validation",
        "device": device,
        "validation_source_episodes": [item["episode_index"] for item in episodes],
        "control_fps": args.control_fps,
        "replan_steps": args.replan_steps,
        "executed_chunk_duration_s": args.replan_steps / args.control_fps,
        "policies": {
            name: {"aggregate": aggregate(reports), "episodes": reports}
            for name, reports in all_reports.items()
        },
        "reference": {
            "screening_pass": "at least 3/4 successful episodes",
            "project_target": "at least 80/100 successful episodes after scaling evaluation",
        },
    }
    report["screening_passed"] = {
        name: values["aggregate"]["success_rate"] >= 0.75
        for name, values in report["policies"].items()
    }
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
