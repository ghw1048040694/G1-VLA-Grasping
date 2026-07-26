#!/usr/bin/env python3
"""Evaluate a G1 SmolVLA policy on language-grounded object-to-box tasks."""

from __future__ import annotations

import argparse
import itertools
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

if lerobot_site_packages := os.environ.get("LEROBOT_SITE_PACKAGES"):
    sys.path.append(lerobot_site_packages)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from run_g1_language_pick_place import (
    BIN_POSITION_M,
    LEFT_PALM_INDEX,
    HOME_POSITIONS_M,
    OBJECT_NAMES,
    OBJECT_SLOTS_XY_M,
    OBJECT_SPAWN_Z_M,
    PALM_NAMES,
    arm_ik_contract,
    build_scene,
    controlled_joints,
    solve_pose,
)
from collect_g1_language_pick_place_dataset import LANGUAGE_VARIANTS
from validate_g1_bimanual_actuation import apply_regularized_dynamics, unitree_gains

CAMERAS = {
    "observation.images.head": "head_camera",
    "observation.images.left_wrist": "left_wrist_camera",
    "observation.images.right_wrist": "right_wrist_camera",
}
LOWER_BODY_JOINTS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_language_pick_place_train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--assist-distance-m", type=float, default=0.075)
    parser.add_argument("--position-jitter-m", type=float, default=0.018)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_policy(checkpoint: Path, train_meta: LeRobotDatasetMetadata, device: str):
    config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, found {type(config)}")
    config.device = device
    config.pretrained_path = checkpoint
    policy = make_policy(config, ds_meta=train_meta).eval()
    if policy.config.action_feature.shape != (31,):
        raise ValueError(
            f"Expected 31 actions, found {policy.config.action_feature.shape}"
        )
    return policy


def render_observation(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    state: np.ndarray,
    instruction: str,
    device: str,
) -> dict:
    batch: dict[str, object] = {
        "observation.state": torch.from_numpy(state.astype(np.float32))
        .unsqueeze(0)
        .to(device),
        "task": [instruction],
    }
    for key, camera in CAMERAS.items():
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


def evaluation_specs(episodes: int, seed: int, jitter: float) -> list[dict]:
    if episodes < 3 or episodes % 3 != 0:
        raise ValueError("--episodes must be at least three and divisible by three")
    rng = np.random.default_rng(seed)
    permutations = tuple(itertools.permutations(range(3)))
    specs = []
    for scene_index in range(episodes // len(OBJECT_NAMES)):
        permutation = permutations[scene_index % len(permutations)]
        offsets = rng.uniform(-jitter, jitter, size=(3, 2))
        for target in OBJECT_NAMES:
            instruction = LANGUAGE_VARIANTS[target][
                scene_index % len(LANGUAGE_VARIANTS[target])
            ]
            specs.append(
                {
                    "episode_index": len(specs),
                    "scene_index": scene_index,
                    "target_object": target,
                    "instruction": instruction,
                    "permutation": permutation,
                    "offsets": offsets.copy(),
                }
            )
    return specs


def object_inside_box(position: np.ndarray) -> bool:
    relative = position - BIN_POSITION_M
    return bool(
        abs(relative[0]) <= 0.105
        and abs(relative[1]) <= 0.125
        and 0.75 <= position[2] <= 0.93
    )


@torch.no_grad()
def run_episode(
    policy,
    spec: dict,
    args: argparse.Namespace,
    expected_upper_names: list[str],
) -> dict:
    episode_index = spec["episode_index"]
    output_dir = args.output_dir / f"episode_{episode_index:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = {
        name: np.asarray(
            (
                *(
                    OBJECT_SLOTS_XY_M[spec["permutation"][index]]
                    + spec["offsets"][index]
                ),
                OBJECT_SPAWN_Z_M[name],
            )
        )
        for index, name in enumerate(OBJECT_NAMES)
    }
    scene_path = output_dir / "scene.xml"
    build_scene(args.asset.resolve(), scene_path, positions)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    controlled_names = list(controlled)
    upper_names = controlled_names[LOWER_BODY_JOINTS:]
    if upper_names != expected_upper_names:
        raise ValueError("Simulation and dataset joint ordering differ")
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
        (arm_index, name): mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            f"{name}_{'left' if arm_index == LEFT_PALM_INDEX else 'right'}_assisted_grasp",
        )
        for arm_index in range(len(PALM_NAMES))
        for name in OBJECT_NAMES
    }

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
    if not home_report["solver_success"]:
        raise RuntimeError("Failed to solve evaluation home pose")
    mujoco.mj_resetData(model, data)
    data.qpos[selected_qpos_ids] = home_pose
    for name, item in controlled.items():
        if "hand_" not in name:
            continue
        low_limit, high_limit = model.jnt_range[item["joint_id"]]
        if abs(float(low_limit)) < 1e-9:
            data.qpos[item["qpos_id"]] = 0.01
        elif abs(float(high_limit)) < 1e-9:
            data.qpos[item["qpos_id"]] = -0.01
    mujoco.mj_forward(model, data)
    initial_targets = np.asarray(
        [data.qpos[controlled[name]["qpos_id"]] for name in controlled_names],
        dtype=np.float64,
    )
    action_lower = np.asarray(
        [model.jnt_range[controlled[name]["joint_id"], 0] for name in upper_names]
    )
    action_upper = np.asarray(
        [model.jnt_range[controlled[name]["joint_id"], 1] for name in upper_names]
    )

    task_renderer = mujoco.Renderer(model, height=240, width=320)
    video_renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.45, -0.05, 0.92)
    camera.distance = 2.15
    camera.azimuth = 145
    camera.elevation = -12
    video_path = output_dir / "closed_loop.mp4"
    writer = imageio.get_writer(
        video_path, fps=args.control_fps, codec="libx264", quality=8
    )

    policy.reset()
    physics_per_control = max(1, round(1.0 / (args.control_fps * model.opt.timestep)))
    control_frames = round(args.duration_s * args.control_fps)
    current_chunk = None
    grabbed_object = None
    grabbed_arm_index = None
    assist_activation_frame = None
    assist_release_frame = None
    inference_times = []
    action_clip_values = 0
    action_values = 0
    joint_limit_violations = 0
    joint_samples = 0
    joint_limit_violations_by_joint = {name: 0 for name in controlled_names}
    joint_samples_by_joint = {name: 0 for name in controlled_names}
    action_delta_squared_sum = 0.0
    action_delta_count = 0
    action_delta_max_abs = 0.0
    previous_action = initial_targets[LOWER_BODY_JOINTS:].copy()

    for frame in range(control_frames):
        state = np.asarray(
            [data.qpos[controlled[name]["qpos_id"]] for name in upper_names],
            dtype=np.float32,
        )
        chunk_index = frame % args.replan_steps
        if chunk_index == 0:
            batch = render_observation(
                task_renderer, data, state, spec["instruction"], args.device
            )
            started = time.perf_counter()
            current_chunk = (
                policy.predict_action_chunk(
                    batch,
                    noise=seeded_noise(
                        policy,
                        args.seed + spec["scene_index"] * 1000 + frame,
                        args.device,
                    ),
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            inference_times.append(time.perf_counter() - started)
        action = current_chunk[min(chunk_index, len(current_chunk) - 1), :31]
        clipped = np.clip(action, action_lower, action_upper)
        action_clip_values += int(np.count_nonzero(np.abs(clipped - action) > 1e-8))
        action_values += len(action)
        action_delta = clipped - previous_action
        action_delta_squared_sum += float(np.sum(action_delta**2))
        action_delta_count += action_delta.size
        action_delta_max_abs = max(
            action_delta_max_abs, float(np.max(np.abs(action_delta)))
        )
        previous_action = clipped.copy()

        if grabbed_object is None and frame >= args.control_fps:
            distances = {
                (arm_index, name): float(
                    np.linalg.norm(data.site_xpos[palm_id] - data.site_xpos[site_id])
                )
                for arm_index, palm_id in enumerate(palm_ids)
                for name, site_id in object_site_ids.items()
            }
            nearest_arm, nearest_object = min(distances, key=distances.get)
            if distances[(nearest_arm, nearest_object)] <= args.assist_distance_m:
                grabbed_object = nearest_object
                grabbed_arm_index = nearest_arm
                assist_activation_frame = frame
                data.eq_active[equality_ids[(nearest_arm, nearest_object)]] = 1
        if grabbed_object is not None and assist_release_frame is None:
            object_position = data.xpos[object_body_ids[grabbed_object]]
            relative = object_position - BIN_POSITION_M
            if abs(relative[0]) <= 0.105 and abs(relative[1]) <= 0.125:
                data.eq_active[equality_ids[(grabbed_arm_index, grabbed_object)]] = 0
                assist_release_frame = frame

        commanded = initial_targets.copy()
        commanded[LOWER_BODY_JOINTS:] = clipped
        for _ in range(physics_per_control):
            for name, item in controlled.items():
                kp, kd = unitree_gains(name)
                kp *= 1.5
                kd *= math.sqrt(1.5)
                qpos = float(data.qpos[item["qpos_id"]])
                qvel = float(data.qvel[item["qvel_id"]])
                torque = kp * (commanded[item["actuator_id"]] - qpos) - kd * qvel
                torque += float(data.qfrc_bias[item["qvel_id"]])
                data.ctrl[item["actuator_id"]] = torque
            mujoco.mj_step(model, data)
            for name, item in controlled.items():
                low_limit, high_limit = model.jnt_range[item["joint_id"]]
                qpos = float(data.qpos[item["qpos_id"]])
                violated = int(
                    qpos < low_limit - 1e-6 or qpos > high_limit + 1e-6
                )
                joint_limit_violations += violated
                joint_samples += 1
                joint_limit_violations_by_joint[name] += violated
                joint_samples_by_joint[name] += 1
        video_renderer.update_scene(data, camera=camera)
        writer.append_data(video_renderer.render())

    writer.close()
    task_renderer.close()
    video_renderer.close()
    final_positions = {
        name: data.xpos[body_id].copy() for name, body_id in object_body_ids.items()
    }
    objects_in_box = [
        name
        for name, position in final_positions.items()
        if object_inside_box(position)
    ]
    target_in_box = spec["target_object"] in objects_in_box
    wrong_objects_in_box = [
        name for name in objects_in_box if name != spec["target_object"]
    ]
    selected_correct_object = grabbed_object == spec["target_object"]
    violation_fraction = joint_limit_violations / max(joint_samples, 1)
    task_success = bool(
        selected_correct_object
        and target_in_box
        and not wrong_objects_in_box
    )
    passed = bool(task_success and violation_fraction <= 0.01)
    report = {
        "episode_index": episode_index,
        "scene_index": spec["scene_index"],
        "language_instruction": spec["instruction"],
        "target_object": spec["target_object"],
        "grabbed_object": grabbed_object,
        "grabbed_arm": (
            PALM_NAMES[grabbed_arm_index].removesuffix("_palm_center")
            if grabbed_arm_index is not None
            else None
        ),
        "selected_correct_object": selected_correct_object,
        "target_in_box": target_in_box,
        "wrong_objects_in_box": wrong_objects_in_box,
        "objects_in_box": objects_in_box,
        "task_success": task_success,
        "joint_limit_violation_fraction": violation_fraction,
        "joint_limit_violation_fraction_by_joint": {
            name: joint_limit_violations_by_joint[name]
            / joint_samples_by_joint[name]
            for name in controlled_names
            if joint_limit_violations_by_joint[name]
        },
        "action_clip_fraction": action_clip_values / max(action_values, 1),
        "action_delta_rmse_rad": math.sqrt(
            action_delta_squared_sum / max(action_delta_count, 1)
        ),
        "action_delta_max_abs_rad": action_delta_max_abs,
        "assist_activation_frame": assist_activation_frame,
        "assist_release_frame": assist_release_frame,
        "mean_policy_inference_s": float(np.mean(inference_times)),
        "final_object_position_m": {
            name: value.tolist() for name, value in final_positions.items()
        },
        "passed": passed,
        "video": str(video_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"EVAL_EPISODE={episode_index + 1}/{args.episodes} "
        f"target={spec['target_object']} grabbed={grabbed_object} passed={passed}",
        flush=True,
    )
    return report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)
    expected_names = train_meta.features["action"]["names"]
    policy = load_policy(args.checkpoint, train_meta, args.device)
    reports = [
        run_episode(policy, spec, args, expected_names)
        for spec in evaluation_specs(args.episodes, args.seed, args.position_jitter_m)
    ]
    per_target = {}
    for target in OBJECT_NAMES:
        subset = [item for item in reports if item["target_object"] == target]
        per_target[target] = {
            "episodes": len(subset),
            "object_selection_accuracy": sum(
                item["selected_correct_object"] for item in subset
            )
            / len(subset),
            "put_in_box_success_rate": sum(item["task_success"] for item in subset)
            / len(subset),
            "strict_success_rate": sum(item["passed"] for item in subset)
            / len(subset),
        }
    summary = {
        "experiment": "G1-Language-Grounded-Manipulation-Closed-Loop",
        "checkpoint": str(args.checkpoint),
        "episodes": len(reports),
        "object_selection_accuracy": sum(
            item["selected_correct_object"] for item in reports
        )
        / len(reports),
        "put_in_box_success_rate": sum(item["task_success"] for item in reports)
        / len(reports),
        "strict_success_rate": sum(item["passed"] for item in reports)
        / len(reports),
        "wrong_object_grasp_rate": sum(
            item["grabbed_object"] is not None and not item["selected_correct_object"]
            for item in reports
        )
        / len(reports),
        "no_grasp_rate": sum(item["grabbed_object"] is None for item in reports)
        / len(reports),
        "mean_joint_limit_violation_fraction": float(
            np.mean([item["joint_limit_violation_fraction"] for item in reports])
        ),
        "mean_action_delta_rmse_rad": float(
            np.mean([item["action_delta_rmse_rad"] for item in reports])
        ),
        "maximum_action_delta_abs_rad": float(
            np.max([item["action_delta_max_abs_rad"] for item in reports])
        ),
        "per_target": per_target,
        "acceptance_thresholds": {
            "object_selection_accuracy_min": 0.80,
            "put_in_box_success_rate_min": 0.70,
            "strict_success_rate_min": 0.70,
            "wrong_object_grasp_rate_max": 0.10,
            "mean_joint_limit_violation_fraction_max": 0.01,
        },
        "reports": reports,
    }
    summary["passed"] = bool(
        summary["object_selection_accuracy"] >= 0.80
        and summary["put_in_box_success_rate"] >= 0.70
        and summary["strict_success_rate"] >= 0.70
        and summary["wrong_object_grasp_rate"] <= 0.10
        and summary["mean_joint_limit_violation_fraction"] <= 0.01
    )
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "reports"}, indent=2
        )
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
