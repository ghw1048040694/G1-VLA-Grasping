#!/usr/bin/env python3
"""Compare one or more G1 SmolVLA checkpoints in MuJoCo closed loop."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
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
import train_g1_world_model as world_model_utils
from train_g1_hybrid_world_model import build_hybrid_from_checkpoint


TASK_CAMERAS = {
    "observation.images.head": "head_camera",
    "observation.images.left_wrist": "left_wrist_camera",
    "observation.images.right_wrist": "right_wrist_camera",
}
PHASE_INSTRUCTIONS = {
    0: "keep both hands at the ready pose",
    1: "move both hands toward the blue tote",
    2: "align both hands with the sides of the blue tote",
    3: "close both hands around the blue tote",
    4: "lift the blue tote upward with both hands",
    5: "hold the blue tote steady in the air",
}
LOWER_BODY_JOINTS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--checkpoint-500", type=Path)
    parser.add_argument("--checkpoint-1000", type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Named checkpoint to evaluate; repeat for three or more policies.",
    )
    parser.add_argument(
        "--hybrid",
        action="append",
        default=[],
        metavar="NAME=BASE,HOLD",
        help="Policy using BASE before phase 5 and HOLD during phase 5.",
    )
    parser.add_argument(
        "--skip-standalone",
        action="store_true",
        help="Evaluate only requested hybrids, not their component policies.",
    )
    parser.add_argument(
        "--policy-selection",
        choices=("both", "step500", "step1000"),
        default="both",
    )
    parser.add_argument(
        "--experiment-id", default="G1WH-27-smolvla-closed-loop-validation"
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_assisted_lift_train")
    parser.add_argument("--validation-start", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--assist-distance-m", type=float, default=0.08)
    parser.add_argument(
        "--assist-solref-timeconst",
        type=float,
        default=0.02,
        help="MuJoCo equality-constraint time constant for assisted grasping.",
    )
    parser.add_argument("--align-distance-m", type=float, default=0.12)
    parser.add_argument("--phase-language-scheduler", action="store_true")
    parser.add_argument(
        "--hold-action-blend-alpha",
        type=float,
        default=1.0,
        help=(
            "Blend factor for new policy actions during phase 5. "
            "1.0 disables smoothing; smaller values retain more of the previous command."
        ),
    )
    parser.add_argument(
        "--hold-gain-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the 1.5x PD gain boost during phase 5.",
    )
    parser.add_argument(
        "--terminal-hold-controller",
        action="store_true",
        help=(
            "Freeze the last executed joint target when phase 5 starts instead "
            "of continuing policy inference."
        ),
    )
    parser.add_argument("--seed", type=int, default=2707)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument(
        "--world-model-ensemble-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Additional hybrid world-model checkpoint; repeat to form an ensemble.",
    )
    parser.add_argument(
        "--world-model-uncertainty-threshold",
        type=float,
        help="Reject candidates whose normalized ensemble disagreement exceeds this value.",
    )
    parser.add_argument(
        "--world-model-uncertainty-keep-fraction",
        type=float,
        help="Keep only this lowest-disagreement fraction of candidates before scoring.",
    )
    parser.add_argument("--world-model-candidates", type=int, default=4)
    parser.add_argument("--world-model-horizon", type=int, default=5)
    parser.add_argument(
        "--world-model-batched-candidates",
        action="store_true",
        help="Generate all stochastic VLA candidates in one batched forward pass.",
    )
    parser.add_argument(
        "--planner-body-safety-margin-fraction",
        type=float,
        default=0.0,
        help="Fraction of each non-hand joint range reserved as a planner safety margin.",
    )
    parser.add_argument(
        "--planner-hand-safety-margin-fraction",
        type=float,
        default=0.0,
        help="Fraction of each hand joint range reserved as a planner safety margin.",
    )
    parser.add_argument(
        "--compare-world-model-planner",
        action="store_true",
        help="Evaluate both VLA-only and VLA plus world-model candidate ranking.",
    )
    parser.add_argument(
        "--compare-unfiltered-ensemble-planner",
        action="store_true",
        help="Also evaluate the same ensemble without uncertainty rejection.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def checkpoint_specs(args: argparse.Namespace) -> dict[str, Path]:
    if args.checkpoint:
        if args.checkpoint_500 or args.checkpoint_1000:
            raise ValueError(
                "Use either repeated --checkpoint NAME=PATH or the legacy "
                "--checkpoint-500/--checkpoint-1000 pair"
            )
        if args.policy_selection != "both":
            raise ValueError("--policy-selection only applies to legacy checkpoints")
        specs = {}
        for value in args.checkpoint:
            if "=" not in value:
                raise ValueError(f"Checkpoint must use NAME=PATH syntax: {value}")
            name, raw_path = value.split("=", 1)
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
                raise ValueError(f"Invalid checkpoint name: {name}")
            if name in specs:
                raise ValueError(f"Duplicate checkpoint name: {name}")
            specs[name] = Path(raw_path)
        return specs

    if args.checkpoint_500 is None or args.checkpoint_1000 is None:
        raise ValueError(
            "Provide repeated --checkpoint NAME=PATH values or both legacy checkpoints"
        )
    specs = {
        "step500": args.checkpoint_500,
        "step1000": args.checkpoint_1000,
    }
    if args.policy_selection != "both":
        specs = {args.policy_selection: specs[args.policy_selection]}
    return specs


def hybrid_specs(
    values: list[str], checkpoints: dict[str, Path]
) -> dict[str, tuple[str, str]]:
    specs = {}
    for value in values:
        if "=" not in value or "," not in value:
            raise ValueError(f"Hybrid must use NAME=BASE,HOLD syntax: {value}")
        name, policy_names = value.split("=", 1)
        base_name, hold_name = policy_names.split(",", 1)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"Invalid hybrid name: {name}")
        if name in checkpoints or name in specs:
            raise ValueError(f"Duplicate policy name: {name}")
        missing = [item for item in (base_name, hold_name) if item not in checkpoints]
        if missing:
            raise ValueError(f"Hybrid {name} references unknown policies: {missing}")
        specs[name] = (base_name, hold_name)
    return specs


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


def seeded_noise(
    policy, seed: int, device: str, batch_size: int = 1
) -> torch.Tensor:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    return torch.randn(
        batch_size,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        device=device,
    )


def repeat_policy_batch(batch: dict, count: int) -> dict:
    repeated = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            repeats = (count,) + (1,) * (value.ndim - 1)
            repeated[key] = value.repeat(repeats)
        elif isinstance(value, list):
            repeated[key] = value * count
        else:
            raise TypeError(f"Unsupported policy batch value for {key}: {type(value)}")
    return repeated


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
        bilateral["left"] |= any(name.startswith("left_hand_") for name in bodies)
        bilateral["right"] |= any(name.startswith("right_hand_") for name in bodies)
        table_contact |= "world" in bodies
    return bilateral, table_contact


def world_model_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    upper_names: list[str],
    tote_body_id: int,
    tote_site_id: int,
    palm_ids: list[int],
    initial_tote_z: float,
    phase: int,
) -> np.ndarray:
    bilateral, table_contact = contact_state(model, data)
    hands = np.asarray(
        (float(bilateral["left"]), float(bilateral["right"])), dtype=np.float32
    )
    lift = float(data.site_xpos[tote_site_id, 2] - initial_tote_z)
    progress = (
        float(phase) / 5.0
        + float(hands.mean())
        + float(np.clip(lift / 0.10, 0.0, 1.0))
    ) / 3.0
    state = np.concatenate(
        (
            [data.qpos[controlled[name]["qpos_id"]] for name in upper_names],
            [data.qvel[controlled[name]["qvel_id"]] for name in upper_names],
            data.xpos[tote_body_id],
            data.xquat[tote_body_id],
            data.cvel[tote_body_id, 3:],
            data.cvel[tote_body_id, :3],
            np.stack([data.site_xpos[item] for item in palm_ids]).reshape(-1),
            hands,
            (float(table_contact), lift, progress),
        )
    ).astype(np.float32)
    if state.shape != (world_model_utils.LAYOUT.state_dim,):
        raise RuntimeError(f"Unexpected world-model state shape: {state.shape}")
    return state


@torch.no_grad()
def select_world_model_candidate(
    policy,
    batch: dict,
    planner: dict,
    current_state: np.ndarray,
    previous_action: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    base_seed: int,
    device: str,
) -> tuple[np.ndarray, dict]:
    candidates = []
    raw_candidates = []
    inference_started = time.perf_counter()
    candidate_seeds = [
        base_seed + candidate_index * 1_000_003
        for candidate_index in range(planner["candidate_count"])
    ]
    if planner["batched_candidates"]:
        noises = torch.cat(
            [seeded_noise(policy, seed, device) for seed in candidate_seeds], dim=0
        )
        raw_batch = (
            policy.predict_action_chunk(
                repeat_policy_batch(batch, planner["candidate_count"]), noise=noises
            )
            .detach()
            .cpu()
            .numpy()
        )
        raw_candidates.extend(raw_batch)
    else:
        for seed in candidate_seeds:
            raw = (
                policy.predict_action_chunk(
                    batch, noise=seeded_noise(policy, seed, device)
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            raw_candidates.append(raw)
    candidates.extend(np.clip(raw, lower, upper) for raw in raw_candidates)
    policy_inference_s = time.perf_counter() - inference_started
    horizon = min(planner["horizon"], min(len(item) for item in candidates))
    candidate_actions = np.stack([item[:horizon] for item in candidates]).astype(np.float32)
    prediction_started = time.perf_counter()
    member_predictions = []
    for model, stats in zip(
        planner["models"], planner["member_stats"], strict=True
    ):
        initial = (current_state - stats["state_mean"]) / stats["state_std"]
        normalized_actions = (
            candidate_actions - stats["action_mean"][None, None, :]
        ) / stats["action_std"][None, None, :]
        initial_batch = np.repeat(
            initial[None, :], len(candidates), axis=0
        ).astype(np.float32)
        normalized_prediction = world_model_utils.rollout(
            model,
            torch.from_numpy(initial_batch).to(device),
            torch.from_numpy(normalized_actions).to(device),
        ).cpu().numpy()
        member_predictions.append(
            normalized_prediction * stats["state_std"][None, None, :]
            + stats["state_mean"][None, None, :]
        )
    world_model_inference_s = time.perf_counter() - prediction_started
    members = np.stack(member_predictions, axis=0)
    predicted = np.mean(members, axis=0)
    endpoint = predicted[:, -1]
    layout = world_model_utils.LAYOUT
    object_low = layout.tote_position[0]
    object_high = layout.task_progress[1]
    reference_stats = planner["member_stats"][0]
    normalized_member_endpoints = (
        members[:, :, -1, object_low:object_high]
        - reference_stats["state_mean"][None, None, object_low:object_high]
    ) / reference_stats["state_std"][None, None, object_low:object_high]
    candidate_uncertainty = np.sqrt(
        np.mean(np.var(normalized_member_endpoints, axis=0), axis=1)
    )
    q_low, q_high = layout.joint_position
    contact_low, contact_high = layout.bilateral_contact
    table_index = layout.table_contact[0]
    lift_index = layout.lift_height[0]
    progress_index = layout.task_progress[0]
    linear_low, linear_high = layout.tote_linear_velocity
    angular_low, angular_high = layout.tote_angular_velocity
    quaternion_low, quaternion_high = layout.tote_quaternion

    scores = []
    components = []
    for index in range(len(candidates)):
        raw = raw_candidates[index][:horizon]
        clipped = candidate_actions[index]
        clip_rmse = float(np.sqrt(np.mean((raw - clipped) ** 2)))
        action_sequence = np.concatenate((previous_action[None, :], clipped), axis=0)
        smoothness = float(np.sqrt(np.mean(np.diff(action_sequence, axis=0) ** 2)))
        predicted_qpos = predicted[index, :, q_low:q_high]
        lower_violation = np.maximum(lower[None, :] - predicted_qpos, 0.0)
        upper_violation = np.maximum(predicted_qpos - upper[None, :], 0.0)
        predicted_violation = float(
            np.sqrt(np.mean((lower_violation + upper_violation) ** 2))
        )
        quaternion = endpoint[index, quaternion_low:quaternion_high]
        quaternion /= max(float(np.linalg.norm(quaternion)), 1e-8)
        upright_error = float(2.0 * np.arccos(np.clip(abs(quaternion[0]), 0.0, 1.0)))
        lift_score = float(np.clip(endpoint[index, lift_index] / 0.10, -0.5, 1.5))
        contact_score = float(np.mean(endpoint[index, contact_low:contact_high]))
        progress_score = float(endpoint[index, progress_index])
        table_contact = float(endpoint[index, table_index])
        linear_speed = float(np.linalg.norm(endpoint[index, linear_low:linear_high]))
        angular_speed = float(np.linalg.norm(endpoint[index, angular_low:angular_high]))
        score = (
            8.0 * progress_score
            + 6.0 * lift_score
            + 2.0 * contact_score
            - 6.0 * table_contact
            - 1.5 * linear_speed
            - 0.3 * angular_speed
            - 0.5 * upright_error
            - 40.0 * clip_rmse
            - 25.0 * predicted_violation
            - 0.5 * smoothness
        )
        scores.append(score)
        components.append(
            {
                "score": score,
                "predicted_progress": progress_score,
                "predicted_lift_m": float(endpoint[index, lift_index]),
                "predicted_contact": contact_score,
                "predicted_table_contact": table_contact,
                "predicted_linear_speed_m_s": linear_speed,
                "predicted_upright_error_rad": upright_error,
                "action_clip_rmse_rad": clip_rmse,
                "predicted_joint_violation_rmse_rad": predicted_violation,
                "action_smoothness_rmse_rad": smoothness,
                "normalized_ensemble_disagreement": float(
                    candidate_uncertainty[index]
                ),
            }
        )
    threshold = planner["uncertainty_threshold"]
    eligible = np.ones(len(scores), dtype=bool)
    keep_fraction = planner["uncertainty_keep_fraction"]
    if keep_fraction is not None and len(planner["models"]) > 1:
        keep_count = max(1, math.ceil(len(scores) * keep_fraction))
        eligible[:] = False
        eligible[np.argsort(candidate_uncertainty)[:keep_count]] = True
    elif threshold is not None and len(planner["models"]) > 1:
        eligible = candidate_uncertainty <= threshold
    all_candidates_uncertain = not bool(np.any(eligible))
    if all_candidates_uncertain:
        selected = 0
    else:
        eligible_scores = np.where(eligible, np.asarray(scores), -np.inf)
        selected = int(np.argmax(eligible_scores))
    sorted_scores = np.sort(np.asarray(scores))
    score_margin = float(sorted_scores[-1] - sorted_scores[-2]) if len(scores) > 1 else 0.0
    return candidates[selected], {
        "selected_index": selected,
        "selected_nonbaseline": selected != 0,
        "score_margin": score_margin,
        "policy_inference_s": policy_inference_s,
        "batched_candidates": planner["batched_candidates"],
        "world_model_inference_s": world_model_inference_s,
        "selected_uncertainty": float(candidate_uncertainty[selected]),
        "uncertainty_rejected_fraction": float(np.mean(~eligible)),
        "all_candidates_uncertain": all_candidates_uncertain,
        "candidate_scores": components,
    }


def stability_metrics(
    states: np.ndarray,
    commands: np.ndarray,
    tote_heights: np.ndarray,
    phases: np.ndarray,
    control_fps: int,
) -> dict[str, float | None]:
    hold_indices = np.flatnonzero(phases == 5)
    result = {
        "hold_action_delta_rmse_rad": None,
        "hold_action_delta_p95_step_norm_rad": None,
        "hold_action_delta_max_abs_rad": None,
        "hold_joint_velocity_rmse_rad_s": None,
        "hold_joint_acceleration_rmse_rad_s2": None,
        "hold_switch_action_delta_rmse_rad": None,
        "hold_switch_action_delta_max_abs_rad": None,
        "hold_final_2s_tote_height_std_m": None,
        "hold_final_2s_tote_height_range_m": None,
    }
    if not len(hold_indices):
        return result

    first_hold = int(hold_indices[0])
    if first_hold > 0:
        switch_delta = commands[first_hold] - commands[first_hold - 1]
        result["hold_switch_action_delta_rmse_rad"] = float(
            np.sqrt(np.mean(switch_delta**2))
        )
        result["hold_switch_action_delta_max_abs_rad"] = float(
            np.max(np.abs(switch_delta))
        )

    hold_pairs = (phases[1:] == 5) & (phases[:-1] == 5)
    if np.any(hold_pairs):
        action_delta = np.diff(commands, axis=0)[hold_pairs]
        joint_velocity = np.diff(states, axis=0)[hold_pairs] * control_fps
        result["hold_action_delta_rmse_rad"] = float(
            np.sqrt(np.mean(action_delta**2))
        )
        result["hold_action_delta_p95_step_norm_rad"] = float(
            np.percentile(np.linalg.norm(action_delta, axis=1), 95)
        )
        result["hold_action_delta_max_abs_rad"] = float(
            np.max(np.abs(action_delta))
        )
        result["hold_joint_velocity_rmse_rad_s"] = float(
            np.sqrt(np.mean(joint_velocity**2))
        )
        if len(joint_velocity) > 1:
            acceleration = np.diff(joint_velocity, axis=0) * control_fps
            result["hold_joint_acceleration_rmse_rad_s2"] = float(
                np.sqrt(np.mean(acceleration**2))
            )

    final_window = min(2 * control_fps, len(hold_indices))
    final_hold = tote_heights[hold_indices[-final_window:]]
    result["hold_final_2s_tote_height_std_m"] = float(np.std(final_hold))
    result["hold_final_2s_tote_height_range_m"] = float(np.ptp(final_hold))
    return result


@torch.no_grad()
def run_episode(
    policy,
    policy_name: str,
    episode: dict,
    args: argparse.Namespace,
    expected_upper_names: list[str],
    device: str,
    hold_policy=None,
    world_model_planner: dict | None = None,
) -> dict:
    episode_index = int(episode["episode_index"])
    episode_instruction = episode["language_instruction"]
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
    upper_qvel_ids = np.asarray(
        [controlled[name]["qvel_id"] for name in upper_names], dtype=np.int64
    )

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
    planner_lower = upper_lower.copy()
    planner_upper = upper_upper.copy()
    for index, name in enumerate(upper_names):
        margin_fraction = (
            args.planner_hand_safety_margin_fraction
            if "hand_" in name
            else args.planner_body_safety_margin_fraction
        )
        margin = margin_fraction * (upper_upper[index] - upper_lower[index])
        planner_lower[index] += margin
        planner_upper[index] -= margin

    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in PALM_NAMES
    ]
    tote_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in TOTE_SITE_NAMES
    ]
    torso_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
    )
    tote_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "warehouse_tote"
    )
    assisted_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        for name in ("left_assisted_grasp", "right_assisted_grasp")
    ]
    for equality_id in assisted_ids:
        model.eq_solref[equality_id] = (args.assist_solref_timeconst, 1.0)
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
    if hold_policy is not None:
        hold_policy.reset()
    dt = float(model.opt.timestep)
    physics_per_control = max(1, round(1.0 / (args.control_fps * dt)))
    control_frames = round(args.duration_s * args.control_fps)
    current_chunk = None
    action_clip_values = 0
    action_values = 0
    joint_limit_violations = 0
    joint_samples = 0
    joint_limit_violations_by_joint = {name: 0 for name in controlled_names}
    joint_samples_by_joint = {name: 0 for name in controlled_names}
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
    world_model_inference_times = []
    planner_score_margins = []
    planner_selected_indices = []
    planner_nonbaseline_selections = 0
    planner_selected_uncertainties = []
    planner_uncertainty_rejected_fractions = []
    planner_all_candidates_uncertain = 0
    scheduler_phase = 0
    hold_latched = False
    phase_history = []
    phase_transitions = []
    previous_action = initial_targets[LOWER_BODY_JOINTS:].copy()
    terminal_hold_target = None
    terminal_hold_activation_s = None
    terminal_hold_capture_error_rmse_rad = None
    previous_hold_substep_velocity = None
    hold_substep_velocity_sq_sum = 0.0
    hold_substep_velocity_count = 0
    hold_substep_velocity_max_abs = 0.0
    hold_substep_acceleration_sq_sum = 0.0
    hold_substep_acceleration_count = 0
    hold_substep_acceleration_max_abs = 0.0
    hold_torso_angular_velocity_sq_sum = 0.0
    hold_tote_angular_velocity_sq_sum = 0.0
    hold_body_velocity_count = 0
    hold_assist_force_sq_sum = 0.0
    hold_assist_force_count = 0
    hold_assist_force_max_abs = 0.0
    hold_assist_position_error_sq_sum = 0.0
    hold_assist_position_error_count = 0
    hold_assist_position_error_max_m = 0.0
    started = time.perf_counter()

    for frame in range(control_frames):
        upper_state = np.asarray(
            [data.qpos[controlled[name]["qpos_id"]] for name in upper_names]
        )
        chunk_index = frame % args.replan_steps
        if chunk_index == 0:
            if args.phase_language_scheduler:
                distances = [
                    np.linalg.norm(data.site_xpos[palm] - data.site_xpos[tote])
                    for palm, tote in zip(palm_ids, tote_site_ids, strict=True)
                ]
                if data.time >= 0.5 and scheduler_phase == 0:
                    scheduler_phase = 1
                if scheduler_phase in (1, 2):
                    if max(distances) <= args.assist_distance_m:
                        scheduler_phase = 3
                    elif max(distances) <= args.align_distance_m:
                        scheduler_phase = max(scheduler_phase, 2)
                if assist_activation_s is not None:
                    scheduler_phase = max(scheduler_phase, 4)
                current_lift = float(data.site_xpos[tote_site_ids[0], 2]) - initial_tote_z
                if hold_latched or current_lift >= 0.10:
                    hold_latched = True
                    scheduler_phase = 5
                task = PHASE_INSTRUCTIONS[scheduler_phase]
                transition = {
                    "frame": frame,
                    "time_s": float(data.time),
                    "phase": scheduler_phase,
                    "task": task,
                }
                if not phase_transitions or phase_transitions[-1]["phase"] != scheduler_phase:
                    phase_transitions.append(transition)
            else:
                task = episode_instruction
            if args.terminal_hold_controller and scheduler_phase == 5:
                if terminal_hold_target is None:
                    terminal_hold_target = previous_action.copy()
                    terminal_hold_activation_s = float(data.time)
                    terminal_hold_capture_error_rmse_rad = float(
                        np.sqrt(np.mean((terminal_hold_target - upper_state) ** 2))
                    )
                current_chunk = np.repeat(
                    terminal_hold_target[None, :], args.replan_steps, axis=0
                )
            else:
                batch = render_observation(task_renderer, data, upper_state, task, device)
                active_policy = (
                    hold_policy
                    if hold_policy is not None and scheduler_phase == 5
                    else policy
                )
                base_seed = args.seed + episode_index * 10_000 + frame
                if world_model_planner is None:
                    inference_started = time.perf_counter()
                    current_chunk = (
                        active_policy.predict_action_chunk(
                            batch,
                            noise=seeded_noise(active_policy, base_seed, device),
                        )[0]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    inference_times.append(time.perf_counter() - inference_started)
                else:
                    planner_state = world_model_state(
                        model,
                        data,
                        controlled,
                        upper_names,
                        tote_body_id,
                        tote_site_ids[0],
                        palm_ids,
                        initial_tote_z,
                        scheduler_phase,
                    )
                    current_chunk, planner_report = select_world_model_candidate(
                        active_policy,
                        batch,
                        world_model_planner,
                        planner_state,
                        previous_action,
                        planner_lower,
                        planner_upper,
                        base_seed,
                        device,
                    )
                    inference_times.append(planner_report["policy_inference_s"])
                    world_model_inference_times.append(
                        planner_report["world_model_inference_s"]
                    )
                    planner_score_margins.append(planner_report["score_margin"])
                    planner_selected_indices.append(planner_report["selected_index"])
                    planner_nonbaseline_selections += int(
                        planner_report["selected_nonbaseline"]
                    )
                    planner_selected_uncertainties.append(
                        planner_report["selected_uncertainty"]
                    )
                    planner_uncertainty_rejected_fractions.append(
                        planner_report["uncertainty_rejected_fraction"]
                    )
                    planner_all_candidates_uncertain += int(
                        planner_report["all_candidates_uncertain"]
                    )
        if current_chunk is None or chunk_index >= len(current_chunk):
            raise RuntimeError("Policy did not produce enough actions for replanning")
        raw_action = current_chunk[chunk_index].astype(np.float64)
        action = np.clip(raw_action, upper_lower, upper_upper)
        action_clip_values += int(np.count_nonzero(np.abs(action - raw_action) > 1e-8))
        action_values += action.size
        if scheduler_phase == 5 and args.hold_action_blend_alpha < 1.0:
            alpha = args.hold_action_blend_alpha
            action = previous_action + alpha * (action - previous_action)
        previous_action = action.copy()
        targets = initial_targets.copy()
        targets[LOWER_BODY_JOINTS:] = action

        for _ in range(physics_per_control):
            if assist_activation_s is None:
                distances = [
                    np.linalg.norm(data.site_xpos[palm] - data.site_xpos[tote])
                    for palm, tote in zip(palm_ids, tote_site_ids, strict=True)
                ]
                phase_allows_assist = (
                    not args.phase_language_scheduler or scheduler_phase >= 3
                )
                if phase_allows_assist and max(distances) <= args.assist_distance_m:
                    for equality_id in assisted_ids:
                        data.eq_active[equality_id] = 1
                    assist_activation_s = float(data.time)
            for target, name in zip(targets, controlled_names, strict=True):
                item = controlled[name]
                kp, kd = unitree_gains(name)
                gain_scale = 1.5
                if scheduler_phase == 5:
                    gain_scale *= args.hold_gain_scale
                kp *= gain_scale
                kd *= math.sqrt(gain_scale)
                qpos = float(data.qpos[item["qpos_id"]])
                qvel = float(data.qvel[item["qvel_id"]])
                torque = kp * (target - qpos) - kd * qvel
                torque += float(data.qfrc_bias[item["qvel_id"]])
                data.ctrl[item["actuator_id"]] = torque
            mujoco.mj_step(model, data)
            if scheduler_phase == 5:
                hold_velocity = data.qvel[upper_qvel_ids].copy()
                hold_substep_velocity_sq_sum += float(np.sum(hold_velocity**2))
                hold_substep_velocity_count += hold_velocity.size
                hold_substep_velocity_max_abs = max(
                    hold_substep_velocity_max_abs,
                    float(np.max(np.abs(hold_velocity))),
                )
                if previous_hold_substep_velocity is not None:
                    hold_acceleration = (
                        hold_velocity - previous_hold_substep_velocity
                    ) / dt
                    hold_substep_acceleration_sq_sum += float(
                        np.sum(hold_acceleration**2)
                    )
                    hold_substep_acceleration_count += hold_acceleration.size
                    hold_substep_acceleration_max_abs = max(
                        hold_substep_acceleration_max_abs,
                        float(np.max(np.abs(hold_acceleration))),
                    )
                previous_hold_substep_velocity = hold_velocity
                hold_torso_angular_velocity_sq_sum += float(
                    np.sum(data.cvel[torso_body_id, :3] ** 2)
                )
                hold_tote_angular_velocity_sq_sum += float(
                    np.sum(data.cvel[tote_body_id, :3] ** 2)
                )
                hold_body_velocity_count += 3
                equality_rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & np.isin(
                    data.efc_id, assisted_ids
                )
                if np.any(equality_rows):
                    assist_force = data.efc_force[equality_rows]
                    hold_assist_force_sq_sum += float(np.sum(assist_force**2))
                    hold_assist_force_count += assist_force.size
                    hold_assist_force_max_abs = max(
                        hold_assist_force_max_abs,
                        float(np.max(np.abs(assist_force))),
                    )
                assist_error = np.asarray(
                    [
                        np.linalg.norm(data.site_xpos[palm] - data.site_xpos[tote])
                        for palm, tote in zip(palm_ids, tote_site_ids, strict=True)
                    ]
                )
                hold_assist_position_error_sq_sum += float(np.sum(assist_error**2))
                hold_assist_position_error_count += assist_error.size
                hold_assist_position_error_max_m = max(
                    hold_assist_position_error_max_m,
                    float(np.max(assist_error)),
                )
            if (
                args.phase_language_scheduler
                and float(data.site_xpos[tote_site_ids[0], 2]) - initial_tote_z >= 0.10
            ):
                hold_latched = True

            for name, item in controlled.items():
                joint_id = item["joint_id"]
                position = float(data.qpos[item["qpos_id"]])
                low, high = model.jnt_range[joint_id]
                joint_limit_violations += int(
                    position < low - 1e-6 or position > high + 1e-6
                )
                joint_samples += 1
                joint_limit_violations_by_joint[name] += int(
                    position < low - 1e-6 or position > high + 1e-6
                )
                joint_samples_by_joint[name] += 1
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
        phase_history.append(scheduler_phase if args.phase_language_scheduler else -1)

    writer.close()
    task_renderer.close()
    video_renderer.close()
    mujoco.mj_forward(model, data)
    final_tote_speed = float(np.linalg.norm(data.cvel[tote_body_id, 3:]))
    final_lift = float(data.site_xpos[tote_site_ids[0], 2]) - initial_tote_z
    states_array = np.asarray(states)
    commands_array = np.asarray(commands)
    tote_heights_array = np.asarray(tote_heights, dtype=np.float32)
    phase_array = np.asarray(phase_history, dtype=np.int64)
    report = {
        "policy": policy_name,
        "checkpoint": str(policy.config.pretrained_path),
        "hold_checkpoint": (
            str(hold_policy.config.pretrained_path)
            if hold_policy is not None
            else None
        ),
        "source_episode": episode_index,
        "episode_instruction": episode_instruction,
        "task_mode": (
            "phase_language_scheduler"
            if args.phase_language_scheduler
            else "episode_instruction"
        ),
        "phase_transitions": phase_transitions,
        "phase_frame_counts": {
            str(phase): phase_history.count(phase)
            for phase in sorted(set(phase_history))
        },
        "tote_x_m": float(episode["tote_x_m"]),
        "control_fps": args.control_fps,
        "replan_steps": args.replan_steps,
        "hold_action_blend_alpha": args.hold_action_blend_alpha,
        "hold_gain_scale": args.hold_gain_scale,
        "terminal_hold_controller": args.terminal_hold_controller,
        "terminal_hold_activation_s": terminal_hold_activation_s,
        "terminal_hold_capture_error_rmse_rad": (
            terminal_hold_capture_error_rmse_rad
        ),
        "action_chunk_size": int(policy.config.chunk_size),
        "assisted_grasp_activation_distance_m": args.assist_distance_m,
        "assisted_grasp_solref_timeconst_s": args.assist_solref_timeconst,
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
        "action_delta_rmse_rad": float(
            np.sqrt(np.mean(np.diff(np.asarray(commands), axis=0) ** 2))
        ),
        "joint_limit_violation_fraction": joint_limit_violations / joint_samples,
        "joint_limit_violation_fraction_by_joint": {
            name: joint_limit_violations_by_joint[name]
            / joint_samples_by_joint[name]
            for name in controlled_names
            if joint_limit_violations_by_joint[name]
        },
        "actuator_saturation_fraction": saturated_samples / actuator_samples,
        "mean_inference_s": float(np.mean(inference_times)),
        "p95_inference_s": float(np.percentile(inference_times, 95)),
        "world_model_planner_enabled": world_model_planner is not None,
        "world_model_candidate_count": (
            world_model_planner["candidate_count"] if world_model_planner else 1
        ),
        "world_model_ensemble_size": (
            len(world_model_planner["models"]) if world_model_planner else 0
        ),
        "world_model_uncertainty_threshold": (
            world_model_planner["uncertainty_threshold"]
            if world_model_planner
            else None
        ),
        "world_model_uncertainty_keep_fraction": (
            world_model_planner["uncertainty_keep_fraction"]
            if world_model_planner
            else None
        ),
        "world_model_horizon": (
            world_model_planner["horizon"] if world_model_planner else None
        ),
        "world_model_batched_candidates": (
            world_model_planner["batched_candidates"]
            if world_model_planner
            else False
        ),
        "planner_body_safety_margin_fraction": (
            args.planner_body_safety_margin_fraction
            if world_model_planner
            else 0.0
        ),
        "planner_hand_safety_margin_fraction": (
            args.planner_hand_safety_margin_fraction
            if world_model_planner
            else 0.0
        ),
        "mean_world_model_inference_s": (
            float(np.mean(world_model_inference_times))
            if world_model_inference_times
            else None
        ),
        "planner_replans": len(planner_selected_indices),
        "planner_nonbaseline_selection_rate": (
            planner_nonbaseline_selections / len(planner_selected_indices)
            if planner_selected_indices
            else None
        ),
        "mean_planner_score_margin": (
            float(np.mean(planner_score_margins)) if planner_score_margins else None
        ),
        "mean_planner_selected_uncertainty": (
            float(np.mean(planner_selected_uncertainties))
            if planner_selected_uncertainties
            else None
        ),
        "mean_planner_uncertainty_rejected_fraction": (
            float(np.mean(planner_uncertainty_rejected_fractions))
            if planner_uncertainty_rejected_fractions
            else None
        ),
        "planner_all_candidates_uncertain_rate": (
            planner_all_candidates_uncertain / len(planner_selected_indices)
            if planner_selected_indices
            else None
        ),
        "planner_selected_index_counts": {
            str(index): planner_selected_indices.count(index)
            for index in sorted(set(planner_selected_indices))
        },
        "wall_time_s": time.perf_counter() - started,
        "video": str(video_path),
        "hold_substep_joint_velocity_rmse_rad_s": (
            math.sqrt(hold_substep_velocity_sq_sum / hold_substep_velocity_count)
            if hold_substep_velocity_count
            else None
        ),
        "hold_substep_joint_velocity_max_abs_rad_s": (
            hold_substep_velocity_max_abs if hold_substep_velocity_count else None
        ),
        "hold_substep_joint_acceleration_rmse_rad_s2": (
            math.sqrt(
                hold_substep_acceleration_sq_sum / hold_substep_acceleration_count
            )
            if hold_substep_acceleration_count
            else None
        ),
        "hold_substep_joint_acceleration_max_abs_rad_s2": (
            hold_substep_acceleration_max_abs
            if hold_substep_acceleration_count
            else None
        ),
        "hold_torso_angular_velocity_rmse_rad_s": (
            math.sqrt(hold_torso_angular_velocity_sq_sum / hold_body_velocity_count)
            if hold_body_velocity_count
            else None
        ),
        "hold_tote_angular_velocity_rmse_rad_s": (
            math.sqrt(hold_tote_angular_velocity_sq_sum / hold_body_velocity_count)
            if hold_body_velocity_count
            else None
        ),
        "hold_assist_constraint_force_rmse": (
            math.sqrt(hold_assist_force_sq_sum / hold_assist_force_count)
            if hold_assist_force_count
            else None
        ),
        "hold_assist_constraint_force_max_abs": (
            hold_assist_force_max_abs if hold_assist_force_count else None
        ),
        "hold_assist_position_error_rmse_m": (
            math.sqrt(
                hold_assist_position_error_sq_sum
                / hold_assist_position_error_count
            )
            if hold_assist_position_error_count
            else None
        ),
        "hold_assist_position_error_max_m": (
            hold_assist_position_error_max_m
            if hold_assist_position_error_count
            else None
        ),
        **stability_metrics(
            states_array,
            commands_array,
            tote_heights_array,
            phase_array,
            args.control_fps,
        ),
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
        observation_joint_position_rad=states_array,
        action_joint_position_rad=commands_array,
        tote_lift_height_m=tote_heights_array,
        scheduled_phase=phase_array,
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
    def mean_available(key: str) -> float | None:
        values = [report[key] for report in reports if report[key] is not None]
        return float(np.mean(values)) if values else None

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
        "mean_action_delta_rmse_rad": float(
            np.mean([report["action_delta_rmse_rad"] for report in reports])
        ),
        "mean_final_tote_linear_speed_m_s": float(
            np.mean([report["final_tote_linear_speed_m_s"] for report in reports])
        ),
        "mean_maximum_abs_waist_pitch_rad": float(
            np.mean([report["maximum_abs_waist_pitch_rad"] for report in reports])
        ),
        "mean_joint_limit_violation_fraction": float(
            np.mean([report["joint_limit_violation_fraction"] for report in reports])
        ),
        "mean_actuator_saturation_fraction": float(
            np.mean([report["actuator_saturation_fraction"] for report in reports])
        ),
        "functional_final_lift_rate": sum(
            report["tote_lift_height_m"] >= report["required_lift_height_m"]
            for report in reports
        )
        / len(reports),
        "mean_inference_s": float(
            np.mean([report["mean_inference_s"] for report in reports])
        ),
        "mean_world_model_inference_s": mean_available(
            "mean_world_model_inference_s"
        ),
        "mean_planner_nonbaseline_selection_rate": mean_available(
            "planner_nonbaseline_selection_rate"
        ),
        "mean_planner_score_margin": mean_available("mean_planner_score_margin"),
        "mean_planner_selected_uncertainty": mean_available(
            "mean_planner_selected_uncertainty"
        ),
        "mean_planner_uncertainty_rejected_fraction": mean_available(
            "mean_planner_uncertainty_rejected_fraction"
        ),
        "mean_planner_all_candidates_uncertain_rate": mean_available(
            "planner_all_candidates_uncertain_rate"
        ),
        "mean_hold_action_delta_rmse_rad": mean_available(
            "hold_action_delta_rmse_rad"
        ),
        "mean_hold_action_delta_p95_step_norm_rad": mean_available(
            "hold_action_delta_p95_step_norm_rad"
        ),
        "mean_hold_action_delta_max_abs_rad": mean_available(
            "hold_action_delta_max_abs_rad"
        ),
        "mean_hold_joint_velocity_rmse_rad_s": mean_available(
            "hold_joint_velocity_rmse_rad_s"
        ),
        "mean_hold_joint_acceleration_rmse_rad_s2": mean_available(
            "hold_joint_acceleration_rmse_rad_s2"
        ),
        "mean_hold_switch_action_delta_rmse_rad": mean_available(
            "hold_switch_action_delta_rmse_rad"
        ),
        "mean_hold_switch_action_delta_max_abs_rad": mean_available(
            "hold_switch_action_delta_max_abs_rad"
        ),
        "mean_hold_final_2s_tote_height_std_m": mean_available(
            "hold_final_2s_tote_height_std_m"
        ),
        "mean_hold_final_2s_tote_height_range_m": mean_available(
            "hold_final_2s_tote_height_range_m"
        ),
        "mean_hold_substep_joint_velocity_rmse_rad_s": mean_available(
            "hold_substep_joint_velocity_rmse_rad_s"
        ),
        "mean_hold_substep_joint_velocity_max_abs_rad_s": mean_available(
            "hold_substep_joint_velocity_max_abs_rad_s"
        ),
        "mean_hold_substep_joint_acceleration_rmse_rad_s2": mean_available(
            "hold_substep_joint_acceleration_rmse_rad_s2"
        ),
        "mean_hold_substep_joint_acceleration_max_abs_rad_s2": mean_available(
            "hold_substep_joint_acceleration_max_abs_rad_s2"
        ),
        "mean_hold_torso_angular_velocity_rmse_rad_s": mean_available(
            "hold_torso_angular_velocity_rmse_rad_s"
        ),
        "mean_hold_tote_angular_velocity_rmse_rad_s": mean_available(
            "hold_tote_angular_velocity_rmse_rad_s"
        ),
        "mean_hold_assist_constraint_force_rmse": mean_available(
            "hold_assist_constraint_force_rmse"
        ),
        "mean_hold_assist_constraint_force_max_abs": mean_available(
            "hold_assist_constraint_force_max_abs"
        ),
        "mean_hold_assist_position_error_rmse_m": mean_available(
            "hold_assist_position_error_rmse_m"
        ),
        "mean_hold_assist_position_error_max_m": mean_available(
            "hold_assist_position_error_max_m"
        ),
        "mean_terminal_hold_capture_error_rmse_rad": mean_available(
            "terminal_hold_capture_error_rmse_rad"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.episodes < 1 or args.control_fps < 1 or args.replan_steps < 1:
        raise ValueError("episodes, control-fps, and replan-steps must be positive")
    if args.align_distance_m <= args.assist_distance_m:
        raise ValueError("align-distance-m must be larger than assist-distance-m")
    if args.assist_solref_timeconst < 2 * 0.002:
        raise ValueError("assist-solref-timeconst must be at least 0.004 s")
    if not 0.0 < args.hold_action_blend_alpha <= 1.0:
        raise ValueError("hold-action-blend-alpha must be in (0, 1]")
    if not 0.0 < args.hold_gain_scale <= 1.0:
        raise ValueError("hold-gain-scale must be in (0, 1]")
    if args.terminal_hold_controller and not args.phase_language_scheduler:
        raise ValueError(
            "--terminal-hold-controller requires --phase-language-scheduler"
        )
    if args.compare_world_model_planner and args.world_model_checkpoint is None:
        raise ValueError(
            "--compare-world-model-planner requires --world-model-checkpoint"
        )
    if (
        args.compare_unfiltered_ensemble_planner
        and not args.world_model_ensemble_checkpoint
    ):
        raise ValueError(
            "--compare-unfiltered-ensemble-planner requires an ensemble"
        )
    if args.world_model_checkpoint is not None:
        if args.world_model_candidates < 2:
            raise ValueError("world-model-candidates must be at least 2")
        if args.world_model_horizon < 1:
            raise ValueError("world-model-horizon must be positive")
    if args.world_model_ensemble_checkpoint and args.world_model_checkpoint is None:
        raise ValueError(
            "--world-model-ensemble-checkpoint requires --world-model-checkpoint"
        )
    if args.world_model_ensemble_checkpoint:
        if len(args.world_model_ensemble_checkpoint) < 2:
            raise ValueError("An ensemble requires at least two additional checkpoints")
        if (
            args.world_model_uncertainty_threshold is None
            and args.world_model_uncertainty_keep_fraction is None
            and not args.compare_unfiltered_ensemble_planner
        ):
            raise ValueError(
                "An ensemble requires a threshold, keep fraction, or unfiltered comparison"
            )
    if (
        args.world_model_uncertainty_threshold is not None
        and args.world_model_uncertainty_threshold <= 0.0
    ):
        raise ValueError("--world-model-uncertainty-threshold must be positive")
    if (
        args.world_model_uncertainty_keep_fraction is not None
        and not 0.0 < args.world_model_uncertainty_keep_fraction <= 1.0
    ):
        raise ValueError("--world-model-uncertainty-keep-fraction must be in (0, 1]")
    if (
        args.world_model_uncertainty_threshold is not None
        and args.world_model_uncertainty_keep_fraction is not None
    ):
        raise ValueError("Use either an uncertainty threshold or keep fraction, not both")
    for name in (
        "planner_body_safety_margin_fraction",
        "planner_hand_safety_margin_fraction",
    ):
        value = getattr(args, name)
        if not 0.0 <= value < 0.5:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 0.5)")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    episodes = source["episodes"][
        args.validation_start : args.validation_start + args.episodes
    ]
    if len(episodes) != args.episodes or not all(item["success"] for item in episodes):
        raise ValueError("Requested validation episodes are missing or failed")
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)
    expected_upper_names = train_meta.features["action"]["names"]
    specs = checkpoint_specs(args)
    hybrids = hybrid_specs(args.hybrid, specs)
    missing = [str(path) for path in specs.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Checkpoint directories do not exist: {missing}")
    all_reports = {}
    world_model_planner = None
    if args.world_model_checkpoint is not None:
        checkpoint_paths = [
            args.world_model_checkpoint,
            *args.world_model_ensemble_checkpoint,
        ]
        checkpoints = [
            torch.load(path, map_location=device, weights_only=False)
            for path in checkpoint_paths
        ]
        world_model_planner = {
            "models": [
                build_hybrid_from_checkpoint(item, torch.device(device)).eval()
                for item in checkpoints
            ],
            "member_stats": [item["normalizers"] for item in checkpoints],
            "candidate_count": args.world_model_candidates,
            "horizon": args.world_model_horizon,
            "batched_candidates": args.world_model_batched_candidates,
            "checkpoint": str(args.world_model_checkpoint),
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "uncertainty_threshold": args.world_model_uncertainty_threshold,
            "uncertainty_keep_fraction": (
                args.world_model_uncertainty_keep_fraction
            ),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_standalone and not hybrids:
        raise ValueError("--skip-standalone requires at least one --hybrid")
    if not args.skip_standalone:
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
            if args.compare_world_model_planner:
                if args.compare_unfiltered_ensemble_planner:
                    unfiltered_name = (
                        f"{policy_name}_world_model_ensemble_unfiltered"
                    )
                    unfiltered_planner = {
                        **world_model_planner,
                        "uncertainty_threshold": None,
                        "uncertainty_keep_fraction": None,
                    }
                    all_reports[unfiltered_name] = [
                        run_episode(
                            policy,
                            unfiltered_name,
                            episode,
                            args,
                            expected_upper_names,
                            device,
                            world_model_planner=unfiltered_planner,
                        )
                        for episode in episodes
                    ]
                planner_name = f"{policy_name}_world_model_mpc"
                all_reports[planner_name] = [
                    run_episode(
                        policy,
                        planner_name,
                        episode,
                        args,
                        expected_upper_names,
                        device,
                        world_model_planner=world_model_planner,
                    )
                    for episode in episodes
                ]
            del policy
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    for policy_name, (base_name, hold_name) in hybrids.items():
        print(
            f"Loading {policy_name}: base={specs[base_name]} hold={specs[hold_name]}",
            flush=True,
        )
        base_policy = load_policy(specs[base_name], train_meta, device)
        hold_policy = load_policy(specs[hold_name], train_meta, device)
        if not args.phase_language_scheduler:
            raise ValueError("Hybrid policies require --phase-language-scheduler")
        all_reports[policy_name] = [
            run_episode(
                base_policy,
                policy_name,
                episode,
                args,
                expected_upper_names,
                device,
                hold_policy=hold_policy,
            )
            for episode in episodes
        ]
        del base_policy, hold_policy
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    report = {
        "experiment": args.experiment_id,
        "device": device,
        "validation_source_episodes": [item["episode_index"] for item in episodes],
        "control_fps": args.control_fps,
        "replan_steps": args.replan_steps,
        "phase_language_scheduler": args.phase_language_scheduler,
        "hold_action_blend_alpha": args.hold_action_blend_alpha,
        "hold_gain_scale": args.hold_gain_scale,
        "terminal_hold_controller": args.terminal_hold_controller,
        "world_model_checkpoint": (
            str(args.world_model_checkpoint)
            if args.world_model_checkpoint is not None
            else None
        ),
        "world_model_ensemble_checkpoints": [
            str(path) for path in args.world_model_ensemble_checkpoint
        ],
        "world_model_uncertainty_threshold": (
            args.world_model_uncertainty_threshold
        ),
        "world_model_uncertainty_keep_fraction": (
            args.world_model_uncertainty_keep_fraction
        ),
        "world_model_candidates": args.world_model_candidates,
        "world_model_horizon": args.world_model_horizon,
        "world_model_batched_candidates": args.world_model_batched_candidates,
        "planner_body_safety_margin_fraction": (
            args.planner_body_safety_margin_fraction
        ),
        "planner_hand_safety_margin_fraction": (
            args.planner_hand_safety_margin_fraction
        ),
        "assist_solref_timeconst_s": args.assist_solref_timeconst,
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
