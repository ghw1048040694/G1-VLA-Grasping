#!/usr/bin/env python3
"""Replay saved G1LANG-396 states to separate VLA and MPC action errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

import evaluate_g1_language_pick_place as base
from audit_g1_396_temporal_expert_divergence import (
    choose_expert,
    load_experts,
    rmse,
)
from evaluate_g1_phase_blind_local_wm_mpc import PhaseBlindLocalPolicy
from g1_finger_contact_geometry import finger_contact_geom_ids
from run_g1_language_pick_place import (
    OBJECT_NAMES,
    PALM_NAMES,
    controlled_joints,
)
from validate_g1_bimanual_actuation import apply_regularized_dynamics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def close_renderer(renderer: mujoco.Renderer) -> None:
    close = getattr(renderer, "close", None)
    if close is not None:
        close()


def history_tensor(policy: PhaseBlindLocalPolicy) -> torch.Tensor:
    history = policy.causal_history
    return torch.cat(
        ([history[0]] * (policy.history_frames - len(history))) + history,
        dim=0,
    ).unsqueeze(0)


def replay_episode(
    args: argparse.Namespace,
    report: dict,
    expert,
) -> dict:
    episode_dir = args.rollout_dir / f"episode_{int(report['episode_index']):04d}"
    with np.load(episode_dir / "trajectory.npz") as trajectory:
        state = trajectory["observation_joint_position_rad"].astype(np.float32)
        velocity = trajectory["observation_joint_velocity_rad_s"].astype(np.float32)
        deployed_action = trajectory["action_joint_position_rad"].astype(np.float32)
        objects = trajectory["object_position_m"].astype(np.float32)
        joint_names = trajectory["joint_names"]
    frame_count = min(args.frames, len(state), len(expert.state))
    model = mujoco.MjModel.from_xml_path(str(episode_dir / "scene.xml"))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    controlled_names = list(controlled)
    upper_names = controlled_names[base.LOWER_BODY_JOINTS:]
    if upper_names != joint_names.tolist():
        raise ValueError("Trajectory joint ordering disagrees with the replay scene")
    qpos_ids = [controlled[name]["qpos_id"] for name in upper_names]
    qvel_ids = [controlled[name]["qvel_id"] for name in upper_names]
    action_lower = np.asarray(
        [model.jnt_range[controlled[name]["joint_id"], 0] for name in upper_names],
        dtype=np.float32,
    )
    action_upper = np.asarray(
        [model.jnt_range[controlled[name]["joint_id"], 1] for name in upper_names],
        dtype=np.float32,
    )
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_NAMES
    ]
    contact_pad_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_palm_contact_pad")
        for side in ("left", "right")
    ]
    object_body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in OBJECT_NAMES
    }
    object_qpos_ids = {
        name: int(
            model.jnt_qposadr[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_freejoint")
            ]
        )
        for name in OBJECT_NAMES
    }
    renderer = mujoco.Renderer(model, height=120, width=160)
    policy = PhaseBlindLocalPolicy(
        args.checkpoint,
        args.world_model_checkpoint,
        args.device,
        control_fps=15,
        wm_horizon=1,
        action_delta_limit=0.08,
    )
    target = OBJECT_NAMES.index(report["target_object"])
    rows = []
    try:
        for frame in range(frame_count):
            data.qpos[qpos_ids] = state[frame]
            data.qvel[qvel_ids] = velocity[frame]
            for object_index, name in enumerate(OBJECT_NAMES):
                address = object_qpos_ids[name]
                data.qpos[address : address + 3] = objects[frame, object_index]
                data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
            mujoco.mj_forward(model, data)
            policy.set_planner_context(
                {
                    "model": model,
                    "data": data,
                    "controlled": controlled,
                    "upper_names": upper_names,
                    "action_lower": action_lower,
                    "action_upper": action_upper,
                    "palm_ids": palm_ids,
                    "contact_pad_ids": contact_pad_ids,
                    "finger_contact_geom_ids": finger_contact_geom_ids(model),
                    "object_body_ids": object_body_ids,
                }
            )
            batch = base.render_observation(
                renderer, data, state[frame], report["language_instruction"], args.device
            )
            policy.observe_history(batch)
            observation = policy.pending_observation
            output = policy.model(
                observation["normalized_state"],
                (observation["velocity"] - policy.stats["velocity_mean"])
                / policy.stats["velocity_std"],
                observation["objects"],
                observation["contact_sites"],
                observation["tactile_contact"],
                observation["vision_color"],
                observation["ids"],
                observation["mask"],
                history_tensor(policy),
            )
            model_target = int(output["target_logits"].argmax(-1)[0])
            predicted_arm = int(
                output["active_arm_logits_by_target"][0, model_target].argmax()
            )
            active_arm = (
                predicted_arm if policy.latched_active_arm is None else policy.latched_active_arm
            )
            sparse = (
                output["action_candidates"][0, model_target, active_arm]
                * policy.stats["action_std"]
                + policy.stats["action_mean"]
            )
            sparse = torch.maximum(
                torch.minimum(
                    sparse, policy.stats["candidate_upper"][model_target, active_arm]
                ),
                policy.stats["candidate_lower"][model_target, active_arm],
            )
            plans = policy._candidate_plans(
                sparse.detach().cpu().numpy(), state[frame]
            )
            selected_plan = policy._select_with_world_model(
                observation["state"],
                observation["velocity"],
                observation["contact_sites"],
                observation["tactile_contact"],
                observation["objects"],
                output,
            )
            replay_choice = policy.candidate_reports[-1]
            stored_choice = report["world_model_candidates"][frame]
            rows.append(
                {
                    "frame": frame,
                    "time_s": frame / 15.0,
                    "model_target_index": model_target,
                    "model_active_arm_index": predicted_arm,
                    "latched_active_arm_index": int(active_arm),
                    "raw_vla_offset0_rmse_to_expert_rad": float(
                        rmse(sparse.detach().cpu().numpy()[0] - expert.action[frame])
                    ),
                    "bounded_scale_0_70_rmse_to_expert_rad": float(
                        rmse(plans[0, 0] - expert.action[frame])
                    ),
                    "bounded_scale_1_00_rmse_to_expert_rad": float(
                        rmse(plans[1, 0] - expert.action[frame])
                    ),
                    "bounded_scale_1_20_rmse_to_expert_rad": float(
                        rmse(plans[2, 0] - expert.action[frame])
                    ),
                    "selected_mpc_rmse_to_expert_rad": float(
                        rmse(selected_plan[0] - expert.action[frame])
                    ),
                    "stored_deployed_rmse_to_expert_rad": float(
                        rmse(deployed_action[frame] - expert.action[frame])
                    ),
                    "replayed_selected_scale": replay_choice["candidate_plan_scales"]
                    [replay_choice["selected_candidate"]],
                    "stored_selected_scale": stored_choice["candidate_plan_scales"]
                    [stored_choice["selected_candidate"]],
                }
            )
    finally:
        close_renderer(renderer)
    return {
        "episode_index": int(report["episode_index"]),
        "target_object": report["target_object"],
        "nearest_strict_expert": expert.name,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    experts = load_experts(args.cache_dir, args.expert_root)
    reports = json.loads((args.rollout_dir / "summary.json").read_text())["reports"]
    output = []
    for report in reports:
        episode_dir = args.rollout_dir / f"episode_{int(report['episode_index']):04d}"
        with np.load(episode_dir / "trajectory.npz") as trajectory:
            state = trajectory["observation_joint_position_rad"]
            action = trajectory["action_joint_position_rad"]
            objects = trajectory["object_position_m"]
        target = OBJECT_NAMES.index(report["target_object"])
        arm = int(report["world_model_candidates"][0]["active_arm_index"])
        expert, _ = choose_expert(experts, target, arm, state, action, objects)
        output.append(replay_episode(args, report, expert))
    result = {
        "experiment": "G1LANG-396-raw-vla-versus-mpc-replay-audit",
        "contract": "read-only MuJoCo rendering of saved rollout states; no physics step or model update",
        "episodes": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
