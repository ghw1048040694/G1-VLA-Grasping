#!/usr/bin/env python3
"""Audit strict-expert recording dynamics against a deployed rollout.

This is a read-only diagnostic.  It separates a MuJoCo execution mismatch
from a policy target mismatch before another model-training attempt is spent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from run_g1_language_pick_place import controlled_joints
from validate_g1_bimanual_actuation import apply_regularized_dynamics, unitree_gains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-dir", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--body-gain-scale", type=float, default=5.0)
    parser.add_argument("--hand-gain-scale", type=float, default=6.0)
    return parser.parse_args()


def metric(values: np.ndarray, names: list[str]) -> dict:
    absolute = np.abs(values)
    flat = int(absolute.argmax())
    frame, joint = np.unravel_index(flat, absolute.shape)
    return {
        "rmse_rad": float(np.sqrt(np.mean(np.square(values)))),
        "max_abs_rad": float(absolute[frame, joint]),
        "max_frame": int(frame),
        "max_joint": names[int(joint)],
    }


def replay_expert_interval(
    source: Path,
    joint_names: list[str],
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    physics_steps: int,
    body_gain_scale: float,
    hand_gain_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(source / "g1_language_pick_place.xml"))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    controlled = controlled_joints(model)
    if list(controlled) != joint_names:
        raise ValueError("Expert joint ordering does not match the source scene")
    qpos_ids = [item["qpos_id"] for item in controlled.values()]
    qvel_ids = [item["qvel_id"] for item in controlled.values()]
    data.qpos[qpos_ids] = qpos
    data.qvel[qvel_ids] = qvel
    mujoco.mj_forward(model, data)
    for _ in range(physics_steps):
        for index, item in enumerate(controlled.values()):
            name = joint_names[index]
            kp, kd = unitree_gains(name)
            scale = hand_gain_scale if "hand_" in name else body_gain_scale
            torque = (
                scale * kp * (action[index] - data.qpos[item["qpos_id"]])
                - np.sqrt(scale) * kd * data.qvel[item["qvel_id"]]
                + data.qfrc_bias[item["qvel_id"]]
            )
            data.ctrl[item["actuator_id"]] = torque
        mujoco.mj_step(model, data)
    return data.qpos[qpos_ids].copy(), data.qvel[qvel_ids].copy()


def main() -> None:
    args = parse_args()
    with np.load(args.expert_dir / "expert_pick_place_episode.npz") as values:
        expert_names = values["joint_names"].tolist()
        expert_state = values["joint_position_rad"].astype(np.float64)
        expert_velocity = values["joint_velocity_rad_s"].astype(np.float64)
        expert_action = values["action_joint_position_rad"].astype(np.float64)
        expert_time = values["time_s"].astype(np.float64)
    with np.load(args.rollout_dir / "trajectory.npz") as values:
        rollout_names = values["joint_names"].tolist()
        rollout_state = values["observation_joint_position_rad"].astype(np.float64)
        rollout_action = values["action_joint_position_rad"].astype(np.float64)
        rollout_fps = int(values["control_fps"])
    if expert_names[12:] != rollout_names:
        raise ValueError("Expert and rollout upper-body joint ordering differs")
    if rollout_fps != args.control_fps:
        raise ValueError("Recorded rollout FPS differs from --control-fps")
    if len(expert_time) < 2:
        raise ValueError("Expert recording has fewer than two samples")
    model = mujoco.MjModel.from_xml_path(
        str(args.expert_dir / "g1_language_pick_place.xml")
    )
    physics_steps = round(1.0 / (args.control_fps * model.opt.timestep))
    replay_qpos, replay_qvel = replay_expert_interval(
        args.expert_dir,
        expert_names,
        expert_state[0],
        expert_velocity[0],
        expert_action[0],
        physics_steps,
        args.body_gain_scale,
        args.hand_gain_scale,
    )
    frame_count = min(args.frames, len(rollout_state), len(expert_state) - 1)
    if frame_count < 1:
        raise ValueError("Not enough overlapping rollout/expert frames")
    upper_names = rollout_names
    # A deployed trajectory is recorded after each control interval.  The
    # matching strict state is therefore expert frame t+1, while the target
    # applied during that interval is strict action t.
    state_error = rollout_state[:frame_count] - expert_state[1 : frame_count + 1, 12:]
    action_error = rollout_action[:frame_count] - expert_action[:frame_count, 12:]
    result = {
        "experiment": "G1LANG-execution-contract-audit",
        "contract": (
            "read-only: replay the strict expert's first recorded target through "
            "the identical MuJoCo PD/gravity-compensation path, then compare a "
            "deployed rollout with the time-aligned strict target/state"
        ),
        "expert_dir": str(args.expert_dir),
        "rollout_dir": str(args.rollout_dir),
        "control_fps": args.control_fps,
        "physics_timestep_s": float(model.opt.timestep),
        "physics_steps_per_control": physics_steps,
        "expert_record_interval_s": float(expert_time[1] - expert_time[0]),
        "strict_recording_replay": {
            "qpos_upper": metric(
                (replay_qpos - expert_state[1])[None, 12:], expert_names[12:]
            ),
            "qvel_upper": metric(
                (replay_qvel - expert_velocity[1])[None, 12:], expert_names[12:]
            ),
        },
        "deployed_vs_strict": {
            "frames": frame_count,
            "upper_state": metric(state_error, upper_names),
            "command_target": metric(action_error, upper_names),
        },
    }
    result["conclusion"] = (
        "execution_contract_matches_to_milliradian_scale_but_policy_commands_diverge"
        if result["strict_recording_replay"]["qpos_upper"]["max_abs_rad"] < 0.005
        and result["deployed_vs_strict"]["command_target"]["max_abs_rad"] > 0.02
        else "requires_further_execution_contract_investigation"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
