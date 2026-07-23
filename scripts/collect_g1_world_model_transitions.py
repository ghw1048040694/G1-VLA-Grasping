#!/usr/bin/env python3
"""Collect control-rate G1 transitions with nominal and perturbed expert actions."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

from train_g1_world_model import LAYOUT, upper_body_indices
from validate_g1_bimanual_actuation import apply_regularized_dynamics, unitree_gains


NOISE_LEVELS_RAD = (0.03, 0.06, 0.10)


def object_name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"unnamed_{index}"


def controlled_joints(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
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
    return controlled


def contact_flags(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float]:
    hands = np.zeros(2, dtype=np.float32)
    table = 0.0
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = {
            object_name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[int(geom)]),
            )
            for geom in (contact.geom1, contact.geom2)
        }
        if "warehouse_tote" not in bodies:
            continue
        hands[0] = max(hands[0], float(any(name.startswith("left_hand_") for name in bodies)))
        hands[1] = max(hands[1], float(any(name.startswith("right_hand_") for name in bodies)))
        table = max(table, float("world" in bodies))
    return hands, table


def state_vector(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    joint_names: np.ndarray,
    upper: np.ndarray,
    tote_body_id: int,
    tote_site_id: int,
    palm_ids: list[int],
    initial_tote_z: float,
    phase: int,
) -> np.ndarray:
    qpos = np.asarray([data.qpos[controlled[name]["qpos_id"]] for name in joint_names])
    qvel = np.asarray([data.qvel[controlled[name]["qvel_id"]] for name in joint_names])
    hands, table = contact_flags(model, data)
    lift = float(data.site_xpos[tote_site_id, 2] - initial_tote_z)
    contact_progress = float(hands.mean())
    lift_progress = float(np.clip(lift / 0.10, 0.0, 1.0))
    progress = (float(phase) / 5.0 + contact_progress + lift_progress) / 3.0
    state = np.concatenate(
        (
            qpos[upper],
            qvel[upper],
            data.xpos[tote_body_id],
            data.xquat[tote_body_id],
            data.cvel[tote_body_id, 3:],
            data.cvel[tote_body_id, :3],
            np.stack([data.site_xpos[item] for item in palm_ids]).reshape(-1),
            hands,
            np.asarray((table, lift, progress)),
        )
    ).astype(np.float32)
    if state.shape != (LAYOUT.state_dim,):
        raise RuntimeError(f"Unexpected state shape: {state.shape}")
    return state


def collect_rollout(
    source: dict,
    output_path: Path,
    noise_std_rad: float,
    seed: int,
    control_fps: int,
    noise_correlation: float,
) -> dict:
    episode_dir = Path(source["episode_dir"])
    scene_path = episode_dir / "g1_assisted_tote_lift.xml"
    expert_path = Path(source["dataset"])
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    with np.load(expert_path) as expert:
        joint_names = expert["joint_names"].astype(str)
        upper = upper_body_indices(joint_names)
        source_qpos = expert["observation_joint_position_rad"].copy()
        source_qvel = expert["observation_joint_velocity_rad_s"].copy()
        source_action = expert["action_joint_position_rad"].copy()
        phases = expert["task_phase"].copy()
        assist = expert["assisted_grasp_active"].copy()
    missing = [name for name in joint_names if name not in controlled]
    if missing:
        raise RuntimeError(f"Scene is missing controlled joints: {missing}")

    mujoco.mj_resetData(model, data)
    for index, name in enumerate(joint_names):
        data.qpos[controlled[name]["qpos_id"]] = source_qpos[0, index]
        data.qvel[controlled[name]["qvel_id"]] = source_qvel[0, index]
    mujoco.mj_forward(model, data)
    tote_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "warehouse_tote")
    tote_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tote_left_assist_site")
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in ("left_palm_center", "right_palm_center")
    ]
    assist_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        for name in ("left_assisted_grasp", "right_assisted_grasp")
    ]
    initial_tote_z = float(data.site_xpos[tote_site_id, 2])
    substeps = max(1, round(1.0 / (control_fps * float(model.opt.timestep))))
    rng = np.random.default_rng(seed)
    noise = np.zeros(len(upper), dtype=np.float64)
    noise_scale = math.sqrt(1.0 - noise_correlation**2) * noise_std_rad
    transitions = len(source_action) - 1
    states = []
    actions = []
    applied_noise = []
    clipped = 0
    action_values = 0
    violations = defaultdict(int)

    for step in range(transitions):
        phase = int(phases[step])
        states.append(
            state_vector(
                model,
                data,
                controlled,
                joint_names,
                upper,
                tote_body_id,
                tote_site_id,
                palm_ids,
                initial_tote_z,
                phase,
            )
        )
        noise = noise_correlation * noise + noise_scale * rng.standard_normal(len(upper))
        target = source_action[step].astype(np.float64)
        target[upper] += noise
        actual_noise = np.zeros(len(upper), dtype=np.float64)
        for upper_index, joint_index in enumerate(upper):
            name = joint_names[joint_index]
            joint_id = controlled[name]["joint_id"]
            low, high = model.jnt_range[joint_id]
            if name == "waist_pitch_joint":
                low, high = max(low, -0.10), min(high, 0.10)
            elif name in ("waist_yaw_joint", "waist_roll_joint"):
                low, high = max(low, -0.05), min(high, 0.05)
            unclipped = target[joint_index]
            target[joint_index] = np.clip(unclipped, low, high)
            clipped += int(target[joint_index] != unclipped)
            action_values += 1
            actual_noise[upper_index] = target[joint_index] - source_action[step, joint_index]
        actions.append(target[upper].astype(np.float32))
        applied_noise.append(actual_noise.astype(np.float32))

        for equality_id in assist_ids:
            data.eq_active[equality_id] = int(assist[step])
        for _ in range(substeps):
            for index, name in enumerate(joint_names):
                item = controlled[name]
                kp, kd = unitree_gains(name)
                kp *= 1.5
                kd *= math.sqrt(1.5)
                qpos = float(data.qpos[item["qpos_id"]])
                qvel = float(data.qvel[item["qvel_id"]])
                torque = kp * (target[index] - qpos) - kd * qvel
                torque += float(data.qfrc_bias[item["qvel_id"]])
                data.ctrl[item["actuator_id"]] = torque
            mujoco.mj_step(model, data)
        for name in joint_names[upper]:
            item = controlled[name]
            low, high = model.jnt_range[item["joint_id"]]
            value = float(data.qpos[item["qpos_id"]])
            violations[name] += int(value < low - 1e-6 or value > high + 1e-6)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f"Non-finite simulation state at step {step}")

    states.append(
        state_vector(
            model,
            data,
            controlled,
            joint_names,
            upper,
            tote_body_id,
            tote_site_id,
            palm_ids,
            initial_tote_z,
            int(phases[-1]),
        )
    )
    states_array = np.asarray(states, dtype=np.float32)
    actions_array = np.asarray(actions, dtype=np.float32)
    noise_array = np.asarray(applied_noise, dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        state=states_array,
        action=actions_array,
        applied_action_noise_rad=noise_array,
        upper_body_joint_names=joint_names[upper],
    )
    lift_index = LAYOUT.lift_height[0]
    return {
        "source_episode_index": int(source["episode_index"]),
        "condition": "nominal" if noise_std_rad == 0.0 else f"noise_{noise_std_rad:.2f}",
        "noise_std_rad": noise_std_rad,
        "seed": seed,
        "transitions": transitions,
        "control_fps": control_fps,
        "action_hold_substeps": substeps,
        "action_clip_fraction": clipped / max(action_values, 1),
        "applied_noise_rmse_rad": float(np.sqrt(np.mean(noise_array**2))),
        "final_lift_height_m": float(states_array[-1, lift_index]),
        "maximum_lift_height_m": float(states_array[:, lift_index].max()),
        "joint_limit_violation_fraction": sum(violations.values()) / max(transitions * len(upper), 1),
        "joint_limit_violation_counts": {key: value for key, value in violations.items() if value},
        "dataset": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--noise-correlation", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=4201)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--experiment-id", default="G1WM-02-control-rate-action-interventions")
    args = parser.parse_args()
    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    sources = [item for item in source_summary["episodes"] if item["success"]][: args.episodes]
    if len(sources) != args.episodes:
        raise ValueError(f"Expected {args.episodes} successful sources, found {len(sources)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for source in sources:
        source_index = int(source["episode_index"])
        conditions = (("nominal", 0.0), ("perturbed", NOISE_LEVELS_RAD[source_index % len(NOISE_LEVELS_RAD)]))
        for condition_name, noise_std in conditions:
            output_path = args.output_dir / f"source_{source_index:04d}_{condition_name}.npz"
            metadata_path = output_path.with_suffix(".json")
            if args.resume and output_path.is_file() and metadata_path.is_file():
                result = json.loads(metadata_path.read_text(encoding="utf-8"))
                print(f"Reusing {output_path.name}", flush=True)
            else:
                result = collect_rollout(
                    source,
                    output_path,
                    noise_std,
                    args.seed + source_index * 2 + int(noise_std > 0.0),
                    args.control_fps,
                    args.noise_correlation,
                )
                metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(result), flush=True)
            results.append(result)
    counts = defaultdict(int)
    for item in results:
        counts[item["condition"]] += 1
    summary = {
        "experiment": args.experiment_id,
        "source_summary": str(args.source_summary),
        "source_episodes": args.episodes,
        "rollouts": len(results),
        "control_fps": args.control_fps,
        "noise_correlation": args.noise_correlation,
        "condition_counts": dict(counts),
        "episodes": results,
        "experiment_passed": len(results) == args.episodes * 2,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
