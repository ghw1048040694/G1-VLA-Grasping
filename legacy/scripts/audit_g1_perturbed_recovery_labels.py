#!/usr/bin/env python3
"""Physically audit strict expert-continuation recovery labels.

This is an offline label-quality check. It restores a perturbed state, converts
the same 35D Cartesian labels used by deployment back through runtime IK, and
executes them with contact-only MuJoCo physics. Object state is never an input
to the policy; it is used here only to restore and score offline samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from build_g1_perturbed_contact_recovery_cache import (
    DEFAULT_RECOVERY_PHASES,
    PHASE_PERTURBATION_SCALE,
    contact_action,
    parse_recovery_phases,
    perturb_active_arm,
    restore_state,
    selected_episodes,
)
from evaluate_g1_compact_vla_closed_loop import contact_action_to_joint_chunk
from run_g1_language_pick_place import OBJECT_NAMES, PALM_NAMES, controlled_joints
from validate_g1_bimanual_actuation import apply_regularized_dynamics, unitree_gains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-target", type=int, default=1)
    parser.add_argument("--samples-per-episode", type=int, default=2)
    parser.add_argument("--chunk-horizon", type=int, default=20)
    parser.add_argument("--replan-chunks", type=int, default=6)
    parser.add_argument(
        "--recovery-phases",
        type=parse_recovery_phases,
        default=DEFAULT_RECOVERY_PHASES,
        help="Comma-separated source phases to audit.",
    )
    parser.add_argument("--control-fps", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def active_hand_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_body: int,
    active_side: str,
) -> bool:
    """Return true only for a real target-object contact with the active hand."""
    for index in range(data.ncon):
        contact = data.contact[index]
        body_ids = (
            int(model.geom_bodyid[contact.geom1]),
            int(model.geom_bodyid[contact.geom2]),
        )
        if target_body not in body_ids:
            continue
        hand_body = body_ids[1 - body_ids.index(target_body)]
        while hand_body > 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, hand_body) or ""
            if name.startswith(f"{active_side}_hand_"):
                return True
            hand_body = int(model.body_parentid[hand_body])
    return False


def command_chunk(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    upper_names: list[str],
    lower_targets: dict[str, float],
    chunk: np.ndarray,
    body_gain: float,
    hand_gain: float,
    physics_per_control: int,
    target_body: int,
    active_side: str,
) -> tuple[int, float, int]:
    """Execute one decoded action chunk with the production PD convention."""
    contact_steps = 0
    maximum_height = -np.inf
    equality_active_steps = 0
    for row in chunk:
        targets = dict(lower_targets)
        targets.update(zip(upper_names, row.tolist()))
        for _ in range(physics_per_control):
            for name, item in controlled.items():
                kp, kd = unitree_gains(name)
                scale = hand_gain if "hand_" in name else body_gain
                kp *= scale
                kd *= np.sqrt(scale)
                qpos = float(data.qpos[item["qpos_id"]])
                qvel = float(data.qvel[item["qvel_id"]])
                data.ctrl[item["actuator_id"]] = (
                    kp * (targets[name] - qpos)
                    - kd * qvel
                    + float(data.qfrc_bias[item["qvel_id"]])
                )
            mujoco.mj_step(model, data)
            equality_active_steps += int(np.count_nonzero(model.eq_active))
            contact_steps += int(active_hand_contact(model, data, target_body, active_side))
            maximum_height = max(maximum_height, float(data.xpos[target_body, 2]))
    return contact_steps, maximum_height, equality_active_steps


def audit_sample(
    episode_dir: Path,
    frame: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> dict:
    summary = json.loads((episode_dir / "language_pick_place_summary.json").read_text())
    target = summary["target_object"]
    target_index = OBJECT_NAMES.index(target)
    with np.load(episode_dir / "expert_pick_place_episode.npz", allow_pickle=True) as source:
        names = source["joint_names"].copy()
        positions = source["joint_position_rad"].copy()
        actions = source["action_joint_position_rad"].copy()
        objects = source["object_position_m"].copy()
        quaternions = source["object_quaternion_wxyz"].copy()
        active_arms = source["active_arm_index"].copy()
        phase = int(source["task_phase"][frame])

    model = mujoco.MjModel.from_xml_path(str(episode_dir / "g1_language_pick_place.xml"))
    apply_regularized_dynamics(model)
    model.eq_active[:] = 0
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    upper_names = list(controlled)[12:]
    source_index = {name: index for index, name in enumerate(names.tolist())}
    source_upper = np.asarray([source_index[name] for name in upper_names], dtype=np.int64)
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_NAMES
    ]
    target_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    baseline_qpos = data.qpos.copy()
    restore_state(
        model, data, controlled, upper_names, names, positions[frame], objects[frame],
        quaternions[frame], baseline_qpos,
    )
    perturbation_scale = PHASE_PERTURBATION_SCALE[phase]
    perturb_active_arm(
        model, data, controlled, target, rng,
        amplitude_scale=perturbation_scale,
    )
    initial_height = float(data.xpos[target_body, 2])
    active_arm = int(active_arms[frame])
    active_side = "left" if active_arm == 0 else "right"
    lower_targets = {
        name: float(data.qpos[item["qpos_id"]])
        for name, item in controlled.items()
        if name not in upper_names
    }
    body_gain = float(summary["controller_gain_scale"]["body"])
    hand_gain = float(summary["controller_gain_scale"]["hand"])
    physics_per_control = max(1, round(1.0 / (args.control_fps * model.opt.timestep)))

    contact_steps = 0
    maximum_height = initial_height
    equality_active_steps = 0
    ik_failed_waypoints = 0
    ik_max_error = 0.0
    plans = []
    for replan in range(args.replan_chunks):
        start = frame + replan * args.chunk_horizon + 1
        stop = start + args.chunk_horizon
        if stop > len(actions):
            break
        expert_chunk = actions[start:stop, source_upper].astype(np.float32)
        current_state = np.asarray(
            [data.qpos[controlled[name]["qpos_id"]] for name in upper_names],
            dtype=np.float32,
        )
        cartesian = contact_action(
            model, data, controlled, upper_names, palm_ids, expert_chunk
        ).numpy()
        decoded, ik_report = contact_action_to_joint_chunk(
            cartesian, model, data, controlled, upper_names, palm_ids, current_state
        )
        current_contact, current_height, current_equality = command_chunk(
            model, data, controlled, upper_names, lower_targets, decoded, body_gain,
            hand_gain, physics_per_control, target_body, active_side,
        )
        contact_steps += current_contact
        maximum_height = max(maximum_height, current_height)
        equality_active_steps += current_equality
        ik_failed_waypoints += int(ik_report["failed_waypoints"])
        ik_max_error = max(ik_max_error, float(ik_report["max_position_error_m"]))
        plans.append(
            {
                "source_start_frame": start,
                "phase": int(phase),
                "contact_steps": current_contact,
                "max_position_error_m": float(ik_report["max_position_error_m"]),
                "failed_waypoints": int(ik_report["failed_waypoints"]),
            }
        )

    lift_m = maximum_height - initial_height
    return {
        "episode": episode_dir.name,
        "target": target,
        "source_frame": frame,
        "source_phase": phase,
        "perturbation_scale": perturbation_scale,
        "active_arm": active_arm,
        "plans": plans,
        "physical_contact_steps": contact_steps,
        "maximum_lift_m": lift_m,
        "cartesian_ik_failed_waypoints": ik_failed_waypoints,
        "cartesian_ik_max_position_error_m": ik_max_error,
        "assisted_equality_active_steps": equality_active_steps,
        "contact_recovery_passed": bool(contact_steps >= 3 and lift_m >= 0.08),
    }


def main() -> None:
    args = parse_args()
    if min(args.episodes_per_target, args.samples_per_episode, args.chunk_horizon, args.replan_chunks) < 1:
        raise ValueError("episode/sample/horizon/replan arguments must be positive")
    rng = np.random.default_rng(args.seed)
    reports = []
    for episode_dir in selected_episodes(args.source_root, args.episodes_per_target):
        with np.load(episode_dir / "expert_pick_place_episode.npz", allow_pickle=True) as source:
            phases = source["task_phase"].copy()
            contacts = source["physical_contact_active"].copy()
            horizon = len(phases) - args.replan_chunks * args.chunk_horizon - 1
        candidates = np.flatnonzero(
            np.isin(phases, args.recovery_phases)
            & (contacts == 0)
            & (np.arange(len(phases)) < horizon)
        )
        if len(candidates) < args.samples_per_episode:
            raise RuntimeError(f"Insufficient auditable source frames in {episode_dir}")
        for frame in rng.choice(candidates, size=args.samples_per_episode, replace=False):
            report = audit_sample(episode_dir, int(frame), args, rng)
            reports.append(report)
            print(
                json.dumps(
                    {key: report[key] for key in (
                        "episode", "target", "source_frame", "source_phase",
                        "physical_contact_steps", "maximum_lift_m",
                        "cartesian_ik_failed_waypoints", "contact_recovery_passed",
                    )},
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = {
        "experiment": "G1LANG_perturbed_recovery_label_physics_audit",
        "samples": len(reports),
        "recovery_phases": list(args.recovery_phases),
        "input_contract": "offline audit of deployment-equivalent Cartesian-to-joint action path",
        "grasp_mode": "strict physical contact only",
        "assisted_grasp_enabled": False,
        "reports": reports,
        "contact_recovery_success_rate": float(
            np.mean([item["contact_recovery_passed"] for item in reports])
        ),
        "total_assisted_equality_active_steps": int(
            sum(item["assisted_equality_active_steps"] for item in reports)
        ),
        "total_cartesian_ik_failed_waypoints": int(
            sum(item["cartesian_ik_failed_waypoints"] for item in reports)
        ),
    }
    summary["passed"] = bool(
        summary["contact_recovery_success_rate"] >= 0.80
        and summary["total_assisted_equality_active_steps"] == 0
        and summary["total_cartesian_ik_failed_waypoints"] == 0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
