#!/usr/bin/env python3
"""Collect strict expert recoveries from actual VLA states in reachable scenes.

The source rollout is produced by a deployed policy with no object state input.
For each failed policy state this tool reconstructs the same continuous slot-
jitter scene, restores only the observed 31-D robot state, and runs the strict
contact-only expert as an offline teacher.  A trajectory is retained only when
the entire recovery physically contacts, lifts, and places its language target
without an equality constraint.  The resulting demonstrations retain three RGB
views, language, robot states, and executed 31-D actions for the next VLA/WM
training round.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from collect_g1_language_pick_place_dataset import contact_profile_for_target_slot
from g1_finger_contact_geometry import finger_contact_geom_ids, finger_contact_sites
from run_g1_language_pick_place import OBJECT_NAMES, OBJECT_SLOTS_XY_M


RECOVERY_CONTRACT_VERSION = "g1lang_v8_contact_relative_pose_transport_recovery"


def parse_frames(value: str) -> tuple[int, ...]:
    try:
        frames = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("frames must be comma-separated integers") from error
    if not frames or min(frames) < 1 or len(set(frames)) != len(frames):
        raise argparse.ArgumentTypeError("frames must be distinct positive integers")
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=OBJECT_NAMES,
        action="append",
        help="Collect only the requested target(s); repeat for more than one.",
    )
    parser.add_argument(
        "--active-arm",
        choices=("left", "right"),
        help=(
            "Keep only policy states whose target geometry selects the requested "
            "arm. This is an offline sampling filter; it does not expose arm truth "
            "to the deployed policy."
        ),
    )
    parser.add_argument(
        "--takeover-frames",
        type=parse_frames,
        default=(5, 8, 12, 15, 30),
        help=(
            "Observed policy frames eligible for recovery. The collector keeps only "
            "the latest object-stable candidate from each rollout."
        ),
    )
    parser.add_argument(
        "--max-object-drift-m",
        type=float,
        default=0.0018,
        help=(
            "Maximum displacement of any scene object from its rollout-start pose "
            "for a policy state to be eligible as recovery supervision."
        ),
    )
    parser.add_argument("--max-recoveries-per-target", type=int, default=6)
    parser.add_argument(
        "--takeover-selection",
        choices=("latest_stable", "closest_finger"),
        default="latest_stable",
        help=(
            "Choose either the latest requested object-stable frame, or the "
            "object-stable frame with the smallest physical finger-to-target distance."
        ),
    )
    parser.add_argument(
        "--all-takeover-frames",
        action="store_true",
        help=(
            "Collect one strict recovery for every eligible requested frame. "
            "Disabled by default so ordinary DAgger collection remains one state "
            "per failed rollout."
        ),
    )
    parser.add_argument(
        "--direct-continuation",
        action="store_true",
        help="Use the offline teacher's direct continuation instead of resetting home.",
    )
    parser.add_argument("--direct-grasp-duration-s", type=float, default=0.9)
    parser.add_argument("--red-direct-grasp-duration-s", type=float)
    parser.add_argument("--yellow-direct-grasp-duration-s", type=float)
    parser.add_argument("--green-direct-grasp-duration-s", type=float)
    parser.add_argument(
        "--adaptive-regrasp",
        action="store_true",
        help=(
            "Ask the offline strict teacher to recompute its lift/place poses "
            "from the physically realized contact pose. This is label generation "
            "only; it is never enabled in VLA deployment."
        ),
    )
    parser.add_argument("--red-triangle-mesh-collision", action="store_true")
    parser.add_argument("--initial-state-recovery-s", type=float, default=1.2)
    parser.add_argument("--initial-home-settle-s", type=float, default=1.0)
    parser.add_argument(
        "--safe-waypoint-recovery",
        action="store_true",
        help="Raise both palms, traverse above home, then descend before the expert task.",
    )
    parser.add_argument("--safe-lift-s", type=float, default=0.8)
    parser.add_argument("--safe-traverse-s", type=float, default=0.8)
    parser.add_argument("--safe-lift-height-m", type=float, default=0.18)
    parser.add_argument("--recovery-lift-duration-s", type=float, default=2.0)
    parser.add_argument("--recovery-lift-hold-s", type=float, default=0.0)
    parser.add_argument("--recovery-initial-lift-stage-height-m", type=float, default=0.0)
    parser.add_argument("--recovery-initial-lift-stage-duration-s", type=float, default=0.45)
    parser.add_argument("--recovery-initial-lift-stage-hold-s", type=float, default=0.20)
    parser.add_argument("--recovery-transport-duration-s", type=float, default=2.5)
    parser.add_argument("--recovery-hand-gain-scale", type=float)
    parser.add_argument("--recovery-closure-multiplier-scale", type=float, default=1.0)
    parser.add_argument("--transport-relative-pose-feedback", action="store_true")
    parser.add_argument("--transport-feedback-hz", type=float, default=20.0)
    parser.add_argument("--transport-feedback-gain", type=float, default=0.65)
    parser.add_argument("--transport-max-correction-m", type=float, default=0.04)
    parser.add_argument("--transport-max-relative-error-m", type=float, default=0.06)
    parser.add_argument("--transport-contact-grace-s", type=float, default=0.08)
    parser.add_argument("--transport-contact-closure-feedback", action="store_true")
    parser.add_argument("--transport-contact-closure-max-boost", type=float, default=0.08)
    parser.add_argument("--transport-contact-closure-rate-per-s", type=float, default=0.05)
    parser.add_argument("--transport-contact-closure-decay-per-s", type=float, default=0.025)
    parser.add_argument("--duration-s", type=float, default=14.0)
    parser.add_argument("--render-skip", type=int, default=3)
    parser.add_argument("--resume-successful", action="store_true")
    return parser.parse_args()


def strict_success(summary: dict | None) -> bool:
    audit = (summary or {}).get("grasp_physics_audit", {})
    return bool(
        summary
        and summary.get("passed")
        and summary.get("grasp_mode") == "contact"
        and audit.get("assisted_constraint_active_steps", 1) == 0
        and audit.get("physical_contact_steps", 0) > 0
        and float(audit.get("target_lift_m", 0.0)) >= 0.10
        and summary.get("target_in_box")
    )


def source_is_policy_failure(summary: dict) -> bool:
    """Only consume states from an observed no-grasp rollout, never successes."""
    return bool(
        summary.get("grabbed_object") is None
        and int(summary.get("hand_object_contact_candidate_steps", 0)) == 0
        and not summary.get("task_success")
    )


def infer_slot_layout(positions: np.ndarray) -> tuple[tuple[int, int, int], np.ndarray, float]:
    """Recover the slot permutation and continuous jitter from rollout geometry."""
    if positions.shape != (len(OBJECT_NAMES), 3):
        raise ValueError(f"Expected object positions (3, 3), found {positions.shape}")
    xy = positions[:, :2]
    best_permutation = None
    best_cost = np.inf
    for permutation in itertools.permutations(range(len(OBJECT_NAMES))):
        cost = sum(
            float(np.linalg.norm(xy[index] - OBJECT_SLOTS_XY_M[slot]))
            for index, slot in enumerate(permutation)
        )
        if cost < best_cost:
            best_cost = cost
            best_permutation = permutation
    assert best_permutation is not None
    offsets = np.stack(
        [xy[index] - OBJECT_SLOTS_XY_M[slot] for index, slot in enumerate(best_permutation)]
    )
    max_error = float(
        max(
            np.linalg.norm(xy[index] - (OBJECT_SLOTS_XY_M[slot] + offsets[index]))
            for index, slot in enumerate(best_permutation)
        )
    )
    return best_permutation, offsets, max_error


def persisted_production_layout(summary: dict) -> tuple[tuple[int, int, int], np.ndarray] | None:
    """Use the evaluator's exact generation contract when it is available."""
    spec = summary.get("production_spec")
    if not isinstance(spec, dict):
        return None
    permutation = spec.get("slot_permutation")
    offsets = spec.get("slot_offsets_xy_m")
    if (
        not isinstance(permutation, list)
        or sorted(permutation) != [0, 1, 2]
        or np.asarray(offsets).shape != (3, 2)
    ):
        return None
    return tuple(int(value) for value in permutation), np.asarray(offsets, dtype=np.float64)


def closest_finger_frame(
    source_dir: Path,
    names: list[str],
    states: np.ndarray,
    positions: np.ndarray,
    target_index: int,
    eligible: list[int],
) -> tuple[int, float]:
    """Select an offline teacher state near the target without changing deployment inputs."""
    scene = source_dir / "scene.xml"
    if not scene.is_file():
        raise FileNotFoundError(scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos_ids = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Unknown rollout joint {name} in {scene}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
    geom_ids = finger_contact_geom_ids(model)
    best_frame = -1
    best_distance = np.inf
    for frame in eligible:
        data.qpos[qpos_ids] = states[frame]
        mujoco.mj_forward(model, data)
        sites = finger_contact_sites(model, data, geom_ids).reshape(-1, 3)
        distance = float(np.linalg.norm(sites - positions[frame, target_index], axis=-1).min())
        if distance < best_distance:
            best_frame, best_distance = frame, distance
    if best_frame < 0:
        raise RuntimeError(f"No closest-finger candidate in {source_dir}")
    return best_frame, best_distance


def source_jobs(args: argparse.Namespace) -> list[dict]:
    jobs = []
    by_target = {name: 0 for name in OBJECT_NAMES}
    requested_targets = set(args.target or OBJECT_NAMES)
    for source_dir in sorted(args.rollout_root.glob("episode_*")):
        summary_path = source_dir / "summary.json"
        trajectory_path = source_dir / "trajectory.npz"
        if not summary_path.is_file() or not trajectory_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        target = summary.get("target_object")
        if (
            target not in OBJECT_NAMES
            or target not in requested_targets
            or not source_is_policy_failure(summary)
        ):
            continue
        with np.load(trajectory_path) as values:
            state = np.asarray(values["observation_joint_position_rad"], dtype=np.float32)
            velocity = np.asarray(values["observation_joint_velocity_rad_s"], dtype=np.float32)
            positions = np.asarray(values["object_position_m"], dtype=np.float64)
            names = values["joint_names"].tolist()
        if state.ndim != 2 or state.shape[1] != 31 or velocity.shape != state.shape:
            raise ValueError(f"Unexpected policy state contract in {trajectory_path}")
        if len(names) != 31:
            raise ValueError(f"Expected 31 joint names in {trajectory_path}")
        persisted = persisted_production_layout(summary)
        if persisted is None:
            permutation, offsets, reconstruction_error = infer_slot_layout(positions[0])
            layout_source = "inferred_from_rollout_initial_positions"
        else:
            permutation, offsets = persisted
            expected_xy = np.stack(
                [OBJECT_SLOTS_XY_M[slot] + offsets[index] for index, slot in enumerate(permutation)]
            )
            reconstruction_error = float(
                np.max(np.linalg.norm(positions[0, :, :2] - expected_xy, axis=-1))
            )
            layout_source = "persisted_production_spec"
        object_drift_m = np.max(
            np.linalg.norm(positions - positions[:1], axis=-1), axis=1
        )
        if args.takeover_selection == "closest_finger":
            eligible = np.flatnonzero(
                object_drift_m <= args.max_object_drift_m
            ).astype(int).tolist()
        else:
            eligible = [
                frame
                for frame in args.takeover_frames
                if frame < len(state) and object_drift_m[frame] <= args.max_object_drift_m
            ]
        if not eligible:
            print(
                "POLICY_STATE_RECOVERY_SKIP="
                + json.dumps(
                    {
                        "source_episode": source_dir.name,
                        "reason": "no_object_stable_requested_frame",
                        "candidate_frames": list(args.takeover_frames),
                        "max_object_drift_mm": args.max_object_drift_m * 1e3,
                    }
                ),
                flush=True,
            )
            continue
        target_index = OBJECT_NAMES.index(target)
        if args.takeover_selection == "closest_finger":
            frame, finger_distance = closest_finger_frame(
                source_dir, names, state, positions, target_index, eligible
            )
            candidates = [(frame, finger_distance)]
        elif args.all_takeover_frames:
            candidates = [(frame, None) for frame in eligible]
        else:
            candidates = [(max(eligible), None)]
        if args.active_arm is not None:
            # The production evaluator selects the arm from the target object's
            # world-y coordinate. The rollout stores object poses only as an
            # offline teacher label, so using it here is a sampling filter and
            # never a deployment input.
            candidates = [
                (frame, distance)
                for frame, distance in candidates
                if ("left" if positions[frame, target_index, 1] >= 0.05 else "right")
                == args.active_arm
            ]
            if not candidates:
                continue
        target_slot = permutation[target_index]
        for frame, finger_distance in candidates:
            jobs.append(
                {
                    "source_dir": source_dir,
                    "source_episode": source_dir.name,
                    "target_object": target,
                    "instruction": summary["language_instruction"],
                    "source_frame": frame,
                    "source_object_drift_m": float(object_drift_m[frame]),
                    "eligible_source_frames": eligible,
                    "takeover_selection": (
                        "all_requested_stable_frames"
                        if args.all_takeover_frames
                        else args.takeover_selection
                    ),
                    "source_finger_target_distance_m": finger_distance,
                    "joint_names": names,
                    "state": state[frame],
                    "velocity": velocity[frame],
                    "source_object_positions_m": positions[frame],
                    "slot_permutation": permutation,
                    "slot_offsets_xy_m": offsets,
                    "target_slot": target_slot,
                    "layout_reconstruction_error_m": reconstruction_error,
                    "layout_source": layout_source,
                }
            )
    if args.takeover_selection == "closest_finger":
        jobs.sort(
            key=lambda item: (
                item["target_object"],
                float(item["source_finger_target_distance_m"]),
                item["source_episode"],
            )
        )
    selected = []
    for job in jobs:
        target = job["target_object"]
        if by_target[target] >= args.max_recoveries_per_target:
            continue
        selected.append(job)
        by_target[target] += 1
    jobs = selected
    missing = [target for target in requested_targets if by_target[target] == 0]
    if missing:
        raise RuntimeError(f"No failed policy states collected for targets: {missing}")
    return jobs


def recovery_payload(args: argparse.Namespace, job: dict) -> dict:
    target_duration = {
        "red_triangle": args.red_direct_grasp_duration_s,
        "yellow_rod": args.yellow_direct_grasp_duration_s,
        "green_cube": args.green_direct_grasp_duration_s,
    }[job["target_object"]]
    return {
        "recovery_contract_version": RECOVERY_CONTRACT_VERSION,
        "source_rollout": str(job["source_dir"]),
        "source_episode": job["source_episode"],
        "source_frame": job["source_frame"],
        "source_object_drift_m": job["source_object_drift_m"],
        "eligible_source_frames": job["eligible_source_frames"],
        "takeover_selection": job["takeover_selection"],
        "source_finger_target_distance_m": job["source_finger_target_distance_m"],
        "target_object": job["target_object"],
        "instruction": job["instruction"],
        "slot_permutation": list(job["slot_permutation"]),
        "slot_offsets_xy_m": job["slot_offsets_xy_m"].tolist(),
        "target_slot": job["target_slot"],
        "layout_reconstruction_error_m": job["layout_reconstruction_error_m"],
        "layout_source": job["layout_source"],
        "input_contract": "policy state31 + language + three RGB cameras; simulator object state is offline teacher-only",
        "labels": "strict contact-only MuJoCo direct-continuation actions",
        "offline_teacher_adaptive_regrasp": bool(args.adaptive_regrasp),
        "red_triangle_mesh_collision": bool(args.red_triangle_mesh_collision),
        "safe_waypoint_recovery": bool(args.safe_waypoint_recovery),
        "safe_lift_s": args.safe_lift_s if args.safe_waypoint_recovery else None,
        "safe_traverse_s": args.safe_traverse_s if args.safe_waypoint_recovery else None,
        "safe_lift_height_m": args.safe_lift_height_m if args.safe_waypoint_recovery else None,
        "recovery_motion_profile": {
            "lift_duration_s": args.recovery_lift_duration_s,
            "lift_hold_s": args.recovery_lift_hold_s,
            "initial_lift_stage_height_m": args.recovery_initial_lift_stage_height_m,
            "initial_lift_stage_duration_s": args.recovery_initial_lift_stage_duration_s,
            "initial_lift_stage_hold_s": args.recovery_initial_lift_stage_hold_s,
            "transport_duration_s": args.recovery_transport_duration_s,
            "hand_gain_scale": args.recovery_hand_gain_scale,
            "closure_multiplier_scale": args.recovery_closure_multiplier_scale,
            "transport_relative_pose_feedback": args.transport_relative_pose_feedback,
            "transport_feedback_hz": args.transport_feedback_hz,
            "transport_feedback_gain": args.transport_feedback_gain,
            "transport_max_correction_m": args.transport_max_correction_m,
            "transport_max_relative_error_m": args.transport_max_relative_error_m,
            "transport_contact_grace_s": args.transport_contact_grace_s,
            "transport_contact_closure_feedback": args.transport_contact_closure_feedback,
            "transport_contact_closure_max_boost": args.transport_contact_closure_max_boost,
            "transport_contact_closure_rate_per_s": args.transport_contact_closure_rate_per_s,
            "transport_contact_closure_decay_per_s": args.transport_contact_closure_decay_per_s,
        },
        "direct_grasp_duration_s": (
            float(args.direct_grasp_duration_s if target_duration is None else target_duration)
            if args.direct_continuation
            else None
        ),
    }


def collect_one(args: argparse.Namespace, index: int, job: dict) -> dict:
    output_dir = args.output_dir / f"recovery_{index:04d}_{job['target_object']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "recovery_spec.json"
    summary_path = output_dir / "language_pick_place_summary.json"
    payload = recovery_payload(args, job)
    existing = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    if (
        args.resume_successful
        and spec_path.is_file()
        and json.loads(spec_path.read_text()) == payload
        and strict_success(existing)
    ):
        return {**payload, "output_dir": str(output_dir), "passed": True, "reused": True}

    state_path = output_dir / "policy_visited_state.npz"
    np.savez_compressed(
        state_path,
        observation_joint_position_rad=job["state"][None],
        observation_joint_velocity_rad_s=job["velocity"][None],
        joint_names=np.asarray(job["joint_names"]),
        source_object_position_m=job["source_object_positions_m"],
    )
    profile = contact_profile_for_target_slot(job["target_object"], job["target_slot"])
    hand_profile_scale = "1.0,1.0,1.0"
    if job["target_object"] == "yellow_rod":
        # Validated by 75 strict slot-1 successes in the bridge/contact corpus.
        profile.update(
            grasp_yaw_deg=90.0,
            contact_grasp_compensation="0.015,-0.050,-0.055",
            hand_closure_multiplier=1.05,
        )
        hand_profile_scale = "1.0,1.04,1.04"
    profile["hand_closure_multiplier"] *= args.recovery_closure_multiplier_scale
    if args.recovery_hand_gain_scale is not None:
        profile["hand_gain_scale"] = args.recovery_hand_gain_scale
    target_duration = {
        "red_triangle": args.red_direct_grasp_duration_s,
        "yellow_rod": args.yellow_direct_grasp_duration_s,
        "green_cube": args.green_direct_grasp_duration_s,
    }[job["target_object"]]
    direct_grasp_duration_s = float(
        args.direct_grasp_duration_s if target_duration is None else target_duration
    )
    command = [
        str(args.python),
        str(args.expert),
        "--asset", str(args.asset.resolve()),
        "--output-dir", str(output_dir),
        "--initial-upper-state-npz", str(state_path),
        "--target-object", job["target_object"],
        "--slot-permutation=" + ",".join(str(value) for value in job["slot_permutation"]),
        "--slot-offsets=" + ",".join(
            f"{value:.8f}" for pair in job["slot_offsets_xy_m"] for value in pair
        ),
        "--language-instruction", job["instruction"],
        "--record-demonstration",
        "--disable-overview-video",
        "--duration-s", str(args.duration_s),
        "--initial-state-recovery-s", str(0.0 if args.direct_continuation else args.initial_state_recovery_s),
        "--initial-home-settle-s", str(0.0 if args.direct_continuation else args.initial_home_settle_s),
        "--initial-velocity-mode", "zero",
        "--render-skip", str(args.render_skip),
        "--allow-occluded-initial-head-view",
        "--grasp-mode", "contact",
        "--body-gain-scale", str(profile["body_gain_scale"]),
        "--hand-gain-scale", str(profile["hand_gain_scale"]),
        "--grasp-yaw-deg", str(profile["grasp_yaw_deg"]),
        "--contact-grasp-compensation=" + profile["contact_grasp_compensation"],
        "--hand-closure-multiplier", str(profile["hand_closure_multiplier"]),
        "--hand-profile-scale", hand_profile_scale,
        "--lift-duration-s", str(args.recovery_lift_duration_s),
        "--lift-hold-duration-s", str(args.recovery_lift_hold_s),
        "--initial-lift-stage-height-m", str(args.recovery_initial_lift_stage_height_m),
        "--initial-lift-stage-duration-s", str(args.recovery_initial_lift_stage_duration_s),
        "--initial-lift-stage-hold-s", str(args.recovery_initial_lift_stage_hold_s),
        "--transport-duration-s", str(args.recovery_transport_duration_s),
    ]
    if args.direct_continuation:
        command.extend(
            (
                "--initial-direct-continuation",
                "--initial-direct-grasp-s", str(direct_grasp_duration_s),
            )
        )
    if args.adaptive_regrasp:
        command.append("--adaptive-regrasp")
    if args.red_triangle_mesh_collision:
        command.append("--red-triangle-mesh-collision")
    if args.safe_waypoint_recovery:
        command.extend(
            (
                "--initial-safe-lift-s", str(args.safe_lift_s),
                "--initial-safe-traverse-s", str(args.safe_traverse_s),
                "--initial-safe-lift-height-m", str(args.safe_lift_height_m),
            )
        )
    if args.transport_relative_pose_feedback:
        command.extend(
            (
                "--transport-relative-pose-feedback",
                "--transport-feedback-hz", str(args.transport_feedback_hz),
                "--transport-feedback-gain", str(args.transport_feedback_gain),
                "--transport-max-correction-m", str(args.transport_max_correction_m),
                "--transport-max-relative-error-m", str(args.transport_max_relative_error_m),
                "--transport-contact-grace-s", str(args.transport_contact_grace_s),
            )
        )
        if args.transport_contact_closure_feedback:
            command.extend(
                (
                    "--transport-contact-closure-feedback",
                    "--transport-contact-closure-max-boost", str(args.transport_contact_closure_max_boost),
                    "--transport-contact-closure-rate-per-s", str(args.transport_contact_closure_rate_per_s),
                    "--transport-contact-closure-decay-per-s", str(args.transport_contact_closure_decay_per_s),
                )
            )
    completed = subprocess.run(command, check=False, env={**os.environ, "MUJOCO_GL": "egl"})
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    passed = strict_success(summary)
    spec_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        **payload,
        "output_dir": str(output_dir),
        "returncode": completed.returncode,
        "direct_grasp_duration_s": direct_grasp_duration_s if args.direct_continuation else None,
        "passed": passed,
        "reused": False,
        "physical_contact_steps": (summary or {}).get("grasp_physics_audit", {}).get("physical_contact_steps", 0),
        "target_lift_m": (summary or {}).get("grasp_physics_audit", {}).get("target_lift_m", 0.0),
        "target_in_box": (summary or {}).get("target_in_box", False),
        "assisted_constraint_active_steps": (summary or {}).get("grasp_physics_audit", {}).get("assisted_constraint_active_steps", None),
    }


def main() -> None:
    args = parse_args()
    if (
        args.max_recoveries_per_target < 1
        or args.render_skip < 1
        or args.max_object_drift_m <= 0.0
    ):
        raise ValueError("recovery count and render skip must be positive")
    if args.duration_s < 12.0 or args.initial_home_settle_s < 0.0:
        raise ValueError("recovery duration must be positive and total duration at least 12 s")
    if args.direct_continuation:
        if args.direct_grasp_duration_s <= 0.0:
            raise ValueError("direct continuation requires a positive grasp duration")
    elif args.initial_state_recovery_s <= 0.0:
        raise ValueError("home-reset recovery requires a positive recovery duration")
    if args.safe_waypoint_recovery and (
        args.direct_continuation
        or args.safe_lift_s <= 0.0
        or args.safe_traverse_s <= 0.0
        or args.safe_lift_height_m <= 0.0
    ):
        raise ValueError("safe waypoint recovery requires positive segments and cannot be direct")
    if (
        args.recovery_lift_duration_s <= 0.0
        or args.recovery_lift_hold_s < 0.0
        or args.recovery_initial_lift_stage_height_m < 0.0
        or args.recovery_initial_lift_stage_duration_s <= 0.0
        or args.recovery_initial_lift_stage_hold_s < 0.0
        or args.recovery_transport_duration_s <= 0.0
        or args.recovery_closure_multiplier_scale <= 0.0
        or (args.recovery_hand_gain_scale is not None and args.recovery_hand_gain_scale <= 0.0)
        or args.transport_feedback_hz <= 0.0
        or not 0.0 <= args.transport_feedback_gain <= 1.0
        or args.transport_max_correction_m <= 0.0
        or args.transport_max_relative_error_m <= 0.0
        or args.transport_contact_grace_s < 0.0
        or args.transport_contact_closure_max_boost < 0.0
        or args.transport_contact_closure_rate_per_s <= 0.0
        or args.transport_contact_closure_decay_per_s <= 0.0
    ):
        raise ValueError("recovery motion durations, gains, and closure scale must be positive")
    if not args.asset.is_file() or not args.expert.is_file() or not args.python.is_file():
        raise FileNotFoundError("asset, expert, and Python executable must exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    requested_targets = set(args.target or OBJECT_NAMES)
    reports = []
    for index, job in enumerate(source_jobs(args)):
        report = collect_one(args, index, job)
        reports.append(report)
        print(
            "POLICY_STATE_RECOVERY=" + json.dumps(
                {
                    "index": index,
                    "target": report["target_object"],
                    "source_frame": report["source_frame"],
                    "passed": report["passed"],
                    "reused": report["reused"],
                }
            ),
            flush=True,
        )
    by_target = {
        target: sum(item["passed"] and item["target_object"] == target for item in reports)
        for target in OBJECT_NAMES
    }
    summary = {
        "experiment": "G1LANG-reachable-policy-state-strict-recovery-dataset",
        "recovery_contract_version": RECOVERY_CONTRACT_VERSION,
        "source_rollout_root": str(args.rollout_root),
        "requested_targets": sorted(requested_targets),
        "requested_policy_states": len(reports),
        "strict_recovery_successes": sum(item["passed"] for item in reports),
        "strict_recovery_success_rate": float(np.mean([item["passed"] for item in reports])),
        "strict_successes_by_target": by_target,
        "takeover_frames": list(args.takeover_frames),
        "takeover_selection": args.takeover_selection,
        "all_takeover_frames": args.all_takeover_frames,
        "direct_continuation": args.direct_continuation,
        "direct_grasp_duration_s": (
            {
                "default": args.direct_grasp_duration_s,
                "red_triangle": args.red_direct_grasp_duration_s,
                "yellow_rod": args.yellow_direct_grasp_duration_s,
                "green_cube": args.green_direct_grasp_duration_s,
            }
            if args.direct_continuation
            else None
        ),
        "offline_teacher_adaptive_regrasp": bool(args.adaptive_regrasp),
        "red_triangle_mesh_collision": bool(args.red_triangle_mesh_collision),
        "safe_waypoint_recovery": bool(args.safe_waypoint_recovery),
        "safe_waypoint": (
            {
                "lift_s": args.safe_lift_s,
                "traverse_s": args.safe_traverse_s,
                "lift_height_m": args.safe_lift_height_m,
            }
            if args.safe_waypoint_recovery
            else None
        ),
        "recovery_motion_profile": {
            "lift_duration_s": args.recovery_lift_duration_s,
            "lift_hold_s": args.recovery_lift_hold_s,
            "transport_duration_s": args.recovery_transport_duration_s,
            "hand_gain_scale": args.recovery_hand_gain_scale,
            "closure_multiplier_scale": args.recovery_closure_multiplier_scale,
            "transport_relative_pose_feedback": args.transport_relative_pose_feedback,
            "transport_feedback_hz": args.transport_feedback_hz,
            "transport_feedback_gain": args.transport_feedback_gain,
            "transport_max_correction_m": args.transport_max_correction_m,
            "transport_max_relative_error_m": args.transport_max_relative_error_m,
            "transport_contact_grace_s": args.transport_contact_grace_s,
            "transport_contact_closure_feedback": args.transport_contact_closure_feedback,
            "transport_contact_closure_max_boost": args.transport_contact_closure_max_boost,
            "transport_contact_closure_rate_per_s": args.transport_contact_closure_rate_per_s,
            "transport_contact_closure_decay_per_s": args.transport_contact_closure_decay_per_s,
        },
        "max_object_drift_m": args.max_object_drift_m,
        "selection_policy": "one latest object-stable candidate per source rollout",
        "initial_state_recovery_s": 0.0 if args.direct_continuation else args.initial_state_recovery_s,
        "initial_home_settle_s": 0.0 if args.direct_continuation else args.initial_home_settle_s,
        "initial_velocity_mode": "zero",
        "recovery_duration_s": args.duration_s,
        "deployment_input_contract": "RGB head/left-wrist/right-wrist + language + state31",
        "teacher_privilege": "object pose is used only inside offline strict expert/physical audit",
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "reports": reports,
    }
    summary["passed"] = bool(
        summary["strict_recovery_success_rate"] >= 0.90
        and all(by_target[target] >= 1 for target in requested_targets)
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
