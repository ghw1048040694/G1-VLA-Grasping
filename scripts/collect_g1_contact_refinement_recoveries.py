#!/usr/bin/env python3
"""Collect short-horizon contact-refinement recoveries from near-object states.

This is deliberately narrower than full pick-place data collection.  It samples
states where a strict expert's active hand is already near the language target,
then asks the same strict contact-only teacher to continue directly into
contact, closure, and lift.  The retained demonstrations are meant to train the
missing final 5-15 cm contact-refinement policy; placement success is recorded
but is not required.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from collect_g1_language_pick_place_dataset import (
    LANGUAGE_VARIANTS,
    contact_profile_for_target_slot,
)
from run_g1_language_pick_place import OBJECT_NAMES, OBJECT_SLOTS_XY_M


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-object", choices=OBJECT_NAMES, action="append")
    parser.add_argument("--samples-per-target", type=int, default=3)
    parser.add_argument(
        "--candidate-start-rank",
        type=int,
        default=0,
        help=(
            "Skip the first N sorted near-contact candidates per target. "
            "Use this when expanding the dataset so later runs do not just recollect the same easiest frames."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--min-distance-m", type=float, default=0.045)
    parser.add_argument("--max-distance-m", type=float, default=0.165)
    parser.add_argument("--max-object-drift-m", type=float, default=0.030)
    parser.add_argument("--state-noise-rad", type=float, default=0.010)
    parser.add_argument("--velocity-noise-rad-s", type=float, default=0.0)
    parser.add_argument(
        "--use-source-frame-object-positions",
        action="store_true",
        help=(
            "Reconstruct the MuJoCo scene with the object XY positions observed "
            "at the selected source frame, preserving near-contact hand-object geometry."
        ),
    )
    parser.add_argument("--direct-grasp-duration-s", type=float, default=0.55)
    parser.add_argument("--duration-s", type=float, default=6.5)
    parser.add_argument("--render-skip", type=int, default=10)
    parser.add_argument("--adaptive-regrasp", action="store_true")
    parser.add_argument("--override-grasp-yaw-deg", type=float)
    parser.add_argument("--override-contact-grasp-compensation")
    parser.add_argument("--override-hand-closure-multiplier", type=float)
    parser.add_argument("--override-hand-profile-scale")
    parser.add_argument("--override-hand-gain-scale", type=float)
    parser.add_argument("--resume-successful", action="store_true")
    return parser.parse_args()


def physical_grasp_success(summary: dict | None) -> bool:
    audit = (summary or {}).get("grasp_physics_audit", {})
    return bool(
        summary
        and summary.get("grasp_mode") == "contact"
        and audit.get("assisted_constraint_active_steps", 1) == 0
        and int(audit.get("physical_contact_steps", 0)) > 0
        and bool(audit.get("physical_grasp_passed", False))
        and float(audit.get("target_lift_m", 0.0)) >= 0.10
    )


def nearest_profile_slot(position_xy: np.ndarray) -> int:
    return int(np.argmin([np.linalg.norm(position_xy - slot) for slot in OBJECT_SLOTS_XY_M]))


def load_source_candidates(episode_dir: Path, args: argparse.Namespace) -> list[dict]:
    npz_path = episode_dir / "expert_pick_place_episode.npz"
    summary_path = episode_dir / "language_pick_place_summary.json"
    if not npz_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Missing expert NPZ or summary in {episode_dir}")
    summary = json.loads(summary_path.read_text())
    z = np.load(npz_path, allow_pickle=True)
    try:
        joint_names = z["joint_names"].astype(str).tolist()
        states = z["joint_position_rad"].astype(np.float32)[:, 12:]
        velocities = z["joint_velocity_rad_s"].astype(np.float32)[:, 12:]
        phases = z["task_phase"].astype(np.int64)
        contacts = z["physical_contact_active"].astype(np.float32)
        objects = z["object_position_m"].astype(np.float32)
        palms = z["palm_position_m"].astype(np.float32)
        arms = z["active_arm_index"].astype(np.int64)
        targets = z["target_object_index"].astype(np.int64)
    finally:
        z.close()
    target = str(summary["target_object"])
    target_index = OBJECT_NAMES.index(target)
    if not bool(summary.get("grasp_physics_audit", {}).get("physical_grasp_passed", False)):
        return []
    if not np.all(targets == target_index):
        raise ValueError(f"Target index changed inside {episode_dir}")
    slot_permutation = tuple(int(value) for value in summary.get("slot_permutation", [0, 1, 2]))
    offsets = np.asarray(summary.get("slot_offsets_xy_m", [[0, 0], [0, 0], [0, 0]]), dtype=np.float64)
    if sorted(slot_permutation) != [0, 1, 2] or offsets.shape != (3, 2):
        raise ValueError(f"Unexpected slot layout in {episode_dir}")
    drift = np.linalg.norm(objects[:, target_index] - objects[0, target_index], axis=-1)
    active_palm = palms[np.arange(len(palms)), arms]
    distance = np.linalg.norm(active_palm - objects[:, target_index], axis=-1)
    valid = np.flatnonzero(
        ((phases == 2) | (phases == 3))
        & (distance >= args.min_distance_m)
        & (distance <= args.max_distance_m)
        & (drift <= args.max_object_drift_m)
    )
    result = []
    for frame in valid.tolist():
        result.append(
            {
                "source_episode_dir": str(episode_dir),
                "source_frame": int(frame),
                "target_object": target,
                "language_instruction": LANGUAGE_VARIANTS[target][frame % len(LANGUAGE_VARIANTS[target])],
                "state": states[frame],
                "velocity": velocities[frame],
                "joint_names": joint_names[12:],
                "source_phase": int(phases[frame]),
                "source_contact": float(contacts[frame]),
                "source_palm_target_distance_m": float(distance[frame]),
                "source_object_drift_m": float(drift[frame]),
                "source_object_positions_m": objects[frame],
                "slot_permutation": slot_permutation,
                "slot_offsets_xy_m": offsets,
                "profile_slot": nearest_profile_slot(objects[0, target_index, :2]),
            }
        )
    return result


def build_jobs(args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed)
    by_target: dict[str, list[dict]] = {target: [] for target in OBJECT_NAMES}
    for episode_dir in args.episode_dir:
        for candidate in load_source_candidates(episode_dir, args):
            by_target[candidate["target_object"]].append(candidate)
    jobs = []
    requested_targets = tuple(args.target_object) if args.target_object else OBJECT_NAMES
    for target, candidates in by_target.items():
        if target not in requested_targets:
            continue
        if not candidates:
            continue
        # Prefer non-contact close approach frames first; then contact-hold frames.
        candidates.sort(
            key=lambda item: (
                int(item["source_contact"] > 0.5),
                abs(item["source_palm_target_distance_m"] - 0.085),
                item["source_object_drift_m"],
            )
        )
        selected = candidates[
            args.candidate_start_rank : args.candidate_start_rank + args.samples_per_target
        ]
        for index, item in enumerate(selected):
            noisy = dict(item)
            state_noise = rng.normal(0.0, args.state_noise_rad, size=item["state"].shape).astype(np.float32)
            velocity_noise = rng.normal(0.0, args.velocity_noise_rad_s, size=item["velocity"].shape).astype(np.float32)
            # Keep perturbations small and away from joint-limit-scale jumps.  This
            # is recovery data, not exploration by violent randomization.
            noisy["state"] = item["state"] + state_noise
            noisy["velocity"] = item["velocity"] + velocity_noise
            noisy["target_sample_index"] = index
            jobs.append(noisy)
    missing = [target for target in requested_targets if not by_target[target]]
    if missing:
        raise RuntimeError(f"No near-contact physical-grasp source frames for targets: {missing}")
    return jobs


def collect_one(args: argparse.Namespace, index: int, job: dict) -> dict:
    output_dir = args.output_dir / f"contact_refine_{index:04d}_{job['target_object']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "contact_refinement_spec.json"
    summary_path = output_dir / "language_pick_place_summary.json"
    slot_offsets_xy_m = np.asarray(job["slot_offsets_xy_m"], dtype=np.float64)
    layout_source = "episode_initial_layout"
    if args.use_source_frame_object_positions:
        slot_offsets_xy_m = np.asarray(
            [
                np.asarray(job["source_object_positions_m"])[object_index, :2]
                - OBJECT_SLOTS_XY_M[slot]
                for object_index, slot in enumerate(job["slot_permutation"])
            ],
            dtype=np.float64,
        )
        layout_source = "source_frame_object_positions"
    payload = {
        "contract": "g1lang_contact_refinement_v1",
        "source_episode_dir": job["source_episode_dir"],
        "source_frame": job["source_frame"],
        "source_phase": job["source_phase"],
        "source_contact": job["source_contact"],
        "source_palm_target_distance_m": job["source_palm_target_distance_m"],
        "source_object_drift_m": job["source_object_drift_m"],
        "target_object": job["target_object"],
        "language_instruction": job["language_instruction"],
        "slot_permutation": list(job["slot_permutation"]),
        "slot_offsets_xy_m": slot_offsets_xy_m.tolist(),
        "layout_source_for_recovery_scene": layout_source,
        "profile_slot": job["profile_slot"],
        "state_noise_rad": args.state_noise_rad,
        "velocity_noise_rad_s": args.velocity_noise_rad_s,
        "direct_grasp_duration_s": args.direct_grasp_duration_s,
        "duration_s": args.duration_s,
        "objective": "short-horizon contact/closure/lift recovery; placement is not required",
    }
    existing = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    if (
        args.resume_successful
        and spec_path.is_file()
        and json.loads(spec_path.read_text()) == payload
        and physical_grasp_success(existing)
    ):
        return {**payload, "output_dir": str(output_dir), "passed": True, "reused": True}
    state_path = output_dir / "near_contact_initial_state.npz"
    np.savez_compressed(
        state_path,
        observation_joint_position_rad=np.asarray(job["state"], dtype=np.float32)[None],
        observation_joint_velocity_rad_s=np.asarray(job["velocity"], dtype=np.float32)[None],
        joint_names=np.asarray(job["joint_names"]),
    )
    profile = contact_profile_for_target_slot(job["target_object"], int(job["profile_slot"]))
    if args.override_grasp_yaw_deg is not None:
        profile["grasp_yaw_deg"] = args.override_grasp_yaw_deg
    if args.override_contact_grasp_compensation is not None:
        profile["contact_grasp_compensation"] = args.override_contact_grasp_compensation
    if args.override_hand_closure_multiplier is not None:
        profile["hand_closure_multiplier"] = args.override_hand_closure_multiplier
    if args.override_hand_gain_scale is not None:
        profile["hand_gain_scale"] = args.override_hand_gain_scale
    hand_profile_scale = args.override_hand_profile_scale or "1,1,1"
    command = [
        str(args.python),
        str(args.expert),
        "--asset", str(args.asset.resolve()),
        "--output-dir", str(output_dir),
        "--initial-upper-state-npz", str(state_path),
        "--target-object", job["target_object"],
        "--slot-permutation=" + ",".join(str(value) for value in job["slot_permutation"]),
        "--slot-offsets=" + ",".join(
            f"{value:.8f}" for pair in slot_offsets_xy_m for value in pair
        ),
        "--language-instruction", job["language_instruction"],
        "--record-demonstration",
        "--disable-demonstration-videos",
        "--disable-overview-video",
        "--duration-s", str(args.duration_s),
        "--initial-direct-continuation",
        "--initial-direct-grasp-s", str(args.direct_grasp_duration_s),
        "--initial-velocity-mode", "zero",
        "--render-skip", str(args.render_skip),
        "--allow-occluded-initial-head-view",
        "--grasp-mode", "contact",
        "--body-gain-scale", str(profile["body_gain_scale"]),
        "--hand-gain-scale", str(profile["hand_gain_scale"]),
        "--hand-profile-scale", hand_profile_scale,
        "--grasp-yaw-deg", str(profile["grasp_yaw_deg"]),
        "--contact-grasp-compensation=" + profile["contact_grasp_compensation"],
        "--hand-closure-multiplier", str(profile["hand_closure_multiplier"]),
    ]
    if args.adaptive_regrasp:
        command.append("--adaptive-regrasp")
    completed = subprocess.run(command, check=False, env={**os.environ, "MUJOCO_GL": "egl"})
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    passed = physical_grasp_success(summary)
    spec_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    audit = (summary or {}).get("grasp_physics_audit", {})
    return {
        **payload,
        "output_dir": str(output_dir),
        "returncode": completed.returncode,
        "passed": passed,
        "reused": False,
        "physical_contact_steps": audit.get("physical_contact_steps", 0),
        "target_lift_m": audit.get("target_lift_m", 0.0),
        "target_in_box": (summary or {}).get("target_in_box", False),
        "summary_passed_full_pick_place": (summary or {}).get("passed", False),
    }


def main() -> None:
    args = parse_args()
    if (
        args.samples_per_target < 1
        or args.candidate_start_rank < 0
        or args.min_distance_m < 0.0
        or args.max_distance_m <= args.min_distance_m
        or args.max_object_drift_m <= 0.0
        or args.state_noise_rad < 0.0
        or args.velocity_noise_rad_s < 0.0
        or args.direct_grasp_duration_s <= 0.0
        or args.duration_s <= 3.5
        or args.render_skip < 1
    ):
        raise ValueError("Invalid contact-refinement collection arguments")
    if not args.python.is_file() or not args.expert.is_file() or not args.asset.is_file():
        raise FileNotFoundError("Missing Python, expert script, or asset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    reports = []
    for index, job in enumerate(jobs):
        report = collect_one(args, index, job)
        reports.append(report)
        print(
            "CONTACT_REFINEMENT_RECOVERY=" + json.dumps(
                {
                    "index": index,
                    "target": report["target_object"],
                    "source_distance_mm": round(1e3 * report["source_palm_target_distance_m"], 1),
                    "source_contact": report["source_contact"],
                    "passed": report["passed"],
                    "lift_mm": round(1e3 * float(report["target_lift_m"]), 1),
                }
            ),
            flush=True,
        )
    by_target = {
        target: sum(item["passed"] and item["target_object"] == target for item in reports)
        for target in (tuple(args.target_object) if args.target_object else OBJECT_NAMES)
    }
    summary = {
        "experiment": "G1LANG-contact-refinement-near-object-recoveries",
        "contract": "g1lang_contact_refinement_v1",
        "source_episode_dirs": [str(path) for path in args.episode_dir],
        "samples_per_target": args.samples_per_target,
        "candidate_start_rank": args.candidate_start_rank,
        "target_filter": args.target_object,
        "requested_recoveries": len(reports),
        "physical_grasp_successes": int(sum(item["passed"] for item in reports)),
        "physical_grasp_success_rate": float(np.mean([item["passed"] for item in reports])) if reports else 0.0,
        "physical_grasp_successes_by_target": by_target,
        "distance_window_m": [args.min_distance_m, args.max_distance_m],
        "max_object_drift_m": args.max_object_drift_m,
        "state_noise_rad": args.state_noise_rad,
        "velocity_noise_rad_s": args.velocity_noise_rad_s,
        "use_source_frame_object_positions": bool(args.use_source_frame_object_positions),
        "direct_grasp_duration_s": args.direct_grasp_duration_s,
        "duration_s": args.duration_s,
        "offline_teacher_adaptive_regrasp": bool(args.adaptive_regrasp),
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "placement_success_not_required": True,
        "reports": reports,
    }
    summary["passed"] = bool(all(count >= 1 for count in by_target.values()))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
