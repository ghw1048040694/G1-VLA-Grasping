#!/usr/bin/env python3
"""Collect strict experts on the exact independent-random layout distribution.

The closed-loop gate samples all three objects independently over the tabletop
subject to a minimum separation.  Earlier production demonstrations only used
small perturbations around three fixed slots, so they cannot establish a
visual closed-loop contract for the actual gate.  This collector closes that
gap while keeping simulator object positions strictly inside the offline
teacher.  Every retained episode must physically contact, lift, and place the
language target with all assisted constraints disabled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from collect_g1_language_pick_place_dataset import (
    LANGUAGE_VARIANTS,
    contact_profile_for_target_slot,
)
from evaluate_g1_language_pick_place import _sample_random_positions
from run_g1_language_pick_place import OBJECT_NAMES, OBJECT_SLOTS_XY_M


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=OBJECT_NAMES,
        action="append",
        help="Collect only these targets; repeat for red and yellow.",
    )
    parser.add_argument("--scenes", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260749)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Use one under WSL/EGL; multiple renderer subprocesses can corrupt contexts.",
    )
    parser.add_argument("--render-skip", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--lift-hold-duration-s", type=float, default=0.0)
    parser.add_argument("--transport-duration-s", type=float, default=2.5)
    parser.add_argument("--red-lift-height-m", type=float, default=0.22)
    parser.add_argument("--red-hand-closure-delay-s", type=float, default=0.0)
    parser.add_argument(
        "--profile-fallbacks",
        action="store_true",
        help=(
            "For each random episode, retry a small set of stronger/slower "
            "strict-contact teacher profiles before marking it failed."
        ),
    )
    parser.add_argument(
        "--adaptive-regrasp",
        action="store_true",
        help=(
            "Offline teacher only: recompute lift/transport from its measured "
            "physical contact pose. Deployment never receives this object state."
        ),
    )
    parser.add_argument("--red-high-transport-waypoint", action="store_true")
    parser.add_argument("--resume-successful", action="store_true")
    parser.add_argument("--min-unique-success-positions", type=int, default=12)
    parser.add_argument("--success-position-dedup-m", type=float, default=0.015)
    parser.add_argument(
        "--red-anchor-curriculum",
        action="store_true",
        help="Expand red starts outward from its validated (0.36,-0.34) anchor.",
    )
    return parser.parse_args()


def strict_success(summary: dict | None) -> bool:
    audit = (summary or {}).get("grasp_physics_audit", {})
    return bool(
        summary
        and summary.get("passed")
        and summary.get("grasp_mode") == "contact"
        and audit.get("assisted_constraint_active_steps", 1) == 0
        and int(audit.get("physical_contact_steps", 0)) > 0
        and float(audit.get("target_lift_m", 0.0)) >= 0.10
        and summary.get("target_in_box")
    )


def nearest_profile_slot(position_xy: np.ndarray) -> int:
    return int(np.argmin([np.linalg.norm(position_xy - slot) for slot in OBJECT_SLOTS_XY_M]))


def shape_specialized_profile(
    base_profile: dict, target: str, position_xy: np.ndarray
) -> dict:
    """Use grasp geometry validated for the shape, mirrored across the table."""
    profile = dict(base_profile)
    if target == "red_triangle":
        profile.update(
            grasp_yaw_deg=0.0,
            contact_grasp_compensation="-0.015,0.030,-0.055",
            hand_closure_multiplier=1.0,
            hand_profile_scale="1.0,1.0,1.0",
        )
    elif target == "yellow_rod":
        # Positive-y objects use the left hand. Mirror the successful right-hand
        # +90 degree cross-rod clamp instead of reusing the same world rotation.
        profile.update(
            grasp_yaw_deg=-90.0 if position_xy[1] >= 0.05 else 90.0,
            contact_grasp_compensation="0.015,-0.050,-0.055",
            hand_closure_multiplier=1.05,
            hand_profile_scale="1.0,1.04,1.04",
        )
    return profile


def build_specs(
    scenes: int,
    seed: int,
    targets: tuple[str, ...],
    red_anchor_curriculum: bool = False,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    specs = []
    for scene_index in range(scenes):
        positions = _sample_random_positions(rng)
        if red_anchor_curriculum and "red_triangle" in targets:
            red_position = np.array(
                (
                    0.36 + (0.006 if scene_index % 2 else 0.0),
                    -0.34 + 0.018 * scene_index,
                )
            )
            for _ in range(1000):
                candidate = _sample_random_positions(rng)
                if all(
                    np.linalg.norm(red_position - candidate[index]) >= 0.22
                    for index in (1, 2)
                ):
                    positions = candidate
                    positions[0] = red_position
                    break
            else:
                raise RuntimeError("Could not construct separated red curriculum scene")
        offsets = positions - np.asarray(OBJECT_SLOTS_XY_M)
        for target in targets:
            target_index = OBJECT_NAMES.index(target)
            profile_slot = nearest_profile_slot(positions[target_index])
            profile = shape_specialized_profile(
                contact_profile_for_target_slot(target, profile_slot),
                target,
                positions[target_index],
            )
            specs.append(
                {
                    "episode_index": len(specs),
                    "scene_index": scene_index,
                    "target_object": target,
                    "language_instruction": LANGUAGE_VARIANTS[target][scene_index % len(LANGUAGE_VARIANTS[target])],
                    "random_object_xy_m": positions.tolist(),
                    "slot_permutation": [0, 1, 2],
                    "slot_offsets_xy_m": offsets.tolist(),
                    "profile_slot": profile_slot,
                    "contact_profile": profile,
                }
            )
    return specs


def unique_success_positions(reports: list[dict], target: str, radius_m: float) -> list[list[float]]:
    """Greedily retain spatially distinct successful target start positions."""
    unique: list[np.ndarray] = []
    for report in reports:
        if not report["passed"] or report["target_object"] != target:
            continue
        position = np.asarray(report["random_object_xy_m"][OBJECT_NAMES.index(target)])
        if all(np.linalg.norm(position - existing) >= radius_m for existing in unique):
            unique.append(position)
    return [position.tolist() for position in unique]


def unique_success_reports(reports: list[dict], radius_m: float) -> list[dict]:
    """Return one strict success for each spatially distinct target start."""
    retained: list[dict] = []
    retained_by_target: dict[str, list[np.ndarray]] = {name: [] for name in OBJECT_NAMES}
    for report in reports:
        if not report["passed"]:
            continue
        target = report["target_object"]
        position = np.asarray(report["random_object_xy_m"][OBJECT_NAMES.index(target)])
        if all(
            np.linalg.norm(position - existing) >= radius_m
            for existing in retained_by_target[target]
        ):
            retained.append(report)
            retained_by_target[target].append(position)
    return retained


def spec_payload(spec: dict) -> dict:
    return {key: value for key, value in spec.items() if key not in {"contact_profile"}}


def robust_profile_variants(base_profile: dict, target: str) -> list[dict]:
    """Small calibrated fallback set for strict-contact random-layout teachers."""
    variants = [dict(base_profile, profile_name="base")]
    variants.append(
        {
            **base_profile,
            "profile_name": "strong_slow",
            "hand_gain_scale": 6.0,
            "hand_closure_multiplier": 1.06 * float(base_profile["hand_closure_multiplier"]),
            "lift_hold_duration_s": 0.35,
            "transport_duration_s": 3.4,
            "duration_s": 13.6,
            "hand_profile_scale": "1.05,1.10,1.10",
        }
    )
    if target == "yellow_rod":
        variants.extend(
            [
                {
                    **base_profile,
                    "profile_name": "yellow_deep_parallel_slow",
                    "grasp_yaw_deg": base_profile["grasp_yaw_deg"],
                    "contact_grasp_compensation": "0.015,-0.050,-0.060",
                    "hand_gain_scale": 6.0,
                    "hand_closure_multiplier": 1.08 * float(base_profile["hand_closure_multiplier"]),
                    "lift_hold_duration_s": 0.45,
                    "transport_duration_s": 3.7,
                    "duration_s": 14.0,
                    "hand_profile_scale": "1.05,1.18,1.18",
                },
                {
                    **base_profile,
                    "profile_name": "yellow_diagonal_clamp_slow",
                    "grasp_yaw_deg": 30.0 * np.sign(base_profile["grasp_yaw_deg"]),
                    "contact_grasp_compensation": "0.000,-0.045,-0.060",
                    "hand_gain_scale": 6.0,
                    "hand_closure_multiplier": 1.06 * float(base_profile["hand_closure_multiplier"]),
                    "lift_hold_duration_s": 0.45,
                    "transport_duration_s": 3.7,
                    "duration_s": 14.0,
                    "hand_profile_scale": "1.05,1.16,1.16",
                },
            ]
        )
    elif target == "red_triangle":
        # Random-position failures contact only thumb/palm: compared with the
        # successful fixed-position transform, the palm is about 10 mm too far
        # forward and 20 mm too high. Search geometry before adding force.
        variants.extend(
            [
                {
                    **base_profile,
                    "profile_name": "red_deeper_lower",
                    "contact_grasp_compensation": "-0.025,0.024,-0.065",
                    "hand_profile_scale": "1.0,1.0,1.0",
                },
                {
                    **base_profile,
                    "profile_name": "red_deep_low_index_support",
                    "contact_grasp_compensation": "-0.025,0.024,-0.070",
                    "hand_profile_scale": "1.0,1.06,1.02",
                },
                {
                    **base_profile,
                    "profile_name": "red_retention_conservative",
                    "contact_grasp_compensation": "-0.020,0.024,-0.065",
                    "hand_gain_scale": 6.0,
                    "hand_closure_multiplier": 1.04
                    * float(base_profile["hand_closure_multiplier"]),
                    "hand_profile_scale": "1.02,1.06,1.04",
                },
            ]
        )
    return variants


def build_command(args: argparse.Namespace, spec: dict, episode_dir: Path, profile: dict) -> list[str]:
    duration_s = float(profile.get("duration_s", args.duration_s))
    lift_hold_duration_s = float(profile.get("lift_hold_duration_s", args.lift_hold_duration_s))
    transport_duration_s = float(profile.get("transport_duration_s", args.transport_duration_s))
    return [
        str(args.python),
        str(args.collector),
        "--asset", str(args.asset.resolve()),
        "--output-dir", str(episode_dir),
        "--target-object", spec["target_object"],
        "--slot-permutation=0,1,2",
        "--slot-offsets=" + ",".join(f"{value:.8f}" for pair in spec["slot_offsets_xy_m"] for value in pair),
        "--language-instruction", spec["language_instruction"],
        "--record-demonstration",
        "--disable-demonstration-videos",
        "--disable-overview-video",
        "--video-fps", "15",
        "--render-width", "320",
        "--render-height", "240",
        "--render-skip", str(args.render_skip),
        "--duration-s", str(duration_s),
        "--lift-hold-duration-s", str(lift_hold_duration_s),
        "--transport-duration-s", str(transport_duration_s),
        "--lift-height-m", str(
            args.red_lift_height_m
            if spec["target_object"] == "red_triangle"
            else 0.22
        ),
        "--hand-closure-delay-s", str(
            args.red_hand_closure_delay_s
            if spec["target_object"] == "red_triangle"
            else 0.0
        ),
        "--grasp-mode", "contact",
        "--body-gain-scale", str(profile["body_gain_scale"]),
        "--hand-gain-scale", str(profile["hand_gain_scale"]),
        "--hand-profile-scale", str(profile.get("hand_profile_scale", "1,1,1")),
        "--grasp-yaw-deg", str(profile["grasp_yaw_deg"]),
        "--contact-grasp-compensation=" + profile["contact_grasp_compensation"],
        "--hand-closure-multiplier", str(profile["hand_closure_multiplier"]),
    ]


def collect_one(args: argparse.Namespace, spec: dict) -> dict:
    episode_dir = args.output_dir / f"episode_{spec['episode_index']:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    spec_path = episode_dir / "collection_spec.json"
    summary_path = episode_dir / "language_pick_place_summary.json"
    payload = spec_payload(spec)
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    if (
        args.resume_successful
        and spec_path.is_file()
        and json.loads(spec_path.read_text()) == payload
        and strict_success(summary)
    ):
        return {**payload, "passed": True, "reused": True, "output_dir": str(episode_dir)}

    profiles = (
        robust_profile_variants(spec["contact_profile"], spec["target_object"])
        if args.profile_fallbacks
        else [dict(spec["contact_profile"], profile_name="base")]
    )
    attempt_reports = []
    completed = None
    summary = None
    passed = False
    for attempt_index, profile in enumerate(profiles):
        command = build_command(args, spec, episode_dir, profile)
        if args.adaptive_regrasp:
            command.append("--adaptive-regrasp")
        if args.red_high_transport_waypoint and spec["target_object"] == "red_triangle":
            command.append("--high-transport-waypoint")
        completed = subprocess.run(command, check=False, env={**os.environ, "MUJOCO_GL": "egl"})
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
        passed = strict_success(summary)
        audit = (summary or {}).get("grasp_physics_audit", {})
        attempt_reports.append(
            {
                "attempt": attempt_index,
                "profile_name": profile.get("profile_name", f"attempt_{attempt_index}"),
                "returncode": completed.returncode,
                "passed": passed,
                "physical_contact_steps": audit.get("physical_contact_steps", 0),
                "target_lift_m": audit.get("target_lift_m", 0.0),
                "target_in_box": (summary or {}).get("target_in_box", False),
            }
        )
        if passed:
            break
    spec_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        **payload,
        "output_dir": str(episode_dir),
        "returncode": completed.returncode if completed is not None else None,
        "passed": passed,
        "reused": False,
        "profile_name": attempt_reports[-1]["profile_name"] if attempt_reports else None,
        "attempts": attempt_reports,
        "physical_contact_steps": (summary or {}).get("grasp_physics_audit", {}).get("physical_contact_steps", 0),
        "target_lift_m": (summary or {}).get("grasp_physics_audit", {}).get("target_lift_m", 0.0),
        "target_in_box": (summary or {}).get("target_in_box", False),
    }


def main() -> None:
    args = parse_args()
    if (
        args.scenes < 1
        or args.workers < 1
        or args.workers > 1
        or args.render_skip < 1
        or args.duration_s <= 0
        or args.lift_hold_duration_s < 0
        or args.transport_duration_s <= 0
        or not 0.10 <= args.red_lift_height_m <= 0.30
        or not 0.0 <= args.red_hand_closure_delay_s <= 0.8
        or args.min_unique_success_positions < 1
        or args.success_position_dedup_m <= 0
    ):
        raise ValueError("Invalid scenes/workers/render/duration settings")
    if not args.python.is_file() or not args.collector.is_file() or not args.asset.is_file():
        raise FileNotFoundError("Missing Python, strict expert collector, or MuJoCo asset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = tuple(dict.fromkeys(args.target or OBJECT_NAMES))
    specs = build_specs(args.scenes, args.seed, targets, args.red_anchor_curriculum)
    reports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(collect_one, args, spec) for spec in specs]
        for future in concurrent.futures.as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                "UNIFORM_RANDOM_STRICT_EXPERT=" + json.dumps(
                    {
                        "complete": len(reports),
                        "total": len(specs),
                        "episode": report["episode_index"],
                        "target": report["target_object"],
                        "passed": report["passed"],
                    }
                ),
                flush=True,
            )
    reports.sort(key=lambda item: item["episode_index"])
    by_target = {
        target: sum(item["passed"] and item["target_object"] == target for item in reports)
        for target in targets
    }
    unique_positions = {
        target: unique_success_positions(reports, target, args.success_position_dedup_m)
        for target in targets
    }
    complete_scenes = [
        all(
            any(
                item["scene_index"] == scene_index
                and item["target_object"] == target
                and item["passed"]
                for item in reports
            )
            for target in targets
        )
        for scene_index in range(args.scenes)
    ]
    summary = {
        "experiment": "G1LANG-uniform-random-strict-contact-experts",
        "layout_contract": "matches evaluate_g1_language_pick_place random independent XY range and separation",
        "seed": args.seed,
        "scenes": args.scenes,
        "episodes": len(reports),
        "strict_successes": int(sum(item["passed"] for item in reports)),
        "strict_successes_by_target": by_target,
        "requested_targets": list(targets),
        "red_anchor_curriculum": bool(args.red_anchor_curriculum),
        "red_lift_height_m": args.red_lift_height_m,
        "red_hand_closure_delay_s": args.red_hand_closure_delay_s,
        "unique_success_positions_by_target": {
            target: len(positions) for target, positions in unique_positions.items()
        },
        "unique_success_xy_m_by_target": unique_positions,
        "success_position_dedup_m": args.success_position_dedup_m,
        "min_unique_success_positions": args.min_unique_success_positions,
        "complete_scene_triplets": int(sum(complete_scenes)),
        "input_contract": "head/left-wrist/right-wrist RGB + language + state31; object positions are offline teacher-only",
        "offline_teacher_adaptive_regrasp": bool(args.adaptive_regrasp),
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "reports": reports,
    }
    summary["passed"] = bool(
        all(
            len(unique_positions[target]) >= args.min_unique_success_positions
            for target in targets
        )
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    unique_manifest = unique_success_reports(reports, args.success_position_dedup_m)
    (args.output_dir / "strict_unique_success_manifest.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in unique_manifest),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
