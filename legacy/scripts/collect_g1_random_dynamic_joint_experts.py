#!/usr/bin/env python3
"""Collect strict dynamic expert trajectories on camera-matched random scenes.

This collector is intentionally sequential.  It reuses the random layouts that
were used for the current RGB calibration, executes the contact-only MuJoCo
expert for every language target, and retains a trajectory only when physical
contact, lift, and placement all pass without an assisted constraint.  These
trajectories are training labels for the direct joint-action VLA; simulator
object state is never part of the deployed policy input.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from collect_g1_language_pick_place_dataset import contact_profile_for_target_slot
from run_g1_language_pick_place import (
    INSTRUCTIONS,
    OBJECT_NAMES,
    OBJECT_SLOTS_XY_M,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-layout-root", type=Path, required=True)
    parser.add_argument(
        "--layout-episodes",
        default="0,50,100",
        help="Comma-separated G1LANG-330 episode indices used as source layouts.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--render-skip", type=int, default=6)
    return parser.parse_args()


def parse_indices(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--layout-episodes must be comma-separated integers") from error
    if not result or len(set(result)) != len(result) or any(item < 0 for item in result):
        raise ValueError("--layout-episodes must contain distinct non-negative indices")
    return result


def layout_positions(path: Path) -> np.ndarray:
    with np.load(path) as arrays:
        positions = np.asarray(arrays["object_position_m"][0], dtype=np.float64)
    if positions.shape != (3, 3):
        raise ValueError(f"Expected first object positions (3, 3), found {positions.shape}")
    return positions


def nearest_slot(position_xy: np.ndarray) -> int:
    return int(np.argmin([np.linalg.norm(position_xy - item) for item in OBJECT_SLOTS_XY_M]))


def successful(summary: dict | None) -> bool:
    audit = (summary or {}).get("grasp_physics_audit", {})
    return bool(
        summary
        and summary.get("passed")
        and summary.get("grasp_mode") == "contact"
        and audit.get("assisted_constraint_active_steps") == 0
        and audit.get("physical_contact_steps", 0) > 0
    )


def main() -> None:
    args = parse_args()
    if args.render_skip < 1:
        raise ValueError("--render-skip must be positive")
    indices = parse_indices(args.layout_episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for layout_index in indices:
        source_dir = args.random_layout_root / f"episode_{layout_index:04d}"
        scene = source_dir / "scene.xml"
        trajectory = source_dir / "trajectory.npz"
        if not scene.is_file() or not trajectory.is_file():
            raise FileNotFoundError(f"Missing scene or trajectory in {source_dir}")
        positions = layout_positions(trajectory)
        offsets = positions[:, :2] - np.asarray(OBJECT_SLOTS_XY_M)
        serialized_offsets = ",".join(f"{value:.8f}" for pair in offsets for value in pair)
        for target_index, target in enumerate(OBJECT_NAMES):
            slot = nearest_slot(positions[target_index, :2])
            profile = contact_profile_for_target_slot(target, slot)
            episode_dir = args.output_dir / f"layout_{layout_index:04d}_{target}"
            command = [
                str(args.python),
                str(args.collector),
                "--asset", str(scene.resolve()),
                "--output-dir", str(episode_dir),
                "--target-object", target,
                "--slot-permutation=0,1,2",
                "--slot-offsets=" + serialized_offsets,
                "--language-instruction", INSTRUCTIONS[target],
                "--record-demonstration",
                "--disable-overview-video",
                "--render-skip", str(args.render_skip),
                "--grasp-mode", "contact",
                "--body-gain-scale", str(profile["body_gain_scale"]),
                "--hand-gain-scale", str(profile["hand_gain_scale"]),
                "--grasp-yaw-deg", str(profile["grasp_yaw_deg"]),
                "--contact-grasp-compensation=" + profile["contact_grasp_compensation"],
                "--hand-closure-multiplier", str(profile["hand_closure_multiplier"]),
            ]
            completed = subprocess.run(command, check=False)
            summary_path = episode_dir / "language_pick_place_summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
            report = {
                "layout_episode": layout_index,
                "target_object": target,
                "target_slot_profile": slot,
                "source_scene": str(scene),
                "source_object_position_m": positions.tolist(),
                "returncode": completed.returncode,
                "eligible": successful(summary),
                "summary": summary,
            }
            reports.append(report)
            print(
                "RANDOM_DYNAMIC_EXPERT="
                + json.dumps(
                    {
                        "layout_episode": layout_index,
                        "target": target,
                        "eligible": report["eligible"],
                        "returncode": completed.returncode,
                    }
                ),
                flush=True,
            )
    eligible = [item for item in reports if item["eligible"]]
    summary = {
        "experiment": "G1LANG-random-camera-matched-dynamic-joint-experts",
        "source_random_layout_root": str(args.random_layout_root),
        "layout_episodes": indices,
        "episodes": len(reports),
        "eligible_physical_dynamic_episodes": len(eligible),
        "by_target": {
            target: sum(item["eligible"] and item["target_object"] == target for item in reports)
            for target in OBJECT_NAMES
        },
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "reports": reports,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))
    if len(eligible) < len(OBJECT_NAMES):
        raise RuntimeError("Need at least one strict dynamic expert trajectory for every target")


if __name__ == "__main__":
    main()
