#!/usr/bin/env python3
"""Collect randomized successful assisted G1 tote-lift episodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


INSTRUCTIONS = (
    "lift the blue tote with both hands",
    "pick up the blue tote using both hands",
    "raise the tote off the table",
    "grasp both sides of the tote and lift it",
)
WORLD_MODEL_KEYS = {
    "tote_position_m",
    "tote_quaternion_wxyz",
    "tote_linear_velocity_m_s",
    "tote_angular_velocity_rad_s",
    "palm_position_m",
    "bilateral_hand_contact",
    "table_contact",
    "tote_lift_height_m",
    "task_progress",
}


def has_world_model_contract(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path) as arrays:
            return WORLD_MODEL_KEYS.issubset(arrays.files)
    except (OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2207)
    parser.add_argument("--tote-x-min", type=float, default=0.43)
    parser.add_argument("--tote-x-max", type=float, default=0.46)
    parser.add_argument(
        "--resume-successful",
        action="store_true",
        help="Reuse already completed successful episodes in the output directory.",
    )
    parser.add_argument(
        "--experiment-id", default="G1WH-23-randomized-assisted-lift-dataset"
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.tote_x_min >= args.tote_x_max:
        parser.error("--tote-x-min must be smaller than --tote-x-max")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    episodes = []

    for episode_index in range(args.episodes):
        tote_x = float(rng.uniform(args.tote_x_min, args.tote_x_max))
        instruction = INSTRUCTIONS[episode_index % len(INSTRUCTIONS)]
        episode_dir = args.output_dir / f"episode_{episode_index:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        summary_path = episode_dir / "assisted_tote_lift_summary.json"
        required_outputs = (
            episode_dir / "expert_lift_episode.npz",
            episode_dir / "head_camera.mp4",
            episode_dir / "left_wrist_camera.mp4",
            episode_dir / "right_wrist_camera.mp4",
        )
        existing_summary = (
            json.loads(summary_path.read_text()) if summary_path.exists() else None
        )
        reuse = bool(
            args.resume_successful
            and existing_summary
            and existing_summary.get("passed")
            and abs(float(existing_summary.get("tote_x_m", float("inf"))) - tote_x)
            < 1e-12
            and all(path.is_file() for path in required_outputs)
            and has_world_model_contract(required_outputs[0])
        )
        if reuse:
            print(f"Reusing successful episode {episode_index:04d}", flush=True)
            summary = existing_summary
            returncode = 0
        else:
            summary_path.unlink(missing_ok=True)
            result = subprocess.run(
                [
                    str(args.python),
                    str(args.collector),
                    "--asset",
                    str(args.asset.resolve()),
                    "--output-dir",
                    str(episode_dir),
                    "--tote-x",
                    str(tote_x),
                    "--record-demonstration",
                    "--video-fps",
                    "15",
                    "--render-width",
                    "320",
                    "--render-height",
                    "240",
                    "--language-instruction",
                    instruction,
                ],
                check=False,
                env={**os.environ, "MUJOCO_GL": "egl"},
            )
            returncode = result.returncode
            summary = (
                json.loads(summary_path.read_text())
                if returncode == 0 and summary_path.exists()
                else None
            )
        success = bool(summary and summary["passed"])
        if success:
            failure_reason = None
        elif returncode != 0:
            failure_reason = f"collector_exit_code_{returncode}"
        elif summary is None:
            failure_reason = "collector_summary_missing"
        else:
            failure_reason = "episode_contract_failed"
        episodes.append(
            {
                "episode_index": episode_index,
                "tote_x_m": tote_x,
                "language_instruction": instruction,
                "success": success,
                "failure_reason": failure_reason,
                "episode_dir": str(episode_dir),
                "dataset": str(episode_dir / "expert_lift_episode.npz"),
            }
        )

    successful = [item for item in episodes if item["success"]]
    manifest_path = args.output_dir / "successful_lift_episodes.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item) + "\n" for item in successful), encoding="utf-8"
    )
    summary = {
        "experiment": args.experiment_id,
        "seed": args.seed,
        "requested_episodes": args.episodes,
        "successful_episodes": len(successful),
        "success_rate": len(successful) / args.episodes,
        "tote_x_range_m": [args.tote_x_min, args.tote_x_max],
        "language_variants": list(INSTRUCTIONS),
        "episodes": episodes,
        "successful_manifest": str(manifest_path),
        "experiment_passed": len(successful) == args.episodes,
    }
    summary_path = args.output_dir / "assisted_lift_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")
    if not summary["experiment_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
