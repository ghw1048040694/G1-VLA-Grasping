#!/usr/bin/env python3
"""Collect randomized successful G1 pre-grasp expert episodes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


INSTRUCTIONS = (
    "move both hands to the tote pre-grasp pose",
    "position both hands beside the tote",
    "bring both hands to the sides of the blue tote",
    "prepare both hands to grasp the tote",
)


def build_scene(source: Path, destination: Path, tote_x: float, tote_y: float) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    tote = root.find(".//body[@name='warehouse_tote']")
    table = root.find(".//geom[@name='calibration_table']")
    if tote is None or table is None:
        raise RuntimeError("Source scene is missing the tote or table")
    tote_pos = [float(value) for value in tote.get("pos", "").split()]
    table_pos = [float(value) for value in table.get("pos", "").split()]
    tote_pos[:2] = [tote_x, tote_y]
    table_pos[0] = tote_x + 0.15
    tote.set("pos", " ".join(str(value) for value in tote_pos))
    table.set("pos", " ".join(str(value) for value in table_pos))
    tree.write(destination, encoding="unicode", xml_declaration=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2107)
    parser.add_argument("--tote-x-min", type=float, default=0.52)
    parser.add_argument("--tote-x-max", type=float, default=0.58)
    parser.add_argument("--experiment-id", default="G1WH-21-randomized-expert-dataset")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    episodes = []
    for episode_index in range(args.episodes):
        tote_x = float(rng.uniform(args.tote_x_min, args.tote_x_max))
        tote_y = 0.0
        instruction = INSTRUCTIONS[episode_index % len(INSTRUCTIONS)]
        episode_dir = args.output_dir / f"episode_{episode_index:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        scene_path = episode_dir / "scene.xml"
        build_scene(args.asset.resolve(), scene_path, tote_x, tote_y)
        episode_summary_path = episode_dir / "safe_reset_expert_summary.json"
        episode_summary_path.unlink(missing_ok=True)
        result = subprocess.run(
            [
                str(args.python),
                str(args.collector),
                "--python",
                str(args.python),
                "--runner",
                str(args.collector.parent / "validate_g1_pregrasp_trajectory.py"),
                "--asset",
                str(scene_path),
                "--reachability-report",
                str(args.reachability_report.resolve()),
                "--output-dir",
                str(episode_dir),
                "--tote-x",
                str(tote_x),
                "--language-instruction",
                instruction,
                "--pregrasp-clearance-m",
                "0.16",
            ],
            check=False,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        summary = (
            json.loads(episode_summary_path.read_text())
            if result.returncode == 0 and episode_summary_path.exists()
            else None
        )
        success = bool(summary and summary["experiment_passed"])
        episodes.append(
            {
                "episode_index": episode_index,
                "tote_x_m": tote_x,
                "tote_y_m": tote_y,
                "language_instruction": instruction,
                "success": success,
                "failure_reason": None if success else f"collector_exit_code_{result.returncode}",
                "episode_dir": str(episode_dir),
                "dataset": str(episode_dir / "expert_episode.npz"),
            }
        )

    successful = [episode for episode in episodes if episode["success"]]
    manifest_path = args.output_dir / "successful_episodes.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(episode) + "\n" for episode in successful), encoding="utf-8"
    )
    summary = {
        "experiment": args.experiment_id,
        "seed": args.seed,
        "requested_episodes": args.episodes,
        "successful_episodes": len(successful),
        "success_rate": len(successful) / args.episodes,
        "tote_x_range_m": [args.tote_x_min, args.tote_x_max],
        "tote_y_range_m": [0.0, 0.0],
        "language_variants": list(INSTRUCTIONS),
        "episodes": episodes,
        "successful_manifest": str(manifest_path),
        "experiment_passed": len(successful) == args.episodes,
    }
    summary_path = args.output_dir / "randomized_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
