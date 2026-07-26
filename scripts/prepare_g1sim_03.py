#!/usr/bin/env python3
"""Prepare a successful multi-object language episode for Isaac Sim replay."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/mnt/d/G1-UpperBody-Sim2Sim/G1SIM-03")
DEFAULT_ASSET_ROOT = Path(
    "/home/ubuntu/unitree_lerobot/unitree_lerobot/eval_robot/assets/g1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--source-experiment", required=True)
    args = parser.parse_args()

    summary_path = args.episode / "summary.json"
    trajectory_path = args.episode / "trajectory.npz"
    scene_path = args.episode / "scene.xml"
    for path in (summary_path, trajectory_path, scene_path, args.asset_root):
        if not path.exists():
            raise FileNotFoundError(path)
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not source_summary["passed"]:
        raise ValueError("G1SIM-03 requires a strict-success source episode")

    with np.load(trajectory_path) as trajectory:
        object_names = trajectory["object_names"].astype(str).tolist()
        grabbed = trajectory["grabbed_object_index"].astype(np.int64)
        valid = grabbed[grabbed >= 0]
        if not len(valid):
            raise ValueError("Source episode never grabbed an object")
        tracked_object_name = object_names[int(valid[0])]
        action_names = trajectory["joint_names"].astype(str).tolist()
        control_fps = int(trajectory["control_fps"])
    if tracked_object_name != source_summary["target_object"]:
        raise ValueError("Source episode grabbed a non-target object")

    assets_dir = args.output / "assets"
    data_dir = args.output / "data"
    scripts_dir = args.output / "scripts"
    usd_dir = args.output / "usd"
    for directory in (assets_dir, data_dir, scripts_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if usd_dir.exists():
        shutil.rmtree(usd_dir)
    usd_dir.mkdir(parents=True)
    report_path = args.output / "isaacsim_report.json"
    if report_path.exists():
        report_path.unlink()

    shutil.copytree(
        args.asset_root / "meshes", assets_dir / "meshes", dirs_exist_ok=True
    )
    scene_tree = ET.parse(scene_path)
    compiler = scene_tree.getroot().find("compiler")
    if compiler is None:
        raise ValueError("MJCF scene has no <compiler> element")
    compiler.set("meshdir", "meshes")
    bundled_scene = assets_dir / "g1_language_episode.xml"
    scene_tree.write(bundled_scene, encoding="unicode")
    shutil.copy2(trajectory_path, data_dir / "trajectory.npz")
    shutil.copy2(summary_path, data_dir / "source_summary.json")
    shutil.copy2(
        PROJECT_ROOT / "sim2sim/isaacsim/g1sim_replay.py",
        scripts_dir / "g1sim_replay.py",
    )

    joint_names = [
        f"{name}_{side}_assisted_grasp"
        for name in object_names
        for side in ("left", "right")
    ]
    manifest = {
        "experiment": "G1SIM-03-language-task-isaacsim-replay",
        "source_experiment": args.source_experiment,
        "source_episode": str(args.episode.resolve()),
        "source_simulator": "MuJoCo",
        "target_simulator": "Isaac Sim 6.0.1 PhysX",
        "task_type": "language_object_to_box",
        "mjcf_file": "assets/g1_language_episode.xml",
        "control_fps": control_fps,
        "physics_fps": 60,
        "action_joint_names": action_names,
        "object_names": object_names,
        "tracked_object_name": tracked_object_name,
        "goal_position_m": source_summary["blue_box_position_m"],
        "activate_assisted_grasp": True,
        "assisted_grasp_joint_names": joint_names,
        "source_passed": True,
        "scope": (
            "Offline multi-object language-task action replay with the exact "
            "source grasp/release event stream. This is not online VLA inference."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"Prepared {args.output}")


if __name__ == "__main__":
    main()
