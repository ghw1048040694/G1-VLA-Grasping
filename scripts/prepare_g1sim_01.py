#!/usr/bin/env python3
"""Prepare a small Windows-readable bundle for the first Isaac Sim replay."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("/mnt/d/G1-UpperBody-Sim2Sim/G1SIM-01")
DEFAULT_ASSET_ROOT = Path(
    "/home/ubuntu/unitree_lerobot/unitree_lerobot/eval_robot/assets/g1"
)
DEFAULT_EPISODE = (
    PROJECT_ROOT
    / "outputs/G1MPC-04_rank_filtered_heldout"
    / "scaled20000_world_model_mpc/episode_0000"
)
DEFAULT_DATASET_INFO = Path(
    "/home/ubuntu/lerobot/datasets/local/g1_scaled_phase_recovery_train/meta/info.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    parser.add_argument("--dataset-info", type=Path, default=DEFAULT_DATASET_INFO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.asset_root, args.episode, args.dataset_info):
        if not path.exists():
            raise FileNotFoundError(path)

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
        args.asset_root / "meshes",
        assets_dir / "meshes",
        dirs_exist_ok=True,
    )

    scene_tree = ET.parse(args.episode / "scene.xml")
    compiler = scene_tree.getroot().find("compiler")
    if compiler is None:
        raise ValueError("MJCF scene has no <compiler> element")
    compiler.set("meshdir", "meshes")
    scene_path = assets_dir / "g1_mpc04_episode0000.xml"
    scene_tree.write(scene_path, encoding="unicode")

    shutil.copy2(args.episode / "trajectory.npz", data_dir / "trajectory.npz")
    shutil.copy2(args.episode / "summary.json", data_dir / "genesis_summary.json")
    shutil.copy2(
        PROJECT_ROOT / "sim2sim/isaacsim/g1sim_replay.py",
        scripts_dir / "g1sim_replay.py",
    )

    dataset_info = json.loads(args.dataset_info.read_text(encoding="utf-8"))
    genesis_summary = json.loads(
        (args.episode / "summary.json").read_text(encoding="utf-8")
    )
    action_feature = dataset_info["features"]["action"]
    manifest = {
        "experiment": "G1SIM-01-isaacsim-offline-action-replay",
        "source_experiment": "G1MPC-04-rank-filtered-independent-heldout",
        "source_policy": "scaled20000_world_model_mpc",
        "source_episode": 0,
        "source_simulator": "MuJoCo/Genesis project pipeline",
        "target_simulator": "Isaac Sim 6.0.1 PhysX",
        "control_fps": dataset_info["fps"],
        "physics_fps": 60,
        "action_joint_names": action_feature["names"],
        "source_passed": genesis_summary["passed"],
        "source_final_lift_height_m": genesis_summary["tote_lift_height_m"],
        "source_maximum_lift_height_m": genesis_summary[
            "maximum_tote_lift_height_m"
        ],
        "source_joint_limit_violation_fraction": genesis_summary[
            "joint_limit_violation_fraction"
        ],
        "scope": (
            "Offline action replay for articulation, joint-order, units, and "
            "basic PhysX response validation. This is not online VLA inference."
        ),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {args.output}")
    print(f"Source strict success: {manifest['source_passed']}")
    print(f"Action dimensions: {len(manifest['action_joint_names'])}")


if __name__ == "__main__":
    main()
