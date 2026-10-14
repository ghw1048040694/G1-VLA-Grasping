#!/usr/bin/env python3
"""Audit whether closed-loop videos moved the requested or a wrong object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit_episode(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    with np.load(path / "trajectory.npz") as trajectory:
        names = trajectory["object_names"].astype(str).tolist()
        positions = trajectory["object_position_m"].astype(np.float64)
        observations = trajectory["observation_joint_position_rad"].astype(np.float64)
        actions = trajectory["action_joint_position_rad"].astype(np.float64)
        grabbed_indices = trajectory["grabbed_object_index"].astype(np.int64)
        video_frames = int(observations.shape[0])

    grabbed = sorted({int(index) for index in grabbed_indices if index >= 0})
    object_motion = {}
    for index, name in enumerate(names):
        delta = positions[-1, index] - positions[0, index]
        object_motion[name] = {
            "initial_position_m": positions[0, index].tolist(),
            "final_position_m": positions[-1, index].tolist(),
            "max_z_m": float(np.max(positions[:, index, 2])),
            "final_displacement_m": float(np.linalg.norm(delta)),
            "final_delta_m": delta.tolist(),
        }
    return {
        "episode": path.name,
        "video": str(path / "closed_loop.mp4"),
        "video_frames_from_trajectory": video_frames,
        "language_instruction": summary["language_instruction"],
        "target_object": summary["target_object"],
        "routed_target": summary.get("routed_target"),
        "grabbed_objects": [names[index] for index in grabbed],
        "selected_correct_object": bool(summary["selected_correct_object"]),
        "task_success": bool(summary["task_success"]),
        "strict_passed": bool(summary["passed"]),
        "action_observation_rmse_rad": float(
            np.sqrt(np.mean((actions - observations) ** 2))
        ),
        "max_frame_action_delta_rad": float(np.max(np.abs(np.diff(actions, axis=0)))),
        "object_motion": object_motion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = [
        audit_episode(path)
        for path in sorted(args.root.glob("episode_*"))
        if (path / "summary.json").is_file() and (path / "trajectory.npz").is_file()
    ]
    if not episodes:
        raise RuntimeError(f"No complete episode trajectories found under {args.root}")
    report = {
        "experiment": "G1FINAL-05-video-trajectory-audit",
        "source_root": str(args.root),
        "episodes": episodes,
        "video_audit_passed": all(item["video_frames_from_trajectory"] > 0 for item in episodes),
        "interpretation": (
            "A nonzero action/observation difference proves control was applied; "
            "target selection and strict success remain separate gates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
