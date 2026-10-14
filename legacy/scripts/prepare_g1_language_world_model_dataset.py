#!/usr/bin/env python3
"""Project multi-object language demonstrations into the 86-state WM contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "joint_names",
    "time_s",
    "joint_position_rad",
    "joint_velocity_rad_s",
    "action_joint_position_rad",
    "task_phase",
    "assist_active",
    "active_arm_index",
    "object_position_m",
    "object_quaternion_wxyz",
    "palm_position_m",
    "target_object_index",
}


def finite_difference(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float32)
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(values, time_s, axis=0, edge_order=edge_order).astype(np.float32)


def quaternion_angular_velocity(
    quaternion_wxyz: np.ndarray, time_s: np.ndarray
) -> np.ndarray:
    quaternion = quaternion_wxyz.astype(np.float64).copy()
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12)
    for index in range(1, len(quaternion)):
        if np.dot(quaternion[index - 1], quaternion[index]) < 0.0:
            quaternion[index] *= -1.0

    velocity = np.zeros((len(quaternion), 3), dtype=np.float64)
    for index in range(1, len(quaternion)):
        previous = quaternion[index - 1]
        current = quaternion[index]
        previous_conjugate = previous * np.asarray((1.0, -1.0, -1.0, -1.0))
        w1, x1, y1, z1 = current
        w2, x2, y2, z2 = previous_conjugate
        relative = np.asarray(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )
        )
        relative /= max(float(np.linalg.norm(relative)), 1e-12)
        vector_norm = float(np.linalg.norm(relative[1:]))
        if vector_norm > 1e-12:
            angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[0], -1.0, 1.0))
            if angle > np.pi:
                angle -= 2.0 * np.pi
            dt = max(float(time_s[index] - time_s[index - 1]), 1e-8)
            velocity[index] = relative[1:] / vector_norm * angle / dt
    if len(velocity) > 1:
        velocity[0] = velocity[1]
    return velocity.astype(np.float32)


def project_episode(source: Path, destination: Path) -> dict:
    with np.load(source) as arrays:
        missing = REQUIRED_KEYS - set(arrays.files)
        if missing:
            raise ValueError(f"{source} is missing fields: {sorted(missing)}")
        frame_count = len(arrays["time_s"])
        target_indices = arrays["target_object_index"].astype(np.int64)
        if target_indices.shape != (frame_count,) or len(np.unique(target_indices)) != 1:
            raise ValueError(f"{source} must contain one fixed target object")
        target_index = int(target_indices[0])
        if target_index not in (0, 1, 2):
            raise ValueError(f"{source} has invalid target index {target_index}")

        rows = np.arange(frame_count)
        time_s = arrays["time_s"].astype(np.float64)
        target_position = arrays["object_position_m"][rows, target_indices].astype(
            np.float32
        )
        target_quaternion = arrays["object_quaternion_wxyz"][
            rows, target_indices
        ].astype(np.float32)
        target_linear_velocity = finite_difference(target_position, time_s)
        target_angular_velocity = quaternion_angular_velocity(
            target_quaternion, time_s
        )
        lift_height = (target_position[:, 2] - target_position[0, 2]).astype(
            np.float32
        )

        assist_active = arrays["assist_active"].astype(np.float32)
        active_arm = arrays["active_arm_index"].astype(np.int64)
        if np.any((active_arm < 0) | (active_arm > 1)):
            raise ValueError(f"{source} has invalid active arm indices")
        bilateral_contact = np.zeros((frame_count, 2), dtype=np.float32)
        bilateral_contact[rows, active_arm] = assist_active
        table_contact = (lift_height <= 0.003).astype(np.float32)
        task_phase = arrays["task_phase"].astype(np.float32)
        phase_scale = max(float(np.max(task_phase)), 1.0)
        task_progress = (task_phase / phase_scale).astype(np.float32)

        payload = {
            "joint_names": arrays["joint_names"],
            "observation_joint_position_rad": arrays["joint_position_rad"].astype(
                np.float32
            ),
            "observation_joint_velocity_rad_s": arrays[
                "joint_velocity_rad_s"
            ].astype(np.float32),
            "action_joint_position_rad": arrays[
                "action_joint_position_rad"
            ].astype(np.float32),
            "tote_position_m": target_position,
            "tote_quaternion_wxyz": target_quaternion,
            "tote_linear_velocity_m_s": target_linear_velocity,
            "tote_angular_velocity_rad_s": target_angular_velocity,
            "palm_position_m": arrays["palm_position_m"].astype(np.float32),
            "bilateral_hand_contact": bilateral_contact,
            "table_contact": table_contact,
            "tote_lift_height_m": lift_height,
            "task_progress": task_progress,
            "target_object_index": target_indices,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return {
        "frames": frame_count,
        "target_object_index": target_index,
        "max_lift_height_m": float(np.max(lift_height)),
        "finite": all(
            np.isfinite(value).all()
            for value in payload.values()
            if value.dtype.kind in "fiu"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_summary = json.loads(args.source_summary.read_text(encoding="utf-8"))
    specs = source_summary.get("episode_specs", [])
    if not specs:
        raise ValueError("Source summary contains no episode_specs")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    target_counts: dict[str, int] = {}
    for spec in specs:
        episode_index = int(spec["episode_index"])
        source = Path(spec["episode_dir"]) / "expert_pick_place_episode.npz"
        destination = args.output_dir / f"episode_{episode_index:04d}.npz"
        audit = project_episode(source, destination)
        target = str(spec["target_object"])
        target_counts[target] = target_counts.get(target, 0) + 1
        episodes.append(
            {
                "episode_index": episode_index,
                "success": bool(spec["passed"]),
                "target_object": target,
                "source_dataset": str(source.resolve()),
                "dataset": str(destination.resolve()),
                **audit,
            }
        )

    summary = {
        "experiment": "G1FINAL-02-language-target-world-model-dataset",
        "contract_version": "g1_language_target_projected_to_wm86_v1",
        "source_summary": str(args.source_summary.resolve()),
        "episodes": episodes,
        "episode_count": len(episodes),
        "successful_episodes": sum(item["success"] for item in episodes),
        "target_counts": target_counts,
        "projection_contract": {
            "legacy_tote_slot": "the episode's language-selected target object",
            "linear_velocity": "finite difference of target position using recorded time_s",
            "angular_velocity": "quaternion relative rotation using recorded time_s",
            "bilateral_contact": "assist_active assigned to active_arm_index",
            "table_contact": "target lift height <= 0.003 m proxy",
            "lift_height": "target z minus initial target z",
            "task_progress": "task_phase divided by the episode maximum phase",
        },
        "limitation": (
            "The 86-state model contains only the routed target object; "
            "non-target objects remain outside the learned dynamics state."
        ),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "episodes"},
            indent=2,
        )
    )
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
