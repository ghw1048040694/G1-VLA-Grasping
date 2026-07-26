#!/usr/bin/env python3
"""Measure whether language alone changes G1 SmolVLA initial action chunks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from evaluate_g1_language_smolvla_heldout import (
    load_policy,
    make_dataset,
    seeded_noise,
)


TARGETS = ("red_triangle", "yellow_rod", "green_cube")
PAIRS = ((0, 1), (0, 2), (1, 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_language_pick_place_train")
    parser.add_argument("--val-repo-id", default="local/g1_language_pick_place_val")
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((left - right) ** 2)))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.scenes < 1:
        raise ValueError("scenes must be positive")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)
    dataset = make_dataset(args, args.checkpoint)
    scene_count = min(args.scenes, dataset.num_episodes // len(TARGETS))
    policy = load_policy(args.checkpoint, train_meta, device)

    records = []
    for scene_index in range(scene_count):
        episode_indices = [scene_index * len(TARGETS) + index for index in range(3)]
        starts = [int(dataset.episode_data_index["from"][index]) for index in episode_indices]
        items = [dataset[index] for index in starts]
        target_indices = [
            int(item["complementary_info.target_object_index"]) for item in items
        ]
        if target_indices != list(range(len(TARGETS))):
            raise ValueError(
                f"Scene {scene_index} target order is {target_indices}, expected [0, 1, 2]"
            )
        tasks = [item["task"] for item in items]
        expert_chunks = torch.stack([item["action"] for item in items]).to(device)
        reference_positions = torch.stack(
            [item["complementary_info.object_position"] for item in items]
        ).reshape(len(TARGETS), len(TARGETS), 3)
        reference_position_delta = torch.linalg.vector_norm(
            reference_positions - reference_positions[0:1], dim=-1
        )
        reference_states = torch.stack([item["observation.state"] for item in items])
        base = items[0]
        predictions = []
        for task in tasks:
            batch = {
                key: value.unsqueeze(0).to(device)
                for key, value in base.items()
                if key.startswith("observation.") and isinstance(value, torch.Tensor)
            }
            batch["task"] = [task]
            predictions.append(
                policy.predict_action_chunk(
                    batch,
                    noise=seeded_noise(policy, 1, args.seed + scene_index, device),
                )[0]
            )
        predictions = torch.stack(predictions)
        distance_matrix = np.asarray(
            [
                [rmse(prediction, expert) for expert in expert_chunks]
                for prediction in predictions
            ]
        )
        nearest = np.argmin(distance_matrix, axis=1)
        predicted_pairwise = [
            rmse(predictions[left], predictions[right]) for left, right in PAIRS
        ]
        expert_pairwise = [
            rmse(expert_chunks[left], expert_chunks[right]) for left, right in PAIRS
        ]
        records.append(
            {
                "scene_index": scene_index,
                "tasks": tasks,
                "distance_to_expert_rmse_rad": distance_matrix.tolist(),
                "nearest_expert_target": [TARGETS[index] for index in nearest],
                "language_classification_correct": [
                    bool(index == nearest[index]) for index in range(len(TARGETS))
                ],
                "predicted_pairwise_rmse_rad": predicted_pairwise,
                "expert_pairwise_rmse_rad": expert_pairwise,
                "reference_scene_max_object_position_delta_m": float(
                    reference_position_delta.max()
                ),
                "reference_initial_state_max_abs_delta_rad": float(
                    (reference_states - reference_states[0:1]).abs().max()
                ),
            }
        )

    correct = [value for row in records for value in row["language_classification_correct"]]
    per_target = {}
    confusion_matrix = {
        target: {prediction: 0 for prediction in TARGETS} for target in TARGETS
    }
    for target_index, target in enumerate(TARGETS):
        target_correct = [
            row["language_classification_correct"][target_index] for row in records
        ]
        per_target[target] = {
            "correct": int(sum(target_correct)),
            "queries": len(target_correct),
            "accuracy": float(np.mean(target_correct)),
        }
        for row in records:
            prediction = row["nearest_expert_target"][target_index]
            confusion_matrix[target][prediction] += 1
    predicted_pairwise = np.asarray(
        [value for row in records for value in row["predicted_pairwise_rmse_rad"]]
    )
    expert_pairwise = np.asarray(
        [value for row in records for value in row["expert_pairwise_rmse_rad"]]
    )
    reference_scene_deltas = [
        row["reference_scene_max_object_position_delta_m"] for row in records
    ]
    reference_state_deltas = [
        row["reference_initial_state_max_abs_delta_rad"] for row in records
    ]
    exact_reference_scenes = bool(
        max(reference_scene_deltas) <= 1e-12
        and max(reference_state_deltas) <= 1e-12
    )
    report = {
        "experiment": "G1LANG-counterfactual-initial-action",
        "checkpoint": str(args.checkpoint),
        "device": device,
        "scenes": scene_count,
        "queries": len(correct),
        "nearest_expert_classification_accuracy": float(np.mean(correct)),
        "per_target": per_target,
        "confusion_matrix_true_by_nearest_expert": confusion_matrix,
        "mean_predicted_pairwise_rmse_rad": float(np.mean(predicted_pairwise)),
        "mean_expert_pairwise_rmse_rad": float(np.mean(expert_pairwise)),
        "language_separation_ratio": float(
            np.mean(predicted_pairwise) / np.mean(expert_pairwise)
        ),
        "chance_accuracy": 1.0 / len(TARGETS),
        "metric_unit": "joint radians",
        "query_contract": (
            "Within each group, all predictions use the first episode image/state and "
            "the same seeded noise; only the language task changes."
        ),
        "reference_contract": (
            "Nearest-expert labels use target-matched validation episodes from the "
            + (
                "same exact scene and initial robot state."
                if exact_reference_scenes
                else "same sequential group; their scene or initial robot state differs."
            )
        ),
        "mean_reference_scene_max_object_position_delta_m": float(
            np.mean(reference_scene_deltas)
        ),
        "max_reference_scene_max_object_position_delta_m": float(
            np.max(reference_scene_deltas)
        ),
        "max_reference_initial_state_max_abs_delta_rad": float(
            np.max(reference_state_deltas)
        ),
        "records": records,
    }
    if not math.isfinite(report["language_separation_ratio"]):
        raise ValueError("Non-finite language separation ratio")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
