#!/usr/bin/env python3
"""Audit whether a language-conditioned G1 policy uses camera input for action grounding."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_g1_language_smolvla_heldout import load_policy, make_dataset, seeded_noise
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


TARGETS = ("red_triangle", "yellow_rod", "green_cube")
IMAGE_KEYS = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_language_paired_train")
    parser.add_argument("--val-repo-id", default="local/g1_language_paired_val")
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def batch_from_item(item: dict, task: str, device: str) -> dict:
    batch = {
        key: value.unsqueeze(0).to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in item.items()
        if key.startswith("observation.") and not key.endswith("_is_pad")
    }
    batch["task"] = [task]
    return batch


def replace_images(batch: dict, source: dict | None) -> dict:
    result = dict(batch)
    for key in IMAGE_KEYS:
        if source is None:
            result[key] = torch.zeros_like(batch[key])
        else:
            result[key] = source[key].unsqueeze(0).to(batch[key].device)
    return result


def action_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((left.float() - right.float()) ** 2)))


def finite(value: object, path: str = "report") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite value at {path}: {value}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.scenes < 1 or args.frame_index < 0:
        raise ValueError("scenes must be positive and frame-index must be non-negative")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)
    dataset = make_dataset(args, args.checkpoint)
    available_scenes = dataset.num_episodes // len(TARGETS)
    scene_count = min(args.scenes, available_scenes)
    if scene_count < 1:
        raise ValueError("Validation dataset contains no complete target triplets")
    starts = [int(dataset.episode_data_index["from"][index]) for index in range(dataset.num_episodes)]
    grouped = []
    for scene_index in range(scene_count):
        items = [
            dataset[starts[scene_index * len(TARGETS) + target] + args.frame_index]
            for target in range(len(TARGETS))
        ]
        target_indices = [int(item["complementary_info.target_object_index"]) for item in items]
        if target_indices != list(range(len(TARGETS))):
            raise ValueError(f"Scene {scene_index} target order is {target_indices}")
        grouped.append(items)

    policy = load_policy(args.checkpoint, train_meta, device).eval()
    base_item = grouped[0][0]
    base_state = base_item["observation.state"]
    scene_records = []
    fixed_state_action_records = []
    target_classifier_records = []
    for scene_index, items in enumerate(grouped):
        intact_actions = []
        zero_actions = []
        intact_posteriors = []
        zero_posteriors = []
        for target_index, task_item in enumerate(grouped[0]):
            task = task_item["task"]
            base_batch = batch_from_item(
                {**base_item, "observation.state": base_state}, task, device
            )
            intact_batch = replace_images(base_batch, items[target_index])
            zero_batch = replace_images(base_batch, None)
            noise = seeded_noise(policy, 1, args.seed + target_index, device)
            intact = policy.predict_action_chunk(intact_batch, noise=noise)[0].detach().cpu()
            zero = policy.predict_action_chunk(zero_batch, noise=noise)[0].detach().cpu()
            intact_actions.append(intact)
            zero_actions.append(zero)
            intact_logits = policy.model._language_action_target_logits.softmax(-1)[0].detach().cpu()
            intact_posteriors.append(intact_logits)
            zero_logits = policy.model._language_action_target_logits.softmax(-1)[0].detach().cpu()
            zero_posteriors.append(zero_logits)
        scene_records.append(
            {
                "scene_index": scene_index,
                "image_mean": {
                    key: float(items[0][key].float().mean()) for key in IMAGE_KEYS
                },
                "same_state_intact_vs_zero_action_rmse_rad": [
                    action_rmse(intact, zero)
                    for intact, zero in zip(intact_actions, zero_actions, strict=True)
                ],
                "same_state_intact_vs_zero_target_posterior_rmse": [
                    action_rmse(intact, zero)
                    for intact, zero in zip(intact_posteriors, zero_posteriors, strict=True)
                ],
                "intact_action_pairwise_rmse_rad": [
                    action_rmse(intact_actions[left], intact_actions[right])
                    for left, right in ((0, 1), (0, 2), (1, 2))
                ],
                "zero_action_pairwise_rmse_rad": [
                    action_rmse(zero_actions[left], zero_actions[right])
                    for left, right in ((0, 1), (0, 2), (1, 2))
                ],
                "intact_target_predictions": [
                    TARGETS[int(torch.argmax(posterior))] for posterior in intact_posteriors
                ],
                "zero_target_predictions": [
                    TARGETS[int(torch.argmax(posterior))] for posterior in zero_posteriors
                ],
            }
        )

        # Hold language and pose fixed while replacing only the camera images
        # with another scene. This isolates visual grounding from state input.
        if scene_index > 0:
            reference_items = grouped[scene_index - 1]
            for target_index, task_item in enumerate(grouped[0]):
                task = task_item["task"]
                base_batch = batch_from_item({**base_item, "observation.state": base_state}, task, device)
                current = replace_images(base_batch, items[target_index])
                reference = replace_images(base_batch, reference_items[target_index])
                noise = seeded_noise(policy, 1, args.seed + target_index, device)
                current_action = policy.predict_action_chunk(current, noise=noise)[0].detach().cpu()
                reference_action = policy.predict_action_chunk(reference, noise=noise)[0].detach().cpu()
                fixed_state_action_records.append(
                    {
                        "scene_pair": [scene_index - 1, scene_index],
                        "target": TARGETS[target_index],
                        "cross_scene_image_action_rmse_rad": action_rmse(current_action, reference_action),
                    }
                )

    report = {
        "experiment": "G1LANG-multimodal-camera-grounding-audit",
        "checkpoint": str(args.checkpoint),
        "device": device,
        "scenes": scene_count,
        "frame_index": args.frame_index,
        "fixed_pose_contract": "All action comparisons use scene-0 observation.state; only camera tensors are replaced.",
        "privileged_object_position_used": False,
        "scene_records": scene_records,
        "cross_scene_fixed_state_records": fixed_state_action_records,
        "mean_same_state_intact_vs_zero_action_rmse_rad": float(
            np.mean([
                value
                for record in scene_records
                for value in record["same_state_intact_vs_zero_action_rmse_rad"]
            ])
        ),
        "mean_cross_scene_fixed_state_image_action_rmse_rad": float(
            np.mean([item["cross_scene_image_action_rmse_rad"] for item in fixed_state_action_records])
            if fixed_state_action_records
            else 0.0
        ),
        "mean_same_state_intact_vs_zero_target_posterior_rmse": float(
            np.mean([
                value
                for record in scene_records
                for value in record["same_state_intact_vs_zero_target_posterior_rmse"]
            ])
        ),
        "camera_input_nonzero": all(
            abs(record["image_mean"][key]) > 1e-6
            for record in scene_records
            for key in IMAGE_KEYS
        ),
        "acceptance": {
            "camera_action_sensitivity_min_rmse_rad": 0.005,
            "requires_cross_scene_fixed_state_action_sensitivity": True,
        },
    }
    report["camera_action_sensitivity_passed"] = bool(
        report["mean_same_state_intact_vs_zero_action_rmse_rad"] >= 0.005
        and report["mean_cross_scene_fixed_state_image_action_rmse_rad"] >= 0.005
    )
    finite(report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"scene_records", "cross_scene_fixed_state_records"}}, indent=2))
    print(f"Saved {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
