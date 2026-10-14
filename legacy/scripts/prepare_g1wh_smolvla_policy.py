#!/usr/bin/env python3
"""Adapt local SmolVLA weights to the G1 upper-body dataset feature contract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment-id", default="G1WH-25-smolvla-upper-body-finetune"
    )
    parser.add_argument("--refresh-normalization-stats", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.base_policy.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Adapted policy already exists: {output}")
        shutil.rmtree(output)
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    dataset_info = json.loads(
        (args.dataset_root / "meta/info.json").read_text(encoding="utf-8")
    )
    state_shape = dataset_info["features"]["observation.state"]["shape"]
    action_shape = dataset_info["features"]["action"]["shape"]
    cameras = {}
    for key, feature in dataset_info["features"].items():
        if feature["dtype"] not in {"video", "image"}:
            continue
        shape = list(feature["shape"])
        names = feature["names"]
        if names[2] in {"channel", "channels"}:
            shape = [shape[2], shape[0], shape[1]]
        cameras[key] = {"type": "VISUAL", "shape": shape}
    if state_shape[0] > source_config["max_state_dim"]:
        raise ValueError(
            f"State dimension {state_shape[0]} exceeds {source_config['max_state_dim']}"
        )
    if action_shape[0] > source_config["max_action_dim"]:
        raise ValueError(
            f"Action dimension {action_shape[0]} exceeds {source_config['max_action_dim']}"
        )
    if len(cameras) != 3:
        raise ValueError(f"Expected three task cameras, found {sorted(cameras)}")

    adapted = dict(source_config)
    adapted["input_features"] = {
        "observation.state": {"type": "STATE", "shape": state_shape},
        **cameras,
    }
    adapted["output_features"] = {"action": {"type": "ACTION", "shape": action_shape}}
    adapted["push_to_hub"] = False
    adapted["repo_id"] = None
    output.mkdir(parents=True)
    (output / "config.json").write_text(
        json.dumps(adapted, indent=2) + "\n", encoding="utf-8"
    )
    source_weights = (source / "model.safetensors").resolve()
    normalization_report = None
    if args.refresh_normalization_stats:
        tensors = load_file(source_weights, device="cpu")
        dataset_meta = LeRobotDatasetMetadata(
            args.dataset_repo_id, root=args.dataset_root
        )
        replacements = {
            "normalize_inputs.buffer_observation_state.mean": (
                "observation.state",
                "mean",
            ),
            "normalize_inputs.buffer_observation_state.std": (
                "observation.state",
                "std",
            ),
            "normalize_targets.buffer_action.mean": ("action", "mean"),
            "normalize_targets.buffer_action.std": ("action", "std"),
            "unnormalize_outputs.buffer_action.mean": ("action", "mean"),
            "unnormalize_outputs.buffer_action.std": ("action", "std"),
        }
        changes = {}
        for key, (feature, statistic) in replacements.items():
            if key not in tensors:
                raise KeyError(f"Source policy is missing normalization tensor: {key}")
            old = tensors[key]
            new = torch.as_tensor(
                dataset_meta.stats[feature][statistic], dtype=old.dtype
            ).reshape(old.shape)
            tensors[key] = new
            changes[key] = {
                "old_min": float(old.min()),
                "old_max": float(old.max()),
                "new_min": float(new.min()),
                "new_max": float(new.max()),
            }
        with safe_open(source_weights, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        save_file(tensors, output / "model.safetensors", metadata=metadata)
        normalization_report = {
            "refreshed_from_dataset": True,
            "tensor_changes": changes,
        }
    else:
        os.symlink(source_weights, output / "model.safetensors")

    report = {
        "experiment": args.experiment_id,
        "source_policy": str(source),
        "network_weights_reused_without_modification": True,
        "all_serialized_tensors_reused_without_modification": (
            not args.refresh_normalization_stats
        ),
        "processors_managed_by_training_stack": True,
        "dataset_repo_id": args.dataset_repo_id,
        "state_dim": state_shape[0],
        "action_dim": action_shape[0],
        "camera_keys": sorted(cameras),
        "max_state_dim": source_config["max_state_dim"],
        "max_action_dim": source_config["max_action_dim"],
        "freeze_vision_encoder": source_config["freeze_vision_encoder"],
        "train_expert_only": source_config["train_expert_only"],
        "normalization": normalization_report,
    }
    (output / "adaptation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
