#!/usr/bin/env python3
"""Evaluate G1 language SmolVLA policies on held-out expert actions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import safetensors.torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.constants import ACTION
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import standardise_state_dict
from g1_language_action_adapter import attach_language_action_adapter


PHASE_NAMES = {
    0: "home",
    1: "high_pregrasp",
    2: "pregrasp_to_grasp",
    3: "grasp_hold",
    4: "lift",
    5: "place_approach",
    6: "place_hold",
    7: "retreat",
}
TARGET_NAMES = ("red_triangle", "yellow_rod", "green_cube")
JOINT_GROUPS = {
    "waist": slice(0, 3),
    "left_arm": slice(3, 10),
    "left_hand": slice(10, 17),
    "right_arm": slice(17, 24),
    "right_hand": slice(24, 31),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-policy", type=Path)
    parser.add_argument("--train-repo-id", default="local/g1_language_pick_place_train")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-repo-id", default="local/g1_language_pick_place_val")
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--inference-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def make_dataset(args: argparse.Namespace, config_path: Path) -> LeRobotDataset:
    metadata = LeRobotDatasetMetadata(args.val_repo_id, root=args.val_root)
    config = PreTrainedConfig.from_pretrained(config_path)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    return LeRobotDataset(
        args.val_repo_id,
        root=args.val_root,
        delta_timestamps=resolve_delta_timestamps(config, metadata),
        video_backend="pyav",
    )


def load_policy(path: Path, train_meta: LeRobotDatasetMetadata, device: str):
    config = PreTrainedConfig.from_pretrained(path)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig for {path}")
    config.device = device
    config.pretrained_path = path
    try:
        policy = make_policy(config, ds_meta=train_meta)
    except RuntimeError as error:
        # Adapter checkpoints contain project-owned keys that the stock
        # LeRobot loader quite intentionally rejects. Instantiate first,
        # attach the adapter, then load the compatible subset explicitly.
        if "unexpected" not in str(error):
            raise
        config.pretrained_path = None
        policy = make_policy(config, ds_meta=train_meta)
        attach_language_action_adapter(policy)
        state = safetensors.torch.load_file(path / "model.safetensors", device=device)
        state, _ = standardise_state_dict(state, set(policy.state_dict().keys()), verbose=False)
        state = {
            key: value for key, value in state.items()
            if not key.startswith(("normalize_inputs", "normalize_targets", "unnormalize_outputs"))
        }
        missing, unexpected = policy.load_state_dict(state, strict=False)
        allowed_missing = {
            key for key in missing
            if key.startswith(("normalize_inputs", "normalize_targets", "unnormalize_outputs"))
            or key.startswith("model.language_action_adapter.action_output_proj.")
            or key.startswith("model.language_action_adapter.action_film_proj.")
            or key.startswith("model.language_action_adapter.target_classifier.")
            or key.startswith("model.language_action_adapter.target_action_chunk_proj.")
            or key.startswith("model.language_action_adapter.target_action_hidden_proj.")
        }
        if unexpected or set(missing) != allowed_missing:
            raise RuntimeError(f"Adapter checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
    else:
        # Source checkpoints predate the adapter; attach it after stock load.
        attach_language_action_adapter(policy)
    policy = policy.eval()
    action_dim = policy.config.action_feature.shape[0]
    if action_dim != 31:
        raise ValueError(f"Expected 31 actions, found {action_dim}")
    return policy


def to_device(batch: dict, device: str) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def seeded_noise(policy, batch_size: int, seed: int, device: str) -> torch.Tensor:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    return torch.randn(
        batch_size,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        device=device,
    )


def seeded_flow(policy, batch_size: int, seed: int, device: str):
    noise = seeded_noise(policy, batch_size, seed, device)
    time_values = torch.distributions.Beta(1.5, 1.0).sample((batch_size,)).to(device)
    return noise, time_values * 0.999 + 0.001


def add_error_sums(row: dict, prefix: str, values: torch.Tensor) -> None:
    row[f"{prefix}_all_squared_sum"] = float(values.square().sum())
    row[f"{prefix}_all_count"] = values.numel()
    for group, group_slice in JOINT_GROUPS.items():
        grouped_values = values[..., group_slice]
        row[f"{prefix}_{group}_squared_sum"] = float(grouped_values.square().sum())
        row[f"{prefix}_{group}_count"] = grouped_values.numel()


def aggregate(records: list[dict]) -> dict:
    result = {
        "samples": len(records),
        "valid_action_steps": sum(row["valid_steps"] for row in records),
        "flow_action_mse": sum(row["flow_squared_sum"] for row in records)
        / sum(row["flow_count"] for row in records),
    }
    for horizon in ("chunk", "first"):
        for group in ("all", *JOINT_GROUPS):
            result[f"{horizon}_{group}_rmse_rad"] = math.sqrt(
                sum(row[f"{horizon}_{group}_squared_sum"] for row in records)
                / sum(row[f"{horizon}_{group}_count"] for row in records)
            )
    for prefix in (
        "predicted_entry_delta",
        "expert_entry_delta",
        "predicted_step_delta",
        "expert_step_delta",
        "chunk_noise",
        "first_noise",
    ):
        result[f"{prefix}_rmse_rad"] = math.sqrt(
            sum(row[f"{prefix}_squared_sum"] for row in records)
            / sum(row[f"{prefix}_count"] for row in records)
        )
    result["predicted_to_expert_entry_delta_ratio"] = (
        result["predicted_entry_delta_rmse_rad"]
        / result["expert_entry_delta_rmse_rad"]
    )
    result["predicted_to_expert_step_delta_ratio"] = (
        result["predicted_step_delta_rmse_rad"]
        / result["expert_step_delta_rmse_rad"]
    )
    return result


def grouped(records: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in records:
        groups[str(row[key])].append(row)
    return {name: aggregate(rows) for name, rows in sorted(groups.items())}


@torch.no_grad()
def evaluate_policy(
    name: str,
    path: Path,
    train_meta: LeRobotDatasetMetadata,
    loader: DataLoader,
    device: str,
    seed: int,
    inference_repeats: int,
) -> tuple[dict, list[dict]]:
    policy = load_policy(path, train_meta, device)
    action_dim = policy.config.action_feature.shape[0]
    records = []
    objective_losses = []
    started = time.perf_counter()
    for batch_index, cpu_batch in enumerate(loader):
        target = cpu_batch[ACTION].to(device)
        valid = ~cpu_batch[f"{ACTION}_is_pad"].to(device)
        batch_size = target.shape[0]

        flow_batch = to_device(cpu_batch, device)
        flow_noise, flow_time = seeded_flow(
            policy, batch_size, seed + batch_index, device
        )
        objective, output = policy.forward(
            flow_batch, noise=flow_noise, time=flow_time
        )
        flow_losses = output["losses_after_forward"][:, :, :action_dim]
        objective_losses.append(float(objective))

        inference_batch = to_device(
            {
                key: value
                for key, value in cpu_batch.items()
                if key.startswith("observation.") or key == "task"
            },
            device,
        )
        predictions = torch.stack(
            [
                policy.predict_action_chunk(
                    inference_batch,
                    noise=seeded_noise(
                        policy,
                        batch_size,
                        seed + 100_000 + repeat * 1_000_000 + batch_index,
                        device,
                    ),
                )
                for repeat in range(inference_repeats)
            ]
        )
        prediction = predictions[0]
        error = prediction - target
        phases = cpu_batch["complementary_info.task_phase"].reshape(-1).tolist()
        targets = (
            cpu_batch["complementary_info.target_object_index"].reshape(-1).tolist()
        )
        episodes = cpu_batch["episode_index"].reshape(-1).tolist()
        frames = cpu_batch["frame_index"].reshape(-1).tolist()
        tasks = cpu_batch["task"]
        for sample in range(batch_size):
            sample_valid = valid[sample]
            chunk_error = error[sample, sample_valid]
            first_error = error[sample, 0]
            sample_flow = flow_losses[sample, sample_valid]
            predicted_actions = prediction[sample, sample_valid]
            expert_actions = target[sample, sample_valid]
            state = cpu_batch["observation.state"][sample].to(device)
            predicted_entry_delta = predicted_actions[0] - state
            expert_entry_delta = expert_actions[0] - state
            predicted_step_delta = torch.diff(predicted_actions, dim=0)
            expert_step_delta = torch.diff(expert_actions, dim=0)
            chunk_noise = predictions[:, sample, sample_valid].var(
                dim=0, unbiased=False
            )
            first_noise = predictions[:, sample, 0].var(dim=0, unbiased=False)
            target_index = int(targets[sample])
            row = {
                "policy": name,
                "episode": int(episodes[sample]),
                "frame": int(frames[sample]),
                "phase": PHASE_NAMES.get(int(phases[sample]), str(phases[sample])),
                "target_object": TARGET_NAMES[target_index],
                "task": tasks[sample],
                "valid_steps": int(sample_valid.sum()),
                "flow_squared_sum": float(sample_flow.sum()),
                "flow_count": sample_flow.numel(),
                "predicted_entry_delta_squared_sum": float(
                    predicted_entry_delta.square().sum()
                ),
                "predicted_entry_delta_count": predicted_entry_delta.numel(),
                "expert_entry_delta_squared_sum": float(
                    expert_entry_delta.square().sum()
                ),
                "expert_entry_delta_count": expert_entry_delta.numel(),
                "predicted_step_delta_squared_sum": float(
                    predicted_step_delta.square().sum()
                ),
                "predicted_step_delta_count": predicted_step_delta.numel(),
                "expert_step_delta_squared_sum": float(
                    expert_step_delta.square().sum()
                ),
                "expert_step_delta_count": expert_step_delta.numel(),
                "chunk_noise_squared_sum": float(chunk_noise.sum()),
                "chunk_noise_count": chunk_noise.numel(),
                "first_noise_squared_sum": float(first_noise.sum()),
                "first_noise_count": first_noise.numel(),
            }
            add_error_sums(row, "chunk", chunk_error)
            add_error_sums(row, "first", first_error)
            row["sample_chunk_all_rmse_rad"] = float(
                torch.sqrt(chunk_error.square().mean())
            )
            row["sample_first_all_rmse_rad"] = float(
                torch.sqrt(first_error.square().mean())
            )
            row["sample_predicted_step_delta_rmse_rad"] = float(
                torch.sqrt(predicted_step_delta.square().mean())
            )
            row["sample_expert_step_delta_rmse_rad"] = float(
                torch.sqrt(expert_step_delta.square().mean())
            )
            row["sample_first_noise_std_rad"] = float(
                torch.sqrt(first_noise.mean())
            )
            records.append(row)
        print(f"{name}: batch {batch_index + 1}/{len(loader)}", flush=True)

    result = {
        "path": str(path),
        "elapsed_s": time.perf_counter() - started,
        "training_objective_loss_mean": float(np.mean(objective_losses)),
        "overall": aggregate(records),
        "by_phase": grouped(records, "phase"),
        "by_target": grouped(records, "target_object"),
        "by_task": grouped(records, "task"),
    }
    del policy
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, records


def ratios(candidate: dict, baseline: dict) -> dict:
    fields = (
        "flow_action_mse",
        "chunk_all_rmse_rad",
        "first_all_rmse_rad",
        "chunk_waist_rmse_rad",
        "chunk_left_arm_rmse_rad",
        "chunk_left_hand_rmse_rad",
        "chunk_right_arm_rmse_rad",
        "chunk_right_hand_rmse_rad",
        "predicted_entry_delta_rmse_rad",
        "predicted_step_delta_rmse_rad",
        "chunk_noise_rmse_rad",
        "first_noise_rmse_rad",
    )
    return {
        field: (
            candidate["overall"][field] / baseline["overall"][field]
            if baseline["overall"][field] > 0
            else None
        )
        for field in fields
    }


def write_csv(path: Path, records: list[dict]) -> None:
    fields = (
        "policy",
        "episode",
        "frame",
        "phase",
        "target_object",
        "task",
        "valid_steps",
        "sample_chunk_all_rmse_rad",
        "sample_first_all_rmse_rad",
        "sample_predicted_step_delta_rmse_rad",
        "sample_expert_step_delta_rmse_rad",
        "sample_first_noise_std_rad",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.inference_repeats < 1:
        raise ValueError("samples, batch-size, and inference-repeats must be positive")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)
    dataset = make_dataset(args, args.checkpoint)
    count = min(args.samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    specs = {"candidate": args.checkpoint}
    if args.baseline_policy:
        specs["baseline"] = args.baseline_policy

    results = {}
    all_records = []
    for name, path in specs.items():
        results[name], records = evaluate_policy(
            name,
            path,
            train_meta,
            loader,
            device,
            args.seed,
            args.inference_repeats,
        )
        all_records.extend(records)

    report = {
        "experiment": "G1LANG-03-heldout-action-diagnostic",
        "device": device,
        "validation_episodes": dataset.num_episodes,
        "validation_frames": len(dataset),
        "evaluated_samples": count,
        "inference_repeats": args.inference_repeats,
        "seed": args.seed,
        "metric_unit": "joint radians",
        "results": results,
        "reference": {
            "first_all_rmse_rad_good": "<=0.08 rad, roughly <=4.6 deg",
            "first_all_rmse_rad_usable": "0.08-0.15 rad, roughly 4.6-8.6 deg",
            "first_all_rmse_rad_poor": ">=0.25 rad, roughly >=14.3 deg",
            "scope": "Local project diagnostic thresholds, not universal VLA standards.",
        },
    }
    if args.baseline_policy:
        report["candidate_vs_baseline_ratio"] = ratios(
            results["candidate"], results["baseline"]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "per_sample.csv", all_records)
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
