#!/usr/bin/env python3
"""Compare base, step-500, and step-1000 G1 SmolVLA policies on held-out data."""

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
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.constants import ACTION
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


PHASE_NAMES = {
    0: "home",
    1: "approach_pregrasp",
    2: "approach_grasp",
    3: "close_hands",
    4: "lift",
    5: "hold",
}
JOINT_GROUPS = {
    "waist": slice(0, 3),
    "left_arm": slice(3, 10),
    "left_hand": slice(10, 17),
    "right_arm": slice(17, 24),
    "right_hand": slice(24, 31),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-policy", type=Path, required=True)
    parser.add_argument("--checkpoint-500", type=Path, required=True)
    parser.add_argument("--checkpoint-1000", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_assisted_lift_train")
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-repo-id", default="local/g1_assisted_lift_val")
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2507)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def make_dataset(args: argparse.Namespace) -> LeRobotDataset:
    metadata = LeRobotDatasetMetadata(args.val_repo_id, root=args.val_root)
    config = PreTrainedConfig.from_pretrained(args.checkpoint_1000)
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
    return make_policy(config, ds_meta=train_meta).eval()


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
    return result


def grouped(records: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in records:
        groups[str(row[key])].append(row)
    return {name: aggregate(rows) for name, rows in sorted(groups.items())}


@torch.no_grad()
def evaluate_policy(name, path, train_meta, loader, device, seed):
    policy = load_policy(path, train_meta, device)
    action_dim = policy.config.action_feature.shape[0]
    if action_dim != 31:
        raise ValueError(f"Expected 31 actions, found {action_dim}")
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
        prediction = policy.predict_action_chunk(
            inference_batch,
            noise=seeded_noise(
                policy, batch_size, seed + 100_000 + batch_index, device
            ),
        )
        error = prediction - target
        phases = cpu_batch["complementary_info.task_phase"].reshape(-1).tolist()
        episodes = cpu_batch["episode_index"].reshape(-1).tolist()
        frames = cpu_batch["frame_index"].reshape(-1).tolist()
        tasks = cpu_batch["task"]
        for sample in range(batch_size):
            sample_valid = valid[sample]
            chunk_error = error[sample, sample_valid]
            first_error = error[sample, 0]
            sample_flow = flow_losses[sample, sample_valid]
            row = {
                "policy": name,
                "episode": int(episodes[sample]),
                "frame": int(frames[sample]),
                "phase": PHASE_NAMES[int(phases[sample])],
                "task": tasks[sample],
                "valid_steps": int(sample_valid.sum()),
                "flow_squared_sum": float(sample_flow.sum()),
                "flow_count": sample_flow.numel(),
            }
            for horizon, values in (("chunk", chunk_error), ("first", first_error)):
                row[f"{horizon}_all_squared_sum"] = float(values.square().sum())
                row[f"{horizon}_all_count"] = values.numel()
                for group, group_slice in JOINT_GROUPS.items():
                    grouped_values = values[..., group_slice]
                    row[f"{horizon}_{group}_squared_sum"] = float(
                        grouped_values.square().sum()
                    )
                    row[f"{horizon}_{group}_count"] = grouped_values.numel()
            row["sample_chunk_all_rmse_rad"] = float(
                torch.sqrt(chunk_error.square().mean())
            )
            row["sample_first_all_rmse_rad"] = float(
                torch.sqrt(first_error.square().mean())
            )
            records.append(row)
        print(f"{name}: batch {batch_index + 1}/{len(loader)}", flush=True)
    result = {
        "path": str(path),
        "elapsed_s": time.perf_counter() - started,
        "training_objective_loss_mean": float(np.mean(objective_losses)),
        "overall": aggregate(records),
        "by_phase": grouped(records, "phase"),
        "by_task": grouped(records, "task"),
    }
    del policy
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, records


def compare(candidate: dict, base: dict) -> dict:
    fields = (
        "flow_action_mse",
        "chunk_all_rmse_rad",
        "first_all_rmse_rad",
        "chunk_waist_rmse_rad",
        "chunk_left_arm_rmse_rad",
        "chunk_left_hand_rmse_rad",
        "chunk_right_arm_rmse_rad",
        "chunk_right_hand_rmse_rad",
    )
    return {
        field: candidate["overall"][field] / base["overall"][field]
        for field in fields
    }


def write_svg(path: Path, comparisons: dict) -> None:
    fields = (
        "chunk_all_rmse_rad",
        "chunk_waist_rmse_rad",
        "chunk_left_arm_rmse_rad",
        "chunk_left_hand_rmse_rad",
        "chunk_right_arm_rmse_rad",
        "chunk_right_hand_rmse_rad",
    )
    labels = ("All", "Waist", "L arm", "L hand", "R arm", "R hand")
    width, height = 980, 480
    left, right, top, bottom = 75, 30, 55, 85
    values = [comparisons[name][field] for name in comparisons for field in fields]
    maximum = max(1.15, max(values) * 1.08)
    plot_h = height - top - bottom
    slot = (width - left - right) / len(fields)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px;fill:#333}.axis{stroke:#aaa}.threshold{stroke:#c62828;stroke-dasharray:6 4}</style>',
        '<text x="20" y="30" font-size="18">G1WH-26 held-out chunk RMSE ratio vs base</text>',
    ]
    y_one = height - bottom - plot_h / maximum
    parts.append(f'<line class="threshold" x1="{left}" y1="{y_one:.1f}" x2="{width-right}" y2="{y_one:.1f}"/>')
    parts.append(f'<text x="20" y="{y_one+5:.1f}">1.0x</text>')
    colors = {"step500": "#1976d2", "step1000": "#2e7d32"}
    for field_index, (field, label) in enumerate(zip(fields, labels, strict=True)):
        base_x = left + field_index * slot
        for candidate_index, name in enumerate(("step500", "step1000")):
            value = comparisons[name][field]
            bar_w = slot * 0.30
            x = base_x + slot * (0.17 + candidate_index * 0.34)
            bar_h = value / maximum * plot_h
            y = height - bottom - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colors[name]}"/>')
            parts.append(f'<text x="{x:.1f}" y="{y-6:.1f}">{value:.2f}x</text>')
        parts.append(f'<text x="{base_x+slot*0.28:.1f}" y="{height-bottom+28}">{label}</text>')
    parts.extend([
        '<rect x="690" y="15" width="14" height="14" fill="#1976d2"/><text x="710" y="27">step500</text>',
        '<rect x="790" y="15" width="14" height="14" fill="#2e7d32"/><text x="810" y="27">step1000</text>',
        '</svg>',
    ])
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.batch_size < 1:
        raise ValueError("samples and batch-size must be positive")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)
    dataset = make_dataset(args)
    count = min(args.samples, len(dataset))
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    specs = {
        "base": args.base_policy,
        "step500": args.checkpoint_500,
        "step1000": args.checkpoint_1000,
    }
    results = {}
    all_records = []
    for model_name, model_path in specs.items():
        results[model_name], records = evaluate_policy(
            model_name, model_path, train_meta, loader, device, args.seed
        )
        all_records.extend(records)
    comparisons = {
        name: compare(results[name], results["base"])
        for name in ("step500", "step1000")
    }
    report = {
        "experiment": "G1WH-26-heldout-action-evaluation",
        "device": device,
        "validation_episodes": dataset.num_episodes,
        "validation_frames": len(dataset),
        "evaluated_samples": count,
        "seed": args.seed,
        "metric_unit": "joint radians",
        "results": results,
        "ratios_vs_base": comparisons,
        "reference": {
            "good": "candidate/base <= 0.8",
            "partial": "0.8 < candidate/base < 1.0",
            "regression": "candidate/base >= 1.0",
            "scope": "Project screening thresholds, not universal SmolVLA standards.",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    fields = (
        "policy",
        "episode",
        "frame",
        "phase",
        "task",
        "valid_steps",
        "sample_chunk_all_rmse_rad",
        "sample_first_all_rmse_rad",
    )
    with (args.output_dir / "per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    write_svg(args.output_dir / "comparison.svg", comparisons)
    print(json.dumps(report, indent=2))
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
