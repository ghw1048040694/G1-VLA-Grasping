#!/usr/bin/env python3
"""Train a phase-free local direct-joint VLA from strict dynamic episodes."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from g1_direct_joint_candidate_vla import ACTION_DIM, CANDIDATE_COUNT, TARGET_COUNT, encode_tasks
from g1_local_joint_vla import LocalJointVLA, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--history-frames", type=int, default=8)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "Optional compatible VLA checkpoint used to initialize fine-tuning. "
            "The new cache still defines all normalization and action bounds."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def flatten(cache: dict) -> dict[str, torch.Tensor | list]:
    episode_count, frames = cache["state"].shape[:2]
    color = cache.get("multiview_color_features", cache["color_features"])
    if color.ndim == 2:
        color = color[:, None].expand(-1, frames, -1)
    objects = cache["visual_object_position"]
    if objects.ndim == 2:
        objects = objects[:, None].expand(-1, frames, -1)
    contact_sites = cache.get(
        "finger_contact_site_position",
        cache.get("contact_pad_position", cache["palm_position"]),
    )
    if contact_sites.ndim == 2:
        contact_sites = contact_sites.reshape(episode_count, frames, -1)
    valid = cache.get(
        "valid_frame", torch.ones((episode_count, frames), dtype=torch.bool)
    ).reshape(-1).bool()
    action_representation = cache.get(
        "action_representation", "absolute_joint_position"
    )
    if action_representation not in {
        "absolute_joint_position",
        "relative_joint_delta",
    }:
        raise ValueError(f"Unsupported cache action representation: {action_representation}")
    return {
        "state": cache["state"].reshape(-1, 31),
        "velocity": cache["velocity"].reshape(-1, 31),
        "contact_sites": contact_sites.reshape(-1, contact_sites.shape[-1]),
        "tactile_contact": cache["physical_contact_active"].reshape(-1, 1),
        "action": cache["candidate_action"].reshape(-1, CANDIDATE_COUNT, ACTION_DIM),
        "arm": cache["active_arm"].reshape(-1),
        "target": cache["target"][:, None].expand(-1, frames).reshape(-1),
        "task_index": cache["task_index"][:, None].expand(-1, frames).reshape(-1),
        "color": color.reshape(-1, color.shape[-1]),
        "objects": objects.reshape(-1, 9),
        "tasks": cache["tasks"],
        "vocab": cache["vocab"],
        "offsets": cache["candidate_offsets_frames"],
        "episodes": episode_count,
        "vision_dim": color.shape[-1],
        "contact_site_dim": contact_sites.shape[-1],
        "action_representation": action_representation,
        "valid": valid,
    }


def statistics(data: dict) -> dict[str, torch.Tensor]:
    valid = data["valid"]
    if not bool(valid.any()):
        raise ValueError("Cache has no valid supervised frames")
    action = data["action"][valid]
    lower = torch.empty(
        (TARGET_COUNT, 2, CANDIDATE_COUNT, ACTION_DIM), dtype=torch.float32
    )
    upper = torch.empty_like(lower)
    fallback_lower = torch.quantile(action, 0.002, dim=0)
    fallback_upper = torch.quantile(action, 0.998, dim=0)
    for target in range(TARGET_COUNT):
        for arm in range(2):
            values = data["action"][
                valid & (data["target"] == target) & (data["arm"] == arm)
            ]
            lower[target, arm] = (
                torch.quantile(values, 0.002, dim=0)
                if len(values)
                else fallback_lower
            )
            upper[target, arm] = (
                torch.quantile(values, 0.998, dim=0)
                if len(values)
                else fallback_upper
            )
    return {
        "state_mean": data["state"][valid].mean(0),
        "state_std": data["state"][valid].std(0).clamp_min(1e-3),
        "velocity_mean": data["velocity"][valid].mean(0),
        "velocity_std": data["velocity"][valid].std(0).clamp_min(1e-3),
        "action_mean": action.mean((0, 1)),
        "action_std": action.std((0, 1)).clamp_min(1e-3),
        "candidate_lower": lower,
        "candidate_upper": upper,
        "candidate_offsets_frames": data["offsets"].clone(),
    }


def causal_history(
    data: dict, stats: dict[str, torch.Tensor], history_frames: int
) -> torch.Tensor:
    if history_frames < 1:
        raise ValueError("history_frames must be positive")
    frames = int(data["state"].shape[0] // data["episodes"])
    current = torch.cat(
        (
            (data["state"] - stats["state_mean"]) / stats["state_std"],
            (data["velocity"] - stats["velocity_mean"]) / stats["velocity_std"],
            data["objects"],
            data["contact_sites"],
            data["tactile_contact"],
            data["color"],
        ),
        dim=-1,
    ).reshape(data["episodes"], frames, -1)
    offsets = torch.arange(history_frames - 1, -1, -1)
    indices = (torch.arange(frames)[:, None] - offsets[None]).clamp_min(0)
    return current[:, indices].reshape(-1, history_frames, current.shape[-1])


def dataset(
    data: dict, stats: dict[str, torch.Tensor], history_frames: int
) -> TensorDataset:
    ids, mask = encode_tasks([data["tasks"][int(index)] for index in data["task_index"]], data["vocab"])
    values = (
        (data["state"] - stats["state_mean"]) / stats["state_std"],
        (data["velocity"] - stats["velocity_mean"]) / stats["velocity_std"],
        data["objects"], data["contact_sites"], data["tactile_contact"], data["color"],
        (data["action"] - stats["action_mean"]) / stats["action_std"],
        data["target"], data["arm"], ids, mask,
        causal_history(data, stats, history_frames),
    )
    return TensorDataset(*[value[data["valid"]] for value in values])


def select(
    values: torch.Tensor, target: torch.Tensor, arm: torch.Tensor
) -> torch.Tensor:
    return values[torch.arange(len(target), device=target.device), target, arm]


def metrics(output: dict, action: torch.Tensor, target: torch.Tensor, arm: torch.Tensor) -> dict:
    prediction = select(output["action_candidates"], target, arm)
    arm_logits = output["active_arm_logits_by_target"][
        torch.arange(len(target), device=target.device), target
    ]
    return {
        "action": F.mse_loss(prediction, action),
        "target": F.cross_entropy(output["target_logits"], target),
        "arm": F.cross_entropy(arm_logits, arm),
        "prediction": prediction,
        "arm_logits": arm_logits,
    }


@torch.no_grad()
def evaluate(model, loader, stats, device):
    model.eval()
    total = {key: 0.0 for key in ("action", "target", "arm")}
    correct_target = correct_arm = rows = 0
    raw_square = raw_count = 0
    for values in loader:
        state, velocity, objects, contact_sites, tactile_contact, color, action, target, arm, ids, mask, history_features = [value.to(device) for value in values]
        output = model(state, velocity, objects, contact_sites, tactile_contact, color, ids, mask, history_features)
        value = metrics(output, action, target, arm)
        count = len(target)
        for key in total:
            total[key] += float(value[key]) * count
        correct_target += int((output["target_logits"].argmax(-1) == target).sum())
        correct_arm += int((value["arm_logits"].argmax(-1) == arm).sum())
        predicted = value["prediction"] * stats["action_std"] + stats["action_mean"]
        expected = action * stats["action_std"] + stats["action_mean"]
        raw_square += float((predicted - expected).square().sum())
        raw_count += predicted.numel()
        rows += count
    return {
        **{key: value / max(rows, 1) for key, value in total.items()},
        "target_accuracy": correct_target / max(rows, 1),
        "active_arm_accuracy": correct_arm / max(rows, 1),
        "joint_action_rmse_rad": math.sqrt(raw_square / max(raw_count, 1)),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train = flatten(torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=False))
    val = flatten(torch.load(args.cache_dir / "val.pt", map_location="cpu", weights_only=False))
    if train["action_representation"] != val["action_representation"]:
        raise ValueError("Train and validation caches disagree on action representation")
    stats = statistics(train)
    train_loader = DataLoader(dataset(train, stats, args.history_frames), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset(val, stats, args.history_frames), batch_size=args.batch_size, shuffle=False)
    model = LocalJointVLA(
        len(train["vocab"]),
        args.hidden_dim,
        int(train["vision_dim"]),
        int(train["contact_site_dim"]),
    ).to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("action_representation", "absolute_joint_position") != train["action_representation"]:
            raise ValueError("Initialization checkpoint action representation differs from cache")
        if initial.get("config") != {
            "vocab_size": len(train["vocab"]),
            "hidden_dim": args.hidden_dim,
            "vision_dim": int(train["vision_dim"]),
            "contact_site_dim": int(train["contact_site_dim"]),
        }:
            raise ValueError("Initialization checkpoint architecture differs from cache/training arguments")
        if initial.get("vocab") != train["vocab"]:
            raise ValueError("Initialization checkpoint vocabulary differs from cache")
        model.load_state_dict(initial["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    calibration = torch.from_numpy(np.load(args.calibration)["coefficients"].astype(np.float32))
    best = math.inf; history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); total = rows = 0
        for values in train_loader:
            state, velocity, objects, contact_sites, tactile_contact, color, action, target, arm, ids, mask, history_features = [value.to(device) for value in values]
            value = metrics(
                model(state, velocity, objects, contact_sites, tactile_contact, color, ids, mask, history_features),
                action,
                target,
                arm,
            )
            loss = value["action"] + 0.20 * value["target"] + 0.08 * value["arm"]
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss) * len(target); rows += len(target)
        value = evaluate(model, val_loader, {key: item.to(device) for key, item in stats.items()}, device)
        value.update(epoch=epoch + 1, train_loss=total / max(rows, 1)); history.append(value)
        print("LOCAL_JOINT_VLA_EPOCH=" + json.dumps(value), flush=True)
        score = value["action"] + 0.20 * value["target"]
        if score < best:
            best = score
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                train["vocab"],
                stats,
                calibration,
                action_representation=train["action_representation"],
            )
    summary = {
        "experiment": "G1LANG-phase-free-local-direct-joint-vla",
        "cache_dir": str(args.cache_dir),
        "source_episodes": {"train": train["episodes"], "validation": val["episodes"]},
        "supervised_frames": {
            "train": int(train["valid"].sum()),
            "validation": int(val["valid"].sum()),
        },
        "candidate_offsets_frames": train["offsets"].tolist(),
        "vision_input_dim": int(train["vision_dim"]),
        "contact_site_input_dim": int(train["contact_site_dim"]),
        "camera_input": (
            "dynamic head/left-wrist/right-wrist RGB colour features"
            if int(train["vision_dim"]) == 27
            else "head RGB colour features"
        ),
        "phase_contract": "none; candidates are only local dynamically executed future joint targets",
        "state_contract": "eight-frame causal qpos31/qvel31/finger geometry/tactile contact + RGB/language",
        "action_representation": train["action_representation"],
        "action_contract": (
            "language target x learned active-arm branch x local joint deltas"
            if train["action_representation"] == "relative_joint_delta"
            else "language target x learned active-arm branch x local joint candidates"
        ),
        "history_frames": args.history_frames,
        "initialization_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "best_validation_score": best, "final": history[-1], "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
