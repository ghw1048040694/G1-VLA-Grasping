#!/usr/bin/env python3
"""Train a temporal VLA to propose direct dynamic 31-D joint candidates."""

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

from g1_direct_joint_candidate_vla import (
    ACTION_DIM,
    CANDIDATE_COUNT,
    PHASE_COUNT,
    TARGET_COUNT,
    DirectJointCandidateVLA,
    encode_tasks,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def statistics(cache: dict) -> dict[str, torch.Tensor]:
    action = cache["candidate_action"]
    lower = torch.empty((TARGET_COUNT, CANDIDATE_COUNT, ACTION_DIM), dtype=torch.float32)
    upper = torch.empty_like(lower)
    fallback_lower = torch.quantile(action.reshape(-1, CANDIDATE_COUNT, ACTION_DIM), 0.002, dim=0)
    fallback_upper = torch.quantile(action.reshape(-1, CANDIDATE_COUNT, ACTION_DIM), 0.998, dim=0)
    for target in range(TARGET_COUNT):
        values = action[cache["target"] == target]
        if len(values) < 2:
            lower[target], upper[target] = fallback_lower, fallback_upper
            continue
        values = values.reshape(-1, CANDIDATE_COUNT, ACTION_DIM)
        lower[target] = torch.quantile(values, 0.002, dim=0)
        upper[target] = torch.quantile(values, 0.998, dim=0)
    return {
        "state_mean": cache["state"].mean((0, 1)),
        "state_std": cache["state"].std((0, 1)).clamp_min(1e-3),
        "action_mean": action.mean((0, 1, 2)),
        "action_std": action.std((0, 1, 2)).clamp_min(1e-3),
        "candidate_lower": lower,
        "candidate_upper": upper,
        "candidate_offsets_frames": cache["candidate_offsets_frames"].clone(),
    }


def make_dataset(cache: dict, stats: dict[str, torch.Tensor]) -> TensorDataset:
    ids, mask = encode_tasks(
        [cache["tasks"][int(index)] for index in cache["task_index"]], cache["vocab"]
    )
    return TensorDataset(
        (cache["state"] - stats["state_mean"]) / stats["state_std"],
        cache["visual_object_position"],
        cache["palm_position"],
        cache["color_features"],
        (cache["candidate_action"] - stats["action_mean"]) / stats["action_std"],
        cache["target"],
        cache["phase"],
        cache["active_arm"],
        ids,
        mask,
    )


def selected_target(values: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return values[torch.arange(len(target), device=target.device), :, target]


def losses(output: dict, expected: torch.Tensor, target: torch.Tensor, phase: torch.Tensor, arm: torch.Tensor):
    predicted = selected_target(output["action_candidates"], target)
    phase_logits = selected_target(output["phase_logits_by_target"], target)
    arm_logits = selected_target(output["active_arm_logits_by_target"], target)
    action = F.mse_loss(predicted, expected)
    phase_loss = F.cross_entropy(phase_logits.reshape(-1, PHASE_COUNT), phase.reshape(-1))
    arm_loss = F.cross_entropy(arm_logits.reshape(-1, 2), arm.reshape(-1))
    target_loss = F.cross_entropy(output["target_logits"], target)
    return action, target_loss, phase_loss, arm_loss, predicted, phase_logits, arm_logits


@torch.no_grad()
def evaluate(model, loader, stats, device):
    model.eval()
    total = {"action": 0.0, "target": 0.0, "phase": 0.0, "active_arm": 0.0}
    correct = {"target": 0, "phase": 0, "active_arm": 0}
    raw_square = 0.0
    raw_count = 0
    episode_count = 0
    for values in loader:
        state, objects, palms, color, action, target, phase, arm, ids, mask = [value.to(device) for value in values]
        output = model(state, objects, palms, color, ids, mask)
        action_loss, target_loss, phase_loss, arm_loss, prediction, phase_logits, arm_logits = losses(
            output, action, target, phase, arm
        )
        batch = len(target)
        total["action"] += float(action_loss) * batch
        total["target"] += float(target_loss) * batch
        total["phase"] += float(phase_loss) * batch
        total["active_arm"] += float(arm_loss) * batch
        correct["target"] += int((output["target_logits"].argmax(-1) == target).sum())
        correct["phase"] += int((phase_logits.argmax(-1) == phase).sum())
        correct["active_arm"] += int((arm_logits.argmax(-1) == arm).sum())
        raw_prediction = prediction * stats["action_std"] + stats["action_mean"]
        raw_expected = action * stats["action_std"] + stats["action_mean"]
        raw_square += float((raw_prediction - raw_expected).square().sum())
        raw_count += raw_prediction.numel()
        episode_count += batch
    return {
        **{key: value / max(episode_count, 1) for key, value in total.items()},
        **{f"{key}_accuracy": value / max(episode_count * (1 if key == "target" else loader.dataset.tensors[0].shape[1]), 1) for key, value in correct.items()},
        "joint_action_rmse_rad": math.sqrt(raw_square / max(raw_count, 1)),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train = torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=False)
    val = torch.load(args.cache_dir / "val.pt", map_location="cpu", weights_only=False)
    required = {"state", "palm_position", "candidate_action", "phase", "active_arm", "target", "task_index", "tasks", "vocab", "color_features", "visual_object_position"}
    missing = sorted(required - set(train))
    if missing:
        raise ValueError(f"Cache missing required fields: {missing}")
    if train["candidate_action"].shape[-2:] != (CANDIDATE_COUNT, ACTION_DIM):
        raise ValueError("Unexpected direct candidate action shape")
    stats = statistics(train)
    train_set = make_dataset(train, stats)
    val_set = make_dataset(val, stats)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    model = DirectJointCandidateVLA(len(train["vocab"]), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    calibration = torch.from_numpy(np.load(args.calibration)["coefficients"].astype(np.float32))
    best = math.inf
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        episodes = 0
        for values in train_loader:
            state, objects, palms, color, action, target, phase, arm, ids, mask = [value.to(device) for value in values]
            output = model(state, objects, palms, color, ids, mask)
            action_loss, target_loss, phase_loss, arm_loss, *_ = losses(output, action, target, phase, arm)
            loss = action_loss + 0.20 * target_loss + 0.12 * phase_loss + 0.08 * arm_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(target)
            episodes += len(target)
        metrics = evaluate(model, val_loader, {key: value.to(device) for key, value in stats.items()}, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = total / max(episodes, 1)
        history.append(metrics)
        print("DIRECT_JOINT_VLA_EPOCH=" + json.dumps(metrics), flush=True)
        score = metrics["action"] + 0.20 * metrics["target"] + 0.12 * metrics["phase"]
        if score < best:
            best = score
            save_checkpoint(args.output_dir / "best.pt", model, train["vocab"], stats, calibration)
    summary = {
        "experiment": "G1LANG-direct-joint-temporal-vla",
        "cache_dir": str(args.cache_dir),
        "state_input_dim": 31,
        "camera_input": "latched head RGB colour geometry",
        "language_input": "tokenized task",
        "model_action_output": "target(3) x learned future joint candidates(4) x 31D",
        "candidate_offsets_frames": train["candidate_offsets_frames"].tolist(),
        "temporal_memory": "target-specific GRU updated from RGB/language/proprioception history",
        "labels": "strict MuJoCo dynamically executed joint-position commands",
        "best_validation_score": best,
        "final": history[-1],
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
