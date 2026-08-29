#!/usr/bin/env python3
"""Train contact-gated direct-joint VLA candidates on strict dynamic labels."""

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

from g1_contact_gated_joint_vla import (
    ACTION_DIM,
    CANDIDATE_COUNT,
    TARGET_COUNT,
    ContactGatedJointVLA,
    encode_tasks,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def statistics(cache: dict) -> dict[str, torch.Tensor]:
    action = cache["candidate_action"]
    lower = torch.empty((TARGET_COUNT, CANDIDATE_COUNT, ACTION_DIM), dtype=torch.float32)
    upper = torch.empty_like(lower)
    fallback_lower = torch.quantile(action, 0.002, dim=0)
    fallback_upper = torch.quantile(action, 0.998, dim=0)
    for target in range(TARGET_COUNT):
        values = action[cache["target"] == target]
        if len(values) < 8:
            lower[target], upper[target] = fallback_lower, fallback_upper
        else:
            lower[target] = torch.quantile(values, 0.002, dim=0)
            upper[target] = torch.quantile(values, 0.998, dim=0)
    return {
        "state_mean": cache["state"].mean(0),
        "state_std": cache["state"].std(0).clamp_min(1e-3),
        "action_mean": action.mean((0, 1)),
        "action_std": action.std((0, 1)).clamp_min(1e-3),
        "candidate_lower": lower,
        "candidate_upper": upper,
    }


def make_dataset(cache, stats):
    ids, mask = encode_tasks([cache["tasks"][int(index)] for index in cache["task_index"]], cache["vocab"])
    return TensorDataset(
        (cache["state"] - stats["state_mean"]) / stats["state_std"],
        cache["visual_object_position"], cache["palm_position"], cache["color_features"],
        (cache["candidate_action"] - stats["action_mean"]) / stats["action_std"],
        cache["target"], cache["desired_stage"], cache["active_arm"], ids, mask,
    )


def select(values, target):
    return values[torch.arange(len(target), device=target.device), target]


def batch_metrics(output, action, target, stage, arm):
    prediction = select(output["action_candidates"], target)
    stage_logits = select(output["next_stage_logits_by_target"], target)
    arm_logits = select(output["active_arm_logits_by_target"], target)
    return {
        "action": F.mse_loss(prediction, action),
        "target": F.cross_entropy(output["target_logits"], target),
        "stage": F.cross_entropy(stage_logits, stage),
        "active_arm": F.cross_entropy(arm_logits, arm),
        "prediction": prediction,
        "stage_logits": stage_logits,
        "arm_logits": arm_logits,
    }


@torch.no_grad()
def evaluate(model, loader, stats, device):
    model.eval()
    total = {key: 0.0 for key in ("action", "target", "stage", "active_arm")}
    correct = {key: 0 for key in ("target", "stage", "active_arm")}
    raw_square = 0.0
    raw_count = 0
    rows = 0
    for values in loader:
        state, objects, palms, color, action, target, stage, arm, ids, mask = [value.to(device) for value in values]
        output = model(state, objects, palms, color, ids, mask)
        metrics = batch_metrics(output, action, target, stage, arm)
        count = len(target)
        for key in total:
            total[key] += float(metrics[key]) * count
        correct["target"] += int((output["target_logits"].argmax(-1) == target).sum())
        correct["stage"] += int((metrics["stage_logits"].argmax(-1) == stage).sum())
        correct["active_arm"] += int((metrics["arm_logits"].argmax(-1) == arm).sum())
        raw_prediction = metrics["prediction"] * stats["action_std"] + stats["action_mean"]
        raw_expected = action * stats["action_std"] + stats["action_mean"]
        raw_square += float((raw_prediction - raw_expected).square().sum())
        raw_count += raw_prediction.numel()
        rows += count
    return {
        **{key: value / max(rows, 1) for key, value in total.items()},
        **{f"{key}_accuracy": value / max(rows, 1) for key, value in correct.items()},
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
    required = {"state", "palm_position", "candidate_action", "desired_stage", "active_arm", "target", "task_index", "tasks", "vocab", "color_features", "visual_object_position"}
    missing = sorted(required - set(train))
    if missing:
        raise ValueError(f"Cache missing required fields: {missing}")
    stats = statistics(train)
    train_loader = DataLoader(make_dataset(train, stats), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(make_dataset(val, stats), batch_size=args.batch_size, shuffle=False)
    model = ContactGatedJointVLA(len(train["vocab"]), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    calibration = torch.from_numpy(np.load(args.calibration)["coefficients"].astype(np.float32))
    best = math.inf
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total = rows = 0
        for values in train_loader:
            state, objects, palms, color, action, target, stage, arm, ids, mask = [value.to(device) for value in values]
            metrics = batch_metrics(model(state, objects, palms, color, ids, mask), action, target, stage, arm)
            loss = metrics["action"] + 0.20 * metrics["target"] + 0.15 * metrics["stage"] + 0.08 * metrics["active_arm"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(target)
            rows += len(target)
        metrics = evaluate(model, val_loader, {key: value.to(device) for key, value in stats.items()}, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = total / max(rows, 1)
        history.append(metrics)
        print("CONTACT_GATED_JOINT_VLA_EPOCH=" + json.dumps(metrics), flush=True)
        score = metrics["action"] + 0.20 * metrics["target"] + 0.15 * metrics["stage"]
        if score < best:
            best = score
            save_checkpoint(args.output_dir / "best.pt", model, train["vocab"], stats, calibration)
    summary = {
        "experiment": "G1LANG-contact-gated-direct-joint-vla",
        "cache_dir": str(args.cache_dir),
        "model_action_output": "target(3) x learned contact-gated phase candidates(4) x joint31",
        "phase_contract": "learned next-stage logits from current RGB/proprioceptive geometry; no clock or scripted stage scheduler",
        "labels": "strict MuJoCo dynamically executed joint commands",
        "best_validation_score": best,
        "final": history[-1],
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
