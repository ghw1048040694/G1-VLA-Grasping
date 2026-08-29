#!/usr/bin/env python3
"""Train the structured local contact VLA on camera-derived geometry."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from g1_geometric_contact_vla import GeometricContactVLA, encode_tasks, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--geometry-noise-m", type=float, default=0.012)
    parser.add_argument(
        "--action-contract",
        choices=("palm_residual", "target_relative"),
        default="palm_residual",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def statistics(cache: dict) -> dict[str, torch.Tensor]:
    return {
        "state_mean": cache["state"].mean(0),
        "state_std": cache["state"].std(0).clamp_min(1e-3),
        "action_mean": cache["action"].mean(0),
        "action_std": cache["action"].std(0).clamp_min(1e-3),
        "action_lower": torch.quantile(cache["action"], 0.001, dim=0),
        "action_upper": torch.quantile(cache["action"], 0.999, dim=0),
    }


def make_dataset(cache: dict, stats: dict[str, torch.Tensor]) -> TensorDataset:
    token_ids, token_mask = encode_tasks(
        [cache["tasks"][int(index)] for index in cache["task_index"]], cache["vocab"]
    )
    return TensorDataset(
        (cache["state"] - stats["state_mean"]) / stats["state_std"],
        cache["visual_object_position"],
        cache["palm_position"],
        cache["color_features"],
        (cache["action"] - stats["action_mean"]) / stats["action_std"],
        cache["target"],
        cache["phase"],
        cache["active_arm"],
        token_ids,
        token_mask,
    )


def action_loss(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    phase: torch.Tensor,
    action_weights: torch.Tensor,
) -> torch.Tensor:
    phase_weight = torch.ones_like(phase, dtype=predicted.dtype)
    phase_weight = torch.where(phase == 0, torch.tensor(1.5, device=phase.device), phase_weight)
    phase_weight = torch.where(phase == 1, torch.tensor(2.0, device=phase.device), phase_weight)
    phase_weight = torch.where(
        (phase == 2) | (phase == 3), torch.tensor(5.0, device=phase.device), phase_weight
    )
    return (((predicted - expected).square() * action_weights).mean(-1) * phase_weight).mean()


@torch.no_grad()
def evaluate(
    model: GeometricContactVLA,
    loader: DataLoader,
    device: str,
    action_weights: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    totals = {"action": 0.0, "target": 0.0, "phase": 0.0, "active_arm": 0.0}
    target_correct = phase_correct = arm_correct = count = 0
    for values in loader:
        state, objects, palms, color, action, target, phase, arm, ids, mask = [
            value.to(device) for value in values
        ]
        output = model(state, objects, palms, color, ids, mask)
        batch_count = len(target)
        totals["action"] += float(action_loss(output["action"], action, phase, action_weights)) * batch_count
        totals["target"] += float(F.cross_entropy(output["target_logits"], target)) * batch_count
        totals["phase"] += float(F.cross_entropy(output["phase_logits"], phase)) * batch_count
        totals["active_arm"] += float(F.cross_entropy(output["active_arm_logits"], arm)) * batch_count
        target_correct += int((output["target_logits"].argmax(-1) == target).sum())
        phase_correct += int((output["phase_logits"].argmax(-1) == phase).sum())
        arm_correct += int((output["active_arm_logits"].argmax(-1) == arm).sum())
        count += batch_count
    return {
        key: value / max(count, 1) for key, value in totals.items()
    } | {
        "target_accuracy": target_correct / max(count, 1),
        "phase_accuracy": phase_correct / max(count, 1),
        "active_arm_accuracy": arm_correct / max(count, 1),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    train_cache = torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=False)
    val_cache = torch.load(args.cache_dir / "val.pt", map_location="cpu", weights_only=False)
    stats = statistics(train_cache)
    stats["target_relative_action"] = torch.tensor(
        int(args.action_contract == "target_relative"), dtype=torch.int64
    )
    train_set = make_dataset(train_cache, stats)
    val_set = make_dataset(val_cache, stats)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    model = GeometricContactVLA(len(train_cache["vocab"]), args.hidden_dim).to(device)
    action_weights = torch.ones(35, device=device)
    action_weights[:3] = 1.0
    action_weights[3:9] = 10.0
    action_weights[9:21] = 2.0
    action_weights[21:] = 3.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    calibration = torch.from_numpy(
        __import__("numpy").load(args.calibration)["coefficients"].astype("float32")
    )
    best = math.inf
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running = samples = 0
        for values in train_loader:
            state, objects, palms, color, action, target, phase, arm, ids, mask = [
                value.to(device) for value in values
            ]
            if args.geometry_noise_m > 0:
                # Calibration error is an RGB deployment uncertainty, not a label.
                objects = objects + torch.randn_like(objects) * args.geometry_noise_m
            override = F.one_hot(target, num_classes=3).to(dtype=state.dtype)
            output = model(state, objects, palms, color, ids, mask, target_override=override)
            loss = (
                action_loss(output["action"], action, phase, action_weights)
                + 0.30 * F.cross_entropy(output["target_logits"], target)
                + 0.10 * F.cross_entropy(output["phase_logits"], phase)
                + 0.10 * F.cross_entropy(output["active_arm_logits"], arm)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss) * len(target)
            samples += len(target)
        metrics = evaluate(model, val_loader, device, action_weights)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running / max(samples, 1)
        history.append(metrics)
        print("GEOMETRIC_CONTACT_VLA_EPOCH=" + json.dumps(metrics), flush=True)
        score = metrics["action"] + 0.30 * metrics["target"]
        if score < best:
            best = score
            save_checkpoint(args.output_dir / "best.pt", model, train_cache["vocab"], stats, calibration)
    summary = {
        "experiment": "G1LANG_geometric_local_contact_vla",
        "cache_dir": str(args.cache_dir),
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "epochs": args.epochs,
        "geometry_noise_m": args.geometry_noise_m,
        "best_validation_score": best,
        "final_validation": history[-1],
        "input_contract": "head-camera RGB geometry + tokenized language + 31D state + measured palm pose",
        "action_contract": (
            "learned 35D one-step Cartesian palm residual + wrist orientation + object-specific hand joints"
            if args.action_contract == "palm_residual"
            else "learned 35D Cartesian palm pose relative to the camera-selected object + wrist orientation + object-specific hand joints"
        ),
        "privileged_positions_at_inference": False,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
