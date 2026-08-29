#!/usr/bin/env python3
"""Train a target x phase candidate VLA on the compact geometry cache."""

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

from g1_phase_candidate_contact_vla import (
    ACTION_DIM,
    PHASE_COUNT,
    TARGET_COUNT,
    PhaseCandidateContactVLA,
    encode_tasks,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--geometry-noise-m", type=float, default=0.010)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def candidate_bounds(cache: dict, fallback_lower: torch.Tensor, fallback_upper: torch.Tensor):
    lower = torch.empty((TARGET_COUNT, PHASE_COUNT, ACTION_DIM), dtype=torch.float32)
    upper = torch.empty_like(lower)
    for target in range(TARGET_COUNT):
        for phase in range(PHASE_COUNT):
            keep = (cache["target"] == target) & (cache["phase"] == phase)
            if int(keep.sum()) < 8:
                lower[target, phase] = fallback_lower
                upper[target, phase] = fallback_upper
                continue
            values = cache["action"][keep]
            lower[target, phase] = torch.quantile(values, 0.002, dim=0)
            upper[target, phase] = torch.quantile(values, 0.998, dim=0)
    return lower, upper


def statistics(cache: dict) -> dict[str, torch.Tensor]:
    action_lower = torch.quantile(cache["action"], 0.001, dim=0)
    action_upper = torch.quantile(cache["action"], 0.999, dim=0)
    per_target_phase_lower, per_target_phase_upper = candidate_bounds(
        cache, action_lower, action_upper
    )
    return {
        "state_mean": cache["state"].mean(0),
        "state_std": cache["state"].std(0).clamp_min(1e-3),
        "action_mean": cache["action"].mean(0),
        "action_std": cache["action"].std(0).clamp_min(1e-3),
        "action_lower": action_lower,
        "action_upper": action_upper,
        "candidate_lower": per_target_phase_lower,
        "candidate_upper": per_target_phase_upper,
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
        (cache["action"] - stats["action_mean"]) / stats["action_std"],
        cache["target"],
        cache["phase"],
        cache["active_arm"],
        ids,
        mask,
    )


def gathered_actions(candidates: torch.Tensor, target: torch.Tensor, phase: torch.Tensor):
    rows = torch.arange(len(target), device=target.device)
    return candidates[rows, target, phase]


def action_loss(
    prediction: torch.Tensor,
    expected: torch.Tensor,
    phase: torch.Tensor,
    active_arm: torch.Tensor,
):
    weights = torch.ones_like(prediction)
    weights[:, 3:9] = 10.0
    weights[:, 9:21] = 2.0
    weights[:, 21:] = 3.0
    # A phase-3 near miss is a failure, even when the other 32 Cartesian and
    # hand dimensions are well predicted.  Weight the selected palm's contact
    # pose directly instead of allowing it to be averaged away by passive-arm
    # values.  The arm identity remains a learned classifier/input-dependent
    # choice; this only changes offline supervision emphasis.
    contact_rows = phase == 3
    for arm in range(2):
        rows = contact_rows & (active_arm == arm)
        if bool(rows.any()):
            weights[rows, 3 + arm * 3 : 6 + arm * 3] = 80.0
            weights[rows, 9 + arm * 6 : 15 + arm * 6] = 12.0
            weights[rows, 21 + arm * 7 : 28 + arm * 7] = 6.0
    phase_weight = torch.where(
        phase >= 2,
        torch.tensor(2.0, device=prediction.device),
        torch.tensor(1.0, device=prediction.device),
    )
    return (((prediction - expected).square() * weights).mean(-1) * phase_weight).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    totals = {"action": 0.0, "target": 0.0, "phase": 0.0, "active_arm": 0.0}
    correct = {"target": 0, "phase": 0, "active_arm": 0}
    count = 0
    for values in loader:
        state, objects, palms, color, action, target, phase, arm, ids, mask = [
            value.to(device) for value in values
        ]
        output = model(state, objects, palms, color, ids, mask)
        selected = gathered_actions(output["action_candidates"], target, phase)
        batch = len(target)
        totals["action"] += float(action_loss(selected, action, phase, arm)) * batch
        totals["target"] += float(F.cross_entropy(output["target_logits"], target)) * batch
        totals["phase"] += float(F.cross_entropy(output["phase_logits"], phase)) * batch
        totals["active_arm"] += float(F.cross_entropy(output["active_arm_logits"], arm)) * batch
        correct["target"] += int((output["target_logits"].argmax(-1) == target).sum())
        correct["phase"] += int((output["phase_logits"].argmax(-1) == phase).sum())
        correct["active_arm"] += int((output["active_arm_logits"].argmax(-1) == arm).sum())
        count += batch
    return {
        **{key: value / max(count, 1) for key, value in totals.items()},
        **{f"{key}_accuracy": value / max(count, 1) for key, value in correct.items()},
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    train = torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=False)
    val = torch.load(args.cache_dir / "val.pt", map_location="cpu", weights_only=False)
    required = {"state", "visual_object_position", "palm_position", "color_features", "action", "target", "phase", "active_arm", "task_index", "tasks", "vocab"}
    missing = sorted(required - set(train))
    if missing:
        raise ValueError(f"Cache is missing required fields: {missing}")
    if int(train["phase"].min()) < 0 or int(train["phase"].max()) >= PHASE_COUNT:
        raise ValueError("This candidate policy only accepts phase 0 through 3 labels")
    stats = statistics(train)
    mixed_contract = train.get("action_contract") == "active_target_passive_palm_residual"
    stats["active_target_passive_palm_residual"] = torch.tensor(
        int(mixed_contract), dtype=torch.int64
    )
    train_set = make_dataset(train, stats)
    val_set = make_dataset(val, stats)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    model = PhaseCandidateContactVLA(len(train["vocab"]), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    calibration = torch.from_numpy(np.load(args.calibration)["coefficients"].astype(np.float32))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = math.inf
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = samples = 0
        for values in train_loader:
            state, objects, palms, color, action, target, phase, arm, ids, mask = [
                value.to(device) for value in values
            ]
            if args.geometry_noise_m > 0:
                objects = objects + torch.randn_like(objects) * args.geometry_noise_m
            override = F.one_hot(target, num_classes=TARGET_COUNT).to(dtype=state.dtype)
            output = model(state, objects, palms, color, ids, mask, target_override=override)
            selected = gathered_actions(output["action_candidates"], target, phase)
            loss = (
                action_loss(selected, action, phase, arm)
                + 0.30 * F.cross_entropy(output["target_logits"], target)
                + 0.18 * F.cross_entropy(output["phase_logits"], phase)
                + 0.10 * F.cross_entropy(output["active_arm_logits"], arm)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(target)
            samples += len(target)
        metrics = evaluate(model, val_loader, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = total / max(samples, 1)
        history.append(metrics)
        print("PHASE_CANDIDATE_VLA_EPOCH=" + json.dumps(metrics), flush=True)
        score = metrics["action"] + 0.30 * metrics["target"] + 0.12 * metrics["phase"]
        if score < best:
            best = score
            save_checkpoint(args.output_dir / "best.pt", model, train["vocab"], stats, calibration)
    phase_counts = torch.bincount(train["phase"], minlength=PHASE_COUNT).tolist()
    summary = {
        "experiment": "G1LANG-phase-candidate-contact-vla",
        "cache_dir": str(args.cache_dir),
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "epochs": args.epochs,
        "geometry_noise_m": args.geometry_noise_m,
        "phase_counts": phase_counts,
        "candidate_layout": "target(3) x learned_approach_phase(4) x contact35",
        "input_contract": "head-camera RGB geometry + tokenized language + 31D state + measured palm pose",
        "action_contract": (
            "learned active-target and passive-palm-residual 35D Cartesian contact pose with per-target-phase reachability bounds"
            if mixed_contract
            else "learned target-relative 35D Cartesian contact pose with per-target-phase reachability bounds"
        ),
        "privileged_positions_at_inference": False,
        "best_validation_score": best,
        "final_validation": history[-1],
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
