#!/usr/bin/env python3
"""Train and compare a G1 world model on control-rate action interventions."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import train_g1_world_model as wm


def load_episodes(summary_path: Path, train_sources: int) -> tuple[list[wm.Episode], list[wm.Episode], dict[int, str]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    train = []
    val = []
    conditions = {}
    for item in summary["episodes"]:
        with np.load(item["dataset"]) as arrays:
            state = arrays["state"].astype(np.float32)
            action = arrays["action"].astype(np.float32)
        if len(state) != len(action) + 1:
            raise ValueError(f"Invalid transition contract in {item['dataset']}")
        source_index = int(item["source_episode_index"])
        unique_id = source_index * 2 + int(item["condition"] != "nominal")
        episode = wm.Episode(unique_id, state, action)
        conditions[unique_id] = item["condition"]
        (train if source_index < train_sources else val).append(episode)
    train_sources_set = {item.episode_index // 2 for item in train}
    val_sources_set = {item.episode_index // 2 for item in val}
    if train_sources_set & val_sources_set:
        raise RuntimeError("Source-episode leakage detected")
    return train, val, conditions


def condition_groups(episodes: list[wm.Episode], conditions: dict[int, str]) -> dict[str, list[wm.Episode]]:
    groups = {
        "all": episodes,
        "nominal": [item for item in episodes if conditions[item.episode_index] == "nominal"],
        "perturbed": [item for item in episodes if conditions[item.episode_index] != "nominal"],
    }
    for condition in sorted(set(conditions[item.episode_index] for item in episodes)):
        if condition != "nominal":
            groups[condition] = [item for item in episodes if conditions[item.episode_index] == condition]
    return groups


def evaluate_groups(
    model: nn.Module,
    raw_groups: dict[str, list[wm.Episode]],
    stats: dict[str, np.ndarray],
    horizons: list[int],
    device: torch.device,
    max_starts: int,
) -> dict:
    return {
        name: wm.evaluate(
            model,
            wm.normalized_copy(episodes, stats),
            stats,
            horizons,
            device,
            max_starts,
        )
        for name, episodes in raw_groups.items()
        if episodes
    }


def build_model_from_checkpoint(checkpoint: dict, device: torch.device) -> nn.Module:
    args = checkpoint["args"]
    model = wm.ResidualWorldModel(
        wm.LAYOUT.state_dim,
        wm.LAYOUT.action_dim,
        int(args["hidden_dim"]),
        int(args["depth"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-summary", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="G1WM-02-intervention-data-aggregation")
    parser.add_argument("--train-sources", type=int, default=100)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-horizon", type=int, default=10)
    parser.add_argument("--rollout-horizons", type=int, nargs="+", default=(1, 5, 10, 20))
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--max-eval-starts", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=4202)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_raw, val_raw, conditions = load_episodes(args.transition_summary, args.train_sources)
    stats = wm.normalizers(train_raw)
    train = wm.normalized_copy(train_raw, stats)
    dataset = wm.SequenceDataset(train, args.train_horizon)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    iterator = iter(loader)
    model = wm.ResidualWorldModel(
        wm.LAYOUT.state_dim,
        wm.LAYOUT.action_dim,
        args.hidden_dim,
        args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.1
    )
    val_groups = condition_groups(val_raw, conditions)
    log_path = args.output_dir / "training_log.jsonl"
    best_score = math.inf
    best_step = 0

    for step in range(1, args.steps + 1):
        try:
            initial, actions, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            initial, actions, targets = next(iterator)
        initial, actions, targets = initial.to(device), actions.to(device), targets.to(device)
        model.train()
        prediction = wm.rollout(model, initial, actions)
        loss = torch.mean((prediction - targets) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        optimizer.step()
        scheduler.step()
        record = {
            "step": step,
            "train_normalized_mse": float(loss),
            "gradient_norm": gradient_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        evaluate_now = step == 1 or step % args.eval_freq == 0 or step == args.steps
        if evaluate_now:
            perturbed = evaluate_groups(
                model,
                {"perturbed": val_groups["perturbed"]},
                stats,
                list(args.rollout_horizons),
                device,
                args.max_eval_starts,
            )["perturbed"]
            score = perturbed[f"h{max(args.rollout_horizons)}"]["normalized_rollout_mse"]
            record["validation_perturbed"] = perturbed
            print(json.dumps(record), flush=True)
            if score < best_score:
                best_score = score
                best_step = step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "normalizers": stats,
                        "layout": wm.asdict(wm.LAYOUT),
                        "args": vars(args),
                    },
                    args.output_dir / "best_model.pt",
                )
        elif step % 100 == 0:
            print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    best = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    new_metrics = evaluate_groups(
        model,
        val_groups,
        stats,
        list(args.rollout_horizons),
        device,
        args.max_eval_starts,
    )
    baseline_checkpoint = torch.load(
        args.baseline_checkpoint, map_location=device, weights_only=False
    )
    baseline_model = build_model_from_checkpoint(baseline_checkpoint, device)
    baseline_metrics = evaluate_groups(
        baseline_model,
        val_groups,
        baseline_checkpoint["normalizers"],
        list(args.rollout_horizons),
        device,
        args.max_eval_starts,
    )
    comparison_horizon = f"h{max(args.rollout_horizons)}"
    old_perturbed = baseline_metrics["perturbed"][comparison_horizon]
    new_perturbed = new_metrics["perturbed"][comparison_horizon]
    old_nominal = baseline_metrics["nominal"][comparison_horizon]
    new_nominal = new_metrics["nominal"][comparison_horizon]
    comparisons = {
        "perturbed_joint_position_error_reduction": 1.0 - new_perturbed["joint_position_rmse_rad"] / old_perturbed["joint_position_rmse_rad"],
        "perturbed_tote_position_error_reduction": 1.0 - new_perturbed["tote_position_rmse_m"] / old_perturbed["tote_position_rmse_m"],
        "nominal_joint_position_error_ratio": new_nominal["joint_position_rmse_rad"] / old_nominal["joint_position_rmse_rad"],
        "nominal_tote_position_error_ratio": new_nominal["tote_position_rmse_m"] / old_nominal["tote_position_rmse_m"],
    }
    checks = {
        "perturbed_h20_joint_position_rmse_below_0.03_rad": new_perturbed["joint_position_rmse_rad"] < 0.03,
        "perturbed_h20_tote_position_rmse_below_2cm": new_perturbed["tote_position_rmse_m"] < 0.02,
        "perturbed_h20_lift_height_rmse_below_2cm": new_perturbed["lift_height_rmse_m"] < 0.02,
        "perturbed_joint_error_improves_baseline_at_least_20pct": comparisons["perturbed_joint_position_error_reduction"] >= 0.20,
        "perturbed_tote_error_improves_baseline_at_least_20pct": comparisons["perturbed_tote_position_error_reduction"] >= 0.20,
        "nominal_joint_error_no_more_than_50pct_worse": comparisons["nominal_joint_position_error_ratio"] <= 1.50,
        "nominal_tote_error_no_more_than_50pct_worse": comparisons["nominal_tote_position_error_ratio"] <= 1.50,
    }
    summary = {
        "experiment": args.experiment_id,
        "device": str(device),
        "train_rollouts": len(train_raw),
        "validation_rollouts": len(val_raw),
        "train_source_indices": sorted({item.episode_index // 2 for item in train_raw}),
        "validation_source_indices": sorted({item.episode_index // 2 for item in val_raw}),
        "source_episode_split_leakage": False,
        "best_step": best_step,
        "best_perturbed_h20_normalized_mse": best_score,
        "comparison_horizon": comparison_horizon,
        "baseline_g1wm01_metrics": baseline_metrics,
        "g1wm02_metrics": new_metrics,
        "comparisons": comparisons,
        "screening_checks": checks,
        "experiment_passed": all(checks.values()),
        "reference_note": "Thresholds are project screening targets, not universal robotics standards.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    preview_episode = val_groups["perturbed"][0]
    wm.save_preview(model, preview_episode, stats, args.output_dir, device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
