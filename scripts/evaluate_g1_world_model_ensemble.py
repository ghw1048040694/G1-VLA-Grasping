#!/usr/bin/env python3
"""Measure whether G1 world-model ensemble disagreement predicts rollout error."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_g1_world_model as wm
import train_g1_world_model_interventions as wi
from train_g1_hybrid_world_model import build_hybrid_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--train-sources", type=int, default=100)
    parser.add_argument("--horizons", type=int, nargs="+", default=(5, 20))
    parser.add_argument("--max-starts", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


@torch.inference_mode()
def evaluate_group(
    episodes: list[wm.Episode],
    models: list[torch.nn.Module],
    stats: list[dict[str, np.ndarray]],
    horizon: int,
    max_starts: int,
    device: torch.device,
) -> dict:
    starts = wm.sampled_starts(episodes, horizon, max_starts)
    member_endpoints = [[] for _ in models]
    targets = []
    for offset in range(0, len(starts), 128):
        batch = starts[offset : offset + 128]
        initial = np.stack([episodes[e].state[s] for e, s in batch])
        actions = np.stack([episodes[e].action[s : s + horizon] for e, s in batch])
        target = np.stack([episodes[e].state[s + horizon] for e, s in batch])
        targets.append(target)
        for index, (model, member_stats) in enumerate(zip(models, stats, strict=True)):
            normalized_initial = (
                initial - member_stats["state_mean"]
            ) / member_stats["state_std"]
            normalized_actions = (
                actions - member_stats["action_mean"]
            ) / member_stats["action_std"]
            prediction = wm.rollout(
                model,
                torch.from_numpy(normalized_initial.astype(np.float32)).to(device),
                torch.from_numpy(normalized_actions.astype(np.float32)).to(device),
            )[:, -1].cpu().numpy()
            member_endpoints[index].append(
                prediction * member_stats["state_std"]
                + member_stats["state_mean"]
            )
    members = np.stack(
        [np.concatenate(endpoint) for endpoint in member_endpoints], axis=0
    )
    target = np.concatenate(targets)
    ensemble = np.mean(members, axis=0)
    reference_mean = stats[0]["state_mean"]
    reference_std = stats[0]["state_std"]
    object_low = wm.LAYOUT.tote_position[0]
    object_high = wm.LAYOUT.task_progress[1]
    normalized_members = (
        members[:, :, object_low:object_high]
        - reference_mean[None, None, object_low:object_high]
    ) / reference_std[None, None, object_low:object_high]
    normalized_target = (
        target[:, object_low:object_high]
        - reference_mean[None, object_low:object_high]
    ) / reference_std[None, object_low:object_high]
    normalized_ensemble = np.mean(normalized_members, axis=0)
    disagreement = np.sqrt(np.mean(np.var(normalized_members, axis=0), axis=1))
    actual_error = np.sqrt(
        np.mean((normalized_ensemble - normalized_target) ** 2, axis=1)
    )
    high_uncertainty = disagreement >= np.quantile(disagreement, 0.75)
    high_error = actual_error >= np.quantile(actual_error, 0.75)
    low_error_mean = float(np.mean(actual_error[~high_uncertainty]))
    return {
        "samples": len(starts),
        "ensemble_metrics": wm.group_metrics(ensemble.copy(), target.copy()),
        "member_metrics": [
            wm.group_metrics(member.copy(), target.copy()) for member in members
        ],
        "disagreement_error_pearson_r": correlation(disagreement, actual_error),
        "top_quartile_uncertainty_error_ratio": float(
            np.mean(actual_error[high_uncertainty]) / max(low_error_mean, 1e-12)
        ),
        "top_quartile_error_recall": float(
            np.sum(high_uncertainty & high_error) / max(np.sum(high_error), 1)
        ),
        "normalized_object_disagreement_mean": float(np.mean(disagreement)),
        "normalized_object_disagreement_p90": float(
            np.quantile(disagreement, 0.90)
        ),
        "normalized_object_disagreement_p95": float(
            np.quantile(disagreement, 0.95)
        ),
        "normalized_object_error_mean": float(np.mean(actual_error)),
    }


def main() -> None:
    args = parse_args()
    if len(args.checkpoint) < 3:
        raise ValueError("At least three checkpoints are required for an ensemble")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    _, validation, conditions = wi.load_episodes(
        args.transition_summary, args.train_sources
    )
    groups = wi.condition_groups(validation, conditions)
    checkpoints = [
        torch.load(path, map_location=device, weights_only=False)
        for path in args.checkpoint
    ]
    models = [
        build_hybrid_from_checkpoint(checkpoint, device).eval()
        for checkpoint in checkpoints
    ]
    stats = [checkpoint["normalizers"] for checkpoint in checkpoints]
    results = {
        condition: {
            f"h{horizon}": evaluate_group(
                episodes,
                models,
                stats,
                horizon,
                args.max_starts,
                device,
            )
            for horizon in args.horizons
        }
        for condition, episodes in groups.items()
        if condition in ("nominal", "perturbed")
    }
    calibration = results["perturbed"][f"h{min(args.horizons)}"]
    summary = {
        "experiment": "G1WM-04-deep-ensemble-uncertainty",
        "checkpoints": [str(path) for path in args.checkpoint],
        "validation_source_leakage": False,
        "results": results,
        "recommended_mpc_normalized_disagreement_threshold": calibration[
            "normalized_object_disagreement_p90"
        ],
        "screening_checks": {
            "h5_disagreement_error_correlation_at_least_0.3": calibration[
                "disagreement_error_pearson_r"
            ]
            >= 0.3,
            "h5_high_uncertainty_error_ratio_at_least_1.5": calibration[
                "top_quartile_uncertainty_error_ratio"
            ]
            >= 1.5,
            "h5_top_error_quartile_recall_at_least_0.5": calibration[
                "top_quartile_error_recall"
            ]
            >= 0.5,
        },
    }
    summary["experiment_passed"] = all(summary["screening_checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
