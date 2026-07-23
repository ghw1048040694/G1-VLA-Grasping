#!/usr/bin/env python3
"""Train a constraint-aware hybrid dynamics model on G1 intervention data."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import train_g1_world_model as wm
import train_g1_world_model_interventions as wi


class HybridWorldModel(nn.Module):
    def __init__(
        self,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        joint_coefficients: np.ndarray,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        hidden_dim: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.register_buffer("state_mean", torch.from_numpy(state_mean))
        self.register_buffer("state_std", torch.from_numpy(state_std))
        self.register_buffer("action_mean", torch.from_numpy(action_mean))
        self.register_buffer("action_std", torch.from_numpy(action_std))
        self.register_buffer("joint_coefficients", torch.from_numpy(joint_coefficients))
        self.register_buffer("joint_lower", torch.from_numpy(joint_lower))
        self.register_buffer("joint_upper", torch.from_numpy(joint_upper))
        layers: list[nn.Module] = [
            nn.Linear(wm.LAYOUT.state_dim + wm.LAYOUT.action_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        self.backbone = nn.Sequential(*layers)
        self.correction_head = nn.Linear(hidden_dim, wm.LAYOUT.state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raw_state = state * self.state_std + self.state_mean
        raw_action = action * self.action_std + self.action_mean
        q_low, q_high = wm.LAYOUT.joint_position
        v_low, v_high = wm.LAYOUT.joint_velocity
        joint_input = torch.stack(
            (
                raw_state[..., q_low:q_high],
                raw_state[..., v_low:v_high],
                raw_action,
                torch.ones_like(raw_action),
            ),
            dim=-1,
        )
        joint_next = torch.einsum(
            "...ji,jki->...jk", joint_input, self.joint_coefficients
        )
        base = torch.cat(
            (
                (joint_next[..., 0] - self.state_mean[q_low:q_high])
                / self.state_std[q_low:q_high],
                (joint_next[..., 1] - self.state_mean[v_low:v_high])
                / self.state_std[v_low:v_high],
                state[..., v_high:],
            ),
            dim=-1,
        )
        correction = self.correction_head(
            self.backbone(torch.cat((state, action), dim=-1))
        )
        predicted_raw = (base + correction) * self.state_std + self.state_mean
        joint_margin = 0.20 * (self.joint_upper - self.joint_lower)
        projected_joint_position = torch.clamp(
            predicted_raw[..., q_low:q_high],
            self.joint_lower - joint_margin,
            self.joint_upper + joint_margin,
        )
        quat_low, quat_high = wm.LAYOUT.tote_quaternion
        quaternion = predicted_raw[..., quat_low:quat_high]
        projected_quaternion = quaternion / torch.clamp(
            torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True), min=1e-8
        )
        contact_low = wm.LAYOUT.bilateral_contact[0]
        contact_high = wm.LAYOUT.table_contact[1]
        projected_contacts = torch.clamp(
            predicted_raw[..., contact_low:contact_high], 0.0, 1.0
        )
        progress_low, progress_high = wm.LAYOUT.task_progress
        projected_progress = torch.clamp(
            predicted_raw[..., progress_low:progress_high], 0.0, 1.0
        )
        predicted_raw = torch.cat(
            (
                projected_joint_position,
                predicted_raw[..., q_high:quat_low],
                projected_quaternion,
                predicted_raw[..., quat_high:contact_low],
                projected_contacts,
                predicted_raw[..., contact_high:progress_low],
                projected_progress,
            ),
            dim=-1,
        )
        return (predicted_raw - self.state_mean) / self.state_std


def fit_joint_dynamics(episodes: list[wm.Episode], ridge: float = 1e-4) -> np.ndarray:
    coefficients = np.zeros((wm.LAYOUT.action_dim, 2, 4), dtype=np.float32)
    q_low, q_high = wm.LAYOUT.joint_position
    v_low, v_high = wm.LAYOUT.joint_velocity
    for joint in range(wm.LAYOUT.action_dim):
        inputs = []
        targets = []
        for episode in episodes:
            inputs.append(
                np.stack(
                    (
                        episode.state[:-1, q_low + joint],
                        episode.state[:-1, v_low + joint],
                        episode.action[:, joint],
                        np.ones(len(episode.action), dtype=np.float32),
                    ),
                    axis=1,
                )
            )
            targets.append(
                np.stack(
                    (
                        episode.state[1:, q_low + joint],
                        episode.state[1:, v_low + joint],
                    ),
                    axis=1,
                )
            )
        x = np.concatenate(inputs).astype(np.float64)
        y = np.concatenate(targets).astype(np.float64)
        regularizer = ridge * np.eye(x.shape[1], dtype=np.float64)
        regularizer[-1, -1] = 0.0
        solution = np.linalg.solve(x.T @ x + regularizer, x.T @ y)
        coefficients[joint] = solution.T.astype(np.float32)
    return coefficients


def physical_joint_limits(scene: Path, joint_names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(scene))
    lower = []
    upper = []
    for name in joint_names.astype(str):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint {name} is missing from {scene}")
        low, high = model.jnt_range[joint_id]
        if name == "waist_pitch_joint":
            low, high = max(low, -0.10), min(high, 0.10)
        elif name in ("waist_yaw_joint", "waist_roll_joint"):
            low, high = max(low, -0.05), min(high, 0.05)
        lower.append(low)
        upper.append(high)
    return np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


def weighted_rollout_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    feature_weights = torch.ones(
        wm.LAYOUT.state_dim, device=prediction.device, dtype=prediction.dtype
    )
    q_low, q_high = wm.LAYOUT.joint_position
    v_low, v_high = wm.LAYOUT.joint_velocity
    feature_weights[q_low:q_high] = 2.0
    feature_weights[q_low + 17 : q_high] = 5.0
    feature_weights[v_low:v_high] = 1.5
    feature_weights[v_low + 17 : v_high] = 2.5
    feature_weights[wm.LAYOUT.tote_position[0] : wm.LAYOUT.task_progress[1]] = 2.0
    horizon_weights = torch.linspace(
        1.0, 2.0, prediction.shape[1], device=prediction.device, dtype=prediction.dtype
    )
    squared = (prediction - target) ** 2
    return torch.mean(squared * feature_weights[None, None, :] * horizon_weights[None, :, None])


def build_hybrid_from_checkpoint(checkpoint: dict, device: torch.device) -> HybridWorldModel:
    args = checkpoint["args"]
    stats = checkpoint["normalizers"]
    model = HybridWorldModel(
        stats["state_mean"],
        stats["state_std"],
        stats["action_mean"],
        stats["action_std"],
        checkpoint["joint_coefficients"],
        checkpoint["joint_lower"],
        checkpoint["joint_upper"],
        int(args["hidden_dim"]),
        int(args["depth"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-summary", type=Path, required=True)
    parser.add_argument("--joint-limit-scene", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="G1WM-03-hybrid-constraint-aware-dynamics")
    parser.add_argument("--train-sources", type=int, default=100)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-horizon", type=int, default=20)
    parser.add_argument("--rollout-horizons", type=int, nargs="+", default=(1, 5, 10, 20))
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--max-eval-starts", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=4303)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    train_raw, val_raw, conditions = wi.load_episodes(
        args.transition_summary, args.train_sources
    )
    with np.load(json.loads(args.transition_summary.read_text())["episodes"][0]["dataset"]) as arrays:
        joint_names = arrays["upper_body_joint_names"].astype(str)
    joint_coefficients = fit_joint_dynamics(train_raw)
    joint_lower, joint_upper = physical_joint_limits(args.joint_limit_scene, joint_names)
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
    model = HybridWorldModel(
        stats["state_mean"],
        stats["state_std"],
        stats["action_mean"],
        stats["action_std"],
        joint_coefficients,
        joint_lower,
        joint_upper,
        args.hidden_dim,
        args.depth,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.1
    )
    val_groups = wi.condition_groups(val_raw, conditions)
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
        loss = weighted_rollout_loss(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        optimizer.step()
        scheduler.step()
        record = {
            "step": step,
            "train_weighted_normalized_mse": float(loss),
            "gradient_norm": gradient_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        evaluate_now = step == 1 or step % args.eval_freq == 0 or step == args.steps
        if evaluate_now:
            perturbed = wi.evaluate_groups(
                model,
                {"perturbed": val_groups["perturbed"]},
                stats,
                list(args.rollout_horizons),
                device,
                args.max_eval_starts,
            )["perturbed"]
            hmax = perturbed[f"h{max(args.rollout_horizons)}"]
            score = hmax["joint_position_rmse_rad"]
            record["validation_perturbed"] = perturbed
            print(json.dumps(record), flush=True)
            if score < best_score:
                best_score = score
                best_step = step
                torch.save(
                    {
                        "model": model.state_dict(),
                        "normalizers": stats,
                        "joint_coefficients": joint_coefficients,
                        "joint_lower": joint_lower,
                        "joint_upper": joint_upper,
                        "joint_names": joint_names.tolist(),
                        "args": vars(args),
                    },
                    args.output_dir / "best_model.pt",
                )
        elif step % 100 == 0:
            print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    best = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model = build_hybrid_from_checkpoint(best, device)
    new_metrics = wi.evaluate_groups(
        model,
        val_groups,
        stats,
        list(args.rollout_horizons),
        device,
        args.max_eval_starts,
    )
    baseline_checkpoint = torch.load(args.baseline_checkpoint, map_location=device, weights_only=False)
    baseline_model = wi.build_model_from_checkpoint(baseline_checkpoint, device)
    baseline_metrics = wi.evaluate_groups(
        baseline_model,
        val_groups,
        baseline_checkpoint["normalizers"],
        list(args.rollout_horizons),
        device,
        args.max_eval_starts,
    )
    comparison_horizon = f"h{max(args.rollout_horizons)}"
    old_p = baseline_metrics["perturbed"][comparison_horizon]
    new_p = new_metrics["perturbed"][comparison_horizon]
    old_n = baseline_metrics["nominal"][comparison_horizon]
    new_n = new_metrics["nominal"][comparison_horizon]
    comparisons = {
        "perturbed_joint_position_error_reduction": 1.0 - new_p["joint_position_rmse_rad"] / old_p["joint_position_rmse_rad"],
        "perturbed_tote_position_error_reduction": 1.0 - new_p["tote_position_rmse_m"] / old_p["tote_position_rmse_m"],
        "nominal_joint_position_error_ratio": new_n["joint_position_rmse_rad"] / old_n["joint_position_rmse_rad"],
        "nominal_tote_position_error_ratio": new_n["tote_position_rmse_m"] / old_n["tote_position_rmse_m"],
    }
    checks = {
        "perturbed_h20_joint_position_rmse_below_0.03_rad": new_p["joint_position_rmse_rad"] < 0.03,
        "perturbed_h20_tote_position_rmse_below_2cm": new_p["tote_position_rmse_m"] < 0.02,
        "perturbed_h20_lift_height_rmse_below_2cm": new_p["lift_height_rmse_m"] < 0.02,
        "perturbed_joint_error_reduced_vs_g1wm02_at_least_40pct": comparisons["perturbed_joint_position_error_reduction"] >= 0.40,
        "nominal_joint_error_no_more_than_50pct_worse": comparisons["nominal_joint_position_error_ratio"] <= 1.50,
        "nominal_tote_error_no_more_than_50pct_worse": comparisons["nominal_tote_position_error_ratio"] <= 1.50,
    }
    summary = {
        "experiment": args.experiment_id,
        "model_type": "per-joint affine servo dynamics plus neural residual and physical projection",
        "train_rollouts": len(train_raw),
        "validation_rollouts": len(val_raw),
        "source_episode_split_leakage": False,
        "best_step": best_step,
        "best_perturbed_joint_position_rmse_rad": best_score,
        "comparison_horizon": comparison_horizon,
        "baseline_g1wm02_metrics": baseline_metrics,
        "g1wm03_metrics": new_metrics,
        "comparisons": comparisons,
        "screening_checks": checks,
        "experiment_passed": all(checks.values()),
        "reference_note": "Thresholds are project screening targets, not universal robotics standards.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    wm.save_preview(model, val_groups["perturbed"][0], stats, args.output_dir, device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
