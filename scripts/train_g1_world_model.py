#!/usr/bin/env python3
"""Train an action-conditioned state world model for the G1 tote-lift task."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


FEATURE_KEYS = (
    "observation_joint_position_rad",
    "observation_joint_velocity_rad_s",
    "action_joint_position_rad",
    "tote_position_m",
    "tote_quaternion_wxyz",
    "tote_linear_velocity_m_s",
    "tote_angular_velocity_rad_s",
    "palm_position_m",
    "bilateral_hand_contact",
    "table_contact",
    "tote_lift_height_m",
    "task_progress",
)


@dataclass(frozen=True)
class Layout:
    joint_position: tuple[int, int] = (0, 31)
    joint_velocity: tuple[int, int] = (31, 62)
    tote_position: tuple[int, int] = (62, 65)
    tote_quaternion: tuple[int, int] = (65, 69)
    tote_linear_velocity: tuple[int, int] = (69, 72)
    tote_angular_velocity: tuple[int, int] = (72, 75)
    palm_position: tuple[int, int] = (75, 81)
    bilateral_contact: tuple[int, int] = (81, 83)
    table_contact: tuple[int, int] = (83, 84)
    lift_height: tuple[int, int] = (84, 85)
    task_progress: tuple[int, int] = (85, 86)
    state_dim: int = 86
    action_dim: int = 31


LAYOUT = Layout()


@dataclass
class Episode:
    episode_index: int
    state: np.ndarray
    action: np.ndarray


class SequenceDataset(Dataset):
    def __init__(self, episodes: list[Episode], horizon: int) -> None:
        self.episodes = episodes
        self.horizon = horizon
        self.index = [
            (episode_id, start)
            for episode_id, episode in enumerate(episodes)
            for start in range(len(episode.state) - horizon)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        episode_id, start = self.index[index]
        episode = self.episodes[episode_id]
        return (
            torch.from_numpy(episode.state[start]),
            torch.from_numpy(episode.action[start : start + self.horizon]),
            torch.from_numpy(episode.state[start + 1 : start + self.horizon + 1]),
        )


class ResidualWorldModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(state_dim + action_dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        self.backbone = nn.Sequential(*layers)
        self.delta_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.delta_head(self.backbone(torch.cat((state, action), dim=-1)))
        return state + delta


def upper_body_indices(joint_names: np.ndarray) -> np.ndarray:
    excluded = ("hip_", "knee_", "ankle_")
    indices = [
        index
        for index, name in enumerate(joint_names.astype(str))
        if not any(part in name for part in excluded)
    ]
    if len(indices) != LAYOUT.action_dim:
        raise ValueError(f"Expected 31 upper-body joints, found {len(indices)}")
    return np.asarray(indices, dtype=np.int64)


def load_episode(path: Path, episode_index: int) -> Episode:
    with np.load(path) as arrays:
        missing = set(FEATURE_KEYS) - set(arrays.files)
        if missing:
            raise ValueError(f"{path} is missing world-model fields: {sorted(missing)}")
        upper = upper_body_indices(arrays["joint_names"])
        state = np.concatenate(
            (
                arrays["observation_joint_position_rad"][:, upper],
                arrays["observation_joint_velocity_rad_s"][:, upper],
                arrays["tote_position_m"],
                arrays["tote_quaternion_wxyz"],
                arrays["tote_linear_velocity_m_s"],
                arrays["tote_angular_velocity_rad_s"],
                arrays["palm_position_m"].reshape(len(arrays["palm_position_m"]), -1),
                arrays["bilateral_hand_contact"],
                arrays["table_contact"][:, None],
                arrays["tote_lift_height_m"][:, None],
                arrays["task_progress"][:, None],
            ),
            axis=1,
        ).astype(np.float32)
        action = arrays["action_joint_position_rad"][:, upper].astype(np.float32)
    if state.shape[1] != LAYOUT.state_dim or action.shape[1] != LAYOUT.action_dim:
        raise ValueError(f"Unexpected shapes in {path}: state={state.shape}, action={action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"Non-finite values in {path}")
    return Episode(episode_index=episode_index, state=state, action=action)


def load_split(summary_path: Path, train_count: int, val_count: int) -> tuple[list[Episode], list[Episode]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    successful = [item for item in summary["episodes"] if item["success"]]
    if len(successful) < train_count + val_count:
        raise ValueError(
            f"Need {train_count + val_count} successful episodes, found {len(successful)}"
        )
    selected = successful[: train_count + val_count]
    episodes = [
        load_episode(Path(item["dataset"]), int(item["episode_index"])) for item in selected
    ]
    train = episodes[:train_count]
    val = episodes[train_count:]
    overlap = {item.episode_index for item in train} & {item.episode_index for item in val}
    if overlap:
        raise RuntimeError(f"Episode leakage detected: {sorted(overlap)}")
    return train, val


def normalizers(episodes: list[Episode]) -> dict[str, np.ndarray]:
    states = np.concatenate([item.state for item in episodes])
    actions = np.concatenate([item.action for item in episodes])
    return {
        "state_mean": states.mean(axis=0).astype(np.float32),
        "state_std": np.maximum(states.std(axis=0), 1e-4).astype(np.float32),
        "action_mean": actions.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(actions.std(axis=0), 1e-4).astype(np.float32),
    }


def normalized_copy(episodes: list[Episode], stats: dict[str, np.ndarray]) -> list[Episode]:
    return [
        Episode(
            episode_index=item.episode_index,
            state=((item.state - stats["state_mean"]) / stats["state_std"]).astype(np.float32),
            action=((item.action - stats["action_mean"]) / stats["action_std"]).astype(np.float32),
        )
        for item in episodes
    ]


def rollout(model: nn.Module, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    predictions = []
    for step in range(actions.shape[1]):
        state = model(state, actions[:, step])
        predictions.append(state)
    return torch.stack(predictions, dim=1)


def sampled_starts(episodes: list[Episode], horizon: int, limit: int) -> list[tuple[int, int]]:
    starts = [
        (episode_id, start)
        for episode_id, episode in enumerate(episodes)
        for start in range(len(episode.state) - horizon)
    ]
    if len(starts) <= limit:
        return starts
    indices = np.linspace(0, len(starts) - 1, limit, dtype=np.int64)
    return [starts[index] for index in indices]


def group_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    def rmse(span: tuple[int, int]) -> float:
        low, high = span
        return float(np.sqrt(np.mean((prediction[:, low:high] - target[:, low:high]) ** 2)))

    q_low, q_high = LAYOUT.tote_quaternion
    pred_q = prediction[:, q_low:q_high]
    true_q = target[:, q_low:q_high]
    pred_q /= np.maximum(np.linalg.norm(pred_q, axis=1, keepdims=True), 1e-8)
    true_q /= np.maximum(np.linalg.norm(true_q, axis=1, keepdims=True), 1e-8)
    dot = np.clip(np.abs(np.sum(pred_q * true_q, axis=1)), 0.0, 1.0)
    angle_deg = np.degrees(2.0 * np.arccos(dot))
    contact_low, _ = LAYOUT.bilateral_contact
    _, table_high = LAYOUT.table_contact
    contact_pred = prediction[:, contact_low:table_high] >= 0.5
    contact_true = target[:, contact_low:table_high] >= 0.5
    return {
        "joint_position_rmse_rad": rmse(LAYOUT.joint_position),
        "joint_velocity_rmse_rad_s": rmse(LAYOUT.joint_velocity),
        "tote_position_rmse_m": rmse(LAYOUT.tote_position),
        "tote_orientation_error_deg": float(np.mean(angle_deg)),
        "tote_linear_velocity_rmse_m_s": rmse(LAYOUT.tote_linear_velocity),
        "tote_angular_velocity_rmse_rad_s": rmse(LAYOUT.tote_angular_velocity),
        "palm_position_rmse_m": rmse(LAYOUT.palm_position),
        "lift_height_rmse_m": rmse(LAYOUT.lift_height),
        "task_progress_rmse": rmse(LAYOUT.task_progress),
        "contact_accuracy": float(np.mean(contact_pred == contact_true)),
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    normalized_episodes: list[Episode],
    stats: dict[str, np.ndarray],
    horizons: list[int],
    device: torch.device,
    max_starts: int,
) -> dict[str, dict[str, float]]:
    model.eval()
    results = {}
    mean = stats["state_mean"]
    std = stats["state_std"]
    for horizon in horizons:
        starts = sampled_starts(normalized_episodes, horizon, max_starts)
        predicted_batches = []
        target_batches = []
        initial_batches = []
        normalized_errors = []
        for offset in range(0, len(starts), 256):
            batch = starts[offset : offset + 256]
            initial = np.stack([normalized_episodes[e].state[s] for e, s in batch])
            actions = np.stack(
                [normalized_episodes[e].action[s : s + horizon] for e, s in batch]
            )
            target = np.stack(
                [normalized_episodes[e].state[s + 1 : s + horizon + 1] for e, s in batch]
            )
            initial_t = torch.from_numpy(initial).to(device)
            actions_t = torch.from_numpy(actions).to(device)
            prediction = rollout(model, initial_t, actions_t).cpu().numpy()
            normalized_errors.append(np.mean((prediction - target) ** 2))
            predicted_batches.append(prediction[:, -1] * std + mean)
            target_batches.append(target[:, -1] * std + mean)
            initial_batches.append(initial * std + mean)
        prediction = np.concatenate(predicted_batches)
        target = np.concatenate(target_batches)
        initial = np.concatenate(initial_batches)
        metrics = group_metrics(prediction, target)
        baseline = group_metrics(initial, target)
        metrics["normalized_rollout_mse"] = float(np.mean(normalized_errors))
        metrics["samples"] = len(starts)
        for key, value in baseline.items():
            metrics[f"persistence_{key}"] = value
            if key.endswith("rmse_rad") or "rmse" in key or key.endswith("error_deg"):
                metrics[f"improvement_vs_persistence_{key}"] = float(
                    1.0 - metrics[key] / max(value, 1e-12)
                )
        results[f"h{horizon}"] = metrics
    return results


def save_preview(
    model: nn.Module,
    episode: Episode,
    stats: dict[str, np.ndarray],
    output_dir: Path,
    device: torch.device,
    horizon: int = 20,
) -> None:
    lift_index = LAYOUT.lift_height[0]
    lift_frames = np.flatnonzero(episode.state[:, lift_index] > 1e-4)
    start = max(0, int(lift_frames[0]) - 5) if len(lift_frames) else 0
    horizon = min(horizon, len(episode.state) - start - 1)
    normalized_state = (episode.state - stats["state_mean"]) / stats["state_std"]
    normalized_action = (episode.action - stats["action_mean"]) / stats["action_std"]
    with torch.inference_mode():
        prediction = rollout(
            model,
            torch.from_numpy(normalized_state[None, start].astype(np.float32)).to(device),
            torch.from_numpy(normalized_action[None, start : start + horizon].astype(np.float32)).to(device),
        )[0].cpu().numpy()
    prediction = prediction * stats["state_std"] + stats["state_mean"]
    target = episode.state[start + 1 : start + horizon + 1]
    steps = np.arange(1, horizon + 1)
    joint_low, joint_high = LAYOUT.joint_position
    joint_rmse = np.sqrt(np.mean((prediction[:, joint_low:joint_high] - target[:, joint_low:joint_high]) ** 2, axis=1))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    lift = lift_index
    progress = LAYOUT.task_progress[0]
    axes[0, 0].plot(steps, target[:, lift], label="true")
    axes[0, 0].plot(steps, prediction[:, lift], label="predicted")
    axes[0, 0].set(title="Tote lift height", xlabel="rollout step", ylabel="m")
    axes[0, 0].legend()
    axes[0, 1].plot(steps, target[:, progress], label="true")
    axes[0, 1].plot(steps, prediction[:, progress], label="predicted")
    axes[0, 1].set(title="Task progress", xlabel="rollout step", ylabel="0-1")
    axes[0, 1].legend()
    pos_low, pos_high = LAYOUT.tote_position
    for axis, label in enumerate("xyz"):
        axes[1, 0].plot(steps, target[:, pos_low + axis], label=f"true {label}")
        axes[1, 0].plot(steps, prediction[:, pos_low + axis], linestyle="--", label=f"pred {label}")
    axes[1, 0].set(title="Tote position", xlabel="rollout step", ylabel="m")
    axes[1, 0].legend(ncol=2, fontsize=8)
    axes[1, 1].plot(steps, joint_rmse)
    axes[1, 1].set(title="Upper-body joint position error", xlabel="rollout step", ylabel="RMSE (rad)")
    fig.suptitle(
        f"G1 world-model rollout, held-out episode {episode.episode_index}, start frame {start}"
    )
    fig.savefig(output_dir / "rollout_preview.png", dpi=150)
    plt.close(fig)

    with (output_dir / "rollout_preview.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("step", "true_lift_m", "predicted_lift_m", "true_progress", "predicted_progress", "joint_position_rmse_rad"))
        for index in range(horizon):
            writer.writerow((steps[index], target[index, lift], prediction[index, lift], target[index, progress], prediction[index, progress], joint_rmse[index]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="G1WM-01-action-conditioned-state-world-model")
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--val-episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-horizon", type=int, default=10)
    parser.add_argument("--rollout-horizons", type=int, nargs="+", default=(1, 5, 10, 20))
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--max-eval-starts", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=4101)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.steps < 1 or args.train_horizon < 1:
        parser.error("--steps and --train-horizon must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    train_raw, val_raw = load_split(args.source_summary, args.train_episodes, args.val_episodes)
    stats = normalizers(train_raw)
    train = normalized_copy(train_raw, stats)
    val = normalized_copy(val_raw, stats)
    dataset = SequenceDataset(train, args.train_horizon)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0, generator=generator)
    iterator = iter(loader)
    model = ResidualWorldModel(LAYOUT.state_dim, LAYOUT.action_dim, args.hidden_dim, args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.1)
    log_path = args.output_dir / "training_log.jsonl"
    best_score = math.inf
    best_step = 0

    for step in range(1, args.steps + 1):
        try:
            initial, actions, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            initial, actions, targets = next(iterator)
        initial = initial.to(device)
        actions = actions.to(device)
        targets = targets.to(device)
        model.train()
        prediction = rollout(model, initial, actions)
        loss = torch.mean((prediction - targets) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        optimizer.step()
        scheduler.step()

        should_evaluate = step == 1 or step % args.eval_freq == 0 or step == args.steps
        record = {
            "step": step,
            "train_normalized_mse": float(loss),
            "gradient_norm": gradient_norm,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        if should_evaluate:
            metrics = evaluate(model, val, stats, list(args.rollout_horizons), device, args.max_eval_starts)
            score = metrics[f"h{max(args.rollout_horizons)}"]["normalized_rollout_mse"]
            record["validation"] = metrics
            print(json.dumps(record), flush=True)
            if score < best_score:
                best_score = score
                best_step = step
                torch.save({"model": model.state_dict(), "normalizers": stats, "layout": asdict(LAYOUT), "args": vars(args)}, args.output_dir / "best_model.pt")
        elif step % 100 == 0:
            print(json.dumps(record), flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    checkpoint = torch.load(args.output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, val, stats, list(args.rollout_horizons), device, args.max_eval_starts)
    h1 = final_metrics.get("h1", {})
    h10 = final_metrics.get("h10", final_metrics[f"h{max(args.rollout_horizons)}"])
    h20 = final_metrics.get("h20", final_metrics[f"h{max(args.rollout_horizons)}"])
    checks = {
        "h1_joint_position_improvement_at_least_50pct": h1.get("improvement_vs_persistence_joint_position_rmse_rad", -math.inf) >= 0.50,
        "h10_joint_position_improvement_at_least_25pct": h10.get("improvement_vs_persistence_joint_position_rmse_rad", -math.inf) >= 0.25,
        "h20_tote_position_rmse_below_5cm": h20["tote_position_rmse_m"] < 0.05,
        "h20_lift_height_rmse_below_3cm": h20["lift_height_rmse_m"] < 0.03,
        "h20_rollout_growth_below_4x_h10": h20["normalized_rollout_mse"] < 4.0 * h10["normalized_rollout_mse"],
    }
    summary = {
        "experiment": args.experiment_id,
        "model_type": "action-conditioned residual multilayer perceptron state world model",
        "device": str(device),
        "train_episode_indices": [item.episode_index for item in train_raw],
        "validation_episode_indices": [item.episode_index for item in val_raw],
        "episode_split_leakage": False,
        "train_transitions": int(sum(len(item.state) - 1 for item in train_raw)),
        "validation_transitions": int(sum(len(item.state) - 1 for item in val_raw)),
        "state_layout": asdict(LAYOUT),
        "best_step": best_step,
        "best_horizon_score": best_score,
        "metrics": final_metrics,
        "screening_checks": checks,
        "experiment_passed": all(checks.values()),
        "reference_note": "Thresholds are project screening targets, not universal robotics standards.",
        "limitation": "This first model is trained on expert trajectories only; policy-failure and out-of-distribution data are not yet represented.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    save_preview(model, val_raw[0], stats, args.output_dir, device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
