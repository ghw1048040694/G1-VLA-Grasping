#!/usr/bin/env python3
"""Train a learned contact-aware world model on G1 language episodes.

The model predicts the next proprioceptive, palm, object and contact state from
the current state, target language class and a 31-D joint-position action.  It
is deliberately independent of MuJoCo equality constraints: object motion is
learned from the recorded contact-only transitions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


OBJECT_NAMES = ("red_triangle", "yellow_rod", "green_cube")
STATE_DIM = 31 + 31 + 9 + 6 + 1 + 8
ACTION_DIM = 31
QPOS = slice(0, 31)
QVEL = slice(31, 62)
OBJECT = slice(62, 71)
PALM = slice(71, 77)
CONTACT = 77
PHASE = slice(78, 86)


@dataclass
class Episode:
    name: str
    state: np.ndarray
    action: np.ndarray
    target: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=9)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def upper_state(arrays: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, int]:
    # The source contract is 12 lower-body joints followed by 31 upper-body.
    qpos = arrays["joint_position_rad"][:, 12:].astype(np.float32)
    qvel = arrays["joint_velocity_rad_s"][:, 12:].astype(np.float32)
    objects = arrays["object_position_m"].reshape(len(qpos), 9).astype(np.float32)
    palms = arrays["palm_position_m"].reshape(len(qpos), 6).astype(np.float32)
    contact = arrays["physical_contact_active"].astype(np.float32)[:, None]
    phase_ids = arrays["task_phase"].astype(np.int64)
    phase = np.eye(8, dtype=np.float32)[np.clip(phase_ids, 0, 7)]
    state = np.concatenate((qpos, qvel, objects, palms, contact, phase), axis=1)
    target = int(arrays["target_object_index"][0])
    action = arrays["action_joint_position_rad"][:, 12:].astype(np.float32)
    if state.shape[1] != STATE_DIM or action.shape[1] != ACTION_DIM:
        raise ValueError(f"unexpected state/action shape {state.shape} {action.shape}")
    return state, action, target


def load_episodes(root: Path) -> list[Episode]:
    episodes = []
    for directory in sorted(root.glob("episode_*")):
        path = directory / "expert_pick_place_episode.npz"
        if not path.is_file():
            continue
        with np.load(path) as arrays:
            state, action, target = upper_state(arrays)
        episodes.append(Episode(directory.name, state, action, target))
    if len(episodes) < 2:
        raise ValueError(f"Need at least two episodes under {root}")
    return episodes


def normalizers(episodes: list[Episode]) -> dict[str, np.ndarray]:
    states = np.concatenate([e.state for e in episodes])
    actions = np.concatenate([e.action for e in episodes])
    return {
        "state_mean": states.mean(0).astype(np.float32),
        "state_std": np.maximum(states.std(0), 1e-4).astype(np.float32),
        "action_mean": actions.mean(0).astype(np.float32),
        "action_std": np.maximum(actions.std(0), 1e-4).astype(np.float32),
    }


class TransitionDataset(Dataset):
    def __init__(self, episodes: list[Episode], stats: dict[str, np.ndarray], horizon: int):
        self.samples = []
        self.horizon = horizon
        for episode in episodes:
            state = (episode.state - stats["state_mean"]) / stats["state_std"]
            action = (episode.action - stats["action_mean"]) / stats["action_std"]
            target = np.eye(3, dtype=np.float32)[episode.target]
            for start in range(len(state) - horizon):
                self.samples.append(
                    (state[start], action[start : start + horizon], state[start + 1 : start + horizon + 1], target)
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        state, action, next_state, target = self.samples[index]
        return tuple(torch.from_numpy(x) for x in (state, action, next_state, target))


class ContactWorldModel(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, hidden_dim: int = 512, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(state_dim + action_dim + 3, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        self.backbone = nn.Sequential(*layers)
        self.delta_head = nn.Linear(hidden_dim, state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = self.delta_head(self.backbone(torch.cat((state, action, target), dim=-1)))
        return state + delta


def rollout(model: ContactWorldModel, state: torch.Tensor, actions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predictions = []
    for step in range(actions.shape[1]):
        state = model(state, actions[:, step], target)
        predictions.append(state)
    return torch.stack(predictions, dim=1)


def weighted_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = torch.ones(STATE_DIM, device=prediction.device, dtype=prediction.dtype)
    weights[QPOS] = 2.0
    weights[QVEL] = 1.5
    weights[OBJECT] = 5.0
    weights[PALM] = 5.0
    weights[CONTACT] = 2.0
    weights[PHASE] = 0.5
    return ((prediction - target) ** 2 * weights).mean()


@torch.no_grad()
def evaluate(model, episodes, stats, horizon, device):
    model.eval()
    metrics = {}
    mean = torch.from_numpy(stats["state_mean"]).to(device)
    std = torch.from_numpy(stats["state_std"]).to(device)
    action_mean = torch.from_numpy(stats["action_mean"]).to(device)
    action_std = torch.from_numpy(stats["action_std"]).to(device)
    for length in (1, 5, 10, horizon):
        errors = {"qpos": [], "object": [], "palm": [], "contact": []}
        for episode in episodes:
            if len(episode.state) <= length:
                continue
            starts = np.linspace(0, len(episode.state) - length - 1, min(64, len(episode.state) - length), dtype=int)
            target_onehot = torch.tensor(np.eye(3, dtype=np.float32)[episode.target], device=device).expand(len(starts), -1)
            initial = torch.from_numpy((episode.state[starts] - stats["state_mean"]) / stats["state_std"]).to(device)
            actions = []
            targets = []
            for start in starts:
                actions.append((episode.action[start : start + length] - stats["action_mean"]) / stats["action_std"])
                targets.append((episode.state[start + 1 : start + length + 1] - stats["state_mean"]) / stats["state_std"])
            action_t = torch.from_numpy(np.stack(actions)).to(device)
            target_t = torch.from_numpy(np.stack(targets)).to(device)
            prediction = rollout(model, initial, action_t, target_onehot)
            pred_raw = prediction[:, -1] * std + mean
            true_raw = target_t[:, -1] * std + mean
            errors["qpos"].append(torch.sqrt(torch.mean((pred_raw[:, QPOS] - true_raw[:, QPOS]) ** 2)).item())
            errors["object"].append(torch.sqrt(torch.mean((pred_raw[:, OBJECT] - true_raw[:, OBJECT]) ** 2)).item())
            errors["palm"].append(torch.sqrt(torch.mean((pred_raw[:, PALM] - true_raw[:, PALM]) ** 2)).item())
            errors["contact"].append(torch.mean(((pred_raw[:, CONTACT] >= 0.5) == (true_raw[:, CONTACT] >= 0.5)).float()).item())
        metrics[f"h{length}"] = {key: float(np.mean(value)) for key, value in errors.items()}
    return metrics


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    episodes = load_episodes(args.dataset_root)
    if not 0 < args.train_episodes < len(episodes):
        raise ValueError("--train-episodes must leave validation episodes")
    train, val = episodes[: args.train_episodes], episodes[args.train_episodes :]
    stats = normalizers(train)
    train_ds = TransitionDataset(train, stats, args.horizon)
    val_ds = TransitionDataset(val, stats, args.horizon)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = ContactWorldModel(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best = math.inf
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for state, action, next_state, target in train_loader:
            state, action, next_state, target = [x.to(device) for x in (state, action, next_state, target)]
            # Small action perturbations expose the model to MPC candidates while
            # retaining the measured contact transition as the supervision target.
            noisy_action = action + 0.02 * torch.randn_like(action)
            prediction = rollout(model, state, noisy_action, target)
            loss = weighted_loss(prediction, next_state)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(state)
        metrics = evaluate(model, val, stats, args.horizon, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = total / max(len(train_ds), 1)
        history.append(metrics)
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print("LANGUAGE_WM_EPOCH=" + json.dumps(metrics), flush=True)
        score = metrics[f"h{args.horizon}"]["object"] + metrics[f"h{args.horizon}"]["palm"]
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "stats": stats, "config": vars(args), "state_dim": STATE_DIM, "action_dim": ACTION_DIM}, args.output_dir / "best.pt")
    summary = {
        "experiment": "G1LANG-96-contact-only-language-world-model",
        "dataset_root": str(args.dataset_root),
        "train_episodes": [e.name for e in train],
        "validation_episodes": [e.name for e in val],
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": {"qpos": [0, 31], "qvel": [31, 62], "object": [62, 71], "palm": [71, 77], "contact": [77, 78], "phase_onehot": [78, 86]},
        "horizon": args.horizon,
        "best_validation_score": best,
        "final_validation": history[-1],
        "history": history,
        "contact_only": True,
        "equality_constraints": False,
        "passed": bool(history[-1][f"h{args.horizon}"]["object"] < 0.06 and history[-1][f"h{args.horizon}"]["palm"] < 0.06),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("history",)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
