#!/usr/bin/env python3
"""Train a raw three-camera object-pose posterior on random evaluator scenes."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from g1_raw_multiview_servo_vla import ViewEncoder


class RandomPerceptionDataset(Dataset):
    def __init__(self, root: Path, names: list[str], augment: bool) -> None:
        self.root = root
        self.names = names
        self.augment = augment

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        episode = self.root / self.names[index]
        frames = []
        for name in ("head_camera.mp4", "left_wrist_camera.mp4", "right_wrist_camera.mp4"):
            capture = cv2.VideoCapture(str(episode / name))
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise RuntimeError(f"Could not read first RGB frame: {episode / name}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with np.load(episode / "trajectory.npz") as value:
            pose = value["object_position_m"][0].astype(np.float32).reshape(-1)
        image = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255.0)
        if self.augment:
            brightness = 0.85 + 0.30 * torch.rand(1)
            contrast = 0.85 + 0.30 * torch.rand(1)
            image = ((image - 0.5) * contrast + 0.5) * brightness
            image = (image + 0.01 * torch.randn_like(image)).clamp(0.0, 1.0)
        return image, torch.from_numpy(pose)


class RandomMultiviewPoseModel(nn.Module):
    def __init__(self, hidden_dim: int = 384) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder = ViewEncoder()
        self.view_embedding = nn.Parameter(torch.randn(3, 192) * 0.02)
        self.fusion = nn.Sequential(
            nn.Linear(3 * 192, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.pose = nn.Linear(hidden_dim, 9)
        self.log_std = nn.Linear(hidden_dim, 9)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 5 or images.shape[1:3] != (3, 3):
            raise ValueError(f"Expected [B,3,3,H,W], got {tuple(images.shape)}")
        pixels = F.interpolate(images.flatten(0, 1), size=(64, 96), mode="bilinear", align_corners=False)
        view = self.encoder(pixels).reshape(len(images), 3, -1) + self.view_embedding[None]
        hidden = self.fusion(view.flatten(1))
        return self.pose(hidden), self.log_std(hidden).clamp(-5.0, -1.0)


def save_checkpoint(path: Path, model: RandomMultiviewPoseModel, mean: torch.Tensor, std: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"hidden_dim": model.hidden_dim},
            "pose_mean": mean.detach().cpu(),
            "pose_std": std.detach().cpu(),
            "input_contract": "single synchronized raw RGB frame from head/left-wrist/right-wrist cameras",
            "labels": "offline MuJoCo object positions used only during training",
            "privileged_positions_at_inference": False,
        },
        path,
    )


def load_checkpoint(path: Path, device: str):
    value = torch.load(path, map_location=device, weights_only=False)
    model = RandomMultiviewPoseModel(**value["config"]).to(device)
    model.load_state_dict(value["model"])
    model.eval()
    return model, value["pose_mean"].to(device), value["pose_std"].to(device)


def metrics(model, loader, mean, std, device) -> dict:
    model.eval()
    squared = 0.0
    values = []
    with torch.no_grad():
        for images, pose in loader:
            images = images.to(device); pose = pose.to(device)
            raw, _ = model(images)
            prediction = raw * std + mean
            error = torch.linalg.vector_norm((prediction - pose).reshape(-1, 3), dim=-1)
            squared += float(error.square().sum())
            values.append(error.cpu())
    error = torch.cat(values)
    return {
        "object_position_rmse_mm": 1e3 * math.sqrt(squared / len(error)),
        "object_position_mean_mm": 1e3 * float(error.mean()),
        "object_position_p95_mm": 1e3 * float(torch.quantile(error, 0.95)),
        "object_position_max_mm": 1e3 * float(error.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260741)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    names = sorted(path.name for path in args.data_root.glob("episode_*") if (path / "trajectory.npz").is_file())
    if len(names) < 30:
        raise ValueError("Need at least 30 complete random perception episodes")
    split = max(3, round(len(names) * 0.20))
    train_names, validation_names = names[:-split], names[-split:]
    train_set = RandomPerceptionDataset(args.data_root, train_names, True)
    validation_set = RandomPerceptionDataset(args.data_root, validation_names, False)
    bootstrap = DataLoader(RandomPerceptionDataset(args.data_root, train_names, False), batch_size=len(train_names))
    _, all_pose = next(iter(bootstrap))
    mean = all_pose.mean(0); std = all_pose.std(0).clamp_min(1e-3)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    mean, std = mean.to(device), std.to(device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = RandomMultiviewPoseModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf"); history = []
    for epoch in range(args.epochs):
        model.train(); total = 0.0; rows = 0
        for images, pose in train_loader:
            images = images.to(device, non_blocking=True); pose = pose.to(device, non_blocking=True)
            raw, _ = model(images); target = (pose - mean) / std
            # With only 120 randomized layouts, learned heteroscedastic NLL
            # initially widens its variance instead of fitting image geometry.
            loss = F.mse_loss(raw, target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss) * len(images); rows += len(images)
        report = metrics(model, validation_loader, mean, std, device)
        report.update(epoch=epoch + 1, train_loss=total / max(rows, 1)); history.append(report)
        print("RANDOM_MULTIVIEW_PERCEPTION_EPOCH=" + json.dumps(report), flush=True)
        if report["object_position_rmse_mm"] < best:
            best = report["object_position_rmse_mm"]
            save_checkpoint(args.output_dir / "best.pt", model, mean, std)
    summary = {
        "experiment": "G1LANG-random-multiview-raw-RGB-pose-posterior",
        "data_root": str(args.data_root),
        "train_episodes": len(train_names),
        "validation_episodes": len(validation_names),
        "input_contract": "single raw RGB frame x3; no simulator object state at inference",
        "best_validation_object_position_rmse_mm": best,
        "history": history,
        "final": history[-1],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
