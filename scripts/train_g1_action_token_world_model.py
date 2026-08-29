#!/usr/bin/env python3
"""Fit the action-token MPC world model in the VLA's learned visual state space."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from g1_action_token_vla import load_checkpoint as load_vla
from g1_direct_joint_candidate_vla import encode_tasks
from g1_temporal_chunk_world_model import TemporalChunkWorldEnsemble, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--vla-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=34)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260744)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


@torch.no_grad()
def visual_pose(cache: dict, checkpoint: Path, device: torch.device, output: Path) -> torch.Tensor:
    if output.is_file():
        return torch.load(output, map_location="cpu", weights_only=False)
    model, vocab, stats, _, _ = load_vla(checkpoint, str(device))
    if vocab != cache["vocab"]:
        raise ValueError("VLA vocabulary disagrees with world-model cache")
    images = np.load(cache["image_path"], mmap_mode="r")
    episodes, frames = cache["state"].shape[:2]
    total = frames + 1
    if images.shape[1] != total:
        raise ValueError("World-model cache must provide successor RGB observations")
    sequence = {
        name: torch.cat((cache[name], cache[f"next_{name}"][:, -1:]), dim=1)
        for name in ("state", "velocity", "contact_sites", "contact")
    }
    result = torch.empty((episodes, total, 9), dtype=torch.float32)
    indices = np.maximum(
        np.arange(total)[:, None] - model.history_frames + 1 + np.arange(model.history_frames)[None], 0
    )
    for episode in range(episodes):
        task = cache["tasks"][int(cache["task_index"][episode])]
        for start in range(0, total, 32):
            stop = min(start + 32, total)
            index = indices[start:stop]
            pixels = torch.from_numpy(np.asarray(images[episode, index]).copy())
            pixels = pixels.permute(0, 1, 2, 5, 3, 4).float().div_(255.0).to(device)
            state = (sequence["state"][episode, index].float().to(device) - stats["state_mean"]) / stats["state_std"]
            velocity = (sequence["velocity"][episode, index].float().to(device) - stats["velocity_mean"]) / stats["velocity_std"]
            sites = (sequence["contact_sites"][episode, index].float().to(device) - stats["contact_sites_mean"]) / stats["contact_sites_std"]
            ids, mask = encode_tasks([task] * (stop - start), vocab)
            prediction = model(pixels, state, velocity, sites, sequence["contact"][episode, index].float().to(device), ids.to(device), mask.to(device))
            result[episode, start:stop] = (prediction["object_mean"] * stats["objects_std"] + stats["objects_mean"]).cpu()
        print(json.dumps({"action_token_visual_pose_episode": episode + 1, "total": episodes}), flush=True)
    torch.save(result, output)
    return result


def world_stats(cache: dict, pose: torch.Tensor) -> dict[str, torch.Tensor]:
    valid = cache["valid_frame"].bool()
    values = {
        "state": cache["state"][valid].float(),
        "velocity": cache["velocity"][valid].float(),
        "objects": pose[:, :-1][valid].float(),
        "contact_sites": cache["contact_sites"][valid].float(),
        "action": cache["action_delta"][valid].float(),
    }
    return {
        f"{name}_{part}": getattr(value, part)(0).clamp_min(1e-3) if part == "std" else getattr(value, part)(0)
        for name, value in values.items()
        for part in ("mean", "std")
    }


class TransitionDataset(Dataset):
    def __init__(self, cache: dict, pose: torch.Tensor, weighted: bool) -> None:
        self.cache = cache
        self.pose = pose.float()
        self.rows = np.argwhere(cache["valid_frame"].bool().numpy())
        self.weights = cache["sample_weight"].numpy()[cache["valid_frame"].bool().numpy()].astype(np.float64) if weighted else np.ones(len(self.rows))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, frame = self.rows[index]
        output = {name: self.cache[name][episode, frame].float() for name in (
            "state", "velocity", "action_delta", "next_state", "next_velocity", "contact_sites", "next_contact_sites", "contact"
        )}
        output["objects"] = self.pose[episode, frame]
        output["next_objects"] = self.pose[episode, frame + 1]
        output["target"] = self.cache["target"][episode].long()
        return output


def z(value: torch.Tensor, stats: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return (value - stats[f"{name}_mean"]) / stats[f"{name}_std"]


def outputs(model, batch, stats):
    return model(
        z(batch["state"], stats, "state"),
        z(batch["velocity"], stats, "velocity"),
        z(batch["objects"], stats, "objects"),
        z(batch["contact_sites"], stats, "contact_sites"),
        batch["contact"],
        z(batch["action_delta"], stats, "action"),
        F.one_hot(batch["target"], 3).float(),
    )


def member_loss(prediction, batch, stats):
    return (
        F.mse_loss(prediction["state"], z(batch["next_state"], stats, "state"))
        + .20 * F.mse_loss(prediction["velocity"], z(batch["next_velocity"], stats, "velocity"))
        + .75 * F.mse_loss(prediction["objects"], z(batch["next_objects"], stats, "objects"))
        + .45 * F.mse_loss(prediction["contact_sites"], z(batch["next_contact_sites"], stats, "contact_sites"))
    )


@torch.no_grad()
def evaluate(model, loader, stats, device):
    model.eval()
    total = square = disagreement = 0.0
    rows = count = 0
    for raw in loader:
        batch = {key: item.to(device) for key, item in raw.items()}
        prediction = outputs(model, batch, stats)
        total += float(torch.stack([member_loss(item, batch, stats) for item in prediction]).mean()) * len(batch["target"])
        mean = torch.stack([item["objects"] for item in prediction]).mean(0) * stats["objects_std"] + stats["objects_mean"]
        square += float((mean - batch["next_objects"]).square().sum())
        members = torch.stack([item["objects"] for item in prediction]) * stats["objects_std"][None, None] + stats["objects_mean"][None, None]
        disagreement += float(members.std(0).square().sum())
        rows += len(batch["target"]); count += mean.numel()
    return {
        "loss": total / max(rows, 1),
        "object_rmse_mm": 1e3 * math.sqrt(square / max(count, 1)),
        "ensemble_object_std_mm": 1e3 * math.sqrt(disagreement / max(count, 1)),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=False)
    validation = torch.load(args.cache_dir / "validation.pt", map_location="cpu", weights_only=False)
    train_pose = visual_pose(train, args.vla_checkpoint, device, args.output_dir / "train_visual_pose.pt")
    validation_pose = visual_pose(validation, args.vla_checkpoint, device, args.output_dir / "validation_visual_pose.pt")
    if device.type == "cuda": torch.cuda.empty_cache()
    stats = {key: item.to(device) for key, item in world_stats(train, train_pose).items()}
    train_set = TransitionDataset(train, train_pose, True)
    validation_set = TransitionDataset(validation, validation_pose, False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=WeightedRandomSampler(train_set.weights, len(train_set), replacement=True), num_workers=0, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = TemporalChunkWorldEnsemble(args.members, int(train["contact_sites"].shape[-1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    best = math.inf
    history = []
    for epoch in range(args.epochs):
        model.train(); total = 0.0; rows = 0
        for raw in train_loader:
            batch = {key: item.to(device, non_blocking=True) for key, item in raw.items()}
            prediction = outputs(model, batch, stats)
            loss = torch.stack([member_loss(item, batch, stats) for item in prediction]).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss) * len(batch["target"]); rows += len(batch["target"])
        report = evaluate(model, validation_loader, stats, device)
        report.update(epoch=epoch + 1, train_loss=total / max(rows, 1)); history.append(report)
        print("ACTION_TOKEN_WORLD_EPOCH=" + json.dumps(report), flush=True)
        if report["loss"] < best:
            best = report["loss"]
            save_checkpoint(args.output_dir / "best.pt", model, {key: item.detach().cpu() for key, item in stats.items()})
    summary = {
        "experiment": "G1LANG-action-token-ensemble-world-model",
        "cache_dir": str(args.cache_dir),
        "vla_checkpoint": str(args.vla_checkpoint),
        "members": args.members,
        "best_validation_score": best,
        "state_contract": "action-token VLA visual pose + state31 + qvel31 + finger sites + tactile",
        "contact_reward_enabled": False,
        "privileged_positions_at_inference": False,
        "history": history,
        "final": history[-1],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
