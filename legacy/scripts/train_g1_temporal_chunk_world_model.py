#!/usr/bin/env python3
"""Train a three-member transition ensemble in temporal-VLA visual state space."""

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

from g1_direct_joint_candidate_vla import encode_tasks
from g1_temporal_chunk_vla import load_checkpoint as load_vla
from g1_temporal_chunk_world_model import TemporalChunkWorldEnsemble, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True); parser.add_argument("--vla-checkpoint", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--members", type=int, default=3); parser.add_argument("--epochs", type=int, default=36); parser.add_argument("--batch-size", type=int, default=96); parser.add_argument("--learning-rate", type=float, default=5e-4); parser.add_argument("--seed", type=int, default=20260730); parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class TransitionDataset(Dataset):
    def __init__(self, cache: dict, visual_pose: torch.Tensor, weighted: bool):
        self.cache = cache; self.visual_pose = visual_pose.float(); valid = cache["valid_frame"].bool().numpy(); self.rows = np.argwhere(valid)
        self.weights = cache["sample_weight"].numpy()[valid].astype(np.float64) if weighted else np.ones(len(self.rows), dtype=np.float64)

    def __len__(self): return len(self.rows)

    def __getitem__(self, index: int):
        episode, frame = self.rows[index]
        result = {key: self.cache[key][episode, frame].float() for key in ("state", "velocity", "action_delta", "next_state", "next_velocity", "contact_sites", "next_contact_sites", "contact")}
        result["objects"] = self.visual_pose[episode, frame]; result["next_objects"] = self.visual_pose[episode, frame + 1]; result["target"] = self.cache["target"][episode].long()
        return result


@torch.no_grad()
def temporal_visual_pose(cache: dict, checkpoint: Path, device: torch.device, output: Path) -> torch.Tensor:
    if output.is_file(): return torch.load(output, map_location="cpu", weights_only=False)
    model, vocab, stats = load_vla(checkpoint, str(device))
    if vocab != cache["vocab"]: raise ValueError("VLA vocabulary disagrees with cache")
    images = np.load(cache["image_path"], mmap_mode="r"); episodes, frames = cache["state"].shape[:2]; total = frames + 1
    if images.shape[1] != total: raise ValueError("Temporal world model requires successor RGB cache")
    sequence = {name: torch.cat((cache[name], cache[f"next_{name}"][:, -1:]), dim=1) for name in ("state", "velocity", "contact_sites", "contact")}
    result = torch.empty((episodes, total, 9), dtype=torch.float32); history_frames = model.history_frames
    for episode in range(episodes):
        all_indices = np.maximum(np.arange(total)[:, None] - history_frames + 1 + np.arange(history_frames)[None], 0)
        task = cache["tasks"][int(cache["task_index"][episode])]
        for start in range(0, total, 32):
            stop = min(start + 32, total); index = all_indices[start:stop]
            pixels = torch.from_numpy(np.asarray(images[episode, index]).copy()).permute(0, 1, 2, 5, 3, 4).float().div_(255.0).to(device)
            state = (sequence["state"][episode, index].float().to(device) - stats["state_mean"]) / stats["state_std"]
            velocity = (sequence["velocity"][episode, index].float().to(device) - stats["velocity_mean"]) / stats["velocity_std"]
            sites = (sequence["contact_sites"][episode, index].float().to(device) - stats["contact_sites_mean"]) / stats["contact_sites_std"]
            ids, mask = encode_tasks([task] * (stop - start), vocab)
            value = model(pixels, state, velocity, sites, sequence["contact"][episode, index].float().to(device), ids.to(device), mask.to(device))
            result[episode, start:stop] = (value["object_mean"] * stats["objects_std"] + stats["objects_mean"]).cpu()
        print(json.dumps({"temporal_visual_pose_episode": episode + 1, "total": episodes}), flush=True)
    torch.save(result, output); return result


def world_stats(cache: dict, pose: torch.Tensor) -> dict[str, torch.Tensor]:
    valid = cache["valid_frame"].bool(); values = {"state": cache["state"][valid].float(), "velocity": cache["velocity"][valid].float(), "objects": pose[:, :-1][valid].float(), "contact_sites": cache["contact_sites"][valid].float(), "action": cache["action_delta"][valid].float()}
    return {f"{name}_{part}": getattr(value, part)(0).clamp_min(1e-3) if part == "std" else getattr(value, part)(0) for name, value in values.items() for part in ("mean", "std")}


def z(value, stats, name): return (value - stats[f"{name}_mean"]) / stats[f"{name}_std"]


def member_loss(output, batch, stats):
    return (F.mse_loss(output["state"], z(batch["next_state"], stats, "state")) + 0.20 * F.mse_loss(output["velocity"], z(batch["next_velocity"], stats, "velocity")) + 0.75 * F.mse_loss(output["objects"], z(batch["next_objects"], stats, "objects")) + 0.45 * F.mse_loss(output["contact_sites"], z(batch["next_contact_sites"], stats, "contact_sites")))


def ensemble_outputs(model, batch, stats):
    target = F.one_hot(batch["target"], 3).float()
    return model(z(batch["state"], stats, "state"), z(batch["velocity"], stats, "velocity"), z(batch["objects"], stats, "objects"), z(batch["contact_sites"], stats, "contact_sites"), batch["contact"], z(batch["action_delta"], stats, "action"), target)


@torch.no_grad()
def evaluate(model, loader, stats, device):
    model.eval(); total = 0.0; rows = 0; object_sq = 0.0; object_count = 0; disagreement = 0.0; disagreement_count = 0
    for raw in loader:
        batch = {key: item.to(device) for key, item in raw.items()}; outputs = ensemble_outputs(model, batch, stats); loss = torch.stack([member_loss(output, batch, stats) for output in outputs]).mean(); count = len(batch["target"]); total += float(loss) * count; rows += count
        objects = torch.stack([output["objects"] for output in outputs]).mean(0) * stats["objects_std"] + stats["objects_mean"]
        object_sq += float((objects - batch["next_objects"]).square().sum()); object_count += objects.numel()
        raw_members = torch.stack([output["objects"] for output in outputs]) * stats["objects_std"][None, None] + stats["objects_mean"][None, None]
        disagreement += float(raw_members.std(0).square().sum()); disagreement_count += raw_members[0].numel()
    return {"loss": total / max(rows, 1), "object_rmse_mm": 1e3 * math.sqrt(object_sq / max(object_count, 1)), "ensemble_object_std_mm": 1e3 * math.sqrt(disagreement / max(disagreement_count, 1))}


def main() -> None:
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"); args.output_dir.mkdir(parents=True, exist_ok=True)
    train_cache = torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=False); val_cache = torch.load(args.cache_dir / "validation.pt", map_location="cpu", weights_only=False)
    train_pose = temporal_visual_pose(train_cache, args.vla_checkpoint, device, args.output_dir / "train_visual_pose.pt"); val_pose = temporal_visual_pose(val_cache, args.vla_checkpoint, device, args.output_dir / "validation_visual_pose.pt")
    if device.type == "cuda": torch.cuda.empty_cache()
    stats = {key: item.to(device) for key, item in world_stats(train_cache, train_pose).items()}
    train_set = TransitionDataset(train_cache, train_pose, True); val_set = TransitionDataset(val_cache, val_pose, False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=WeightedRandomSampler(train_set.weights, len(train_set), replacement=True), num_workers=0, pin_memory=device.type == "cuda"); val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = TemporalChunkWorldEnsemble(args.members, int(train_cache["contact_sites"].shape[-1])).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4); best = math.inf; history = []
    for epoch in range(args.epochs):
        model.train(); total = 0.0; rows = 0
        for raw in train_loader:
            batch = {key: item.to(device, non_blocking=True) for key, item in raw.items()}; outputs = ensemble_outputs(model, batch, stats); loss = torch.stack([member_loss(output, batch, stats) for output in outputs]).mean(); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total += float(loss) * len(batch["target"]); rows += len(batch["target"])
        report = evaluate(model, val_loader, stats, device); report.update(epoch=epoch + 1, train_loss=total / max(rows, 1)); history.append(report); print("TEMPORAL_CHUNK_WORLD_EPOCH=" + json.dumps(report), flush=True)
        if report["loss"] < best: best = report["loss"]; save_checkpoint(args.output_dir / "best.pt", model, {key: item.detach().cpu() for key, item in stats.items()})
    summary = {"experiment": "G1LANG-temporal-chunk-world-model-ensemble", "cache_dir": str(args.cache_dir), "vla_checkpoint": str(args.vla_checkpoint), "members": args.members, "best_validation_score": best, "state_contract": "temporal VLA estimated visual pose + state31 + qvel31 + measured finger sites + tactile; no simulator object truth", "contact_reward_enabled": False, "privileged_positions_at_inference": False, "history": history, "final": history[-1]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__": main()
