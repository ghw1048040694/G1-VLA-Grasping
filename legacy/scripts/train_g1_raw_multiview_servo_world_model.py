#!/usr/bin/env python3
"""Train the raw visual-state world model from strict dynamic transitions."""

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

from g1_raw_multiview_servo_vla import load_checkpoint as load_vla
from g1_raw_multiview_servo_world_model import RawServoWorldModel, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--vla-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class TransitionDataset(Dataset):
    def __init__(self, cache: dict, visual_pose: torch.Tensor, weighted: bool):
        self.cache = cache; self.visual_pose = visual_pose.float(); self.images = None
        valid = cache["valid_frame"].bool().numpy(); self.rows = np.argwhere(valid)
        self.weights = cache["sample_weight"].numpy()[valid].astype(np.float64) if weighted else np.ones(len(self.rows), dtype=np.float64)

    def _images(self):
        if self.images is None: self.images = np.load(self.cache["image_path"], mmap_mode="r")
        return self.images

    def __len__(self): return len(self.rows)

    def __getitem__(self, index: int):
        episode, frame = self.rows[index]
        value = {key: self.cache[key][episode, frame].float() for key in ("state", "velocity", "action_delta", "next_state", "next_velocity", "contact_sites", "next_contact_sites", "contact", "next_contact", "next_objects")}
        value["objects"] = self.visual_pose[episode, frame]
        # The deployed planner rolls its own visual-state estimate forward.
        # Train that same estimated-state transition rather than mixing an
        # estimated input with privileged next-step simulator coordinates.
        value["next_objects"] = self.visual_pose[episode, frame + 1]
        value["target"] = self.cache["target"][episode].long()
        return value


@torch.no_grad()
def visual_pose(cache: dict, checkpoint: Path, device: torch.device, output: Path) -> torch.Tensor:
    if output.is_file(): return torch.load(output, map_location="cpu", weights_only=False)
    model, vocab, stats = load_vla(checkpoint, str(device))
    if vocab != cache["vocab"]: raise ValueError("VLA vocabulary disagrees with transition cache")
    images = np.load(cache["image_path"], mmap_mode="r")
    episodes, frames = cache["state"].shape[:2]
    if images.shape[1] != frames + 1:
        raise ValueError(
            "World-model training requires a cache with successor RGB frames; "
            "rebuild the raw multiview cache."
        )
    # Cache labels contain state/action for `frames` control instants plus a
    # one-step successor label.  Rendered videos have the same extra frame.
    # Allocate it so a valid frame can target its own next visual estimate.
    result = torch.empty((episodes, frames + 1, 9), dtype=torch.float32)
    sequence = {
        name: torch.cat((cache[name], cache[f"next_{name}"][:, -1:]), dim=1)
        for name in ("state", "velocity", "contact_sites", "contact")
    }
    for episode in range(episodes):
        for start in range(0, frames + 1, 64):
            stop = min(start + 64, frames + 1)
            pixel = torch.from_numpy(np.asarray(images[episode, start:stop]).copy()).permute(0, 1, 4, 2, 3).float().div_(255.0).to(device)
            state = (sequence["state"][episode, start:stop].float().to(device) - stats["state_mean"]) / stats["state_std"]
            velocity = (sequence["velocity"][episode, start:stop].float().to(device) - stats["velocity_mean"]) / stats["velocity_std"]
            sites = (sequence["contact_sites"][episode, start:stop].float().to(device) - stats["contact_sites_mean"]) / stats["contact_sites_std"]
            task_index = int(cache["task_index"][episode]); task = cache["tasks"][task_index]
            from g1_direct_joint_candidate_vla import encode_tasks
            ids, mask = encode_tasks([task] * (stop - start), vocab)
            value = model(pixel, state, velocity, sites, sequence["contact"][episode, start:stop].float().to(device), ids.to(device), mask.to(device))
            result[episode, start:stop] = (value["object_mean"] * stats["objects_std"] + stats["objects_mean"]).cpu()
        print(json.dumps({"visual_pose_episode": episode + 1, "total": episodes}), flush=True)
    torch.save(result, output)
    return result


def world_stats(cache: dict, pose: torch.Tensor) -> dict[str, torch.Tensor]:
    valid = cache["valid_frame"].bool()
    values = {
        "state": cache["state"][valid].float(), "velocity": cache["velocity"][valid].float(),
        "objects": pose[:, :-1][valid].float(), "contact_sites": cache["contact_sites"][valid].float(),
        "action": cache["action_delta"][valid].float(),
    }
    result = {}
    for name, value in values.items():
        result[f"{name}_mean"] = value.mean(0); result[f"{name}_std"] = value.std(0).clamp_min(1e-3)
    return result


def z(value, stats, name): return (value - stats[f"{name}_mean"]) / stats[f"{name}_std"]


def model_loss(model, batch, stats):
    target = F.one_hot(batch["target"], 3).float()
    output = model(z(batch["state"], stats, "state"), z(batch["velocity"], stats, "velocity"), z(batch["objects"], stats, "objects"), z(batch["contact_sites"], stats, "contact_sites"), batch["contact"], z(batch["action_delta"], stats, "action"), target)
    losses = {
        "state": F.mse_loss(output["state"], z(batch["next_state"], stats, "state")),
        "velocity": F.mse_loss(output["velocity"], z(batch["next_velocity"], stats, "velocity")),
        "objects": F.mse_loss(output["objects"], z(batch["next_objects"], stats, "objects")),
        "sites": F.mse_loss(output["contact_sites"], z(batch["next_contact_sites"], stats, "contact_sites")),
        "contact": F.binary_cross_entropy_with_logits(output["contact_logit"], batch["next_contact"]),
    }
    return losses["state"] + 0.20 * losses["velocity"] + 0.65 * losses["objects"] + 0.35 * losses["sites"] + 0.15 * losses["contact"], losses


@torch.no_grad()
def evaluate(model, loader, stats, device):
    model.eval(); total = {name: 0.0 for name in ("loss", "state", "velocity", "objects", "sites", "contact")}; rows = 0; object_sq = 0.0; object_count = 0; contact_ok = 0
    for raw in loader:
        batch = {key: item.to(device) for key, item in raw.items()}; loss, values = model_loss(model, batch, stats); count = len(batch["target"]); rows += count
        total["loss"] += float(loss) * count
        for key, value in values.items(): total[key] += float(value) * count
        target = F.one_hot(batch["target"], 3).float(); output = model(z(batch["state"], stats, "state"), z(batch["velocity"], stats, "velocity"), z(batch["objects"], stats, "objects"), z(batch["contact_sites"], stats, "contact_sites"), batch["contact"], z(batch["action_delta"], stats, "action"), target)
        objects = output["objects"] * stats["objects_std"] + stats["objects_mean"]
        object_sq += float((objects - batch["next_objects"]).square().sum()); object_count += objects.numel()
        contact_ok += int(((output["contact_logit"].sigmoid() >= .5) == (batch["next_contact"] >= .5)).sum())
    return {**{key: value / max(rows, 1) for key, value in total.items()}, "object_rmse_mm": 1e3 * math.sqrt(object_sq / max(object_count, 1)), "contact_accuracy": contact_ok / max(rows, 1)}


def main() -> None:
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_cache = torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=False); val_cache = torch.load(args.cache_dir / "validation.pt", map_location="cpu", weights_only=False)
    train_pose = visual_pose(train_cache, args.vla_checkpoint, device, args.output_dir / "train_visual_pose.pt"); val_pose = visual_pose(val_cache, args.vla_checkpoint, device, args.output_dir / "validation_visual_pose.pt")
    if device.type == "cuda": torch.cuda.empty_cache()
    stats = {key: item.to(device) for key, item in world_stats(train_cache, train_pose).items()}
    train_set = TransitionDataset(train_cache, train_pose, True); val_set = TransitionDataset(val_cache, val_pose, False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=WeightedRandomSampler(train_set.weights, len(train_set), replacement=True), num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = RawServoWorldModel(int(train_cache["contact_sites"].shape[-1])).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    best = math.inf; history=[]
    for epoch in range(args.epochs):
        model.train(); total=0.0; rows=0
        for raw in train_loader:
            batch={key:item.to(device, non_blocking=True) for key,item in raw.items()}; loss,_=model_loss(model,batch,stats); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); total += float(loss)*len(batch["target"]); rows += len(batch["target"])
        report=evaluate(model,val_loader,stats,device); report.update(epoch=epoch+1,train_loss=total/max(rows,1)); history.append(report); print("RAW_VISUAL_SERVO_WORLD_EPOCH="+json.dumps(report),flush=True)
        if report["loss"] < best: best=report["loss"]; save_checkpoint(args.output_dir/"best.pt",model,{key:item.detach().cpu() for key,item in stats.items()})
    summary={"experiment":"G1LANG-raw-multiview-visual-servo-world-model","cache_dir":str(args.cache_dir),"vla_checkpoint":str(args.vla_checkpoint),"best_validation_score":best,"state_contract":"VLA-estimated object pose + state31 + qvel31 + measured finger sites + tactile; no simulator object truth","history":history,"final":history[-1],"privileged_positions_at_inference":False}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__ == "__main__": main()
