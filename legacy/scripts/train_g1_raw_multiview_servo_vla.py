#!/usr/bin/env python3
"""Train the raw-RGB visual-servo VLA from a disk-backed strict cache."""

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
from g1_raw_multiview_servo_vla import RawMultiviewServoVLA, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class CacheDataset(Dataset):
    def __init__(self, cache: dict, weighted: bool):
        self.cache = cache
        self.images = None
        valid = cache["valid_frame"].bool().numpy()
        self.rows = np.argwhere(valid)
        self.weights = cache["sample_weight"].numpy()[valid].astype(np.float64) if weighted else np.ones(len(self.rows), dtype=np.float64)
        # Instructions are recorded once per episode while samples are drawn
        # per frame.  Expand them explicitly instead of reshaping a shorter
        # episode-level tensor into a frame-level tensor.
        ids, mask = encode_tasks(
            [cache["tasks"][int(index)] for index in cache["task_index"].tolist()],
            cache["vocab"],
        )
        episodes, frames = cache["state"].shape[:2]
        self.ids = ids[:, None, :].expand(episodes, frames, -1)
        self.mask = mask[:, None, :].expand(episodes, frames, -1)

    def _images(self):
        if self.images is None:
            self.images = np.load(self.cache["image_path"], mmap_mode="r")
        return self.images

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, frame = self.rows[index]
        image = torch.from_numpy(np.asarray(self._images()[episode, frame]).copy()).permute(0, 3, 1, 2).float().div_(255.0)
        return {
            "images": image,
            "state": self.cache["state"][episode, frame].float(),
            "velocity": self.cache["velocity"][episode, frame].float(),
            "contact_sites": self.cache["contact_sites"][episode, frame].float(),
            "contact": self.cache["contact"][episode, frame].float(),
            "action": self.cache["action_delta"][episode, frame].float(),
            "objects": self.cache["objects"][episode, frame].float(),
            "target": self.cache["target"][episode].long(),
            "arm": self.cache["active_arm"][episode, frame].long(),
            "ids": self.ids[episode, frame],
            "mask": self.mask[episode, frame],
        }


def stats(cache: dict) -> dict[str, torch.Tensor]:
    valid = cache["valid_frame"].bool()
    def flatten(name: str) -> torch.Tensor:
        return cache[name][valid].float()
    result = {}
    for name in ("state", "velocity", "action_delta", "objects", "contact_sites"):
        values = flatten(name)
        result[f"{name}_mean"] = values.mean(0)
        result[f"{name}_std"] = values.std(0).clamp_min(1e-3)
    return result


def normalized(batch: dict[str, torch.Tensor], value: dict[str, torch.Tensor]) -> tuple:
    return (
        (batch["state"] - value["state_mean"]) / value["state_std"],
        (batch["velocity"] - value["velocity_mean"]) / value["velocity_std"],
        (batch["contact_sites"] - value["contact_sites_mean"]) / value["contact_sites_std"],
        (batch["action"] - value["action_delta_mean"]) / value["action_delta_std"],
        (batch["objects"] - value["objects_mean"]) / value["objects_std"],
    )


def loss_and_metrics(model, batch, value):
    state, velocity, sites, action, objects = normalized(batch, value)
    output = model(batch["images"], state, velocity, sites, batch["contact"], batch["ids"], batch["mask"])
    action_scale = output["action_log_std"].exp()
    action_loss = (0.5 * ((output["action_mean"] - action) / action_scale).square() + output["action_log_std"]).mean()
    pose_scale = output["object_log_std"].exp()
    pose_loss = (0.5 * ((output["object_mean"] - objects) / pose_scale).square() + output["object_log_std"]).mean()
    target_loss = F.cross_entropy(output["target_logits"], batch["target"])
    arm_loss = F.cross_entropy(output["active_arm_logits"], batch["arm"])
    loss = action_loss + 0.45 * pose_loss + 0.20 * target_loss + 0.08 * arm_loss
    predicted_action = output["action_mean"] * value["action_delta_std"] + value["action_delta_mean"]
    predicted_objects = output["object_mean"] * value["objects_std"] + value["objects_mean"]
    return loss, output, predicted_action, predicted_objects, {"action": action_loss, "pose": pose_loss, "target": target_loss, "arm": arm_loss}


@torch.no_grad()
def evaluate(model, loader, value, device):
    model.eval(); total = {name: 0.0 for name in ("loss", "action", "pose", "target", "arm")}; rows = 0; action_sq = pose_sq = target_pose_sq = 0.0; action_count = pose_count = target_pose_count = 0; target_ok = arm_ok = 0
    for raw in loader:
        batch = {key: item.to(device) for key, item in raw.items()}
        loss, output, action, pose, parts = loss_and_metrics(model, batch, value)
        count = len(batch["target"]); rows += count
        total["loss"] += float(loss) * count
        for key, item in parts.items(): total[key] += float(item) * count
        action_sq += float((action - batch["action"]).square().sum()); action_count += action.numel()
        pose_sq += float((pose - batch["objects"]).square().sum()); pose_count += pose.numel()
        target_row = torch.arange(count, device=device)
        target_pose = pose.reshape(count, 3, 3)[target_row, batch["target"]]
        target_truth = batch["objects"].reshape(count, 3, 3)[target_row, batch["target"]]
        target_pose_sq += float((target_pose - target_truth).square().sum()); target_pose_count += target_pose.numel()
        target_ok += int((output["target_logits"].argmax(-1) == batch["target"]).sum())
        arm_ok += int((output["active_arm_logits"].argmax(-1) == batch["arm"]).sum())
    return {
        **{key: item / max(rows, 1) for key, item in total.items()},
        "joint_delta_rmse_rad": math.sqrt(action_sq / max(action_count, 1)),
        "object_position_rmse_mm": 1e3 * math.sqrt(pose_sq / max(pose_count, 1)),
        "target_object_position_rmse_mm": 1e3 * math.sqrt(target_pose_sq / max(target_pose_count, 1)),
        "target_accuracy": target_ok / max(rows, 1),
        "active_arm_accuracy": arm_ok / max(rows, 1),
    }


def main() -> None:
    args = parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    train_cache = torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=False)
    val_cache = torch.load(args.cache_dir / "validation.pt", map_location="cpu", weights_only=False)
    value = {key: item.to(device) for key, item in stats(train_cache).items()}
    train_set = CacheDataset(train_cache, weighted=True); val_set = CacheDataset(val_cache, weighted=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=WeightedRandomSampler(train_set.weights, len(train_set), replacement=True), num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = RawMultiviewServoVLA(len(train_cache["vocab"]), int(train_cache["contact_sites"].shape[-1])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True); best = math.inf; best_epoch = None; history = []
    for epoch in range(args.epochs):
        model.train(); total = 0.0; rows = 0
        for raw in train_loader:
            batch = {key: item.to(device, non_blocking=True) for key, item in raw.items()}
            loss, _, _, _, _ = loss_and_metrics(model, batch, value)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss) * len(batch["target"]); rows += len(batch["target"])
        report = evaluate(model, val_loader, value, device); report.update(epoch=epoch + 1, train_loss=total / max(rows, 1)); history.append(report)
        print("RAW_VISUAL_SERVO_EPOCH=" + json.dumps(report), flush=True)
        # CEM uses the action mean and the target object's estimated pose,
        # not the calibrated NLL.  Select the checkpoint on those deployment
        # quantities and retain the final state for an explicit audit.
        score = report["joint_delta_rmse_rad"] / 0.002 + report["target_object_position_rmse_mm"] / 20.0
        save_checkpoint(args.output_dir / "last.pt", model, train_cache["vocab"], {key: item.detach().cpu() for key, item in value.items()})
        if score < best:
            best = score; best_epoch = epoch + 1; save_checkpoint(args.output_dir / "best.pt", model, train_cache["vocab"], {key: item.detach().cpu() for key, item in value.items()})
    summary = {
        "experiment": "G1LANG-raw-multiview-visual-servo-vla",
        "cache_dir": str(args.cache_dir), "best_validation_score": best, "best_epoch": best_epoch,
        "input_contract": "raw synchronized RGB x3 + language + state31 + qvel31 + measured finger sites + tactile",
        "action_representation": "relative_joint_delta", "privileged_positions_at_inference": False,
        "history": history, "final": history[-1],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
