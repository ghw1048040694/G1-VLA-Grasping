#!/usr/bin/env python3
"""Train a proposal VLA with discrete, multimodal strict-expert action tokens."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from scipy.optimize import linear_sum_assignment

from g1_action_token_vla import ActionTokenVLA, save_checkpoint
from g1_direct_joint_candidate_vla import encode_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--token-count", type=int, default=72)
    parser.add_argument("--history-frames", type=int, default=4)
    parser.add_argument("--action-chunk", type=int, default=8)
    parser.add_argument("--future-offsets", default="0,12,24,36,48")
    parser.add_argument("--epochs", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--freeze-visual-encoder", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=20260743)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples-per-epoch", type=int, default=0, help="0 uses all train rows; positive values are intended for contract smoke tests.")
    parser.add_argument("--max-validation-rows", type=int, default=0, help="0 validates all rows; positive values are intended for contract smoke tests.")
    parser.add_argument("--source-root-boost-pattern", default="", help="Substring identifying a quality-audited rare source family to upweight without duplicating episodes.")
    parser.add_argument("--source-root-boost", type=float, default=1.0)
    return parser.parse_args()


def statistics(cache: dict) -> dict[str, torch.Tensor]:
    valid = cache["valid_frame"].bool()
    result = {}
    for name in ("state", "velocity", "action_delta", "objects", "contact_sites"):
        values = cache[name][valid].float()
        result[f"{name}_mean"] = values.mean(0)
        result[f"{name}_std"] = values.std(0).clamp_min(1e-3)
    return result


def valid_rows(cache: dict, action_chunk: int) -> list[tuple[int, int]]:
    rows = []
    for episode, valid in enumerate(cache["valid_frame"].bool()):
        length = int(valid.sum())
        candidates = list(range(max(0, length - action_chunk + 1)))
        if candidates:
            energy = torch.stack([
                cache["action_delta"][episode, frame : frame + action_chunk].float().square().mean().sqrt()
                for frame in candidates
            ])
            moving = torch.where(energy >= 0.0015)[0]
            if len(moving):
                # Drop leading/trailing scripted waits. The deployment policy
                # must start moving immediately from the canonical home view.
                candidates = candidates[int(moving[0]) : int(moving[-1]) + 1]
        rows.extend((episode, frame) for frame in candidates)
    if not rows:
        raise ValueError("No action chunks in cache")
    return rows


def action_chunks(cache: dict, rows: list[tuple[int, int]], action_chunk: int) -> torch.Tensor:
    return torch.stack(
        [cache["action_delta"][episode, frame : frame + action_chunk].float() for episode, frame in rows]
    )


def kmeans(vectors: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """Small deterministic k-means; weighted sampling preserves rare approach/lift actions."""
    generator = torch.Generator().manual_seed(seed)
    energy = vectors.square().mean(1).sqrt()
    probability = (energy / energy.mean().clamp_min(1e-5)).clamp(.15, 12.0)
    probability = probability / probability.sum()
    sample_count = min(len(vectors), 18000)
    sample = vectors[torch.multinomial(probability, sample_count, replacement=False, generator=generator)]
    centers = sample[torch.randperm(sample_count, generator=generator)[:count]].clone()
    for _ in range(36):
        assignment = torch.cdist(sample, centers).argmin(1)
        next_centers = []
        for index in range(count):
            selected = sample[assignment == index]
            next_centers.append(
                selected.mean(0) if len(selected) else sample[torch.randint(sample_count, (1,), generator=generator)[0]]
            )
        centers = torch.stack(next_centers)
    return centers


class TokenDataset(Dataset):
    def __init__(
        self,
        cache: dict,
        action_chunk: int,
        history_frames: int,
        action_std: torch.Tensor,
        codebook_normalized: torch.Tensor,
        future_offsets: tuple[int, ...],
        weighted: bool,
        source_root_boost_pattern: str = "",
        source_root_boost: float = 1.0,
    ) -> None:
        self.cache = cache
        self.action_chunk = action_chunk
        self.history_frames = history_frames
        self.future_offsets = future_offsets
        self.images = None
        self.rows = valid_rows(cache, action_chunk)
        physical_chunks = action_chunks(cache, self.rows, action_chunk).flatten(1)
        chunks = physical_chunks / action_std.repeat(action_chunk)
        self.codes = torch.cdist(chunks, codebook_normalized).argmin(1)
        self.row_to_code = {row: int(code) for row, code in zip(self.rows, self.codes)}
        self.episode_frame_bounds = {}
        for episode, frame in self.rows:
            if episode not in self.episode_frame_bounds:
                self.episode_frame_bounds[episode] = [frame, frame]
            self.episode_frame_bounds[episode][1] = frame
        self.lengths = [int(row.sum()) for row in cache["valid_frame"].bool()]
        self.ids, self.mask = encode_tasks(
            [cache["tasks"][int(index)] for index in cache["task_index"].tolist()], cache["vocab"]
        )
        normalized_energy = chunks.square().mean(1).sqrt().numpy()
        self.energy = physical_chunks.square().mean(1).sqrt().numpy()
        if weighted:
            source = cache.get("source_root", [str(index) for index in range(len(self.lengths))])
            duplicate_count = {name: source.count(name) for name in set(source)}
            unique_target_count = {}
            for episode, name in enumerate(source):
                unique_target_count.setdefault(int(cache["target"][episode]), set()).add(name)
            phase = []
            for episode, frame in self.rows:
                if "task_phase" in cache:
                    item_phase = int(cache["task_phase"][episode, frame])
                    phase.append(
                        item_phase if item_phase >= 0
                        else min(7, int(8 * frame / max(self.lengths[episode], 1)))
                    )
                else:
                    phase.append(min(7, int(8 * frame / max(self.lengths[episode], 1))))
            group_count = {}
            for row, item_phase in zip(self.rows, phase):
                key = (row[0], item_phase)
                group_count[key] = group_count.get(key, 0) + 1
            median_energy = max(float(np.median(normalized_energy)), 1e-5)
            self.weights = np.asarray([
                (0.25 + min(float(item_energy) / median_energy, 4.0))
                / group_count[(episode, item_phase)]
                / duplicate_count[source[episode]]
                / len(unique_target_count[int(cache["target"][episode])])
                for (episode, _), item_phase, item_energy in zip(self.rows, phase, normalized_energy)
            ], dtype=np.float64)
            if source_root_boost_pattern:
                if source_root_boost < 1.0:
                    raise ValueError("source_root_boost must be >= 1")
                for index, (episode, _) in enumerate(self.rows):
                    if source_root_boost_pattern in source[episode]:
                        self.weights[index] *= source_root_boost
        else:
            self.weights = np.ones(len(self.rows), dtype=np.float64)

    def _images(self):
        if self.images is None:
            self.images = np.load(self.cache["image_path"], mmap_mode="r")
        return self.images

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, frame = self.rows[index]
        history = np.maximum(np.arange(frame - self.history_frames + 1, frame + 1), 0)
        image = torch.from_numpy(np.asarray(self._images()[episode, history]).copy())
        image = image.permute(0, 1, 4, 2, 3).float().div_(255.0)
        earliest, latest = self.episode_frame_bounds[episode]
        future_frames = [min(max(frame + offset, earliest), latest) for offset in self.future_offsets]
        future = [self.row_to_code[(episode, future_frame)] for future_frame in future_frames]
        future_actions = torch.stack(
            [self.cache["action_delta"][episode, future_frame : future_frame + self.action_chunk].float() for future_frame in future_frames]
        )
        return {
            "images": image,
            "state": self.cache["state"][episode, history].float(),
            "velocity": self.cache["velocity"][episode, history].float(),
            "sites": self.cache["contact_sites"][episode, history].float(),
            "tactile": self.cache["contact"][episode, history].float(),
            "objects": self.cache["objects"][episode, frame].float(),
            "future_codes": torch.tensor(future, dtype=torch.long),
            "future_actions": future_actions,
            "target": self.cache["target"][episode].long(),
            "arm": self.cache["active_arm"][episode, frame].long(),
            "immediate_energy": torch.tensor(self.energy[index], dtype=torch.float32),
            "prefix": torch.tensor(
                frame <= earliest + max(1, (latest - earliest) // 4), dtype=torch.bool
            ),
            "ids": self.ids[episode],
            "mask": self.mask[episode],
        }


def model_input(batch: dict[str, torch.Tensor], value: dict[str, torch.Tensor]):
    return (
        batch["images"],
        (batch["state"] - value["state_mean"]) / value["state_std"],
        (batch["velocity"] - value["velocity_mean"]) / value["velocity_std"],
        (batch["sites"] - value["contact_sites_mean"]) / value["contact_sites_std"],
        batch["tactile"],
        batch["ids"],
        batch["mask"],
    )


def losses(model, batch, value, codebook):
    output = model(*model_input(batch, value))
    token_target = torch.zeros_like(output["action_token_logits"])
    token_target.scatter_(1, batch["future_codes"], 1.0)
    token_loss = F.binary_cross_entropy_with_logits(
        output["action_token_logits"], token_target, pos_weight=torch.full((token_target.shape[1],), 10.0, device=token_target.device)
    )
    immediate_token_loss = F.cross_entropy(
        output["action_token_logits"], batch["future_codes"][:, 0]
    )
    rows = torch.arange(len(batch["target"]), device=token_target.device)[:, None]
    candidate_action = codebook[batch["future_codes"]] + output["candidate_action_residual"][rows, batch["future_codes"]]
    action_loss = F.smooth_l1_loss(candidate_action, batch["future_actions"], beta=.004)
    pose = (batch["objects"] - value["objects_mean"]) / value["objects_std"]
    pose_loss = F.mse_loss(output["object_mean"], pose)
    target_loss = F.cross_entropy(output["target_logits"], batch["target"])
    arm_loss = F.cross_entropy(output["active_arm_logits"], batch["arm"])
    loss = .45 * token_loss + immediate_token_loss + .70 * action_loss + .45 * pose_loss + .12 * target_loss + .06 * arm_loss
    return loss, output, {"token": token_loss, "immediate_token": immediate_token_loss, "action": action_loss, "pose": pose_loss, "target": target_loss, "arm": arm_loss}, candidate_action


@torch.no_grad()
def evaluate(model, loader, value, codebook, device):
    model.eval()
    total = 0.0
    pose_square = 0.0
    pose_count = 0
    action_square = 0.0
    action_count = 0
    future_covered = 0
    immediate_covered = 0
    immediate_top1 = nontrivial_rows = nontrivial_recalled = 0
    target_top1_ok = torch.zeros(3, dtype=torch.long, device=device)
    target_rows = torch.zeros(3, dtype=torch.long, device=device)
    prefix_top1_ok = torch.zeros(3, dtype=torch.long, device=device)
    prefix_rows = torch.zeros(3, dtype=torch.long, device=device)
    top1_action_square = 0.0
    top1_action_count = 0
    target_ok = arm_ok = rows = 0
    for raw in loader:
        batch = {key: item.to(device) for key, item in raw.items()}
        loss, output, _, candidate_action = losses(model, batch, value, codebook)
        count = len(batch["target"])
        total += float(loss) * count
        pose = output["object_mean"] * value["objects_std"] + value["objects_mean"]
        pose_square += float((pose - batch["objects"]).square().sum())
        pose_count += pose.numel()
        action_square += float((candidate_action - batch["future_actions"]).square().sum())
        action_count += candidate_action.numel()
        top = output["action_token_logits"].topk(min(12, output["action_token_logits"].shape[1]), dim=1).indices
        top1 = top[:, 0]
        immediate_top1 += int((top1 == batch["future_codes"][:, 0]).sum())
        top1_ok = top1 == batch["future_codes"][:, 0]
        for target_index in range(3):
            target_mask = batch["target"] == target_index
            target_rows[target_index] += target_mask.sum()
            target_top1_ok[target_index] += (top1_ok & target_mask).sum()
            prefix_mask = target_mask & batch["prefix"]
            prefix_rows[target_index] += prefix_mask.sum()
            prefix_top1_ok[target_index] += (top1_ok & prefix_mask).sum()
        top1_action = codebook[top1] + output["candidate_action_residual"][torch.arange(count, device=device), top1]
        top1_action_square += float((top1_action - batch["future_actions"][:, 0]).square().sum())
        top1_action_count += top1_action.numel()
        selected_energy = top1_action.square().mean((1, 2)).sqrt()
        nontrivial = batch["immediate_energy"] >= 0.002
        nontrivial_rows += int(nontrivial.sum())
        nontrivial_recalled += int(((selected_energy >= 0.002) & nontrivial).sum())
        future_covered += int((top[:, :, None] == batch["future_codes"][:, None]).any((1, 2)).sum())
        immediate_covered += int((top == batch["future_codes"][:, :1]).any(1).sum())
        target_ok += int((output["target_logits"].argmax(-1) == batch["target"]).sum())
        arm_ok += int((output["active_arm_logits"].argmax(-1) == batch["arm"]).sum())
        rows += count
    result = {
        "loss": total / max(rows, 1),
        "object_position_rmse_mm": 1e3 * math.sqrt(pose_square / max(pose_count, 1)),
        "future_candidate_action_rmse_mrad": 1e3 * math.sqrt(action_square / max(action_count, 1)),
        "proposal_future_recall_at_12": future_covered / max(rows, 1),
        "proposal_immediate_recall_at_12": immediate_covered / max(rows, 1),
        "proposal_immediate_top1_accuracy": immediate_top1 / max(rows, 1),
        "top1_action_rmse_mrad": 1e3 * math.sqrt(top1_action_square / max(top1_action_count, 1)),
        "top1_nontrivial_action_recall": nontrivial_recalled / max(nontrivial_rows, 1),
        "target_accuracy": target_ok / max(rows, 1),
        "active_arm_accuracy": arm_ok / max(rows, 1),
    }
    names = ("red", "yellow", "green")
    for index, name in enumerate(names):
        result[f"{name}_immediate_top1_accuracy"] = (
            int(target_top1_ok[index]) / max(int(target_rows[index]), 1)
        )
        result[f"{name}_prefix_top1_accuracy"] = (
            int(prefix_top1_ok[index]) / max(int(prefix_rows[index]), 1)
        )
    result["worst_target_immediate_top1_accuracy"] = min(
        result[f"{name}_immediate_top1_accuracy"] for name in names
    )
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    future_offsets = tuple(int(value) for value in args.future_offsets.split(","))
    if not future_offsets or min(future_offsets) < 0:
        raise ValueError("--future-offsets must contain non-negative frame offsets")
    train = torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=False)
    validation = torch.load(args.cache_dir / "validation.pt", map_location="cpu", weights_only=False)
    value = statistics(train)
    train_rows = valid_rows(train, args.action_chunk)
    normalized_chunks = action_chunks(train, train_rows, args.action_chunk).flatten(1) / value["action_delta_std"].repeat(args.action_chunk)
    codebook_normalized = kmeans(normalized_chunks, args.token_count, args.seed)
    codebook = (codebook_normalized * value["action_delta_std"].repeat(args.action_chunk)).reshape(args.token_count, args.action_chunk, 31)
    warm_value = None
    token_alignment = None
    if args.warm_start is not None:
        warm_value = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        old_codebook = warm_value["action_codebook"].float().flatten(1)
        if old_codebook.shape == codebook.flatten(1).shape:
            distance = torch.cdist(old_codebook, codebook.flatten(1)).numpy()
            old_ids, new_ids = linear_sum_assignment(distance)
            order = np.empty(args.token_count, dtype=np.int64)
            order[old_ids] = new_ids
            codebook = codebook[torch.from_numpy(order)]
            codebook_normalized = codebook.flatten(1) / value["action_delta_std"].repeat(args.action_chunk)
            token_alignment = {
                "same_index_fraction_before_alignment": float(np.mean(order == np.arange(args.token_count))),
                "mean_matched_rmse_rad": float(
                    distance[old_ids, new_ids].mean() / math.sqrt(codebook[0].numel())
                ),
                "new_index_for_old_index": order.tolist(),
            }
    train_set = TokenDataset(train, args.action_chunk, args.history_frames, value["action_delta_std"], codebook_normalized, future_offsets, True, args.source_root_boost_pattern, args.source_root_boost)
    validation_set = TokenDataset(validation, args.action_chunk, args.history_frames, value["action_delta_std"], codebook_normalized, future_offsets, False)
    train_samples = len(train_set) if args.samples_per_epoch <= 0 else min(args.samples_per_epoch, len(train_set))
    validation_data = validation_set if args.max_validation_rows <= 0 else Subset(validation_set, range(min(args.max_validation_rows, len(validation_set))))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=WeightedRandomSampler(train_set.weights, train_samples, replacement=True), num_workers=0, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    model = ActionTokenVLA(len(train["vocab"]), args.token_count, int(train["contact_sites"].shape[-1]), args.history_frames, args.action_chunk).to(device)
    transferred = []
    if args.warm_start is not None:
        source = warm_value["model"]
        destination = model.state_dict()
        for name, tensor in source.items():
            if name in destination and destination[name].shape == tensor.shape:
                destination[name] = tensor
                transferred.append(name)
        model.load_state_dict(destination)
        if not transferred:
            raise ValueError("Warm-start checkpoint has no shape-compatible visual or temporal weights")
        # Preserve the old function under the new input normalization.
        old_stats = warm_value.get("stats", {})
        with torch.no_grad():
            linear = model.proprio[0]
            offset = 0
            for name, size in (("state", 31), ("velocity", 31), ("contact_sites", model.contact_site_dim)):
                old_mean, old_std = old_stats.get(f"{name}_mean"), old_stats.get(f"{name}_std")
                if old_mean is not None and old_std is not None:
                    old_mean = old_mean.to(linear.weight.device)
                    old_std = old_std.to(linear.weight.device)
                    new_mean = value[f"{name}_mean"].to(linear.weight.device)
                    new_std = value[f"{name}_std"].to(linear.weight.device)
                    weight = linear.weight[:, offset : offset + size].clone()
                    linear.weight[:, offset : offset + size] = weight * (new_std / old_std)
                    linear.bias.add_(weight @ ((new_mean - old_mean) / old_std))
                offset += size
            old_object_mean = old_stats.get("objects_mean")
            old_object_std = old_stats.get("objects_std")
            if old_object_mean is not None and old_object_std is not None:
                old_object_mean = old_object_mean.to(model.object_head.weight.device)
                old_object_std = old_object_std.to(model.object_head.weight.device)
                new_object_mean = value["objects_mean"].to(model.object_head.weight.device)
                new_object_std = value["objects_std"].to(model.object_head.weight.device)
                scale = old_object_std / new_object_std
                old_weight = model.object_head.weight.clone()
                old_bias = model.object_head.bias.clone()
                model.object_head.weight.copy_(old_weight * scale[:, None])
                model.object_head.bias.copy_(
                    old_bias * scale + (old_object_mean - new_object_mean) / new_object_std
                )
    if args.freeze_visual_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            if not (name.startswith("token_head.") or name.startswith("token_embedding.") or name.startswith("action_residual.")):
                parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    gpu_value = {key: item.to(device) for key, item in value.items()}
    gpu_codebook = codebook.to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = -math.inf
    history = []
    for epoch in range(args.epochs):
        model.train(); total = 0.0; rows = 0
        for raw in train_loader:
            batch = {key: item.to(device, non_blocking=True) for key, item in raw.items()}
            loss, _, _, _ = losses(model, batch, gpu_value, gpu_codebook)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss) * len(batch["target"]); rows += len(batch["target"])
        report = evaluate(model, validation_loader, gpu_value, gpu_codebook, device)
        report.update(epoch=epoch + 1, train_loss=total / max(rows, 1)); history.append(report)
        print("ACTION_TOKEN_VLA_EPOCH=" + json.dumps(report), flush=True)
        score = (
            .30 * report["proposal_immediate_top1_accuracy"]
            + .20 * report["worst_target_immediate_top1_accuracy"]
            + .20 * report["yellow_prefix_top1_accuracy"]
            + .20 * report["top1_nontrivial_action_recall"]
            + .10 * report["proposal_future_recall_at_12"]
            - report["object_position_rmse_mm"] / 1000.0
            - report["top1_action_rmse_mrad"] / 1000.0
        )
        if score > best:
            best = score
            save_checkpoint(args.output_dir / "best.pt", model, train["vocab"], value, codebook, future_offsets)
    summary = {
        "experiment": "G1LANG-discrete-action-token-multiview-vla",
        "cache_dir": str(args.cache_dir),
        "token_count": args.token_count,
        "action_chunk": args.action_chunk,
        "future_offsets_frames": list(future_offsets),
        "input_contract": "causal raw RGB x3 + language + state31 + qvel31 + finger sites + tactile",
        "action_representation": "learned discrete strict-expert action-chunk proposals",
        "privileged_positions_at_inference": False,
        "warm_start": str(args.warm_start) if args.warm_start is not None else None,
        "warm_start_transferred_parameters": transferred,
        "warm_start_token_alignment": token_alignment,
        "sampling_contract": "deduplicated target-episode-stage balance with leading/trailing wait trimming and activity weighting",
        "samples_per_epoch": train_samples,
        "validation_rows_evaluated": len(validation_data),
        "source_root_boost_pattern": args.source_root_boost_pattern or None,
        "source_root_boost": args.source_root_boost,
        "source_root_boost_sampling_mass": (
            float(sum(weight for weight, (episode, _) in zip(train_set.weights, train_set.rows) if args.source_root_boost_pattern and args.source_root_boost_pattern in train.get("source_root", [""])[episode]) / sum(train_set.weights))
            if args.source_root_boost_pattern else 0.0
        ),
        "visual_encoder_frozen": bool(args.freeze_visual_encoder),
        "backbone_frozen": bool(args.freeze_backbone),
        "best_validation_score": best,
        "history": history,
        "final": history[-1],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
