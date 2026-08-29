#!/usr/bin/env python3
"""Prepare and train the compact RGB-language-state G1 policy."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from g1_compact_vla import (
    CAMERA_KEYS,
    CompactG1VLA,
    FrozenResNet18,
    SpatialFrozenResNet18,
    build_vocab,
    encode_tasks,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-horizon", type=int, default=20)
    parser.add_argument(
        "--action-offset",
        type=int,
        default=0,
        help="Frame offset for action chunks; 0 aligns each observation with its recorded command.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--phase-action-weight", type=float, default=2.0)
    parser.add_argument("--phase0-action-weight", type=float, default=1.0)
    parser.add_argument(
        "--contact-phase-action-weight",
        type=float,
        help=(
            "Optional action-loss weight for physical approach/contact phases 2 and 3. "
            "This raises contact precision without changing the phase classifier."
        ),
    )
    parser.add_argument(
        "--phase-self-conditioning",
        action="store_true",
        help="Use the model-predicted phase posterior for action conditioning during training.",
    )
    parser.add_argument("--target-loss-weight", type=float, default=0.25)
    parser.add_argument("--checkpoint-target-weight", type=float, default=0.0)
    parser.add_argument("--target-conditioned-actions", action="store_true")
    parser.add_argument("--target-mixture-actions", action="store_true")
    parser.add_argument("--arm-action-weight", type=float, default=3.0)
    parser.add_argument(
        "--hand-action-weight",
        type=float,
        default=1.5,
        help="Loss weight for the 7 hand joints of each arm.",
    )
    parser.add_argument("--spatial-vision", action="store_true")
    parser.add_argument(
        "--contact-action",
        action="store_true",
        help=(
            "Train the 35D Cartesian contact contract stored in contact_action: "
            "waist, two palm poses, and fourteen hand joints."
        ),
    )
    parser.add_argument(
        "--target-relative-contact-action",
        action="store_true",
        help=(
            "Use contact_relative_action, where both palm positions are relative "
            "to the language-selected camera-localized object."
        ),
    )
    parser.add_argument(
        "--palm-residual-contact-action",
        action="store_true",
        help=(
            "Use contact_palm_residual_action, where future Cartesian palm "
            "waypoints are residuals from the current measured palm positions."
        ),
    )
    parser.add_argument(
        "--palm-lookahead-residual-contact-action",
        action="store_true",
        help=(
            "Train the final expert waypoint as a one-step visual-servo palm "
            "residual. Requires --chunk-horizon=1."
        ),
    )
    parser.add_argument(
        "--visual-grasp-residual-conditioning",
        action="store_true",
        help=(
            "Condition the action head on the language-selected visual target "
            "position relative to the current measured left/right palms."
        ),
    )
    parser.add_argument("--palm-action-weight", type=float, default=6.0)
    parser.add_argument("--orientation-action-weight", type=float, default=2.0)
    parser.add_argument(
        "--action-object-teacher-forcing",
        action="store_true",
        help=(
            "Condition the action head on the supervised object position during "
            "training. Deployment must provide the corresponding camera estimate."
        ),
    )
    parser.add_argument(
        "--action-object-noise-m",
        type=float,
        default=0.0,
        help="Isotropic metric noise applied to teacher-forced action object positions.",
    )
    parser.add_argument(
        "--action-delta",
        action="store_true",
        help="Train on action minus current joint state and restore absolute targets at inference.",
    )
    parser.add_argument(
        "--disable-phase-action-conditioning",
        action="store_true",
        help="Keep the phase auxiliary head but do not feed its prediction to the action head.",
    )
    parser.add_argument(
        "--active-arm-conditioning",
        action="store_true",
        help="Condition actions on the model-predicted active arm (left/right).",
    )
    parser.add_argument(
        "--color-features",
        action="store_true",
        help="Use head-camera color centroid features for spatial grounding.",
    )
    parser.add_argument(
        "--temporal-history",
        action="store_true",
        help="Use recurrent history of prior proprioception and head-camera color observations.",
    )
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument(
        "--palm-conditioning",
        action="store_true",
        help="Predict and condition actions on the expert palm-position trajectory.",
    )
    parser.add_argument("--palm-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--palm-pose-conditioning",
        action="store_true",
        help="Also supervise and condition actions on 6D wrist orientation per hand.",
    )
    return parser.parse_args()


def load_tasks(root: Path) -> list[str]:
    return [
        json.loads(line)["task"]
        for line in (root / "meta/tasks.jsonl").read_text().splitlines()
        if line.strip()
    ]


def episode_rows(root: Path) -> dict[int, dict[str, np.ndarray]]:
    rows: dict[int, list[dict]] = {}
    for path in sorted((root / "data").glob("**/*.parquet")):
        table = pq.read_table(path)
        columns = table.to_pydict()
        for index in range(table.num_rows):
            episode = int(columns["episode_index"][index])
            rows.setdefault(episode, []).append(
                {
                    key: columns[key][index]
                    for key in (
                        "frame_index",
                        "observation.state",
                        "action",
                        "complementary_info.object_position",
                        "complementary_info.target_object_index",
                        "complementary_info.task_phase",
                        "complementary_info.active_arm_index",
                        "task_index",
                    )
                }
            )
    result = {}
    for episode, values in rows.items():
        values.sort(key=lambda item: int(item["frame_index"]))
        result[episode] = {
            key: np.asarray([item[key] for item in values])
            for key in values[0]
        }
    return result


@torch.no_grad()
def encode_video(path: Path, encoder: FrozenResNet18, device: str) -> torch.Tensor:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
    if not frames:
        raise RuntimeError(f"Video contains no frames: {path}")
    embeddings = []
    for start in range(0, len(frames), 64):
        batch = torch.stack(frames[start : start + 64]).permute(0, 3, 1, 2)
        embeddings.append(encoder(batch.to(device)))
    return torch.cat(embeddings, dim=0).cpu().half()


@torch.no_grad()
def encode_color_features(path: Path) -> torch.Tensor:
    """Extract stable RGB spatial cues from the head camera.

    The features are normalized x/y centroids and pixel area for red, yellow,
    and green objects. They are camera observations, not simulator state.
    """
    features = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            image = frame.to_ndarray(format="rgb24").astype(np.float32)
            red = (image[..., 0] > 120) & (image[..., 0] > image[..., 1] * 1.25) & (image[..., 0] > image[..., 2] * 1.25)
            yellow = (image[..., 0] > 120) & (image[..., 1] > 90) & (image[..., 2] < 120)
            green = (image[..., 1] > 80) & (image[..., 1] > image[..., 0] * 1.25) & (image[..., 1] > image[..., 2] * 1.15)
            row = []
            height, width = image.shape[:2]
            for mask in (red, yellow, green):
                yy, xx = np.where(mask)
                if len(xx) == 0:
                    row.extend((0.5, 0.5, 0.0))
                else:
                    row.extend((float(xx.mean() / width), float(yy.mean() / height), float(len(xx) / (width * height))))
            features.append(row)
    if not features:
        raise RuntimeError(f"Video contains no frames: {path}")
    return torch.tensor(features, dtype=torch.float32)


def build_cache(
    root: Path,
    cache_path: Path,
    horizon: int,
    device: str,
    spatial_vision: bool = False,
    action_offset: int = 0,
) -> dict:
    tasks = load_tasks(root)
    vocab = build_vocab(tasks)
    rows = episode_rows(root)
    encoder = (
        SpatialFrozenResNet18() if spatial_vision else FrozenResNet18()
    ).to(device).eval()
    vision_parts = []
    state_parts = []
    action_parts = []
    object_parts = []
    target_parts = []
    phase_parts = []
    active_arm_parts = []
    color_feature_parts = []
    task_parts = []
    episode_parts = []
    frame_parts = []
    for episode in sorted(rows):
        values = rows[episode]
        frame_count = len(values["frame_index"])
        embeddings = []
        for camera_key in CAMERA_KEYS:
            video_path = (
                root
                / "videos/chunk-000"
                / camera_key
                / f"episode_{episode:06d}.mp4"
            )
            encoded = encode_video(video_path, encoder, device)
            if len(encoded) != frame_count:
                raise ValueError(
                    f"Frame mismatch for {video_path}: {len(encoded)} != {frame_count}"
                )
            embeddings.append(encoded)
            if camera_key == "observation.images.head":
                color_features = encode_color_features(video_path)
                if len(color_features) != frame_count:
                    raise ValueError(
                        f"Color feature frame mismatch for {video_path}: {len(color_features)} != {frame_count}"
                    )
        vision_parts.append(torch.stack(embeddings, dim=1))
        color_feature_parts.append(color_features)
        states = torch.from_numpy(values["observation.state"].astype(np.float32))
        actions = torch.from_numpy(values["action"].astype(np.float32))
        chunks = []
        for frame in range(frame_count):
            indices = [
                min(max(frame + action_offset + offset, 0), frame_count - 1)
                for offset in range(horizon)
            ]
            chunks.append(actions[indices])
        state_parts.append(states)
        action_parts.append(torch.stack(chunks))
        object_parts.append(
            torch.from_numpy(values["complementary_info.object_position"].astype(np.float32))
        )
        target_parts.append(
            torch.from_numpy(
                values["complementary_info.target_object_index"].astype(np.int64)
            )
        )
        phase_parts.append(
            torch.from_numpy(values["complementary_info.task_phase"].astype(np.int64))
        )
        active_arm_parts.append(
            torch.from_numpy(
                values["complementary_info.active_arm_index"].astype(np.int64)
            )
        )
        task_parts.append(torch.from_numpy(values["task_index"].astype(np.int64)))
        episode_parts.append(torch.full((frame_count,), episode, dtype=torch.long))
        frame_parts.append(torch.from_numpy(values["frame_index"].astype(np.int64)))
        print(f"CACHE_EPISODE={episode} frames={frame_count}", flush=True)
    cache = {
        "vision": torch.cat(vision_parts),
        "state": torch.cat(state_parts),
        "action": torch.cat(action_parts),
        "object_position": torch.cat(object_parts),
        "target": torch.cat(target_parts),
        "phase": torch.cat(phase_parts),
        "active_arm": torch.cat(active_arm_parts),
        "color_features": torch.cat(color_feature_parts),
        "task_index": torch.cat(task_parts),
        "episode_index": torch.cat(episode_parts),
        "frame_index": torch.cat(frame_parts),
        "vocab": vocab,
        "tasks": tasks,
        "horizon": horizon,
        "action_offset": action_offset,
        "spatial_vision": spatial_vision,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Saved cache {cache_path} samples={len(cache['state'])}", flush=True)
    return cache


def cache_stats(
    train: dict,
    action_delta: bool = False,
    action_key: str = "action",
    chunk_horizon: int | None = None,
) -> dict[str, torch.Tensor]:
    state = train["state"].float()
    raw_actions = train[action_key].float()
    if chunk_horizon is not None:
        if raw_actions.shape[1] < chunk_horizon:
            raise ValueError(
                f"Cache horizon {raw_actions.shape[1]} is shorter than requested "
                f"horizon {chunk_horizon}"
            )
        raw_actions = raw_actions[:, :chunk_horizon]
    if action_delta and raw_actions.shape[-1] != state.shape[-1]:
        raise ValueError("--action-delta is only valid for the 31D joint action contract")
    actions = raw_actions - state[:, None, :] if action_delta else raw_actions
    actions = actions.reshape(-1, actions.shape[-1])
    objects = train["object_position"].float()
    stats = {
        "state_mean": state.mean(0),
        "state_std": state.std(0).clamp_min(1e-3),
        "action_mean": actions.mean(0),
        "action_std": actions.std(0).clamp_min(1e-3),
        "object_mean": objects.mean(0),
        "object_std": objects.std(0).clamp_min(1e-3),
    }
    if "palm_position" in train:
        palms = train["palm_position"].float()
        stats["palm_mean"] = palms.mean(0)
        stats["palm_std"] = palms.std(0).clamp_min(1e-3)
    if "palm_orientation" in train:
        orientation = train["palm_orientation"].float()
        stats["palm_orientation_mean"] = orientation.mean(0)
        stats["palm_orientation_std"] = orientation.std(0).clamp_min(1e-3)
    if "palm_position" in train:
        residual = target_palm_residual(train)
        stats["grasp_residual_mean"] = residual.mean(0)
        stats["grasp_residual_std"] = residual.std(0).clamp_min(1e-3)
    return stats


def target_palm_residual(cache: dict) -> torch.Tensor:
    """Return selected-object position minus the two current palm positions."""
    if "palm_position" not in cache:
        raise ValueError("visual grasp residual requires palm_position in the cache")
    objects = cache["object_position"].float().reshape(-1, 3, 3)
    target = cache["target"].long()
    selected = objects[torch.arange(len(objects)), target]
    palms = cache["palm_position"].float().reshape(-1, 2, 3)
    return (selected[:, None, :] - palms).reshape(-1, 6)


def history_indices(cache: dict, history_length: int) -> torch.Tensor:
    if history_length < 1:
        raise ValueError("history_length must be positive")
    episodes = cache["episode_index"].long().tolist()
    frames = cache["frame_index"].long().tolist()
    lookup = {
        (int(episode), int(frame)): index
        for index, (episode, frame) in enumerate(zip(episodes, frames))
    }
    result = torch.empty((len(episodes), history_length), dtype=torch.long)
    for index, (episode, frame) in enumerate(zip(episodes, frames)):
        values = []
        for offset in range(history_length - 1, -1, -1):
            values.append(lookup.get((int(episode), max(0, int(frame) - offset)), index))
        result[index] = torch.tensor(values, dtype=torch.long)
    return result


def make_dataset(
    cache: dict,
    stats: dict[str, torch.Tensor],
    action_delta: bool = False,
    temporal_history: bool = False,
    history_length: int = 4,
    action_key: str = "action",
    chunk_horizon: int | None = None,
    visual_grasp_residual: bool = False,
):
    state = (cache["state"].float() - stats["state_mean"]) / stats["state_std"]
    raw_state = cache["state"].float()
    raw_action = cache[action_key].float()
    if chunk_horizon is not None:
        if raw_action.shape[1] < chunk_horizon:
            raise ValueError(
                f"Cache horizon {raw_action.shape[1]} is shorter than requested "
                f"horizon {chunk_horizon}"
            )
        raw_action = raw_action[:, :chunk_horizon]
    if action_delta:
        raw_action = raw_action - raw_state[:, None, :]
    action = (
        raw_action - stats["action_mean"]
    ) / stats["action_std"]
    objects = (
        cache["object_position"].float() - stats["object_mean"]
    ) / stats["object_std"]
    token_ids, token_mask = encode_tasks(
        [cache["tasks"][int(index)] for index in cache["task_index"]],
        cache["vocab"],
    )
    values = [
        cache["vision"].float(),
        state,
        action,
        objects,
        cache["target"].long(),
        cache["phase"].long(),
        cache["active_arm"].long(),
        cache["color_features"].float(),
    ]
    if visual_grasp_residual:
        residual = target_palm_residual(cache)
        residual = (
            residual - stats["grasp_residual_mean"]
        ) / stats["grasp_residual_std"]
        values.append(residual)
    if "palm_position" in cache:
        palms = (cache["palm_position"].float() - stats["palm_mean"]) / stats["palm_std"]
    else:
        palms = None
    if palms is not None:
        values.append(palms)
    if "palm_orientation" in cache:
        orientation = (
            cache["palm_orientation"].float() - stats["palm_orientation_mean"]
        ) / stats["palm_orientation_std"]
        values.append(orientation)
    if temporal_history:
        indices = history_indices(cache, history_length)
        values.extend((state[indices], cache["color_features"].float()[indices]))
    values.extend((token_ids, token_mask))
    return TensorDataset(*values)


def unpack_batch(
    items: list[torch.Tensor],
    temporal_history: bool,
    palm_conditioning: bool,
    palm_pose_conditioning: bool,
    visual_grasp_residual: bool,
):
    """Unpack cache fields while keeping optional action conditioning explicit."""
    (
        vision, state, action, objects, target, phase, active_arm,
        color_features, *rest,
    ) = items
    grasp_residual = rest.pop(0) if visual_grasp_residual else None
    palm = orientation = None
    if palm_conditioning:
        palm = rest.pop(0)
        if palm_pose_conditioning:
            orientation = rest.pop(0)
    if temporal_history:
        history_state = rest.pop(0)
        history_color = rest.pop(0)
    else:
        history_state = history_color = None
    token_ids = rest.pop(0)
    token_mask = rest.pop(0)
    if rest:
        raise ValueError(f"Unexpected dataset fields: {len(rest)}")
    return (
        vision, state, action, objects, target, phase, active_arm,
        color_features, grasp_residual, palm, orientation, history_state,
        history_color, token_ids, token_mask,
    )


def evaluate(
    model,
    loader,
    device,
    action_weights,
    phase_action_weight: float,
    phase0_action_weight: float,
    contact_phase_action_weight: float | None = None,
    temporal_history: bool = False,
    palm_conditioning: bool = False,
    palm_pose_conditioning: bool = False,
    action_object_teacher_forcing: bool = False,
    visual_grasp_residual: bool = False,
):
    model.eval()
    totals = {"action": 0.0, "object": 0.0, "target": 0.0, "phase": 0.0, "active_arm": 0.0, "palm": 0.0, "palm_orientation": 0.0, "count": 0}
    correct_target = 0
    correct_phase = 0
    correct_active_arm = 0
    with torch.no_grad():
        for batch in loader:
            items = [item.to(device) for item in batch]
            (
                vision, state, action, objects, target, phase, active_arm,
                color_features, grasp_residual, palm, orientation, history_state,
                history_color, token_ids, token_mask,
            ) = unpack_batch(
                items,
                temporal_history,
                palm_conditioning,
                palm_pose_conditioning,
                visual_grasp_residual,
            )
            output = model(
                vision,
                state,
                token_ids,
                token_mask,
                color_features=color_features,
                history_state=history_state,
                history_color_features=history_color,
                object_position_override=(
                    objects if action_object_teacher_forcing else None
                ),
                palm_position_override=palm if palm_conditioning else None,
                visual_grasp_residual=grasp_residual,
            )
            action_loss = ((output["action"] - action) ** 2 * action_weights).mean()
            phase_weight = torch.where(
                phase == 0,
                torch.tensor(phase0_action_weight, device=device),
                torch.where(
                torch.isin(
                    phase.clamp(0, 7),
                    torch.tensor([1, 2, 3, 4, 5], device=device),
                ),
                torch.tensor(phase_action_weight, device=device),
                torch.tensor(1.0, device=device),
                ),
            )
            if contact_phase_action_weight is not None:
                phase_weight = torch.where(
                    torch.isin(
                        phase.clamp(0, 7),
                        torch.tensor([2, 3], device=device),
                    ),
                    torch.tensor(contact_phase_action_weight, device=device),
                    phase_weight,
                )
            action_loss = (
                ((output["action"] - action) ** 2 * action_weights).mean(dim=(1, 2))
                * phase_weight
            ).mean()
            object_loss = F.mse_loss(output["object_position"], objects)
            palm_loss = (
                F.mse_loss(output["palm_position"], palm)
                if palm is not None
                else torch.tensor(0.0, device=device)
            )
            target_loss = F.cross_entropy(output["target_logits"], target)
            phase_loss = F.cross_entropy(output["phase_logits"], phase)
            active_arm_loss = F.cross_entropy(output["active_arm_logits"], active_arm)
            count = len(target)
            for key, value in (
                ("action", action_loss),
                ("object", object_loss),
                ("target", target_loss),
                ("phase", phase_loss),
                ("active_arm", active_arm_loss),
                ("palm", palm_loss),
                (
                    "palm_orientation",
                    F.mse_loss(output["palm_orientation"], orientation)
                    if orientation is not None
                    else torch.tensor(0.0, device=device),
                ),
            ):
                totals[key] += float(value) * count
            totals["count"] += count
            correct_target += int((output["target_logits"].argmax(-1) == target).sum())
            correct_phase += int((output["phase_logits"].argmax(-1) == phase).sum())
            correct_active_arm += int(
                (output["active_arm_logits"].argmax(-1) == active_arm).sum()
            )
    count = max(totals.pop("count"), 1)
    return {
        **{key: value / count for key, value in totals.items()},
        "target_accuracy": correct_target / count,
        "phase_accuracy": correct_phase / count,
        "active_arm_accuracy": correct_active_arm / count,
    }


def train_model(args: argparse.Namespace, train_cache: dict, val_cache: dict) -> None:
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    # Token ids must share the training vocabulary.  Each cache is prepared
    # independently, so keeping the validation vocabulary silently changes
    # word indices at evaluation time.
    val_cache["vocab"] = train_cache["vocab"]
    if args.target_relative_contact_action and not args.contact_action:
        raise ValueError("--target-relative-contact-action requires --contact-action")
    if args.palm_residual_contact_action and not args.contact_action:
        raise ValueError("--palm-residual-contact-action requires --contact-action")
    if args.palm_lookahead_residual_contact_action and not args.contact_action:
        raise ValueError(
            "--palm-lookahead-residual-contact-action requires --contact-action"
        )
    if args.palm_lookahead_residual_contact_action and args.chunk_horizon != 1:
        raise ValueError(
            "--palm-lookahead-residual-contact-action requires --chunk-horizon=1"
        )
    if sum(
        (
            args.target_relative_contact_action,
            args.palm_residual_contact_action,
            args.palm_lookahead_residual_contact_action,
        )
    ) > 1:
        raise ValueError("Choose one relative Cartesian contact action contract")
    if args.visual_grasp_residual_conditioning:
        if not (
            args.palm_residual_contact_action
            or args.palm_lookahead_residual_contact_action
        ):
            raise ValueError(
                "--visual-grasp-residual-conditioning requires "
                "--palm-residual-contact-action"
            )
        if not args.palm_conditioning:
            raise ValueError(
                "--visual-grasp-residual-conditioning requires --palm-conditioning"
            )
    action_key = (
        "contact_palm_lookahead_residual_action"
        if args.palm_lookahead_residual_contact_action
        else (
            "contact_palm_residual_action"
            if args.palm_residual_contact_action
            else (
                "contact_relative_action"
                if args.target_relative_contact_action
                else ("contact_action" if args.contact_action else "action")
            )
        )
    )
    if action_key not in train_cache or action_key not in val_cache:
        raise ValueError(f"{action_key} is missing from the train or validation cache")
    if args.contact_action and args.action_delta:
        raise ValueError("--contact-action cannot be combined with --action-delta")
    if args.action_object_noise_m < 0.0:
        raise ValueError("--action-object-noise-m must be non-negative")
    action_dim = int(train_cache[action_key].shape[-1])
    expected_action_dim = 35 if args.contact_action else 31
    if action_dim != expected_action_dim:
        raise ValueError(
            f"Expected {expected_action_dim}D {action_key}, found {action_dim}D"
        )
    stats = cache_stats(
        train_cache, args.action_delta, action_key, args.chunk_horizon
    )
    stats["target_relative_contact_action"] = torch.tensor(
        int(args.target_relative_contact_action), dtype=torch.int64
    )
    stats["palm_residual_contact_action"] = torch.tensor(
        int(
            args.palm_residual_contact_action
            or args.palm_lookahead_residual_contact_action
        ),
        dtype=torch.int64,
    )
    stats["palm_lookahead_residual_contact_action"] = torch.tensor(
        int(args.palm_lookahead_residual_contact_action), dtype=torch.int64
    )
    train_dataset = make_dataset(
        train_cache,
        stats,
        args.action_delta,
        args.temporal_history,
        args.history_length,
        action_key,
        args.chunk_horizon,
        args.visual_grasp_residual_conditioning,
    )
    val_dataset = make_dataset(
        val_cache,
        stats,
        args.action_delta,
        args.temporal_history,
        args.history_length,
        action_key,
        args.chunk_horizon,
        args.visual_grasp_residual_conditioning,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    model = CompactG1VLA(
        vocab_size=len(train_cache["vocab"]),
        chunk_horizon=args.chunk_horizon,
        action_dim=action_dim,
        use_target_conditioning=args.target_conditioned_actions,
        use_target_mixture=args.target_mixture_actions,
        action_is_delta=args.action_delta,
        use_phase_action_conditioning=(
            args.phase_self_conditioning or not args.disable_phase_action_conditioning
        ),
        use_active_arm_conditioning=args.active_arm_conditioning,
        use_color_features=args.color_features,
        spatial_vision=args.spatial_vision,
        use_temporal_history=args.temporal_history,
        history_length=args.history_length,
        use_palm_conditioning=args.palm_conditioning,
        use_palm_pose_conditioning=args.palm_pose_conditioning,
        use_visual_grasp_residual=args.visual_grasp_residual_conditioning,
    ).to(device)
    action_weights = torch.ones(action_dim, device=device)
    if args.contact_action:
        action_weights[:3] = args.arm_action_weight
        action_weights[3:9] = args.palm_action_weight
        action_weights[9:21] = args.orientation_action_weight
        action_weights[21:35] = args.hand_action_weight
    else:
        action_weights[3:10] = args.arm_action_weight
        action_weights[10:17] = args.hand_action_weight
        action_weights[17:24] = args.arm_action_weight
        action_weights[24:31] = args.hand_action_weight
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=2e-3,
        weight_decay=1e-4,
    )
    best = math.inf
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        samples = 0
        for batch in train_loader:
            items = [item.to(device) for item in batch]
            (
                vision, state, action, objects, target, phase, active_arm,
                color_features, grasp_residual, palm, orientation, history_state,
                history_color, token_ids, token_mask,
            ) = unpack_batch(
                items,
                args.temporal_history,
                args.palm_conditioning,
                args.palm_pose_conditioning,
                args.visual_grasp_residual_conditioning,
            )
            target_override = F.one_hot(target, num_classes=3).float()
            phase_override = (
                None
                if args.phase_self_conditioning
                else F.one_hot(phase, num_classes=8).float()
            )
            active_arm_override = F.one_hot(active_arm, num_classes=2).float()
            action_objects = None
            if args.action_object_teacher_forcing:
                action_objects = objects
                if args.action_object_noise_m > 0.0:
                    noise = torch.randn_like(objects) * (
                        args.action_object_noise_m
                        / stats["object_std"].to(device=objects.device)
                    )
                    action_objects = action_objects + noise
            action_grasp_residual = grasp_residual
            if action_grasp_residual is not None and action_objects is not None:
                object_noise_m = (action_objects - objects) * stats["object_std"].to(
                    device=objects.device
                )
                target_noise_m = object_noise_m.reshape(-1, 3, 3)[
                    torch.arange(len(target), device=target.device), target
                ]
                action_grasp_residual = action_grasp_residual + (
                    target_noise_m.repeat(1, 2)
                    / stats["grasp_residual_std"].to(device=objects.device)
                )
            output = model(
                vision,
                state,
                token_ids,
                token_mask,
                target_override=target_override,
                phase_override=phase_override,
                active_arm_override=active_arm_override,
                color_features=color_features,
                history_state=history_state,
                history_color_features=history_color,
                object_position_override=action_objects,
                palm_position_override=palm if args.palm_conditioning else None,
                visual_grasp_residual=action_grasp_residual,
            )
            phase_weight = torch.where(
                phase == 0,
                torch.tensor(args.phase0_action_weight, device=device),
                torch.where(
                torch.isin(
                    phase.clamp(0, 7),
                    torch.tensor([1, 2, 3, 4, 5], device=device),
                ),
                torch.tensor(args.phase_action_weight, device=device),
                torch.tensor(1.0, device=device),
                ),
            )
            if args.contact_phase_action_weight is not None:
                phase_weight = torch.where(
                    torch.isin(
                        phase.clamp(0, 7),
                        torch.tensor([2, 3], device=device),
                    ),
                    torch.tensor(args.contact_phase_action_weight, device=device),
                    phase_weight,
                )
            action_loss = (
                ((output["action"] - action) ** 2 * action_weights).mean(dim=(1, 2))
                * phase_weight
            ).mean()
            object_loss = F.mse_loss(output["object_position"], objects)
            target_loss = F.cross_entropy(output["target_logits"], target)
            phase_loss = F.cross_entropy(output["phase_logits"], phase)
            active_arm_loss = F.cross_entropy(output["active_arm_logits"], active_arm)
            loss = (
                action_loss
                + 0.25 * object_loss
                + args.target_loss_weight * target_loss
                + 0.15 * phase_loss
                + 0.15 * active_arm_loss
                + args.palm_loss_weight * (
                    F.mse_loss(output["palm_position"], palm)
                    if palm is not None
                    else torch.tensor(0.0, device=device)
                )
                + args.palm_loss_weight * (
                    F.mse_loss(output["palm_orientation"], orientation)
                    if orientation is not None
                    else torch.tensor(0.0, device=device)
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss) * len(target)
            samples += len(target)
        metrics = evaluate(
            model,
            val_loader,
            device,
            action_weights,
            args.phase_action_weight,
            args.phase0_action_weight,
            args.contact_phase_action_weight,
            args.temporal_history,
            args.palm_conditioning,
            args.palm_pose_conditioning,
            args.action_object_teacher_forcing,
            args.visual_grasp_residual_conditioning,
        )
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running / max(samples, 1)
        history.append(metrics)
        print("COMPACT_VLA_EPOCH=" + json.dumps(metrics), flush=True)
        score = (
            metrics["action"]
            + 0.25 * metrics["object"]
            + args.checkpoint_target_weight * metrics["target"]
        )
        if score < best:
            best = score
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                train_cache["vocab"],
                stats,
                {
                    "vocab_size": len(train_cache["vocab"]),
                    "chunk_horizon": args.chunk_horizon,
                    "state_dim": 31,
                    "action_dim": action_dim,
                    "use_target_conditioning": args.target_conditioned_actions,
                    "use_target_mixture": args.target_mixture_actions,
                    "action_is_delta": args.action_delta,
                    "use_phase_action_conditioning": (
                        args.phase_self_conditioning
                        or not args.disable_phase_action_conditioning
                    ),
                    "use_active_arm_conditioning": args.active_arm_conditioning,
                    "use_color_features": args.color_features,
                    "spatial_vision": args.spatial_vision,
                    "use_temporal_history": args.temporal_history,
                    "history_length": args.history_length,
                    "use_palm_conditioning": args.palm_conditioning,
                    "use_palm_pose_conditioning": args.palm_pose_conditioning,
                    "use_visual_grasp_residual": args.visual_grasp_residual_conditioning,
                },
            )
    summary = {
        "experiment": "G1LANG-44-compact-multimodal-vla",
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "chunk_horizon": args.chunk_horizon,
        "action_offset": args.action_offset,
        "epochs": args.epochs,
        "best_validation_score": best,
        "final_validation": history[-1],
        "history": history,
        "camera_inputs": list(CAMERA_KEYS),
        "language_input": "tokenized task text",
        "state_input_dim": 31,
        "action_output_dim": action_dim,
        "action_contract": (
            "waist3+palm_position6+wrist_orientation6D12+hand_joints14"
            if args.contact_action
            else "upper_body_joint_position31"
        ),
        "contact_action": args.contact_action,
        "target_relative_contact_action": args.target_relative_contact_action,
        "palm_residual_contact_action": args.palm_residual_contact_action,
        "palm_lookahead_residual_contact_action": (
            args.palm_lookahead_residual_contact_action
        ),
        "visual_grasp_residual_conditioning": args.visual_grasp_residual_conditioning,
        "arm_action_weight": args.arm_action_weight,
        "palm_action_weight": args.palm_action_weight,
        "orientation_action_weight": args.orientation_action_weight,
        "hand_action_weight": args.hand_action_weight,
        "phase0_action_weight": args.phase0_action_weight,
        "contact_phase_action_weight": args.contact_phase_action_weight,
        "action_is_delta": args.action_delta,
        "use_phase_action_conditioning": (
            args.phase_self_conditioning
            or not args.disable_phase_action_conditioning
        ),
        "phase_self_conditioning": args.phase_self_conditioning,
        "use_active_arm_conditioning": args.active_arm_conditioning,
        "use_color_features": args.color_features,
        "spatial_vision": args.spatial_vision,
        "use_temporal_history": args.temporal_history,
        "history_length": args.history_length,
        "use_palm_conditioning": args.palm_conditioning,
        "palm_loss_weight": args.palm_loss_weight,
        "use_palm_pose_conditioning": args.palm_pose_conditioning,
        "action_object_teacher_forcing": args.action_object_teacher_forcing,
        "action_object_noise_m": args.action_object_noise_m,
        "privileged_positions_at_inference": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.output_dir / 'summary.json'}", flush=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_cache_path = args.cache_dir / "train.pt"
    val_cache_path = args.cache_dir / "val.pt"
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    train_cache = (
        torch.load(train_cache_path, weights_only=False)
        if train_cache_path.exists()
        else build_cache(
            args.train_root,
            train_cache_path,
            args.chunk_horizon,
            device,
            args.spatial_vision,
            args.action_offset,
        )
    )
    val_cache = (
        torch.load(val_cache_path, weights_only=False)
        if val_cache_path.exists()
        else build_cache(
            args.val_root,
            val_cache_path,
            args.chunk_horizon,
            device,
            args.spatial_vision,
            args.action_offset,
        )
    )
    if not args.prepare_only:
        train_model(args, train_cache, val_cache)


if __name__ == "__main__":
    main()
