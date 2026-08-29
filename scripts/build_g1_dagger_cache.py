#!/usr/bin/env python3
"""Build a compact VLA cache from failed on-policy rollouts and expert labels.

The rollout observations are kept exactly as observed by the policy.  Only the
action, phase, and target supervision come from the matched strict
contact-only expert trajectory.  Object positions are auxiliary labels from
the rollout physics log and are never policy inputs at inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch

from g1_compact_vla import CAMERA_KEYS, FrozenResNet18, SpatialFrozenResNet18, build_vocab


ROLLOUT_CAMERAS = {
    "observation.images.head": "head_camera.mp4",
    "observation.images.left_wrist": "left_wrist_camera.mp4",
    "observation.images.right_wrist": "right_wrist_camera.mp4",
}
OBJECT_NAMES = ("red_triangle", "yellow_rod", "green_cube")
TASKS = [
    "put the red triangular prism into the blue box",
    "put the yellow rod into the blue box",
    "put the green cube into the blue box",
    "place the yellow stick inside the blue container",
    "place the green block inside the blue container",
    "move the red triangular object into the blue bin",
    "move the yellow cylindrical object into the blue bin",
    "move the green cubic object into the blue bin",
    "place the red triangle inside the blue container",
]
TASK_TARGETS = (0, 1, 2, 1, 2, 0, 1, 2, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path)
    parser.add_argument("--state-correction-gain", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-horizon", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spatial-vision", action="store_true")
    return parser.parse_args()


def encode_video(path: Path, encoder: FrozenResNet18, device: str) -> torch.Tensor:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
    if not frames:
        raise RuntimeError(f"Video contains no frames: {path}")
    encoded = []
    for start in range(0, len(frames), 64):
        batch = torch.stack(frames[start : start + 64]).permute(0, 3, 1, 2)
        encoded.append(encoder(batch.to(device)))
    return torch.cat(encoded).cpu().half()


def expert_for_target(expert_root: Path, target: int) -> tuple[np.ndarray, dict]:
    # The first canonical episode for each target is a strict, contact-only
    # trajectory.  Its language and action style are the labels for DAgger.
    for path in sorted(expert_root.glob("episode_*/expert_pick_place_episode.npz")):
        episode_dir = path.parent
        summary = json.loads(
            (episode_dir / "language_pick_place_summary.json").read_text()
        )
        if OBJECT_NAMES.index(summary["target_object"]) != target:
            continue
        if not summary.get("passed"):
            continue
        arrays = np.load(path, allow_pickle=True)
        if np.any(arrays["assist_active"] != 0):
            arrays.close()
            continue
        return arrays, summary
    raise FileNotFoundError(f"No strict expert episode for target index {target}")


def remap_expert_frame(frame: int, rollout_frames: int, expert_frames: int) -> int:
    if rollout_frames <= 1:
        return 0
    return int(round(frame * (expert_frames - 1) / (rollout_frames - 1)))


def build_train(args: argparse.Namespace, encoder: FrozenResNet18, device: str) -> dict:
    tasks = TASKS
    vocab = build_vocab(tasks)
    parts = {key: [] for key in (
        "vision", "state", "action", "object_position", "target", "phase",
        "task_index", "episode_index", "frame_index",
    )}
    episode_dirs = sorted(args.rollout_root.glob("episode_*"))
    if not episode_dirs:
        raise FileNotFoundError(f"No rollout episodes under {args.rollout_root}")
    for new_episode, episode_dir in enumerate(episode_dirs):
        rollout = np.load(episode_dir / "trajectory.npz", allow_pickle=True)
        summary = json.loads((episode_dir / "summary.json").read_text())
        target_name = summary["target_object"]
        target = OBJECT_NAMES.index(target_name)
        expert, expert_summary = expert_for_target(args.expert_root, target)
        instruction = tasks[target]
        if target_name != expert_summary["target_object"]:
            raise ValueError("Expert target mismatch")
        states = torch.from_numpy(rollout["observation_joint_position_rad"].astype(np.float32))
        rollout_objects = torch.from_numpy(
            rollout["object_position_m"].reshape(len(states), -1).astype(np.float32)
        )
        frame_count = len(states)
        expert_actions = expert["action_joint_position_rad"][:, 12:].astype(np.float32)
        expert_phases = expert["task_phase"].astype(np.int64)
        expert_targets = expert["target_object_index"].astype(np.int64)
        if not np.all(expert_targets == target):
            raise ValueError(f"Expert target labels changed in {expert_summary}")
        expert_states = expert["joint_position_rad"][:, 12:].astype(np.float32)
        action_chunks = []
        phases = []
        for frame in range(frame_count):
            expert_frame = remap_expert_frame(frame, frame_count, len(expert_actions))
            indices = [min(expert_frame + offset, len(expert_actions) - 1)
                       for offset in range(1, args.chunk_horizon + 1)]
            # Approximate the strict expert's corrective response at the
            # policy's off-trajectory state.  The expert action is an absolute
            # joint target, so subtracting the observed tracking error pulls
            # the target back toward the canonical trajectory instead of
            # replaying an incompatible absolute pose.
            state_error = states[frame].numpy() - expert_states[expert_frame]
            corrected = expert_actions[indices] - args.state_correction_gain * state_error
            corrected = np.clip(corrected, expert_actions[indices] - 0.25,
                                expert_actions[indices] + 0.25)
            action_chunks.append(torch.from_numpy(corrected.astype(np.float32)))
            phases.append(int(expert_phases[expert_frame]))
        vision = []
        for key, filename in ROLLOUT_CAMERAS.items():
            encoded = encode_video(episode_dir / filename, encoder, device)
            if len(encoded) != frame_count:
                raise ValueError(
                    f"Frame mismatch {episode_dir / filename}: {len(encoded)} != {frame_count}"
                )
            vision.append(encoded)
        parts["vision"].append(torch.stack(vision, dim=1))
        parts["state"].append(states)
        parts["action"].append(torch.stack(action_chunks))
        parts["object_position"].append(rollout_objects)
        parts["target"].append(torch.full((frame_count,), target, dtype=torch.long))
        parts["phase"].append(torch.tensor(phases, dtype=torch.long))
        variant_indices = [i for i, value in enumerate(TASK_TARGETS) if value == target]
        parts["task_index"].append(torch.full((frame_count,), variant_indices[0], dtype=torch.long))
        parts["episode_index"].append(torch.full((frame_count,), new_episode, dtype=torch.long))
        parts["frame_index"].append(torch.arange(frame_count, dtype=torch.long))
        # Reuse the on-policy observation under the other language variants
        # for this target.  The labels remain the same target-specific expert
        # action, while the language head learns all evaluation phrasings.
        for variant_index in variant_indices[1:]:
            for key in ("vision", "state", "action", "object_position", "target", "phase"):
                parts[key].append(parts[key][-1].clone())
            parts["task_index"].append(torch.full((frame_count,), variant_index, dtype=torch.long))
            parts["episode_index"].append(torch.full((frame_count,), new_episode, dtype=torch.long))
            parts["frame_index"].append(torch.arange(frame_count, dtype=torch.long))
        print(f"DAGGER_EPISODE={new_episode} target={target_name} frames={frame_count}", flush=True)
        expert.close()
        rollout.close()
    result = {
        key: torch.cat(values) for key, values in parts.items()
    } | {"vocab": vocab, "tasks": tasks, "horizon": args.chunk_horizon}
    if args.base_cache is not None:
        base = torch.load(args.base_cache, map_location="cpu", weights_only=False)
        for key in parts:
            result[key] = torch.cat((base[key], result[key]))
    return result


def build_val(source_path: Path, train_cache: dict, horizon: int) -> dict:
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    # Reuse only successful expert validation frames, truncate existing chunks,
    # and map every source instruction to the canonical three-task vocabulary.
    source["action"] = source["action"][:, :horizon].contiguous()
    source["tasks"] = list(train_cache["tasks"])
    targets = source["target"].long()
    source["task_index"] = targets.clone()
    source["vocab"] = train_cache["vocab"]
    source["horizon"] = horizon
    return source


def main() -> None:
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    encoder = (
        SpatialFrozenResNet18() if args.spatial_vision else FrozenResNet18()
    ).to(device).eval()
    train = build_train(args, encoder, device)
    train["spatial_vision"] = args.spatial_vision
    val = build_val(args.val_cache, train, args.chunk_horizon)
    val["spatial_vision"] = args.spatial_vision
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train, args.output_dir / "train.pt")
    torch.save(val, args.output_dir / "val.pt")
    summary = {
        "experiment": "G1LANG-117-dagger-cache",
        "rollout_root": str(args.rollout_root),
        "expert_root": str(args.expert_root),
        "train_samples": len(train["state"]),
        "validation_samples": len(val["state"]),
        "chunk_horizon": args.chunk_horizon,
        "camera_inputs": list(CAMERA_KEYS),
        "state_input_dim": 31,
        "action_output_dim": 31,
        "labels": "strict contact-only canonical expert action aligned by normalized episode time",
        "state_correction_gain": args.state_correction_gain,
        "privileged_positions_at_inference": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
