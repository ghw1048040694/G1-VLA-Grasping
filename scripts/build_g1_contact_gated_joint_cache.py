#!/usr/bin/env python3
"""Build contact-gated direct-joint phase candidates from dynamic experts.

Every candidate is a command that was physically executed in a strict MuJoCo
trajectory.  Unlike the temporal cache, rows are conditioned only on the
current visual/proprioceptive state and expose the four semantic phases to the
learned planner: home, approach, pregrasp, and contact.  Lift/place commands
are deliberately excluded from this contact gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from build_g1_direct_joint_dynamic_cache import (
    OBJECT_NAMES,
    TOKEN_RE,
    build_vocab,
    episode_paths,
    load_episode,
)


PHASE_START = np.asarray((0, 8, 35, 51), dtype=np.int64)
PHASE_END = np.asarray((7, 34, 50, 72), dtype=np.int64)
LOOKAHEAD_FRAMES = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-stride", type=int, default=6)
    return parser.parse_args()


def desired_stage(frame: int) -> int:
    # The label is the *next* verified contact-gated candidate.  Switching a
    # few frames before an endpoint prevents a stateless policy from holding
    # the approach/pregrasp endpoint forever once it reaches that state.
    if frame <= PHASE_END[1] - LOOKAHEAD_FRAMES:
        return 1
    if frame <= PHASE_END[2] - LOOKAHEAD_FRAMES:
        return 2
    return 3


def action_candidates(action: np.ndarray, frame: int) -> np.ndarray:
    indices = []
    for start, end in zip(PHASE_START, PHASE_END):
        # A candidate is a short lookahead only inside its own dynamic phase;
        # after the phase it holds the verified endpoint rather than advancing
        # to lift/place without a learned contact transition.
        index = min(max(frame, int(start)) + LOOKAHEAD_FRAMES, int(end))
        indices.append(index)
    return action[np.asarray(indices)]


def pack(episodes, tasks, vocab) -> dict:
    state = []
    palms = []
    actions = []
    stage = []
    arm = []
    target = []
    task_index = []
    color = []
    visual = []
    true = []
    source_episode = []
    for item in episodes:
        frames = range(int(PHASE_END[-1]) + 1)
        row_count = len(frames)
        state.append(item.state[list(frames)])
        palms.append(item.palms[list(frames)])
        actions.append(np.stack([action_candidates(item.action, frame) for frame in frames]))
        stage.append(np.asarray([desired_stage(frame) for frame in frames], dtype=np.int64))
        arm.append(item.active_arm[list(frames)])
        target.append(np.full(row_count, item.target, dtype=np.int64))
        task_index.append(np.full(row_count, tasks.index(item.task), dtype=np.int64))
        color.append(np.repeat(item.color_features[None], row_count, axis=0))
        visual.append(np.repeat(item.visual_objects[None], row_count, axis=0))
        true.append(item.true_objects[list(frames)])
        source_episode.extend([f"{item.source_root}/{item.name}"] * row_count)
    return {
        "state": torch.from_numpy(np.concatenate(state)),
        "palm_position": torch.from_numpy(np.concatenate(palms)),
        "candidate_action": torch.from_numpy(np.concatenate(actions)),
        "desired_stage": torch.from_numpy(np.concatenate(stage)),
        "active_arm": torch.from_numpy(np.concatenate(arm)),
        "target": torch.from_numpy(np.concatenate(target)),
        "task_index": torch.from_numpy(np.concatenate(task_index)),
        "color_features": torch.from_numpy(np.concatenate(color)),
        "visual_object_position": torch.from_numpy(np.concatenate(visual)),
        "audit_true_object_position": torch.from_numpy(np.concatenate(true)),
        "source_episode": source_episode,
        "tasks": tasks,
        "vocab": vocab,
        "candidate_phase_start_frames": torch.from_numpy(PHASE_START.copy()),
        "candidate_phase_end_frames": torch.from_numpy(PHASE_END.copy()),
        "candidate_lookahead_frames": torch.tensor(LOOKAHEAD_FRAMES, dtype=torch.int64),
        "action_contract": "strict_dynamic_contact_gated_direct_joint_candidates",
    }


def main() -> None:
    args = parse_args()
    if args.validation_stride < 2:
        raise ValueError("--validation-stride must be at least two")
    calibration = torch.from_numpy(np.load(args.calibration)["coefficients"].astype(np.float32))
    episodes, rejected = [], []
    for root in args.dataset_root:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for directory in episode_paths(root):
            item = load_episode(directory, root, calibration, int(PHASE_END[-1]) + 1)
            if item is None:
                rejected.append(str(directory))
            else:
                episodes.append(item)
    if len(episodes) < args.validation_stride:
        raise ValueError("Not enough strict dynamic episodes")
    episodes.sort(key=lambda item: (item.source_root, item.name))
    counters = [0] * len(OBJECT_NAMES)
    train, val = [], []
    for item in episodes:
        index = counters[item.target]
        counters[item.target] += 1
        (val if index % args.validation_stride == 0 else train).append(item)
    tasks = sorted({item.task for item in episodes})
    vocab = build_vocab(tasks)
    train_cache, val_cache = pack(train, tasks, vocab), pack(val, tasks, vocab)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_cache, args.output_dir / "train.pt")
    torch.save(val_cache, args.output_dir / "val.pt")
    error = torch.linalg.vector_norm(
        torch.cat(
            [
                cache["visual_object_position"].reshape(-1, 3, 3)
                - cache["audit_true_object_position"].reshape(-1, 3, 3)
                for cache in (train_cache, val_cache)
            ]
        ),
        dim=-1,
    )
    summary = {
        "experiment": "G1LANG-contact-gated-direct-joint-dynamic-cache",
        "dataset_roots": [str(root) for root in args.dataset_root],
        "strict_dynamic_episodes": len(episodes),
        "train_episodes": len(train),
        "validation_episodes": len(val),
        "train_rows": len(train_cache["state"]),
        "validation_rows": len(val_cache["state"]),
        "candidate_phase_start_frames": PHASE_START.tolist(),
        "candidate_phase_end_frames": PHASE_END.tolist(),
        "candidate_lookahead_frames": LOOKAHEAD_FRAMES,
        "next_stage_switch_frames": [
            int(PHASE_END[1] - LOOKAHEAD_FRAMES),
            int(PHASE_END[2] - LOOKAHEAD_FRAMES),
        ],
        "labels": "strict MuJoCo dynamically executed approach/pregrasp/contact joint commands",
        "camera_input": "latched initial head RGB colour geometry",
        "privileged_positions_at_inference": False,
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "initial_camera_audit_object_error_m": {
            "mean": float(error.mean()),
            "p95": float(torch.quantile(error, 0.95)),
            "max": float(error.max()),
        },
        "rejected_non_strict_episodes": rejected,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rejected_non_strict_episodes"}, indent=2))


if __name__ == "__main__":
    main()
