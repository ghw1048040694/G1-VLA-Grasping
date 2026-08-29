#!/usr/bin/env python3
"""Adapt the contact World Model to strict phase-candidate policy rollouts.

The base demonstrations supply the broad contact dynamics prior.  Real VLA
rollouts add the states and joint targets that MPC actually evaluates, so the
planner cannot rely on an expert-only optimistic dynamics distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_g1_language_world_model import (
    ACTION_DIM,
    CONTACT,
    OBJECT,
    PALM,
    PHASE,
    STATE_DIM,
    ContactWorldModel,
    Episode,
    TransitionDataset,
    evaluate,
    load_episodes,
    normalizers,
    rollout,
    weighted_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, default=9)
    parser.add_argument("--rollout-repeat", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def load_policy_rollouts(root: Path) -> list[Episode]:
    episodes = []
    for directory in sorted(root.glob("episode_*")):
        trajectory_path = directory / "trajectory.npz"
        summary_path = directory / "summary.json"
        if not trajectory_path.is_file() or not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        candidates = summary.get("reports")
        # Per-episode summaries are the authoritative source. The root summary
        # is intentionally not used so a partial run cannot enter training.
        if candidates is not None:
            continue
        with np.load(trajectory_path) as arrays:
            qpos = arrays["observation_joint_position_rad"].astype(np.float32)
            qvel = arrays["observation_joint_velocity_rad_s"].astype(np.float32)
            objects = arrays["object_position_m"].reshape(len(qpos), 9).astype(np.float32)
            palms = arrays["palm_position_m"].reshape(len(qpos), 6).astype(np.float32)
            action = arrays["action_joint_position_rad"].astype(np.float32)
            grabbed = arrays["grabbed_object_index"].astype(np.int8)
            target = int(arrays["task_target_index"][0])
        selected = summary.get("world_model_candidates", [])
        if selected and len(selected) != len(qpos):
            raise ValueError(
                f"Candidate count {len(selected)} does not match trajectory {len(qpos)} in {directory}"
            )
        phase = np.zeros((len(qpos), 8), dtype=np.float32)
        if selected:
            ids = np.asarray([item["selected_phase"] for item in selected], dtype=np.int64)
            phase[np.arange(len(qpos)), np.clip(ids, 0, 7)] = 1.0
        else:
            phase[:, 0] = 1.0
        contact = (grabbed >= 0).astype(np.float32)[:, None]
        state = np.concatenate((qpos, qvel, objects, palms, contact, phase), axis=1)
        if state.shape[1] != STATE_DIM or action.shape[1] != ACTION_DIM:
            raise ValueError(f"Unexpected rollout state/action shape {state.shape}/{action.shape}")
        episodes.append(Episode(directory.name, state, action, target))
    if not episodes:
        raise FileNotFoundError(f"No completed rollout episodes under {root}")
    return episodes


def main() -> None:
    args = parse_args()
    if args.rollout_repeat < 1 or args.horizon < 1:
        raise ValueError("rollout repeat and horizon must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    base = load_episodes(args.dataset_root)
    if not 0 < args.train_episodes < len(base):
        raise ValueError("--train-episodes must leave at least one base validation episode")
    base_train, base_val = base[: args.train_episodes], base[args.train_episodes :]
    policy_rollouts = load_policy_rollouts(args.rollout_root)
    train_episodes = [*base_train, *policy_rollouts * args.rollout_repeat]
    stats = normalizers(train_episodes)
    train_set = TransitionDataset(train_episodes, stats, args.horizon)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    model = ContactWorldModel(hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = math.inf
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = samples = 0
        for state, action, next_state, target in train_loader:
            state, action, next_state, target = [value.to(device) for value in (state, action, next_state, target)]
            prediction = rollout(model, state, action + 0.02 * torch.randn_like(action), target)
            loss = weighted_loss(prediction, next_state)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(state)
            samples += len(state)
        base_metrics = evaluate(model, base_val, stats, args.horizon, device)
        policy_metrics = evaluate(model, policy_rollouts, stats, min(5, args.horizon), device)
        metrics = {
            "epoch": epoch + 1,
            "train_loss": total / max(samples, 1),
            "base_validation": base_metrics,
            "onpolicy_rollout": policy_metrics,
        }
        history.append(metrics)
        print("PHASE_CANDIDATE_WORLD_MODEL_EPOCH=" + json.dumps(metrics), flush=True)
        score = (
            base_metrics[f"h{args.horizon}"]["object"]
            + base_metrics[f"h{args.horizon}"]["palm"]
            + policy_metrics[f"h{min(5, args.horizon)}"]["palm"]
        )
        if score < best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "stats": stats,
                    "config": {"hidden_dim": args.hidden_dim, "depth": args.depth},
                    "state_dim": STATE_DIM,
                    "action_dim": ACTION_DIM,
                },
                args.output_dir / "best.pt",
            )
    summary = {
        "experiment": "G1LANG-phase-candidate-onpolicy-world-model",
        "base_dataset_root": str(args.dataset_root),
        "rollout_root": str(args.rollout_root),
        "base_train_episodes": [episode.name for episode in base_train],
        "base_validation_episodes": [episode.name for episode in base_val],
        "onpolicy_rollout_episodes": [episode.name for episode in policy_rollouts],
        "rollout_repeat": args.rollout_repeat,
        "train_transition_count": len(train_set),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_layout": {
            "qpos": [0, 31], "qvel": [31, 62], "object": [62, 71],
            "palm": [71, 77], "contact": [77, 78], "phase_onehot": [78, 86],
        },
        "input_provenance": "strict MuJoCo observed qpos/qvel/object/palm/contact and actually executed joint action; object state is only a World Model training label",
        "privileged_positions_at_inference": False,
        "best_validation_score": best,
        "final": history[-1],
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
