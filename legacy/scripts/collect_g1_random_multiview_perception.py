#!/usr/bin/env python3
"""Collect evaluator-matched random RGB observations for visual supervision.

This is not a manipulation rollout.  The robot holds its current joint state
for one control interval while the standard strict evaluator renders the same
three cameras used by deployment.  MuJoCo object poses are offline labels only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import evaluate_g1_language_pick_place as base


class HoldPositionPolicy:
    """A data-collection policy that never commands an expert action."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(chunk_size=1, max_action_dim=31)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise=None) -> torch.Tensor:
        del noise
        return batch["observation.state"].reshape(1, 1, 31)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260741)
    parser.add_argument("--observation-width", type=int, default=160)
    parser.add_argument("--observation-height", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 6 or args.episodes % 3:
        raise ValueError("--episodes must be at least six and divisible by three")
    action_names = json.loads((args.train_root / "meta/info.json").read_text())["features"]["action"]["names"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.control_fps = 15
    args.duration_s = 1.0 / args.control_fps
    args.layout_mode = "random"
    args.position_jitter_m = 0.018
    args.video_width = 320
    args.video_height = 240
    args.disable_video = True
    args.save_observation_videos = True
    args.disable_assisted_grasp = True
    args.replan_steps = 1
    args.body_gain_scale = 5.0
    args.hand_gain_scale = 6.0
    args.action_delta_limit_rad = 0.0
    args.device = "cpu"
    policy = HoldPositionPolicy()
    reports = []
    for spec in base.evaluation_specs(args.episodes, args.seed, args.position_jitter_m, "random"):
        report = base.run_episode(policy, spec, args, action_names)
        reports.append(report)
        print(
            f"RANDOM_PERCEPTION={len(reports)}/{args.episodes} "
            f"target={report['target_object']}",
            flush=True,
        )
    summary = {
        "experiment": "G1LANG-random-multiview-perception-supervision",
        "episodes": len(reports),
        "seed": args.seed,
        "input_contract": "evaluator-matched raw head/left-wrist/right-wrist RGB at home state",
        "labels": "offline MuJoCo object positions after one passive control interval",
        "layout_mode": "random",
        "policy_action_source": "hold current measured state; no expert or deployment action",
        "privileged_positions_at_inference": False,
        "reports": reports,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
