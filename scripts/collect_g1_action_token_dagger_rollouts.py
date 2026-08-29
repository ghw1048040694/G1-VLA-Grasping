#!/usr/bin/env python3
"""Collect pure on-policy action-token rollouts for strict DAgger recovery.

The policy receives only its deployment inputs.  No physical branch oracle is
used: at every control tick it executes the highest-logit action token from the
checkpoint selected by the predicted language target.  MuJoCo object state is
saved only as an offline teacher/audit label for later recovery collection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import evaluate_g1_language_pick_place as base
from collect_g1_policy_intervention_world_model_data import (
    TargetRoutedActionTokenProposals,
    tactile_contact,
)
from g1_finger_contact_geometry import finger_contact_sites


class TargetRoutedTop1Policy:
    """LeRobot evaluator adapter for language-routed top-1 action tokens."""

    def __init__(self, checkpoints: list[Path], device: str, limit: float):
        self.proposals = TargetRoutedActionTokenProposals(checkpoints, device, 1, limit)
        self.device = self.proposals.device
        self.config = SimpleNamespace(chunk_size=1, max_action_dim=31)
        self.needs_temporal_history = True
        self.ctx = None
        self.last_state = None
        self.decisions: list[dict] = []

    def reset(self) -> None:
        self.proposals.reset()
        self.ctx = None
        self.last_state = None
        self.decisions = []

    def set_planner_context(self, context: dict) -> None:
        self.ctx = context

    def observe_history(self, batch: dict) -> None:
        if self.ctx is None:
            raise RuntimeError("Planner context is unavailable")
        data, controlled = self.ctx["data"], self.ctx["controlled"]
        velocity = np.asarray(
            [data.qvel[controlled[name]["qvel_id"]] for name in self.ctx["upper_names"]],
            dtype=np.float32,
        )
        sites = finger_contact_sites(
            self.ctx["model"], data, self.ctx["finger_contact_geom_ids"]
        ).astype(np.float32)
        hand_geoms = set(self.ctx["finger_contact_geom_ids"]) | set(self.ctx["contact_pad_ids"])
        object_bodies = set(self.ctx["object_body_ids"].values())
        tactile = tactile_contact(self.ctx["model"], data, hand_geoms, object_bodies)
        self.proposals.observe(batch, velocity, sites, tactile)
        self.last_state = batch["observation.state"].to(self.device)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise=None) -> torch.Tensor:
        del batch, noise
        if self.last_state is None:
            raise RuntimeError("No policy observation is available")
        prediction = self.proposals.predict()
        delta = torch.from_numpy(prediction["deltas"][0]).to(self.device)
        action = self.last_state[0] + delta
        encoded_token = int(prediction["token_ids"][0])
        self.decisions.append(
            {
                "target_index": int(prediction["target"]),
                "active_arm_index": int(prediction["arm"]),
                "checkpoint_target_route": encoded_token // 10_000,
                "token_id": encoded_token % 10_000,
                "mean_abs_delta_mrad": float(delta.abs().mean() * 1e3),
            }
        )
        return action.reshape(1, 1, 31)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument(
        "--vla-checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Repeat exactly three times in red, yellow, green target order.",
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument(
        "--target",
        choices=base.OBJECT_NAMES,
        action="append",
        help="Keep only the requested target rollout(s); repeat for multiple targets.",
    )
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--layout-mode", choices=("slot_jitter", "random"), default="random")
    parser.add_argument("--position-jitter-m", type=float, default=0.018)
    parser.add_argument("--action-delta-limit-rad", type=float, default=0.08)
    parser.add_argument("--body-gain-scale", type=float, default=5.0)
    parser.add_argument("--hand-gain-scale", type=float, default=6.0)
    parser.add_argument("--observation-width", type=int, default=96)
    parser.add_argument("--observation-height", type=int, default=64)
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = list(args.vla_checkpoint)
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if len(checkpoints) != 3:
        raise ValueError("Repeat --vla-checkpoint exactly three times: red, yellow, green")
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed rollout: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.disable_assisted_grasp = True
    args.disable_video = True
    args.save_observation_videos = True
    args.replan_steps = 1
    names = json.loads((args.train_root / "meta/info.json").read_text())["features"]["action"]["names"]
    policy = TargetRoutedTop1Policy(checkpoints, args.device, args.action_delta_limit_rad)
    reports = []
    requested_targets = set(args.target or base.OBJECT_NAMES)
    requested_specs = args.episodes * 3 if args.target else args.episodes
    specs = [
        spec for spec in base.evaluation_specs(
            requested_specs, args.seed, args.position_jitter_m, args.layout_mode
        )
        if spec["target_object"] in requested_targets
    ][: args.episodes]
    for output_index, spec in enumerate(specs):
        spec = dict(spec)
        spec["episode_index"] = output_index
        report = base.run_episode(policy, spec, args, names)
        report["policy_decisions"] = list(policy.decisions)
        reports.append(report)
    target_matches = [
        decision["target_index"] == base.OBJECT_NAMES.index(report["target_object"])
        for report in reports
        for decision in report["policy_decisions"]
    ]
    summary = {
        "experiment": "G1LANG-action-token-language-router-pure-policy-DAgger-rollout",
        "checkpoints_by_target": [str(path) for path in checkpoints],
        "episodes": len(reports),
        "requested_targets": sorted(requested_targets),
        "input_contract": "raw RGB x3 + language + state31 + qvel31 + measured finger sites + tactile",
        "action_source": "language-target-routed highest-logit action token; no branch oracle",
        "teacher_labels_present": False,
        "privileged_positions_at_inference": False,
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "language_target_accuracy": float(np.mean(target_matches)),
        "real_contact_episodes": int(sum(r["hand_object_contact_candidate_steps"] > 0 for r in reports)),
        "full_pick_place_successes": int(sum(r["task_success"] for r in reports)),
        "reports": reports,
    }
    summary["passed"] = bool(summary["language_target_accuracy"] >= 0.99)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
