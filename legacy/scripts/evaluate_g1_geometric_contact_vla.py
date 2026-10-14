#!/usr/bin/env python3
"""Strict MuJoCo closed-loop evaluation for the geometric local contact VLA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import evaluate_g1_language_pick_place as base
from evaluate_g1_compact_vla_closed_loop import contact_action_to_joint_chunk
from g1_geometric_contact_vla import (
    calibrated_object_positions,
    encode_tasks,
    load_checkpoint,
)


CONTACT_ACTION_DIM = 35
JOINT_ACTION_DIM = 31


def color_features_from_head(image: torch.Tensor) -> torch.Tensor:
    image = image[:, :3].clamp(0, 1)
    red = (image[:, 0] > 0.47) & (image[:, 0] > image[:, 1] * 1.25) & (image[:, 0] > image[:, 2] * 1.25)
    yellow = (image[:, 0] > 0.47) & (image[:, 1] > 0.35) & (image[:, 2] < 0.47)
    green = (image[:, 1] > 0.31) & (image[:, 1] > image[:, 0] * 1.25) & (image[:, 1] > image[:, 2] * 1.15)
    height, width = image.shape[-2:]
    x_grid = torch.arange(width, device=image.device, dtype=image.dtype).view(1, 1, width)
    y_grid = torch.arange(height, device=image.device, dtype=image.dtype).view(1, height, 1)
    values = []
    for mask in (red, yellow, green):
        count = mask.flatten(1).sum(1).float().clamp_min(1.0)
        values.extend(
            (
                (mask.float() * x_grid).sum((1, 2)) / count / width,
                (mask.float() * y_grid).sum((1, 2)) / count / height,
                count / (height * width),
            )
        )
    return torch.stack(values, dim=1)


class GeometricContactPolicy:
    def __init__(self, checkpoint: Path, device: str):
        self.device = device
        self.model, self.vocab, self.stats, self.calibration = load_checkpoint(checkpoint, device)
        self.target_relative_action = bool(
            int(self.stats.get("target_relative_action", torch.tensor(0)).item())
        )
        self.config = SimpleNamespace(chunk_size=1, max_action_dim=CONTACT_ACTION_DIM)
        self.needs_temporal_history = False
        self.planner_context = None
        self.reset()

    def reset(self) -> None:
        self.planner_context = None
        self.latched_camera_objects = None
        self.contact_ik_reports: list[dict] = []
        self.last_target_probabilities = None
        self.last_phase_probabilities = None
        self.last_active_arm_probabilities = None
        self.last_object_position = None
        self.last_action_object_position = None
        self.last_palm_position = None
        self.last_action_palm_position = None

    def set_planner_context(self, context: dict) -> None:
        self.planner_context = context

    def _measured_palms(self) -> torch.Tensor:
        if self.planner_context is None:
            raise RuntimeError("Geometric VLA requires MuJoCo palm kinematics")
        context = self.planner_context
        return torch.tensor(
            np.concatenate([context["data"].site_xpos[site].copy() for site in context["palm_ids"]]),
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 6)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise=None) -> torch.Tensor:
        del noise
        if self.planner_context is None:
            raise RuntimeError("Missing MuJoCo evaluation context")
        state_raw = batch["observation.state"].to(self.device)
        state = (state_raw - self.stats["state_mean"]) / self.stats["state_std"]
        color = color_features_from_head(batch["observation.images.head"].to(self.device))
        # The head camera sees all table objects at reset. Once the torso and
        # active arm occlude that view, reacquiring centroids creates a false
        # target jump. Until physical contact, the objects are static, so hold
        # the first camera-only estimate rather than using simulator state.
        if self.latched_camera_objects is None:
            self.latched_camera_objects = calibrated_object_positions(
                color, self.calibration
            ).reshape(1, 9).detach()
        objects = self.latched_camera_objects
        palms = self._measured_palms()
        token_ids, token_mask = encode_tasks(batch["task"], self.vocab)
        output = self.model(
            state,
            objects,
            palms,
            color,
            token_ids.to(self.device),
            token_mask.to(self.device),
        )
        action = output["action"] * self.stats["action_std"] + self.stats["action_mean"]
        action = torch.maximum(
            torch.minimum(action, self.stats["action_upper"]), self.stats["action_lower"]
        )
        action = action.clone()
        if self.target_relative_action:
            target_index = output["target_logits"].argmax(-1)
            target_position = objects.reshape(-1, 3, 3)[
                torch.arange(len(objects), device=self.device), target_index
            ]
            action[:, 3:6] += target_position
            action[:, 6:9] += target_position
        else:
            action[:, 3:9] += palms
        context = self.planner_context
        joint_action, report = contact_action_to_joint_chunk(
            action[0].detach().cpu().numpy().reshape(1, CONTACT_ACTION_DIM),
            context["model"],
            context["data"],
            context["controlled"],
            context["upper_names"],
            context["palm_ids"],
            state_raw[0].detach().cpu().numpy(),
        )
        self.contact_ik_reports.append(report)
        self.last_target_probabilities = output["target_logits"].softmax(-1)[0].cpu().tolist()
        self.last_phase_probabilities = output["phase_logits"].softmax(-1)[0].cpu().tolist()
        self.last_active_arm_probabilities = output["active_arm_logits"].softmax(-1)[0].cpu().tolist()
        self.last_object_position = objects[0].reshape(3, 3).cpu().tolist()
        self.last_action_object_position = self.last_object_position
        self.last_palm_position = palms[0].reshape(2, 3).cpu().tolist()
        self.last_action_palm_position = action[0, 3:9].reshape(2, 3).cpu().tolist()
        return torch.from_numpy(joint_action).unsqueeze(0).to(self.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--replan-steps", type=int, default=1)
    parser.add_argument("--observation-width", type=int, default=160)
    parser.add_argument("--observation-height", type=int, default=120)
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--position-jitter-m", type=float, default=0.018)
    parser.add_argument("--layout-mode", choices=("slot_jitter", "random"), default="slot_jitter")
    parser.add_argument("--action-delta-limit-rad", type=float, default=0.12)
    parser.add_argument("--body-gain-scale", type=float, default=1.5)
    parser.add_argument("--hand-gain-scale", type=float, default=1.5)
    parser.add_argument("--initial-state-from-train", action="store_true")
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--save-observation-videos", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 3 or args.episodes % 3 != 0:
        raise ValueError("--episodes must be at least three and divisible by three")
    metadata = json.loads((args.train_root / "meta/info.json").read_text())
    expected_names = metadata["features"]["action"]["names"]
    policy = GeometricContactPolicy(args.checkpoint, args.device)
    args.disable_assisted_grasp = True
    specs = base.evaluation_specs(args.episodes, args.seed, args.position_jitter_m, args.layout_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for spec in specs:
        report = base.run_episode(policy, spec, args, expected_names)
        report["predicted_target_probabilities"] = policy.last_target_probabilities
        report["predicted_phase_probabilities"] = policy.last_phase_probabilities
        report["predicted_active_arm_probabilities"] = policy.last_active_arm_probabilities
        report["camera_object_position_m"] = policy.last_object_position
        report["action_object_position_m"] = policy.last_action_object_position
        report["measured_palm_position_m"] = policy.last_palm_position
        report["action_palm_target_m"] = policy.last_action_palm_position
        report["contact_ik"] = {
            "chunks": len(policy.contact_ik_reports),
            "waypoints": int(sum(item["waypoints"] for item in policy.contact_ik_reports)),
            "failed_waypoints": int(sum(item["failed_waypoints"] for item in policy.contact_ik_reports)),
            "max_position_error_m": max((item["max_position_error_m"] for item in policy.contact_ik_reports), default=0.0),
            "max_orientation_error_rad": max((item["max_orientation_error_rad"] for item in policy.contact_ik_reports), default=0.0),
        }
        reports.append(report)
    summary = {
        "experiment": "G1LANG-geometric-local-contact-vla-closed-loop",
        "checkpoint": str(args.checkpoint),
        "episodes": len(reports),
        "camera_inputs": ["observation.images.head"],
        "language_input": "tokenized task text",
        "state_input_dim": 31,
        "model_action_output_dim": CONTACT_ACTION_DIM,
        "controller_action_output_dim": JOINT_ACTION_DIM,
        "action_contract": (
            "learned_camera_target_relative_language_state_contact35_to_mujoco_ik_to_joint31"
            if policy.target_relative_action
            else "learned_camera_geometry_language_state_contact35_to_mujoco_ik_to_joint31"
        ),
        "assisted_grasp_enabled": False,
        "reports": reports,
        "object_selection_accuracy": float(np.mean([item["selected_correct_object"] for item in reports])),
        "put_in_box_success_rate": float(np.mean([item["task_success"] for item in reports])),
        "strict_success_rate": float(np.mean([item["passed"] for item in reports])),
        "wrong_object_grasp_rate": float(np.mean([
            item["grabbed_object"] is not None and not item["selected_correct_object"] for item in reports
        ])),
        "no_grasp_rate": float(np.mean([item["grabbed_object"] is None for item in reports])),
        "mean_joint_limit_violation_fraction": float(np.mean([item["joint_limit_violation_fraction"] for item in reports])),
        "mean_action_safety_clip_fraction": float(np.mean([item["action_safety_clip_fraction"] for item in reports])),
        "privileged_positions_at_inference": False,
        "world_model_planner_enabled": False,
    }
    summary["contact_ik_failed_waypoint_rate"] = sum(
        item["contact_ik"]["failed_waypoints"] for item in reports
    ) / max(sum(item["contact_ik"]["waypoints"] for item in reports), 1)
    summary["passed"] = bool(
        summary["object_selection_accuracy"] >= 0.80
        and summary["put_in_box_success_rate"] >= 0.70
        and summary["strict_success_rate"] >= 0.70
        and summary["wrong_object_grasp_rate"] <= 0.10
        and summary["mean_joint_limit_violation_fraction"] <= 0.01
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
