#!/usr/bin/env python3
"""Strict closed-loop contact gate for the stateless direct-joint VLA and WM-MPC.

The deployed policy reads a latched head-camera RGB observation, language, and
the measured upper-body pose plus palm sites.  It has no scripted clock, no
Cartesian/IK controller, and never reads MuJoCo object positions for planning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import evaluate_g1_language_pick_place as base
from g1_contact_gated_joint_vla import (
    ACTION_DIM,
    CANDIDATE_COUNT,
    PHASE_COUNT,
    TARGET_COUNT,
    encode_tasks,
    load_checkpoint,
)
from g1_geometric_contact_vla import calibrated_object_positions
from train_g1_language_world_model import (
    CONTACT,
    OBJECT,
    PALM,
    STATE_DIM,
    ContactWorldModel,
    rollout,
)


# The contact-gated candidates are expert commands from these physical task
# phases.  This maps a candidate's learned semantic label into the world-model
# state encoding; it is not an external stage scheduler.
WORLD_PHASE_FOR_CANDIDATE = (0, 1, 2, 3)


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


class ContactGatedJointPolicy:
    def __init__(
        self,
        checkpoint: Path,
        world_model_checkpoint: Path,
        device: str,
        control_fps: int,
        wm_horizon: int,
        action_delta_limit: float,
    ):
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model, self.vocab, self.stats, self.calibration = load_checkpoint(checkpoint, str(self.device))
        world_checkpoint = torch.load(world_model_checkpoint, map_location=self.device, weights_only=False)
        if int(world_checkpoint.get("state_dim", STATE_DIM)) != STATE_DIM or int(world_checkpoint.get("action_dim", ACTION_DIM)) != ACTION_DIM:
            raise ValueError("World Model must use the audited 86-D state and 31-D action contract")
        config = world_checkpoint["config"]
        self.world_model = ContactWorldModel(
            hidden_dim=int(config.get("hidden_dim", 512)),
            depth=int(config.get("depth", 3)),
        ).to(self.device)
        self.world_model.load_state_dict(world_checkpoint["model"])
        self.world_model.eval()
        self.world_stats = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in world_checkpoint["stats"].items()
        }
        self.control_fps = int(control_fps)
        self.wm_horizon = int(wm_horizon)
        self.action_delta_limit = float(action_delta_limit)
        self.config = SimpleNamespace(chunk_size=1, max_action_dim=ACTION_DIM)
        self.needs_temporal_history = False
        self.reset()

    def reset(self) -> None:
        self.planner_context = None
        self.latched_camera_objects = None
        self.camera_tracker_reports: list[dict] = []
        self.previous_state = None
        self.candidate_reports: list[dict] = []
        self.last_target_probabilities = None
        self.last_stage_probabilities = None
        self.last_active_arm_probabilities = None
        self.last_object_position = None
        self.last_palm_position = None

    def set_planner_context(self, context: dict) -> None:
        self.planner_context = context

    def _tracked_camera_objects(self, color: torch.Tensor) -> torch.Tensor:
        measured = calibrated_object_positions(color, self.calibration).reshape(1, TARGET_COUNT, 3)
        if self.latched_camera_objects is None:
            self.latched_camera_objects = measured.detach()
            status = "latched_initial_head_rgb"
        else:
            status = "held_after_initial_observation"
        self.camera_tracker_reports.append(
            {
                "status": status,
                "color_area": color[0, 2::3].detach().cpu().tolist(),
                "measured_object_position_m": measured[0].detach().cpu().tolist(),
                "tracked_object_position_m": self.latched_camera_objects[0].detach().cpu().tolist(),
            }
        )
        return self.latched_camera_objects

    def _measured_palms(self) -> torch.Tensor:
        if self.planner_context is None:
            raise RuntimeError("Missing MuJoCo kinematic context")
        return torch.tensor(
            np.concatenate(
                [self.planner_context["data"].site_xpos[site].copy() for site in self.planner_context["palm_ids"]]
            ),
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 6)

    def _bounded(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        context = self.planner_context
        result = np.clip(
            action,
            np.asarray(context["action_lower"], dtype=np.float32),
            np.asarray(context["action_upper"], dtype=np.float32),
        )
        if self.action_delta_limit > 0:
            result = state + np.clip(result - state, -self.action_delta_limit, self.action_delta_limit)
        return result.astype(np.float32)

    def _wm_state(
        self,
        state: torch.Tensor,
        velocity: torch.Tensor,
        objects: torch.Tensor,
        palms: torch.Tensor,
        stage: int,
    ) -> torch.Tensor:
        phase = torch.zeros(PHASE_COUNT, dtype=torch.float32, device=self.device)
        phase[WORLD_PHASE_FOR_CANDIDATE[stage]] = 1.0
        raw = torch.cat(
            (
                state[0], velocity[0], objects.reshape(-1), palms.reshape(-1),
                torch.zeros(1, dtype=torch.float32, device=self.device), phase,
            )
        )
        if raw.shape != (STATE_DIM,):
            raise RuntimeError(f"Unexpected World Model state shape {tuple(raw.shape)}")
        return (raw - self.world_stats["state_mean"]) / self.world_stats["state_std"]

    @torch.no_grad()
    def _select_with_world_model(
        self,
        state: torch.Tensor,
        palms: torch.Tensor,
        objects: torch.Tensor,
        output: dict,
    ) -> np.ndarray:
        target_probabilities = output["target_logits"].softmax(-1)[0]
        target = int(target_probabilities.argmax())
        stage_probabilities = output["next_stage_logits_by_target"][0, target].softmax(-1)
        arm_probabilities = output["active_arm_logits_by_target"][0, target].softmax(-1)
        active_arm = int(arm_probabilities.argmax())
        actions = output["action_candidates"][0, target]
        actions = actions * self.stats["action_std"] + self.stats["action_mean"]
        lower = self.stats["candidate_lower"][target]
        upper = self.stats["candidate_upper"][target]
        actions = torch.maximum(torch.minimum(actions, upper), lower)
        raw_state = state[0].detach().cpu().numpy()
        bounded = np.stack([self._bounded(item.detach().cpu().numpy(), raw_state) for item in actions])
        candidate_tensor = torch.from_numpy(bounded).to(self.device)
        velocity = torch.zeros_like(state) if self.previous_state is None else (state - self.previous_state) * self.control_fps
        self.previous_state = state.detach().clone()
        initial = torch.stack(
            [self._wm_state(state, velocity, objects, palms, stage) for stage in range(CANDIDATE_COUNT)]
        )
        target_onehot = torch.eye(TARGET_COUNT, device=self.device)[target].expand(CANDIDATE_COUNT, -1)
        normalized_actions = (candidate_tensor - self.world_stats["action_mean"]) / self.world_stats["action_std"]
        predicted = rollout(
            self.world_model,
            initial,
            normalized_actions[:, None].expand(-1, self.wm_horizon, -1),
            target_onehot,
        )
        raw = predicted * self.world_stats["state_std"] + self.world_stats["state_mean"]
        predicted_palms = raw[:, -1, PALM].reshape(CANDIDATE_COUNT, 2, 3)
        predicted_objects = raw[:, -1, OBJECT].reshape(CANDIDATE_COUNT, TARGET_COUNT, 3)
        distance = torch.linalg.vector_norm(predicted_palms[:, active_arm] - predicted_objects[:, target], dim=-1)
        contact = raw[:, :, CONTACT].clamp(0.0, 1.0).max(-1).values
        motion = torch.sqrt(torch.mean((candidate_tensor - state[0]) ** 2, dim=-1))
        stage_surprise = -torch.log(stage_probabilities.clamp_min(1e-5))
        # Every term comes from a learned model: learned stage probability,
        # learned dynamics/contact prediction, and measured joint state.
        score = distance + 0.08 * (1.0 - contact) + 0.02 * motion + 0.03 * stage_surprise
        choice = int(score.argmin())
        self.candidate_reports.append(
            {
                "target_index": target,
                "active_arm_index": active_arm,
                "candidate_stage_probabilities": stage_probabilities.detach().cpu().tolist(),
                "candidate_scores": score.detach().cpu().tolist(),
                "candidate_predicted_distance_m": distance.detach().cpu().tolist(),
                "candidate_predicted_contact": contact.detach().cpu().tolist(),
                "candidate_joint_delta_rmse_rad": motion.detach().cpu().tolist(),
                "candidate_stage_surprise": stage_surprise.detach().cpu().tolist(),
                "selected_candidate": choice,
                "candidate_count": CANDIDATE_COUNT,
                "selection_mode": "learned_contact_gated_world_model_mpc",
            }
        )
        self.last_target_probabilities = target_probabilities.cpu().tolist()
        self.last_stage_probabilities = stage_probabilities.cpu().tolist()
        self.last_active_arm_probabilities = arm_probabilities.cpu().tolist()
        return bounded[choice]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise=None) -> torch.Tensor:
        del noise
        if self.planner_context is None:
            raise RuntimeError("Missing MuJoCo evaluation context")
        state = batch["observation.state"].to(self.device)
        normalized_state = (state - self.stats["state_mean"]) / self.stats["state_std"]
        color = color_features_from_head(batch["observation.images.head"].to(self.device))
        objects = self._tracked_camera_objects(color).reshape(1, 9)
        palms = self._measured_palms()
        token_ids, token_mask = encode_tasks(batch["task"], self.vocab)
        output = self.model(
            normalized_state, objects, palms, color,
            token_ids.to(self.device), token_mask.to(self.device),
        )
        action = self._select_with_world_model(state, palms, objects, output)
        self.last_object_position = objects[0].reshape(TARGET_COUNT, 3).cpu().tolist()
        self.last_palm_position = palms[0].reshape(2, 3).cpu().tolist()
        return torch.from_numpy(action[None, None]).to(self.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--replan-steps", type=int, default=1)
    parser.add_argument("--observation-width", type=int, default=160)
    parser.add_argument("--observation-height", type=int, default=120)
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--position-jitter-m", type=float, default=0.018)
    parser.add_argument("--layout-mode", choices=("slot_jitter", "random"), default="random")
    parser.add_argument("--action-delta-limit-rad", type=float, default=0.0)
    parser.add_argument("--body-gain-scale", type=float, default=1.5)
    parser.add_argument("--hand-gain-scale", type=float, default=1.5)
    parser.add_argument("--wm-horizon", type=int, default=1)
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--save-observation-videos", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 3 or args.episodes % 3:
        raise ValueError("--episodes must be at least three and divisible by three")
    if args.wm_horizon < 1:
        raise ValueError("--wm-horizon must be positive")
    metadata = json.loads((args.train_root / "meta/info.json").read_text())
    policy = ContactGatedJointPolicy(
        args.checkpoint, args.world_model_checkpoint, args.device,
        args.control_fps, args.wm_horizon, args.action_delta_limit_rad,
    )
    args.disable_assisted_grasp = True
    specs = base.evaluation_specs(args.episodes, args.seed, args.position_jitter_m, args.layout_mode)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for spec in specs:
        report = base.run_episode(policy, spec, args, metadata["features"]["action"]["names"])
        report["predicted_target_probabilities"] = policy.last_target_probabilities
        report["predicted_next_stage_probabilities"] = policy.last_stage_probabilities
        report["predicted_active_arm_probabilities"] = policy.last_active_arm_probabilities
        report["camera_object_position_m"] = policy.last_object_position
        report["measured_palm_position_m"] = policy.last_palm_position
        report["camera_tracking"] = policy.camera_tracker_reports
        report["world_model_candidates"] = policy.candidate_reports
        reports.append(report)
    contacts = [item["hand_object_contact_candidate_steps"] > 0 for item in reports]
    language_target_accuracy = float(np.mean([
        item["world_model_candidates"][-1]["target_index"] == base.OBJECT_NAMES.index(item["target_object"])
        for item in reports
    ]))
    summary = {
        "experiment": "G1LANG-contact-gated-direct-joint-vla-world-model-mpc-contact-gate",
        "checkpoint": str(args.checkpoint),
        "world_model_checkpoint": str(args.world_model_checkpoint),
        "episodes": len(reports),
        "camera_inputs": ["observation.images.head"],
        "language_input": "tokenized task text",
        "state_input_dim": 31,
        "model_action_output_dim": ACTION_DIM,
        "candidate_layout": "learned target(3) x contact-gated direct-joint candidates(4) x joint31",
        "phase_contract": "learned next-stage logits conditioned on RGB/proprioceptive geometry; no clock, no scripted stage scheduler, no recurrent time counter",
        "action_contract": "learned direct 31D joint target; no Cartesian target and no inverse kinematics",
        "world_model_planner_enabled": True,
        "world_model_state_contract": "latched camera objects + pose-difference velocity + measured palms; no simulator object truth",
        "world_model_horizon": args.wm_horizon,
        "candidate_selection_mode": "learned stage probability + learned world-model distance/contact MPC",
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "privileged_positions_at_inference": False,
        "reports": reports,
        "language_target_accuracy": language_target_accuracy,
        "object_selection_accuracy": language_target_accuracy,
        "real_contact_rate": float(np.mean(contacts)),
        "no_grasp_rate": float(np.mean([item["grabbed_object"] is None for item in reports])),
        "mean_joint_limit_violation_fraction": float(np.mean([item["joint_limit_violation_fraction"] for item in reports])),
        "mean_action_safety_clip_fraction": float(np.mean([item["action_safety_clip_fraction"] for item in reports])),
        "full_pick_place_success_rate": float(np.mean([item["task_success"] for item in reports])),
    }
    summary["contact_gate_passed"] = bool(
        all(contacts)
        and summary["object_selection_accuracy"] >= 0.80
        and summary["mean_joint_limit_violation_fraction"] <= 0.01
    )
    summary["passed"] = summary["contact_gate_passed"]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
