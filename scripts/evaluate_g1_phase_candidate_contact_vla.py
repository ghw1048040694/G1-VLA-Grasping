#!/usr/bin/env python3
"""Strict MuJoCo contact gate for phase-candidate VLA plus learned WM-MPC.

The deployed policy reads head RGB, language and 31-D joint pose.  Camera RGB
is latched before the arm can occlude the table.  Candidate actions are learned
by the VLA, converted through the standard IK contract, and selected only by a
learned contact world model.  No object simulator state, equality constraint,
assisted grasp, magnet, or scripted motion is used at inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import evaluate_g1_language_pick_place as base
from evaluate_g1_compact_vla_closed_loop import (
    contact_action_to_joint_chunk,
    rotation_matrix_from_6d,
)
from g1_geometric_contact_vla import calibrated_object_positions
from g1_phase_candidate_contact_vla import (
    PHASE_COUNT,
    encode_tasks,
    load_checkpoint,
)
from train_g1_language_world_model import (
    ACTION_DIM,
    CONTACT,
    OBJECT,
    PALM,
    PHASE,
    QPOS,
    QVEL,
    STATE_DIM,
    ContactWorldModel,
    rollout,
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


class PhaseCandidateContactPolicy:
    def __init__(
        self,
        checkpoint: Path,
        world_model_checkpoint: Path,
        device: str,
        control_fps: int,
        wm_horizon: int,
        action_delta_limit: float,
        phase_probability_cost: float,
        dynamic_camera_tracking: bool,
        phase3_contact_pad_goal: bool,
        forced_candidate_phase: int | None,
    ):
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model, self.vocab, self.stats, self.calibration = load_checkpoint(checkpoint, str(self.device))
        self.mixed_palm_contract = bool(
            int(self.stats.get("active_target_passive_palm_residual", torch.tensor(0)).item())
        )
        world_checkpoint = torch.load(world_model_checkpoint, map_location=self.device, weights_only=False)
        if int(world_checkpoint.get("state_dim", STATE_DIM)) != STATE_DIM or int(world_checkpoint.get("action_dim", ACTION_DIM)) != ACTION_DIM:
            raise ValueError("World Model must use the audited 86D state and 31D action contract")
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
        self.phase_probability_cost = float(phase_probability_cost)
        self.dynamic_camera_tracking = bool(dynamic_camera_tracking)
        self.phase3_contact_pad_goal = bool(phase3_contact_pad_goal)
        self.forced_candidate_phase = forced_candidate_phase
        self.config = SimpleNamespace(chunk_size=1, max_action_dim=CONTACT_ACTION_DIM)
        self.needs_temporal_history = False
        self.reset()

    def reset(self) -> None:
        self.planner_context = None
        self.latched_camera_objects = None
        self.camera_reference_area = None
        self.camera_tracker_reports: list[dict] = []
        self.previous_state = None
        self.contact_ik_reports: list[dict] = []
        self.candidate_reports: list[dict] = []
        self.last_target_probabilities = None
        self.last_phase_probabilities = None
        self.last_active_arm_probabilities = None
        self.last_object_position = None
        self.last_palm_position = None
        self.last_action_palm_position = None
        self.last_selected_candidate = None

    def _tracked_camera_objects(self, color: torch.Tensor) -> torch.Tensor:
        """Track camera-localized objects while rejecting occluded color masks."""
        measured = calibrated_object_positions(color, self.calibration).reshape(1, 3, 3)
        areas = color[:, 2::3]
        if self.latched_camera_objects is None:
            self.latched_camera_objects = measured.detach()
            self.camera_reference_area = areas.detach().clamp_min(2.5e-4)
            valid = torch.ones_like(areas, dtype=torch.bool)
        elif self.dynamic_camera_tracking:
            valid = areas >= torch.maximum(
                torch.full_like(areas, 2.5e-4), self.camera_reference_area * 0.28
            )
            maximum_step = torch.tensor(
                (0.025, 0.025, 0.015), dtype=measured.dtype, device=self.device
            ).view(1, 1, 3)
            delta = (measured - self.latched_camera_objects).clamp(
                -maximum_step, maximum_step
            )
            filtered = self.latched_camera_objects + 0.40 * delta
            self.latched_camera_objects = torch.where(
                valid[..., None], filtered, self.latched_camera_objects
            ).detach()
        else:
            valid = torch.zeros_like(areas, dtype=torch.bool)
        self.camera_tracker_reports.append(
            {
                "color_area": areas[0].detach().cpu().tolist(),
                "valid": valid[0].to(dtype=torch.int64).detach().cpu().tolist(),
                "measured_object_position_m": measured[0].detach().cpu().tolist(),
                "tracked_object_position_m": self.latched_camera_objects[0]
                .detach()
                .cpu()
                .tolist(),
            }
        )
        return self.latched_camera_objects

    @staticmethod
    def _contact_pad_site_action(action: torch.Tensor, active_arm: int) -> np.ndarray:
        """Convert a learned phase-3 contact-pad goal into a palm-site IK goal."""
        corrected = action.detach().cpu().numpy().astype(np.float64, copy=True)
        orientation_start = 9 + active_arm * 6
        orientation = rotation_matrix_from_6d(
            corrected[orientation_start : orientation_start + 6]
        )
        # In the G1 asset the palm site is 7 cm ahead of the contact pad.
        # The y term accounts for the pads' mirrored 3 mm lateral mounting.
        site_from_pad = np.array(
            (0.070, -0.003 if active_arm == 0 else 0.003, 0.0), dtype=np.float64
        )
        position_start = 3 + active_arm * 3
        corrected[position_start : position_start + 3] += orientation @ site_from_pad
        return corrected.astype(np.float32)

    def set_planner_context(self, context: dict) -> None:
        self.planner_context = context

    def _measured_palms(self) -> torch.Tensor:
        if self.planner_context is None:
            raise RuntimeError("Phase candidate VLA requires MuJoCo palm kinematics")
        return torch.tensor(
            np.concatenate(
                [self.planner_context["data"].site_xpos[site].copy() for site in self.planner_context["palm_ids"]]
            ),
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 6)

    def _wm_state(
        self,
        state_raw: torch.Tensor,
        palms: torch.Tensor,
        objects: torch.Tensor,
        phase: int,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        current = state_raw[0]
        phase_onehot = torch.zeros(8, dtype=torch.float32, device=self.device)
        phase_onehot[int(np.clip(phase, 0, 7))] = 1.0
        raw = torch.cat(
            (
                current,
                velocity,
                objects.reshape(-1),
                palms.reshape(-1),
                torch.zeros(1, dtype=torch.float32, device=self.device),
                phase_onehot,
            )
        )
        if raw.shape != (STATE_DIM,):
            raise RuntimeError(f"Unexpected World Model state shape {tuple(raw.shape)}")
        return (raw - self.world_stats["state_mean"]) / self.world_stats["state_std"]

    def _bounded_joint_candidate(self, joint: np.ndarray) -> np.ndarray:
        context = self.planner_context
        previous = np.asarray(context["previous_action"], dtype=np.float32)
        lower = np.asarray(context["action_lower"], dtype=np.float32)
        upper = np.asarray(context["action_upper"], dtype=np.float32)
        bounded = np.clip(joint, lower, upper)
        if self.action_delta_limit > 0.0:
            bounded = previous + np.clip(
                bounded - previous,
                -self.action_delta_limit,
                self.action_delta_limit,
            )
        return bounded.astype(np.float32)

    @torch.no_grad()
    def _select_with_world_model(
        self,
        state_raw: torch.Tensor,
        palms: torch.Tensor,
        objects: torch.Tensor,
        output: dict[str, torch.Tensor],
    ) -> tuple[np.ndarray, dict]:
        context = self.planner_context
        target_probabilities = output["target_logits"].softmax(-1)[0]
        phase_probabilities = output["phase_logits"].softmax(-1)[0]
        target = int(target_probabilities.argmax())
        active_arm = int(output["active_arm_logits"].argmax(-1)[0])
        target_position = objects.reshape(1, 3, 3)[0, target]
        normalized = output["action_candidates"][0, target]
        actions = normalized * self.stats["action_std"] + self.stats["action_mean"]
        lower = self.stats["candidate_lower"][target]
        upper = self.stats["candidate_upper"][target]
        actions = torch.maximum(torch.minimum(actions, upper), lower).clone()
        if self.mixed_palm_contract:
            for arm in range(2):
                columns = slice(3 + arm * 3, 6 + arm * 3)
                if arm == active_arm:
                    actions[:, columns] += target_position
                else:
                    actions[:, columns] += palms[0, arm * 3 : arm * 3 + 3]
        else:
            actions[:, 3:6] += target_position
            actions[:, 6:9] += target_position

        joints = []
        ik_actions = []
        ik_reports = []
        for phase, action in enumerate(actions):
            ik_action = (
                self._contact_pad_site_action(action, active_arm)
                if phase == 3 and self.phase3_contact_pad_goal
                else action.detach().cpu().numpy().astype(np.float32)
            )
            joint_chunk, report = contact_action_to_joint_chunk(
                ik_action.reshape(1, CONTACT_ACTION_DIM),
                context["model"],
                context["data"],
                context["controlled"],
                context["upper_names"],
                context["palm_ids"],
                state_raw[0].detach().cpu().numpy(),
            )
            joints.append(self._bounded_joint_candidate(joint_chunk[0]))
            ik_actions.append(ik_action)
            ik_reports.append(report)
        joint_tensor = torch.from_numpy(np.stack(joints)).to(self.device)
        current_state = state_raw[0]
        if self.previous_state is None:
            velocity = torch.zeros_like(current_state)
        else:
            velocity = (current_state - self.previous_state) * self.control_fps
        self.previous_state = current_state.detach().clone()

        # The World Model is trained with the phase of the action it observes.
        # Score every VLA candidate under its own phase rather than conditioning
        # all four rollouts on a separate, potentially ambiguous phase classifier.
        initial = torch.stack(
            [
                self._wm_state(state_raw, palms, objects, phase, velocity)
                for phase in range(PHASE_COUNT)
            ]
        )
        target_onehot = torch.eye(3, device=self.device)[target].expand(PHASE_COUNT, -1)
        normalized_actions = (
            joint_tensor - self.world_stats["action_mean"]
        ) / self.world_stats["action_std"]
        repeated = normalized_actions[:, None, :].expand(-1, self.wm_horizon, -1)
        predicted = rollout(self.world_model, initial, repeated, target_onehot)
        raw = predicted * self.world_stats["state_std"] + self.world_stats["state_mean"]
        predicted_palms = raw[:, :, PALM].reshape(PHASE_COUNT, self.wm_horizon, 2, 3)
        predicted_objects = raw[:, :, OBJECT].reshape(PHASE_COUNT, self.wm_horizon, 3, 3)
        distance = torch.linalg.vector_norm(
            predicted_palms[:, :, active_arm] - predicted_objects[:, :, target], dim=-1
        )
        predicted_contact = raw[:, :, CONTACT].clamp(0.0, 1.0)
        failed = torch.tensor(
            [item["failed_waypoints"] for item in ik_reports], device=self.device, dtype=torch.float32
        )
        phase_penalty = -torch.log(phase_probabilities.clamp_min(1e-5))
        motion_cost = torch.sqrt(torch.mean((joint_tensor - state_raw[0]) ** 2, dim=-1))
        score = (
            distance.mean(-1)
            + 0.60 * distance[:, -1]
            + 0.05 * (1.0 - predicted_contact.max(-1).values)
            + self.phase_probability_cost * phase_penalty
            + 0.02 * motion_cost
            + 2.0 * failed
        )
        choice = (
            int(score.argmin())
            if self.forced_candidate_phase is None
            else self.forced_candidate_phase
        )
        detail = {
            "target_index": target,
            "active_arm_index": active_arm,
            "selected_phase": choice,
            "phase_scores": score.detach().cpu().tolist(),
            "phase_predicted_distance_m": distance[:, -1].detach().cpu().tolist(),
            "phase_predicted_contact": predicted_contact.max(-1).values.detach().cpu().tolist(),
            "phase_ik_failed": [int(item["failed_waypoints"]) for item in ik_reports],
            "phase_ik_position_error_m": [
                float(item["max_position_error_m"]) for item in ik_reports
            ],
            "phase_ik_orientation_error_rad": [
                float(item["max_orientation_error_rad"]) for item in ik_reports
            ],
            "phase_ik_site_target_m": np.stack(ik_actions)[:, 3:9]
            .reshape(PHASE_COUNT, 2, 3)
            .tolist(),
            "phase_contact_pad_goal_m": actions[:, 3:9]
            .reshape(PHASE_COUNT, 2, 3)
            .detach()
            .cpu()
            .tolist(),
            "phase_joint_delta_max_rad": torch.max(
                torch.abs(joint_tensor - state_raw[0]), dim=-1
            ).values.detach().cpu().tolist(),
            "candidate_count": PHASE_COUNT,
            "selection_mode": (
                "world_model_mpc"
                if self.forced_candidate_phase is None
                else f"forced_learned_phase_{self.forced_candidate_phase}_data_collection"
            ),
        }
        self.contact_ik_reports.append(ik_reports[choice])
        self.candidate_reports.append(detail)
        self.last_action_palm_position = detail["phase_ik_site_target_m"][choice]
        return joints[choice], detail

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise=None) -> torch.Tensor:
        del noise
        if self.planner_context is None:
            raise RuntimeError("Missing MuJoCo evaluation context")
        state_raw = batch["observation.state"].to(self.device)
        state = (state_raw - self.stats["state_mean"]) / self.stats["state_std"]
        color = color_features_from_head(batch["observation.images.head"].to(self.device))
        objects = self._tracked_camera_objects(color).reshape(1, 9)
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
        joint, selection = self._select_with_world_model(state_raw, palms, objects, output)
        self.last_target_probabilities = output["target_logits"].softmax(-1)[0].cpu().tolist()
        self.last_phase_probabilities = output["phase_logits"].softmax(-1)[0].cpu().tolist()
        self.last_active_arm_probabilities = output["active_arm_logits"].softmax(-1)[0].cpu().tolist()
        self.last_object_position = objects[0].reshape(3, 3).cpu().tolist()
        self.last_palm_position = palms[0].reshape(2, 3).cpu().tolist()
        return torch.from_numpy(joint[None, None, :]).to(self.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=8.0)
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
    parser.add_argument("--wm-horizon", type=int, default=5)
    parser.add_argument("--phase-probability-cost", type=float, default=0.0)
    parser.add_argument("--dynamic-camera-tracking", action="store_true")
    parser.add_argument(
        "--phase3-contact-pad-goal",
        action="store_true",
        help="Interpret phase-3 VLA palm targets as contact-pad rather than palm-site goals.",
    )
    parser.add_argument(
        "--force-candidate-phase",
        type=int,
        choices=range(PHASE_COUNT),
        help="Data-collection only: execute one learned VLA phase candidate without MPC selection.",
    )
    parser.add_argument("--initial-state-from-train", action="store_true")
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--save-observation-videos", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 3 or args.episodes % 3 != 0:
        raise ValueError("--episodes must be at least three and divisible by three")
    if args.wm_horizon < 1:
        raise ValueError("--wm-horizon must be positive")
    metadata = json.loads((args.train_root / "meta/info.json").read_text())
    expected_names = metadata["features"]["action"]["names"]
    policy = PhaseCandidateContactPolicy(
        args.checkpoint,
        args.world_model_checkpoint,
        args.device,
        args.control_fps,
        args.wm_horizon,
        args.action_delta_limit_rad,
        args.phase_probability_cost,
        args.dynamic_camera_tracking,
        args.phase3_contact_pad_goal,
        args.force_candidate_phase,
    )
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
        report["measured_palm_position_m"] = policy.last_palm_position
        report["candidate_action_relative_palm_m"] = policy.last_action_palm_position
        report["camera_tracking"] = policy.camera_tracker_reports
        report["world_model_candidates"] = policy.candidate_reports
        report["contact_ik"] = {
            "chunks": len(policy.contact_ik_reports),
            "waypoints": int(sum(item["waypoints"] for item in policy.contact_ik_reports)),
            "failed_waypoints": int(sum(item["failed_waypoints"] for item in policy.contact_ik_reports)),
            "max_position_error_m": max((item["max_position_error_m"] for item in policy.contact_ik_reports), default=0.0),
            "max_orientation_error_rad": max((item["max_orientation_error_rad"] for item in policy.contact_ik_reports), default=0.0),
        }
        reports.append(report)
    contact_success = [item["hand_object_contact_candidate_steps"] > 0 for item in reports]
    language_target_accuracy = float(
        np.mean(
            [
                item["world_model_candidates"][-1]["target_index"]
                == base.OBJECT_NAMES.index(item["target_object"])
                for item in reports
            ]
        )
    )
    summary = {
        "experiment": "G1LANG-phase-candidate-vla-world-model-mpc-contact-gate",
        "checkpoint": str(args.checkpoint),
        "world_model_checkpoint": str(args.world_model_checkpoint),
        "episodes": len(reports),
        "camera_inputs": ["observation.images.head"],
        "language_input": "tokenized task text",
        "state_input_dim": 31,
        "model_action_output_dim": CONTACT_ACTION_DIM,
        "controller_action_output_dim": JOINT_ACTION_DIM,
        "candidate_layout": "learned target(3) x phase(4) contact actions",
        "action_contract": (
            "learned camera active-target/passive-palm-residual contact35 to MuJoCo IK to joint31"
            if policy.mixed_palm_contract
            else "learned camera target-relative contact35 to MuJoCo IK to joint31"
        ),
        "world_model_planner_enabled": True,
        "world_model_state_contract": "camera-localized objects + pose-difference velocity + measured palms; no simulator object truth",
        "world_model_horizon": args.wm_horizon,
        "world_model_candidate_phase_conditioned": True,
        "phase_probability_cost": args.phase_probability_cost,
        "dynamic_camera_tracking": args.dynamic_camera_tracking,
        "phase3_contact_pad_goal": args.phase3_contact_pad_goal,
        "candidate_selection_mode": (
            "world_model_mpc"
            if args.force_candidate_phase is None
            else f"forced_learned_phase_{args.force_candidate_phase}_data_collection"
        ),
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "privileged_positions_at_inference": False,
        "reports": reports,
        "language_target_accuracy": language_target_accuracy,
        "object_selection_accuracy": language_target_accuracy,
        "real_contact_rate": float(np.mean(contact_success)),
        "no_grasp_rate": float(np.mean([item["grabbed_object"] is None for item in reports])),
        "mean_joint_limit_violation_fraction": float(np.mean([item["joint_limit_violation_fraction"] for item in reports])),
        "mean_action_safety_clip_fraction": float(np.mean([item["action_safety_clip_fraction"] for item in reports])),
        "full_pick_place_success_rate": float(np.mean([item["task_success"] for item in reports])),
    }
    summary["contact_ik_failed_waypoint_rate"] = sum(
        item["contact_ik"]["failed_waypoints"] for item in reports
    ) / max(sum(item["contact_ik"]["waypoints"] for item in reports), 1)
    summary["contact_gate_passed"] = bool(
        all(contact_success)
        and summary["object_selection_accuracy"] >= 0.80
        and summary["mean_joint_limit_violation_fraction"] <= 0.01
    )
    summary["passed"] = summary["contact_gate_passed"]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
