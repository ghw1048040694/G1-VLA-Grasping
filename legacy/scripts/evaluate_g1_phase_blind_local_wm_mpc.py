#!/usr/bin/env python3
"""Strict random physical gate for the phase-free local VLA and world model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

import evaluate_g1_language_pick_place as base
from evaluate_g1_contact_gated_joint_vla import color_features_from_head
from g1_direct_joint_candidate_vla import ACTION_DIM, CANDIDATE_COUNT, TARGET_COUNT, encode_tasks
from g1_geometric_contact_vla import calibrated_object_positions
from g1_finger_contact_geometry import finger_contact_sites
from g1_local_joint_vla import load_checkpoint
from train_g1_phase_blind_world_model import (
    PhaseBlindContactWorldModel,
    rollout,
    state_layout,
)


HEAD_VISIBILITY_AREA = torch.tensor((0.0013020833, 0.00390625, 0.0013020833))


def multiview_color_features(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return calibrated head cues plus synchronized head/wrist RGB cues."""
    head = color_features_from_head(batch["observation.images.head"].to(device))
    views = [
        head,
        color_features_from_head(batch["observation.images.left_wrist"].to(device)),
        color_features_from_head(batch["observation.images.right_wrist"].to(device)),
    ]
    return head, torch.cat(views, dim=-1)


class PhaseBlindLocalPolicy:
    def __init__(self, checkpoint: Path, world_model_checkpoint: Path, device: str, control_fps: int, wm_horizon: int, action_delta_limit: float):
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model, self.vocab, self.stats, self.calibration = load_checkpoint(checkpoint, str(self.device))
        self.action_representation = getattr(
            self.model, "action_representation", "absolute_joint_position"
        )
        value = torch.load(world_model_checkpoint, map_location=self.device, weights_only=False)
        world_action_representation = value.get(
            "action_representation", "absolute_joint_position"
        )
        if self.action_representation != world_action_representation:
            raise ValueError(
                "VLA and world model action representations disagree: "
                f"{self.action_representation} != {world_action_representation}"
            )
        if self.action_representation not in {
            "absolute_joint_position",
            "relative_joint_delta",
        }:
            raise ValueError(
                f"Unsupported action representation: {self.action_representation}"
            )
        self.contact_site_dim = int(value.get("contact_site_dim", 6))
        self.object_slice, self.contact_site_slice, self.contact_slice = state_layout(
            self.contact_site_dim
        )
        self.state_dim = self.contact_slice.stop
        if int(value.get("state_dim", -1)) != self.state_dim or int(value.get("action_dim", -1)) != ACTION_DIM:
            raise ValueError("World-model state/action contract does not match physical finger geometry")
        if int(self.model.contact_site_dim) != self.contact_site_dim:
            raise ValueError("VLA and world model disagree on the finger-geometry input")
        self.world_model = PhaseBlindContactWorldModel(**value["config"]).to(self.device)
        self.world_model.load_state_dict(value["model"]); self.world_model.eval()
        self.world_stats = {key: torch.as_tensor(item, dtype=torch.float32, device=self.device) for key, item in value["stats"].items()}
        self.control_fps = int(control_fps); self.wm_horizon = int(wm_horizon); self.action_delta_limit = float(action_delta_limit)
        self.candidate_offsets = self.stats["candidate_offsets_frames"].detach().cpu().numpy().astype(np.int64)
        if not np.array_equal(self.candidate_offsets, np.asarray((0, 2, 4, 6))):
            raise ValueError("Expected the learned local plan at offsets 0, 2, 4, and 6")
        self.plan_frames = int(self.candidate_offsets[-1] + 1)
        self.plan_scales = np.asarray((0.70, 1.00, 1.20), dtype=np.float32)
        self.config = SimpleNamespace(chunk_size=self.plan_frames, max_action_dim=ACTION_DIM)
        self.needs_temporal_history = True
        self.history_frames = 8
        self.reset()

    def reset(self):
        self.planner_context = None; self.latched_camera_objects = None; self.previous_state = None
        self.camera_tracker_reports = []; self.candidate_reports = []
        self.last_target_probabilities = None; self.last_active_arm_probabilities = None
        self.last_object_position = None; self.last_contact_site_position = None; self.last_tactile_contact = None
        self.causal_history = []; self.pending_observation = None; self.latched_active_arm = None

    def set_planner_context(self, context: dict):
        self.planner_context = context

    def _tracked_camera_objects(self, color: torch.Tensor) -> torch.Tensor:
        measured = calibrated_object_positions(color, self.calibration).reshape(1, TARGET_COUNT, 3)
        area = color[0, 2::3]
        threshold = HEAD_VISIBILITY_AREA.to(device=color.device, dtype=color.dtype)
        visible = bool(torch.all(area >= threshold))
        if self.latched_camera_objects is None and visible:
            # Match the cache contract: use the first complete observation
            # available causally, then retain it if the hand later occludes it.
            self.latched_camera_objects = measured.detach()
            status = "latched_first_causal_head_rgb_observation"
        elif self.latched_camera_objects is None:
            status = "head_geometry_waiting_for_all_objects_visible"
            self.camera_tracker_reports.append({"status": status, "head_all_objects_visible": visible, "color_area": area.detach().cpu().tolist(), "measured_object_position_m": measured[0].detach().cpu().tolist(), "tracked_object_position_m": None})
            return torch.zeros_like(measured)
        else:
            status = "held_after_initial_observation"
        self.camera_tracker_reports.append({"status": status, "head_all_objects_visible": visible, "color_area": area.detach().cpu().tolist(), "measured_object_position_m": measured[0].detach().cpu().tolist(), "tracked_object_position_m": self.latched_camera_objects[0].detach().cpu().tolist()})
        return self.latched_camera_objects

    def _measured_contact_sites(self) -> torch.Tensor:
        if self.planner_context is None: raise RuntimeError("Missing MuJoCo kinematic context")
        values = finger_contact_sites(
            self.planner_context["model"],
            self.planner_context["data"],
            self.planner_context["finger_contact_geom_ids"],
        )
        if values.shape != (self.contact_site_dim,):
            raise RuntimeError("Measured finger contact geometry disagrees with world-model contract")
        return torch.from_numpy(values).to(self.device).reshape(1, -1)

    def _measured_tactile_contact(self) -> torch.Tensor:
        """Causal hand tactile flag with no object pose, colour, or identity."""
        if self.planner_context is None:
            raise RuntimeError("Missing MuJoCo kinematic context")
        tactile_geom_ids = set(self.planner_context["finger_contact_geom_ids"])
        tactile_geom_ids.update(self.planner_context["contact_pad_ids"])
        data = self.planner_context["data"]
        active = any(
            contact.geom1 in tactile_geom_ids or contact.geom2 in tactile_geom_ids
            for contact in data.contact[: data.ncon]
        )
        return torch.tensor([[float(active)]], dtype=torch.float32, device=self.device)

    def _bounded(self, action: np.ndarray, state: np.ndarray) -> np.ndarray:
        result = np.clip(action, np.asarray(self.planner_context["action_lower"], dtype=np.float32), np.asarray(self.planner_context["action_upper"], dtype=np.float32))
        if self.action_delta_limit > 0:
            result = state + np.clip(result - state, -self.action_delta_limit, self.action_delta_limit)
        return result.astype(np.float32)

    def _candidate_plans(self, sparse_actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Turn learned t/t+2/t+4/t+6 commands into causal seven-step plans."""
        if sparse_actions.shape != (len(self.candidate_offsets), ACTION_DIM):
            raise ValueError(f"Unexpected sparse VLA plan {sparse_actions.shape}")
        timeline = np.arange(self.plan_frames, dtype=np.float32)
        nominal = np.stack(
            [
                np.interp(timeline, self.candidate_offsets, sparse_actions[:, joint])
                for joint in range(ACTION_DIM)
            ],
            axis=-1,
        ).astype(np.float32)
        plans = []
        for scale in self.plan_scales:
            desired = (
                state[None] + scale * nominal
                if self.action_representation == "relative_joint_delta"
                else state[None] + scale * (nominal - state[None])
            )
            previous = state.copy()
            bounded = []
            for command in desired:
                previous = self._bounded(command, previous)
                bounded.append(previous)
            plans.append(np.stack(bounded))
        return np.stack(plans)

    def _candidate_kinematic_distance(
        self, plans: np.ndarray, target_position: np.ndarray
    ) -> np.ndarray:
        """Evaluate model proposals by FK against RGB-derived target geometry.

        This uses only the measured robot joint state and the same latched head
        camera geometry supplied to the VLA. It never reads a MuJoCo object body
        pose, contact, task phase, or equality constraint.
        """
        if self.planner_context is None:
            raise RuntimeError("Missing MuJoCo kinematic context")
        model = self.planner_context["model"]
        source = self.planner_context["data"]
        controlled = self.planner_context["controlled"]
        qpos_ids = [controlled[name]["qpos_id"] for name in self.planner_context["upper_names"]]
        data = mujoco.MjData(model)
        data.qpos[:] = source.qpos
        data.qvel[:] = source.qvel
        values = np.empty((len(plans), self.plan_frames), dtype=np.float32)
        for candidate, plan in enumerate(plans):
            for frame, action in enumerate(plan):
                data.qpos[qpos_ids] = action
                mujoco.mj_forward(model, data)
                sites = finger_contact_sites(
                    model, data, self.planner_context["finger_contact_geom_ids"]
                ).reshape(-1, 3)
                values[candidate, frame] = np.linalg.norm(
                    sites - target_position[None], axis=-1
                ).min()
        return values

    def _wm_state(self, state: torch.Tensor, velocity: torch.Tensor, objects: torch.Tensor, contact_sites: torch.Tensor, tactile_contact: torch.Tensor) -> torch.Tensor:
        raw = torch.cat((state[0], velocity[0], objects.reshape(-1), contact_sites.reshape(-1), tactile_contact.reshape(-1)))
        if raw.shape != (self.state_dim,): raise RuntimeError(f"Unexpected phase-blind WM state {tuple(raw.shape)}")
        return (raw - self.world_stats["state_mean"]) / self.world_stats["state_std"]

    @torch.no_grad()
    def _select_with_world_model(self, state: torch.Tensor, velocity: torch.Tensor, contact_sites: torch.Tensor, tactile_contact: torch.Tensor, objects: torch.Tensor, output: dict) -> np.ndarray:
        target_probabilities = output["target_logits"].softmax(-1)[0]; target = int(target_probabilities.argmax())
        arm_probabilities = output["active_arm_logits_by_target"][0, target].softmax(-1)
        predicted_arm = int(arm_probabilities.argmax())
        if self.latched_active_arm is None:
            self.latched_active_arm = predicted_arm
        active_arm = self.latched_active_arm
        sparse_actions = (
            output["action_candidates"][0, target, active_arm] * self.stats["action_std"]
            + self.stats["action_mean"]
        )
        sparse_actions = torch.maximum(
            torch.minimum(sparse_actions, self.stats["candidate_upper"][target, active_arm]),
            self.stats["candidate_lower"][target, active_arm],
        )
        raw_state = state[0].detach().cpu().numpy()
        plans = self._candidate_plans(sparse_actions.detach().cpu().numpy(), raw_state)
        candidate_sources = [f"learned_scale_{scale:.2f}" for scale in self.plan_scales]
        candidate_tensor = torch.from_numpy(plans).to(self.device)
        candidate_count = len(candidate_tensor)
        initial = self._wm_state(state, velocity, objects, contact_sites, tactile_contact).expand(candidate_count, -1)
        target_onehot = torch.eye(TARGET_COUNT, device=self.device)[target].expand(candidate_count, -1)
        world_actions = (
            candidate_tensor - state[0]
            if self.action_representation == "relative_joint_delta"
            else candidate_tensor
        )
        normalized_actions = (
            world_actions - self.world_stats["action_mean"]
        ) / self.world_stats["action_std"]
        prediction = rollout(self.world_model, initial, normalized_actions, target_onehot)
        raw = prediction * self.world_stats["state_std"] + self.world_stats["state_mean"]
        predicted_contact_sites = raw[:, -1, self.contact_site_slice].reshape(
            candidate_count, -1, 3
        )
        del predicted_contact_sites
        predicted_contact_sites = raw[:, :, self.contact_site_slice].reshape(
            candidate_count, self.plan_frames, -1, 3
        )
        predicted_objects = raw[:, :, self.object_slice].reshape(
            candidate_count, self.plan_frames, TARGET_COUNT, 3
        )
        distance_by_step = torch.linalg.vector_norm(
            predicted_contact_sites - predicted_objects[:, :, target, None, :], dim=-1
        ).min(-1).values
        closest_distance = distance_by_step.min(-1).values
        terminal_distance = distance_by_step[:, -1]
        # Keep the contact output for auditing, but never treat it as measured
        # contact or let a hallucinated probability terminate approach motion.
        contact = raw[:, :, self.contact_slice].squeeze(-1).clamp(0.0, 1.0).max(-1).values
        motion = torch.sqrt(torch.mean((candidate_tensor - state[0]).square(), dim=(1, 2)))
        kinematic_distance = self._candidate_kinematic_distance(
            plans,
            objects[0].reshape(TARGET_COUNT, 3)[target].detach().cpu().numpy(),
        )
        kinematic_closest = torch.from_numpy(kinematic_distance.min(-1)).to(self.device)
        kinematic_terminal = torch.from_numpy(kinematic_distance[:, -1]).to(self.device)
        if float(tactile_contact.item()) < 0.5:
            # Before a physical contact is measured, the MPC must prefer a
            # learned proposal whose actual robot geometry continues closing
            # the RGB-derived gap. Once tactile contact exists, the learned
            # world-model score is free to select lift/transport proposals.
            score = (
                0.25 * closest_distance
                + 0.05 * terminal_distance
                + 0.65 * kinematic_closest
                + 0.30 * kinematic_terminal
                + 0.02 * motion
            )
        else:
            score = closest_distance + 0.15 * terminal_distance + 0.02 * motion
        choice = int(score.argmin())
        self.candidate_reports.append({"target_index": target, "active_arm_index": active_arm, "model_active_arm_index": predicted_arm, "active_arm_latched": True, "measured_tactile_contact": float(tactile_contact.item()), "candidate_sources": candidate_sources, "candidate_scores": score.detach().cpu().tolist(), "candidate_predicted_closest_distance_m": closest_distance.detach().cpu().tolist(), "candidate_predicted_terminal_distance_m": terminal_distance.detach().cpu().tolist(), "candidate_kinematic_closest_distance_m": kinematic_closest.detach().cpu().tolist(), "candidate_kinematic_terminal_distance_m": kinematic_terminal.detach().cpu().tolist(), "candidate_predicted_contact": contact.detach().cpu().tolist(), "candidate_joint_delta_rmse_rad": motion.detach().cpu().tolist(), "selected_candidate": choice, "candidate_plan_scales": self.plan_scales.tolist(), "candidate_offsets_frames": self.candidate_offsets.tolist(), "action_representation": self.action_representation, "selection_mode": "learned_vla_world_model_mpc_with_rgb_fk_closing_cost"})
        self.last_target_probabilities = target_probabilities.cpu().tolist(); self.last_active_arm_probabilities = arm_probabilities.cpu().tolist()
        return plans[choice]

    def _prepare_observation(self, batch: dict) -> dict:
        if self.planner_context is None: raise RuntimeError("Missing MuJoCo evaluation context")
        state = batch["observation.state"].to(self.device)
        normalized_state = (state - self.stats["state_mean"]) / self.stats["state_std"]
        head_color, vision_color = multiview_color_features(batch, self.device)
        objects = self._tracked_camera_objects(head_color).reshape(1, 9)
        contact_sites = self._measured_contact_sites()
        tactile_contact = self._measured_tactile_contact()
        velocity = (
            torch.zeros_like(state)
            if self.previous_state is None
            else (state - self.previous_state) * self.control_fps
        )
        ids, mask = encode_tasks(batch["task"], self.vocab)
        return {
            "state": state,
            "normalized_state": normalized_state,
            "velocity": velocity,
            "objects": objects,
            "contact_sites": contact_sites,
            "tactile_contact": tactile_contact,
            "vision_color": vision_color,
            "ids": ids.to(self.device),
            "mask": mask.to(self.device),
        }

    @torch.no_grad()
    def observe_history(self, batch: dict) -> None:
        observation = self._prepare_observation(batch)
        feature = torch.cat((
            observation["normalized_state"],
            (observation["velocity"] - self.stats["velocity_mean"]) / self.stats["velocity_std"],
            observation["objects"],
            observation["contact_sites"],
            observation["tactile_contact"],
            observation["vision_color"],
        ), dim=-1).detach()
        self.causal_history.append(feature)
        self.causal_history = self.causal_history[-self.history_frames:]
        self.pending_observation = observation
        self.previous_state = observation["state"].detach().clone()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise=None) -> torch.Tensor:
        del noise
        if self.pending_observation is None:
            self.observe_history(batch)
        observation = self.pending_observation
        self.pending_observation = None
        history = self.causal_history
        if not history:
            raise RuntimeError("Missing causal VLA observation history")
        history_tensor = torch.cat(
            ([history[0]] * (self.history_frames - len(history))) + history,
            dim=0,
        ).unsqueeze(0)
        action = self._select_with_world_model(
            observation["state"],
            observation["velocity"],
            observation["contact_sites"],
            observation["tactile_contact"],
            observation["objects"],
            self.model(
                observation["normalized_state"],
                (observation["velocity"] - self.stats["velocity_mean"]) / self.stats["velocity_std"],
                observation["objects"],
                observation["contact_sites"],
                observation["tactile_contact"],
                observation["vision_color"],
                observation["ids"],
                observation["mask"],
                history_tensor,
            ),
        )
        self.last_object_position = observation["objects"][0].reshape(TARGET_COUNT, 3).cpu().tolist()
        self.last_contact_site_position = observation["contact_sites"][0].reshape(-1, 3).cpu().tolist()
        self.last_tactile_contact = float(observation["tactile_contact"].item())
        if action.ndim == 1:
            action = action[None, None]
        elif action.ndim == 2:
            action = action[None]
        else:
            raise RuntimeError(f"Unexpected learned action rank {action.ndim}")
        return torch.from_numpy(action).to(self.device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--world-model-checkpoint", type=Path, required=True); parser.add_argument("--train-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3); parser.add_argument("--seed", type=int, default=20260729); parser.add_argument("--control-fps", type=int, default=15); parser.add_argument("--duration-s", type=float, default=8.0); parser.add_argument("--replan-steps", type=int, default=7)
    parser.add_argument("--observation-width", type=int, default=160); parser.add_argument("--observation-height", type=int, default=120); parser.add_argument("--video-width", type=int, default=320); parser.add_argument("--video-height", type=int, default=240); parser.add_argument("--position-jitter-m", type=float, default=0.018); parser.add_argument("--layout-mode", choices=("slot_jitter", "random"), default="random")
    parser.add_argument("--action-delta-limit-rad", type=float, default=0.08); parser.add_argument("--body-gain-scale", type=float, default=5.0); parser.add_argument("--hand-gain-scale", type=float, default=6.0); parser.add_argument("--wm-horizon", type=int, default=1); parser.add_argument("--disable-video", action="store_true"); parser.add_argument("--save-observation-videos", action="store_true"); parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 3 or args.episodes % 3: raise ValueError("--episodes must be at least three and divisible by three")
    metadata = json.loads((args.train_root / "meta/info.json").read_text())
    policy = PhaseBlindLocalPolicy(args.checkpoint, args.world_model_checkpoint, args.device, args.control_fps, args.wm_horizon, args.action_delta_limit_rad)
    args.disable_assisted_grasp = True; specs = base.evaluation_specs(args.episodes, args.seed, args.position_jitter_m, args.layout_mode); args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for spec in specs:
        report = base.run_episode(policy, spec, args, metadata["features"]["action"]["names"])
        report["predicted_target_probabilities"] = policy.last_target_probabilities; report["predicted_active_arm_probabilities"] = policy.last_active_arm_probabilities; report["camera_object_position_m"] = policy.last_object_position; report["measured_finger_contact_site_position_m"] = policy.last_contact_site_position; report["measured_tactile_contact"] = policy.last_tactile_contact; report["camera_tracking"] = policy.camera_tracker_reports; report["world_model_candidates"] = policy.candidate_reports
        reports.append(report)
    contacts = [report["hand_object_contact_candidate_steps"] > 0 for report in reports]
    language_accuracy = float(np.mean([report["world_model_candidates"][-1]["target_index"] == base.OBJECT_NAMES.index(report["target_object"]) for report in reports]))
    summary = {"experiment": "G1LANG-phase-blind-local-vla-world-model-mpc-contact-gate", "checkpoint": str(args.checkpoint), "world_model_checkpoint": str(args.world_model_checkpoint), "episodes": len(reports), "camera_inputs": ["observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"], "language_input": "tokenized task text", "state_input_dim": 31, "model_action_output_dim": ACTION_DIM, "candidate_layout": "learned target(3) x local dynamic future candidates(0/2/4/6 frames) x joint31", "phase_contract": "none; no stage input, stage output, recurrent clock, or scripted scheduler", "world_model_state_contract": "head-camera objects + pose-difference velocity + sampled physical finger collision geometry + predicted contact; no MuJoCo object truth and no task phase", "candidate_selection_mode": "phase-blind learned world-model finger-geometry distance MPC; contact prediction audit-only", "action_delta_limit_rad": args.action_delta_limit_rad, "assisted_grasp_enabled": False, "equality_constraints": False, "magnet_enabled": False, "privileged_positions_at_inference": False, "reports": reports, "language_target_accuracy": language_accuracy, "object_selection_accuracy": language_accuracy, "real_contact_rate": float(np.mean(contacts)), "no_grasp_rate": float(np.mean([report["grabbed_object"] is None for report in reports])), "mean_joint_limit_violation_fraction": float(np.mean([report["joint_limit_violation_fraction"] for report in reports])), "mean_action_safety_clip_fraction": float(np.mean([report["action_safety_clip_fraction"] for report in reports])), "full_pick_place_success_rate": float(np.mean([report["task_success"] for report in reports]))}
    summary["contact_gate_passed"] = bool(all(contacts) and summary["object_selection_accuracy"] >= 0.80 and summary["mean_joint_limit_violation_fraction"] <= 0.01); summary["passed"] = summary["contact_gate_passed"]
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
