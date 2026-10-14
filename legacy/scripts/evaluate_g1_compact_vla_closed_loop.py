#!/usr/bin/env python3
"""Run the compact RGB-language-state policy in MuJoCo closed loop."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import evaluate_g1_language_pick_place as base
from g1_compact_vla import encode_tasks, load_checkpoint
from train_g1_language_world_model import ContactWorldModel, CONTACT, OBJECT, PALM, PHASE


CONTACT_ACTION_DIM = 35
JOINT_ACTION_DIM = 31


def rotation_matrix_from_6d(values: np.ndarray) -> np.ndarray:
    """Project the first two predicted rotation columns onto SO(3)."""
    columns = np.asarray(values, dtype=np.float64).reshape(3, 2)
    first = columns[:, 0]
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        first = np.array((1.0, 0.0, 0.0))
    else:
        first = first / first_norm
    second = columns[:, 1] - first * np.dot(first, columns[:, 1])
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-8:
        fallback = np.array((0.0, 1.0, 0.0))
        if abs(float(np.dot(first, fallback))) > 0.9:
            fallback = np.array((0.0, 0.0, 1.0))
        second = fallback - first * np.dot(first, fallback)
        second_norm = np.linalg.norm(second)
    second = second / second_norm
    third = np.cross(first, second)
    return np.column_stack((first, second, third))


def contact_action_to_joint_chunk(
    contact_chunk: np.ndarray,
    model: mujoco.MjModel,
    simulation_data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    upper_names: list[str],
    palm_ids: list[int],
    current_state: np.ndarray,
    max_iterations: int = 30,
) -> tuple[np.ndarray, dict]:
    """Convert learned Cartesian palm poses to joint targets with MuJoCo IK."""
    if contact_chunk.ndim != 2 or contact_chunk.shape[1] != CONTACT_ACTION_DIM:
        raise ValueError(
            f"Expected [H,{CONTACT_ACTION_DIM}] contact action, got {contact_chunk.shape}"
        )
    if current_state.shape != (JOINT_ACTION_DIM,):
        raise ValueError(f"Expected 31D robot state, got {current_state.shape}")
    selected_names, selected_qpos_ids, lower, upper = base.arm_ik_contract(model)
    selected_qvel_ids = np.asarray(
        [controlled[name]["qvel_id"] for name in selected_names], dtype=np.int32
    )
    selected_upper_indices = np.asarray(
        [upper_names.index(name) for name in selected_names], dtype=np.int32
    )
    if len(selected_names) != 17 or len(palm_ids) != 2:
        raise ValueError("Contact IK expects waist3 + two 7D arms and two palm sites")

    ik_data = mujoco.MjData(model)
    ik_data.qpos[:] = simulation_data.qpos
    ik_data.qvel[:] = simulation_data.qvel
    values = np.clip(ik_data.qpos[selected_qpos_ids], lower, upper)
    joint_chunk = np.repeat(current_state[None, :], len(contact_chunk), axis=0)
    position_errors = []
    orientation_errors = []
    failed_waypoints = 0
    jacobian_position = np.zeros((3, model.nv), dtype=np.float64)
    jacobian_rotation = np.zeros((3, model.nv), dtype=np.float64)

    for step, contact_action in enumerate(contact_chunk):
        target_waist = contact_action[:3].astype(np.float64)
        target_positions = contact_action[3:9].reshape(2, 3).astype(np.float64)
        target_orientations = np.stack(
            (
                rotation_matrix_from_6d(contact_action[9:15]),
                rotation_matrix_from_6d(contact_action[15:21]),
            )
        )
        for _ in range(max_iterations):
            ik_data.qpos[selected_qpos_ids] = values
            mujoco.mj_forward(model, ik_data)
            errors = []
            jacobian_rows = []
            current_position_errors = []
            current_orientation_errors = []
            for palm_index, site_id in enumerate(palm_ids):
                position_error = target_positions[palm_index] - ik_data.site_xpos[site_id]
                current_rotation = ik_data.site_xmat[site_id].reshape(3, 3)
                orientation_error = Rotation.from_matrix(
                    target_orientations[palm_index] @ current_rotation.T
                ).as_rotvec()
                mujoco.mj_jacSite(
                    model,
                    ik_data,
                    jacobian_position,
                    jacobian_rotation,
                    site_id,
                )
                errors.extend((position_error, 0.06 * orientation_error))
                jacobian_rows.extend(
                    (
                        jacobian_position[:, selected_qvel_ids].copy(),
                        0.06 * jacobian_rotation[:, selected_qvel_ids].copy(),
                    )
                )
                current_position_errors.append(float(np.linalg.norm(position_error)))
                current_orientation_errors.append(
                    float(np.linalg.norm(orientation_error))
                )
            if max(current_position_errors) <= 0.003 and max(
                current_orientation_errors
            ) <= 0.04:
                break
            jacobian = np.vstack(jacobian_rows)
            error = np.concatenate(errors)
            waist_jacobian = np.zeros((3, len(values)), dtype=np.float64)
            waist_jacobian[:, :3] = 0.20 * np.eye(3)
            jacobian = np.vstack((jacobian, waist_jacobian))
            error = np.concatenate((error, 0.20 * (target_waist - values[:3])))
            damping = 2e-4
            normal = jacobian.T @ jacobian + damping * np.eye(len(values))
            delta = np.linalg.solve(normal, jacobian.T @ error)
            delta = np.clip(delta, -0.12, 0.12)
            values = np.clip(values + delta, lower + 1e-6, upper - 1e-6)

        ik_data.qpos[selected_qpos_ids] = values
        mujoco.mj_forward(model, ik_data)
        final_position = max(
            float(np.linalg.norm(ik_data.site_xpos[site] - target))
            for site, target in zip(palm_ids, target_positions)
        )
        final_orientation = max(
            float(
                np.linalg.norm(
                    Rotation.from_matrix(
                        target @ ik_data.site_xmat[site].reshape(3, 3).T
                    ).as_rotvec()
                )
            )
            for site, target in zip(palm_ids, target_orientations)
        )
        position_errors.append(final_position)
        orientation_errors.append(final_orientation)
        failed_waypoints += int(final_position > 0.02 or final_orientation > 0.25)
        joint_chunk[step, selected_upper_indices] = values
        joint_chunk[step, 10:17] = contact_action[21:28]
        joint_chunk[step, 24:31] = contact_action[28:35]

    return joint_chunk.astype(np.float32), {
        "waypoints": len(contact_chunk),
        "failed_waypoints": failed_waypoints,
        "max_position_error_m": max(position_errors, default=0.0),
        "mean_position_error_m": float(np.mean(position_errors)),
        "max_orientation_error_rad": max(orientation_errors, default=0.0),
        "mean_orientation_error_rad": float(np.mean(orientation_errors)),
    }


def color_features_from_head(image: torch.Tensor) -> torch.Tensor:
    """Return normalized centroids/areas for red, yellow, and green pixels."""
    image = image[:, :3].clamp(0, 1)
    red = (image[:, 0] > 0.47) & (image[:, 0] > image[:, 1] * 1.25) & (image[:, 0] > image[:, 2] * 1.25)
    yellow = (image[:, 0] > 0.47) & (image[:, 1] > 0.35) & (image[:, 2] < 0.47)
    green = (image[:, 1] > 0.31) & (image[:, 1] > image[:, 0] * 1.25) & (image[:, 1] > image[:, 2] * 1.15)
    height, width = image.shape[-2:]
    rows = []
    for mask in (red, yellow, green):
        count = mask.flatten(1).sum(1).float()
        x_grid = torch.arange(width, device=image.device, dtype=image.dtype).view(1, 1, width)
        y_grid = torch.arange(height, device=image.device, dtype=image.dtype).view(1, height, 1)
        x = (mask.float() * x_grid).sum((1, 2)) / count.clamp_min(1.0) / width
        y = (mask.float() * y_grid).sum((1, 2)) / count.clamp_min(1.0) / height
        area = count / (height * width)
        rows.extend((x, y, area))
    return torch.stack(rows, dim=1)


class CompactPolicy:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        world_model_checkpoint: Path | None = None,
        candidate_count: int = 8,
        planner_noise_scale: float = 0.015,
        hand_checkpoint: Path | None = None,
        visual_calibration: Path | None = None,
        latch_model_object: bool = False,
    ):
        self.device = device
        self.model, self.vocab, self.stats = load_checkpoint(checkpoint, device)
        relative_flag = self.stats.get("target_relative_contact_action")
        self.target_relative_contact_action = bool(
            relative_flag is not None and int(relative_flag.item()) != 0
        )
        palm_residual_flag = self.stats.get("palm_residual_contact_action")
        self.palm_residual_contact_action = bool(
            palm_residual_flag is not None and int(palm_residual_flag.item()) != 0
        )
        lookahead_flag = self.stats.get("palm_lookahead_residual_contact_action")
        self.palm_lookahead_residual_contact_action = bool(
            lookahead_flag is not None and int(lookahead_flag.item()) != 0
        )
        self.visual_grasp_residual_conditioning = bool(
            self.model.use_visual_grasp_residual
        )
        self.hand_model = None
        self.hand_vocab = None
        self.hand_stats = None
        if hand_checkpoint is not None:
            self.hand_model, self.hand_vocab, self.hand_stats = load_checkpoint(
                hand_checkpoint, device
            )
        self.config = SimpleNamespace(
            chunk_size=self.model.chunk_horizon, max_action_dim=self.model.action_dim
        )
        self.last_target_probabilities: list[float] | None = None
        self.last_phase_probabilities: list[float] | None = None
        self.last_active_arm_probabilities: list[float] | None = None
        self.last_object_position: list[list[float]] | None = None
        self.last_action_object_position: list[list[float]] | None = None
        self.last_palm_position: list[list[float]] | None = None
        self.last_action_palm_position: list[list[float]] | None = None
        self.contact_ik_reports: list[dict] = []
        self.history_states: list[torch.Tensor] = []
        self.history_colors: list[torch.Tensor] = []
        self.needs_temporal_history = bool(self.model.use_temporal_history)
        self.latched_active_arm_index: torch.Tensor | None = None
        self.planner_context = None
        self.planner_candidate_count = candidate_count
        self.planner_noise_scale = planner_noise_scale
        self.visual_calibration = None
        self.latch_model_object = latch_model_object
        self.latched_model_object_position: torch.Tensor | None = None
        if visual_calibration is not None:
            calibration = np.load(visual_calibration)
            self.visual_calibration = torch.from_numpy(
                calibration["coefficients"].astype(np.float32)
            ).to(device)
        if self.visual_grasp_residual_conditioning and self.visual_calibration is None:
            raise ValueError(
                "A palm-residual grasp policy requires --visual-calibration; "
                "deployment may not substitute simulator object positions."
            )
        if self.palm_residual_contact_action and not self.visual_grasp_residual_conditioning:
            raise ValueError(
                "Palm-residual action checkpoints must enable visual grasp "
                "residual conditioning."
            )
        if self.palm_lookahead_residual_contact_action and self.model.chunk_horizon != 1:
            raise ValueError("Lookahead residual checkpoints must output exactly one action")
        self.latched_visual_object_position: torch.Tensor | None = None
        self.latched_visual_object_position_raw: torch.Tensor | None = None
        self.planner_selected_nonbaseline = 0
        self.planner_replans = 0
        self.planner_cost_reductions: list[float] = []
        self.world_model = None
        self.world_model_stats = None
        if world_model_checkpoint is not None:
            checkpoint_data = torch.load(world_model_checkpoint, map_location=device, weights_only=False)
            config = checkpoint_data["config"]
            self.world_model = ContactWorldModel(
                hidden_dim=int(config.get("hidden_dim", 512)),
                depth=int(config.get("depth", 3)),
            ).to(device)
            self.world_model.load_state_dict(checkpoint_data["model"])
            self.world_model.eval()
            self.world_model_stats = {
                key: torch.from_numpy(value).to(device)
                for key, value in checkpoint_data["stats"].items()
            }

    def set_planner_context(self, context: dict) -> None:
        self.planner_context = context

    def reset(self) -> None:
        self.last_target_probabilities = None
        self.last_phase_probabilities = None
        self.last_active_arm_probabilities = None
        self.last_object_position = None
        self.last_action_object_position = None
        self.last_palm_position = None
        self.last_action_palm_position = None
        self.contact_ik_reports = []
        self.history_states = []
        self.history_colors = []
        self.latched_active_arm_index = None
        self.planner_context = None
        self.planner_selected_nonbaseline = 0
        self.planner_replans = 0
        self.planner_cost_reductions = []
        self.latched_visual_object_position = None
        self.latched_visual_object_position_raw = None
        self.latched_model_object_position = None

    @torch.no_grad()
    def observe_history(self, batch: dict) -> None:
        """Record one camera/proprioception observation for the temporal encoder."""
        if not self.model.use_temporal_history:
            return
        state = batch["observation.state"].to(self.device)
        state = (state - self.stats["state_mean"]) / self.stats["state_std"]
        image = batch["observation.images.head"].to(self.device)
        color = color_features_from_head(image)
        self.history_states.append(state.detach()[0])
        self.history_colors.append(color.detach()[0])
        limit = self.model.history_length
        del self.history_states[:-limit]
        del self.history_colors[:-limit]

    def _history_inputs(
        self,
        state: torch.Tensor,
        color_features: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.model.use_temporal_history:
            return None, None
        states = list(self.history_states)
        colors = list(self.history_colors)
        if not states:
            states = [state[0].detach()]
            colors = [color_features[0].detach()]
        while len(states) < self.model.history_length:
            states.insert(0, states[0])
            colors.insert(0, colors[0])
        history_state = torch.stack(states[-self.model.history_length :], dim=0).unsqueeze(0)
        history_color = torch.stack(colors[-self.model.history_length :], dim=0).unsqueeze(0)
        return history_state, history_color

    def _calibrated_object_position(
        self, color_features: torch.Tensor
    ) -> torch.Tensor | None:
        if self.visual_calibration is None:
            return None
        if self.latched_visual_object_position is None:
            values = []
            for object_index in range(3):
                z = color_features[:, 3 * object_index : 3 * object_index + 3]
                x, y, area = z.unbind(-1)
                design = torch.stack(
                    (torch.ones_like(x), x, y, area, x * x, y * y, x * y), dim=-1
                )
                values.append(design @ self.visual_calibration[object_index])
            estimate = torch.stack(values, dim=1)
            # Keep the visual estimate in the calibrated table workspace while
            # retaining its learned z coordinate for each object shape.
            estimate[..., 0] = estimate[..., 0].clamp(0.28, 0.68)
            estimate[..., 1] = estimate[..., 1].clamp(-0.42, 0.28)
            estimate[..., 2] = estimate[..., 2].clamp(0.73, 1.05)
            raw_estimate = estimate.reshape(estimate.shape[0], -1)
            self.latched_visual_object_position_raw = raw_estimate.detach()
            self.latched_visual_object_position = (
                (raw_estimate - self.stats["object_mean"]) / self.stats["object_std"]
            ).detach()
        return self.latched_visual_object_position

    def _measured_palm_position(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read palms from robot kinematics, never from object simulator state."""
        if self.planner_context is None:
            raise RuntimeError("Palm-residual decoding requires MuJoCo robot context")
        context = self.planner_context
        raw = torch.tensor(
            np.concatenate(
                [context["data"].site_xpos[site].copy() for site in context["palm_ids"]]
            ),
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 6).expand(batch_size, -1)
        normalized = (
            raw - self.stats["palm_mean"]
        ) / self.stats["palm_std"]
        return raw, normalized

    def _visual_grasp_residual(
        self,
        object_position: torch.Tensor,
        target_index: torch.Tensor,
        current_palm_position: torch.Tensor,
    ) -> torch.Tensor:
        """Form camera-localized target-to-palm residual in training units."""
        object_world = (
            object_position * self.stats["object_std"] + self.stats["object_mean"]
        ).reshape(-1, 3, 3)
        selected = object_world[
            torch.arange(len(object_world), device=self.device), target_index
        ]
        residual = (selected[:, None, :] - current_palm_position.reshape(-1, 2, 3))
        residual = residual.reshape(-1, 6)
        return (
            residual - self.stats["grasp_residual_mean"]
        ) / self.stats["grasp_residual_std"]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise=None):
        images = torch.stack(
            [batch[key].to(self.device) for key in base.CAMERAS], dim=1
        )
        batch_size, camera_count, channels, height, width = images.shape
        embeddings = self.model.vision(
            images.reshape(batch_size * camera_count, channels, height, width)
        ).reshape(batch_size, camera_count, -1)
        token_ids, token_mask = encode_tasks(batch["task"], self.vocab)
        token_ids = token_ids.to(self.device)
        token_mask = token_mask.to(self.device)
        state = batch["observation.state"].to(self.device)
        state = (state - self.stats["state_mean"]) / self.stats["state_std"]
        color_features = color_features_from_head(images[:, 0])
        history_state, history_color = self._history_inputs(state, color_features)
        visual_object_position = self._calibrated_object_position(color_features)
        current_palm_world = current_palm = None
        if self.palm_residual_contact_action:
            current_palm_world, current_palm = self._measured_palm_position(batch_size)
        output = self.model(
            embeddings,
            state,
            token_ids,
            token_mask,
            color_features=color_features,
            history_state=history_state,
            history_color_features=history_color,
            object_position_override=visual_object_position,
            palm_position_override=current_palm,
        )
        model_object_override = None
        if self.latch_model_object:
            if self.latched_model_object_position is None:
                self.latched_model_object_position = output["object_position"].detach()
            model_object_override = self.latched_model_object_position
            if visual_object_position is None:
                output = self.model(
                    embeddings,
                    state,
                    token_ids,
                    token_mask,
                    color_features=color_features,
                    history_state=history_state,
                    history_color_features=history_color,
                    object_position_override=model_object_override,
                    palm_position_override=current_palm,
                )
        # Target-mixture policies use the language head to select one learned
        # object-specific action expert.  Blending all experts with uncertain
        # probabilities at inference produces a physically meaningless hand
        # pose even when the argmax target is correct.
        if (
            self.model.use_target_mixture
            or self.model.use_active_arm_conditioning
            or self.visual_grasp_residual_conditioning
        ):
            target_index = output["target_logits"].argmax(-1)
            target_override = torch.nn.functional.one_hot(
                target_index, num_classes=3
            ).to(dtype=output["target_logits"].dtype)
            if self.latched_active_arm_index is None:
                self.latched_active_arm_index = output["active_arm_logits"].argmax(-1)
            active_arm_index = self.latched_active_arm_index
            active_arm_override = torch.nn.functional.one_hot(
                active_arm_index, num_classes=2
            ).to(dtype=output["active_arm_logits"].dtype)
            action_object_override = (
                visual_object_position
                if visual_object_position is not None
                else model_object_override
            )
            grasp_residual = (
                self._visual_grasp_residual(
                    action_object_override,
                    target_index,
                    current_palm_world,
                )
                if self.visual_grasp_residual_conditioning
                else None
            )
            output = self.model(
                embeddings,
                state,
                token_ids,
                token_mask,
                target_override=target_override,
                active_arm_override=active_arm_override,
                color_features=color_features,
                history_state=history_state,
                history_color_features=history_color,
                object_position_override=action_object_override,
                palm_position_override=current_palm,
                visual_grasp_residual=grasp_residual,
            )
            if self.latch_model_object and visual_object_position is None:
                output = self.model(
                    embeddings,
                    state,
                    token_ids,
                    token_mask,
                    target_override=target_override,
                    active_arm_override=active_arm_override,
                    color_features=color_features,
                    history_state=history_state,
                    history_color_features=history_color,
                    object_position_override=self.latched_model_object_position,
                    palm_position_override=current_palm,
                    visual_grasp_residual=grasp_residual,
                )
        target_probabilities = output["target_logits"].softmax(-1)[0].cpu()
        phase_probabilities = output["phase_logits"].softmax(-1)[0].cpu()
        active_arm_probabilities = output["active_arm_logits"].softmax(-1)[0].cpu()
        self.last_target_probabilities = target_probabilities.tolist()
        self.last_phase_probabilities = phase_probabilities.tolist()
        self.last_active_arm_probabilities = active_arm_probabilities.tolist()
        object_position = (
            output["object_position"] * self.stats["object_std"]
            + self.stats["object_mean"]
        )
        self.last_object_position = object_position[0].reshape(3, 3).detach().cpu().tolist()
        action_object_position = (
            output["action_object_position"] * self.stats["object_std"]
            + self.stats["object_mean"]
        )
        self.last_action_object_position = action_object_position[0].reshape(3, 3).detach().cpu().tolist()
        palm_position = (
            output["palm_position"] * self.stats.get("palm_std", torch.ones(6, device=self.device))
            + self.stats.get("palm_mean", torch.zeros(6, device=self.device))
        )
        action_palm_position = (
            output["action_palm_position"] * self.stats.get("palm_std", torch.ones(6, device=self.device))
            + self.stats.get("palm_mean", torch.zeros(6, device=self.device))
        )
        self.last_palm_position = palm_position[0].reshape(2, 3).detach().cpu().tolist()
        self.last_action_palm_position = action_palm_position[0].reshape(2, 3).detach().cpu().tolist()
        action = (
            output["action"] * self.stats["action_std"] + self.stats["action_mean"]
        )
        if self.model.action_is_delta:
            action = action + batch["observation.state"].to(self.device)[:, None, :]
        if self.model.action_dim == CONTACT_ACTION_DIM:
            if self.palm_residual_contact_action:
                if current_palm_world is None:
                    raise RuntimeError("Palm-residual action has no measured palm pose")
                action = action.clone()
                action[..., 3:9] += current_palm_world[:, None, :]
            elif self.target_relative_contact_action:
                target_index = output["target_logits"].argmax(-1)
                action_target_position = action_object_position.reshape(-1, 3, 3)[
                    torch.arange(batch_size, device=self.device), target_index
                ]
                action = action.clone()
                action[..., 3:6] += action_target_position[:, None, :]
                action[..., 6:9] += action_target_position[:, None, :]
            if self.planner_context is None:
                raise RuntimeError("Contact-aware action decoding requires MuJoCo context")
            context = self.planner_context
            joint_chunk, ik_report = contact_action_to_joint_chunk(
                action[0].detach().cpu().numpy(),
                context["model"],
                context["data"],
                context["controlled"],
                context["upper_names"],
                context["palm_ids"],
                batch["observation.state"][0].detach().cpu().numpy(),
            )
            self.contact_ik_reports.append(ik_report)
            action = torch.from_numpy(joint_chunk).unsqueeze(0).to(self.device)
        elif self.model.action_dim != JOINT_ACTION_DIM:
            raise ValueError(f"Unsupported policy action dimension: {self.model.action_dim}")
        if self.hand_model is not None:
            hand_embeddings = self.hand_model.vision(
                images.reshape(batch_size * camera_count, channels, height, width)
            ).reshape(batch_size, camera_count, -1)
            hand_ids, hand_mask = encode_tasks(batch["task"], self.hand_vocab)
            hand_ids = hand_ids.to(self.device)
            hand_mask = hand_mask.to(self.device)
            hand_state = batch["observation.state"].to(self.device)
            hand_state = (
                hand_state - self.hand_stats["state_mean"]
            ) / self.hand_stats["state_std"]
            hand_output = self.hand_model(
                hand_embeddings,
                hand_state,
                hand_ids,
                hand_mask,
                color_features=color_features,
            )
            if (
                self.hand_model.use_target_mixture
                or self.hand_model.use_active_arm_conditioning
            ):
                hand_target = hand_output["target_logits"].argmax(-1)
                hand_active_arm = hand_output["active_arm_logits"].argmax(-1)
                hand_output = self.hand_model(
                    hand_embeddings,
                    hand_state,
                    hand_ids,
                    hand_mask,
                    target_override=torch.nn.functional.one_hot(
                        hand_target, num_classes=3
                    ).to(dtype=hand_output["target_logits"].dtype),
                    active_arm_override=torch.nn.functional.one_hot(
                        hand_active_arm, num_classes=2
                    ).to(dtype=hand_output["active_arm_logits"].dtype),
                    color_features=color_features,
                )
            hand_action = (
                hand_output["action"] * self.hand_stats["action_std"]
                + self.hand_stats["action_mean"]
            )
            if self.hand_model.action_is_delta:
                hand_action = hand_action + batch["observation.state"].to(
                    self.device
                )[:, None, :]
            # The primary action has already been decoded to the 31D joint
            # contract.  The optional clean-expert checkpoint supplies only
            # the contact-sensitive hand joints.
            action = action.clone()
            action[..., 10:17] = hand_action[..., 10:17]
            action[..., 24:31] = hand_action[..., 24:31]
        if self.world_model is None or self.planner_context is None:
            return action

        # Build the WM latent from proprioception, VLA visual localization and
        # MuJoCo forward kinematics for the palms.  Object positions are never
        # read from the simulator here.
        context = self.planner_context
        controlled = context["controlled"]
        data = context["data"]
        qvel = torch.tensor(
            [[float(data.qvel[item["qvel_id"]]) for name, item in controlled.items() if name in context["upper_names"]]],
            device=self.device,
            dtype=torch.float32,
        )
        palm = torch.tensor(
            np.stack([data.site_xpos[site].copy() for site in context["palm_ids"]]),
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, 6)
        object_estimate = object_position
        phase = output["phase_logits"].softmax(-1)
        phase_onehot = phase
        wm_state = torch.cat((
            batch["observation.state"].to(self.device),
            qvel,
            object_estimate,
            palm,
            torch.zeros((1, 1), device=self.device),
            phase_onehot,
        ), dim=-1)
        wm_state = (wm_state - self.world_model_stats["state_mean"]) / self.world_model_stats["state_std"]
        target_probabilities = output["target_logits"].softmax(-1)
        target_index = int(target_probabilities.argmax(-1).item())
        active_arm_index = int(
            (
                self.latched_active_arm_index
                if self.latched_active_arm_index is not None
                else output["active_arm_logits"].argmax(-1)
            ).item()
        )
        candidate_count = max(2, self.planner_candidate_count)
        base_action = action[0]
        candidates = [base_action]
        current_action = batch["observation.state"][0]
        candidates.append(current_action.repeat(self.model.chunk_horizon, 1))
        generator = torch.Generator(device=self.device)
        if noise is not None:
            generator.manual_seed(int(torch.sum(noise).abs().item() * 1000000) % (2**31 - 1))
        while len(candidates) < candidate_count:
            scale = self.planner_noise_scale * (1.0 + 0.5 * ((len(candidates) - 2) % 3))
            candidates.append(base_action + scale * torch.randn(base_action.shape, device=self.device, generator=generator))
        candidates_t = torch.stack(candidates)
        wm_action = (candidates_t - self.world_model_stats["action_mean"]) / self.world_model_stats["action_std"]
        wm_state_batch = wm_state.expand(candidate_count, -1)
        target_batch = torch.eye(3, device=self.device)[target_index].expand(candidate_count, -1)
        rollout = []
        current = wm_state_batch
        horizon = min(self.model.chunk_horizon, int(getattr(self.world_model, "horizon", self.model.chunk_horizon)))
        for step in range(horizon):
            current = self.world_model(current, wm_action[:, step], target_batch)
            rollout.append(current)
        predicted = torch.stack(rollout, dim=1) * self.world_model_stats["state_std"] + self.world_model_stats["state_mean"]
        palms = predicted[:, :, PALM].reshape(candidate_count, horizon, 2, 3)
        objects = predicted[:, :, OBJECT].reshape(candidate_count, horizon, 3, 3)
        distances = torch.linalg.vector_norm(palms[:, :, active_arm_index] - objects[:, :, target_index], dim=-1)
        contact_penalty = (predicted[:, :, CONTACT] < 0.5).float().mean(-1)
        smoothness = torch.mean((wm_action[:, 1:] - wm_action[:, :-1]) ** 2, dim=(1, 2))
        scores = distances.mean(-1) + 0.04 * distances[:, -1] + 0.03 * contact_penalty + 0.01 * smoothness
        selected = int(scores.argmin().item())
        self.planner_replans += 1
        self.planner_selected_nonbaseline += int(selected != 0)
        self.planner_cost_reductions.append(float(1.0 - scores[selected].item() / max(scores[0].item(), 1e-8)))
        return candidates_t[selected : selected + 1]
        return action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--hand-checkpoint",
        type=Path,
        help="Optional second VLA checkpoint whose 7D hand actions are fused into the primary action chunk.",
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--observation-width", type=int, default=160)
    parser.add_argument("--observation-height", type=int, default=120)
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--position-jitter-m", type=float, default=0.018)
    parser.add_argument(
        "--layout-mode", choices=("slot_jitter", "random"), default="slot_jitter"
    )
    parser.add_argument("--assist-distance-m", type=float, default=0.075)
    parser.add_argument("--action-delta-limit-rad", type=float, default=0.12)
    parser.add_argument("--body-gain-scale", type=float, default=1.5)
    parser.add_argument("--hand-gain-scale", type=float, default=1.5)
    parser.add_argument(
        "--disable-video",
        action="store_true",
        help="Skip EGL video rendering while retaining physics and trajectory logs.",
    )
    parser.add_argument(
        "--save-observation-videos",
        action="store_true",
        help="Save head/wrist camera streams for DAgger diagnostics.",
    )
    parser.add_argument(
        "--preserve-source-camera",
        action="store_true",
        help="Use the camera calibration embedded in the training scene.",
    )
    parser.add_argument("--initial-state-from-train", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument(
        "--visual-calibration",
        type=Path,
        help="Optional fixed-camera color-to-world calibration used as the action spatial input.",
    )
    parser.add_argument(
        "--latch-model-object",
        action="store_true",
        help="Latch the model's first visual object estimate for the episode.",
    )
    parser.add_argument("--world-model-candidates", type=int, default=8)
    parser.add_argument("--planner-noise-scale", type=float, default=0.015)
    parser.add_argument(
        "--allow-assisted-grasp",
        action="store_true",
        help="Diagnostic only; strict mode leaves all equality constraints disabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 3 or args.episodes % 3 != 0:
        raise ValueError("--episodes must be at least three and divisible by three")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    metadata = json.loads((args.train_root / "meta/info.json").read_text())
    expected_names = metadata["features"]["action"]["names"]
    policy = CompactPolicy(
        args.checkpoint,
        args.device,
        world_model_checkpoint=args.world_model_checkpoint,
        candidate_count=args.world_model_candidates,
        planner_noise_scale=args.planner_noise_scale,
        hand_checkpoint=args.hand_checkpoint,
        visual_calibration=args.visual_calibration,
        latch_model_object=args.latch_model_object,
    )
    if policy.palm_lookahead_residual_contact_action and args.replan_steps != 1:
        raise ValueError("Lookahead residual evaluation requires --replan-steps=1")
    args.disable_assisted_grasp = not args.allow_assisted_grasp
    specs = base.evaluation_specs(
        args.episodes, args.seed, args.position_jitter_m, args.layout_mode
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.preserve_source_camera:
        os.environ["G1_PRESERVE_SOURCE_CAMERA"] = "1"
    reports = []
    for spec in specs:
        report = base.run_episode(policy, spec, args, expected_names)
        report["predicted_target_probabilities"] = policy.last_target_probabilities
        report["predicted_phase_probabilities"] = policy.last_phase_probabilities
        report["predicted_active_arm_probabilities"] = (
            policy.last_active_arm_probabilities
        )
        report["predicted_object_position_m"] = policy.last_object_position
        report["action_object_position_m"] = policy.last_action_object_position
        report["predicted_palm_position_m"] = policy.last_palm_position
        report["action_palm_position_m"] = policy.last_action_palm_position
        report["contact_ik"] = {
            "chunks": len(policy.contact_ik_reports),
            "waypoints": int(
                sum(item["waypoints"] for item in policy.contact_ik_reports)
            ),
            "failed_waypoints": int(
                sum(item["failed_waypoints"] for item in policy.contact_ik_reports)
            ),
            "max_position_error_m": max(
                (item["max_position_error_m"] for item in policy.contact_ik_reports),
                default=0.0,
            ),
            "max_orientation_error_rad": max(
                (
                    item["max_orientation_error_rad"]
                    for item in policy.contact_ik_reports
                ),
                default=0.0,
            ),
        }
        report["latched_active_arm_index"] = (
            int(policy.latched_active_arm_index.item())
            if policy.latched_active_arm_index is not None
            else None
        )
        reports.append(report)
    summary = {
        "experiment": "G1LANG-44-compact-multimodal-vla-closed-loop",
        "checkpoint": str(args.checkpoint),
        "hand_checkpoint": str(args.hand_checkpoint) if args.hand_checkpoint else None,
        "episodes": len(reports),
        "camera_inputs": list(base.CAMERAS),
        "language_input": "tokenized task text",
        "state_input_dim": 31,
        "model_action_output_dim": policy.model.action_dim,
        "controller_action_output_dim": JOINT_ACTION_DIM,
        "action_contract": (
            (
            "learned_target_relative_contact35+camera_object_position"
                "_to_mujoco_ik_to_joint31"
                if policy.target_relative_contact_action
                else (
                    "learned_visual_grasp_residual_contact35+camera_target+"
                    "measured_palm_to_mujoco_ik_to_joint31"
                    if policy.palm_residual_contact_action
                    else "learned_cartesian_contact35_to_mujoco_ik_to_joint31"
                )
            )
            if policy.model.action_dim == CONTACT_ACTION_DIM
            else "learned_joint31"
        ),
        "assisted_grasp_enabled": bool(args.allow_assisted_grasp),
        "reports": reports,
        "object_selection_accuracy": float(
            np.mean([item["selected_correct_object"] for item in reports])
        ),
        "put_in_box_success_rate": float(np.mean([item["task_success"] for item in reports])),
        "strict_success_rate": float(np.mean([item["passed"] for item in reports])),
        "wrong_object_grasp_rate": float(
            np.mean(
                [
                    item["grabbed_object"] is not None
                    and not item["selected_correct_object"]
                    for item in reports
                ]
            )
        ),
        "no_grasp_rate": float(np.mean([item["grabbed_object"] is None for item in reports])),
        "mean_joint_limit_violation_fraction": float(
            np.mean([item["joint_limit_violation_fraction"] for item in reports])
        ),
        "mean_action_safety_clip_fraction": float(
            np.mean([item["action_safety_clip_fraction"] for item in reports])
        ),
        "action_delta_limit_rad": args.action_delta_limit_rad,
        "body_gain_scale": args.body_gain_scale,
        "hand_gain_scale": args.hand_gain_scale,
        "privileged_positions_at_inference": False,
        "target_relative_contact_action": policy.target_relative_contact_action,
        "palm_residual_contact_action": policy.palm_residual_contact_action,
        "palm_lookahead_residual_contact_action": (
            policy.palm_lookahead_residual_contact_action
        ),
        "visual_grasp_residual_conditioning": policy.visual_grasp_residual_conditioning,
        "contact_ik_failed_waypoint_rate": (
            sum(item["contact_ik"]["failed_waypoints"] for item in reports)
            / max(sum(item["contact_ik"]["waypoints"] for item in reports), 1)
        ),
        "world_model_planner_enabled": policy.world_model is not None,
        "world_model_candidate_count": policy.planner_candidate_count if policy.world_model is not None else 0,
        "world_model_planner_replans": policy.planner_replans,
        "world_model_nonbaseline_selection_rate": policy.planner_selected_nonbaseline / max(policy.planner_replans, 1),
        "world_model_mean_cost_reduction": float(np.mean(policy.planner_cost_reductions)) if policy.planner_cost_reductions else None,
    }
    summary["passed"] = bool(
        summary["object_selection_accuracy"] >= 0.80
        and summary["put_in_box_success_rate"] >= 0.70
        and summary["strict_success_rate"] >= 0.70
        and summary["wrong_object_grasp_rate"] <= 0.10
        and summary["mean_joint_limit_violation_fraction"] <= 0.01
    )
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
