#!/usr/bin/env python3
"""Strict physical closed-loop gate for the calibrated policy-intervention WM."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from torch.nn import functional as F

import evaluate_g1_language_pick_place as base
from g1_action_token_vla import load_checkpoint as load_vla
from g1_direct_joint_candidate_vla import encode_tasks
from g1_finger_contact_geometry import finger_contact_geom_ids, finger_contact_sites
from g1_policy_intervention_world_model import load_checkpoint as load_world
from g1_task_value_world_model import (
    load_checkpoint as load_task_value_world,
    score_task_value_candidates,
)


def task_value_stable_positive_count(label_distribution: dict) -> int:
    """Support both direct label summaries and nested data-gate summaries."""
    if "stable_grasp_positive" in label_distribution:
        return int(label_distribution.get("stable_grasp_positive", 0))
    total = 0
    for split in ("train", "validation"):
        value = label_distribution.get(split, {})
        if isinstance(value, dict):
            total += int(value.get("stable_grasp_positive", 0))
    return total


class CalibratedInterventionMPCPolicy:
    """Learned WM selection with a fresh RGB/tactile re-observation."""

    def __init__(
        self,
        vla_checkpoint: Path,
        world_checkpoint: Path,
        device: str,
        proposal_count: int,
        limit: float,
        uncertainty_weight: float,
        rank_penalty_m: float,
        min_improvement_m: float,
        task_value_world_checkpoint: Path | None = None,
    ) -> None:
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.vla, self.vocab, self.stats, self.codebook, _ = load_vla(vla_checkpoint, str(self.device))
        self.world, self.world_stats = load_world(world_checkpoint, str(self.device))
        if self.vla.contact_site_dim != self.world.contact_site_dim:
            raise ValueError("VLA and world-model finger-contact contracts disagree")
        self.task_world = None
        self.task_world_stats = None
        self.task_world_label_distribution = None
        if task_value_world_checkpoint is not None:
            self.task_world, self.task_world_stats, self.task_world_label_distribution = load_task_value_world(
                task_value_world_checkpoint, str(self.device)
            )
            if self.vla.contact_site_dim != self.task_world.contact_site_dim:
                raise ValueError("VLA and task-value world-model finger-contact contracts disagree")
            if self.vla.action_chunk != self.task_world.action_chunk:
                raise ValueError("VLA and task-value world-model action chunk contracts disagree")
            if task_value_stable_positive_count(self.task_world_label_distribution) <= 0:
                raise ValueError(
                    "Task-value world-model checkpoint has no stable-grasp positives; "
                    "recollect branch data before closed-loop evaluation."
                )
        self.proposal_count = min(int(proposal_count), self.vla.token_count)
        self.limit = float(limit)
        self.uncertainty_weight = float(uncertainty_weight)
        self.rank_penalty_m = float(rank_penalty_m)
        self.min_improvement_m = float(min_improvement_m)
        self.config = SimpleNamespace(chunk_size=1, max_action_dim=31)
        self.needs_temporal_history = True
        self.reset()

    def reset(self) -> None:
        self.ctx = None
        self.history: deque[dict[str, torch.Tensor | list[str]]] = deque(maxlen=self.vla.history_frames)
        self.candidate_reports: list[dict] = []
        self.last_target_probabilities = None
        self.last_active_arm_probabilities = None
        self.last_object_position = None
        self.last_tactile_contact = None

    def set_planner_context(self, context: dict) -> None:
        self.ctx = context

    def _z(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return (value - self.world_stats[f"{name}_mean"]) / self.world_stats[f"{name}_std"]

    def _task_z(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return (value - self.task_world_stats[f"{name}_mean"]) / self.task_world_stats[f"{name}_std"]

    def _sites(self) -> torch.Tensor:
        return torch.from_numpy(
            finger_contact_sites(self.ctx["model"], self.ctx["data"], self.ctx["finger_contact_geom_ids"])
        ).float().to(self.device).reshape(1, -1)

    def _tactile(self) -> torch.Tensor:
        model, data = self.ctx["model"], self.ctx["data"]
        hand_geoms = set(self.ctx["finger_contact_geom_ids"]) | set(self.ctx["contact_pad_ids"])
        object_bodies = set(self.ctx["object_body_ids"].values())
        active = any(
            (contact.geom1 in hand_geoms and int(model.geom_bodyid[contact.geom2]) in object_bodies)
            or (contact.geom2 in hand_geoms and int(model.geom_bodyid[contact.geom1]) in object_bodies)
            for contact in data.contact[: data.ncon]
        )
        return torch.tensor([[float(active)]], dtype=torch.float32, device=self.device)

    def observe_history(self, batch: dict) -> None:
        data, controlled = self.ctx["data"], self.ctx["controlled"]
        self.history.append(
            {
                "images": torch.stack([batch[key][0].to(self.device) for key in base.CAMERAS], dim=0),
                "state": batch["observation.state"].to(self.device),
                "velocity": torch.tensor(
                    [[data.qvel[controlled[name]["qvel_id"]] for name in self.ctx["upper_names"]]],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "sites": self._sites(),
                "tactile": self._tactile(),
                "task": list(batch["task"]),
            }
        )

    def _stack(self) -> dict[str, torch.Tensor]:
        rows = list(self.history)
        rows = [rows[0]] * (self.vla.history_frames - len(rows)) + rows
        result = {
            name: torch.stack([row[name] for row in rows], dim=1)  # type: ignore[index]
            for name in ("state", "velocity", "sites", "tactile")
        }
        result["images"] = torch.stack([row["images"] for row in rows], dim=0).unsqueeze(0)  # type: ignore[index]
        return result

    def _visible_distance(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        count = len(batch["target"])
        objects = batch["objects"].reshape(count, 3, 3)
        sites = batch["sites"].reshape(count, 2, -1, 3)
        rows = torch.arange(count, device=self.device)
        active_sites = sites[rows, batch["arm"]]
        target = objects[rows, batch["target"]]
        return torch.linalg.vector_norm(active_sites - target[:, None], dim=-1).min(-1).values[:, None]

    def _distance(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
        correction = output["distance_correction_m"].squeeze(-1)
        # At deployment this batch is exactly one VLA proposal set from one
        # observation, so centering keeps absolute geometry calibrated while
        # preserving the learned action-dependent candidate ordering.
        correction = correction - correction.mean()
        return (self._visible_distance(batch).squeeze(-1) + correction).clamp_min(0.0)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise=None) -> torch.Tensor:
        del batch, noise
        observation = self._stack()
        ids, mask = encode_tasks(self.history[-1]["task"], self.vocab)  # type: ignore[index]
        output = self.vla(
            observation["images"],
            (observation["state"] - self.stats["state_mean"]) / self.stats["state_std"],
            (observation["velocity"] - self.stats["velocity_mean"]) / self.stats["velocity_std"],
            (observation["sites"] - self.stats["contact_sites_mean"]) / self.stats["contact_sites_std"],
            observation["tactile"],
            ids.to(self.device),
            mask.to(self.device),
        )
        target = int(output["target_logits"].argmax(-1)[0])
        arm = int(output["active_arm_logits"].argmax(-1)[0])
        objects = output["object_mean"] * self.stats["objects_std"] + self.stats["objects_mean"]
        logits = output["action_token_logits"][0]
        token_ids = logits.topk(self.proposal_count).indices
        candidate_chunks = (
            self.codebook[token_ids] + output["candidate_action_residual"][0, token_ids]
        ).clamp(-self.limit, self.limit)
        raw_delta = candidate_chunks[:, 0]
        state = observation["state"][0, -1]
        previous = torch.as_tensor(self.ctx["previous_action"], dtype=torch.float32, device=self.device)
        lower = torch.as_tensor(self.ctx["action_lower"], dtype=torch.float32, device=self.device)
        upper = torch.as_tensor(self.ctx["action_upper"], dtype=torch.float32, device=self.device)
        requested = (state[None] + raw_delta).clamp(lower, upper)
        command = previous[None] + (requested - previous[None]).clamp(-self.limit, self.limit)
        applied_delta = command - state[None]
        count = len(command)
        wm_batch = {
            "state": state[None].expand(count, -1),
            "velocity": observation["velocity"][0, -1].expand(count, -1),
            "objects": objects.expand(count, -1),
            "sites": observation["sites"][0, -1].expand(count, -1),
            "tactile": observation["tactile"][0, -1].expand(count, -1),
            "action_delta": applied_delta,
            "target": torch.full((count,), target, dtype=torch.long, device=self.device),
            "arm": torch.full((count,), arm, dtype=torch.long, device=self.device),
        }
        members = self.world(
            self._z(wm_batch["state"], "state"),
            self._z(wm_batch["velocity"], "velocity"),
            self._z(wm_batch["objects"], "objects"),
            self._z(wm_batch["sites"], "sites"),
            wm_batch["tactile"],
            self._z(wm_batch["action_delta"], "action_delta"),
            F.one_hot(wm_batch["target"], 3).float(),
            F.one_hot(wm_batch["arm"], 2).float(),
            self._visible_distance(wm_batch),
        )
        distances = torch.stack([self._distance(member, wm_batch) for member in members])
        distance_mean = distances.mean(0)
        distance_std = distances.std(0, correction=0)
        rank_cost = torch.arange(count, dtype=torch.float32, device=self.device) * self.rank_penalty_m
        one_step_cost = distance_mean + self.uncertainty_weight * distance_std + rank_cost
        cost = one_step_cost
        task_diagnostics = None
        if self.task_world is not None:
            task_batch = {
                "state": wm_batch["state"],
                "velocity": wm_batch["velocity"],
                "objects": wm_batch["objects"],
                "sites": wm_batch["sites"],
                "tactile": wm_batch["tactile"],
                "action_chunk": candidate_chunks.reshape(count, -1),
                "target": wm_batch["target"],
                "arm": wm_batch["arm"],
            }
            task_members = self.task_world(
                self._task_z(task_batch["state"], "state"),
                self._task_z(task_batch["velocity"], "velocity"),
                self._task_z(task_batch["objects"], "objects"),
                self._task_z(task_batch["sites"], "sites"),
                task_batch["tactile"],
                self._task_z(task_batch["action_chunk"], "action_chunk"),
                F.one_hot(task_batch["target"], 3).float(),
                F.one_hot(task_batch["arm"], 2).float(),
                self._visible_distance(task_batch),
            )
            cost, task_diagnostics = score_task_value_candidates(
                task_members,
                candidate_chunks,
                uncertainty_weight=self.uncertainty_weight,
                rank_penalty_m=self.rank_penalty_m,
            )
        selected = int(cost.argmin())
        # Do not replace the VLA's highest-probability action for a negligible
        # model advantage. This prevents selector churn at effectively equal
        # one-step distances while retaining meaningful visual feedback.
        if float(cost[selected]) >= float(cost[0] - self.min_improvement_m):
            selected = 0
        tactile = float(observation["tactile"][0, -1])
        if tactile >= 0.5 and self.task_world is None:
            selected = 0
            selection_mode = "measured_contact_handoff_to_vla_top_proposal"
        elif self.task_world is not None:
            selection_mode = "full_chunk_task_value_world_model_mpc"
        else:
            selection_mode = "calibrated_one_step_policy_intervention_world_model_mpc"
        self.last_target_probabilities = output["target_logits"].softmax(-1)[0].cpu().tolist()
        self.last_active_arm_probabilities = output["active_arm_logits"].softmax(-1)[0].cpu().tolist()
        self.last_object_position = objects[0].cpu().reshape(3, 3).tolist()
        self.last_tactile_contact = tactile
        candidate_report = {
            "target_index": target,
            "active_arm_index": arm,
            "selection_mode": selection_mode,
            "selected_rank": selected,
            "selected_token": int(token_ids[selected]),
            "predicted_selected_distance_mm": float(distance_mean[selected] * 1e3),
            "predicted_selected_distance_std_mm": float(distance_std[selected] * 1e3),
            "predicted_top_vla_distance_mm": float(distance_mean[0] * 1e3),
            "candidate_tokens": token_ids.cpu().tolist(),
            "candidate_distance_mm": (distance_mean * 1e3).cpu().tolist(),
            "candidate_distance_std_mm": (distance_std * 1e3).cpu().tolist(),
            "candidate_one_step_cost_mm": (one_step_cost * 1e3).cpu().tolist(),
            "candidate_selector_cost_mm": (cost * 1e3).cpu().tolist(),
            "candidate_mean_abs_delta_mrad": (applied_delta.abs().mean(-1) * 1e3).cpu().tolist(),
        }
        if task_diagnostics is not None:
            candidate_report.update(
                task_value_selected_stable_grasp_prob=float(
                    task_diagnostics["mean_stable_grasp_prob"][selected]
                ),
                task_value_selected_contact_fraction=float(
                    task_diagnostics["mean_contact_fraction"][selected]
                ),
                task_value_selected_lift_mm=float(
                    task_diagnostics["mean_lift_m"][selected] * 1e3
                ),
                task_value_candidate_stable_grasp_prob=task_diagnostics[
                    "mean_stable_grasp_prob"
                ].cpu().tolist(),
                task_value_candidate_contact_fraction=task_diagnostics[
                    "mean_contact_fraction"
                ].cpu().tolist(),
                task_value_candidate_lift_mm=(
                    task_diagnostics["mean_lift_m"] * 1e3
                ).cpu().tolist(),
                task_value_candidate_smoothness_mrad=(
                    task_diagnostics["smoothness_rad"] * 1e3
                ).cpu().tolist(),
            )
        self.candidate_reports.append(candidate_report)
        return command[selected].reshape(1, 1, 31)


def physical_distance_audit(report: dict) -> dict:
    """Offline audit only: compare the deployed WM prediction to MuJoCo geometry."""
    trajectory = np.load(report["trajectory"])
    try:
        states = trajectory["observation_joint_position_rad"]
        objects = trajectory["object_position_m"]
        names = trajectory["joint_names"].astype(str).tolist()
    finally:
        trajectory.close()
    model = mujoco.MjModel.from_xml_path(str(Path(report["trajectory"]).parent / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    baseline_qpos = data.qpos.copy()
    controlled = base.controlled_joints(model)
    finger_ids = finger_contact_geom_ids(model)
    object_bodies = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in base.OBJECT_NAMES}
    target = base.OBJECT_NAMES.index(report["target_object"])
    candidate_rows = report["world_model_candidates"]
    actual = []
    predicted = []
    for frame, (state, object_positions) in enumerate(zip(states, objects)):
        data.qpos[:] = baseline_qpos
        data.qvel[:] = 0.0
        for name, value in zip(names, state):
            data.qpos[controlled[name]["qpos_id"]] = value
        for index, name in enumerate(base.OBJECT_NAMES):
            body = object_bodies[name]
            joint = int(model.body_jntadr[body])
            address = int(model.jnt_qposadr[joint])
            data.qpos[address : address + 3] = object_positions[index]
            data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        sites = finger_contact_sites(model, data, finger_ids).reshape(2, -1, 3)
        row = candidate_rows[min(frame, len(candidate_rows) - 1)]
        arm = int(row["active_arm_index"])
        actual.append(float(np.linalg.norm(sites[arm] - object_positions[target], axis=-1).min()))
        predicted.append(float(row["predicted_selected_distance_mm"]) / 1e3)
    actual_array = np.asarray(actual)
    predicted_array = np.asarray(predicted)
    error = np.abs(predicted_array - actual_array)
    return {
        "frames": len(actual),
        "actual_nearest_finger_target_mm": {
            "initial": float(actual_array[0] * 1e3),
            "final": float(actual_array[-1] * 1e3),
            "minimum": float(actual_array.min() * 1e3),
        },
        "selected_world_model_distance_error_mm": {
            "mean": float(error.mean() * 1e3),
            "p95": float(np.percentile(error, 95) * 1e3),
            "max": float(error.max() * 1e3),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--vla-checkpoint", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--task-value-world-model-checkpoint", type=Path)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260748)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=14.0)
    parser.add_argument("--layout-mode", choices=("slot_jitter", "random"), default="random")
    parser.add_argument("--position-jitter-m", type=float, default=0.018)
    parser.add_argument("--proposal-count", type=int, default=12)
    parser.add_argument("--action-delta-limit-rad", type=float, default=0.08)
    parser.add_argument("--body-gain-scale", type=float, default=5.0)
    parser.add_argument("--hand-gain-scale", type=float, default=6.0)
    parser.add_argument("--uncertainty-weight", type=float, default=0.50)
    parser.add_argument("--rank-penalty-mm", type=float, default=0.75)
    parser.add_argument("--min-improvement-mm", type=float, default=1.5)
    parser.add_argument("--observation-width", type=int, default=160)
    parser.add_argument("--observation-height", type=int, default=120)
    parser.add_argument("--video-width", type=int, default=320)
    parser.add_argument("--video-height", type=int, default=240)
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--save-observation-videos", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 3 or args.episodes % 3:
        raise ValueError("--episodes must be divisible by three")
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.disable_assisted_grasp = True
    args.replan_steps = 1
    names = json.loads((args.train_root / "meta/info.json").read_text())["features"]["action"]["names"]
    policy = CalibratedInterventionMPCPolicy(
        args.vla_checkpoint,
        args.world_model_checkpoint,
        args.device,
        args.proposal_count,
        args.action_delta_limit_rad,
        args.uncertainty_weight,
        args.rank_penalty_mm / 1e3,
        args.min_improvement_mm / 1e3,
        args.task_value_world_model_checkpoint,
    )
    reports = []
    for spec in base.evaluation_specs(args.episodes, args.seed, args.position_jitter_m, args.layout_mode):
        report = base.run_episode(policy, spec, args, names)
        report.update(
            predicted_target_probabilities=policy.last_target_probabilities,
            predicted_active_arm_probabilities=policy.last_active_arm_probabilities,
            camera_object_position_m=policy.last_object_position,
            measured_tactile_contact=policy.last_tactile_contact,
            world_model_candidates=policy.candidate_reports,
        )
        report["physical_world_model_distance_audit"] = physical_distance_audit(report)
        reports.append(report)
    contacts = [report["hand_object_contact_candidate_steps"] > 0 for report in reports]
    language = float(np.mean([
        report["world_model_candidates"][-1]["target_index"] == base.OBJECT_NAMES.index(report["target_object"])
        for report in reports
    ]))
    selector_error = [
        report["physical_world_model_distance_audit"]["selected_world_model_distance_error_mm"]["mean"]
        for report in reports
    ]
    summary = {
        "experiment": "G1LANG-policy-intervention-calibrated-world-model-strict-closed-loop-gate",
        "vla_checkpoint": str(args.vla_checkpoint),
        "world_model_checkpoint": str(args.world_model_checkpoint),
        "task_value_world_model_checkpoint": (
            str(args.task_value_world_model_checkpoint)
            if args.task_value_world_model_checkpoint is not None
            else None
        ),
        "episodes": len(reports),
        "input_contract": "raw RGB x3 + language + state31 + qvel31 + measured finger sites + tactile",
        "action_source": (
            "action-token VLA candidates selected by a full-chunk task-value world model"
            if args.task_value_world_model_checkpoint is not None
            else "action-token VLA candidates selected before contact by a calibrated learned one-step world model"
        ),
        "world_model_rollout": (
            "8-step candidate value prediction with fresh RGB/tactile re-observation after executed step"
            if args.task_value_world_model_checkpoint is not None
            else "one physical-control-step learned rollout, followed by fresh RGB/tactile re-observation"
        ),
        "privileged_positions_at_inference": False,
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "reports": reports,
        "language_target_accuracy": language,
        "real_contact_rate": float(np.mean(contacts)),
        "no_grasp_rate": float(np.mean([report["grabbed_object"] is None for report in reports])),
        "full_pick_place_success_rate": float(np.mean([report["task_success"] for report in reports])),
        "mean_deployed_world_model_distance_error_mm": float(np.mean(selector_error)),
    }
    summary["passed"] = bool(
        summary["language_target_accuracy"] >= 0.80
        and summary["real_contact_rate"] >= 0.70
        and summary["full_pick_place_success_rate"] >= 0.70
        and summary["mean_deployed_world_model_distance_error_mm"] <= 45.0
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
