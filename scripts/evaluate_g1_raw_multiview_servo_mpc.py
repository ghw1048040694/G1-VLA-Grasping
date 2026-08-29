#!/usr/bin/env python3
"""Strict MuJoCo gate for raw-RGB VLA action-prior + learned WM CEM-MPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from torch.nn import functional as F

import evaluate_g1_language_pick_place as base
from g1_direct_joint_candidate_vla import encode_tasks
from g1_finger_contact_geometry import finger_contact_sites
from g1_raw_multiview_servo_vla import load_checkpoint as load_vla
from g1_raw_multiview_servo_world_model import load_checkpoint as load_world


ACTION_DIM = 31
TARGET_COUNT = 3


class RawVisualServoMPCPolicy:
    """A learned visual action prior refined by a learned CEM world model."""

    def __init__(self, vla_checkpoint: Path, world_checkpoint: Path, device: str, horizon: int, population: int, iterations: int, delta_limit: float):
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model, self.vocab, self.stats = load_vla(vla_checkpoint, str(self.device))
        self.world_model, self.world_stats = load_world(world_checkpoint, str(self.device))
        if self.model.contact_site_dim != self.world_model.contact_site_dim:
            raise ValueError("VLA and world model finger-contact contracts disagree")
        self.horizon = int(horizon); self.population = int(population); self.iterations = int(iterations); self.delta_limit = float(delta_limit)
        self.config = SimpleNamespace(chunk_size=1, max_action_dim=ACTION_DIM)
        self.needs_temporal_history = True
        self.reset()

    def reset(self):
        self.planner_context = None; self.previous_state = None; self.observation = None
        self.candidate_reports = []; self.last_target_probabilities = None; self.last_active_arm_probabilities = None
        self.last_object_position = None; self.last_contact_site_position = None; self.last_tactile_contact = None
        self.object_belief = None; self.last_object_state_source = None

    def set_planner_context(self, context: dict): self.planner_context = context

    def _contact_sites(self) -> torch.Tensor:
        if self.planner_context is None: raise RuntimeError("Missing MuJoCo context")
        value = finger_contact_sites(self.planner_context["model"], self.planner_context["data"], self.planner_context["finger_contact_geom_ids"])
        return torch.from_numpy(value).float().to(self.device).reshape(1, -1)

    def _tactile(self) -> torch.Tensor:
        if self.planner_context is None: raise RuntimeError("Missing MuJoCo context")
        geoms = set(self.planner_context["finger_contact_geom_ids"]); geoms.update(self.planner_context["contact_pad_ids"])
        object_bodies = set(self.planner_context["object_body_ids"].values())
        model = self.planner_context["model"]
        data = self.planner_context["data"]
        active = any(
            (contact.geom1 in geoms and int(model.geom_bodyid[contact.geom2]) in object_bodies)
            or (contact.geom2 in geoms and int(model.geom_bodyid[contact.geom1]) in object_bodies)
            for contact in data.contact[:data.ncon]
        )
        return torch.tensor([[float(active)]], device=self.device)

    def observe_history(self, batch: dict):
        state = batch["observation.state"].to(self.device)
        if self.planner_context is None: raise RuntimeError("Missing MuJoCo context")
        data = self.planner_context["data"]
        controlled = self.planner_context["controlled"]
        velocity = torch.tensor(
            [[data.qvel[controlled[name]["qvel_id"]] for name in self.planner_context["upper_names"]]],
            dtype=torch.float32,
            device=self.device,
        )
        images = torch.stack([batch[key][0].to(self.device) for key in base.CAMERAS], dim=0).unsqueeze(0)
        ids, mask = encode_tasks(batch["task"], self.vocab)
        self.observation = {
            "images": images, "state": state, "velocity": velocity,
            "sites": self._contact_sites(), "tactile": self._tactile(),
            "ids": ids.to(self.device), "mask": mask.to(self.device),
        }
        self.previous_state = state.detach().clone()

    def _z(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return (value - self.world_stats[f"{name}_mean"]) / self.world_stats[f"{name}_std"]

    def _raw(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return value * self.world_stats[f"{name}_std"] + self.world_stats[f"{name}_mean"]

    @torch.no_grad()
    def _advance_object_belief(self, observation: dict, objects: torch.Tensor, action: torch.Tensor, target: int) -> None:
        """Carry the learned object state through occlusion after real contact."""
        onehot = F.one_hot(torch.tensor([target], device=self.device), TARGET_COUNT).float()
        transition = self.world_model(
            self._z(observation["state"], "state"),
            self._z(observation["velocity"], "velocity"),
            self._z(objects, "objects"),
            self._z(observation["sites"], "contact_sites"),
            observation["tactile"],
            self._z(action.reshape(1, -1), "action"),
            onehot,
        )
        self.object_belief = self._raw(transition["objects"], "objects").detach()

    @torch.no_grad()
    def _rollout_cost(self, state, velocity, objects, sites, tactile, action_sequences, target: int, active_arm: int) -> tuple[torch.Tensor, dict]:
        count = len(action_sequences)
        current = {
            "state": self._z(state, "state").expand(count, -1),
            "velocity": self._z(velocity, "velocity").expand(count, -1),
            "objects": self._z(objects, "objects").expand(count, -1),
            "sites": self._z(sites, "contact_sites").expand(count, -1),
            "tactile": tactile.expand(count, -1),
        }
        onehot = F.one_hot(torch.full((count,), target, device=self.device), TARGET_COUNT).float()
        distance_rows = []; contact_rows = []; object_rows = []
        for step in range(self.horizon):
            out = self.world_model(current["state"], current["velocity"], current["objects"], current["sites"], current["tactile"], self._z(action_sequences[:, step], "action"), onehot)
            current = {"state": out["state"], "velocity": out["velocity"], "objects": out["objects"], "sites": out["contact_sites"], "tactile": out["contact_logit"].sigmoid()}
            object_rows.append(self._raw(current["objects"], "objects").reshape(count, TARGET_COUNT, 3))
            site_rows = self._raw(current["sites"], "contact_sites").reshape(count, 2, -1, 3)
            active_sites = site_rows[:, active_arm]
            distance_rows.append(torch.linalg.vector_norm(active_sites - object_rows[-1][:, target, None], dim=-1).min(-1).values)
            contact_rows.append(current["tactile"].squeeze(-1))
        distance = torch.stack(distance_rows, dim=1); contact = torch.stack(contact_rows, dim=1); objects_out = torch.stack(object_rows, dim=1)
        motion = action_sequences.square().mean((1, 2)).sqrt()
        if float(tactile.item()) < 0.5:
            cost = distance.min(1).values + 0.35 * distance[:, -1] - 0.08 * contact.max(1).values + 0.025 * motion
        else:
            box = torch.tensor(base.BIN_POSITION_M, dtype=torch.float32, device=self.device)
            box_distance = torch.linalg.vector_norm(objects_out[:, -1, target] - box, dim=-1)
            cost = box_distance - 0.04 * objects_out[:, -1, target, 2] + 0.02 * motion
        return cost, {"closest_distance_m": distance.min(1).values, "terminal_distance_m": distance[:, -1], "predicted_contact": contact.max(1).values}

    @torch.no_grad()
    def _cem(self, output: dict, observation: dict) -> tuple[torch.Tensor, int, int, dict]:
        target_prob = output["target_logits"].softmax(-1)[0]; target = int(target_prob.argmax())
        arm_prob = output["active_arm_logits"].softmax(-1)[0]; active_arm = int(arm_prob.argmax())
        mean = output["action_mean"] * self.stats["action_delta_std"] + self.stats["action_delta_mean"]
        std = (output["action_log_std"].exp() * self.stats["action_delta_std"]).clamp(0.004, 0.030)
        mean = mean.expand(self.horizon, -1).clone(); std = std.expand(self.horizon, -1).clone()
        generator = torch.Generator(device=self.device); generator.manual_seed(int(self.planner_context["base_seed"]))
        reports = []
        for iteration in range(self.iterations):
            noise = torch.randn((self.population, self.horizon, ACTION_DIM), generator=generator, device=self.device)
            candidates = (mean[None] + std[None] * noise).clamp(-self.delta_limit, self.delta_limit)
            cost, metrics = self._rollout_cost(observation["state"], observation["velocity"], observation["objects"], observation["sites"], observation["tactile"], candidates, target, active_arm)
            elite_count = max(4, self.population // 8); elite = candidates[cost.topk(elite_count, largest=False).indices]
            mean = elite.mean(0); std = elite.std(0).clamp(0.0025, 0.025)
            reports.append({"iteration": iteration + 1, "best_cost": float(cost.min()), "mean_cost": float(cost.mean()), "best_closest_distance_m": float(metrics["closest_distance_m"][cost.argmin()]), "best_predicted_contact": float(metrics["predicted_contact"][cost.argmin()])})
        final = mean[0]
        return final, target, active_arm, {"iterations": reports, "prior_std_mean_rad": float((output["action_log_std"].exp() * self.stats["action_delta_std"]).mean())}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise=None) -> torch.Tensor:
        del noise
        if self.observation is None: self.observe_history(batch)
        observation = self.observation; self.observation = None
        output = self.model(
            observation["images"],
            (observation["state"] - self.stats["state_mean"]) / self.stats["state_std"],
            (observation["velocity"] - self.stats["velocity_mean"]) / self.stats["velocity_std"],
            (observation["sites"] - self.stats["contact_sites_mean"]) / self.stats["contact_sites_std"],
            observation["tactile"], observation["ids"], observation["mask"],
        )
        visual_pose = output["object_mean"] * self.stats["objects_std"] + self.stats["objects_mean"]
        visual_pose = torch.stack((visual_pose[:, 0::3].clamp(0.25, 0.70), visual_pose[:, 1::3].clamp(-0.45, 0.35), visual_pose[:, 2::3].clamp(0.72, 1.15)), dim=-1).reshape(1, 9)
        if float(observation["tactile"].item()) >= 0.5 and self.object_belief is not None:
            pose = self.object_belief
            self.last_object_state_source = "learned_world_model_contact_belief"
        else:
            pose = visual_pose
            self.last_object_state_source = "raw_rgb_vla_visual_state"
        action, target, arm, planner = self._cem(output, {**observation, "objects": pose})
        self._advance_object_belief(observation, pose, action, target)
        self.last_target_probabilities = output["target_logits"].softmax(-1)[0].cpu().tolist(); self.last_active_arm_probabilities = output["active_arm_logits"].softmax(-1)[0].cpu().tolist()
        self.last_object_position = pose[0].reshape(TARGET_COUNT, 3).cpu().tolist(); self.last_contact_site_position = observation["sites"][0].reshape(-1, 3).cpu().tolist(); self.last_tactile_contact = float(observation["tactile"].item())
        self.candidate_reports.append({"target_index": target, "active_arm_index": arm, "object_state_source": self.last_object_state_source, "selection_mode": "learned_raw_rgb_vla_prior_plus_learned_world_model_cem_mpc", "planner": planner})
        absolute = observation["state"][0] + action
        return absolute.reshape(1, 1, ACTION_DIM)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True); parser.add_argument("--vla-checkpoint", type=Path, required=True); parser.add_argument("--world-model-checkpoint", type=Path, required=True); parser.add_argument("--train-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3); parser.add_argument("--seed", type=int, default=20260730); parser.add_argument("--control-fps", type=int, default=15); parser.add_argument("--duration-s", type=float, default=14.0); parser.add_argument("--layout-mode", choices=("slot_jitter", "random"), default="random"); parser.add_argument("--position-jitter-m", type=float, default=.018)
    parser.add_argument("--horizon", type=int, default=4); parser.add_argument("--population", type=int, default=96); parser.add_argument("--iterations", type=int, default=2); parser.add_argument("--action-delta-limit-rad", type=float, default=.08); parser.add_argument("--body-gain-scale", type=float, default=5.0); parser.add_argument("--hand-gain-scale", type=float, default=6.0); parser.add_argument("--observation-width", type=int, default=160); parser.add_argument("--observation-height", type=int, default=120); parser.add_argument("--video-width", type=int, default=320); parser.add_argument("--video-height", type=int, default=240); parser.add_argument("--disable-video", action="store_true"); parser.add_argument("--save-observation-videos", action="store_true"); parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 3 or args.episodes % 3: raise ValueError("--episodes must be divisible by three")
    metadata = json.loads((args.train_root / "meta/info.json").read_text()); args.output_dir.mkdir(parents=True, exist_ok=True); args.disable_assisted_grasp = True; args.replan_steps = 1
    policy = RawVisualServoMPCPolicy(args.vla_checkpoint, args.world_model_checkpoint, args.device, args.horizon, args.population, args.iterations, args.action_delta_limit_rad)
    reports=[]
    for spec in base.evaluation_specs(args.episodes, args.seed, args.position_jitter_m, args.layout_mode):
        report=base.run_episode(policy,spec,args,metadata["features"]["action"]["names"]); report.update(predicted_target_probabilities=policy.last_target_probabilities,predicted_active_arm_probabilities=policy.last_active_arm_probabilities,camera_object_position_m=policy.last_object_position,object_state_source=policy.last_object_state_source,measured_finger_contact_site_position_m=policy.last_contact_site_position,measured_tactile_contact=policy.last_tactile_contact,world_model_candidates=policy.candidate_reports); reports.append(report)
    contacts=[item["hand_object_contact_candidate_steps"]>0 for item in reports]; language=float(np.mean([item["world_model_candidates"][-1]["target_index"]==base.OBJECT_NAMES.index(item["target_object"]) for item in reports]))
    summary={"experiment":"G1LANG-raw-multiview-vla-learned-world-model-cem-mpc-strict-gate","vla_checkpoint":str(args.vla_checkpoint),"world_model_checkpoint":str(args.world_model_checkpoint),"episodes":len(reports),"input_contract":"raw head/left-wrist/right-wrist RGB + language + state31; measured qvel/finger geometry/tactile derived causally from robot state/contact","candidate_selection_mode":"VLA action distribution sampled and optimized by learned-world-model CEM MPC", "privileged_positions_at_inference":False,"assisted_grasp_enabled":False,"equality_constraints":False,"magnet_enabled":False,"reports":reports,"language_target_accuracy":language,"object_selection_accuracy":language,"real_contact_rate":float(np.mean(contacts)),"no_grasp_rate":float(np.mean([item["grabbed_object"] is None for item in reports])),"mean_joint_limit_violation_fraction":float(np.mean([item["joint_limit_violation_fraction"] for item in reports])),"mean_action_safety_clip_fraction":float(np.mean([item["action_safety_clip_fraction"] for item in reports])),"full_pick_place_success_rate":float(np.mean([item["task_success"] for item in reports]) )}
    summary["passed"]=bool(all(contacts) and summary["object_selection_accuracy"]>=.8 and summary["full_pick_place_success_rate"]==1.0 and summary["mean_joint_limit_violation_fraction"]<=.01)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps({key:value for key,value in summary.items() if key!="reports"},indent=2))


if __name__ == "__main__": main()
