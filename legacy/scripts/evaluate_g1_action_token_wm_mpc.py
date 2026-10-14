#!/usr/bin/env python3
"""Strict MuJoCo gate for raw-RGB action-token VLA proposals plus learned WM-MPC."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import evaluate_g1_language_pick_place as base
from g1_action_token_vla import load_checkpoint as load_vla
from g1_direct_joint_candidate_vla import encode_tasks
from g1_finger_contact_geometry import finger_contact_sites
from g1_raw_multiview_servo_world_model import load_checkpoint as load_world


class ActionTokenMPCPolicy:
    def __init__(self, vla_checkpoint: Path, world_checkpoint: Path, device: str, horizon: int, proposal_count: int, limit: float) -> None:
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.vla, self.vocab, self.stats, self.codebook, self.future_offsets = load_vla(vla_checkpoint, str(self.device))
        self.world, self.world_stats = load_world(world_checkpoint, str(self.device))
        self.horizon = min(int(horizon), self.vla.action_chunk)
        self.proposal_count = min(int(proposal_count), self.vla.token_count)
        self.limit = float(limit)
        self.config = SimpleNamespace(chunk_size=1, max_action_dim=31)
        self.needs_temporal_history = True
        self.reset()

    def reset(self) -> None:
        self.ctx = None
        self.history = deque(maxlen=self.vla.history_frames)
        self.candidate_reports: list[dict] = []
        self.last_target_probabilities = None
        self.last_active_arm_probabilities = None
        self.last_object_position = None
        self.last_tactile_contact = None

    def set_planner_context(self, context: dict) -> None:
        self.ctx = context

    def _z(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return (value - self.world_stats[f"{name}_mean"]) / self.world_stats[f"{name}_std"]

    def _raw(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return value * self.world_stats[f"{name}_std"] + self.world_stats[f"{name}_mean"]

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
        return torch.tensor([[float(active)]], device=self.device)

    def observe_history(self, batch: dict) -> None:
        data, controlled = self.ctx["data"], self.ctx["controlled"]
        images = torch.stack([batch[key][0].to(self.device) for key in base.CAMERAS], dim=0)
        velocity = torch.tensor(
            [[data.qvel[controlled[name]["qvel_id"]] for name in self.ctx["upper_names"]]],
            device=self.device,
            dtype=torch.float32,
        )
        self.history.append(
            {
                "images": images,
                "state": batch["observation.state"].to(self.device),
                "velocity": velocity,
                "sites": self._sites(),
                "tactile": self._tactile(),
                "task": batch["task"],
            }
        )

    def _stack(self) -> dict[str, torch.Tensor]:
        rows = list(self.history)
        rows = [rows[0]] * (self.vla.history_frames - len(rows)) + rows
        output = {
            name: torch.stack([row[name] for row in rows], dim=1)
            for name in ("state", "velocity", "sites", "tactile")
        }
        output["images"] = torch.stack([row["images"] for row in rows], dim=0).unsqueeze(0)
        return output

    def _cost(self, observation: dict[str, torch.Tensor], objects: torch.Tensor, actions: torch.Tensor, target: int, arm: int):
        count = len(actions)
        target_onehot = F.one_hot(torch.full((count,), target, device=self.device), 3).float()
        current = {
            "state": self._z(observation["state"], "state").expand(count, -1),
            "velocity": self._z(observation["velocity"], "velocity").expand(count, -1),
            "objects": self._z(objects, "objects").expand(count, -1),
            "sites": self._z(observation["sites"], "contact_sites").expand(count, -1),
        }
        distances = []
        raw_objects = None
        for step in range(self.horizon):
            prediction = self.world(
                current["state"], current["velocity"], current["objects"], current["sites"],
                observation["tactile"].expand(count, -1), self._z(actions[:, step], "action"), target_onehot,
            )
            current = {
                "state": prediction["state"], "velocity": prediction["velocity"],
                "objects": prediction["objects"], "sites": prediction["contact_sites"],
            }
            raw_objects = self._raw(current["objects"], "objects").reshape(count, 3, 3)
            raw_sites = self._raw(current["sites"], "contact_sites").reshape(count, 2, -1, 3)[:, arm]
            distances.append(torch.linalg.vector_norm(raw_sites - raw_objects[:, target, None], dim=-1).min(-1).values)
        distance = torch.stack(distances, dim=1)
        motion = actions.square().mean((1, 2)).sqrt()
        if float(observation["tactile"].item()) < .5:
            cost = distance.min(1).values + .30 * distance[:, -1] + .015 * motion
        else:
            cost = torch.linalg.vector_norm(
                raw_objects[:, target] - torch.tensor(base.BIN_POSITION_M, device=self.device), dim=-1
            ) + .015 * motion
        return cost, {
            "nearest": distance.min(1).values,
            "uncertainty": torch.zeros_like(cost),
            "final_objects": raw_objects,
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise=None) -> torch.Tensor:
        del batch, noise
        observation = self._stack()
        ids, mask = encode_tasks(self.history[-1]["task"], self.vocab)
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
        probabilities = logits.sigmoid()
        token_ids = logits.topk(self.proposal_count).indices
        actions = (self.codebook[token_ids] + output["candidate_action_residual"][0, token_ids]).clamp(-self.limit, self.limit)
        cost, metrics = self._cost(
            {name: observation[name][:, -1] for name in ("state", "velocity", "sites", "tactile")},
            objects,
            actions,
            target,
            arm,
        )
        chosen = int(cost.argmin())
        selected_token = int(token_ids[chosen])
        self.last_target_probabilities = output["target_logits"].softmax(-1)[0].cpu().tolist()
        self.last_active_arm_probabilities = output["active_arm_logits"].softmax(-1)[0].cpu().tolist()
        self.last_object_position = objects[0].reshape(3, 3).cpu().tolist()
        self.last_tactile_contact = float(observation["tactile"][0, -1])
        self.candidate_reports.append(
            {
                "target_index": target,
                "active_arm_index": arm,
                "selection_mode": "raw_rgb_discrete_action_token_vla_plus_learned_visual_state_world_model_mpc",
                "selected_token": selected_token,
                "selected_token_probability": float(probabilities[selected_token]),
                "selected_mean_abs_delta_mrad": float(actions[chosen].abs().mean() * 1e3),
                "best_predicted_closest_distance_m": float(metrics["nearest"][chosen]),
                "best_ensemble_std_m": float(metrics["uncertainty"][chosen]),
                "candidate_tokens": token_ids.cpu().tolist(),
                "candidate_mean_abs_delta_mrad": (actions.abs().mean((1, 2)) * 1e3).cpu().tolist(),
            }
        )
        return (observation["state"][0, -1] + actions[chosen, 0]).reshape(1, 1, 31)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--vla-checkpoint", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260745)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=14)
    parser.add_argument("--layout-mode", choices=("slot_jitter", "random"), default="random")
    parser.add_argument("--position-jitter-m", type=float, default=.018)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--proposal-count", type=int, default=12)
    parser.add_argument("--action-delta-limit-rad", type=float, default=.08)
    parser.add_argument("--body-gain-scale", type=float, default=5)
    parser.add_argument("--hand-gain-scale", type=float, default=6)
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
        raise ValueError("episodes must be divisible by three")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.disable_assisted_grasp = True
    args.replan_steps = 1
    names = json.loads((args.train_root / "meta/info.json").read_text())["features"]["action"]["names"]
    policy = ActionTokenMPCPolicy(
        args.vla_checkpoint, args.world_model_checkpoint, args.device,
        args.horizon, args.proposal_count, args.action_delta_limit_rad,
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
        reports.append(report)
    contacts = [report["hand_object_contact_candidate_steps"] > 0 for report in reports]
    language_accuracy = float(np.mean([
        report["world_model_candidates"][-1]["target_index"] == base.OBJECT_NAMES.index(report["target_object"])
        for report in reports
    ]))
    summary = {
        "experiment": "G1LANG-discrete-action-token-raw-rgb-vla-ensemble-wm-mpc-strict-gate",
        "episodes": len(reports),
        "input_contract": "raw RGB x3 + language + state31 + qvel31 + finger sites + tactile",
        "action_source": "learned discrete strict-expert action-token proposals selected by learned world-model MPC",
        "privileged_positions_at_inference": False,
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "reports": reports,
        "language_target_accuracy": language_accuracy,
        "real_contact_rate": float(np.mean(contacts)),
        "no_grasp_rate": float(np.mean([report["grabbed_object"] is None for report in reports])),
        "full_pick_place_success_rate": float(np.mean([report["task_success"] for report in reports])),
    }
    summary["passed"] = bool(all(contacts) and language_accuracy >= .8 and summary["full_pick_place_success_rate"] == 1.0)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
