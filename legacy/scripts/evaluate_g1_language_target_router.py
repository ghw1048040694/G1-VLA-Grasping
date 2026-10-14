#!/usr/bin/env python3
"""Evaluate a learned language router followed by target-specific SmolVLA policies."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import evaluate_g1_language_pick_place as base
import train_g1_world_model as world_model_utils
from evaluate_g1_language_smolvla_heldout import load_policy as load_adapter_policy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


TARGETS = tuple(base.OBJECT_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--specialist-checkpoint",
        action="append",
        required=True,
        metavar="TARGET=PATH",
        help="Repeat once for red_triangle, yellow_rod, and green_cube.",
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_language_paired_train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--assist-distance-m", type=float, default=0.075)
    parser.add_argument("--position-jitter-m", type=float, default=0.018)
    parser.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument("--world-model-candidates", type=int, default=4)
    parser.add_argument("--world-model-horizon", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def parse_specialists(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Specialist must use TARGET=PATH syntax: {value}")
        target, raw_path = value.split("=", 1)
        if target not in TARGETS or target in result:
            raise ValueError(f"Invalid or duplicate specialist target: {target}")
        result[target] = Path(raw_path)
    missing = [target for target in TARGETS if target not in result]
    if missing:
        raise ValueError(f"Missing specialists: {missing}")
    for target, path in result.items():
        if not (path / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing {target} checkpoint: {path}")
    return result


def load_world_model(checkpoint_path: Path, device: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    layout = checkpoint["layout"]
    if layout["state_dim"] != 86 or layout["action_dim"] != 31:
        raise ValueError(f"Unexpected world-model layout: {layout}")
    model = world_model_utils.ResidualWorldModel(
        int(layout["state_dim"]),
        int(layout["action_dim"]),
        int(args["hidden_dim"]),
        int(args["depth"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return {
        "model": model,
        "stats": checkpoint["normalizers"],
        "checkpoint": str(checkpoint_path),
    }


def repeat_policy_batch(batch: dict, count: int) -> dict:
    repeated = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            repeats = (count,) + (1,) * (value.ndim - 1)
            repeated[key] = value.repeat(repeats)
        elif isinstance(value, list):
            repeated[key] = value * count
        else:
            raise TypeError(f"Unsupported policy batch value for {key}: {type(value)}")
    return repeated


class RoutedPolicy:
    """Expose the LeRobot policy interface while routing from a learned posterior."""

    def __init__(self, router, specialist_loader, planner: dict | None, args):
        self.router = router
        self.specialist_loader = specialist_loader
        self.planner = planner
        self.args = args
        self.specialist = None
        self.specialist_target: str | None = None
        self.route_target: str | None = None
        self.route_logits: list[float] | None = None
        self.planner_context: dict | None = None
        self.planner_reports: list[dict] = []

    @property
    def config(self):
        if self.specialist is not None:
            return self.specialist.config
        return self.router.config

    def reset(self) -> None:
        self.router.reset()
        if self.specialist is not None:
            self.specialist.reset()
        self.route_target = None
        self.route_logits = None
        self.planner_context = None
        self.planner_reports = []

    def set_planner_context(self, context: dict) -> None:
        self.planner_context = context

    def _get_specialist(self, target: str):
        if self.specialist_target == target and self.specialist is not None:
            return self.specialist
        if self.specialist is not None:
            del self.specialist
            self.specialist = None
            self.specialist_target = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.specialist = self.specialist_loader(target)
        self.specialist_target = target
        return self.specialist

    def _world_model_state(self) -> np.ndarray:
        if self.planner_context is None or self.route_target is None:
            raise RuntimeError("Planner context and routed target are required")
        context = self.planner_context
        data = context["data"]
        controlled = context["controlled"]
        upper_names = context["upper_names"]
        body_id = context["object_body_ids"][self.route_target]
        initial_z = float(context["initial_object_positions"][self.route_target][2])
        lift = float(data.xpos[body_id, 2] - initial_z)
        contacts = np.zeros(2, dtype=np.float32)
        if (
            context["grabbed_object"] == self.route_target
            and context["grabbed_arm_index"] is not None
            and context["assist_active"]
        ):
            contacts[int(context["grabbed_arm_index"])] = 1.0
        progress = float(context["frame"]) / max(context["control_frames"] - 1, 1)
        state = np.concatenate(
            (
                [data.qpos[controlled[name]["qpos_id"]] for name in upper_names],
                [data.qvel[controlled[name]["qvel_id"]] for name in upper_names],
                data.xpos[body_id],
                data.xquat[body_id],
                data.cvel[body_id, 3:],
                data.cvel[body_id, :3],
                np.stack([data.site_xpos[item] for item in context["palm_ids"]]).reshape(-1),
                contacts,
                (float(lift <= 0.003), lift, progress),
            )
        ).astype(np.float32)
        if state.shape != (world_model_utils.LAYOUT.state_dim,):
            raise RuntimeError(f"Unexpected world-model state shape: {state.shape}")
        return state

    def _select_world_model_candidate(
        self, specialist, batch: dict, baseline_noise: torch.Tensor
    ) -> torch.Tensor:
        if self.planner is None or self.planner_context is None:
            raise RuntimeError("World-model planner is not configured")
        context = self.planner_context
        policy_started = time.perf_counter()
        noises = [baseline_noise]
        noises.extend(
            base.seeded_noise(
                specialist,
                context["base_seed"] + index * 1_000_003,
                self.args.device,
            )
            for index in range(1, self.args.world_model_candidates)
        )
        raw_candidates = (
            specialist.predict_action_chunk(
                repeat_policy_batch(batch, self.args.world_model_candidates),
                noise=torch.cat(noises, dim=0),
            )
            .detach()
            .cpu()
            .numpy()[:, :, :31]
        )
        policy_inference_s = time.perf_counter() - policy_started

        horizon = min(
            self.args.world_model_horizon,
            min(len(candidate) for candidate in raw_candidates),
        )
        lower = context["action_lower"]
        upper = context["action_upper"]
        candidates = np.stack(
            [np.clip(candidate[:horizon], lower, upper) for candidate in raw_candidates]
        ).astype(np.float32)
        state = self._world_model_state()
        stats = self.planner["stats"]
        normalized_state = (state - stats["state_mean"]) / stats["state_std"]
        normalized_actions = (
            candidates - stats["action_mean"][None, None, :]
        ) / stats["action_std"][None, None, :]
        initial = np.repeat(
            normalized_state[None, :], len(candidates), axis=0
        ).astype(np.float32)
        world_model_started = time.perf_counter()
        predicted = world_model_utils.rollout(
            self.planner["model"],
            torch.from_numpy(initial).to(self.args.device),
            torch.from_numpy(normalized_actions).to(self.args.device),
        ).detach().cpu().numpy()
        predicted = (
            predicted * stats["state_std"][None, None, :]
            + stats["state_mean"][None, None, :]
        )
        world_model_inference_s = time.perf_counter() - world_model_started

        layout = world_model_utils.LAYOUT
        target_low, target_high = layout.tote_position
        palm_low, palm_high = layout.palm_position
        joint_low, joint_high = layout.joint_position
        contact_low, contact_high = layout.bilateral_contact
        progress_index = layout.task_progress[0]
        endpoint = predicted[:, -1]
        goal = base.BIN_POSITION_M.copy()
        goal[2] += 0.05
        grasped = context["grabbed_object"] == self.route_target
        scores = []
        components = []
        for index in range(len(candidates)):
            target_position = endpoint[index, target_low:target_high]
            palms = endpoint[index, palm_low:palm_high].reshape(2, 3)
            approach_distance = float(
                np.min(np.linalg.norm(palms - target_position[None, :], axis=1))
            )
            goal_distance = float(np.linalg.norm(target_position - goal))
            predicted_qpos = predicted[index, :, joint_low:joint_high]
            violations = np.maximum(lower[None, :] - predicted_qpos, 0.0) + np.maximum(
                predicted_qpos - upper[None, :], 0.0
            )
            violation_rmse = float(np.sqrt(np.mean(violations**2)))
            clip_rmse = float(
                np.sqrt(np.mean((raw_candidates[index][:horizon] - candidates[index]) ** 2))
            )
            action_sequence = np.concatenate(
                (context["previous_action"][None, :], candidates[index]), axis=0
            )
            smoothness = float(np.sqrt(np.mean(np.diff(action_sequence, axis=0) ** 2)))
            contact = float(np.mean(endpoint[index, contact_low:contact_high]))
            progress = float(endpoint[index, progress_index])
            task_distance = goal_distance if grasped else approach_distance
            score = (
                -20.0 * task_distance
                + 4.0 * contact
                + 2.0 * progress
                - 25.0 * violation_rmse
                - 20.0 * clip_rmse
                - 0.5 * smoothness
            )
            scores.append(score)
            components.append(
                {
                    "score": score,
                    "predicted_approach_distance_m": approach_distance,
                    "predicted_goal_distance_m": goal_distance,
                    "predicted_contact": contact,
                    "predicted_progress": progress,
                    "predicted_joint_violation_rmse_rad": violation_rmse,
                    "action_clip_rmse_rad": clip_rmse,
                    "action_smoothness_rmse_rad": smoothness,
                }
            )
        selected = int(np.argmax(scores))
        self.planner_reports.append(
            {
                "frame": int(context["frame"]),
                "routed_target": self.route_target,
                "grasped": bool(grasped),
                "selected_index": selected,
                "selected_nonbaseline": selected != 0,
                "score_margin": float(
                    scores[selected]
                    - max(score for index, score in enumerate(scores) if index != selected)
                ),
                "policy_inference_s": policy_inference_s,
                "world_model_inference_s": world_model_inference_s,
                "candidates": components,
            }
        )
        return torch.from_numpy(candidates[selected][None, ...]).to(self.args.device)

    def planner_episode_report(self) -> dict | None:
        if self.planner is None:
            return None
        return {
            "checkpoint": self.planner["checkpoint"],
            "candidate_count": self.args.world_model_candidates,
            "horizon": self.args.world_model_horizon,
            "replans": len(self.planner_reports),
            "nonbaseline_selections": sum(
                item["selected_nonbaseline"] for item in self.planner_reports
            ),
            "mean_policy_inference_s": float(
                np.mean([item["policy_inference_s"] for item in self.planner_reports])
            ),
            "mean_world_model_inference_s": float(
                np.mean(
                    [item["world_model_inference_s"] for item in self.planner_reports]
                )
            ),
            "reports": self.planner_reports,
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict, noise: torch.Tensor | None = None):
        if self.route_target is None:
            # The router's action is discarded. Its target posterior is the only
            # signal used to select the specialist.
            self.router.predict_action_chunk(batch, noise=noise)
            logits = getattr(self.router.model, "_language_action_target_logits", None)
            if logits is None:
                raise RuntimeError("Router did not expose target posterior")
            logits = logits.detach().float().cpu()[0]
            self.route_logits = logits.tolist()
            self.route_target = TARGETS[int(torch.argmax(logits).item())]
        specialist = self._get_specialist(self.route_target)
        if self.planner is None:
            return specialist.predict_action_chunk(batch, noise=noise)
        if noise is None:
            raise RuntimeError("MPC requires deterministic baseline noise")
        return self._select_world_model_candidate(specialist, batch, noise)


def main() -> None:
    args = parse_args()
    if args.world_model_checkpoint is not None and args.world_model_candidates < 2:
        raise ValueError("--world-model-candidates must be at least two")
    if args.world_model_horizon < 1:
        raise ValueError("--world-model-horizon must be positive")
    specialists = parse_specialists(args.specialist_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)

    router = load_adapter_policy(args.router_checkpoint, train_meta, args.device)
    planner = (
        load_world_model(args.world_model_checkpoint, args.device)
        if args.world_model_checkpoint is not None
        else None
    )
    expected_names = train_meta.features["action"]["names"]

    def specialist_loader(target: str):
        return base.load_policy(specialists[target], train_meta, args.device)

    routed = RoutedPolicy(router, specialist_loader, planner, args)
    specs = base.evaluation_specs(args.episodes, args.seed, args.position_jitter_m)
    reports = []
    # Grouping by expected target keeps one specialist resident for each group;
    # the route itself is still selected from the learned posterior per episode.
    for expected_target in TARGETS:
        for spec in (item for item in specs if item["target_object"] == expected_target):
            report = base.run_episode(routed, spec, args, expected_names)
            report["routed_target"] = routed.route_target
            report["router_logits"] = routed.route_logits
            report["router_correct"] = routed.route_target == expected_target
            planner_report = routed.planner_episode_report()
            if planner_report is not None:
                report["world_model_planner"] = planner_report
            reports.append(report)

    reports.sort(key=lambda item: item["episode_index"])
    summary = {
        "experiment": (
            "G1FINAL-03-language-target-world-model-mpc"
            if planner is not None
            else "G1FINAL-01-learned-language-target-router"
        ),
        "router_checkpoint": str(args.router_checkpoint),
        "specialist_checkpoints": {
            target: str(path) for target, path in specialists.items()
        },
        "world_model_checkpoint": (
            str(args.world_model_checkpoint)
            if args.world_model_checkpoint is not None
            else None
        ),
        "world_model_candidates": args.world_model_candidates if planner else None,
        "world_model_horizon": args.world_model_horizon if planner else None,
        "episodes": len(reports),
        "router_accuracy": sum(item["router_correct"] for item in reports) / len(reports),
        "object_selection_accuracy": sum(
            item["selected_correct_object"] for item in reports
        )
        / len(reports),
        "put_in_box_success_rate": sum(item["task_success"] for item in reports)
        / len(reports),
        "strict_success_rate": sum(item["passed"] for item in reports) / len(reports),
        "wrong_object_grasp_rate": sum(
            item["grabbed_object"] is not None and not item["selected_correct_object"]
            for item in reports
        )
        / len(reports),
        "mean_joint_limit_violation_fraction": sum(
            item["joint_limit_violation_fraction"] for item in reports
        )
        / len(reports),
        "per_target": {},
        "acceptance_thresholds": {
            "router_accuracy_min": 0.90,
            "object_selection_accuracy_min": 0.80,
            "put_in_box_success_rate_min": 0.70,
            "strict_success_rate_min": 0.70,
            "wrong_object_grasp_rate_max": 0.10,
            "mean_joint_limit_violation_fraction_max": 0.01,
        },
        "reports": reports,
    }
    if planner is not None:
        planner_reports = [item["world_model_planner"] for item in reports]
        summary["world_model_planner_summary"] = {
            "total_replans": sum(item["replans"] for item in planner_reports),
            "total_nonbaseline_selections": sum(
                item["nonbaseline_selections"] for item in planner_reports
            ),
            "mean_policy_inference_s": float(
                np.mean([item["mean_policy_inference_s"] for item in planner_reports])
            ),
            "mean_world_model_inference_s": float(
                np.mean(
                    [
                        item["mean_world_model_inference_s"]
                        for item in planner_reports
                    ]
                )
            ),
        }
    for target in TARGETS:
        subset = [item for item in reports if item["target_object"] == target]
        summary["per_target"][target] = {
            "episodes": len(subset),
            "router_accuracy": sum(item["router_correct"] for item in subset)
            / len(subset),
            "object_selection_accuracy": sum(
                item["selected_correct_object"] for item in subset
            )
            / len(subset),
            "put_in_box_success_rate": sum(item["task_success"] for item in subset)
            / len(subset),
            "strict_success_rate": sum(item["passed"] for item in subset) / len(subset),
        }
    summary["passed"] = bool(
        summary["router_accuracy"] >= 0.90
        and summary["object_selection_accuracy"] >= 0.80
        and summary["put_in_box_success_rate"] >= 0.70
        and summary["strict_success_rate"] >= 0.70
        and summary["wrong_object_grasp_rate"] <= 0.10
        and summary["mean_joint_limit_violation_fraction"] <= 0.01
    )
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))
    print(f"Saved {path}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
