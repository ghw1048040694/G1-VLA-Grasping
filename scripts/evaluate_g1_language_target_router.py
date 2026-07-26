#!/usr/bin/env python3
"""Evaluate a learned language router followed by target-specific SmolVLA policies."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

import evaluate_g1_language_pick_place as base
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


class RoutedPolicy:
    """Expose the LeRobot policy interface while routing from a learned posterior."""

    def __init__(self, router, specialist_loader):
        self.router = router
        self.specialist_loader = specialist_loader
        self.specialist = None
        self.specialist_target: str | None = None
        self.route_target: str | None = None
        self.route_logits: list[float] | None = None

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
        return specialist.predict_action_chunk(batch, noise=noise)


def main() -> None:
    args = parse_args()
    specialists = parse_specialists(args.specialist_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)

    router = load_adapter_policy(args.router_checkpoint, train_meta, args.device)
    expected_names = train_meta.features["action"]["names"]

    def specialist_loader(target: str):
        return base.load_policy(specialists[target], train_meta, args.device)

    routed = RoutedPolicy(router, specialist_loader)
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
            reports.append(report)

    reports.sort(key=lambda item: item["episode_index"])
    summary = {
        "experiment": "G1FINAL-01-learned-language-target-router",
        "router_checkpoint": str(args.router_checkpoint),
        "specialist_checkpoints": {
            target: str(path) for target, path in specialists.items()
        },
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
