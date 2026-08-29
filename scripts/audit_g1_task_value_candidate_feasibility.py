#!/usr/bin/env python3
"""Summarize teacher-only task-value labels across VLA candidate branches."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from run_g1_language_pick_place import OBJECT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], point: float) -> float:
    return float(np.percentile(values, point)) if values else float("nan")


def task_metrics(cache: dict, split: str) -> dict:
    required = {
        "decision_id",
        "target",
        "horizon_target_contact_fraction",
        "horizon_target_contact_last",
        "horizon_target_lift_m",
        "horizon_hand_object_comotion_m",
        "horizon_target_bin_xy_distance_m",
        "horizon_stable_grasp",
    }
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"{split} cache lacks task-value labels: {sorted(missing)}")
    rows_by_decision: dict[int, list[int]] = defaultdict(list)
    decision_ids = cache["decision_id"].long().tolist()
    for row, decision_id in enumerate(decision_ids):
        rows_by_decision[int(decision_id)].append(row)
    target = cache["target"].long().numpy()
    contact_fraction = cache["horizon_target_contact_fraction"].float().numpy()
    contact_last = cache["horizon_target_contact_last"].float().numpy()
    lift = cache["horizon_target_lift_m"].float().numpy()
    comotion = cache["horizon_hand_object_comotion_m"].float().numpy()
    bin_xy = cache["horizon_target_bin_xy_distance_m"].float().numpy()
    stable = cache["horizon_stable_grasp"].float().numpy()
    per_target: dict[str, dict[str, list[float] | int]] = {
        name: {
            "decisions": 0,
            "has_target_contact": 0,
            "has_sustained_target_contact": 0,
            "has_lift_15mm": 0,
            "has_stable_grasp": 0,
            "best_contact_fraction": [],
            "best_lift_mm": [],
            "best_comotion_mm": [],
            "best_bin_xy_mm": [],
        }
        for name in OBJECT_NAMES
    }
    for rows in rows_by_decision.values():
        target_index = int(target[rows[0]])
        if target_index < 0 or target_index >= len(OBJECT_NAMES):
            continue
        report = per_target[OBJECT_NAMES[target_index]]
        report["decisions"] = int(report["decisions"]) + 1
        best_contact = float(contact_fraction[rows].max())
        best_lift = float(lift[rows].max())
        stable_rows = stable[rows] >= 0.5
        report["has_target_contact"] = int(report["has_target_contact"]) + int(best_contact > 0.0)
        report["has_sustained_target_contact"] = int(report["has_sustained_target_contact"]) + int(
            np.any((contact_fraction[rows] >= 0.75) & (contact_last[rows] >= 0.5))
        )
        report["has_lift_15mm"] = int(report["has_lift_15mm"]) + int(best_lift >= 0.015)
        report["has_stable_grasp"] = int(report["has_stable_grasp"]) + int(stable_rows.any())
        report["best_contact_fraction"].append(best_contact)  # type: ignore[union-attr]
        report["best_lift_mm"].append(1e3 * best_lift)  # type: ignore[union-attr]
        report["best_comotion_mm"].append(1e3 * float(comotion[rows].min()))  # type: ignore[union-attr]
        report["best_bin_xy_mm"].append(1e3 * float(bin_xy[rows].min()))  # type: ignore[union-attr]
    result = {}
    for name, values in per_target.items():
        decisions = int(values["decisions"])
        result[name] = {
            "decisions": decisions,
            "candidate_target_contact_rate": int(values["has_target_contact"]) / max(decisions, 1),
            "candidate_sustained_contact_rate": int(values["has_sustained_target_contact"]) / max(decisions, 1),
            "candidate_lift_15mm_rate": int(values["has_lift_15mm"]) / max(decisions, 1),
            "candidate_stable_grasp_rate": int(values["has_stable_grasp"]) / max(decisions, 1),
            "best_candidate_contact_fraction": {
                "mean": float(np.mean(values["best_contact_fraction"])) if decisions else float("nan"),
                "p95": percentile(values["best_contact_fraction"], 95),
            },
            "best_candidate_lift_mm": {
                "mean": float(np.mean(values["best_lift_mm"])) if decisions else float("nan"),
                "p95": percentile(values["best_lift_mm"], 95),
            },
            "best_candidate_hand_object_comotion_mm": {
                "mean": float(np.mean(values["best_comotion_mm"])) if decisions else float("nan"),
                "p95": percentile(values["best_comotion_mm"], 95),
            },
            "best_candidate_bin_xy_distance_mm": {
                "mean": float(np.mean(values["best_bin_xy_mm"])) if decisions else float("nan"),
                "p95": percentile(values["best_bin_xy_mm"], 95),
            },
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    reports = {}
    for split in ("train", "validation"):
        path = args.cache_dir / f"{split}.pt"
        cache = torch.load(path, map_location="cpu", weights_only=False)
        reports[split] = task_metrics(cache, split)
    validation = reports["validation"]
    report = {
        "experiment": "G1LANG-task-value-candidate-feasibility-audit",
        "cache_dir": str(args.cache_dir),
        "labels": (
            "teacher-only active-hand target contact survival, lift, co-motion, and bin progress "
            "over VLA action-chunk branches"
        ),
        "inference_contract": "no simulator object truth; labels are offline-only",
        "splits": reports,
        "all_targets_have_stable_candidate_in_validation": bool(
            all(validation[name]["candidate_stable_grasp_rate"] > 0.0 for name in OBJECT_NAMES)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
