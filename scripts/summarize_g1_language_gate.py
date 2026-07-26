#!/usr/bin/env python3
"""Aggregate the four pre-registered G1 language counterfactual seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


TARGETS = ("red_triangle", "yellow_rod", "green_cube")
EXPECTED_SEEDS = (20260727, 20260728, 20260729, 20260730)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    seeds = [int(report["flow_noise_seed"]) for report in reports]
    if sorted(seeds) != list(EXPECTED_SEEDS):
        raise ValueError(f"Expected seeds {EXPECTED_SEEDS}, found {seeds}")
    if any(int(report["queries"]) != 30 for report in reports):
        raise ValueError("Every gate report must contain exactly 30 queries")
    checkpoints = {str(report["checkpoint"]) for report in reports}
    if len(checkpoints) != 1:
        raise ValueError(f"Gate reports use multiple checkpoints: {checkpoints}")

    queries = sum(int(report["queries"]) for report in reports)
    per_target = {
        target: sum(int(report["per_target"][target]["correct"]) for report in reports)
        for target in TARGETS
    }
    correct = sum(per_target.values())
    required = math.ceil(args.threshold * queries)
    separation_ratios = [float(report["language_separation_ratio"]) for report in reports]
    exact_references = all(
        float(report["max_reference_scene_max_object_position_delta_m"]) <= 1e-12
        and float(report["max_reference_initial_state_max_abs_delta_rad"]) <= 1e-12
        for report in reports
    )
    summary = {
        "experiment": "G1LANG-four-seed-language-gate",
        "checkpoint": checkpoints.pop(),
        "seeds": seeds,
        "per_seed_correct": [
            sum(int(report["per_target"][target]["correct"]) for target in TARGETS)
            for report in reports
        ],
        "queries": queries,
        "correct": correct,
        "accuracy": correct / queries,
        "required_correct": required,
        "threshold": args.threshold,
        "per_target_correct": per_target,
        "mean_language_separation_ratio": sum(separation_ratios) / len(separation_ratios),
        "exact_reference_contract": exact_references,
        "gate_passed": correct >= required and exact_references,
        "source_summaries": [str(path) for path in args.summaries],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
