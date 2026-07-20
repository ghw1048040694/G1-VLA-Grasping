#!/usr/bin/env python3
"""Compare damping and reflected-armature models for isolated G1 control."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
from pathlib import Path


PROFILES = ("raw", "damping", "armature", "both")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for profile in PROFILES:
        candidate_dir = args.output_dir / profile
        subprocess.run(
            [
                str(args.python),
                str(args.runner),
                "--asset",
                str(args.asset),
                "--output-dir",
                str(candidate_dir),
                "--gain-profile",
                "unitree",
                "--dynamics-profile",
                profile,
            ],
            check=True,
            env={**os.environ, "MPLBACKEND": "Agg"},
        )
        report = json.loads((candidate_dir / "joint_response_summary.json").read_text())
        rows = list(csv.DictReader((candidate_dir / "joint_response_metrics.csv").open()))
        leakage = [float(row["max_other_joint_leakage_rad"]) for row in rows]
        diagnostic_score = statistics.mean(
            [
                report["body_mean_steady_rmse_rad"],
                report["hand_mean_steady_rmse_rad"],
                statistics.median(leakage),
            ]
        )
        candidates.append(
            {
                "profile": profile,
                "passed_joints": report["passed_joints"],
                "body_mean_steady_rmse_rad": report["body_mean_steady_rmse_rad"],
                "hand_mean_steady_rmse_rad": report["hand_mean_steady_rmse_rad"],
                "minimum_leakage_rad": min(leakage),
                "median_leakage_rad": statistics.median(leakage),
                "maximum_leakage_rad": max(leakage),
                "diagnostic_score_rad": diagnostic_score,
            }
        )

    selected = min(
        candidates,
        key=lambda item: (-item["passed_joints"], item["diagnostic_score_rad"]),
    )
    summary = {
        "experiment": "G1WH-07-joint-dynamics-ablation",
        "controlled_variables": "Unitree gains, isolated targets, timing, fixed base, gravity, contacts",
        "diagnostic_score": "Mean of body RMSE, hand RMSE, and median leakage; lower is better",
        "candidates": candidates,
        "selected_candidate": selected,
    }
    summary_path = args.output_dir / "dynamics_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
