#!/usr/bin/env python3
"""Run and summarize controlled PD-gain candidates for G1 actuation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


CANDIDATES = {
    "A_baseline": (80.0, 4.0, 8.0, 0.30),
    "B_more_damping": (80.0, 8.0, 8.0, 0.80),
    "C_more_stiffness": (160.0, 8.0, 16.0, 0.80),
    "D_balanced": (120.0, 10.0, 12.0, 1.20),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, (body_kp, body_kd, hand_kp, hand_kd) in CANDIDATES.items():
        candidate_dir = args.output_dir / name
        command = [
            str(args.python),
            str(args.runner),
            "--asset",
            str(args.asset),
            "--output-dir",
            str(candidate_dir),
            "--body-kp",
            str(body_kp),
            "--body-kd",
            str(body_kd),
            "--hand-kp",
            str(hand_kp),
            "--hand-kd",
            str(hand_kd),
        ]
        subprocess.run(command, check=True, env={**os.environ, "MUJOCO_GL": "egl"})
        report = json.loads((candidate_dir / "actuation_audit.json").read_text())
        body_rmse = report["body_hold_rmse_rad"]
        hand_rmse = report["hand_hold_rmse_rad"]
        results.append(
            {
                "name": name,
                "body_kp": body_kp,
                "body_kd": body_kd,
                "hand_kp": hand_kp,
                "hand_kd": hand_kd,
                "body_hold_rmse_rad": body_rmse,
                "hand_hold_rmse_rad": hand_rmse,
                "combined_rmse_rad": (body_rmse + hand_rmse) / 2.0,
                "video": report["video"],
            }
        )

    selected = min(results, key=lambda item: item["combined_rmse_rad"])
    summary = {
        "experiment": "G1WH-04-pd-gain-sweep",
        "controlled_variables": "Same asset, targets, timing, gravity, contacts, and fixed base",
        "selection_metric": "Mean of body and hand hold RMSE in radians",
        "candidates": results,
        "selected_candidate": selected,
    }
    summary_path = args.output_dir / "gain_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
