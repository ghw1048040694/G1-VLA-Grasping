#!/usr/bin/env python3
"""Sweep the hand-open margin from joint limits at the selected closure."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


MARGINS_RAD = (0.0, 0.01, 0.03, 0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for margin in MARGINS_RAD:
        group_name = f"open_margin_{int(margin * 1000):03d}mrad"
        group_dir = args.output_dir / group_name
        subprocess.run(
            [
                str(args.python),
                str(args.runner),
                "--asset",
                str(args.asset.resolve()),
                "--output-dir",
                str(group_dir),
                "--gain-profile",
                "unitree",
                "--dynamics-profile",
                "both",
                "--enable-gravity",
                "--enable-contacts",
                "--base-mode",
                "model",
                "--hand-target-scale",
                "0.8",
                "--hand-open-margin",
                str(margin),
                "--experiment-name",
                "G1WH-11-hand-open-limit-margin",
            ],
            check=True,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        report = json.loads((group_dir / "actuation_audit.json").read_text())
        non_floor_contacts = [
            pair
            for pair in report["contact_pairs_ranked"]
            if pair["body1"] != "world" and pair["body2"] != "world"
        ]
        violating_joints = {
            name: fraction
            for name, fraction in report[
                "joint_limit_violation_fraction_by_joint"
            ].items()
            if fraction > 0.0
        }
        groups.append(
            {
                "group": group_name,
                "hand_open_margin_rad": margin,
                "body_hold_rmse_rad": report["body_hold_rmse_rad"],
                "hand_hold_rmse_rad": report["hand_hold_rmse_rad"],
                "joint_limit_violation_fraction": report[
                    "joint_limit_violation_fraction"
                ],
                "violating_joints": violating_joints,
                "non_floor_contact_pair_count": len(non_floor_contacts),
                "non_floor_contacts": non_floor_contacts,
                "video": report["video"],
            }
        )

    summary = {
        "experiment": "G1WH-11-hand-open-limit-margin",
        "controlled_variables": (
            "Fixed-base asset, Unitree gains, regularized dynamics, gravity, contacts, "
            "body target, timing, 80 percent hand closure"
        ),
        "independent_variable": "hand_open_margin_rad",
        "groups": groups,
    }
    summary_path = args.output_dir / "open_margin_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
