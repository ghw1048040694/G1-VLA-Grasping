#!/usr/bin/env python3
"""Sweep shoulder lead time to avoid the transient hand-to-hip collision."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


LEADS_SECONDS = (0.0, 0.25, 0.5, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for lead in LEADS_SECONDS:
        group_name = f"shoulder_lead_{int(lead * 1000):04d}ms"
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
                "0.01",
                "--shoulder-lead-seconds",
                str(lead),
                "--experiment-name",
                "G1WH-12-shoulder-lead-timing-sweep",
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
        groups.append(
            {
                "group": group_name,
                "shoulder_lead_seconds": lead,
                "body_hold_rmse_rad": report["body_hold_rmse_rad"],
                "hand_hold_rmse_rad": report["hand_hold_rmse_rad"],
                "joint_limit_violation_fraction": report[
                    "joint_limit_violation_fraction"
                ],
                "non_floor_contact_pair_count": len(non_floor_contacts),
                "maximum_non_floor_contact_force_n": max(
                    (pair["maximum_normal_force_n"] for pair in non_floor_contacts),
                    default=0.0,
                ),
                "maximum_non_floor_penetration_m": max(
                    (pair["maximum_penetration_m"] for pair in non_floor_contacts),
                    default=0.0,
                ),
                "non_floor_contacts": non_floor_contacts,
                "video": report["video"],
            }
        )

    summary = {
        "experiment": "G1WH-12-shoulder-lead-timing-sweep",
        "controlled_variables": (
            "Fixed-base asset, Unitree gains, regularized dynamics, gravity, contacts, "
            "targets, 80 percent hand closure, 0.01 rad hand-open margin"
        ),
        "independent_variable": "shoulder_lead_seconds",
        "groups": groups,
    }
    summary_path = args.output_dir / "shoulder_lead_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
