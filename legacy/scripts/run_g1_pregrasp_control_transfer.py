#!/usr/bin/env python3
"""Transfer the selected hold controller back to a continuous pre-grasp motion."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path


GROUPS = (
    ("A_baseline_pd", False, 1.0),
    ("B_gravity_compensation", True, 1.0),
    ("C_gravity_compensation_kp1p5", True, 1.5),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for group_name, gravity_compensation, kp_scale in GROUPS:
        kd_scale = math.sqrt(kp_scale)
        group_dir = args.output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(args.python),
            str(args.runner),
            "--asset",
            str(args.asset.resolve()),
            "--reachability-report",
            str(args.reachability_report.resolve()),
            "--output-dir",
            str(group_dir),
            "--tote-x",
            "0.55",
            "--shoulder-lead-seconds",
            "0.0",
            "--kp-scale",
            str(kp_scale),
            "--kd-scale",
            str(kd_scale),
        ]
        if gravity_compensation:
            command.append("--gravity-compensation")
        subprocess.run(
            command,
            check=True,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        audit = json.loads((group_dir / "pregrasp_trajectory_audit.json").read_text())
        groups.append({"group": group_name, **audit})

    selected = groups[-1] if groups[-1]["passed"] else None
    summary = {
        "experiment": "G1WH-19-continuous-control-transfer",
        "controlled_conditions": {
            "selected_tote_x_m": 0.55,
            "shoulder_lead_seconds": 0.0,
            "ik_orientation_weight_m_per_rad": 0.08,
        },
        "causal_sequence": [
            "baseline PD",
            "add gravity compensation",
            "add selected feedback gains",
        ],
        "groups": groups,
        "selected_group": selected["group"] if selected else None,
        "experiment_passed": selected is not None,
    }
    summary_path = args.output_dir / "continuous_control_transfer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
