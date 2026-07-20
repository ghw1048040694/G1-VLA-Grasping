#!/usr/bin/env python3
"""Sweep feedback gains for the compensated G1 pre-grasp hold."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path


KP_SCALES = (1.0, 1.25, 1.5, 2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--hold-runner", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for kp_scale in KP_SCALES:
        kd_scale = math.sqrt(kp_scale)
        group_name = f"kp_{str(kp_scale).replace('.', 'p')}"
        group_dir = args.output_dir / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(args.python),
                str(args.hold_runner),
                "--asset",
                str(args.asset.resolve()),
                "--reachability-report",
                str(args.reachability_report.resolve()),
                "--output-dir",
                str(group_dir),
                "--enable-gravity",
                "--gravity-compensation",
                "--kp-scale",
                str(kp_scale),
                "--kd-scale",
                str(kd_scale),
            ],
            check=True,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        audit = json.loads((group_dir / "pregrasp_hold_audit.json").read_text())
        groups.append(
            {
                "group": group_name,
                "kp_scale": kp_scale,
                "kd_scale": kd_scale,
                "dynamic_hold": audit,
                "passed": bool(audit["passed"]),
            }
        )

    passing = [group for group in groups if group["passed"]]
    selected = min(passing, key=lambda item: item["kp_scale"]) if passing else None
    summary = {
        "experiment": "G1WH-18-feedback-gain-hold-sweep",
        "independent_variable": "kp_scale",
        "controlled_change": "kd_scale=sqrt(kp_scale)",
        "selection_rule": "Lowest feedback gain scale among passing groups",
        "groups": groups,
        "selected_kp_scale": selected["kp_scale"] if selected else None,
        "selected_kd_scale": selected["kd_scale"] if selected else None,
        "experiment_passed": selected is not None,
    }
    summary_path = args.output_dir / "feedback_gain_hold_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
