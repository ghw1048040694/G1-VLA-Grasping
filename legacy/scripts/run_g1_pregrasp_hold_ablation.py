#!/usr/bin/env python3
"""Compare gravity and gravity compensation at a static pre-grasp pose."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


GROUPS = {
    "A_no_gravity_pd": [],
    "B_gravity_pd": ["--enable-gravity"],
    "C_gravity_pd_compensation": ["--enable-gravity", "--gravity-compensation"],
}


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
    for group_name, flags in GROUPS.items():
        group_dir = args.output_dir / group_name
        subprocess.run(
            [
                str(args.python),
                str(args.runner),
                "--asset",
                str(args.asset.resolve()),
                "--reachability-report",
                str(args.reachability_report.resolve()),
                "--output-dir",
                str(group_dir),
                *flags,
            ],
            check=True,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        report = json.loads((group_dir / "pregrasp_hold_audit.json").read_text())
        groups.append({"group": group_name, **report})

    summary = {
        "experiment": "G1WH-16-pregrasp-hold-control-ablation",
        "controlled_variables": "Initial IK pose, target, asset, contacts, gains, duration",
        "groups": groups,
        "experiment_passed": groups[-1]["passed"],
    }
    summary_path = args.output_dir / "pregrasp_hold_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
