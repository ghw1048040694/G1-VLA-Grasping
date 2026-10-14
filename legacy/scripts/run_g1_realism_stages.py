#!/usr/bin/env python3
"""Restore gravity and contacts in controlled stages for G1 bimanual motion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


STAGES = {
    "A_no_gravity_no_contact": [],
    "B_gravity_no_contact": ["--enable-gravity"],
    "C_gravity_and_contact": ["--enable-gravity", "--enable-contacts"],
}


def build_fixed_base_asset(source: Path, destination: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(equality, "weld", name="warehouse_fixed_base", body1="pelvis")
    tree.write(destination, encoding="unicode", xml_declaration=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_asset = args.output_dir / "g1_warehouse_fixed_base.xml"
    build_fixed_base_asset(args.asset.resolve(), fixed_asset)

    stages = []
    for stage_name, flags in STAGES.items():
        stage_dir = args.output_dir / stage_name
        subprocess.run(
            [
                str(args.python),
                str(args.runner),
                "--asset",
                str(fixed_asset),
                "--output-dir",
                str(stage_dir),
                "--gain-profile",
                "unitree",
                "--dynamics-profile",
                "both",
                "--experiment-name",
                "G1WH-08-staged-realism-validation",
                "--base-mode",
                "model",
                *flags,
            ],
            check=True,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        report = json.loads((stage_dir / "actuation_audit.json").read_text())
        stages.append(
            {
                "stage": stage_name,
                "gravity_enabled": report["gravity_enabled"],
                "contacts_enabled": report["contacts_enabled"],
                "body_hold_rmse_rad": report["body_hold_rmse_rad"],
                "hand_hold_rmse_rad": report["hand_hold_rmse_rad"],
                "mean_contact_count": report["mean_contact_count"],
                "maximum_contact_count": report["maximum_contact_count"],
                "actuator_saturation_fraction": report["actuator_saturation_fraction"],
                "joint_limit_violation_fraction": report["joint_limit_violation_fraction"],
                "video": report["video"],
            }
        )

    summary = {
        "experiment": "G1WH-08-staged-realism-validation",
        "controlled_variables": "Asset, Unitree gains, regularized dynamics, targets, timing, fixed base",
        "fixed_base_asset": str(fixed_asset.resolve()),
        "stages": stages,
    }
    summary_path = args.output_dir / "realism_stage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
