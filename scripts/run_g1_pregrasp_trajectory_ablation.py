#!/usr/bin/env python3
"""Build the selected scene and compare continuous pre-grasp timings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


GROUPS = (
    ("x045_simultaneous", 0.45, 0.0),
    ("x045_shoulder_lead_250ms", 0.45, 0.25),
    ("x055_simultaneous", 0.55, 0.0),
    ("x055_shoulder_lead_250ms", 0.55, 0.25),
)


def build_selected_scene(source: Path, destination: Path, tote_x: float) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    tote = root.find(".//body[@name='warehouse_tote']")
    table = root.find(".//geom[@name='calibration_table']")
    if tote is None or table is None:
        raise RuntimeError("Reachability asset is missing the tote or table")
    tote_pos = [float(value) for value in tote.get("pos", "").split()]
    table_pos = [float(value) for value in table.get("pos", "").split()]
    tote_pos[0] = tote_x
    table_pos[0] = tote_x + 0.15
    tote.set("pos", " ".join(str(value) for value in tote_pos))
    table.set("pos", " ".join(str(value) for value in table_pos))
    tree.write(destination, encoding="unicode", xml_declaration=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reachability = json.loads(args.reachability_report.read_text())
    if reachability["selected_tote_x_m"] is None:
        raise RuntimeError("G1WH-14 did not select a reachable tote position")

    groups = []
    for group_name, tote_x, lead in GROUPS:
        group_dir = args.output_dir / group_name
        selected_asset = group_dir / "g1_warehouse_pregrasp.xml"
        group_dir.mkdir(parents=True, exist_ok=True)
        build_selected_scene(args.asset.resolve(), selected_asset, tote_x)
        subprocess.run(
            [
                str(args.python),
                str(args.runner),
                "--asset",
                str(selected_asset),
                "--reachability-report",
                str(args.reachability_report.resolve()),
                "--output-dir",
                str(group_dir),
                "--tote-x",
                str(tote_x),
                "--shoulder-lead-seconds",
                str(lead),
            ],
            check=True,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        report = json.loads((group_dir / "pregrasp_trajectory_audit.json").read_text())
        groups.append({"group": group_name, **report})

    passing = [group for group in groups if group["passed"]]
    selected = (
        max(
            passing,
            key=lambda item: (item["selected_tote_x_m"], -item["shoulder_lead_seconds"]),
        )
        if passing
        else None
    )
    summary = {
        "experiment": "G1WH-15-continuous-pregrasp-trajectory",
        "independent_variables": ["selected_tote_x_m", "shoulder_lead_seconds"],
        "selection_rule": (
            "Farthest tote x, then shortest shoulder lead, among passing groups"
        ),
        "groups": groups,
        "selected_group": selected["group"] if selected else None,
        "experiment_passed": selected is not None,
    }
    summary_path = args.output_dir / "pregrasp_trajectory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
