#!/usr/bin/env python3
"""Convert one G1FINAL-07 hybrid episode to the G1SIM-03 source contract."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


LOWER_BODY_JOINTS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_summary_path = args.episode / "language_pick_place_summary.json"
    source_data_path = args.episode / "expert_pick_place_episode.npz"
    source_scene_path = args.episode / "g1_language_pick_place.xml"
    for path in (source_summary_path, source_data_path, source_scene_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if not source_summary.get("passed"):
        raise ValueError("G1SIM-03 source must be a strict-success episode")

    with np.load(source_data_path) as source:
        joint_names = source["joint_names"].astype(str).tolist()
        observations = source["joint_position_rad"].astype(np.float32)
        actions = source["action_joint_position_rad"].astype(np.float32)
        phases = source["task_phase"].astype(np.int64)
        assist_active = source["assist_active"].astype(bool)
        active_arm = source["active_arm_index"].astype(np.int64)
        object_positions = source["object_position_m"].astype(np.float32)
        target_indices = source["target_object_index"].astype(np.int64)

    if observations.ndim != 2 or observations.shape[1] != len(joint_names):
        raise ValueError("Unexpected source joint-position shape")
    if actions.shape != observations.shape:
        raise ValueError("Source observation/action shapes differ")
    if len(joint_names) < LOWER_BODY_JOINTS:
        raise ValueError("Source trajectory has no lower-body prefix")
    upper_names = joint_names[LOWER_BODY_JOINTS:]
    upper_observations = observations[:, LOWER_BODY_JOINTS:]
    upper_actions = actions[:, LOWER_BODY_JOINTS:]
    object_names = list(source_summary["object_order"])
    target_index = int(np.unique(target_indices)[0])
    if target_index != object_names.index(source_summary["target_object"]):
        raise ValueError("Source target index does not match source summary")

    grabbed_object_index = np.full(len(assist_active), -1, dtype=np.int8)
    grabbed_object_index[assist_active] = target_index
    grabbed_arm_index = np.full(len(assist_active), -1, dtype=np.int8)
    grabbed_arm_index[assist_active] = active_arm[assist_active]
    control_fps = 15
    args.output.mkdir(parents=True, exist_ok=True)
    # Isaac Sim's MJCF importer can exit inside native code on inline mesh
    # assets. Keep the target's size/color, but use a primitive for the replay
    # scene; the original MuJoCo scene and video retain the triangular mesh.
    scene_tree = ET.parse(source_scene_path)
    root = scene_tree.getroot()
    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") == "red_triangle_prism_mesh":
                asset.remove(mesh)
    for geom in root.iter("geom"):
        if geom.get("name") == "red_triangle_visual":
            geom.attrib.pop("mesh", None)
            geom.set("type", "box")
            geom.set("size", "0.045 0.040 0.030")
    equality = root.find("equality")
    if equality is not None:
        target_prefix = f"{source_summary['target_object']}_"
        for constraint in list(equality):
            name = constraint.get("name", "")
            if "assisted_grasp" in name and not name.startswith(target_prefix):
                equality.remove(constraint)
    scene_tree.write(args.output / "scene.xml", encoding="unicode")
    (args.output / "summary.json").write_text(
        json.dumps(source_summary, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "trajectory.npz",
        joint_names=np.asarray(upper_names),
        observation_joint_position_rad=upper_observations,
        action_joint_position_rad=upper_actions,
        scheduled_phase=phases,
        assist_active=assist_active.astype(np.int8),
        grabbed_object_index=grabbed_object_index,
        grabbed_arm_index=grabbed_arm_index,
        object_names=np.asarray(object_names),
        object_position_m=object_positions,
        task_target_index=target_indices,
        control_fps=np.asarray(control_fps, dtype=np.int64),
    )
    manifest = {
        "source_experiment": "G1FINAL-07-learned-router-classical-controller",
        "source_episode": str(args.episode.resolve()),
        "controller": "classical_ik_interpolation_assisted_grasp",
        "target_object": source_summary["target_object"],
        "upper_body_action_dim": len(upper_names),
        "frames": len(upper_actions),
        "strict_success": True,
        "scope": "G1SIM-03 source conversion for offline hybrid trajectory replay",
        "isaac_scene_sanitization": (
            "inline red triangle mesh replaced by box primitive; non-target "
            "assisted-grasp constraints removed"
        ),
    }
    (args.output / "conversion_summary.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"Prepared source episode {args.output}")


if __name__ == "__main__":
    main()
