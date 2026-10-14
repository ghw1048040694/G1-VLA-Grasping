#!/usr/bin/env python3
"""Build a language-balanced fixed-scene dataset from validated contact trials."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from collect_g1_language_pick_place_dataset import (
    DATASET_CONTRACT_VERSION,
    LANGUAGE_VARIANTS,
)
from run_g1_language_pick_place import EXPERT_CONTRACT_VERSION, OBJECT_NAMES


LINKED_FILES = (
    "expert_pick_place_episode.npz",
    "g1_language_pick_place.xml",
    "head_camera.mp4",
    "left_wrist_camera.mp4",
    "right_wrist_camera.mp4",
)


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-groups", type=int, default=20)
    args = parser.parse_args()
    if args.scene_groups < 3:
        raise ValueError("--scene-groups must be at least three")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    canonical = {}
    for target_index, target in enumerate(OBJECT_NAMES):
        source = args.source_root / f"episode_{target_index:04d}"
        summary = json.loads(
            (source / "language_pick_place_summary.json").read_text()
        )
        audit = summary.get("grasp_physics_audit", {})
        if not (
            summary.get("passed")
            and summary.get("expert_contract_version") == EXPERT_CONTRACT_VERSION
            and summary.get("target_object") == target
            and summary.get("grasp_mode") == "contact"
            and audit.get("assisted_constraint_active_steps") == 0
            and audit.get("physical_grasp_passed")
        ):
            raise ValueError(f"Canonical contact episode did not pass: {source}")
        canonical[target] = (source, summary)

    specs = []
    for scene_index in range(args.scene_groups):
        variant_index = scene_index % len(next(iter(LANGUAGE_VARIANTS.values())))
        for target_index, target in enumerate(OBJECT_NAMES):
            episode_index = scene_index * len(OBJECT_NAMES) + target_index
            episode_dir = args.output_root / f"episode_{episode_index:04d}"
            episode_dir.mkdir()
            source, source_summary = canonical[target]
            for name in LINKED_FILES:
                link_or_copy(source / name, episode_dir / name)
            instruction = LANGUAGE_VARIANTS[target][variant_index]
            spec = {
                "dataset_contract_version": DATASET_CONTRACT_VERSION,
                "episode_index": episode_index,
                "scene_index": scene_index,
                "target_object": target,
                "language_variant_index": variant_index,
                "instruction": instruction,
                "slot_permutation": [0, 1, 2],
                "slot_offsets_xy_m": [[0.0, 0.0] for _ in OBJECT_NAMES],
            }
            (episode_dir / "collection_spec.json").write_text(
                json.dumps(spec, indent=2) + "\n", encoding="utf-8"
            )
            summary = dict(source_summary)
            summary["language_instruction"] = instruction
            summary["scene"] = str(episode_dir / "g1_language_pick_place.xml")
            summary["video"] = None
            (episode_dir / "language_pick_place_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            specs.append({**spec, "passed": True, "episode_dir": str(episode_dir)})

    target_counts = {target: args.scene_groups for target in OBJECT_NAMES}
    report = {
        "experiment": "G1-Language-Contact-Only-Calibration-Dataset",
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "expert_contract_version": EXPERT_CONTRACT_VERSION,
        "seed": None,
        "episodes": len(specs),
        "successful_episodes": len(specs),
        "success_rate": 1.0,
        "position_jitter_m": 0.0,
        "fixed_scene_calibration": True,
        "canonical_source_root": str(args.source_root),
        "paired_scene_groups": args.scene_groups,
        "paired_scene_contract_valid": True,
        "targets": list(OBJECT_NAMES),
        "target_success_counts": target_counts,
        "language_variants": {
            key: list(value) for key, value in LANGUAGE_VARIANTS.items()
        },
        "episode_specs": specs,
        "passed": True,
    }
    (args.output_root / "language_pick_place_dataset_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_root / "episodes.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in specs), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "episode_specs"}, indent=2))


if __name__ == "__main__":
    main()
