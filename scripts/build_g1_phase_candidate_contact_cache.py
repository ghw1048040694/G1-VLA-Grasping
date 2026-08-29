#!/usr/bin/env python3
"""Build compact geometry labels for phase-candidate contact actions.

The active grasping palm is expressed relative to the language-selected object.
The passive palm is expressed relative to its measured current pose.  This
prevents the passive arm from being pulled across the workspace when a local
target-relative grasp policy is evaluated away from an expert frame.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from g1_geometric_contact_vla import calibrated_object_positions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-phase", type=int, default=3)
    return parser.parse_args()


def build_split(source: Path, calibration: torch.Tensor, max_phase: int) -> dict:
    cache = torch.load(source, map_location="cpu", weights_only=False)
    required = {
        "state", "color_features", "palm_position", "contact_action",
        "object_position", "target", "phase", "active_arm", "task_index",
        "vocab", "tasks",
    }
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(f"Missing keys in {source}: {missing}")
    action = cache["contact_action"][:, -1].float().clone()
    objects = cache["object_position"].float().reshape(-1, 3, 3)
    target = cache["target"].long()
    active_arm = cache["active_arm"].long()
    selected = objects[torch.arange(len(objects)), target]
    # Each label keeps both learned palms. Only their coordinate frames differ.
    # The active palm has an object-relative grasp target; the other action is
    # a local visual-servo residual from its measured current palm position.
    for arm in range(2):
        indices = slice(3 + arm * 3, 6 + arm * 3)
        active = active_arm == arm
        action[active, indices] -= selected[active]
        action[~active, indices] -= cache["palm_position"][~active, arm * 3 : arm * 3 + 3]
    with torch.no_grad():
        visual = calibrated_object_positions(cache["color_features"].float(), calibration)
    result = {
        "state": cache["state"].float(),
        "color_features": cache["color_features"].float(),
        "visual_object_position": visual.reshape(len(visual), 9).float(),
        "palm_position": cache["palm_position"].float(),
        "action": action,
        "target": target,
        "phase": cache["phase"].long(),
        "active_arm": active_arm,
        "task_index": cache["task_index"].long(),
        "vocab": cache["vocab"],
        "tasks": cache["tasks"],
        "action_contract": "active_target_passive_palm_residual",
    }
    keep = result["phase"] <= max_phase
    if not bool(keep.any()):
        raise ValueError(f"No labels at or below phase {max_phase}")
    count = len(keep)
    for key, value in list(result.items()):
        if isinstance(value, torch.Tensor) and value.ndim > 0 and len(value) == count:
            result[key] = value[keep]
    del cache
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    if not 0 <= args.max_phase <= 7:
        raise ValueError("--max-phase must be in [0,7]")
    coefficients = torch.from_numpy(
        np.load(args.calibration)["coefficients"].astype(np.float32)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = build_split(args.source_cache / "train.pt", coefficients, args.max_phase)
    val = build_split(args.source_cache / "val.pt", coefficients, args.max_phase)
    torch.save(train, args.output_dir / "train.pt")
    torch.save(val, args.output_dir / "val.pt")
    summary = {
        "experiment": "G1LANG-phase-candidate-mixed-contact-cache",
        "source_cache": str(args.source_cache),
        "calibration": str(args.calibration),
        "train_samples": len(train["state"]),
        "validation_samples": len(val["state"]),
        "maximum_expert_phase": args.max_phase,
        "input_contract": "head-camera RGB geometry + language + 31D state + measured palm pose",
        "label_contract": "active palm relative to expert target; passive palm residual from measured palm",
        "privileged_positions_at_inference": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
