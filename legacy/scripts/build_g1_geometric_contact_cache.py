#!/usr/bin/env python3
"""Create a compact camera-geometry cache from the audited contact cache."""

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
    parser.add_argument(
        "--action-contract",
        choices=("palm_residual", "target_relative"),
        default="palm_residual",
    )
    parser.add_argument(
        "--max-phase",
        type=int,
        help="Keep only labels at or before this expert phase for a curriculum cache.",
    )
    return parser.parse_args()


def build_split(
    source: Path,
    calibration: torch.Tensor,
    action_contract: str,
    max_phase: int | None,
) -> dict:
    cache = torch.load(source, map_location="cpu", weights_only=False)
    action_key = "contact_palm_lookahead_residual_action"
    required = (
        "state",
        "color_features",
        "palm_position",
        action_key,
        "target",
        "phase",
        "active_arm",
        "task_index",
        "vocab",
        "tasks",
    )
    missing = [key for key in required if key not in cache]
    if action_contract == "target_relative":
        missing.extend(
            key
            for key in ("contact_action", "object_position")
            if key not in cache and key not in missing
        )
    if missing:
        raise ValueError(f"Missing keys in {source}: {missing}")
    with torch.no_grad():
        objects = calibrated_object_positions(cache["color_features"].float(), calibration)
    result = {
        "state": cache["state"].float(),
        "color_features": cache["color_features"].float(),
        "visual_object_position": objects.reshape(len(objects), 9).float(),
        "palm_position": cache["palm_position"].float(),
        "action": cache[action_key][:, 0].float(),
        "target": cache["target"].long(),
        "phase": cache["phase"].long(),
        "active_arm": cache["active_arm"].long(),
        "task_index": cache["task_index"].long(),
        "vocab": cache["vocab"],
        "tasks": cache["tasks"],
    }
    if action_contract == "target_relative":
        # Labels may use the expert object position. This is an offline label
        # transform only; the policy later reconstructs absolute palm targets
        # from its camera estimate.
        action = cache["contact_action"][:, -1].float().clone()
        objects = cache["object_position"].float().reshape(-1, 3, 3)
        selected = objects[
            torch.arange(len(objects), dtype=torch.long), cache["target"].long()
        ]
        action[:, 3:6] -= selected
        action[:, 6:9] -= selected
        result["action"] = action
    if max_phase is not None:
        keep = result["phase"] <= max_phase
        if not bool(keep.any()):
            raise ValueError(f"No samples remain at --max-phase={max_phase}")
        count = len(keep)
        for key, value in list(result.items()):
            if isinstance(value, torch.Tensor) and value.ndim > 0 and len(value) == count:
                result[key] = value[keep]
    del cache
    gc.collect()
    return result


def main() -> None:
    args = parse_args()
    calibration_file = np.load(args.calibration)
    coefficients = torch.from_numpy(calibration_file["coefficients"].astype(np.float32))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_phase is not None and not 0 <= args.max_phase <= 7:
        raise ValueError("--max-phase must be in [0,7]")
    train = build_split(
        args.source_cache / "train.pt", coefficients, args.action_contract, args.max_phase
    )
    val = build_split(
        args.source_cache / "val.pt", coefficients, args.action_contract, args.max_phase
    )
    torch.save(train, args.output_dir / "train.pt")
    torch.save(val, args.output_dir / "val.pt")
    summary = {
        "experiment": "G1LANG_geometric_camera_contact_cache",
        "source_cache": str(args.source_cache),
        "calibration": str(args.calibration),
        "train_samples": len(train["state"]),
        "validation_samples": len(val["state"]),
        "input_contract": "head-camera RGB color geometry + language + 31D state + measured palm pose",
        "label_contract": (
            "strict expert Cartesian palm residual lookahead action"
            if args.action_contract == "palm_residual"
            else "strict expert Cartesian palm pose relative to selected object"
        ),
        "action_contract": args.action_contract,
        "maximum_expert_phase": args.max_phase,
        "privileged_positions_at_inference": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
