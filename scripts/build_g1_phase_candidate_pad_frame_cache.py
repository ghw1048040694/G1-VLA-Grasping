#!/usr/bin/env python3
"""Retarget phase-3 VLA labels from palm centers to physical contact pads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def project_rotation_columns(values: torch.Tensor) -> torch.Tensor:
    """Match the Gram-Schmidt 6D rotation projection used by deployment IK."""
    columns = values.reshape(-1, 3, 2).to(dtype=torch.float32)
    first = torch.nn.functional.normalize(columns[:, :, 0], dim=-1, eps=1e-8)
    second = columns[:, :, 1] - first * (first * columns[:, :, 1]).sum(-1, keepdim=True)
    second = torch.nn.functional.normalize(second, dim=-1, eps=1e-8)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def retarget(cache: dict) -> tuple[dict, int]:
    required = {"action", "phase", "active_arm"}
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(f"Cache is missing required fields: {missing}")
    action = cache["action"].clone()
    phase = cache["phase"]
    active_arm = cache["active_arm"]
    changed = 0
    for arm, lateral_offset in ((0, -0.003), (1, 0.003)):
        rows = (phase == 3) & (active_arm == arm)
        if not bool(rows.any()):
            continue
        orientation_start = 9 + arm * 6
        positions_start = 3 + arm * 3
        rotation = project_rotation_columns(
            action[rows, orientation_start : orientation_start + 6]
        )
        local_site_from_pad = torch.tensor(
            (0.070, lateral_offset, 0.0), dtype=action.dtype
        )
        world_site_from_pad = rotation @ local_site_from_pad
        action[rows, positions_start : positions_start + 3] += world_site_from_pad
        changed += int(rows.sum())
    result = dict(cache)
    result["action"] = action
    result["contact_action_frame"] = "palm_site_target_for_physical_contact_pad"
    return result, changed


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    totals = {}
    for split in ("train", "val"):
        source = torch.load(args.source_cache / f"{split}.pt", map_location="cpu", weights_only=False)
        corrected, changed = retarget(source)
        torch.save(corrected, args.output_dir / f"{split}.pt")
        totals[split] = {"samples": len(corrected["action"]), "phase3_retargeted": changed}
    summary = {
        "experiment": "G1LANG-phase-candidate-contact-pad-frame-cache",
        "source_cache": str(args.source_cache),
        "action_contract": "learned phase-3 palm-site target whose physical contact pad aligns with the object",
        "site_from_pad_local_m": {
            "left": [0.070, -0.003, 0.0],
            "right": [0.070, 0.003, 0.0],
        },
        "splits": totals,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
