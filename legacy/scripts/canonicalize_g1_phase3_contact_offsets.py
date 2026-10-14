#!/usr/bin/env python3
"""Remove camera-estimate noise from phase-3 physical-contact labels.

Only offline labels use the MuJoCo audit.  At deployment the learned VLA still
receives RGB geometry, language, and the robot state; its active-palm offset
is added to the RGB-localized target by the policy adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from build_g1_phase_candidate_physical_camera_recovery_cache import audited_contact_targets
from run_g1_language_pick_place import LEFT_PALM_INDEX, OBJECT_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--contact-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def offset_for_arm(values: dict, arm: int) -> torch.Tensor:
    offset = torch.as_tensor(values["site_offset"], dtype=torch.float32)
    if arm != int(values["audited_arm"]):
        offset = offset * torch.tensor((1.0, -1.0, 1.0), dtype=torch.float32)
    return offset


def canonicalize(cache: dict, targets: dict[str, dict]) -> tuple[dict, dict]:
    action = cache["action"].clone()
    phase3 = cache["phase"] == 3
    summary = {}
    for target, name in enumerate(OBJECT_NAMES):
        rows_for_target = phase3 & (cache["target"] == target)
        summary[name] = {}
        for arm in (LEFT_PALM_INDEX, 1 - LEFT_PALM_INDEX):
            rows = rows_for_target & (cache["active_arm"] == arm)
            count = int(rows.sum())
            if count:
                columns = slice(3 + arm * 3, 6 + arm * 3)
                action[rows, columns] = offset_for_arm(targets[name], arm)
            summary[name]["left" if arm == LEFT_PALM_INDEX else "right"] = count
    result = dict(cache)
    result["action"] = action
    result["action_contract"] = "active_target_passive_palm_residual"
    result["phase3_contact_labels"] = "MuJoCo-audited target-relative palm-site offsets; RGB error excluded from labels"
    result["active_target_coordinate_frame"] = "camera-localized target plus learned audited contact offset"
    return result, summary


def main() -> None:
    args = parse_args()
    targets = audited_contact_targets(args.contact_audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "G1LANG-canonical-phase3-physical-contact-offsets",
        "source_cache": str(args.source_cache),
        "contact_audit": str(args.contact_audit),
        "input_contract": "head-camera RGB geometry + language + state31 + measured palms",
        "action_contract": "learned audited active-palm offset applied to camera-localized target; passive-palm residual",
        "privileged_positions_at_inference": False,
        "splits": {},
    }
    for split in ("train", "val"):
        source = torch.load(args.source_cache / f"{split}.pt", map_location="cpu", weights_only=False)
        corrected, counts = canonicalize(source, targets)
        torch.save(corrected, args.output_dir / f"{split}.pt")
        summary["splits"][split] = {
            "samples": len(corrected["action"]),
            "phase3_labels_by_target_and_active_arm": counts,
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
