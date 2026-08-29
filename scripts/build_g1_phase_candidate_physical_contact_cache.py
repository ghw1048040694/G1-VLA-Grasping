#!/usr/bin/env python3
"""Replace phase-3 labels with MuJoCo-audited physical contact poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from scipy.spatial.transform import Rotation


OBJECT_NAMES = ("red_triangle", "yellow_rod", "green_cube")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--contact-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def audit_targets(path: Path) -> dict[int, dict]:
    audit = json.loads(path.read_text())
    result = {}
    for index, name in enumerate(OBJECT_NAMES):
        entry = audit["results"][name]
        pose = entry.get("best_contact_pose")
        if pose is None or not pose.get("contact_geoms"):
            raise ValueError(f"Audit has no physical contact pose for {name}")
        center = torch.tensor(entry["object_center_m"], dtype=torch.float32)
        site = torch.tensor(pose["palm_site_target_m"], dtype=torch.float32)
        active_arm = 0 if entry["active_arm"] == "left" else 1
        orientation = Rotation.from_euler("z", float(pose["yaw_deg"]), degrees=True)
        result[index] = {
            "active_arm": active_arm,
            "site_offset": site - center,
            "orientation6": torch.tensor(
                orientation.as_matrix()[:, :2].reshape(-1), dtype=torch.float32
            ),
            "waist": torch.tensor(pose["arm_pose_rad"][:3], dtype=torch.float32),
            "audit": pose,
        }
    return result


def retarget(cache: dict, targets: dict[int, dict]) -> tuple[dict, dict]:
    required = {"action", "phase", "target", "active_arm"}
    missing = sorted(required - set(cache))
    if missing:
        raise ValueError(f"Cache is missing required fields: {missing}")
    action = cache["action"].clone()
    phase = cache["phase"]
    target = cache["target"]
    active_arm = cache["active_arm"]
    counts = {}
    for index, values in targets.items():
        rows = (phase == 3) & (target == index)
        by_arm = {"audited": 0, "mirrored": 0}
        for arm in (0, 1):
            arm_rows = rows & (active_arm == arm)
            if not bool(arm_rows.any()):
                continue
            if arm == int(values["active_arm"]):
                site_offset = values["site_offset"]
                orientation6 = values["orientation6"]
                waist = values["waist"]
                source = "audited"
            else:
                # G1 arms and palm pads are mirrored across the sagittal plane.
                site_offset = values["site_offset"] * torch.tensor((1.0, -1.0, 1.0))
                yaw = -float(values["audit"]["yaw_deg"])
                orientation6 = torch.tensor(
                    Rotation.from_euler("z", yaw, degrees=True)
                    .as_matrix()[:, :2]
                    .reshape(-1),
                    dtype=torch.float32,
                )
                waist = values["waist"] * torch.tensor((-1.0, -1.0, 1.0))
                source = "mirrored"
            position_start = 3 + arm * 3
            orientation_start = 9 + arm * 6
            action[arm_rows, :3] = waist
            action[arm_rows, position_start : position_start + 3] = site_offset
            action[arm_rows, orientation_start : orientation_start + 6] = orientation6
            by_arm[source] += int(arm_rows.sum())
        counts[OBJECT_NAMES[index]] = by_arm
    result = dict(cache)
    result["action"] = action
    result["contact_action_frame"] = "audited_physical_palm_site_contact_pose"
    return result, counts


def main() -> None:
    args = parse_args()
    targets = audit_targets(args.contact_audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "G1LANG-phase-candidate-physical-contact-cache",
        "source_cache": str(args.source_cache),
        "contact_audit": str(args.contact_audit),
        "action_contract": "learned phase-3 MuJoCo-audited palm-site pose with physical hand-object contact",
        "objects": {},
        "splits": {},
    }
    for index, values in targets.items():
        summary["objects"][OBJECT_NAMES[index]] = {
            "active_arm": "left" if values["active_arm"] == 0 else "right",
            "palm_site_offset_from_object_m": values["site_offset"].tolist(),
            "yaw_deg": float(values["audit"]["yaw_deg"]),
            "audit_pad_center_distance_m": float(values["audit"]["pad_center_distance_m"]),
        }
    for split in ("train", "val"):
        source = torch.load(args.source_cache / f"{split}.pt", map_location="cpu", weights_only=False)
        corrected, counts = retarget(source, targets)
        torch.save(corrected, args.output_dir / f"{split}.pt")
        summary["splits"][split] = {
            "samples": len(corrected["action"]),
            "phase3_retargeted_by_object": counts,
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
