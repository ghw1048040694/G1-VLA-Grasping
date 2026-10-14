#!/usr/bin/env python3
"""Collect the production G1 language-grounded pick-and-place dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from run_g1_language_pick_place import EXPERT_CONTRACT_VERSION, OBJECT_NAMES

DATASET_CONTRACT_VERSION = "g1lang_dataset_v4_contact_profile_triplets"
LANGUAGE_VARIANTS = {
    "red_triangle": (
        "put the red triangular prism into the blue box",
        "place the red triangle inside the blue container",
        "move the red triangular object into the blue bin",
    ),
    "yellow_rod": (
        "put the yellow rod into the blue box",
        "place the yellow stick inside the blue container",
        "move the yellow cylindrical object into the blue bin",
    ),
    "green_cube": (
        "put the green cube into the blue box",
        "place the green block inside the blue container",
        "move the green cubic object into the blue bin",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--position-jitter-m", type=float, default=0.018)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--render-skip",
        type=int,
        default=3,
        help="Render every Nth simulation step while preserving the video frame rate.",
    )
    parser.add_argument("--resume-successful", action="store_true")
    parser.add_argument(
        "--fixed-scene",
        action="store_true",
        help="Use the calibrated identity slot arrangement with zero jitter.",
    )
    parser.add_argument(
        "--center-slot-x-shift-m",
        type=float,
        default=0.0,
        help="Shift the middle workspace slot in x before applying jitter.",
    )
    parser.add_argument(
        "--red-object-x-shift-m",
        type=float,
        default=0.0,
        help="Shift the red triangle in x before applying jitter.",
    )
    parser.add_argument(
        "--red-outer-y-jitter-m",
        type=float,
        default=None,
        help="Optional narrower y jitter for red triangles in outer slots.",
    )
    parser.add_argument(
        "--yellow-left-x-shift-m",
        type=float,
        default=0.0,
        help="Shift yellow rods inward when assigned to the left outer slot.",
    )
    parser.add_argument(
        "--yellow-outer-y-jitter-m",
        type=float,
        default=None,
        help="Optional narrower y jitter for yellow rods in the right outer slot.",
    )
    parser.add_argument(
        "--yellow-right-x-shift-m",
        type=float,
        default=0.0,
        help="Shift yellow rods inward in the right outer slot.",
    )
    parser.add_argument(
        "--yellow-right-y-shift-m",
        type=float,
        default=0.0,
        help="Shift yellow rods toward the box in the right outer slot.",
    )
    return parser.parse_args()


def required_episode_files(episode_dir: Path) -> dict[Path, int]:
    return {
        episode_dir / "collection_spec.json": 100,
        episode_dir / "language_pick_place_summary.json": 1024,
        episode_dir / "expert_pick_place_episode.npz": 1024,
        episode_dir / "head_camera.mp4": 1024,
        episode_dir / "left_wrist_camera.mp4": 1024,
        episode_dir / "right_wrist_camera.mp4": 1024,
    }


def build_episode_specs(
    episodes: int,
    seed: int,
    position_jitter_m: float,
    fixed_scene: bool = False,
    center_slot_x_shift_m: float = 0.0,
    red_object_x_shift_m: float = 0.0,
    red_outer_y_jitter_m: float | None = None,
    yellow_left_x_shift_m: float = 0.0,
    yellow_outer_y_jitter_m: float | None = None,
    yellow_right_x_shift_m: float = 0.0,
    yellow_right_y_shift_m: float = 0.0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    permutations = tuple(itertools.permutations(range(len(OBJECT_NAMES))))
    specs = []
    for scene_index in range(episodes // len(OBJECT_NAMES)):
        permutation = (0, 1, 2) if fixed_scene else permutations[scene_index % len(permutations)]
        offsets = (
            np.zeros((len(OBJECT_NAMES), 2))
            if fixed_scene
            else rng.uniform(
                -position_jitter_m,
                position_jitter_m,
                size=(len(OBJECT_NAMES), 2),
            )
        )
        if abs(center_slot_x_shift_m) > 0.0:
            for object_index, slot_index in enumerate(permutation):
                # The red triangle's right-hand center grasp is calibrated at
                # the nominal x=0.34 m boundary; the yellow/green center
                # profiles need the +2 cm inward shift.
                if slot_index == 1 and object_index != 0:
                    offsets[object_index, 0] += center_slot_x_shift_m
        if permutation[0] == 1:
            # The right-hand red-triangle center grasp has a narrow x
            # feasibility window; retain y jitter while pinning x at 0.34 m.
            offsets[0, 0] = 0.0
        else:
            # Both outer red-triangle grasps use the corresponding nominal
            # x plus a small fixed inward margin; y remains randomized.
            offsets[0, 0] = red_object_x_shift_m
            if red_outer_y_jitter_m is not None:
                offsets[0, 1] = float(
                    np.clip(offsets[0, 1], -red_outer_y_jitter_m, red_outer_y_jitter_m)
                )
        if permutation[1] == 2:
            offsets[1, 0] += yellow_left_x_shift_m
        elif permutation[1] == 1:
            # The yellow rod center grasp uses x=0.36 m; retain y jitter but
            # avoid the lower-x edge where the rod loses the middle finger.
            offsets[1, 0] = center_slot_x_shift_m
        elif yellow_outer_y_jitter_m is not None:
            offsets[1, 1] = float(
                np.clip(offsets[1, 1], -yellow_outer_y_jitter_m, yellow_outer_y_jitter_m)
            )
        if permutation[1] == 0:
            offsets[1, 0] += yellow_right_x_shift_m
            offsets[1, 1] += yellow_right_y_shift_m
        permutation_cycle = scene_index // len(permutations)
        language_variant_index = (
            permutation_cycle + scene_index % len(permutations)
        ) % len(next(iter(LANGUAGE_VARIANTS.values())))
        for target_index, target in enumerate(OBJECT_NAMES):
            slot_index = permutation[target_index]
            profile = contact_profile_for_target_slot(target, slot_index)
            specs.append(
                {
                    "dataset_contract_version": DATASET_CONTRACT_VERSION,
                    "episode_index": scene_index * len(OBJECT_NAMES) + target_index,
                    "scene_index": scene_index,
                    "target_object": target,
                    "language_variant_index": language_variant_index,
                    "instruction": LANGUAGE_VARIANTS[target][language_variant_index],
                    "slot_permutation": list(permutation),
                    "slot_offsets_xy_m": offsets.tolist(),
                    "contact_profile": profile,
                }
            )
    return specs


def contact_profile_for_target_slot(target: str, slot_index: int) -> dict:
    """Return the calibrated physical-contact expert profile for one slot."""
    profile = {
        "body_gain_scale": 5.0,
        "hand_gain_scale": 6.0,
        "grasp_yaw_deg": 45.0 if target == "green_cube" else 0.0,
        "contact_grasp_compensation": "-0.015,0.030,-0.055",
        "hand_closure_multiplier": 1.0,
    }
    if slot_index == 1:
        if target == "red_triangle":
            profile.update(
                grasp_yaw_deg=30.0,
                contact_grasp_compensation="-0.015,-0.030,-0.055",
            )
        elif target == "green_cube":
            profile["hand_closure_multiplier"] = 1.2
    elif slot_index == 2:
        if target == "red_triangle":
            profile["grasp_yaw_deg"] = 30.0
        elif target == "yellow_rod":
            profile.update(grasp_yaw_deg=30.0, hand_closure_multiplier=0.9)
    elif slot_index == 0 and target == "yellow_rod":
        profile.update(
            grasp_yaw_deg=90.0,
            contact_grasp_compensation="0.015,-0.050,-0.055",
        )
    return profile


def validate_paired_scene_groups(specs: list[dict]) -> None:
    if len(specs) % len(OBJECT_NAMES) != 0:
        raise ValueError("Episode specs do not form complete target triplets")
    for group_start in range(0, len(specs), len(OBJECT_NAMES)):
        group = specs[group_start : group_start + len(OBJECT_NAMES)]
        reference = group[0]
        if [item["target_object"] for item in group] != list(OBJECT_NAMES):
            raise ValueError(f"Scene group {reference['scene_index']} has wrong targets")
        for item in group:
            if item["scene_index"] != reference["scene_index"]:
                raise ValueError("Scene group contains multiple scene indices")
            if item["slot_permutation"] != reference["slot_permutation"]:
                raise ValueError("Scene group contains multiple slot permutations")
            if item["slot_offsets_xy_m"] != reference["slot_offsets_xy_m"]:
                raise ValueError("Scene group contains multiple continuous jitters")
            if item["language_variant_index"] != reference["language_variant_index"]:
                raise ValueError("Scene group contains multiple language variants")


def episode_spec_payload(spec: dict) -> dict:
    return {
        key: spec[key]
        for key in (
            "dataset_contract_version",
            "episode_index",
            "scene_index",
            "target_object",
            "language_variant_index",
            "instruction",
            "slot_permutation",
            "slot_offsets_xy_m",
            "contact_profile",
        )
    }


def summary_matches_spec(summary: dict | None, spec: dict) -> bool:
    return bool(
        summary
        and summary.get("passed")
        and summary.get("expert_contract_version") == EXPERT_CONTRACT_VERSION
        and summary.get("target_object") == spec["target_object"]
        and summary.get("slot_permutation") == spec["slot_permutation"]
        and np.allclose(
            summary.get("slot_offsets_xy_m"),
            spec["slot_offsets_xy_m"],
            rtol=0.0,
            atol=5e-8,
        )
        and summary.get("language_instruction") == spec["instruction"]
    )


def collect_episode(args: argparse.Namespace, spec: dict) -> dict:
    episode_index = spec["episode_index"]
    episode_dir = args.output_dir / f"episode_{episode_index:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    summary_path = episode_dir / "language_pick_place_summary.json"
    spec_path = episode_dir / "collection_spec.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else None
    existing_spec = json.loads(spec_path.read_text()) if spec_path.exists() else None
    reuse = bool(
        args.resume_successful
        and existing_spec == episode_spec_payload(spec)
        and summary_matches_spec(existing, spec)
        and all(
            path.is_file() and path.stat().st_size > minimum_size
            for path, minimum_size in required_episode_files(episode_dir).items()
        )
    )
    if reuse:
        summary = existing
        print(f"Reusing episode {episode_index:04d}", flush=True)
    else:
        profile = spec["contact_profile"]
        command = [
            str(args.python),
            str(args.collector),
            "--asset",
            str(args.asset.resolve()),
            "--output-dir",
            str(episode_dir),
            "--target-object",
            spec["target_object"],
            "--slot-permutation="
            + ",".join(str(value) for value in spec["slot_permutation"]),
            "--slot-offsets="
            + ",".join(
                f"{value:.8f}" for pair in spec["slot_offsets_xy_m"] for value in pair
            ),
            "--language-instruction",
            spec["instruction"],
            "--record-demonstration",
            "--video-fps",
            "15",
            "--render-width",
            "320",
            "--render-height",
            "240",
            "--render-skip",
            str(args.render_skip),
            "--disable-overview-video",
            "--grasp-mode",
            "contact",
            "--body-gain-scale",
            str(profile["body_gain_scale"]),
            "--hand-gain-scale",
            str(profile["hand_gain_scale"]),
            "--grasp-yaw-deg",
            str(profile["grasp_yaw_deg"]),
            "--contact-grasp-compensation="
            + profile["contact_grasp_compensation"],
            "--hand-closure-multiplier",
            str(profile["hand_closure_multiplier"]),
        ]
        completed = subprocess.run(
            command,
            check=False,
            env={**os.environ, "MUJOCO_GL": "egl"},
        )
        summary = (
            json.loads(summary_path.read_text())
            if completed.returncode == 0 and summary_path.exists()
            else None
        )
        if summary_matches_spec(summary, spec):
            spec_path.write_text(
                json.dumps(episode_spec_payload(spec), indent=2) + "\n",
                encoding="utf-8",
            )
    return {
        **spec,
        "passed": summary_matches_spec(summary, spec),
        "episode_dir": str(episode_dir),
    }


def main() -> None:
    args = parse_args()
    if args.episodes < 3 or args.episodes % len(OBJECT_NAMES) != 0:
        raise ValueError("--episodes must be at least three and divisible by three")
    if not 0.0 <= args.position_jitter_m <= 0.025:
        raise ValueError("--position-jitter-m must be between 0 and 0.025")
    if not 1 <= args.workers <= 4:
        raise ValueError("--workers must be between one and four")
    if args.render_skip < 1:
        raise ValueError("--render-skip must be at least one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_specs = build_episode_specs(
        args.episodes,
        args.seed,
        args.position_jitter_m,
        args.fixed_scene,
        args.center_slot_x_shift_m,
        args.red_object_x_shift_m,
        args.red_outer_y_jitter_m,
        args.yellow_left_x_shift_m,
        args.yellow_outer_y_jitter_m,
        args.yellow_right_x_shift_m,
        args.yellow_right_y_shift_m,
    )
    validate_paired_scene_groups(episode_specs)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(collect_episode, args, spec): spec for spec in episode_specs
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            completed_count = len(results)
            successful_count = sum(item["passed"] for item in results)
            print(
                f"DATASET_PROGRESS={completed_count}/{args.episodes} "
                f"passed={successful_count}/{completed_count}",
                flush=True,
            )
    results.sort(key=lambda item: item["episode_index"])
    failed = [item["episode_index"] for item in results if not item["passed"]]
    if failed:
        raise RuntimeError(f"Production episodes failed: {failed}")

    target_counts = {
        target: sum(
            item["target_object"] == target and item["passed"] for item in results
        )
        for target in OBJECT_NAMES
    }
    summary = {
        "experiment": "G1-Language-Grounded-Manipulation-Production-Dataset",
        "dataset_contract_version": DATASET_CONTRACT_VERSION,
        "expert_contract_version": EXPERT_CONTRACT_VERSION,
        "seed": args.seed,
        "episodes": args.episodes,
        "successful_episodes": sum(item["passed"] for item in results),
        "success_rate": sum(item["passed"] for item in results) / len(results),
        "position_jitter_m": args.position_jitter_m,
        "center_slot_x_shift_m": args.center_slot_x_shift_m,
        "red_object_x_shift_m": args.red_object_x_shift_m,
        "red_outer_y_jitter_m": args.red_outer_y_jitter_m,
        "yellow_left_x_shift_m": args.yellow_left_x_shift_m,
        "yellow_outer_y_jitter_m": args.yellow_outer_y_jitter_m,
        "yellow_right_x_shift_m": args.yellow_right_x_shift_m,
        "yellow_right_y_shift_m": args.yellow_right_y_shift_m,
        "paired_scene_groups": args.episodes // len(OBJECT_NAMES),
        "paired_scene_contract_valid": True,
        "scene_group_shared_fields": [
            "slot_permutation",
            "slot_offsets_xy_m",
            "language_variant_index",
        ],
        "targets": list(OBJECT_NAMES),
        "target_success_counts": target_counts,
        "language_variants": {
            key: list(value) for key, value in LANGUAGE_VARIANTS.items()
        },
        "language_variant_scene_counts": {
            str(index): sum(
                item["target_object"] == OBJECT_NAMES[0]
                and item["language_variant_index"] == index
                for item in results
            )
            for index in range(len(next(iter(LANGUAGE_VARIANTS.values()))))
        },
        "episode_specs": results,
        "passed": all(item["passed"] for item in results),
    }
    summary_path = args.output_dir / "language_pick_place_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest_path = args.output_dir / "episodes.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item) + "\n" for item in results), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "episode_specs"},
            indent=2,
        )
    )
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
