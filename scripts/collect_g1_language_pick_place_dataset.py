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

from run_g1_language_pick_place import OBJECT_NAMES

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
    parser.add_argument("--resume-successful", action="store_true")
    return parser.parse_args()


def required_episode_files(episode_dir: Path) -> tuple[Path, ...]:
    return (
        episode_dir / "language_pick_place_summary.json",
        episode_dir / "expert_pick_place_episode.npz",
        episode_dir / "head_camera.mp4",
        episode_dir / "left_wrist_camera.mp4",
        episode_dir / "right_wrist_camera.mp4",
    )


def collect_episode(args: argparse.Namespace, spec: dict) -> dict:
    episode_index = spec["episode_index"]
    episode_dir = args.output_dir / f"episode_{episode_index:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    summary_path = episode_dir / "language_pick_place_summary.json"
    existing = json.loads(summary_path.read_text()) if summary_path.exists() else None
    reuse = bool(
        args.resume_successful
        and existing
        and existing.get("passed")
        and existing.get("target_object") == spec["target_object"]
        and existing.get("slot_permutation") == spec["slot_permutation"]
        and np.allclose(
            existing.get("slot_offsets_xy_m"),
            spec["slot_offsets_xy_m"],
            atol=1e-12,
        )
        and existing.get("language_instruction") == spec["instruction"]
        and all(
            path.is_file() and path.stat().st_size > 1024
            for path in required_episode_files(episode_dir)
        )
    )
    if reuse:
        summary = existing
        print(f"Reusing episode {episode_index:04d}", flush=True)
    else:
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
    return {
        **spec,
        "passed": bool(summary and summary.get("passed")),
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    permutations = tuple(itertools.permutations(range(3)))
    episode_specs = []
    for episode_index in range(args.episodes):
        target = OBJECT_NAMES[episode_index % len(OBJECT_NAMES)]
        permutation = permutations[
            (episode_index // len(OBJECT_NAMES)) % len(permutations)
        ]
        instruction_variants = LANGUAGE_VARIANTS[target]
        instruction = instruction_variants[
            (episode_index // len(permutations)) % len(instruction_variants)
        ]
        offsets = rng.uniform(
            -args.position_jitter_m,
            args.position_jitter_m,
            size=(3, 2),
        )
        episode_specs.append(
            {
                "episode_index": episode_index,
                "target_object": target,
                "instruction": instruction,
                "slot_permutation": list(permutation),
                "slot_offsets_xy_m": offsets.tolist(),
            }
        )

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
        "seed": args.seed,
        "episodes": args.episodes,
        "successful_episodes": sum(item["passed"] for item in results),
        "success_rate": sum(item["passed"] for item in results) / len(results),
        "position_jitter_m": args.position_jitter_m,
        "targets": list(OBJECT_NAMES),
        "target_success_counts": target_counts,
        "language_variants": {
            key: list(value) for key, value in LANGUAGE_VARIANTS.items()
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
