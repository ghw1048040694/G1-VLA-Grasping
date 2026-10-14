#!/usr/bin/env python3
"""Convert only passed contact-only jitter episodes into LeRobot datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert_g1_language_pick_place_to_lerobot import convert_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_contact_jitter6_train")
    parser.add_argument("--val-repo-id", default="local/g1_contact_jitter6_val")
    parser.add_argument("--train-episodes", type=int, default=12)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    episode_dirs = []
    rejected = []
    for directory in sorted(args.source_root.glob("episode_*")):
        summary_path = directory / "language_pick_place_summary.json"
        if not summary_path.is_file():
            rejected.append(directory.name)
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("passed"):
            episode_dirs.append(directory)
        else:
            rejected.append(directory.name)
    if not 0 < args.train_episodes < len(episode_dirs):
        raise ValueError(f"train split must leave validation episodes; successful={len(episode_dirs)}")
    train_dirs = episode_dirs[: args.train_episodes]
    val_dirs = episode_dirs[args.train_episodes :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "G1LANG-103-contact-jitter6-successful-lerobot",
        "source_root": str(args.source_root),
        "successful_source_episodes": len(episode_dirs),
        "rejected_source_episodes": rejected,
        "train": convert_split(train_dirs, args.train_repo_id, args.train_root, args.fps, args.overwrite),
        "validation": convert_split(val_dirs, args.val_repo_id, args.val_root, args.fps, args.overwrite),
    }
    report["passed"] = bool(
        report["train"]["action_dim"] == 31
        and report["validation"]["action_dim"] == 31
        and report["train"]["episodes"] == len(train_dirs)
        and report["validation"]["episodes"] == len(val_dirs)
    )
    (args.output_dir / "conversion_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("train", "validation")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
