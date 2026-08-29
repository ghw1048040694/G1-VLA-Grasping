#!/usr/bin/env python3
"""Re-split one action-token cache by the underlying policy-rollout group."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def group_for_root(root: str) -> str:
    spec_path = Path(root) / "recovery_spec.json"
    if spec_path.is_file():
        spec = json.loads(spec_path.read_text())
        if spec.get("source_episode"):
            return str(spec["source_episode"])
    summary_path = Path(root) / "language_pick_place_summary.json"
    summary = json.loads(summary_path.read_text())
    source = summary.get("initial_state_recovery", {}).get("source")
    if source:
        return str(source).split("/policy_visited_state.npz")[0]
    return root


def subset(source: dict, indices: list[int], output_dir: Path, split: str) -> dict:
    result = dict(source)
    episode_count = int(source["target"].shape[0])
    for key, value in source.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == episode_count:
            result[key] = value[indices].clone()
        elif isinstance(value, list) and len(value) == episode_count:
            result[key] = [value[index] for index in indices]
    images = np.load(source["image_path"], mmap_mode="r")
    image_path = output_dir / f"{split}_images.npy"
    out = np.lib.format.open_memmap(
        image_path, mode="w+", dtype=images.dtype, shape=(len(indices), *images.shape[1:])
    )
    out[:] = images[indices]
    out.flush()
    result["image_path"] = str(image_path)
    torch.save(result, output_dir / f"{split}.pt")
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between zero and one")
    source = torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=False)
    groups: dict[str, list[int]] = {}
    for index, root in enumerate(source["source_root"]):
        groups.setdefault(group_for_root(root), []).append(index)
    names = np.asarray(sorted(groups))
    rng = np.random.default_rng(args.seed)
    count = min(len(names) - 1, max(1, round(len(names) * args.validation_ratio)))
    validation_groups = set(rng.choice(names, count, replace=False).tolist())
    train_indices = [index for group, indices in groups.items() if group not in validation_groups for index in indices]
    validation_indices = [index for group, indices in groups.items() if group in validation_groups for index in indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = subset(source, train_indices, args.output_dir, "train")
    validation = subset(source, validation_indices, args.output_dir, "validation")
    summary = {
        "experiment": "G1LANG-source-group-resplit-action-token-cache",
        "source_cache": str(args.cache_dir),
        "groups": len(groups),
        "train_groups": len(groups) - len(validation_groups),
        "validation_groups": len(validation_groups),
        "train_episodes": len(train_indices),
        "validation_episodes": len(validation_indices),
        "train_rows": int(train["valid_frame"].sum()),
        "validation_rows": int(validation["valid_frame"].sum()),
        "validation_group_names": sorted(validation_groups),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
