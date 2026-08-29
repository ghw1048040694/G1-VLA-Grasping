#!/usr/bin/env python3
"""Merge action-token VLA caches while remapping language task IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from build_g1_action_token_cache_from_expert_npz import vocabulary


PAD_FIELDS = (
    "state",
    "velocity",
    "action_delta",
    "objects",
    "contact_sites",
    "contact",
    "active_arm",
    "valid_frame",
    "sample_weight",
    "task_phase",
)
EPISODE_FIELDS = ("target", "task_index")
TEXT_FIELDS = ("episode_name", "source_root")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat-last", type=int, default=1)
    return parser.parse_args()


def load_split(cache_dirs: list[Path], split: str, repeat_last: int) -> list[tuple[Path, dict]]:
    items: list[tuple[Path, dict]] = []
    expanded = list(cache_dirs)
    # Reweight training only. Duplicating validation episodes makes metrics
    # look more certain without adding any independent validation evidence.
    if split == "train" and len(cache_dirs) > 1 and repeat_last > 1:
        expanded.extend([cache_dirs[-1]] * (repeat_last - 1))
    for cache_dir in expanded:
        items.append((cache_dir, torch.load(cache_dir / f"{split}.pt", map_location="cpu", weights_only=False)))
    return items


def merge_split(
    items: list[tuple[Path, dict]],
    output_dir: Path,
    split: str,
    tasks: list[str],
    vocab: dict[str, int],
    exclude_source_roots: set[str] | None = None,
) -> dict:
    first = items[0][1]
    max_frames = max(int(item["valid_frame"].shape[1]) for _, item in items)
    selections = []
    seen_validation_roots: set[str] = set()
    for _, item in items:
        roots = item.get("source_root", [""] * int(item["valid_frame"].shape[0]))
        selected = []
        for index, root in enumerate(roots):
            if exclude_source_roots is not None and root in exclude_source_roots:
                continue
            if split == "validation" and root and root in seen_validation_roots:
                continue
            selected.append(index)
            if split == "validation" and root:
                seen_validation_roots.add(root)
        selections.append(selected)
    total_episodes = sum(len(indices) for indices in selections)
    if total_episodes == 0:
        raise ValueError(f"No {split} episodes remain after source-root exclusion")
    image_shape = tuple(first["image_shape"])
    cameras, height, width, channels = image_shape[1:]
    image_path = output_dir / f"{split}_images.npy"
    out_images = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.uint8,
        shape=(total_episodes, max_frames + 1, cameras, height, width, channels),
    )
    merged: dict[str, torch.Tensor | list[str] | str | tuple | list] = {}
    cursor = 0
    for field in PAD_FIELDS:
        prototype = first[field] if field in first else first["active_arm"]
        shape = (total_episodes, max_frames, *prototype.shape[2:])
        tensor = torch.zeros(shape, dtype=prototype.dtype)
        if field == "task_phase":
            tensor.fill_(-1)
        merged[field] = tensor
    for field in EPISODE_FIELDS:
        merged[field] = torch.zeros((total_episodes,), dtype=first[field].dtype)
    for field in TEXT_FIELDS:
        merged[field] = []
    task_to_index = {task: index for index, task in enumerate(tasks)}
    for (cache_dir, item), selected in zip(items, selections):
        images = np.load(item["image_path"], mmap_mode="r")
        episodes, frames = len(selected), int(item["valid_frame"].shape[1])
        out_images[cursor : cursor + episodes, : images.shape[1]] = images[selected]
        for field in PAD_FIELDS:
            if field in item:
                merged[field][cursor : cursor + episodes, :frames] = item[field][selected]  # type: ignore[index]
        merged["target"][cursor : cursor + episodes] = item["target"][selected]  # type: ignore[index]
        remapped_tasks = [
            task_to_index[item["tasks"][int(index)]]
            for index in item["task_index"][selected].tolist()
        ]
        merged["task_index"][cursor : cursor + episodes] = torch.tensor(remapped_tasks, dtype=first["task_index"].dtype)  # type: ignore[index]
        names = item.get("episode_name", [str(index) for index in range(len(item["target"]))])
        roots = item.get("source_root", [""] * len(item["target"]))
        merged["episode_name"].extend([f"{cache_dir.name}:{names[index]}" for index in selected])  # type: ignore[union-attr]
        merged["source_root"].extend([roots[index] for index in selected])  # type: ignore[union-attr]
        cursor += episodes
    out_images.flush()
    merged.update(
        image_path=str(image_path),
        image_shape=(max_frames + 1, cameras, height, width, channels),
        tasks=tasks,
        vocab=vocab,
        camera_order=list(first["camera_order"]),
        action_representation=first.get("action_representation", "relative_joint_delta"),
        privileged_positions_at_inference=False,
        merge_sources=[str(path) for path, _ in items],
    )
    torch.save(merged, output_dir / f"{split}.pt")
    return merged


def main() -> None:
    args = parse_args()
    if args.repeat_last < 1:
        raise ValueError("--repeat-last must be positive")
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed cache: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_items = load_split(args.cache_dir, "train", args.repeat_last)
    validation_items = load_split(args.cache_dir, "validation", args.repeat_last)
    tasks = []
    for _, item in train_items + validation_items:
        for task in item["tasks"]:
            if task not in tasks:
                tasks.append(task)
    vocab = vocabulary(tasks)
    train = merge_split(train_items, args.output_dir, "train", tasks, vocab)
    validation = merge_split(
        validation_items,
        args.output_dir,
        "validation",
        tasks,
        vocab,
        exclude_source_roots=set(train.get("source_root", [])),
    )
    summary = {
        "experiment": "G1LANG-merged-action-token-vla-cache",
        "cache_dirs": [str(path) for path in args.cache_dir],
        "repeat_last": args.repeat_last,
        "tasks": tasks,
        "train_episodes": int(train["valid_frame"].shape[0]),
        "validation_episodes": int(validation["valid_frame"].shape[0]),
        "train_rows": int(train["valid_frame"].sum()),
        "validation_rows": int(validation["valid_frame"].sum()),
        "image_path_train": train["image_path"],
        "image_path_validation": validation["image_path"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
