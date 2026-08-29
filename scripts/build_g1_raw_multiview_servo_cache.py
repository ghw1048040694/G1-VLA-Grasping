#!/usr/bin/env python3
"""Build a disk-backed raw-RGB cache for the G1 visual-servo VLA.

The cache deliberately keeps rendered RGB pixels instead of reducing every
camera to colour centroids. MuJoCo object poses are stored only as supervised
training labels and are never part of the deployment observation contract.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from build_g1_direct_joint_dynamic_cache import (
    CAMERA_VIDEO_NAMES,
    CANDIDATE_OFFSETS,
    OBJECT_NAMES,
    finger_contact_site_sequence,
    strict_summary,
)


@dataclass(frozen=True)
class EpisodeSource:
    directory: Path
    source_root: Path
    is_recovery: bool
    valid_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-frames", type=int, default=170)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--base-validation-groups", type=int, default=10)
    parser.add_argument("--recovery-supervision-frames", type=int, default=72)
    parser.add_argument("--recovery-weight", type=float, default=12.0)
    return parser.parse_args()


def vocabulary(tasks: list[str]) -> dict[str, int]:
    import re

    token_re = re.compile(r"[a-z0-9]+")
    result = {"<pad>": 0, "<unk>": 1}
    for task in tasks:
        for token in token_re.findall(task.lower()):
            result.setdefault(token, len(result))
    return result


def base_sources(root: Path, sequence_frames: int, validation_groups: int) -> tuple[list[EpisodeSource], list[EpisodeSource]]:
    entries = []
    for directory in sorted(root.glob("episode_*")):
        if strict_summary(directory) is None:
            continue
        try:
            index = int(directory.name.split("_")[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Expected numbered base episode: {directory}") from error
        entries.append((index // len(OBJECT_NAMES), directory))
    groups = sorted({group for group, _ in entries})
    if len(groups) <= validation_groups:
        raise ValueError("Not enough independent base layout groups for validation")
    validation = set(groups[-validation_groups:])
    train = [EpisodeSource(path, root, False, sequence_frames) for group, path in entries if group not in validation]
    val = [EpisodeSource(path, root, False, sequence_frames) for group, path in entries if group in validation]
    return train, val


def recovery_sources(roots: list[Path], frames: int) -> list[EpisodeSource]:
    result = []
    for root in roots:
        for directory in sorted(root.glob("recovery_*")):
            if strict_summary(directory) is not None:
                result.append(EpisodeSource(directory, root, True, frames))
    return result


def read_images(directory: Path, frames: int, height: int, width: int) -> np.ndarray:
    result = np.empty((frames, len(CAMERA_VIDEO_NAMES), height, width, 3), dtype=np.uint8)
    for camera_index, name in enumerate(CAMERA_VIDEO_NAMES):
        capture = cv2.VideoCapture(str(directory / name))
        if not capture.isOpened():
            raise FileNotFoundError(directory / name)
        count = 0
        while count < frames:
            ok, image = capture.read()
            if not ok:
                break
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result[count, camera_index] = cv2.resize(
                image, (width, height), interpolation=cv2.INTER_AREA
            )
            count += 1
        capture.release()
        if count != frames:
            raise ValueError(f"Expected {frames} frames in {directory / name}, got {count}")
    return result


def load_labels(source: EpisodeSource, frames: int) -> dict[str, np.ndarray | str | int]:
    summary = strict_summary(source.directory)
    if summary is None:
        raise ValueError(f"Source is not a strict physical success: {source.directory}")
    with np.load(source.directory / "expert_pick_place_episode.npz") as values:
        names = values["joint_names"]
        qpos = values["joint_position_rad"].astype(np.float32)
        velocity = values["joint_velocity_rad_s"].astype(np.float32)
        action = values["action_joint_position_rad"].astype(np.float32)
        objects = values["object_position_m"].astype(np.float32)
        contact = values["physical_contact_active"].astype(np.float32)
        arms = values["active_arm_index"].astype(np.int64)
    final = frames + 1
    if min(len(qpos), len(velocity), len(action), len(objects), len(contact), len(arms)) < final:
        raise ValueError(f"Episode is shorter than {final} labels: {source.directory}")
    state = qpos[:frames, 12:]
    target = action[:frames, 12:] - state
    sites = finger_contact_site_sequence(source.directory, names, qpos[:frames])
    next_sites = finger_contact_site_sequence(source.directory, names, qpos[1:final])
    if state.shape != (frames, 31) or target.shape != (frames, 31):
        raise ValueError(f"Unexpected 31D direct-joint contract in {source.directory}")
    return {
        "state": state,
        "velocity": velocity[:frames, 12:],
        "action_delta": target,
        "next_state": qpos[1:final, 12:],
        "next_velocity": velocity[1:final, 12:],
        "objects": objects[:frames].reshape(frames, 9),
        "next_objects": objects[1:final].reshape(frames, 9),
        "contact_sites": sites,
        "next_contact_sites": next_sites,
        "contact": contact[:frames, None],
        "next_contact": contact[1:final, None],
        "active_arm": arms[:frames],
        "target": OBJECT_NAMES.index(summary["target_object"]),
        "task": str(summary["language_instruction"]),
    }


def write_split(
    name: str,
    sources: list[EpisodeSource],
    args: argparse.Namespace,
    tasks: list[str],
    vocab: dict[str, int],
) -> dict:
    frames = int(args.sequence_frames)
    image_path = args.output_dir / f"{name}_images.npy"
    images = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.uint8,
        # Preserve the observation at t + 1.  The action label at t already
        # has successor robot/object labels, and a learned visual-state world
        # model must see the matching successor RGB observation as well.
        shape=(len(sources), frames + 1, len(CAMERA_VIDEO_NAMES), args.image_height, args.image_width, 3),
    )
    values: dict[str, list] = {
        key: []
        for key in (
            "state", "velocity", "action_delta", "next_state", "next_velocity",
            "objects", "next_objects", "contact_sites", "next_contact_sites", "contact", "next_contact",
            "active_arm", "target", "task_index", "valid_frame", "sample_weight",
            "is_recovery", "episode_name", "source_root",
        )
    }
    for index, source in enumerate(sources):
        label = load_labels(source, frames)
        images[index] = read_images(source.directory, frames + 1, args.image_height, args.image_width)
        values["state"].append(label["state"])
        values["velocity"].append(label["velocity"])
        values["action_delta"].append(label["action_delta"])
        values["next_state"].append(label["next_state"])
        values["next_velocity"].append(label["next_velocity"])
        values["objects"].append(label["objects"])
        values["next_objects"].append(label["next_objects"])
        values["contact_sites"].append(label["contact_sites"])
        values["next_contact_sites"].append(label["next_contact_sites"])
        values["contact"].append(label["contact"])
        values["next_contact"].append(label["next_contact"])
        values["active_arm"].append(label["active_arm"])
        values["target"].append(int(label["target"]))
        values["task_index"].append(tasks.index(str(label["task"])))
        valid = np.arange(frames) < min(frames, source.valid_frames)
        values["valid_frame"].append(valid)
        values["sample_weight"].append(np.where(valid, args.recovery_weight if source.is_recovery else 1.0, 0.0).astype(np.float32))
        values["is_recovery"].append(source.is_recovery)
        values["episode_name"].append(source.directory.name)
        values["source_root"].append(str(source.source_root))
        print(json.dumps({"cache_split": name, "episode": source.directory.name, "recovery": source.is_recovery, "index": index + 1, "total": len(sources)}), flush=True)
    images.flush()
    packed = {
        key: (
            torch.from_numpy(np.stack(value))
            if key not in {"episode_name", "source_root"}
            else value
        )
        for key, value in values.items()
    }
    packed.update(
        image_path=str(image_path),
        image_shape=(frames + 1, len(CAMERA_VIDEO_NAMES), args.image_height, args.image_width, 3),
        tasks=tasks,
        vocab=vocab,
        camera_order=list(CAMERA_VIDEO_NAMES),
        action_representation="relative_joint_delta",
        privileged_positions_at_inference=False,
    )
    torch.save(packed, args.output_dir / f"{name}.pt")
    return {
        "episodes": len(sources),
        "frames": frames,
        "valid_rows": int(packed["valid_frame"].sum()),
        "recovery_episodes": int(sum(values["is_recovery"])),
        "image_path": str(image_path),
    }


def main() -> None:
    args = parse_args()
    if args.sequence_frames < 2 or args.image_height < 32 or args.image_width < 32:
        raise ValueError("sequence and image dimensions are too small")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train, val = base_sources(args.dataset_root, args.sequence_frames, args.base_validation_groups)
    recoveries = recovery_sources(args.recovery_root, min(args.sequence_frames, args.recovery_supervision_frames))
    if not recoveries:
        raise ValueError("At least one strict recovery root is required for this contract")
    train.extend(recoveries)
    all_tasks = []
    for source in train + val:
        summary = strict_summary(source.directory)
        assert summary is not None
        all_tasks.append(str(summary["language_instruction"]))
    tasks = list(dict.fromkeys(all_tasks))
    vocab = vocabulary(tasks)
    summary = {
        "experiment": "G1LANG-raw-multiview-visual-servo-cache",
        "input_contract": "raw synchronized head/left-wrist/right-wrist RGB + language + state31 + causal qvel31",
        "labels": "strict MuJoCo executed relative joint deltas and offline object pose auxiliary labels",
        "privileged_positions_at_inference": False,
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "recovery_roots": [str(path) for path in args.recovery_root],
        "train": write_split("train", train, args, tasks, vocab),
        "validation": write_split("validation", val, args, tasks, vocab),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
