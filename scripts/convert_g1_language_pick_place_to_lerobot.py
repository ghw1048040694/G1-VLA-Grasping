#!/usr/bin/env python3
"""Convert production G1 language pick-and-place episodes to LeRobot datasets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

CAMERAS = {
    "observation.images.head": "head_camera.mp4",
    "observation.images.left_wrist": "left_wrist_camera.mp4",
    "observation.images.right_wrist": "right_wrist_camera.mp4",
}
OBJECT_NAMES = ("red_triangle", "yellow_rod", "green_cube")
LOWER_BODY_JOINTS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_language_pick_place_train")
    parser.add_argument("--val-repo-id", default="local/g1_language_pick_place_val")
    parser.add_argument("--train-episodes", type=int, default=150)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def dataset_features(joint_names: list[str]) -> dict:
    joint_vector = {
        "dtype": "float32",
        "shape": (len(joint_names),),
        "names": joint_names,
    }
    object_position_names = [
        f"{name}.{axis}" for name in OBJECT_NAMES for axis in ("x", "y", "z")
    ]
    object_quaternion_names = [
        f"{name}.{axis}" for name in OBJECT_NAMES for axis in ("w", "x", "y", "z")
    ]
    features = {
        "observation.state": dict(joint_vector),
        "action": dict(joint_vector),
        "complementary_info.joint_velocity": dict(joint_vector),
        "complementary_info.object_position": {
            "dtype": "float32",
            "shape": (len(object_position_names),),
            "names": object_position_names,
        },
        "complementary_info.object_quaternion": {
            "dtype": "float32",
            "shape": (len(object_quaternion_names),),
            "names": object_quaternion_names,
        },
        "complementary_info.target_object_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["target_object_index"],
        },
        "complementary_info.task_phase": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["phase_id"],
        },
        "complementary_info.assisted_grasp_active": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["assisted_grasp_active"],
        },
        "complementary_info.active_arm_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["active_arm_index_0_left_1_right"],
        },
    }
    for key in CAMERAS:
        features[key] = {
            "dtype": "video",
            "shape": (240, 320, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def load_episode(episode_dir: Path) -> tuple[np.lib.npyio.NpzFile, dict, list[str]]:
    summary = json.loads(
        (episode_dir / "language_pick_place_summary.json").read_text(encoding="utf-8")
    )
    if not summary.get("passed"):
        raise ValueError(f"Refusing failed source episode: {episode_dir}")
    arrays = np.load(episode_dir / "expert_pick_place_episode.npz")
    joint_names = arrays["joint_names"].tolist()
    if len(joint_names) != 43:
        raise ValueError(
            f"Expected 43 source joints in {episode_dir}, found {len(joint_names)}"
        )
    upper_names = joint_names[LOWER_BODY_JOINTS:]
    if len(upper_names) != 31 or not upper_names[0].startswith("waist_"):
        raise ValueError(f"Unexpected upper-body joint contract in {episode_dir}")
    frame_count = len(arrays["time_s"])
    expected_shapes = {
        "joint_position_rad": (frame_count, 43),
        "joint_velocity_rad_s": (frame_count, 43),
        "action_joint_position_rad": (frame_count, 43),
        "object_position_m": (frame_count, 3, 3),
        "object_quaternion_wxyz": (frame_count, 3, 4),
        "active_arm_index": (frame_count,),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(
                f"Unexpected {key} shape {arrays[key].shape}, expected {shape}"
            )
    return arrays, summary, upper_names


def convert_split(
    episode_dirs: list[Path],
    repo_id: str,
    root: Path,
    fps: int,
    overwrite: bool,
) -> dict:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset already exists: {root}")
        shutil.rmtree(root)
    first, _, joint_names = load_episode(episode_dirs[0])
    first.close()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type="unitree_g1_language_grounded_upper_body",
        features=dataset_features(joint_names),
        use_videos=True,
        image_writer_threads=4,
    )
    records = []
    tasks = set()
    target_counts = {name: 0 for name in OBJECT_NAMES}
    for source_index, episode_dir in enumerate(episode_dirs):
        arrays, summary, names = load_episode(episode_dir)
        if names != joint_names:
            raise ValueError(f"Joint ordering changed in {episode_dir}")
        readers = {
            key: imageio.get_reader(episode_dir / filename)
            for key, filename in CAMERAS.items()
        }
        task = summary["language_instruction"]
        target = summary["target_object"]
        frame_count = len(arrays["time_s"])
        try:
            for frame_index in range(frame_count):
                frame = {
                    "observation.state": arrays["joint_position_rad"][
                        frame_index, LOWER_BODY_JOINTS:
                    ].astype(np.float32),
                    "action": arrays["action_joint_position_rad"][
                        frame_index, LOWER_BODY_JOINTS:
                    ].astype(np.float32),
                    "complementary_info.joint_velocity": arrays["joint_velocity_rad_s"][
                        frame_index, LOWER_BODY_JOINTS:
                    ].astype(np.float32),
                    "complementary_info.object_position": arrays["object_position_m"][
                        frame_index
                    ]
                    .reshape(-1)
                    .astype(np.float32),
                    "complementary_info.object_quaternion": arrays[
                        "object_quaternion_wxyz"
                    ][frame_index]
                    .reshape(-1)
                    .astype(np.float32),
                    "complementary_info.target_object_index": np.asarray(
                        [arrays["target_object_index"][frame_index]], dtype=np.int64
                    ),
                    "complementary_info.task_phase": np.asarray(
                        [arrays["task_phase"][frame_index]], dtype=np.int64
                    ),
                    "complementary_info.assisted_grasp_active": np.asarray(
                        [arrays["assist_active"][frame_index]], dtype=np.int64
                    ),
                    "complementary_info.active_arm_index": np.asarray(
                        [arrays["active_arm_index"][frame_index]], dtype=np.int64
                    ),
                }
                for key, reader in readers.items():
                    image = np.asarray(reader.get_data(frame_index))
                    if image.shape != (240, 320, 3):
                        raise ValueError(f"Unexpected {key} image shape {image.shape}")
                    frame[key] = image
                dataset.add_frame(frame, task=task)
            dataset.save_episode()
        finally:
            arrays.close()
            for reader in readers.values():
                reader.close()
        tasks.add(task)
        target_counts[target] += 1
        records.append(
            {
                "dataset_episode": source_index,
                "source_episode": episode_dir.name,
                "target_object": target,
                "task": task,
                "frames": frame_count,
            }
        )
        print(f"CONVERT_PROGRESS={source_index + 1}/{len(episode_dirs)}", flush=True)
    dataset.stop_image_writer()
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    decoded = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    for index in (0, metadata.total_frames - 1):
        item = decoded[index]
        for key in CAMERAS:
            if tuple(item[key].shape) != (3, 240, 320):
                raise ValueError(f"Decoded {key} has shape {tuple(item[key].shape)}")
    return {
        "repo_id": repo_id,
        "root": str(root),
        "episodes": metadata.total_episodes,
        "frames": metadata.total_frames,
        "tasks": metadata.total_tasks,
        "fps": metadata.fps,
        "state_dim": metadata.features["observation.state"]["shape"][0],
        "action_dim": metadata.features["action"]["shape"][0],
        "camera_keys": metadata.camera_keys,
        "target_counts": target_counts,
        "records": records,
    }


def main() -> None:
    args = parse_args()
    source_summary = json.loads(
        (args.source_root / "language_pick_place_dataset_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if not source_summary.get("passed"):
        raise ValueError("Source production dataset has not passed")
    episode_dirs = sorted(args.source_root.glob("episode_*"))
    if not 0 < args.train_episodes < len(episode_dirs):
        raise ValueError("--train-episodes must leave at least one validation episode")
    train_dirs = episode_dirs[: args.train_episodes]
    val_dirs = episode_dirs[args.train_episodes :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "G1-Language-Grounded-Manipulation-LeRobot-Dataset",
        "source_root": str(args.source_root),
        "source_episodes": len(episode_dirs),
        "language_mode": "full_episode_instruction",
        "train": convert_split(
            train_dirs, args.train_repo_id, args.train_root, args.fps, args.overwrite
        ),
        "validation": convert_split(
            val_dirs, args.val_repo_id, args.val_root, args.fps, args.overwrite
        ),
    }
    expected_train_targets = {
        name: sum(
            json.loads((path / "language_pick_place_summary.json").read_text())[
                "target_object"
            ]
            == name
            for path in train_dirs
        )
        for name in OBJECT_NAMES
    }
    expected_val_targets = {
        name: sum(
            json.loads((path / "language_pick_place_summary.json").read_text())[
                "target_object"
            ]
            == name
            for path in val_dirs
        )
        for name in OBJECT_NAMES
    }
    report["passed"] = bool(
        report["train"]["episodes"] == len(train_dirs)
        and report["validation"]["episodes"] == len(val_dirs)
        and report["train"]["action_dim"] == 31
        and report["validation"]["action_dim"] == 31
        and len(report["train"]["camera_keys"]) == 3
        and len(report["validation"]["camera_keys"]) == 3
        and report["train"]["target_counts"] == expected_train_targets
        and report["validation"]["target_counts"] == expected_val_targets
    )
    path = args.output_dir / "conversion_summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
