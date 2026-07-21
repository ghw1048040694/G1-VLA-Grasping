#!/usr/bin/env python3
"""Convert successful G1 assisted-lift episodes into LeRobot train/val datasets."""

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
LOWER_BODY_JOINTS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_assisted_lift_train")
    parser.add_argument("--val-repo-id", default="local/g1_assisted_lift_val")
    parser.add_argument("--train-episodes", type=int, default=16)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def dataset_features(joint_names: list[str]) -> dict:
    vector = {
        "dtype": "float32",
        "shape": (len(joint_names),),
        "names": joint_names,
    }
    features = {
        "observation.state": dict(vector),
        "action": dict(vector),
        "complementary_info.joint_velocity": dict(vector),
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
    }
    for key in CAMERAS:
        features[key] = {
            "dtype": "video",
            "shape": (240, 320, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def load_episode(episode_dir: Path) -> tuple[dict, dict, list[str]]:
    summary = json.loads(
        (episode_dir / "assisted_tote_lift_summary.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (episode_dir / "episode_metadata.json").read_text(encoding="utf-8")
    )
    if not summary["passed"] or not metadata["episode_success"]:
        raise ValueError(f"Refusing failed episode: {episode_dir}")
    arrays = np.load(episode_dir / "expert_lift_episode.npz")
    joint_names = arrays["joint_names"].tolist()
    if len(joint_names) != 43 or any(
        not name.startswith(("left_", "right_")) for name in joint_names[:12]
    ):
        raise ValueError(f"Unexpected 43-joint ordering in {episode_dir}")
    upper_body_names = joint_names[LOWER_BODY_JOINTS:]
    if len(upper_body_names) != 31 or not upper_body_names[0].startswith("waist_"):
        raise ValueError(f"Unexpected upper-body action contract in {episode_dir}")
    return arrays, metadata, upper_body_names


def convert_split(
    episode_dirs: list[Path], repo_id: str, root: Path, fps: int, overwrite: bool
) -> dict:
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset already exists: {root}")
        shutil.rmtree(root)
    first_arrays, _, joint_names = load_episode(episode_dirs[0])
    first_arrays.close()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type="unitree_g1_upper_body_assisted_lift",
        features=dataset_features(joint_names),
        use_videos=True,
        image_writer_threads=4,
    )
    per_episode = []
    for episode_dir in episode_dirs:
        arrays, metadata, names = load_episode(episode_dir)
        if names != joint_names:
            raise ValueError(f"Joint names changed in {episode_dir}")
        readers = {
            key: imageio.get_reader(episode_dir / filename)
            for key, filename in CAMERAS.items()
        }
        frame_count = len(arrays["timestamp_s"])
        try:
            for frame_index in range(frame_count):
                frame = {
                    "observation.state": arrays["observation_joint_position_rad"][
                        frame_index, LOWER_BODY_JOINTS:
                    ].astype(np.float32),
                    "action": arrays["action_joint_position_rad"][
                        frame_index, LOWER_BODY_JOINTS:
                    ].astype(np.float32),
                    "complementary_info.joint_velocity": arrays[
                        "observation_joint_velocity_rad_s"
                    ][frame_index, LOWER_BODY_JOINTS:].astype(np.float32),
                    "complementary_info.task_phase": np.array(
                        [arrays["task_phase"][frame_index]], dtype=np.int64
                    ),
                    "complementary_info.assisted_grasp_active": np.array(
                        [arrays["assisted_grasp_active"][frame_index]], dtype=np.int64
                    ),
                }
                for key, reader in readers.items():
                    image = np.asarray(reader.get_data(frame_index))
                    if image.shape != (240, 320, 3):
                        raise ValueError(
                            f"Unexpected {key} shape {image.shape} in {episode_dir}"
                        )
                    frame[key] = image
                dataset.add_frame(frame, task=metadata["language_instruction"])
            dataset.save_episode()
        finally:
            arrays.close()
            for reader in readers.values():
                reader.close()
        per_episode.append(
            {
                "source_episode": episode_dir.name,
                "frames": frame_count,
                "task": metadata["language_instruction"],
            }
        )
    dataset.stop_image_writer()
    metadata = LeRobotDatasetMetadata(repo_id, root=root)
    decoded_dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    decoded_samples = 0
    offset = 0
    for episode in per_episode:
        for index in (offset, offset + episode["frames"] - 1):
            item = decoded_dataset[index]
            for key in CAMERAS:
                image = item[key]
                if tuple(image.shape) != (3, 240, 320) or not bool(
                    np.isfinite(image.numpy()).all()
                ):
                    raise ValueError(f"PyAV decode failed for {key} at dataset index {index}")
            decoded_samples += 1
        offset += episode["frames"]
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
        "video_decode_backend": "pyav",
        "decoded_endpoint_samples": decoded_samples,
        "episode_records": per_episode,
    }


def main() -> None:
    args = parse_args()
    source_summary = json.loads(
        (args.source_root / "assisted_lift_dataset_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if not source_summary["experiment_passed"]:
        raise ValueError("Source dataset did not pass its collection contract")
    episode_dirs = sorted(args.source_root.glob("episode_*"))
    if args.episode_limit is not None:
        if args.episode_limit < 2:
            raise ValueError("episode-limit must be at least 2")
        episode_dirs = episode_dirs[: args.episode_limit]
    if not 0 < args.train_episodes < len(episode_dirs):
        raise ValueError("train-episodes must leave at least one validation episode")
    train_dirs = episode_dirs[: args.train_episodes]
    val_dirs = episode_dirs[args.train_episodes :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "G1WH-24-lerobot-vla-dataset-contract",
        "source_experiment": source_summary["experiment"],
        "source_episodes": len(episode_dirs),
        "lower_body_dimensions_removed": LOWER_BODY_JOINTS,
        "assisted_grasp_constraint": True,
        "train": convert_split(
            train_dirs, args.train_repo_id, args.train_root, args.fps, args.overwrite
        ),
        "validation": convert_split(
            val_dirs, args.val_repo_id, args.val_root, args.fps, args.overwrite
        ),
    }
    report["passed"] = (
        report["train"]["episodes"] == args.train_episodes
        and report["validation"]["episodes"] == len(val_dirs)
        and report["train"]["action_dim"] == 31
        and report["validation"]["action_dim"] == 31
        and len(report["train"]["camera_keys"]) == 3
        and len(report["validation"]["camera_keys"]) == 3
        and report["train"]["decoded_endpoint_samples"]
        == 2 * report["train"]["episodes"]
        and report["validation"]["decoded_endpoint_samples"]
        == 2 * report["validation"]["episodes"]
    )
    path = args.output_dir / "conversion_summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
