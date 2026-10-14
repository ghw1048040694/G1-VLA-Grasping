#!/usr/bin/env python3
"""Regenerate language-task camera videos with the current MJCF camera contract."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from run_g1_language_pick_place import (
    HEAD_CAMERA_FOVY,
    HEAD_CAMERA_XYAXES,
    OBJECT_NAMES,
    build_scene,
)


CAMERAS = {
    "head_camera": "head_camera.mp4",
    "left_wrist_camera": "left_wrist_camera.mp4",
    "right_wrist_camera": "right_wrist_camera.mp4",
}
JOINT_NAMES = "joint_names"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Regenerate only head_camera and copy the unchanged wrist videos.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_camera_contract(source: Path, destination: Path) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    camera = root.find(".//camera[@name='head_camera']")
    if camera is None:
        raise RuntimeError(f"Missing head_camera in {source}")
    camera.set("xyaxes", HEAD_CAMERA_XYAXES)
    camera.set("fovy", HEAD_CAMERA_FOVY)
    tree.write(destination, encoding="unicode", xml_declaration=False)


def set_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_names: list[str],
    joint_position: np.ndarray,
    object_positions: np.ndarray,
    object_quaternions: np.ndarray,
) -> None:
    for name, value in zip(joint_names, joint_position, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Missing joint {name}")
        data.qpos[int(model.jnt_qposadr[joint_id])] = float(value)
    for index, name in enumerate(OBJECT_NAMES):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise RuntimeError(f"Missing object body {name}")
        joint_id = int(model.body_jntadr[body_id])
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"Object body {name} is not free-jointed")
        qpos_id = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_id : qpos_id + 3] = object_positions[index]
        data.qpos[qpos_id + 3 : qpos_id + 7] = object_quaternions[index]
    mujoco.mj_forward(model, data)


def regenerate_episode(
    source_episode: Path,
    output_episode: Path,
    fps: int,
    width: int,
    height: int,
    overwrite: bool,
    head_only: bool,
) -> dict:
    output_episode.mkdir(parents=True, exist_ok=True)
    arrays = np.load(source_episode / "expert_pick_place_episode.npz")
    summary = json.loads(
        (source_episode / "language_pick_place_summary.json").read_text(
            encoding="utf-8"
        )
    )
    source_scene = Path(summary["scene"])
    if not source_scene.is_file():
        raise FileNotFoundError(source_scene)
    positions = {
        name: np.asarray(arrays["object_position_m"][0, index], dtype=np.float64)
        for index, name in enumerate(OBJECT_NAMES)
    }
    corrected_scene = output_episode / "g1_language_pick_place.xml"
    build_scene(source_scene, corrected_scene, positions)
    model = mujoco.MjModel.from_xml_path(str(corrected_scene))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera_names = ("head_camera",) if head_only else tuple(CAMERAS)
    writers = {
        name: imageio.get_writer(
            output_episode / filename, fps=fps, codec="libx264", quality=8
        )
        for name, filename in CAMERAS.items()
        if name in camera_names
    }
    joint_names = arrays[JOINT_NAMES].tolist()
    frame_count = len(arrays["time_s"])
    try:
        for frame_index in range(frame_count):
            set_state(
                model,
                data,
                joint_names,
                arrays["joint_position_rad"][frame_index],
                arrays["object_position_m"][frame_index],
                arrays["object_quaternion_wxyz"][frame_index],
            )
            for camera_name, writer in writers.items():
                renderer.update_scene(data, camera=camera_name)
                writer.append_data(renderer.render())
    finally:
        for writer in writers.values():
            writer.close()
        renderer.close()
        arrays.close()
    for name in ("collection_spec.json", "language_pick_place_summary.json", "expert_pick_place_episode.npz"):
        shutil.copy2(source_episode / name, output_episode / name)
    if head_only:
        for filename in ("left_wrist_camera.mp4", "right_wrist_camera.mp4"):
            shutil.copy2(source_episode / filename, output_episode / filename)
    return {
        "source_episode": source_episode.name,
        "frames": frame_count,
        "camera_contract": {"head_xyaxes": HEAD_CAMERA_XYAXES, "head_fovy": HEAD_CAMERA_FOVY},
        "target_object": summary["target_object"],
    }


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("fps, width, and height must be positive")
    if args.episode_start < 0:
        raise ValueError("episode-start must be non-negative")
    if args.output_root.exists() and not args.overwrite and args.episode_start == 0:
        existing = list(args.output_root.glob("episode_*"))
        if existing:
            raise FileExistsError(f"Output already contains episodes: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_summary_path = args.source_root / "language_pick_place_dataset_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    episodes = sorted(args.source_root.glob("episode_*"))
    if args.episode_limit is not None:
        episodes = episodes[: args.episode_limit]
    episodes = episodes[args.episode_start :]
    if not episodes:
        raise ValueError("No source episodes found")
    output_summary = dict(source_summary)
    output_episode_count = len(list(args.output_root.glob("episode_*"))) + len(episodes)
    if isinstance(source_summary.get("episode_specs"), list):
        output_summary["episode_specs"] = source_summary["episode_specs"][:output_episode_count]
    output_summary["requested_episodes"] = output_episode_count
    output_summary["successful_episodes"] = output_episode_count
    output_summary["paired_scene_groups"] = output_episode_count // len(OBJECT_NAMES)
    (args.output_root / source_summary_path.name).write_text(
        json.dumps(output_summary, indent=2) + "\n", encoding="utf-8"
    )
    records = []
    for index, source_episode in enumerate(episodes):
        output_episode = args.output_root / source_episode.name
        record = regenerate_episode(
            source_episode,
            output_episode,
            args.fps,
            args.width,
            args.height,
            args.overwrite,
            args.head_only,
        )
        records.append(record)
        print(f"RENDER_PROGRESS={index + 1}/{len(episodes)}", flush=True)
    report = {
        "experiment": "G1LANG-camera-contract-regeneration",
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "episodes": len(records),
        "source_dataset_contract_version": source_summary.get("dataset_contract_version"),
        "camera_contract": {"head_xyaxes": HEAD_CAMERA_XYAXES, "head_fovy": HEAD_CAMERA_FOVY},
        "records": records,
        "passed": len(records) == len(episodes),
    }
    (args.output_root / "camera_regeneration_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
