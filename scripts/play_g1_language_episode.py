#!/usr/bin/env python3
"""Play a recorded G1 language episode in the interactive MuJoCo viewer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--hide-ui", action="store_true")
    return parser.parse_args()


def load_episode(episode_dir: Path) -> tuple[mujoco.MjModel, dict[str, np.ndarray]]:
    scene_path = episode_dir / "g1_language_pick_place.xml"
    trajectory_path = episode_dir / "expert_pick_place_episode.npz"
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path.resolve()))
    with np.load(trajectory_path, allow_pickle=True) as archive:
        trajectory = {name: archive[name].copy() for name in archive.files}
    required = {
        "time_s",
        "joint_names",
        "joint_position_rad",
        "object_position_m",
        "object_quaternion_wxyz",
    }
    missing = sorted(required - trajectory.keys())
    if missing:
        raise ValueError(f"Trajectory is missing fields: {missing}")
    frame_count = len(trajectory["time_s"])
    if trajectory["joint_position_rad"].shape[0] != frame_count:
        raise ValueError("Joint and time arrays have different frame counts")
    if trajectory["object_position_m"].shape[:2] != (frame_count, 3):
        raise ValueError("Expected three object positions per frame")
    return model, trajectory


def set_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: dict[str, np.ndarray],
    frame: int,
) -> None:
    joint_names = trajectory["joint_names"].tolist()
    joint_positions = trajectory["joint_position_rad"][frame]
    for name, position in zip(joint_names, joint_positions):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"Joint from trajectory is absent in scene: {name}")
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_address] = float(position)

    for object_index, name in enumerate(("red_triangle", "yellow_rod", "green_cube")):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Object body is absent in scene: {name}")
        joint_id = int(model.body_jntadr[body_id])
        if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Object body is not free: {name}")
        qpos_address = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_address : qpos_address + 3] = trajectory["object_position_m"][
            frame, object_index
        ]
        data.qpos[qpos_address + 3 : qpos_address + 7] = trajectory[
            "object_quaternion_wxyz"
        ][frame, object_index]
    mujoco.mj_forward(model, data)


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    model, trajectory = load_episode(args.episode_dir.resolve())
    data = mujoco.MjData(model)
    frame_count = len(trajectory["time_s"])
    state = {"paused": False, "loop": bool(args.loop), "frame": 0}

    def key_callback(key: int) -> None:
        if key == 32:  # space
            state["paused"] = not state["paused"]
        elif key in (ord("r"), ord("R")):
            state["frame"] = 0
            state["paused"] = False
        elif key in (ord("l"), ord("L")):
            state["loop"] = not state["loop"]
        elif key == 256:  # GLFW_KEY_ESCAPE
            state["quit"] = True

    state["quit"] = False
    print(
        f"Playing {args.episode_dir} ({frame_count} frames, "
        f"{1.0 / np.mean(np.diff(trajectory['time_s'])):.1f} FPS).",
        flush=True,
    )
    print("Viewer keys: Space=pause, R=restart, L=toggle loop, Esc=close.", flush=True)

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_callback,
        show_left_ui=not args.hide_ui,
        show_right_ui=not args.hide_ui,
    ) as viewer:
        viewer.cam.lookat[:] = (0.45, 0.0, 0.92)
        viewer.cam.distance = 1.65
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -12
        while viewer.is_running() and not state["quit"]:
            frame = state["frame"]
            set_frame(model, data, trajectory, frame)
            viewer.sync()
            if state["paused"]:
                time.sleep(0.03)
                continue
            if frame + 1 >= frame_count:
                if state["loop"]:
                    state["frame"] = 0
                    continue
                state["paused"] = True
                continue
            frame_delta = float(trajectory["time_s"][frame + 1] - trajectory["time_s"][frame])
            state["frame"] += 1
            time.sleep(max(0.0, frame_delta / args.speed))


if __name__ == "__main__":
    main()
