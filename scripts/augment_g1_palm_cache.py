#!/usr/bin/env python3
"""Add strict-expert palm-position labels to a compact VLA cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import mujoco


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-in", type=Path, required=True)
    parser.add_argument("--cache-out", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--episode-offset", type=int, default=0)
    args = parser.parse_args()
    cache = torch.load(args.cache_in, map_location="cpu", weights_only=False)
    parts = []
    for episode in sorted(set(cache["episode_index"].long().tolist())):
        source_index = episode + args.episode_offset
        path = args.expert_root / f"episode_{source_index:04d}" / "expert_pick_place_episode.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=True) as arrays:
            palm = arrays["palm_position_m"].astype(np.float32)
            joint_names = arrays["joint_names"].astype(str).tolist()
            joint_position = arrays["joint_position_rad"].astype(np.float32)
        scene_path = path.parent / "g1_language_pick_place.xml"
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        site_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in ("left_palm_center", "right_palm_center")
        ]
        if any(site_id < 0 for site_id in site_ids):
            raise ValueError(f"Missing palm sites in {scene_path}")
        orientation = np.empty((len(joint_position), 12), dtype=np.float32)
        for frame_index, values in enumerate(joint_position):
            for name, value in zip(joint_names, values):
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[model.jnt_qposadr[joint_id]] = value
            mujoco.mj_forward(model, data)
            orientation[frame_index] = np.concatenate(
                [data.site_xmat[site_id].reshape(3, 3)[:, :2].reshape(-1) for site_id in site_ids]
            )
        mask = cache["episode_index"].long() == episode
        count = int(mask.sum())
        if len(palm) != count:
            raise ValueError(f"{path}: {len(palm)} frames, cache episode has {count}")
        parts.append((episode, palm, orientation))
    palm_by_episode = {episode: palm for episode, palm, _ in parts}
    orientation_by_episode = {episode: orientation for episode, _, orientation in parts}
    palm_values = np.empty((len(cache["state"]), 6), dtype=np.float32)
    for episode, palm in palm_by_episode.items():
        mask = cache["episode_index"].long().numpy() == episode
        palm_values[mask] = palm.reshape(-1, 6)
    cache["palm_position"] = torch.from_numpy(palm_values)
    orientation_values = np.empty((len(cache["state"]), 12), dtype=np.float32)
    for episode, orientation in orientation_by_episode.items():
        mask = cache["episode_index"].long().numpy() == episode
        orientation_values[mask] = orientation
    cache["palm_orientation"] = torch.from_numpy(orientation_values)
    args.cache_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.cache_out)
    print(f"Saved {args.cache_out} samples={len(cache['palm_position'])}")


if __name__ == "__main__":
    main()
