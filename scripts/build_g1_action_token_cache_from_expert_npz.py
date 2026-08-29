#!/usr/bin/env python3
"""Replay strict-contact expert NPZs into an action-token VLA training cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

import evaluate_g1_language_pick_place as base
from collect_g1_language_pick_place_dataset import LANGUAGE_VARIANTS
from g1_finger_contact_geometry import finger_contact_geom_ids, finger_contact_sites
from run_g1_language_pick_place import OBJECT_NAMES, controlled_joints, build_scene


CAMERAS = tuple(base.CAMERAS.keys())
LOWER_BODY_JOINTS = base.LOWER_BODY_JOINTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-height", type=int, default=120)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--validation-ratio", type=float, default=0.34)
    parser.add_argument("--augment-language-variants", action="store_true")
    parser.add_argument(
        "--require-physical-grasp",
        action="store_true",
        help="Skip episodes whose strict MuJoCo audit did not physically grasp/lift the target.",
    )
    parser.add_argument(
        "--max-task-phase",
        type=int,
        help="Trim each expert NPZ to frames with task_phase <= this value, e.g. 4 keeps approach/contact/lift and drops place/release.",
    )
    parser.add_argument(
        "--pair-policy-visited-start",
        action="store_true",
        help=(
            "For direct DAgger recoveries, replace frame 0 with the exact saved "
            "policy-visited state and pair it with the teacher target one control "
            "step later. This removes the zero-motion startup label."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def vocabulary(tasks: list[str]) -> dict[str, int]:
    import re

    token_re = re.compile(r"[a-z0-9]+")
    vocab = {"<pad>": 0, "<unk>": 1}
    for task in tasks:
        for token in token_re.findall(task.lower()):
            vocab.setdefault(token, len(vocab))
    return vocab


def trim_array(value: np.ndarray, keep: np.ndarray) -> np.ndarray:
    return value[keep] if value.shape[:1] == (len(keep),) else value


def load_episode(episode_dir: Path, max_task_phase: int | None = None, pair_policy_visited_start: bool = False) -> dict:
    npz = np.load(episode_dir / "expert_pick_place_episode.npz", allow_pickle=True)
    summary = json.loads((episode_dir / "language_pick_place_summary.json").read_text())
    joint_names = npz["joint_names"].astype(str).tolist()
    qpos = npz["joint_position_rad"].astype(np.float32).copy()
    qvel = npz["joint_velocity_rad_s"].astype(np.float32).copy()
    action = npz["action_joint_position_rad"].astype(np.float32).copy()
    objects = npz["object_position_m"].astype(np.float32).copy()
    start_pair_audit = None
    if pair_policy_visited_start:
        visited_path = episode_dir / "policy_visited_state.npz"
        if not visited_path.is_file() or len(qpos) < 2:
            raise ValueError(f"Direct DAgger pairing requires visited state and >=2 frames: {episode_dir}")
        visited = np.load(visited_path, allow_pickle=True)
        visited_names = visited["joint_names"].astype(str).tolist()
        visited_qpos = visited["observation_joint_position_rad"].reshape(-1).astype(np.float32)
        visited_qvel = visited["observation_joint_velocity_rad_s"].reshape(-1).astype(np.float32)
        for expert_index, name in enumerate(joint_names):
            if name not in visited_names:
                continue
            source_index = visited_names.index(name)
            qpos[0, expert_index] = visited_qpos[source_index]
            qvel[0, expert_index] = visited_qvel[source_index]
        # A takeover can also need to undo an incorrectly closed hand.  Store
        # the first correction as a deployment-safe bounded step rather than
        # an impossible one-frame jump to the teacher's absolute hand target.
        raw_first_delta = action[1] - qpos[0]
        action[0] = qpos[0] + np.clip(raw_first_delta, -0.08, 0.08)
        if "source_object_position_m" in visited:
            objects[0] = visited["source_object_position_m"].astype(np.float32)
        first_delta = action[0] - qpos[0]
        start_pair_audit = {
            "exact_state_max_error_rad": 0.0,
            "paired_teacher_step": 1,
            "paired_action_mean_abs_rad": float(np.mean(np.abs(first_delta))),
            "paired_action_max_abs_rad": float(np.max(np.abs(first_delta))),
            "raw_teacher_target_max_delta_rad": float(np.max(np.abs(raw_first_delta))),
            "per_joint_pairing_limit_rad": 0.08,
        }
    keep = np.ones(len(npz["time_s"]), dtype=bool)
    if max_task_phase is not None:
        keep &= npz["task_phase"].astype(np.int64) <= int(max_task_phase)
        if int(keep.sum()) < 16:
            raise ValueError(f"After --max-task-phase={max_task_phase}, too few frames remain in {episode_dir}")
    return {
        "episode_dir": episode_dir,
        "summary": summary,
        "joint_names": joint_names,
        "time_s": npz["time_s"].astype(np.float32)[keep],
        "qpos": qpos[keep],
        "qvel": qvel[keep],
        "action": action[keep],
        "objects": objects[keep],
        "object_quaternion": npz["object_quaternion_wxyz"].astype(np.float32)[keep],
        "contact": npz["physical_contact_active"].astype(np.float32)[keep],
        "task_phase": npz["task_phase"].astype(np.int64)[keep],
        "arms": npz["active_arm_index"].astype(np.int64)[keep],
        "target": npz["target_object_index"].astype(np.int64)[keep],
        "max_task_phase": max_task_phase,
        "npz": npz,
        "split_group": str(
            json.loads((episode_dir / "recovery_spec.json").read_text()).get("source_episode")
            if (episode_dir / "recovery_spec.json").is_file()
            else summary.get("initial_state_recovery", {}).get("source", episode_dir)
        ).split("/policy_visited_state.npz")[0],
        "start_pair_audit": start_pair_audit,
    }


def replay_images(model: mujoco.MjModel, data: mujoco.MjData, objects: np.ndarray, qpos: np.ndarray, frames: int, height: int, width: int, image_path: Path) -> None:
    renderer = mujoco.Renderer(model, height=height, width=width)
    out = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.uint8,
        shape=(frames, len(CAMERAS), height, width, 3),
    )
    body_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in OBJECT_NAMES}
    for frame in range(frames):
        data.qpos[:] = 0
        data.qvel[:] = 0
        for joint_name, value in zip(qpos["joint_names"], qpos["qpos"][frame]):
            if joint_name not in qpos["controlled"]:
                continue
            item = qpos["controlled"][joint_name]
            data.qpos[item["qpos_id"]] = value
        for index, name in enumerate(OBJECT_NAMES):
            body = body_ids[name]
            joint = int(model.body_jntadr[body])
            address = int(model.jnt_qposadr[joint])
            data.qpos[address : address + 3] = objects[frame, index]
            data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        for camera_index, camera in enumerate(CAMERAS):
            renderer.update_scene(data, camera=base.CAMERAS[camera])
            out[frame, camera_index] = renderer.render().copy()
    out.flush()
    close = getattr(renderer, "close", None)
    if close is not None:
        close()


def build_episode_cache(
    asset: Path,
    episodes: list[dict],
    tasks: list[str],
    vocab: dict[str, int],
    output_dir: Path,
    split: str,
    image_height: int,
    image_width: int,
) -> dict:
    image_path = output_dir / f"{split}_images.npy"
    frames_per_episode = [max(1, len(ep["qpos"]) - 1) for ep in episodes]
    max_frames = max(frames_per_episode)
    image_mem = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(episodes), max_frames + 1, len(CAMERAS), image_height, image_width, 3),
    )
    fields = {
        key: []
        for key in (
            "state", "velocity", "action_delta", "objects", "contact_sites", "contact", "task_phase",
            "active_arm", "target", "task_index", "valid_frame", "sample_weight",
            "episode_name", "source_root",
        )
    }
    for episode_index, episode in enumerate(episodes):
        frames = max(1, len(episode["qpos"]) - 1)
        scene_path = output_dir / f"{split}_scene_{episode_index:04d}.xml"
        build_scene(asset.resolve(), scene_path, {name: episode["objects"][0, i] for i, name in enumerate(OBJECT_NAMES)})
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        controlled = controlled_joints(model)
        upper_names = list(controlled)[LOWER_BODY_JOINTS:]
        upper_indices = [episode["joint_names"].index(name) for name in upper_names]
        qpos_upper = episode["qpos"][:frames, upper_indices]
        qvel_upper = episode["qvel"][:frames, upper_indices]
        action_upper = episode["action"][:frames, upper_indices] - qpos_upper
        image_episode = np.lib.format.open_memmap(
            image_path,
            mode="r+",
            dtype=np.uint8,
            shape=(len(episodes), max_frames + 1, len(CAMERAS), image_height, image_width, 3),
        )
        body_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in OBJECT_NAMES}
        finger_ids = finger_contact_geom_ids(model)
        renderer = mujoco.Renderer(model, height=image_height, width=image_width)
        sites = []
        for frame in range(frames + 1):
            data.qpos[:] = 0
            data.qvel[:] = 0
            if frame < frames:
                for joint_name, value in zip(upper_names, qpos_upper[min(frame, frames - 1)]):
                    data.qpos[controlled[joint_name]["qpos_id"]] = value
                for joint_name, value in zip(upper_names, qvel_upper[min(frame, frames - 1)]):
                    data.qvel[controlled[joint_name]["qvel_id"]] = value
            else:
                for joint_name, value in zip(upper_names, episode["qpos"][frame, upper_indices]):
                    data.qpos[controlled[joint_name]["qpos_id"]] = value
            for index, name in enumerate(OBJECT_NAMES):
                body = body_ids[name]
                joint = int(model.body_jntadr[body])
                address = int(model.jnt_qposadr[joint])
                data.qpos[address : address + 3] = episode["objects"][min(frame, frames - 1), index]
                data.qpos[address + 3 : address + 7] = episode["object_quaternion"][min(frame, frames - 1), index]
            mujoco.mj_forward(model, data)
            if frame < frames:
                sites.append(finger_contact_sites(model, data, finger_ids).astype(np.float32))
            for camera_index, camera in enumerate(CAMERAS):
                renderer.update_scene(data, camera=base.CAMERAS[camera])
                image_episode[episode_index, frame, camera_index] = renderer.render().copy()
        close = getattr(renderer, "close", None)
        if close is not None:
            close()
        sites = np.stack(sites)
        task = str(episode["summary"]["language_instruction"])
        def pad(array: np.ndarray, value: int | float | bool = 0) -> np.ndarray:
            padded = np.full((max_frames, *array.shape[1:]), value, dtype=array.dtype)
            padded[:frames] = array
            return padded

        valid = np.zeros(max_frames, dtype=bool)
        valid[:frames] = True
        fields["state"].append(pad(qpos_upper))
        fields["velocity"].append(pad(qvel_upper))
        fields["action_delta"].append(pad(action_upper))
        fields["objects"].append(pad(episode["objects"][:frames].reshape(frames, 9)))
        fields["contact_sites"].append(pad(sites))
        fields["contact"].append(pad(episode["contact"][:frames, None]))
        fields["task_phase"].append(pad(episode["task_phase"][:frames]))
        fields["active_arm"].append(pad(episode["arms"][:frames]))
        fields["target"].append(np.asarray(int(episode["target"][0]), dtype=np.int64))
        fields["task_index"].append(np.asarray(tasks.index(task), dtype=np.int64))
        fields["valid_frame"].append(valid)
        fields["sample_weight"].append(valid.astype(np.float32))
        fields["episode_name"].append(episode["episode_dir"].name)
        fields["source_root"].append(str(episode["episode_dir"]))
    packed = {key: torch.from_numpy(np.stack(value)) if key not in {"episode_name", "source_root"} else value for key, value in fields.items()}
    packed.update(
        image_path=str(image_path),
        image_shape=(max_frames + 1, len(CAMERAS), image_height, image_width, 3),
        tasks=tasks,
        vocab=vocab,
        camera_order=list(CAMERAS),
        action_representation="relative_joint_delta",
        privileged_positions_at_inference=False,
    )
    torch.save(packed, output_dir / f"{split}.pt")
    return packed


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = [load_episode(path, args.max_task_phase, args.pair_policy_visited_start) for path in args.episode_dir]
    if args.require_physical_grasp:
        episodes = [
            episode
            for episode in episodes
            if bool(episode["summary"].get("grasp_physics_audit", {}).get("physical_grasp_passed", False))
        ]
    if not episodes:
        raise ValueError("No expert episodes remain after filtering")
    if args.augment_language_variants:
        augmented = []
        for episode in episodes:
            target = str(episode["summary"]["target_object"])
            for variant_index, instruction in enumerate(LANGUAGE_VARIANTS[target]):
                item = dict(episode)
                item["summary"] = dict(episode["summary"])
                item["summary"]["language_instruction"] = instruction
                item["language_variant_index"] = variant_index
                augmented.append(item)
        episodes = augmented
    tasks = sorted({str(ep["summary"]["language_instruction"]) for ep in episodes})
    vocab = vocabulary(tasks)
    rng = np.random.default_rng(args.seed)
    val_idx: set[int] = set()
    if args.validation_ratio > 0.0:
        targets = sorted({str(ep["summary"]["target_object"]) for ep in episodes})
        for target in targets:
            indices = [i for i, episode in enumerate(episodes) if str(episode["summary"]["target_object"]) == target]
            groups = {}
            for index in indices:
                groups.setdefault(episodes[index]["split_group"], []).append(index)
            group_names = np.asarray(sorted(groups))
            if len(group_names) <= 1:
                continue
            val_count = min(len(group_names) - 1, max(1, int(round(len(group_names) * args.validation_ratio))))
            for group in rng.choice(group_names, val_count, replace=False).tolist():
                val_idx.update(groups[group])
    train_eps = [ep for i, ep in enumerate(episodes) if i not in val_idx]
    val_eps = [ep for i, ep in enumerate(episodes) if i in val_idx]
    if not val_eps:
        val_eps = train_eps
    train_cache = build_episode_cache(args.asset, train_eps, tasks, vocab, args.output_dir, "train", args.image_height, args.image_width)
    val_cache = build_episode_cache(args.asset, val_eps, tasks, vocab, args.output_dir, "validation", args.image_height, args.image_width)
    summary = {
        "experiment": "G1LANG-expert-npz-action-token-vla-cache",
        "episodes": len(episodes),
        "tasks": tasks,
        "train_rows": int(train_cache["valid_frame"].sum()),
        "validation_rows": int(val_cache["valid_frame"].sum()),
        "image_path_train": train_cache["image_path"],
        "image_path_validation": val_cache["image_path"],
        "policy_visited_start_pairing": bool(args.pair_policy_visited_start),
        "start_pair_audits": [ep["start_pair_audit"] for ep in episodes if ep["start_pair_audit"] is not None],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
