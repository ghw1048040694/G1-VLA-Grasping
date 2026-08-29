#!/usr/bin/env python3
"""Build a task-value branch cache from strict-contact expert NPZ episodes.

This is an offline teacher bridge.  It does not make deployment use privileged
object state; it creates positive grasp/lift branches so the task-value world
model has a real target before the VLA is retrained to propose such chunks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

import evaluate_g1_language_pick_place as base
from g1_finger_contact_geometry import finger_contact_geom_ids, finger_contact_sites
from run_g1_language_pick_place import OBJECT_NAMES, controlled_joints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--validation-ratio", type=float, default=0.34)
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def tactile_contact(model: mujoco.MjModel, data: mujoco.MjData, hand_geoms: set[int], object_bodies: set[int]) -> float:
    return float(
        any(
            (contact.geom1 in hand_geoms and int(model.geom_bodyid[contact.geom2]) in object_bodies)
            or (contact.geom2 in hand_geoms and int(model.geom_bodyid[contact.geom1]) in object_bodies)
            for contact in data.contact[: data.ncon]
        )
    )


def target_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_geoms: set[int],
    target_geoms: set[int],
) -> float:
    return float(
        any(
            (contact.geom1 in hand_geoms and contact.geom2 in target_geoms)
            or (contact.geom2 in hand_geoms and contact.geom1 in target_geoms)
            for contact in data.contact[: data.ncon]
        )
    )


def active_hand_geometries(finger_ids: list[int], pad_ids: list[int], arm_index: int) -> set[int]:
    links_per_hand = len(finger_ids) // 2
    begin = int(arm_index) * links_per_hand
    return set(finger_ids[begin : begin + links_per_hand]) | {pad_ids[int(arm_index)]}


def append_row(store: dict[str, list[np.ndarray]], **values) -> None:
    for key, value in values.items():
        store[key].append(np.asarray(value))


def load_episode(episode_dir: Path, horizon: int, source_index: int) -> tuple[dict[str, list[np.ndarray]], dict]:
    npz_path = episode_dir / "expert_pick_place_episode.npz"
    scene_path = episode_dir / "g1_language_pick_place.xml"
    summary_path = episode_dir / "language_pick_place_summary.json"
    if not npz_path.is_file() or not scene_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Missing expert NPZ, scene XML, or summary in {episode_dir}")
    z = np.load(npz_path, allow_pickle=True)
    summary = json.loads(summary_path.read_text())
    joint_names = z["joint_names"].astype(str).tolist()
    states43 = z["joint_position_rad"].astype(np.float32)
    velocities43 = z["joint_velocity_rad_s"].astype(np.float32)
    actions43 = z["action_joint_position_rad"].astype(np.float32)
    objects = z["object_position_m"].astype(np.float32)
    contact_trace = z["physical_contact_active"].astype(np.float32)
    active_arm = z["active_arm_index"].astype(np.int64)
    target_index = z["target_object_index"].astype(np.int64)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    upper_names = list(controlled)[base.LOWER_BODY_JOINTS :]
    upper_indices = np.asarray([joint_names.index(name) for name in upper_names])
    object_bodies = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in OBJECT_NAMES
    }
    object_geoms = {
        name: {geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) == body}
        for name, body in object_bodies.items()
    }
    finger_ids = finger_contact_geom_ids(model)
    pad_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_palm_contact_pad")
        for side in ("left", "right")
    ]
    hand_geoms = set(finger_ids) | set(pad_ids)
    rows = {
        key: []
        for key in (
            "state", "velocity", "objects", "sites", "tactile", "action_delta", "target", "arm",
            "next_sites", "next_objects_true", "next_target_distance_m", "next_contact",
            "action_chunk", "horizon_final_distance_m", "horizon_min_distance_m", "horizon_contact",
            "initial_target_contact", "horizon_target_contact_fraction", "horizon_target_contact_last",
            "horizon_target_lift_m", "horizon_hand_object_comotion_m", "horizon_target_bin_xy_distance_m",
            "horizon_stable_grasp", "decision_id", "candidate_rank", "candidate_token",
            "selected_for_rollout", "source_episode",
        )
    }
    max_start = len(states43) - horizon - 1
    for frame in range(max(0, max_start)):
        mujoco.mj_resetData(model, data)
        for name in joint_names:
            if name not in controlled:
                continue
            data.qpos[controlled[name]["qpos_id"]] = states43[frame, joint_names.index(name)]
            data.qvel[controlled[name]["qvel_id"]] = velocities43[frame, joint_names.index(name)]
        for object_id, name in enumerate(OBJECT_NAMES):
            body = object_bodies[name]
            joint = int(model.body_jntadr[body])
            address = int(model.jnt_qposadr[joint])
            data.qpos[address : address + 3] = objects[frame, object_id]
            data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(model, data)
        sites = finger_contact_sites(model, data, finger_ids).astype(np.float32)
        arm = int(active_arm[frame])
        target = int(target_index[frame])
        active_geoms = active_hand_geometries(finger_ids, pad_ids, arm)
        initial_contact = target_contact(model, data, active_geoms, object_geoms[OBJECT_NAMES[target]])
        future_slice = slice(frame, frame + horizon)
        state = states43[frame, upper_indices]
        velocity = velocities43[frame, upper_indices]
        action_chunk = actions43[future_slice, upper_indices] - states43[future_slice, upper_indices]
        next_frame = frame + horizon
        next_objects = objects[next_frame].reshape(-1)
        next_sites = sites
        target_positions = objects[future_slice, target]
        active_center_start = z["palm_position_m"][frame, arm].astype(np.float32)
        active_center_end = z["palm_position_m"][next_frame, arm].astype(np.float32)
        target_motion = objects[next_frame, target] - objects[frame, target]
        hand_motion = active_center_end - active_center_start
        active_sites = sites.reshape(2, -1, 3)[arm]
        distance_now = float(np.linalg.norm(active_sites - objects[frame, target], axis=-1).min())
        min_distance = distance_now
        final_distance = float(
            np.linalg.norm(z["palm_position_m"][next_frame, arm] - objects[next_frame, target])
        )
        contact_fraction = float(contact_trace[future_slice].mean())
        contact_last = float(contact_trace[frame + horizon - 1])
        target_lift = float(target_positions[:, 2].max() - objects[frame, target, 2])
        append_row(
            rows,
            state=state,
            velocity=velocity,
            objects=objects[frame].reshape(-1),
            sites=sites,
            tactile=tactile_contact(model, data, hand_geoms, set(object_bodies.values())),
            action_delta=action_chunk[0],
            target=target,
            arm=arm,
            next_sites=next_sites,
            next_objects_true=next_objects,
            next_target_distance_m=final_distance,
            next_contact=contact_last,
            action_chunk=action_chunk.reshape(-1),
            horizon_final_distance_m=final_distance,
            horizon_min_distance_m=min_distance,
            horizon_contact=float(contact_fraction > 0.0),
            initial_target_contact=initial_contact,
            horizon_target_contact_fraction=contact_fraction,
            horizon_target_contact_last=contact_last,
            horizon_target_lift_m=target_lift,
            horizon_hand_object_comotion_m=float(np.linalg.norm(target_motion - hand_motion)),
            horizon_target_bin_xy_distance_m=float(
                np.linalg.norm(objects[next_frame, target, :2] - base.BIN_POSITION_M[:2])
            ),
            horizon_stable_grasp=int(contact_fraction >= 0.75 and contact_last > 0.5 and target_lift >= 0.015),
            decision_id=source_index * 100000 + frame,
            candidate_rank=0,
            candidate_token=-1,
            selected_for_rollout=1,
            source_episode=source_index,
        )
    z.close()
    report = {
        "episode_dir": str(episode_dir),
        "target_object": summary["target_object"],
        "rows": len(rows["state"]),
        "physical_grasp_passed": summary["grasp_physics_audit"]["physical_grasp_passed"],
        "target_lift_m": summary["grasp_physics_audit"]["target_lift_m"],
    }
    return rows, report


def pack(rows: dict[str, list[np.ndarray]]) -> dict[str, torch.Tensor]:
    ints = {"target", "arm", "decision_id", "candidate_rank", "candidate_token", "selected_for_rollout", "source_episode"}
    return {
        key: torch.from_numpy(np.stack(value)).long() if key in ints else torch.from_numpy(np.stack(value)).float()
        for key, value in rows.items()
    }


def main() -> None:
    args = parse_args()
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    if not (0.0 <= args.validation_ratio < 1.0):
        raise ValueError("--validation-ratio must be in [0,1)")
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed cache: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = {"train": {}, "validation": {}}
    reports = []
    rng = np.random.default_rng(args.seed)
    episode_dirs = list(args.episode_dir)
    validation_count = (
        max(1, round(len(episode_dirs) * args.validation_ratio))
        if len(episode_dirs) > 1 and args.validation_ratio > 0.0
        else 0
    )
    validation_indices = set(rng.choice(len(episode_dirs), validation_count, replace=False).tolist())
    for index, episode_dir in enumerate(episode_dirs):
        rows, report = load_episode(episode_dir, args.horizon, index)
        split = "validation" if index in validation_indices else "train"
        if not all_rows[split]:
            all_rows[split] = {key: [] for key in rows}
        for key, value in rows.items():
            all_rows[split][key].extend(value)
        report["split"] = split
        reports.append(report)
    for split, rows in all_rows.items():
        if not rows:
            rows = {key: value[:] for key, value in all_rows["train"].items()}
        cache = pack(rows)
        cache.update(
            input_contract="strict-contact expert trajectory state + teacher object pose; offline positive branch pretraining only",
            labels="MuJoCo physical expert horizon contact, lift, co-motion, and bin progress",
            privileged_positions_at_inference=False,
            simulator_object_truth_at_inference=False,
            expert_positive_pretraining=True,
        )
        torch.save(cache, args.output_dir / f"{split}.pt")
    summary = {
        "experiment": "G1LANG-strict-contact-expert-task-value-positive-cache",
        "episodes": len(episode_dirs),
        "horizon": args.horizon,
        "reports": reports,
        "purpose": "seed task-value WM/VLA candidate contract with real contact-lift positives before deployment-realistic resampling",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
