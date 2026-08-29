#!/usr/bin/env python3
"""Read-only temporal comparison of a closed-loop rollout and strict experts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from evaluate_g1_language_pick_place import OBJECT_NAMES
from g1_finger_contact_geometry import finger_contact_geom_ids, finger_contact_sites


STATE_THRESHOLD_RAD = 0.025
ACTION_THRESHOLD_RAD = 0.030
GAP_THRESHOLD_M = 0.020
CONSECUTIVE_FRAMES = 3


@dataclass(frozen=True)
class ExpertEpisode:
    name: str
    source_root: str
    target: int
    state: np.ndarray
    action: np.ndarray
    active_arm: np.ndarray
    contact_sites: np.ndarray
    object_position: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rmse(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=-1))


def first_sustained(values: np.ndarray, threshold: float) -> int | None:
    above = values > threshold
    for index in range(max(0, len(above) - CONSECUTIVE_FRAMES + 1)):
        if bool(above[index : index + CONSECUTIVE_FRAMES].all()):
            return index
    return None


def resolve(path: str) -> Path:
    return Path(path).resolve()


def load_experts(cache_dir: Path, expert_root: Path) -> list[ExpertEpisode]:
    experts: list[ExpertEpisode] = []
    seen: set[tuple[str, str]] = set()
    expected_root = expert_root.resolve()
    for name in ("train.pt", "val.pt"):
        cache = torch.load(cache_dir / name, map_location="cpu", weights_only=False)
        sites = cache["finger_contact_site_position"].numpy()
        episode_count, frames = cache["state"].shape[:2]
        sites = sites.reshape(episode_count, frames, -1)
        for index, source_root in enumerate(cache["source_root"]):
            key = (source_root, cache["episode_name"][index])
            if key in seen or resolve(source_root) != expected_root:
                continue
            seen.add(key)
            experts.append(
                ExpertEpisode(
                    name=cache["episode_name"][index],
                    source_root=source_root,
                    target=int(cache["target"][index]),
                    state=cache["state"][index].numpy(),
                    action=cache["candidate_action"][index, :, 0].numpy(),
                    active_arm=cache["active_arm"][index].numpy(),
                    contact_sites=sites[index],
                    object_position=cache["audit_true_object_position"][index]
                    .numpy()
                    .reshape(frames, len(OBJECT_NAMES), 3),
                )
            )
    if not experts:
        raise ValueError(f"No strict experts from {expert_root} found in {cache_dir}")
    return experts


def rollout_contact_sites(scene: Path, joint_names: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    geom_ids = finger_contact_geom_ids(model)
    qpos_addresses = []
    for name in joint_names.tolist():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id < 0:
            raise ValueError(f"Joint {name} is missing from {scene}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
    result = []
    for state in qpos:
        data.qpos[qpos_addresses] = state
        mujoco.mj_forward(model, data)
        result.append(finger_contact_sites(model, data, geom_ids))
    return np.asarray(result, dtype=np.float32)


def active_gap(
    sites: np.ndarray, objects: np.ndarray, target: int, active_arm: np.ndarray
) -> np.ndarray:
    per_arm_values = sites.reshape(len(sites), 2, -1, 3)
    chosen = per_arm_values[np.arange(len(sites)), active_arm]
    return np.linalg.norm(chosen - objects[:, target, None, :], axis=-1).min(axis=-1)


def choose_expert(
    experts: list[ExpertEpisode],
    target: int,
    active_arm: int,
    rollout_state: np.ndarray,
    rollout_action: np.ndarray,
    rollout_objects: np.ndarray,
) -> tuple[ExpertEpisode, dict]:
    candidates = [
        expert
        for expert in experts
        if expert.target == target and int(expert.active_arm[0]) == active_arm
    ]
    if not candidates:
        raise ValueError(f"No strict expert for target={target}, active_arm={active_arm}")
    rows = []
    for expert in candidates:
        state_error = float(rmse(rollout_state[0] - expert.state[0]))
        action_error = float(rmse(rollout_action[0] - expert.action[0]))
        object_error = float(
            np.sqrt(np.mean(np.square(rollout_objects[0] - expert.object_position[0])))
        )
        # Object layout is the primary identity signal. State and the first
        # deployed action disambiguate expert branches sharing a layout.
        score = object_error / 0.010 + state_error / 0.020 + action_error / 0.020
        rows.append((score, state_error, action_error, object_error, expert))
    score, state_error, action_error, object_error, expert = min(rows, key=lambda row: row[0])
    return expert, {
        "selection_score": score,
        "initial_state_rmse_rad": state_error,
        "initial_action_rmse_rad": action_error,
        "initial_all_object_position_rmse_m": object_error,
    }


def main() -> None:
    args = parse_args()
    experts = load_experts(args.cache_dir, args.expert_root)
    reports = json.loads((args.rollout_dir / "summary.json").read_text())["reports"]
    episodes = []
    for report in reports:
        episode_dir = args.rollout_dir / f"episode_{int(report['episode_index']):04d}"
        with np.load(episode_dir / "trajectory.npz") as trajectory:
            state = trajectory["observation_joint_position_rad"].astype(np.float32)
            action = trajectory["action_joint_position_rad"].astype(np.float32)
            objects = trajectory["object_position_m"].astype(np.float32)
            names = trajectory["joint_names"]
        target = OBJECT_NAMES.index(report["target_object"])
        candidates = report["world_model_candidates"]
        latched_arm = int(candidates[0]["active_arm_index"])
        expert, selection = choose_expert(
            experts, target, latched_arm, state, action, objects
        )
        frames = min(len(state), len(expert.state))
        state = state[:frames]
        action = action[:frames]
        objects = objects[:frames]
        deployed_sites = rollout_contact_sites(episode_dir / "scene.xml", names, state)
        expert_active_arm = expert.active_arm[:frames]
        deployed_active_arm = np.full(frames, latched_arm, dtype=np.int64)
        state_error = rmse(state - expert.state[:frames])
        action_error = rmse(action - expert.action[:frames])
        deployed_gap = active_gap(deployed_sites, objects, target, deployed_active_arm)
        expert_gap = active_gap(
            expert.contact_sites[:frames],
            expert.object_position[:frames],
            target,
            expert_active_arm,
        )
        gap_error = deployed_gap - expert_gap
        first_action = first_sustained(action_error, ACTION_THRESHOLD_RAD)
        first_state = first_sustained(state_error, STATE_THRESHOLD_RAD)
        first_gap = first_sustained(gap_error, GAP_THRESHOLD_M)
        first_any = min(
            value for value in (first_action, first_state, first_gap) if value is not None
        ) if any(value is not None for value in (first_action, first_state, first_gap)) else None
        selected_scales = np.asarray(
            [row["candidate_plan_scales"][row["selected_candidate"]] for row in candidates[:frames]],
            dtype=np.float32,
        )
        first_context = None
        if first_any is not None and first_any < len(candidates):
            first_context = {
                "frame": first_any,
                "time_s": first_any / 15.0,
                "deployed_action_rmse_rad": float(action_error[first_any]),
                "state_rmse_rad": float(state_error[first_any]),
                "deployed_minus_expert_gap_m": float(gap_error[first_any]),
                "selected_plan_scale": float(selected_scales[first_any]),
                "candidate_scores": candidates[first_any]["candidate_scores"],
            }
        episodes.append(
            {
                "episode_index": int(report["episode_index"]),
                "target_object": report["target_object"],
                "latched_active_arm_index": latched_arm,
                "nearest_strict_expert": {
                    "name": expert.name,
                    "source_root": expert.source_root,
                    **selection,
                },
                "frames_compared": frames,
                "active_arm_matches_expert_fraction": float(
                    np.mean(deployed_active_arm == expert_active_arm)
                ),
                "deployed_action_rmse_rad": {
                    "mean": float(action_error.mean()),
                    "p95": float(np.quantile(action_error, 0.95)),
                    "max": float(action_error.max()),
                },
                "state_rmse_rad": {
                    "mean": float(state_error.mean()),
                    "p95": float(np.quantile(state_error, 0.95)),
                    "max": float(state_error.max()),
                },
                "active_finger_gap_m": {
                    "deployed_mean": float(deployed_gap.mean()),
                    "deployed_min": float(deployed_gap.min()),
                    "expert_mean": float(expert_gap.mean()),
                    "expert_min": float(expert_gap.min()),
                    "deployed_minus_expert_mean": float(gap_error.mean()),
                },
                "first_sustained_departure": {
                    "action_frame": first_action,
                    "state_frame": first_state,
                    "finger_gap_frame": first_gap,
                    "any_frame": first_any,
                    "context": first_context,
                },
                "selected_plan_scales": {
                    "0.70_fraction": float(np.mean(np.isclose(selected_scales, 0.70))),
                    "1.00_fraction": float(np.mean(np.isclose(selected_scales, 1.00))),
                    "1.20_fraction": float(np.mean(np.isclose(selected_scales, 1.20))),
                },
            }
        )
    summary = {
        "experiment": "G1LANG-396-temporal-strict-expert-audit",
        "audit_contract": (
            "read-only post-hoc comparison; strict-expert object positions are used "
            "only to identify and assess the matching source trajectory, never at inference"
        ),
        "thresholds": {
            "action_rmse_rad": ACTION_THRESHOLD_RAD,
            "state_rmse_rad": STATE_THRESHOLD_RAD,
            "finger_gap_excess_m": GAP_THRESHOLD_M,
            "consecutive_frames": CONSECUTIVE_FRAMES,
        },
        "strict_expert_count": len(experts),
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
