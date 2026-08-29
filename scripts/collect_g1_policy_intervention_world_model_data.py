#!/usr/bin/env python3
"""Collect real MuJoCo transitions for action-token VLA candidate actions.

Each decision starts from a state actually visited by a VLA-candidate rollout.
All top-k VLA proposals are independently executed through the same PD loop,
then labelled with their measured successor geometry and short-horizon physical
task outcomes. Object positions and target contact identity are teacher labels
only; the learned model receives the VLA's RGB-derived object estimate recorded
at that decision.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch.nn import functional as F

import evaluate_g1_language_pick_place as base
from g1_action_token_vla import load_checkpoint as load_vla
from g1_direct_joint_candidate_vla import encode_tasks
from g1_finger_contact_geometry import finger_contact_geom_ids, finger_contact_sites
from run_g1_language_pick_place import (
    HOME_POSITIONS_M,
    OBJECT_NAMES,
    PALM_NAMES,
    arm_ik_contract,
    build_scene,
    controlled_joints,
    solve_pose,
)
from validate_g1_bimanual_actuation import apply_regularized_dynamics, unitree_gains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument(
        "--vla-checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint to propose actions from; repeat to form a fixed-size multi-policy candidate union.",
    )
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--validation-episodes", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--proposal-count", type=int, default=12)
    parser.add_argument(
        "--multi-policy-mode",
        choices=("union", "target"),
        default="union",
        help=(
            "With repeated checkpoints, either pool a fixed candidate budget or route the full "
            "budget by predicted language target. Target mode expects checkpoints ordered as "
            "red_triangle, yellow_rod, green_cube."
        ),
    )
    parser.add_argument(
        "--rollout-horizon",
        type=int,
        default=1,
        help="Physical candidate-action rollout horizon; deployment still replans after action one.",
    )
    parser.add_argument("--action-delta-limit-rad", type=float, default=0.08)
    parser.add_argument("--body-gain-scale", type=float, default=5.0)
    parser.add_argument("--hand-gain-scale", type=float, default=6.0)
    parser.add_argument("--explore-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260746)
    parser.add_argument("--observation-width", type=int, default=160)
    parser.add_argument("--observation-height", type=int, default=120)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class ActionTokenProposals:
    """The deployed VLA observation/history contract, without a world model."""

    def __init__(self, checkpoint: Path, device: str, proposal_count: int, limit: float):
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.model, self.vocab, self.stats, self.codebook, _ = load_vla(checkpoint, str(self.device))
        self.proposal_count = min(int(proposal_count), self.model.token_count)
        self.limit = float(limit)
        self.history: deque[dict[str, torch.Tensor | list[str]]] = deque(maxlen=self.model.history_frames)

    def reset(self) -> None:
        self.history.clear()

    def observe(
        self,
        batch: dict,
        velocity: np.ndarray,
        sites: np.ndarray,
        tactile: float,
    ) -> None:
        self.history.append(
            {
                "images": torch.stack([batch[key][0].to(self.device) for key in base.CAMERAS], dim=0),
                "state": batch["observation.state"].to(self.device),
                "velocity": torch.from_numpy(velocity[None]).to(self.device),
                "sites": torch.from_numpy(sites[None]).to(self.device),
                "tactile": torch.tensor([[tactile]], dtype=torch.float32, device=self.device),
                "task": list(batch["task"]),
            }
        )

    def _stack(self) -> dict[str, torch.Tensor]:
        rows = list(self.history)
        rows = [rows[0]] * (self.model.history_frames - len(rows)) + rows
        result = {
            name: torch.stack([row[name] for row in rows], dim=1)  # type: ignore[index]
            for name in ("state", "velocity", "sites", "tactile")
        }
        result["images"] = torch.stack([row["images"] for row in rows], dim=0).unsqueeze(0)  # type: ignore[index]
        return result

    @torch.no_grad()
    def predict(self) -> dict[str, np.ndarray | int]:
        observation = self._stack()
        task = self.history[-1]["task"]  # type: ignore[index]
        ids, mask = encode_tasks(task, self.vocab)
        output = self.model(
            observation["images"],
            (observation["state"] - self.stats["state_mean"]) / self.stats["state_std"],
            (observation["velocity"] - self.stats["velocity_mean"]) / self.stats["velocity_std"],
            (observation["sites"] - self.stats["contact_sites_mean"]) / self.stats["contact_sites_std"],
            observation["tactile"],
            ids.to(self.device),
            mask.to(self.device),
        )
        logits = output["action_token_logits"][0]
        token_ids = logits.topk(self.proposal_count).indices
        # The first chunk action is exactly what the closed-loop evaluator can
        # execute before it re-observes RGB and tactile state on the next tick.
        chunks = (
            self.codebook[token_ids] + output["candidate_action_residual"][0, token_ids]
        ).clamp(-self.limit, self.limit)
        visual_objects = output["object_mean"] * self.stats["objects_std"] + self.stats["objects_mean"]
        return {
            "deltas": chunks[:, 0].cpu().numpy().astype(np.float32),
            "chunks": chunks.cpu().numpy().astype(np.float32),
            "objects": visual_objects[0].cpu().numpy().astype(np.float32),
            "target": int(output["target_logits"].argmax(-1)[0]),
            "arm": int(output["active_arm_logits"].argmax(-1)[0]),
            "token_ids": token_ids.cpu().numpy().astype(np.int64),
        }


class MultiActionTokenProposals:
    """Union equally sized top-k candidate sets from several frozen VLAs.

    The first checkpoint remains the primary source for visual object, target,
    and arm predictions.  Only action candidates are pooled; no simulator truth
    is used to form the union.
    """

    def __init__(self, checkpoints: list[Path], device: str, proposal_count: int, limit: float):
        if proposal_count < len(checkpoints):
            raise ValueError("--proposal-count must be at least the number of checkpoints")
        base_count, remainder = divmod(int(proposal_count), len(checkpoints))
        allocations = [base_count + int(index < remainder) for index in range(len(checkpoints))]
        self.models = [
            ActionTokenProposals(checkpoint, device, count, limit)
            for checkpoint, count in zip(checkpoints, allocations)
        ]
        self.device = self.models[0].device

    def reset(self) -> None:
        for model in self.models:
            model.reset()

    def observe(self, batch: dict, velocity: np.ndarray, sites: np.ndarray, tactile: float) -> None:
        for model in self.models:
            model.observe(batch, velocity, sites, tactile)

    def predict(self) -> dict[str, np.ndarray | int]:
        predictions = [model.predict() for model in self.models]
        primary = predictions[0]
        return {
            "deltas": np.concatenate([prediction["deltas"] for prediction in predictions], axis=0),
            "chunks": np.concatenate([prediction["chunks"] for prediction in predictions], axis=0),
            "objects": primary["objects"],
            "target": primary["target"],
            "arm": primary["arm"],
            # Preserve candidate provenance without changing the intervention
            # cache schema.  Existing token IDs are far below this offset.
            "token_ids": np.concatenate(
                [prediction["token_ids"] + 10_000 * index for index, prediction in enumerate(predictions)],
                axis=0,
            ),
        }


class TargetRoutedActionTokenProposals:
    """Route a full candidate budget using only the VLA's language target class."""

    def __init__(self, checkpoints: list[Path], device: str, proposal_count: int, limit: float):
        if len(checkpoints) != len(OBJECT_NAMES):
            raise ValueError(
                f"target routing requires {len(OBJECT_NAMES)} checkpoints ordered as {OBJECT_NAMES}"
            )
        unique: dict[Path, ActionTokenProposals] = {}
        self.models = []
        for checkpoint in checkpoints:
            key = checkpoint.resolve()
            if key not in unique:
                unique[key] = ActionTokenProposals(checkpoint, device, proposal_count, limit)
            self.models.append(unique[key])
        self.unique_models = list(unique.values())
        self.device = self.models[0].device

    def reset(self) -> None:
        for model in self.unique_models:
            model.reset()

    def observe(self, batch: dict, velocity: np.ndarray, sites: np.ndarray, tactile: float) -> None:
        for model in self.unique_models:
            model.observe(batch, velocity, sites, tactile)

    def predict(self) -> dict[str, np.ndarray | int]:
        primary_model = self.models[0]
        primary = primary_model.predict()
        target = int(primary["target"])
        selected_model = self.models[target]
        selected = dict(primary if selected_model is primary_model else selected_model.predict())
        selected["target"] = target
        selected["token_ids"] = selected["token_ids"] + 10_000 * target
        return selected


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
    """Return only the selected arm's physical hand contact geometry."""
    links_per_hand = len(finger_ids) // 2
    begin = int(arm_index) * links_per_hand
    return set(finger_ids[begin : begin + links_per_hand]) | {pad_ids[int(arm_index)]}


def upper_state(data, controlled, upper_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([data.qpos[controlled[name]["qpos_id"]] for name in upper_names], dtype=np.float32),
        np.asarray([data.qvel[controlled[name]["qvel_id"]] for name in upper_names], dtype=np.float32),
    )


def execute_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    upper_names: list[str],
    initial_targets: np.ndarray,
    target: np.ndarray,
    physics_per_control: int,
    body_gain_scale: float,
    hand_gain_scale: float,
) -> None:
    commanded = initial_targets.copy()
    commanded[base.LOWER_BODY_JOINTS :] = target
    for _ in range(physics_per_control):
        for name, item in controlled.items():
            kp, kd = unitree_gains(name)
            gain = hand_gain_scale if "hand_" in name else body_gain_scale
            qpos = float(data.qpos[item["qpos_id"]])
            qvel = float(data.qvel[item["qvel_id"]])
            data.ctrl[item["actuator_id"]] = (
                gain * kp * (commanded[item["actuator_id"]] - qpos)
                - math.sqrt(gain) * kd * qvel
                + float(data.qfrc_bias[item["qvel_id"]])
            )
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("MuJoCo became non-finite while collecting an intervention")


def bounded_target(
    state: np.ndarray,
    delta: np.ndarray,
    previous_action: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    limit: float,
) -> np.ndarray:
    requested = np.clip(state + delta, lower, upper)
    return previous_action + np.clip(requested - previous_action, -limit, limit)


def initialize_scene(args: argparse.Namespace, spec: dict, expected_upper_names: list[str]):
    episode_dir = args.output_dir / f"scene_{spec['episode_index']:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    positions = {
        name: np.asarray(
            (
                *(base.OBJECT_SLOTS_XY_M[spec["permutation"][index]] + spec["offsets"][index]),
                base.OBJECT_SPAWN_Z_M[name],
            )
        )
        for index, name in enumerate(OBJECT_NAMES)
    }
    scene_path = episode_dir / "scene.xml"
    build_scene(args.asset.resolve(), scene_path, positions)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    upper_names = list(controlled)[base.LOWER_BODY_JOINTS :]
    if upper_names != expected_upper_names:
        raise ValueError("Simulation and dataset joint ordering differ")
    _, selected_qpos_ids, lower, upper = arm_ik_contract(model)
    palm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_NAMES]
    mujoco.mj_resetData(model, data)
    data.eq_active[:] = 0
    mujoco.mj_forward(model, data)
    home_pose, report = solve_pose(
        model,
        data,
        palm_ids,
        selected_qpos_ids,
        lower,
        upper,
        HOME_POSITIONS_M,
        np.zeros(len(selected_qpos_ids)),
        orientation_weight=0.06,
    )
    if not report["solver_success"]:
        raise RuntimeError("Failed to solve collection home pose")
    mujoco.mj_resetData(model, data)
    data.eq_active[:] = 0
    data.qpos[selected_qpos_ids] = home_pose
    for name, item in controlled.items():
        if "hand_" not in name:
            continue
        joint_lower, joint_upper = model.jnt_range[item["joint_id"]]
        data.qpos[item["qpos_id"]] = 0.01 if abs(float(joint_lower)) < 1e-9 else -0.01 if abs(float(joint_upper)) < 1e-9 else 0.5 * (joint_lower + joint_upper)
    mujoco.mj_forward(model, data)
    initial_targets = np.asarray([data.qpos[controlled[name]["qpos_id"]] for name in controlled], dtype=np.float64)
    action_lower = np.asarray([model.jnt_range[controlled[name]["joint_id"], 0] for name in upper_names], dtype=np.float32)
    action_upper = np.asarray([model.jnt_range[controlled[name]["joint_id"], 1] for name in upper_names], dtype=np.float32)
    finger_ids = finger_contact_geom_ids(model)
    pads = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_palm_contact_pad") for side in ("left", "right")]
    if min(pads) < 0:
        raise RuntimeError("Scene is missing physical palm contact pads")
    object_bodies = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in OBJECT_NAMES}
    object_geoms = {
        name: {geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) == body}
        for name, body in object_bodies.items()
    }
    return {
        "episode_dir": episode_dir,
        "model": model,
        "data": data,
        "controlled": controlled,
        "upper_names": upper_names,
        "initial_targets": initial_targets,
        "action_lower": action_lower,
        "action_upper": action_upper,
        "finger_ids": finger_ids,
        "hand_geoms": set(finger_ids) | set(pads),
        "hand_geoms_by_arm": [
            active_hand_geometries(finger_ids, pads, arm_index)
            for arm_index in range(2)
        ],
        "object_bodies": object_bodies,
        "object_geoms": object_geoms,
    }


def append_row(store: dict[str, list[np.ndarray]], **values: np.ndarray | float | int) -> None:
    for key, value in values.items():
        store[key].append(np.asarray(value))


def collect_episode(
    args: argparse.Namespace,
    spec: dict,
    expected_upper_names: list[str],
    proposals: ActionTokenProposals,
    decision_start: int,
) -> tuple[dict[str, list[np.ndarray]], dict, int]:
    env = initialize_scene(args, spec, expected_upper_names)
    model, data, controlled = env["model"], env["data"], env["controlled"]
    upper_names = env["upper_names"]
    renderer = mujoco.Renderer(model, height=args.observation_height, width=args.observation_width)
    state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    saved_state = np.empty(state_size, dtype=np.float64)
    clone = mujoco.MjData(model)
    physics_per_control = max(1, round(1.0 / (args.control_fps * model.opt.timestep)))
    object_bodies: dict[str, int] = env["object_bodies"]
    rows = {
        name: []
        for name in (
            "state", "velocity", "objects", "sites", "tactile", "action_delta", "target", "arm",
            "next_sites", "next_objects_true", "next_target_distance_m", "next_contact",
            "action_chunk", "horizon_final_distance_m", "horizon_min_distance_m", "horizon_contact",
            "initial_target_contact", "horizon_target_contact_fraction", "horizon_target_contact_last",
            "horizon_target_lift_m", "horizon_hand_object_comotion_m", "horizon_target_bin_xy_distance_m",
            "horizon_stable_grasp",
            "decision_id", "candidate_rank", "candidate_token", "selected_for_rollout", "source_episode",
        )
    }
    proposals.reset()
    previous_action = env["initial_targets"][base.LOWER_BODY_JOINTS :].astype(np.float32).copy()
    rng = np.random.default_rng(args.seed + spec["episode_index"] * 977)
    decision_id = decision_start
    physical_distances: list[float] = []
    contacts = 0
    for frame in range(round(args.duration_s * args.control_fps)):
        state, velocity = upper_state(data, controlled, upper_names)
        sites = finger_contact_sites(model, data, env["finger_ids"]).astype(np.float32)
        tactile = tactile_contact(model, data, env["hand_geoms"], set(object_bodies.values()))
        batch = base.render_observation(renderer, data, state, spec["instruction"], str(proposals.device))
        proposals.observe(batch, velocity, sites, tactile)
        proposal = proposals.predict()
        target_index = int(proposal["target"])
        arm_index = int(proposal["arm"])
        target_name = OBJECT_NAMES[target_index]
        active_hand_geoms = env["hand_geoms_by_arm"][arm_index]
        initial_objects = np.stack(
            [data.xpos[object_bodies[name]].copy() for name in OBJECT_NAMES]
        ).astype(np.float32)
        initial_target_contact = target_contact(
            model, data, active_hand_geoms, env["object_geoms"][target_name]
        )
        candidate_delta = np.asarray(proposal["deltas"], dtype=np.float32)
        candidate_chunks = np.asarray(proposal["chunks"], dtype=np.float32)[:, : args.rollout_horizon]
        candidate_targets = np.stack(
            [
                bounded_target(
                    state,
                    item,
                    previous_action,
                    env["action_lower"],
                    env["action_upper"],
                    args.action_delta_limit_rad,
                )
                for item in candidate_delta
            ]
        )
        mujoco.mj_getState(model, data, saved_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        outcomes = []
        for rank, command in enumerate(candidate_targets):
            mujoco.mj_setState(model, clone, saved_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(model, clone)
            branch_previous = previous_action.copy()
            branch_distances = []
            branch_contact_steps = 0
            branch_contact_last = 0.0
            branch_target_max_z = float(initial_objects[target_index, 2])
            initial_active_sites = sites.reshape(2, -1, 3)[arm_index].mean(axis=0)
            for horizon_step, proposed_delta in enumerate(candidate_chunks[rank]):
                branch_state, _ = upper_state(clone, controlled, upper_names)
                branch_command = bounded_target(
                    branch_state,
                    proposed_delta,
                    branch_previous,
                    env["action_lower"],
                    env["action_upper"],
                    args.action_delta_limit_rad,
                )
                execute_control(
                    model,
                    clone,
                    controlled,
                    upper_names,
                    env["initial_targets"],
                    branch_command,
                    physics_per_control,
                    args.body_gain_scale,
                    args.hand_gain_scale,
                )
                branch_previous = branch_command
                interim_sites = finger_contact_sites(model, clone, env["finger_ids"]).astype(np.float32)
                interim_objects = np.stack([clone.xpos[object_bodies[name]].copy() for name in OBJECT_NAMES]).astype(np.float32)
                branch_distances.append(
                    float(
                        np.linalg.norm(
                            interim_sites.reshape(2, -1, 3)[arm_index] - interim_objects[target_index],
                            axis=-1,
                        ).min()
                    )
                )
                branch_target_contact = target_contact(
                    model, clone, active_hand_geoms, env["object_geoms"][target_name]
                )
                branch_contact_steps += int(branch_target_contact > 0.5)
                branch_contact_last = branch_target_contact
                branch_target_max_z = max(
                    branch_target_max_z,
                    float(clone.xpos[object_bodies[target_name], 2]),
                )
            next_sites = finger_contact_sites(model, clone, env["finger_ids"]).astype(np.float32)
            next_objects = np.stack([clone.xpos[object_bodies[name]].copy() for name in OBJECT_NAMES]).astype(np.float32)
            active_sites = next_sites.reshape(2, -1, 3)[arm_index]
            distance = float(np.linalg.norm(active_sites - next_objects[target_index], axis=-1).min())
            final_active_center = active_sites.mean(axis=0)
            target_motion = next_objects[target_index] - initial_objects[target_index]
            hand_motion = final_active_center - initial_active_sites
            contact_fraction = branch_contact_steps / max(len(candidate_chunks[rank]), 1)
            target_lift = branch_target_max_z - float(initial_objects[target_index, 2])
            outcomes.append(
                {
                    "command": command.astype(np.float32),
                    "next_sites": next_sites,
                    "next_objects": next_objects,
                    "distance": distance,
                    "horizon_final_distance": branch_distances[-1],
                    "horizon_min_distance": min(branch_distances),
                    "horizon_contact": float(branch_contact_steps > 0),
                    "target_contact_fraction": contact_fraction,
                    "target_contact_last": branch_contact_last,
                    "target_lift": target_lift,
                    "hand_object_comotion": float(np.linalg.norm(target_motion - hand_motion)),
                    "target_bin_xy_distance": float(
                        np.linalg.norm(
                            next_objects[target_index, :2] - base.BIN_POSITION_M[:2]
                        )
                    ),
                    # A short-horizon grasp label deliberately requires both
                    # sustained selected-hand contact and upward object motion.
                    # It is teacher-only, never a deployment input.
                    "stable_grasp": int(
                        contact_fraction >= 0.75
                        and branch_contact_last > 0.5
                        and target_lift >= 0.015
                    ),
                    "contact": tactile_contact(model, clone, env["hand_geoms"], set(object_bodies.values())),
                    "target_contact": target_contact(model, clone, active_hand_geoms, env["object_geoms"][target_name]),
                    "rank": rank,
                }
            )
        # The rollout uses a candidate action, so subsequent collection states
        # are policy-reachable. The physical-distance oracle is collection-only
        # and is never exposed to deployment inputs or the learned policy.
        if rng.random() < args.explore_probability:
            chosen = int(rng.integers(len(outcomes)))
        else:
            chosen = int(np.argmin([item["distance"] for item in outcomes]))
        for outcome in outcomes:
            append_row(
                rows,
                state=state,
                velocity=velocity,
                objects=np.asarray(proposal["objects"], dtype=np.float32),
                sites=sites,
                tactile=tactile,
                action_delta=outcome["command"] - state,
                action_chunk=candidate_chunks[outcome["rank"]].reshape(-1),
                target=target_index,
                arm=arm_index,
                next_sites=outcome["next_sites"],
                next_objects_true=outcome["next_objects"].reshape(-1),
                next_target_distance_m=outcome["distance"],
                next_contact=outcome["target_contact"],
                horizon_final_distance_m=outcome["horizon_final_distance"],
                horizon_min_distance_m=outcome["horizon_min_distance"],
                horizon_contact=outcome["horizon_contact"],
                initial_target_contact=initial_target_contact,
                horizon_target_contact_fraction=outcome["target_contact_fraction"],
                horizon_target_contact_last=outcome["target_contact_last"],
                horizon_target_lift_m=outcome["target_lift"],
                horizon_hand_object_comotion_m=outcome["hand_object_comotion"],
                horizon_target_bin_xy_distance_m=outcome["target_bin_xy_distance"],
                horizon_stable_grasp=outcome["stable_grasp"],
                decision_id=decision_id,
                candidate_rank=outcome["rank"],
                candidate_token=np.asarray(proposal["token_ids"])[outcome["rank"]],
                selected_for_rollout=int(outcome["rank"] == chosen),
                source_episode=spec["episode_index"],
            )
        mujoco.mj_setState(model, data, saved_state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        mujoco.mj_forward(model, data)
        execute_control(
            model,
            data,
            controlled,
            upper_names,
            env["initial_targets"],
            outcomes[chosen]["command"],
            physics_per_control,
            args.body_gain_scale,
            args.hand_gain_scale,
        )
        previous_action = outcomes[chosen]["command"].copy()
        physical_distances.append(float(outcomes[chosen]["distance"]))
        contacts += int(outcomes[chosen]["target_contact"] > 0.5)
        decision_id += 1
    close = getattr(renderer, "close", None)
    if close is not None:
        close()
    report = {
        "episode_index": spec["episode_index"],
        "target_object": spec["target_object"],
        "instruction": spec["instruction"],
        "decisions": len(physical_distances),
        "candidate_transitions": len(rows["decision_id"]),
        "initial_selected_distance_mm": 1e3 * physical_distances[0],
        "final_selected_distance_mm": 1e3 * physical_distances[-1],
        "minimum_selected_distance_mm": 1e3 * min(physical_distances),
        "target_contact_steps": contacts,
        "collection_rollout_selector": f"physical {args.rollout_horizon}-step-distance oracle over current VLA candidates; labels remain deployment-realistic",
    }
    print("POLICY_INTERVENTION_EPISODE=" + json.dumps(report), flush=True)
    return rows, report, decision_id


def pack(rows: dict[str, list[np.ndarray]]) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(np.stack(value)).float() if key not in {"target", "arm", "decision_id", "candidate_rank", "candidate_token", "selected_for_rollout", "source_episode"} else torch.from_numpy(np.stack(value)).long() for key, value in rows.items()}


def main() -> None:
    args = parse_args()
    if args.episodes < 6 or args.episodes % 3 or not (1 <= args.validation_episodes < args.episodes):
        raise ValueError("--episodes must be divisible by three and leave at least one validation scene")
    if args.rollout_horizon < 1 or args.rollout_horizon > 8:
        raise ValueError("--rollout-horizon must be in [1, 8]")
    if not (0.0 <= args.explore_probability < 1.0):
        raise ValueError("--explore-probability must be in [0, 1)")
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed collection: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.train_root / "meta/info.json").read_text())
    expected_upper_names = metadata["features"]["action"]["names"]
    checkpoints = list(args.vla_checkpoint)
    if len(checkpoints) == 1:
        proposals = ActionTokenProposals(
            checkpoints[0], args.device, args.proposal_count, args.action_delta_limit_rad
        )
        policy_mode = "single"
    elif args.multi_policy_mode == "target":
        proposals = TargetRoutedActionTokenProposals(
            checkpoints, args.device, args.proposal_count, args.action_delta_limit_rad
        )
        policy_mode = "language_target_router"
    else:
        proposals = MultiActionTokenProposals(
            checkpoints, args.device, args.proposal_count, args.action_delta_limit_rad
        )
        policy_mode = "fixed_budget_union"
    specs = base.evaluation_specs(args.episodes, args.seed, 0.018, "random")
    all_rows: dict[str, dict[str, list[np.ndarray]]] = {"train": {}, "validation": {}}
    reports = []
    decision_id = 0
    for spec in specs:
        rows, report, decision_id = collect_episode(args, spec, expected_upper_names, proposals, decision_id)
        split = "validation" if spec["episode_index"] >= args.episodes - args.validation_episodes else "train"
        if not all_rows[split]:
            all_rows[split] = {key: [] for key in rows}
        for key, value in rows.items():
            all_rows[split][key].extend(value)
        report["split"] = split
        reports.append(report)
    for split, rows in all_rows.items():
        value = pack(rows)
        value.update(
            input_contract="VLA RGB-derived object pose + measured state31/qvel31/finger sites/tactile + executed candidate action + language target + VLA arm",
            labels="MuJoCo physical successor finger sites, object positions, target distance, and contact",
            privileged_positions_at_inference=False,
            simulator_object_truth_at_inference=False,
        )
        torch.save(value, args.output_dir / f"{split}.pt")
    summary = {
        "experiment": "G1LANG-policy-visited-action-token-world-model-interventions",
        "vla_checkpoint": str(checkpoints[0]) if len(checkpoints) == 1 else [str(path) for path in checkpoints],
        "candidate_policy_mode": policy_mode,
        "candidate_union": policy_mode == "fixed_budget_union",
        "candidate_policy_count": len(checkpoints),
        "episodes": args.episodes,
        "validation_episodes": args.validation_episodes,
        "transitions": {split: len(rows["decision_id"]) for split, rows in all_rows.items()},
        "proposal_count": args.proposal_count,
        "rollout_horizon": args.rollout_horizon,
        "input_contract": "VLA RGB-derived object pose + measured state31/qvel31/finger sites/tactile + executed candidate action + language target + VLA arm",
        "labels": "physical MuJoCo successor geometry and distance; object truth is teacher-only",
        "rollout_state_source": "VLA-candidate actions, with offline physical-distance oracle only to reach and cover closer states",
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "reports": reports,
    }
    summary["passed"] = bool(all(report["candidate_transitions"] > 0 for report in reports))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))


if __name__ == "__main__":
    main()
