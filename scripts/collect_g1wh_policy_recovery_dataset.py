#!/usr/bin/env python3
"""Collect expert hold recoveries from states visited by the G1 SmolVLA policy."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

# MuJoCo and PyTorch come from g1ub_genesis; LeRobot's remaining Python
# dependencies are exposed from the dedicated LeRobot environment.
if lerobot_site_packages := os.environ.get("LEROBOT_SITE_PACKAGES"):
    sys.path.append(lerobot_site_packages)

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from evaluate_g1wh_smolvla_closed_loop import (
    LOWER_BODY_JOINTS,
    PHASE_INSTRUCTIONS,
    TASK_CAMERAS,
    controlled_joints,
    load_policy,
    render_observation,
    seeded_noise,
    solve_home_pose,
)
from run_g1_assisted_tote_lift import (
    PALM_NAMES,
    TOTE_SITE_NAMES,
    WAIST_PITCH_LIMIT_RAD,
    build_scene,
    object_name,
    smoothstep,
)
from validate_g1_bimanual_actuation import apply_regularized_dynamics, unitree_gains

RECOVERY_INSTRUCTION = "stabilize and hold the blue tote steady in the air"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_assisted_lift_phase_train")
    parser.add_argument("--source-episodes", type=int, default=16)
    parser.add_argument("--seed-variants", type=int, default=2)
    parser.add_argument("--control-fps", type=int, default=15)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--takeover-delay-s", type=float, default=0.67)
    parser.add_argument("--recovery-blend-s", type=float, default=2.0)
    parser.add_argument("--assist-distance-m", type=float, default=0.08)
    parser.add_argument("--align-distance-m", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=3307)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode-limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_expert_hold_target(
    source_episode: dict, controlled_names: list[str]
) -> np.ndarray:
    with np.load(source_episode["dataset"]) as arrays:
        names = arrays["joint_names"].tolist()
        if names != controlled_names:
            raise ValueError("Expert and simulation joint ordering differ")
        return np.asarray(arrays["action_joint_position_rad"][-1], dtype=np.float64)


@torch.no_grad()
def collect_episode(
    policy,
    source_episode: dict,
    seed_variant: int,
    collection_index: int,
    args: argparse.Namespace,
    expected_upper_names: list[str],
    device: str,
) -> dict:
    source_index = int(source_episode["episode_index"])
    output_dir = args.output_dir / f"episode_{collection_index:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "scene.xml"
    build_scene(args.asset.resolve(), scene_path, float(source_episode["tote_x_m"]))
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    controlled_names = list(controlled)
    upper_names = controlled_names[LOWER_BODY_JOINTS:]
    if upper_names != expected_upper_names:
        raise ValueError("Simulation and LeRobot upper-body joint order differ")

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    _, selected_qpos_ids, home_pose = solve_home_pose(model, data)
    mujoco.mj_resetData(model, data)
    data.qpos[selected_qpos_ids] = home_pose
    for name, item in controlled.items():
        if "hand_" not in name:
            continue
        low, high = model.jnt_range[item["joint_id"]]
        if abs(float(low)) < 1e-9:
            data.qpos[item["qpos_id"]] = 0.01
        elif abs(float(high)) < 1e-9:
            data.qpos[item["qpos_id"]] = -0.01
    mujoco.mj_forward(model, data)
    initial_targets = np.asarray(
        [data.qpos[controlled[name]["qpos_id"]] for name in controlled_names],
        dtype=np.float64,
    )
    upper_lower = np.asarray(
        [model.jnt_range[controlled[name]["joint_id"], 0] for name in upper_names]
    )
    upper_upper = np.asarray(
        [model.jnt_range[controlled[name]["joint_id"], 1] for name in upper_names]
    )
    for index, name in enumerate(upper_names):
        if name == "waist_pitch_joint":
            upper_lower[index], upper_upper[index] = (
                -WAIST_PITCH_LIMIT_RAD,
                WAIST_PITCH_LIMIT_RAD,
            )
        elif name in ("waist_yaw_joint", "waist_roll_joint"):
            upper_lower[index], upper_upper[index] = -0.05, 0.05

    expert_target = load_expert_hold_target(source_episode, controlled_names)
    expert_upper_target = np.clip(
        expert_target[LOWER_BODY_JOINTS:], upper_lower, upper_upper
    )
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_NAMES
    ]
    tote_site_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in TOTE_SITE_NAMES
    ]
    assisted_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
        for name in ("left_assisted_grasp", "right_assisted_grasp")
    ]
    tote_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "warehouse_tote")
    initial_tote_z = float(data.site_xpos[tote_site_ids[0], 2])

    task_renderer = mujoco.Renderer(model, height=240, width=320)
    video_renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.42, 0.0, 0.95)
    camera.distance = 2.25
    camera.azimuth = 145
    camera.elevation = -8
    video_path = output_dir / "policy_to_expert_recovery.mp4"
    video_writer = imageio.get_writer(
        video_path, fps=args.control_fps, codec="libx264", quality=8
    )
    camera_writers = {
        key: imageio.get_writer(
            output_dir / f"{camera_name}.mp4",
            fps=args.control_fps,
            codec="libx264",
            quality=8,
        )
        for key, camera_name in (
            ("observation.images.head", "head_camera"),
            ("observation.images.left_wrist", "left_wrist_camera"),
            ("observation.images.right_wrist", "right_wrist_camera"),
        )
    }

    policy.reset()
    dt = float(model.opt.timestep)
    physics_per_control = max(1, round(1.0 / (args.control_fps * dt)))
    control_frames = round(args.duration_s * args.control_fps)
    current_chunk = None
    scheduler_phase = 0
    hold_latched = False
    hold_enter_time = None
    takeover_time = None
    takeover_upper_action = None
    takeover_lift = None
    takeover_speed = None
    assist_activation_s = None
    phase_transitions = []
    recorded_times = []
    recorded_qpos = []
    recorded_qvel = []
    recorded_actions = []
    recorded_phases = []
    recorded_assist = []
    recorded_lifts = []
    inference_times = []
    bilateral_seen = {"left": False, "right": False}
    final_table_contact = False
    final_joint_violations = 0
    final_joint_samples = 0
    final_non_hand_joint_violations = 0
    final_non_hand_joint_samples = 0
    final_saturated = 0
    final_actuator_samples = 0
    final_non_hand_saturated = 0
    final_non_hand_actuator_samples = 0
    started = time.perf_counter()

    for frame in range(control_frames):
        upper_state = np.asarray(
            [data.qpos[controlled[name]["qpos_id"]] for name in upper_names]
        )
        upper_velocity = np.asarray(
            [data.qvel[controlled[name]["qvel_id"]] for name in upper_names]
        )
        chunk_index = frame % args.replan_steps
        if takeover_time is None and chunk_index == 0:
            distances = [
                np.linalg.norm(data.site_xpos[palm] - data.site_xpos[tote])
                for palm, tote in zip(palm_ids, tote_site_ids, strict=True)
            ]
            if data.time >= 0.5 and scheduler_phase == 0:
                scheduler_phase = 1
            if scheduler_phase in (1, 2):
                if max(distances) <= args.assist_distance_m:
                    scheduler_phase = 3
                elif max(distances) <= args.align_distance_m:
                    scheduler_phase = max(scheduler_phase, 2)
            if assist_activation_s is not None:
                scheduler_phase = max(scheduler_phase, 4)
            current_lift = float(data.site_xpos[tote_site_ids[0], 2]) - initial_tote_z
            if hold_latched or current_lift >= 0.10:
                hold_latched = True
                scheduler_phase = 5
            task = PHASE_INSTRUCTIONS[scheduler_phase]
            if (
                not phase_transitions
                or phase_transitions[-1]["phase"] != scheduler_phase
            ):
                phase_transitions.append(
                    {
                        "frame": frame,
                        "time_s": float(data.time),
                        "phase": scheduler_phase,
                        "task": task,
                    }
                )
            if scheduler_phase == 5 and hold_enter_time is None:
                hold_enter_time = float(data.time)
            batch = render_observation(task_renderer, data, upper_state, task, device)
            inference_started = time.perf_counter()
            current_chunk = (
                policy.predict_action_chunk(
                    batch,
                    noise=seeded_noise(
                        policy,
                        args.seed
                        + seed_variant * 1_000_000
                        + source_index * 10_000
                        + frame,
                        device,
                    ),
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            inference_times.append(time.perf_counter() - inference_started)

        if (
            takeover_time is None
            and hold_enter_time is not None
            and float(data.time) - hold_enter_time >= args.takeover_delay_s
        ):
            takeover_time = float(data.time)
            takeover_upper_action = upper_state.copy()
            mujoco.mj_forward(model, data)
            takeover_lift = float(data.site_xpos[tote_site_ids[0], 2]) - initial_tote_z
            takeover_speed = float(np.linalg.norm(data.cvel[tote_body_id, 3:]))

        if takeover_time is None:
            if current_chunk is None or chunk_index >= len(current_chunk):
                raise RuntimeError("Policy did not produce enough actions")
            action = np.clip(
                current_chunk[chunk_index].astype(np.float64), upper_lower, upper_upper
            )
        else:
            blend = smoothstep(
                (float(data.time) - takeover_time) / args.recovery_blend_s
            )
            action = (1.0 - blend) * takeover_upper_action + blend * expert_upper_target
            full_state = np.asarray(
                [data.qpos[controlled[name]["qpos_id"]] for name in controlled_names],
                dtype=np.float32,
            )
            full_velocity = np.asarray(
                [data.qvel[controlled[name]["qvel_id"]] for name in controlled_names],
                dtype=np.float32,
            )
            full_action = initial_targets.copy()
            full_action[LOWER_BODY_JOINTS:] = action
            recorded_times.append(float(data.time) - takeover_time)
            recorded_qpos.append(full_state)
            recorded_qvel.append(full_velocity)
            recorded_actions.append(full_action.astype(np.float32))
            recorded_phases.append(5)
            recorded_assist.append(1)
            recorded_lifts.append(
                float(data.site_xpos[tote_site_ids[0], 2]) - initial_tote_z
            )
            for key, camera_name in TASK_CAMERAS.items():
                task_renderer.update_scene(data, camera=camera_name)
                camera_writers[key].append_data(task_renderer.render())

        targets = initial_targets.copy()
        targets[LOWER_BODY_JOINTS:] = action
        for _ in range(physics_per_control):
            if assist_activation_s is None:
                distances = [
                    np.linalg.norm(data.site_xpos[palm] - data.site_xpos[tote])
                    for palm, tote in zip(palm_ids, tote_site_ids, strict=True)
                ]
                if scheduler_phase >= 3 and max(distances) <= args.assist_distance_m:
                    for equality_id in assisted_ids:
                        data.eq_active[equality_id] = 1
                    assist_activation_s = float(data.time)
            for target, name in zip(targets, controlled_names, strict=True):
                item = controlled[name]
                kp, kd = unitree_gains(name)
                kp *= 1.5
                kd *= math.sqrt(1.5)
                qpos = float(data.qpos[item["qpos_id"]])
                qvel = float(data.qvel[item["qvel_id"]])
                torque = kp * (target - qpos) - kd * qvel
                torque += float(data.qfrc_bias[item["qvel_id"]])
                data.ctrl[item["actuator_id"]] = torque
            mujoco.mj_step(model, data)
            if float(data.site_xpos[tote_site_ids[0], 2]) - initial_tote_z >= 0.10:
                hold_latched = True

            bilateral = {"left": False, "right": False}
            table_contact = False
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                bodies = {
                    object_name(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(model.geom_bodyid[int(contact.geom1)]),
                    ),
                    object_name(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(model.geom_bodyid[int(contact.geom2)]),
                    ),
                }
                if "warehouse_tote" in bodies:
                    bilateral["left"] |= any(
                        name.startswith("left_hand_") for name in bodies
                    )
                    bilateral["right"] |= any(
                        name.startswith("right_hand_") for name in bodies
                    )
                    table_contact |= "world" in bodies
            for side in bilateral_seen:
                bilateral_seen[side] |= bilateral[side]

            if data.time >= args.duration_s - 0.5:
                final_table_contact |= table_contact
                for name, item in controlled.items():
                    joint_id = item["joint_id"]
                    position = float(data.qpos[item["qpos_id"]])
                    low, high = model.jnt_range[joint_id]
                    final_joint_violations += int(
                        position < low - 1e-6 or position > high + 1e-6
                    )
                    final_joint_samples += 1
                    force_limit = max(
                        abs(float(value)) for value in model.jnt_actfrcrange[joint_id]
                    )
                    final_saturated += int(
                        force_limit > 0
                        and abs(float(data.actuator_force[item["actuator_id"]]))
                        >= 0.98 * force_limit
                    )
                    final_actuator_samples += 1
                    if "hand_" not in name:
                        final_non_hand_joint_violations += int(
                            position < low - 1e-6 or position > high + 1e-6
                        )
                        final_non_hand_joint_samples += 1
                        final_non_hand_saturated += int(
                            force_limit > 0
                            and abs(float(data.actuator_force[item["actuator_id"]]))
                            >= 0.98 * force_limit
                        )
                        final_non_hand_actuator_samples += 1

        video_renderer.update_scene(data, camera=camera)
        video_writer.append_data(video_renderer.render())

    video_writer.close()
    for writer in camera_writers.values():
        writer.close()
    task_renderer.close()
    video_renderer.close()
    mujoco.mj_forward(model, data)
    final_lift = float(data.site_xpos[tote_site_ids[0], 2]) - initial_tote_z
    final_speed = float(np.linalg.norm(data.cvel[tote_body_id, 3:]))
    waist = controlled["waist_pitch_joint"]
    final_abs_waist = abs(float(data.qpos[waist["qpos_id"]]))
    joint_violation_fraction = (
        final_joint_violations / final_joint_samples if final_joint_samples else 1.0
    )
    saturation_fraction = (
        final_saturated / final_actuator_samples if final_actuator_samples else 1.0
    )
    non_hand_joint_violation_fraction = (
        final_non_hand_joint_violations / final_non_hand_joint_samples
        if final_non_hand_joint_samples
        else 1.0
    )
    non_hand_saturation_fraction = (
        final_non_hand_saturated / final_non_hand_actuator_samples
        if final_non_hand_actuator_samples
        else 1.0
    )
    strict_task_passed = bool(
        takeover_time is not None
        and len(recorded_times) >= 30
        and final_lift >= 0.10
        and final_speed < 0.05
        and final_abs_waist <= 0.11
        and all(bilateral_seen.values())
        and not final_table_contact
        and joint_violation_fraction == 0.0
        and saturation_fraction < 0.05
        and assist_activation_s is not None
    )
    # Assisted palm-to-tote constraints can trap thumb joints outside their
    # limits in policy-visited states. Keep this visible, but accept recovery
    # data only when the waist/arms recover safely and the physical task passes.
    passed = bool(
        takeover_time is not None
        and len(recorded_times) >= 30
        and final_lift >= 0.10
        and final_speed < 0.05
        and final_abs_waist <= 0.11
        and all(bilateral_seen.values())
        and not final_table_contact
        and non_hand_joint_violation_fraction == 0.0
        and non_hand_saturation_fraction < 0.05
        and assist_activation_s is not None
    )
    report = {
        "experiment": "G1WH-33-policy-state-expert-recovery-dataset",
        "collection_episode": collection_index,
        "source_episode": source_index,
        "seed_variant": seed_variant,
        "tote_x_m": float(source_episode["tote_x_m"]),
        "policy_checkpoint": str(args.checkpoint),
        "phase_transitions": phase_transitions,
        "hold_enter_time_s": hold_enter_time,
        "expert_takeover_time_s": takeover_time,
        "expert_takeover_delay_s": args.takeover_delay_s,
        "recovery_blend_s": args.recovery_blend_s,
        "recovery_instruction": RECOVERY_INSTRUCTION,
        "recovery_frames": len(recorded_times),
        "takeover_lift_height_m": takeover_lift,
        "takeover_tote_speed_m_s": takeover_speed,
        "final_lift_height_m": final_lift,
        "final_tote_speed_m_s": final_speed,
        "final_abs_waist_pitch_rad": final_abs_waist,
        "bilateral_hand_contact_seen": bilateral_seen,
        "final_table_contact": final_table_contact,
        "final_joint_limit_violation_fraction": joint_violation_fraction,
        "final_actuator_saturation_fraction": saturation_fraction,
        "final_non_hand_joint_limit_violation_fraction": non_hand_joint_violation_fraction,
        "final_non_hand_actuator_saturation_fraction": non_hand_saturation_fraction,
        "assisted_grasp_activated": assist_activation_s is not None,
        "mean_policy_inference_s": (
            float(np.mean(inference_times)) if inference_times else None
        ),
        "wall_time_s": time.perf_counter() - started,
        "video": str(video_path),
        "strict_task_passed": strict_task_passed,
        "recovery_data_accepted": passed,
        "passed": passed,
    }
    (output_dir / "assisted_tote_lift_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if recorded_times:
        np.savez_compressed(
            output_dir / "expert_lift_episode.npz",
            timestamp_s=np.asarray(recorded_times, dtype=np.float32),
            joint_names=np.asarray(controlled_names),
            observation_joint_position_rad=np.asarray(recorded_qpos, dtype=np.float32),
            observation_joint_velocity_rad_s=np.asarray(
                recorded_qvel, dtype=np.float32
            ),
            action_joint_position_rad=np.asarray(recorded_actions, dtype=np.float32),
            task_phase=np.asarray(recorded_phases, dtype=np.int64),
            assisted_grasp_active=np.asarray(recorded_assist, dtype=np.int8),
            tote_lift_height_m=np.asarray(recorded_lifts, dtype=np.float32),
        )
        metadata = {
            "task": "assisted_bimanual_tote_hold_recovery",
            "language_instruction": RECOVERY_INSTRUCTION,
            "episode_success": passed,
            "assisted_grasp_constraint": True,
            "frames": len(recorded_times),
            "fps": args.control_fps,
            "image_width": 320,
            "image_height": 240,
            "joint_count": len(controlled_names),
            "task_cameras": ["head_camera", "left_wrist_camera", "right_wrist_camera"],
            "dataset": str(output_dir / "expert_lift_episode.npz"),
            "source_episode": source_index,
            "seed_variant": seed_variant,
        }
        (output_dir / "episode_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"recovery {collection_index}: source={source_index} seed={seed_variant} "
        f"passed={passed} takeover={takeover_lift} final={final_lift:.3f}m "
        f"speed={final_speed:.3f}m/s",
        flush=True,
    )
    return report


def main() -> None:
    args = parse_args()
    if args.source_episodes < 1 or args.seed_variants < 1:
        raise ValueError("source-episodes and seed-variants must be positive")
    if args.replan_steps < 1 or args.control_fps < 1:
        raise ValueError("replan-steps and control-fps must be positive")
    if args.takeover_delay_s < 0 or args.recovery_blend_s <= 0:
        raise ValueError("takeover delay must be nonnegative and blend time positive")
    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    source = json.loads(args.source_summary.read_text(encoding="utf-8"))
    episodes = source["episodes"][: args.source_episodes]
    if len(episodes) != args.source_episodes or not all(e["success"] for e in episodes):
        raise ValueError("Requested source episodes are missing or unsuccessful")
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)
    expected_upper_names = train_meta.features["action"]["names"]
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    policy = load_policy(args.checkpoint, train_meta, device)
    jobs = [
        (source_episode, seed_variant)
        for seed_variant in range(args.seed_variants)
        for source_episode in episodes
    ]
    if args.episode_limit is not None:
        jobs = jobs[: args.episode_limit]
    reports = [
        collect_episode(
            policy,
            source_episode,
            seed_variant,
            collection_index,
            args,
            expected_upper_names,
            device,
        )
        for collection_index, (source_episode, seed_variant) in enumerate(jobs)
    ]
    successful = [report for report in reports if report["passed"]]
    manifest = args.output_dir / "successful_lift_episodes.jsonl"
    manifest.write_text(
        "".join(json.dumps(report) + "\n" for report in successful),
        encoding="utf-8",
    )
    summary = {
        "experiment": "G1WH-33-policy-state-expert-recovery-dataset",
        "method": "DAgger-style policy rollout followed by expert recovery",
        "source_split": "training episodes only",
        "source_episode_count": args.source_episodes,
        "seed_variants": args.seed_variants,
        "requested_episodes": len(jobs),
        "successful_episodes": len(successful),
        "success_rate": len(successful) / len(jobs),
        "recovery_instruction": RECOVERY_INSTRUCTION,
        "total_recovery_frames": sum(r["recovery_frames"] for r in successful),
        "mean_takeover_lift_height_m": float(
            np.mean(
                [
                    r["takeover_lift_height_m"]
                    for r in reports
                    if r["takeover_lift_height_m"] is not None
                ]
            )
        ),
        "mean_final_lift_height_m": float(
            np.mean([r["final_lift_height_m"] for r in reports])
        ),
        "mean_final_tote_speed_m_s": float(
            np.mean([r["final_tote_speed_m_s"] for r in reports])
        ),
        "episodes": reports,
        "successful_manifest": str(manifest),
        "experiment_passed": len(successful) == len(jobs),
    }
    path = args.output_dir / "assisted_lift_dataset_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {path}")
    if not summary["experiment_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
