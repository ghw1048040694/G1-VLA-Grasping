#!/usr/bin/env python3
"""Run learned language routing followed by the verified classical G1 controller."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

if lerobot_site_packages := os.environ.get("LEROBOT_SITE_PACKAGES"):
    sys.path.append(lerobot_site_packages)

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

import evaluate_g1_language_pick_place as policy_eval
import run_g1_language_pick_place as classical
from evaluate_g1_language_smolvla_heldout import load_policy as load_router
from validate_g1_bimanual_actuation import apply_regularized_dynamics


LOWER_BODY_JOINTS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument(
        "--train-repo-id", default="local/g1_language_paired_train"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--position-jitter-m", type=float, default=0.018)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    return parser.parse_args()


def scene_positions(spec: dict) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(
            (
                *(
                    classical.OBJECT_SLOTS_XY_M[spec["permutation"][index]]
                    + spec["offsets"][index]
                ),
                classical.OBJECT_SPAWN_Z_M[name],
            )
        )
        for index, name in enumerate(classical.OBJECT_NAMES)
    }


def initial_router_batch(
    asset: Path,
    scene_path: Path,
    positions: dict[str, np.ndarray],
    instruction: str,
    expected_upper_names: list[str],
    device: str,
) -> dict:
    classical.build_scene(asset.resolve(), scene_path, positions)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = classical.controlled_joints(model)
    controlled_names = list(controlled)
    upper_names = controlled_names[LOWER_BODY_JOINTS:]
    if upper_names != expected_upper_names:
        raise ValueError("Simulation and dataset joint ordering differ")

    _, selected_qpos_ids, lower, upper = classical.arm_ik_contract(model)
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in classical.PALM_NAMES
    ]
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    home_pose, home_report = classical.solve_pose(
        model,
        data,
        palm_ids,
        selected_qpos_ids,
        lower,
        upper,
        classical.HOME_POSITIONS_M,
        np.zeros(len(selected_qpos_ids)),
        orientation_weight=0.06,
    )
    if not home_report["solver_success"]:
        raise RuntimeError(f"Failed to solve routing home pose: {home_report}")

    mujoco.mj_resetData(model, data)
    data.qpos[selected_qpos_ids] = home_pose
    for name, item in controlled.items():
        if "hand_" not in name:
            continue
        low_limit, high_limit = model.jnt_range[item["joint_id"]]
        if abs(float(low_limit)) < 1e-9:
            data.qpos[item["qpos_id"]] = 0.01
        elif abs(float(high_limit)) < 1e-9:
            data.qpos[item["qpos_id"]] = -0.01
    mujoco.mj_forward(model, data)
    state = np.asarray(
        [data.qpos[controlled[name]["qpos_id"]] for name in upper_names],
        dtype=np.float32,
    )
    renderer = mujoco.Renderer(model, height=240, width=320)
    try:
        return policy_eval.render_observation(
            renderer, data, state, instruction, device
        )
    finally:
        renderer.close()


@torch.no_grad()
def route_target(router, batch: dict, seed: int, device: str) -> tuple[str, list[float]]:
    router.reset()
    router.predict_action_chunk(
        batch, noise=policy_eval.seeded_noise(router, seed, device)
    )
    logits = getattr(router.model, "_language_action_target_logits", None)
    if logits is None:
        raise RuntimeError("Router did not expose its target posterior")
    values = logits.detach().float().cpu()[0]
    target = classical.OBJECT_NAMES[int(torch.argmax(values).item())]
    return target, values.tolist()


def posterior_metrics(logits: list[list[float]]) -> dict[str, float]:
    """Summarize routing confidence without changing the routing decision."""
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("Router logits must be a 2-D array with at least two classes")
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    ordered = np.sort(scores, axis=1)
    return {
        "mean_router_top1_probability": float(probabilities.max(axis=1).mean()),
        "min_router_top1_probability": float(probabilities.max(axis=1).min()),
        "mean_router_logit_margin": float((ordered[:, -1] - ordered[:, -2]).mean()),
        "min_router_logit_margin": float((ordered[:, -1] - ordered[:, -2]).min()),
    }


def controller_args(
    args: argparse.Namespace, spec: dict, output_dir: Path, routed_target: str
) -> argparse.Namespace:
    offsets = ",".join(
        f"{float(value):.9g}" for pair in spec["offsets"] for value in pair
    )
    permutation = ",".join(str(value) for value in spec["permutation"])
    return argparse.Namespace(
        asset=args.asset,
        output_dir=output_dir,
        target_object=routed_target,
        slot_permutation=permutation,
        slot_offsets=offsets,
        language_instruction=spec["instruction"],
        record_demonstration=True,
        video_fps=args.video_fps,
        render_width=args.render_width,
        render_height=args.render_height,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)
    expected_upper_names = train_meta.features["action"]["names"]
    router = load_router(args.router_checkpoint, train_meta, args.device)
    specs = policy_eval.evaluation_specs(
        args.episodes, args.seed, args.position_jitter_m
    )
    reports = []

    for spec in specs:
        episode_dir = args.output_dir / f"episode_{spec['episode_index']:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        positions = scene_positions(spec)
        batch = initial_router_batch(
            args.asset,
            episode_dir / "router_scene.xml",
            positions,
            spec["instruction"],
            expected_upper_names,
            args.device,
        )
        routed_target, logits = route_target(
            router,
            batch,
            args.seed + spec["scene_index"] * 1000,
            args.device,
        )
        execution = classical.run_episode(
            controller_args(args, spec, episode_dir, routed_target)
        )
        route_correct = routed_target == spec["target_object"]
        hybrid_success = bool(route_correct and execution["passed"])
        report = {
            "episode_index": spec["episode_index"],
            "scene_index": spec["scene_index"],
            "language_instruction": spec["instruction"],
            "expected_target": spec["target_object"],
            "routed_target": routed_target,
            "router_logits": logits,
            "router_correct": route_correct,
            "controller": "classical_ik_interpolation_assisted_grasp",
            "controller_passed": execution["passed"],
            "hybrid_success": hybrid_success,
            "target_in_box": execution["target_in_box"],
            "wrong_objects_in_box": execution["wrong_objects_in_box"],
            "joint_limit_violation_fraction": execution[
                "joint_limit_violation_fraction"
            ],
            "video": execution["video"],
            "execution_summary": str(
                episode_dir / "language_pick_place_summary.json"
            ),
        }
        (episode_dir / "hybrid_summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(report)
        print(
            f"HYBRID_EPISODE={spec['episode_index'] + 1}/{args.episodes} "
            f"expected={spec['target_object']} routed={routed_target} "
            f"passed={hybrid_success}",
            flush=True,
        )

    summary = {
        "experiment": "G1FINAL-07-learned-router-classical-controller",
        "claim_boundary": (
            "Learned language target routing followed by a privileged classical "
            "IK/interpolation controller; this is not end-to-end learned VLA control."
        ),
        "router_checkpoint": str(args.router_checkpoint),
        "controller": "classical_ik_interpolation_assisted_grasp",
        "episodes": len(reports),
        "router_accuracy": sum(item["router_correct"] for item in reports)
        / len(reports),
        "controller_success_rate": sum(
            item["controller_passed"] for item in reports
        )
        / len(reports),
        "hybrid_success_rate": sum(item["hybrid_success"] for item in reports)
        / len(reports),
        "mean_joint_limit_violation_fraction": float(
            np.mean(
                [item["joint_limit_violation_fraction"] for item in reports]
            )
        ),
        **posterior_metrics([item["router_logits"] for item in reports]),
        "reports": reports,
    }
    summary["passed"] = bool(
        summary["router_accuracy"] >= 0.90
        and summary["hybrid_success_rate"] >= 0.90
        and summary["mean_joint_limit_violation_fraction"] <= 0.01
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "reports"}, indent=2))
    print(f"Saved {summary_path}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
