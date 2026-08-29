#!/usr/bin/env python3
"""Refit the RGB centroid calibration on randomized G1 evaluation layouts.

The object positions are used only as offline targets.  The saved coefficients
take exactly the nine RGB centroid/area features available to the deployed
policy and produce its camera-localized object positions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from build_g1_phase_candidate_onpolicy_cache import color_features
from run_g1_language_pick_place import (
    HOME_POSITIONS_M,
    OBJECT_NAMES,
    PALM_NAMES,
    arm_ik_contract,
    controlled_joints,
    solve_pose,
)
from validate_g1_bimanual_actuation import apply_regularized_dynamics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-max-frame", type=int, default=20)
    parser.add_argument("--initial-weight", type=float, default=50.0)
    return parser.parse_args()


def design(features: np.ndarray, object_index: int) -> np.ndarray:
    x, y, area = features[:, object_index * 3 : object_index * 3 + 3].T
    return np.column_stack((np.ones(len(features)), x, y, area, x * x, y * y, x * y))


def fit_coefficients(
    base_features: np.ndarray,
    base_positions: np.ndarray,
    initial_features: np.ndarray,
    initial_positions: np.ndarray,
    initial_weight: float,
) -> np.ndarray:
    coefficients = np.zeros((3, 7, 3), dtype=np.float32)
    for index in range(3):
        matrix = np.concatenate((design(base_features, index), design(initial_features, index)))
        target = np.concatenate((base_positions[:, index], initial_positions[:, index]))
        weights = np.concatenate(
            (np.ones(len(base_features)), np.full(len(initial_features), initial_weight))
        )
        weighted = np.sqrt(weights)[:, None]
        coefficients[index] = np.linalg.lstsq(matrix * weighted, target * weighted, rcond=None)[0]
    return coefficients


def predict(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    values = np.stack(
        [design(features, index) @ coefficients[index] for index in range(3)], axis=1
    )
    values[..., 0] = np.clip(values[..., 0], 0.28, 0.68)
    values[..., 1] = np.clip(values[..., 1], -0.42, 0.28)
    values[..., 2] = np.clip(values[..., 2], 0.73, 1.05)
    return values


def errors(predicted: np.ndarray, expected: np.ndarray) -> dict:
    value = np.linalg.norm(predicted - expected, axis=-1)
    return {
        "mean_error_m": float(value.mean()),
        "p95_error_m": float(np.quantile(value, 0.95)),
        "max_error_m": float(value.max()),
        "mean_error_by_object_m": value.mean(0).tolist(),
    }


def initial_rgb_observation(episode_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(episode_dir / "scene.xml"))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    names, qpos_ids, lower, upper = arm_ik_contract(model)
    palm_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_NAMES]
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    home_pose, report = solve_pose(
        model,
        data,
        palm_ids,
        qpos_ids,
        lower,
        upper,
        HOME_POSITIONS_M,
        np.zeros(len(names)),
        orientation_weight=0.06,
    )
    if not report["solver_success"]:
        raise RuntimeError(f"Could not recreate the evaluation home pose for {episode_dir}")
    mujoco.mj_resetData(model, data)
    data.qpos[qpos_ids] = home_pose
    for name, item in controlled.items():
        if "hand_" in name:
            low, high = model.jnt_range[item["joint_id"]]
            data.qpos[item["qpos_id"]] = 0.01 if abs(float(low)) < 1e-9 else -0.01 if abs(float(high)) < 1e-9 else 0.5 * (low + high)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=120, width=160)
    renderer.update_scene(data, camera="head_camera")
    feature = color_features(renderer.render().copy()).numpy()
    close = getattr(renderer, "close", None)
    if close is not None:
        close()
    positions = np.stack(
        [data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy() for name in OBJECT_NAMES]
    )
    return feature, positions


def main() -> None:
    args = parse_args()
    if args.base_max_frame < 1 or args.initial_weight <= 0.0:
        raise ValueError("base max frame and initial weight must be positive")
    base = torch.load(args.base_cache, map_location="cpu", weights_only=False)
    keep = base["frame_index"].numpy() < args.base_max_frame
    base_features = base["color_features"][keep].float().numpy()
    base_positions = base["object_position"][keep].float().numpy().reshape(-1, 3, 3)
    del base
    pairs = [initial_rgb_observation(path) for path in sorted(args.rollout_root.glob("episode_*"))]
    if len(pairs) < 6:
        raise ValueError("At least six randomized initial observations are required")
    initial_features = np.stack([item[0] for item in pairs])
    initial_positions = np.stack([item[1] for item in pairs])
    coefficients = fit_coefficients(
        base_features,
        base_positions,
        initial_features,
        initial_positions,
        args.initial_weight,
    )
    predicted = predict(initial_features, coefficients)
    loo = []
    for holdout in range(len(initial_features)):
        keep = np.arange(len(initial_features)) != holdout
        fold = fit_coefficients(
            base_features,
            base_positions,
            initial_features[keep],
            initial_positions[keep],
            args.initial_weight,
        )
        loo.append(predict(initial_features[holdout : holdout + 1], fold)[0])
    summary = {
        "experiment": "G1LANG-randomized-initial-rgb-camera-calibration",
        "base_cache": str(args.base_cache),
        "rollout_root": str(args.rollout_root),
        "base_samples": int(len(base_features)),
        "randomized_initial_samples": int(len(initial_features)),
        "randomized_initial_weight": float(args.initial_weight),
        "input_contract": "head-camera RGB color centroid/area features",
        "privileged_positions_at_inference": False,
        "fit": errors(predicted, initial_positions),
        "leave_one_episode_out": errors(np.stack(loo), initial_positions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, coefficients=coefficients)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
