#!/usr/bin/env python3
"""Identify each G1 actuator response independently under fixed test conditions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


def joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def gains_for_joint(name: str, profile: str) -> tuple[float, float]:
    if profile == "uniform":
        return (8.0, 0.30) if "hand_" in name else (80.0, 4.0)
    if "hand_" in name:
        return 1.5, 0.2
    if "wrist_" in name:
        return 40.0, 1.5
    weak_body = (
        "ankle_pitch" in name
        or "shoulder_" in name
        or "elbow_" in name
    )
    return (80.0, 3.0) if weak_body else (300.0, 3.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--gain-profile", choices=("uniform", "unitree"), default="uniform")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.asset.resolve()))
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()

    actuator_joint_ids = [int(model.actuator_trnid[index, 0]) for index in range(model.nu)]
    actuator_names = [joint_name(model, joint_id) for joint_id in actuator_joint_ids]
    if model.nu != 43 or len(set(actuator_joint_ids)) != 43:
        raise RuntimeError("Expected 43 unique joint actuators")

    qpos_ids = {name: int(model.jnt_qposadr[joint_id]) for name, joint_id in zip(actuator_names, actuator_joint_ids)}
    qvel_ids = {name: int(model.jnt_dofadr[joint_id]) for name, joint_id in zip(actuator_names, actuator_joint_ids)}
    initial_joint_positions = {name: float(initial_qpos[qpos_ids[name]]) for name in actuator_names}
    dt = float(model.opt.timestep)
    total_steps = int(args.duration / dt)
    hold_start = int((args.duration - 0.25) / dt)
    rows = []

    for selected_index, selected_name in enumerate(actuator_names):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = initial_qpos
        mujoco.mj_forward(model, data)
        base_qpos = data.qpos[:7].copy()

        joint_id = actuator_joint_ids[selected_index]
        low, high = model.jnt_range[joint_id]
        q0 = initial_joint_positions[selected_name]
        positive_room = high - q0
        negative_room = q0 - low
        direction = 1.0 if positive_room >= negative_room else -1.0
        available_room = max(positive_room, negative_room)
        delta = direction * min(0.30, 0.40 * available_room)
        target = q0 + delta

        selected_positions = []
        rise_time_s = None
        max_progress = 0.0
        for step in range(total_steps):
            time_s = step * dt
            selected_scale = smoothstep(time_s / 0.30)
            targets = dict(initial_joint_positions)
            targets[selected_name] = q0 + delta * selected_scale

            for actuator_id, name in enumerate(actuator_names):
                current = data.qpos[qpos_ids[name]]
                velocity = data.qvel[qvel_ids[name]]
                kp, kd = gains_for_joint(name, args.gain_profile)
                data.ctrl[actuator_id] = kp * (targets[name] - current) - kd * velocity

            mujoco.mj_step(model, data)
            data.qpos[:7] = base_qpos
            data.qvel[:6] = 0.0
            mujoco.mj_forward(model, data)
            current_position = float(data.qpos[qpos_ids[selected_name]])
            selected_positions.append(current_position)
            progress = (current_position - q0) / delta
            max_progress = max(max_progress, progress)
            if rise_time_s is None and progress >= 0.90:
                rise_time_s = time_s

        hold_positions = np.asarray(selected_positions[hold_start:])
        steady_error = float(np.sqrt(np.mean(np.square(hold_positions - target))))
        final_position = float(selected_positions[-1])
        final_progress = (final_position - q0) / delta
        leakage = max(
            abs(float(data.qpos[qpos_ids[name]]) - initial_joint_positions[name])
            for name in actuator_names
            if name != selected_name
        )
        overshoot = max(0.0, max_progress - 1.0)
        direction_correct = final_progress > 0.0
        passed = direction_correct and steady_error < 0.10 and leakage < 0.10
        rows.append(
            {
                "joint": selected_name,
                "group": "hand" if "hand_" in selected_name else "body",
                "target_delta_rad": delta,
                "final_progress_ratio": final_progress,
                "steady_rmse_rad": steady_error,
                "rise_time_s": rise_time_s if rise_time_s is not None else "",
                "overshoot_ratio": overshoot,
                "max_other_joint_leakage_rad": leakage,
                "direction_correct": direction_correct,
                "passed": passed,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "joint_response_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sorted_rows = sorted(rows, key=lambda item: float(item["steady_rmse_rad"]), reverse=True)
    figure, axis = plt.subplots(figsize=(12, 9))
    labels = [item["joint"].replace("_joint", "") for item in sorted_rows]
    values = [float(item["steady_rmse_rad"]) for item in sorted_rows]
    colors = ["#c23b33" if value >= 0.10 else "#2f7d4a" for value in values]
    axis.barh(range(len(labels)), values, color=colors)
    axis.set_yticks(range(len(labels)), labels=labels, fontsize=7)
    axis.invert_yaxis()
    axis.axvline(0.10, color="#202020", linestyle="--", linewidth=1)
    axis.set_xlabel("Steady-state RMSE (rad), lower is better")
    experiment_name = (
        "G1WH-05-isolated-joint-response"
        if args.gain_profile == "uniform"
        else "G1WH-06-unitree-gain-profile"
    )
    axis.set_title(experiment_name)
    figure.tight_layout()
    chart_path = args.output_dir / "joint_response_errors.png"
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)

    passed_rows = [item for item in rows if item["passed"]]
    body_errors = [float(item["steady_rmse_rad"]) for item in rows if item["group"] == "body"]
    hand_errors = [float(item["steady_rmse_rad"]) for item in rows if item["group"] == "hand"]
    report = {
        "experiment": experiment_name,
        "gain_profile": args.gain_profile,
        "controller": (
            {"body_kp": 80.0, "body_kd": 4.0, "hand_kp": 8.0, "hand_kd": 0.30}
            if args.gain_profile == "uniform"
            else {
                "strong_body_kp_kd": [300.0, 3.0],
                "weak_body_kp_kd": [80.0, 3.0],
                "wrist_kp_kd": [40.0, 1.5],
                "hand_kp_kd": [1.5, 0.2],
            }
        ),
        "pass_rule": "Correct direction, steady RMSE < 0.10 rad, leakage < 0.10 rad",
        "tested_joints": len(rows),
        "passed_joints": len(passed_rows),
        "body_mean_steady_rmse_rad": float(np.mean(body_errors)),
        "hand_mean_steady_rmse_rad": float(np.mean(hand_errors)),
        "worst_five": sorted_rows[:5],
        "metrics_csv": str(csv_path),
        "error_chart": str(chart_path),
    }
    report_path = args.output_dir / "joint_response_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
