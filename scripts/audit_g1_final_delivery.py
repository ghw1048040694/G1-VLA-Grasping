#!/usr/bin/env python3
"""Create a machine-readable final evidence audit for the G1 project."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summary_gate(path: Path) -> dict:
    report = load_json(path)
    return {
        "path": str(path),
        "episodes": report.get("episodes"),
        "router_accuracy": report.get("router_accuracy"),
        "object_selection_accuracy": report.get("object_selection_accuracy"),
        "strict_success_rate": report.get("strict_success_rate"),
        "wrong_object_grasp_rate": report.get("wrong_object_grasp_rate"),
        "passed": bool(report.get("passed", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/G1FINAL-04_project_audit.json",
    )
    args = parser.parse_args()

    specialist = load_json(ROOT / "outputs/G1FINAL-01_target_specialists_audit.json")
    world_model = load_json(
        ROOT / "outputs/G1FINAL-02_language_world_model_10000step/summary.json"
    )
    sim2sim = load_json(Path("/mnt/d/G1-UpperBody-Sim2Sim/G1SIM-02/isaacsim_report.json"))
    baseline = summary_gate(ROOT / "outputs/G1FINAL-01_target_router_smoke_3ep/summary.json")
    mpc = summary_gate(ROOT / "outputs/G1FINAL-03_target_router_mpc_smoke_3ep/summary.json")
    video_audit = load_json(ROOT / "outputs/G1FINAL-05_video_audit.json")

    strict_source_available = bool(
        baseline["strict_success_rate"] is not None
        and baseline["strict_success_rate"] > 0.0
    )
    report = {
        "experiment": "G1FINAL-04-project-delivery-audit",
        "audit_version": 1,
        "python": sys.executable,
        "platform": platform.platform(),
        "delivery_status": "final-evidence-package-ready",
        "end_to_end_language_success": False,
        "training": {
            "specialist_checkpoint_audit_passed": bool(specialist.get("audit_passed")),
            "world_model_experiment_passed": bool(world_model.get("experiment_passed")),
            "world_model_best_step": world_model.get("best_step"),
            "world_model_screening_checks": world_model.get("screening_checks", {}),
        },
        "closed_loop": {
            "baseline": baseline,
            "world_model_mpc": mpc,
            "video_trajectory_audit": {
                "path": "outputs/G1FINAL-05_video_audit.json",
                "passed": bool(video_audit.get("video_audit_passed")),
                "episodes": [
                    {
                        "episode": item["episode"],
                        "target_object": item["target_object"],
                        "grabbed_objects": item["grabbed_objects"],
                        "selected_correct_object": item["selected_correct_object"],
                        "task_success": item["task_success"],
                        "action_observation_rmse_rad": item[
                            "action_observation_rmse_rad"
                        ],
                    }
                    for item in video_audit["episodes"]
                ],
            },
        },
        "sim2sim": {
            "g1sim_02_replay_completed": bool(sim2sim.get("replay_completed")),
            "g1sim_02_joint_contract_passed": bool(sim2sim.get("joint_contract_passed")),
            "g1sim_02_task_transfer_passed": bool(sim2sim.get("task_transfer_passed")),
            "g1sim_02_final_lift_m": sim2sim.get("tote_final_lift_height_m"),
            "g1sim_02_max_lift_m": sim2sim.get("tote_maximum_lift_height_m"),
            "g1sim_03_source_episode_available": strict_source_available,
            "g1sim_03_status": (
                "ready_for_replay" if strict_source_available else "blocked_no_strict_success_source"
            ),
        },
        "stability_incidents": [
            {
                "artifact": "outputs/G1FINAL-01_yellow_fp32_freezestate100_retry_smoke.log",
                "status": "failed_cuda_illegal_memory_access",
            },
            {
                "artifact": "outputs/G1FINAL-01_target_router_baseline_30ep.log",
                "status": "failed_mujoco_2_3_schema_import",
            },
            {
                "artifact": "outputs/G1FINAL-01_target_router_baseline_30ep_v2.log",
                "status": "failed_cublas_after_three_saved_episodes",
            },
            {
                "artifact": "outputs/G1FINAL-01_target_router_postreboot_3ep.log",
                "status": "failed_cuda_and_initial_simulation_nan",
            },
            {
                "artifact": "outputs/G1FINAL-01_target_router_postreboot_fp32_3ep.log",
                "status": "failed_cublas_fp32_image_encoder",
            },
            {
                "artifact": "outputs/G1FINAL-01_yellow_clean100_recovery_smoke.log",
                "status": "failed_bf16_vlm_attention_after_zero_finite_updates",
            },
            {
                "artifact": "outputs/G1FINAL-01_yellow_fullfp32_20_smoke.log",
                "status": "one_finite_update_then_cublas_backward_failure",
            },
        ],
        "architecture_claim": (
            "language classifier -> target specialist -> World Model/MPC -> Sim2Sim; "
            "the SmolVLA end-to-end language gate is not passed"
        ),
        "required_follow_up": [
            "Reset the WSL2/DXG GPU context before any further SmolVLA inference.",
            "Do not prepare G1SIM-03 until a strict-success language episode exists.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
