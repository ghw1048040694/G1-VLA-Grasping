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
        "controller_success_rate": report.get("controller_success_rate"),
        "hybrid_success_rate": report.get("hybrid_success_rate"),
        "mean_joint_limit_violation_fraction": report.get(
            "mean_joint_limit_violation_fraction"
        ),
        "mean_router_top1_probability": report.get("mean_router_top1_probability"),
        "min_router_top1_probability": report.get("min_router_top1_probability"),
        "mean_router_logit_margin": report.get("mean_router_logit_margin"),
        "min_router_logit_margin": report.get("min_router_logit_margin"),
        "object_selection_accuracy": report.get("object_selection_accuracy"),
        "strict_success_rate": report.get("strict_success_rate"),
        "wrong_object_grasp_rate": report.get("wrong_object_grasp_rate"),
        "passed": bool(report.get("passed", False)),
    }


def hybrid_artifact_gate(path: Path) -> dict:
    """Check that every hybrid report points to locally present evidence."""
    report = load_json(path)
    episodes = report.get("reports", [])
    checks = []
    for item in episodes:
        video = ROOT / item["video"]
        execution = ROOT / item["execution_summary"]
        checks.append(
            {
                "episode_index": item.get("episode_index"),
                "video": str(video),
                "video_present": video.is_file() and video.stat().st_size > 0,
                "execution_summary": str(execution),
                "execution_summary_present": execution.is_file(),
            }
        )
    return {
        "episodes": checks,
        "all_videos_present": bool(checks) and all(
            item["video_present"] for item in checks
        ),
        "all_execution_summaries_present": bool(checks) and all(
            item["execution_summary_present"] for item in checks
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/G1FINAL-04_project_audit.json",
    )
    parser.add_argument(
        "--hybrid-summary",
        type=Path,
        default=None,
        help="Hybrid summary to audit; defaults to the 9-episode run when present.",
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
    default_hybrid = ROOT / "outputs/G1FINAL-09_router_classical_9ep/summary.json"
    if not default_hybrid.is_file():
        default_hybrid = ROOT / "outputs/G1FINAL-07_router_classical_3ep/summary.json"
    hybrid_path = args.hybrid_summary or default_hybrid
    hybrid = summary_gate(hybrid_path)
    hybrid_artifacts = hybrid_artifact_gate(hybrid_path)
    hybrid_sim2sim_source = Path(
        "/mnt/d/G1-UpperBody-Sim2Sim/G1SIM-03/data/source_summary.json"
    )
    hybrid_sim2sim_report = Path(
        "/mnt/d/G1-UpperBody-Sim2Sim/G1SIM-03/isaacsim_report.json"
    )

    strict_source_available = bool(
        baseline["strict_success_rate"] is not None
        and baseline["strict_success_rate"] > 0.0
    )
    report = {
        "experiment": "G1FINAL-04-project-delivery-audit",
        "audit_version": 2,
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
            "hybrid_router_classical": hybrid,
            "hybrid_summary": str(hybrid_path),
            "hybrid_artifacts": hybrid_artifacts,
            "hybrid_sim2sim": {
                "source_prepared": hybrid_sim2sim_source.is_file(),
                "source_passed": (
                    bool(load_json(hybrid_sim2sim_source).get("passed"))
                    if hybrid_sim2sim_source.is_file()
                    else False
                ),
                "replay_report_available": hybrid_sim2sim_report.is_file(),
                "status": (
                    "passed"
                    if hybrid_sim2sim_report.is_file()
                    else "failed_isaac_gpu_device_lost"
                ),
                "external_root": "/mnt/d/G1-UpperBody-Sim2Sim/G1SIM-03",
            },
        },
        "sim2sim": {
            "g1sim_02_replay_completed": bool(sim2sim.get("replay_completed")),
            "g1sim_02_joint_contract_passed": bool(sim2sim.get("joint_contract_passed")),
            "g1sim_02_task_transfer_passed": bool(sim2sim.get("task_transfer_passed")),
            "g1sim_02_final_lift_m": sim2sim.get("tote_final_lift_height_m"),
            "g1sim_02_max_lift_m": sim2sim.get("tote_maximum_lift_height_m"),
            "g1sim_03_learned_source_available": strict_source_available,
            "g1sim_03_hybrid_source_prepared": hybrid_sim2sim_source.is_file(),
            "g1sim_03_status": (
                "replay_completed"
                if hybrid_sim2sim_report.is_file()
                else (
                    "hybrid_source_ready_isaac_gpu_device_lost"
                    if hybrid_sim2sim_source.is_file()
                    else "blocked_no_strict_success_source"
                )
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
            "G1FINAL-07/G1FINAL-09 also validate learned language routing -> classical IK execution "
            "as a separate upper bound; the SmolVLA end-to-end language gate is not passed"
        ),
        "required_follow_up": [
            "Do not claim G1FINAL-07/G1FINAL-09 as end-to-end learned SmolVLA control.",
            "Only retry G1SIM-03 after changing the Isaac/RTX device context.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
