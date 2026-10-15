"""One validated configuration and explicit stage for G1 rebuilding."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import sys
from pathlib import Path

from .artifacts import BASELINE_EXPERIMENTS, inventory_artifacts, sha256_file, write_manifest
from .config import ConfigError, load_config
from .contracts import ActionContract, ContractError, canonical_sha256, read_joint_order
from .runtime.experiment import ExperimentRun, canonical_hash, write_new_json

STAGES = ("plan", "inventory", "registry", "audit", "limits", "pairing", "teacher-smoke", "teacher", "external-fetch", "external-audit", "external-reconcile", "weights-audit", "offline-fetch", "offline-infer", "source-native-fetch", "grasp-memory", "diagnose", "act-diagnose", "act-ablate", "vision-fetch", "vision-ablate", "coverage-audit", "residual-check", "residual-train", "report", "collect", "build", "replay", "train", "eval")


def readiness(config):
    root = Path(config["paths"]["root"])
    action = config["action"]
    checks, blockers = {}, []
    metadata_path = Path(action["joint_metadata_path"])
    try:
        names = read_joint_order(json.loads(metadata_path.read_text(encoding="utf-8")))
        digest = canonical_sha256(list(names))
        checks["joint_order"] = {"names": list(names), "sha256": digest, "matches_config": digest == action["joint_order_sha256"]}
        if not checks["joint_order"]["matches_config"]:
            blockers.append("Source joint ordering differs from the frozen configuration")
    except (OSError, ValueError, ContractError) as exc:
        checks["joint_order"] = {"error": str(exc)}
        blockers.append("Source joint metadata must be repaired")
    asset = Path(config["scene"]["asset_path"]) if config["scene"]["asset_path"] else None
    checks["asset"] = {"path": str(asset) if asset else None, "present": bool(asset and asset.is_file())}
    if asset and asset.is_file():
        checks["asset"]["sha256"] = sha256_file(asset)
    else:
        blockers.append("Scene asset is missing")
    try:
        contract = ActionContract.from_config(config)
        checks["action_contract"] = {"ready": True, "sha256": contract.contract_sha256, "metadata": contract.metadata()}
    except (ValueError, ContractError) as exc:
        checks["action_contract"] = {"ready": False, "reason": str(exc)}
        blockers.append(str(exc))
    versions = {}
    for package in ("numpy", "torch", "mujoco", "scipy", "imageio"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    checks["installed_versions"] = versions
    checks["cadence"] = {key: config["control"][key] for key in ("nominal_control_hz", "control_hz", "physics_dt_s", "physics_substeps")}
    checks["registry"] = {"path": config["data"]["registry_path"], "present": Path(config["data"]["registry_path"]).is_file()}
    return {"schema_version": 1, "project_root": str(root), "configuration_valid": True,
            "training_enabled": config["train"]["enabled"], "training_authorization": config["train"]["authorization"],
            "active_training_backend": config["train"]["backend"],
            "wm_enabled": False, "training_started": False,
            "ready_for_inventory": True, "ready_for_execution": not blockers,
            "checks": checks, "execution_blockers": blockers,
            "pipeline_status": "Audits, physical teacher and original-protocol offline inference are admitted. Conditional training requires dataset and execution admission; there is no automatic stage chaining."}


def _protection(inventory):
    return {"schema_version": 1, "read_only": True,
            "records": [row for row in inventory["records"] if row.get("protected")],
            "inventory_summary": inventory["summary"]}


def execute_stage(stage, config, run):
    report = readiness(config)
    if stage == "coverage-audit":
        from .trajectory_coverage import run_trajectory_coverage
        result = run_trajectory_coverage(config, run.path)
        report.update(stage=stage, trajectory_coverage=result)
        return report, result.get("passed") is True
    if stage in ("residual-check", "residual-train"):
        from .state_residual import run_residual_admission, train_state_residual
        result = (run_residual_admission if stage == "residual-check" else train_state_residual)(config, run.path)
        report.update(stage=stage, state_residual=result,
                      training_started=result.get("training_started", False),
                      optimizer_steps=result.get("optimizer_steps", 0))
        return report, result.get("passed") is True
    if stage in ("vision-fetch", "vision-ablate"):
        from .vision_pretraining import fetch_visual_backbone, train_visual_pretraining
        result = (fetch_visual_backbone if stage == "vision-fetch" else train_visual_pretraining)(config, run.path)
        report.update(stage=stage, visual_pretraining=result,
                      training_started=result.get("training_started", False),
                      optimizer_steps=result.get("optimizer_steps", 0))
        return report, result.get("passed") is True
    if stage == "diagnose":
        from .native_diagnostics import run_native_diagnostics
        diagnosed = run_native_diagnostics(config, run.path)
        report.update(stage=stage, native_diagnostics=diagnosed)
        return report, diagnosed.get("passed") is True
    if stage in ("act-diagnose", "act-ablate"):
        from .act_diagnostics import run_act_diagnostics, train_act_ablation
        diagnosed = (run_act_diagnostics if stage == "act-diagnose" else train_act_ablation)(config, run.path)
        report.update(stage=stage, act_diagnostics=diagnosed,
                      training_started=diagnosed.get("training_started", False),
                      optimizer_steps=diagnosed.get("optimizer_steps", 0))
        return report, diagnosed.get("passed") is True
    if stage == "source-native-fetch":
        from .external_sources import fetch_source_native_videos
        fetched = fetch_source_native_videos(config, run.path)
        report.update(stage=stage, source_native_fetch=fetched)
        return report, fetched["status"] == "complete_verified_source_videos"
    if config["train"]["backend"] == "grasp_square_act" and stage == "build":
        from .grasp_data import build_grasp_square_data
        built = build_grasp_square_data(config, run.path)
        report.update(stage=stage, source_native_build=built)
        return report, built.get("passed") is True
    if stage == "grasp-memory":
        from .grasp_act import run_grasp_act_memory
        measured = run_grasp_act_memory(config, run.path)
        report.update(stage=stage, grasp_act_memory=measured)
        return report, measured.get("memory_passed") is True
    if config["train"]["backend"] == "grasp_square_act" and stage in ("train", "eval"):
        from .grasp_act import train_grasp_act, eval_grasp_act
        result = (train_grasp_act if stage == "train" else eval_grasp_act)(config, run.path)
        report.update(stage=stage, source_native_act=result,
                      training_started=result.get("training_started", False),
                      optimizer_steps=result.get("optimizer_steps", 0))
        return report, result.get("training_passed" if stage == "train" else "evaluation_passed") is True
    if stage in ("external-fetch", "offline-fetch"):
        from .external_sources import fetch_official_sample, fetch_native_policy
        fetched = (fetch_official_sample if stage == "external-fetch" else fetch_native_policy)(config, run.path)
        report.update(stage=stage, source_fetch=fetched)
        return report, True
    if stage == "external-audit":
        from .external_data import audit_official_sample
        audited = audit_official_sample(config, run.path)
        report.update(stage=stage, external_data_audit=audited)
        cameras_complete = (not config["external_data"]["probe_videos"] or
            audited["bounded_video_probes_passed"] == len(config["external_data"]["episode_ids"]) * 4)
        return report, audited["status"] == "passed" and cameras_complete
    if stage == "external-reconcile":
        from .external_data import reconcile_official_audits
        reconciled = reconcile_official_audits(config, run.path)
        report.update(stage=stage, external_reconciliation=reconciled)
        return report, reconciled.get("status") == "passed"
    if stage == "weights-audit":
        from .weight_audit import audit_public_weights
        import torch
        import platform
        free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        hardware = {"gpu_total_mib": total / 2**20, "gpu_free_mib": free / 2**20,
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
        weights = audit_public_weights(config["paths"]["root"], config, hardware=hardware)
        write_new_json(run.path / "reports" / "weight_audit.json", weights)
        report.update(stage=stage, weight_audit=weights)
        return report, weights["status"] == "metadata_audited"
    if stage == "offline-infer":
        from .offline_inference import run_native_act_inference
        inferred = run_native_act_inference(config, run.path)
        report.update(stage=stage, offline_inference=inferred)
        return report, inferred.get("inference_passed") is True
    if stage == "limits":
        from .control_limits import audit_control_limits
        limits = audit_control_limits(config["paths"]["root"], config)
        path = run.path / "reports" / "control_limits.json"
        write_new_json(path, limits)
        report.update(stage=stage, control_limits=limits, control_limits_path=str(path))
        return report, limits["status"] == "passed"
    if stage == "pairing":
        from .data_pairing import audit_raw_pairing
        pairing = audit_raw_pairing(config["paths"]["root"], config, run_dir=run.path / "reports")
        report.update(stage=stage, raw_pairing_summary=pairing["summary"],
                      raw_pairing_path=str(run.path / "reports" / "raw_pairing.json"),
                      errors=pairing["errors"], split_leakage=pairing["split_leakage"])
        return report, not pairing["errors"] and not pairing["split_leakage"]
    if stage in ("teacher-smoke", "teacher"):
        from .teacher_acceptance import run_teacher_acceptance
        acceptance = run_teacher_acceptance(config, run.path, smoke=stage == "teacher-smoke")
        report.update(stage=stage, teacher_acceptance=acceptance)
        return report, bool(acceptance["passed"])
    if stage == "inventory":
        scan = list(config["data"]["source_roots"]) + ["resources", ".runtime", "../../notes/Lerobot.md"]
        inventory = inventory_artifacts(config["paths"]["root"], scan_paths=scan, exclude_paths=["outputs/rebuild", ".g1-rebuild", config["artifacts"]["runs_dir"]])
        # Research weights are slightly above the general large-file cutoff.
        # Fingerprint these protected anchors without hashing the 24 GiB cache candidates.
        for row in inventory["records"]:
            relative = row.get("relative_path") or ""
            anchor = bool(set(row.get("experiment_ids", [])) & BASELINE_EXPERIMENTS)
            if relative.startswith("resources/checkpoints/"):
                row.update({"protected": True, "action": "keep"})
                row["reasons"].append("Preserve the explicit historical reference checkpoint.")
                anchor = True
            if anchor and row.get("kind") == "file" and row.get("sha256") is None:
                row["sha256"] = sha256_file(row["path"])
                row["hash_status"] = "complete"
        inventory["summary"]["actions"] = dict(sorted(Counter(row["action"] for row in inventory["records"]).items()))
        inventory_path = run.path / "manifests" / "inventory.json"
        protected_path = run.path / "manifests" / "protected_artifacts.json"
        write_manifest(inventory, inventory_path)
        protection = _protection(inventory)
        write_manifest(protection, protected_path)
        for path, value in ((Path(config["artifacts"]["inventory_path"]), inventory),
                            (Path(config["artifacts"]["protection_manifest_path"]), protection)):
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                write_manifest(value, path)
        report.update({"stage": stage, "inventory": inventory["summary"], "inventory_path": str(inventory_path),
                       "protection_manifest_path": str(protected_path), "errors": inventory["errors"], "deleted_files": 0})
        return report, not inventory["errors"]
    if stage == "registry":
        from .data_ingest import ingest_raw_registry
        from .data_registry import validate_registry, write_registry
        data = ingest_raw_registry(config["paths"]["root"])
        path = run.path / "manifests" / "raw_registry.jsonl"
        write_registry(data["records"], path)
        configured = Path(config["data"]["registry_path"])
        if not configured.exists():
            configured.parent.mkdir(parents=True, exist_ok=True)
            write_registry(data["records"], configured)
        validation = validate_registry(data["records"])
        report.update({"stage": stage, "registry_path": str(path), "registry": data["summary"],
                       "validation": validation, "errors": data["errors"], "historical_data_used_for_training": False})
        return report, not data["errors"] and validation["ok"]
    if stage == "audit":
        from .data_registry import read_registry, validate_registry
        path = Path(config["data"]["registry_path"])
        if path.is_file():
            report["registry_validation"] = validate_registry(read_registry(path))
        report["stage"] = stage
        report["audit_scope"] = "Configuration, source ordering, checkpoint boundaries and historical registry; no physics steps."
        return report, report["ready_for_execution"] and report.get("registry_validation", {}).get("ok", False)
    if stage == "train":
        from .training import train
        gate = train(config, run.path)
        report.update({"stage": stage, "training_gate": gate})
        return report, False
    report.update({"stage": stage, "status": "blocked",
                   "reason": "This preparation phase has no admitted " + stage + " runner. No historical script will be invoked."})
    return report, False


def build_parser():
    parser = argparse.ArgumentParser(prog="g1-vla", description="G1 rebuild: explicit stages with dataset and execution admission")
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("experiment", help="Use the single validated project configuration")
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--stage", choices=STAGES, default="plan")
    experiment.add_argument("--dry-run", action="store_true")
    experiment.add_argument("--run-id")
    for alias in ("doctor", "prepare", "smoke", "report"):
        command = commands.add_parser(alias, help="Read-only preparation alias")
        command.add_argument("--config", type=Path)
    return parser


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(arguments)
    root = Path(__file__).resolve().parents[2]
    config_path = args.config or root / "configs" / "experiment.json"
    try:
        config = load_config(config_path)
        stage = args.stage if args.command == "experiment" else ("report" if args.command == "report" else "plan")
        if stage == "report":
            runs = Path(config["artifacts"]["runs_dir"]) / "runs"
            items = []
            for path in sorted(runs.glob("*/summary.json"), reverse=True)[:20]:
                items.append(json.loads(path.read_text(encoding="utf-8")))
            print(json.dumps({"schema_version": 1, "runs": items, "training_started": False}, ensure_ascii=False, indent=2))
            return 0
        if stage == "plan" or getattr(args, "dry_run", False) or args.command != "experiment":
            report = readiness(config)
            report.update({"stage": stage, "dry_run": True, "effective_config_sha256": canonical_hash(config)})
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        run = ExperimentRun(Path(config["paths"]["root"]), Path(config["artifacts"]["runs_dir"]), stage, config,
                            ["g1-vla", *arguments], getattr(args, "run_id", None))
        run.begin()
        try:
            report, success = execute_stage(stage, config, run)
        except Exception as exc:
            report, success = {"stage": stage, "error": type(exc).__name__ + ": " + str(exc)}, False
        print(json.dumps(run.finish(report, success), ensure_ascii=False, indent=2))
        return 0 if success else 2
    except (ConfigError, ValueError, OSError) as exc:
        print("g1-vla: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
