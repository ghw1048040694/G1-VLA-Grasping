"""Immutable teacher layout suite, exact denominator and independent raw audit.

This is orchestration for the package teacher and pairing implementations. No
training cache, policy, optimizer, resampling or replacement labels are created.
Smoke is separate evidence and never contributes a success to the formal suite.
"""

import copy
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from .artifacts import sha256_file
from .contracts import ActionContract, canonical_sha256
from .data_pairing import _load_npz, audit_raw_pairing, encode_paired_chunks
from .expert import run_teacher_episode
from .runtime.experiment import write_new_json
from .sim.scene import FIXED_BASE_EQUALITY, OBJECT_NAMES, assert_fixed_scene
from .teacher_evidence import PHASES, export_teacher_evidence, phase_scores


class TeacherAcceptanceError(ValueError):
    """An acceptance manifest or its execution boundary is invalid."""


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TeacherAcceptanceError("duplicate layout manifest JSON key: {}".format(key))
        result[key] = value
    return result


def _spec(value):
    if not isinstance(value, dict) or set(value) != {"layout_id", "positions_m"}:
        raise TeacherAcceptanceError("each layout must contain only layout_id and positions_m")
    identity = value["layout_id"]
    if (not isinstance(identity, str) or not identity.strip() or identity in (".", "..")
            or Path(identity).name != identity or any(character in identity for character in ("/", "\\"))):
        raise TeacherAcceptanceError("layout_id must be one nonempty safe directory name")
    positions = value["positions_m"]
    if not isinstance(positions, dict) or set(positions) != {"green_cube"}:
        raise TeacherAcceptanceError("layout positions_m must contain exactly green_cube")
    xyz = positions["green_cube"]
    if (not isinstance(xyz, list) or len(xyz) != 3 or any(
            isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number)
            for number in xyz)):
        raise TeacherAcceptanceError("green_cube position requires three finite numeric coordinates")
    # Numeric physical identity should not depend on 1 versus 1.0 JSON spelling.
    normalized = {"green_cube": [float(number) for number in xyz]}
    return {"layout_id": identity, "positions_m": normalized,
            "layout_fingerprint": canonical_sha256(normalized)}


def read_layout_manifest(path, *, expected_count):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise TeacherAcceptanceError("layout manifest must be a bounded regular JSON file")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
        raise TeacherAcceptanceError("eval.num_layouts must be a positive integer")
    raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json)
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "scope", "smoke", "layouts"}:
        raise TeacherAcceptanceError("manifest requires schema_version, scope, smoke and layouts only")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise TeacherAcceptanceError("unknown layout manifest schema version")
    if not isinstance(raw["scope"], str) or not raw["scope"].strip():
        raise TeacherAcceptanceError("manifest scope must be a nonempty string")
    if not isinstance(raw["layouts"], list) or len(raw["layouts"]) != expected_count:
        raise TeacherAcceptanceError("full manifest count must equal configured eval.num_layouts")
    smoke = _spec(raw["smoke"])
    layouts = [_spec(value) for value in raw["layouts"]]
    identities = [value["layout_id"] for value in layouts] + [smoke["layout_id"]]
    if len(set(identities)) != len(identities):
        raise TeacherAcceptanceError("smoke and formal layout_id values must be unique")
    fingerprints = [value["layout_fingerprint"] for value in layouts]
    if len(set(fingerprints)) != len(fingerprints):
        raise TeacherAcceptanceError("formal layouts must have unique physical positions")
    return raw, {"schema_version": 1, "scope": raw["scope"], "smoke": smoke, "layouts": layouts}


def _exception(error):
    return {"error_type": type(error).__name__, "message": str(error)}


def _summary(report):
    keys = ("passed", "failure", "control_steps", "physics_steps", "maximum_lift_m",
            "physical_contact_substeps", "maximum_contact_normal_force_n",
            "joint_limit_violation_substeps", "all_numeric_finite", "maximum_limiter_adjustment_rad",
            "command_limit_audit", "actual_tracking", "phases", "report", "raw_episode",
            "sustained_lift_hold_s", "physical_grasp_passed", "closure_start_event",
            "robot_self_collision_substeps_by_pair", "diagnostic_video",
            "active_execution_phase", "planned_next_phase", "failed_ik_label",
            "maximum_green_hand_contact_penetration_m", "maximum_green_table_contact_penetration_m")
    result = {key: report.get(key) for key in keys}
    trajectory = report.get("trajectory")
    result["trajectory"] = ({key: trajectory.get(key) for key in (
        "backend", "progress_blocked_count", "limiter_mismatch_count",
        "maximum_closed_rotation_error_rad", "maximum_taskpoint_tracking_error_m",
        "rotation_measurement_scope")}
        if isinstance(trajectory, dict) else None)
    if result["trajectory"] is not None:
        result["trajectory"]["planning_failed"] = trajectory.get("planning_failure") is not None
    return result


def _write_scores(path, layouts):
    """One row per declared layout, including every exception and failed phase."""
    fields = ["layout_id", "passed", "failure", "exception"] + list(PHASES) + [
        "maximum_lift_m", "sustained_lift_hold_s", "physical_contact_substeps",
        "maximum_green_hand_contact_penetration_m", "maximum_green_table_contact_penetration_m",
        "joint_limit_violation_substeps", "command_limits_passed", "numeric_finite",
        "pairing_eligible", "sensor_complete", "raw_videos_complete", "diagnostic_videos_complete",
        "trajectory_backend", "trajectory_progress_blocked_count", "trajectory_limiter_mismatch_count",
        "trajectory_planning_failed", "maximum_closed_rotation_error_rad", "maximum_taskpoint_tracking_error_m"]
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for layout in layouts:
            teacher, pairing = layout["teacher"] or {}, layout["pairing"] or {}
            row = {key: teacher.get(key) for key in fields if key in teacher}
            row.update(layout_id=layout["layout_id"], passed=layout["passed"],
                       failure=teacher.get("failure"), exception=json.dumps(layout["exception"], ensure_ascii=False),
                       command_limits_passed=(teacher.get("command_limit_audit") or {}).get("passed"),
                       numeric_finite=teacher.get("all_numeric_finite"),
                       pairing_eligible=pairing.get("pairing_eligible"), sensor_complete=pairing.get("sensor_complete"),
                       raw_videos_complete=(layout["evidence"] or {}).get("complete"),
                       diagnostic_videos_complete=(teacher.get("diagnostic_video") or {}).get("complete"))
            trajectory = teacher.get("trajectory") or {}
            for key in ("backend", "progress_blocked_count", "limiter_mismatch_count", "planning_failed"):
                row["trajectory_" + key] = trajectory.get(key)
            for key in ("maximum_closed_rotation_error_rad", "maximum_taskpoint_tracking_error_m"):
                row[key] = trajectory.get(key)
            for phase in PHASES:
                score = (layout["phase_scores"] or {}).get(phase, {})
                row[phase] = "completed" if score.get("completed") else "failed" if score.get("entered") else "not_entered"
            writer.writerow(row)


def _scene_audit(teacher_report, spec, episode_dir, config):
    """Verify frozen external identity against both manifest and actual XML."""
    scene = teacher_report.get("scene")
    if not isinstance(scene, dict):
        raise TeacherAcceptanceError("teacher scene manifest is missing")
    if (scene.get("object_names") != ["green_cube"] or scene.get("assisted_grasp") is not False
            or scene.get("fixed_base_equality") != FIXED_BASE_EQUALITY):
        raise TeacherAcceptanceError("teacher scene must declare only green_cube and its strict base weld")
    if canonical_sha256(scene.get("positions_m")) != spec["layout_fingerprint"]:
        raise TeacherAcceptanceError("scene declared positions differ from the frozen external layout")
    if teacher_report.get("layout_fingerprint") != spec["layout_fingerprint"]:
        raise TeacherAcceptanceError("teacher layout fingerprint differs from the frozen external layout")
    declared_path = Path(scene.get("scene", ""))
    path = declared_path.resolve()
    if declared_path.is_symlink() or path != (episode_dir / "scene.xml").resolve() or not path.is_file():
        raise TeacherAcceptanceError("teacher scene must be its regular run-local scene.xml")
    if path.stat().st_size > 16 * 1024 * 1024 or sha256_file(path) != scene.get("scene_sha256"):
        raise TeacherAcceptanceError("actual scene XML does not match its declared hash")
    source = Path(config["scene"]["asset_path"])
    if not source.is_absolute():
        source = Path(config["paths"]["root"]) / source
    if Path(scene.get("source", "")).resolve() != source.resolve() or scene.get("source_sha256") != sha256_file(source):
        raise TeacherAcceptanceError("teacher scene source differs from the unified configured source asset")
    xml = ET.parse(path).getroot()
    assert_fixed_scene(xml)
    bodies = [body for body in xml.findall(".//body") if body.get("name") in set(OBJECT_NAMES) | {"blue_receptacle", "warehouse_tote"}]
    if len(bodies) != 1 or bodies[0].get("name") != "green_cube":
        raise TeacherAcceptanceError("actual XML must contain only the green_cube task object")
    if bodies[0] not in xml.findall("./worldbody/body"):
        raise TeacherAcceptanceError("green_cube must have an explicit world-frame initial pose")
    free_bodies = [body.get("name") for body in xml.findall(".//body")
                   if body.find("freejoint") is not None or any(joint.get("type") == "free" for joint in body.findall("joint"))]
    if any(name not in ("pelvis", "green_cube") for name in free_bodies):
        raise TeacherAcceptanceError("actual XML contains an undeclared free task body")
    actual_positions = {"green_cube": [float(value) for value in bodies[0].get("pos", "").split()]}
    if canonical_sha256(actual_positions) != spec["layout_fingerprint"]:
        raise TeacherAcceptanceError("actual XML green position differs from the frozen layout; declare coordinates at XML precision")
    return {"passed": True, "scene_sha256": sha256_file(path), "source_sha256": sha256_file(source),
            "actual_positions_m": actual_positions, "layout_fingerprint": spec["layout_fingerprint"],
            "object_names": ["green_cube"], "assisted_grasp": False, "fixed_base_equality": FIXED_BASE_EQUALITY}


def run_teacher_acceptance(config, run_dir, *, smoke=False, focus=False):
    """Run predeclared layouts and keep every failure in the suite denominator."""
    if type(smoke) is not bool or type(focus) is not bool or (smoke and focus):
        raise TeacherAcceptanceError("smoke must be an explicit boolean")
    if config["train"]["enabled"] or config["wm"]["enabled"]:
        raise TeacherAcceptanceError("teacher acceptance requires disabled training and world model")
    root = Path(config["paths"]["root"]).resolve()
    destination = Path(run_dir).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise TeacherAcceptanceError("teacher run_dir must be inside its project for independently auditable lineage") from exc
    configured = config["eval"]["layouts_path"]
    if configured is None:
        raise TeacherAcceptanceError("eval.layouts_path must identify the predeclared immutable manifest")
    path = Path(configured)
    path = path if path.is_absolute() else root / path
    original, manifest = read_layout_manifest(path, expected_count=config["eval"]["num_layouts"])
    rate = config["eval"]["gates"]["expert_lift_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise TeacherAcceptanceError("expert_lift_rate must be a finite probability")
    contract = ActionContract.from_config(config, project_root=root)
    required = math.ceil(config["eval"]["num_layouts"] * rate)
    manifest_path = destination / "teacher_layout_manifest.json"
    provenance_path = destination / "teacher_layout_manifest_provenance.json"
    output_path = destination / "teacher_acceptance.json"
    if output_path.exists() or (destination / "teacher").exists() or (destination / "new_pairing").exists():
        raise TeacherAcceptanceError("teacher acceptance evidence is immutable; choose a new run directory")
    # These exclusive writes occur before the first teacher invocation. A reused
    # run cannot silently change positions, denominator or the acceptance gate.
    write_new_json(manifest_path, original)
    write_new_json(provenance_path, {
        "schema_version": 1, "source_path": str(path.resolve()), "source_file_sha256": sha256_file(path),
        "source_canonical_sha256": canonical_sha256(original), "frozen_manifest_sha256": sha256_file(manifest_path),
        "interpreted_manifest_sha256": canonical_sha256(manifest), "scope": manifest["scope"],
        "formal_layout_count": len(manifest["layouts"]), "formal_required_successes": required,
        "expert_lift_rate": rate, "smoke_separate_from_formal_suite": True,
        "action_contract_sha256": contract.contract_sha256, "training_started": False,
    })
    selected = [manifest["smoke"]] if smoke else manifest["layouts"]
    if focus:
        ids = config["eval"]["diagnostic_layout_ids"]
        known = {row["layout_id"] for row in manifest["layouts"]}
        if not ids or len(set(ids)) != len(ids) or not set(ids).issubset(known):
            raise TeacherAcceptanceError("focus IDs must be unique members of the original frozen layouts")
        selected = [row for row in manifest["layouts"] if row["layout_id"] in ids]
    results, records = [], []
    for spec in selected:
        episode_dir = destination / "teacher" / spec["layout_id"]
        raw_path = episode_dir / "expert_pick_place_episode.npz"
        result = {"layout_id": spec["layout_id"], "positions_m": spec["positions_m"],
                  "layout_fingerprint": spec["layout_fingerprint"], "passed": False,
                  "teacher": None, "failure_reasons": [], "exception": None,
                  "scene_audit": None,
                  "raw_path": str(raw_path) if raw_path.is_file() else None,
                  "pairing": None, "chunks": None, "evidence": None, "evidence_error": None,
                  "phase_scores": None, "training_eligible": False, "split": "diagnostic"}
        episode_report = {}
        try:
            episode_report = run_teacher_episode(config, copy.deepcopy(spec), episode_dir)
            if not isinstance(episode_report, dict):
                raise TeacherAcceptanceError("teacher must return its explicit episode report")
            result["teacher"] = _summary(episode_report)
            result["phase_scores"] = phase_scores(episode_report)
            try:
                result["scene_audit"] = _scene_audit(episode_report, spec, episode_dir, config)
            except (OSError, ValueError, TypeError, ET.ParseError) as error:
                result["scene_audit"] = {"passed": False, **_exception(error)}
            reported_raw = Path(episode_report.get("raw_episode", raw_path)).resolve()
            if reported_raw != raw_path.resolve():
                raise TeacherAcceptanceError("teacher raw must remain in its declared layout directory")
        except Exception as error:
            result["exception"] = _exception(error)
            result["failure_reasons"].append("teacher_exception")
        if raw_path.is_file() and not raw_path.is_symlink():
            result["raw_path"] = str(raw_path)
            records.append({
                "record_id": canonical_sha256({"run_dir": str(destination), "layout_id": spec["layout_id"]}),
                "raw_path": str(raw_path), "raw_role": "expert_trajectory", "raw_sha256": sha256_file(raw_path),
                "regime": "expert", "parent_root": str(root), "source_rollout": str(episode_dir),
                "parent_rollout_id": spec["layout_id"], "layout_fingerprint": spec["layout_fingerprint"],
                "target_object": "green_cube", "training_eligible": False, "split": "diagnostic",
            })
            try:
                result["evidence"] = export_teacher_evidence(episode_report, raw_path, episode_dir, config)
            except Exception as error:
                # A failed evidence export cannot silently remove a failed
                # layout or change the established physical success criterion.
                result["evidence_error"] = _exception(error)
        else:
            result["failure_reasons"].append("raw_episode_missing")
        results.append(result)
    pairing_error = None
    try:
        pairing_report = audit_raw_pairing(root, config, records=records, run_dir=destination / "new_pairing")
    except Exception as error:
        pairing_error = _exception(error)
        pairing_report = {"records": [], "summary": {"record_count": len(records), "error_count": 1,
                          "pairing_eligible_count": 0, "training_eligible_count": 0}, "errors": [pairing_error]}
    paired_by_path = {str(Path(row["raw_path"]).resolve()): row for row in pairing_report["records"]}
    for result in results:
        teacher = result["teacher"] or {}
        checks = {
            "scene_layout_audit_failed": (result["scene_audit"] or {}).get("passed") is True,
            "teacher_physical_gate_failed": teacher.get("passed") is True,
            "command_limit_audit_failed": (teacher.get("command_limit_audit") or {}).get("passed") is True
                                          and (teacher.get("command_limit_audit") or {}).get("checked") is True,
            "joint_safety_gate_failed": type(teacher.get("joint_limit_violation_substeps")) is int
                                        and teacher.get("joint_limit_violation_substeps") == 0,
            "numeric_finiteness_gate_failed": teacher.get("all_numeric_finite") is True,
        }
        paired = paired_by_path.get(result["raw_path"]) if result["raw_path"] else None
        result["pairing"] = paired
        checks["raw_pairing_audit_failed"] = paired is not None and paired.get("pairing_eligible") is True
        checks["sensor_recording_incomplete"] = paired is not None and paired.get("sensor_complete") is True
        if paired is not None and paired.get("pairing_status") == "exact_control_interval":
            try:
                episode = _load_npz(Path(result["raw_path"]))
                chunks = encode_paired_chunks(episode, contract)
                consistent = chunks["action_contract_consistent"]
                result["chunks"] = {
                    "real_future_chunk_count": len(consistent),
                    "contract_consistent_chunk_count": int(consistent.sum()),
                    "contract_inconsistent_chunk_count": int((~consistent).sum()),
                    "restart_wait_interval_count": int(episode["restart_wait"].sum()),
                    "anchor_record_indices": chunks["anchor_record_index"].tolist(),
                    "chunk_steps": contract.chunk_steps, "end_padding": False,
                    "label_clipping": False, "training_eligible": False,
                }
            except Exception as error:
                result["chunk_error"] = _exception(error)
                result["failure_reasons"].append("paired_chunk_encoding_failed")
        result["failure_reasons"].extend(label for label, passed in checks.items() if not passed)
        result["failure_reasons"] = sorted(set(result["failure_reasons"]))
        result["passed"] = not bool(result["failure_reasons"])
    successes = sum(result["passed"] for result in results)
    threshold = len(selected) if focus else 1 if smoke else required
    passed = successes >= threshold
    summary = {
        "layout_count": len(selected), "evaluated_layout_count": len(results), "denominator": len(selected),
        "successes": successes, "failures": len(selected) - successes, "success_rate": successes / len(selected),
        "required_successes": threshold, "formal_required_successes": required,
        "formal_layout_count": len(manifest["layouts"]), "exception_layout_count": sum(row["exception"] is not None for row in results),
        "raw_episode_count": len(records), "pairing_eligible_count": pairing_report["summary"]["pairing_eligible_count"],
        "real_future_chunk_count": sum((row["chunks"] or {}).get("real_future_chunk_count", 0) for row in results),
        "contract_consistent_chunk_count": sum((row["chunks"] or {}).get("contract_consistent_chunk_count", 0) for row in results),
        "contract_inconsistent_chunk_count": sum((row["chunks"] or {}).get("contract_inconsistent_chunk_count", 0) for row in results),
        "training_eligible_count": 0,
        "video_complete_layout_count": sum((row["evidence"] or {}).get("complete") is True for row in results),
        "video_export_error_count": sum(row["evidence_error"] is not None for row in results),
        "phase_completed_counts": {phase: sum(((row["phase_scores"] or {}).get(phase) or {}).get("completed") is True
                                                for row in results) for phase in PHASES},
    }
    command_audits = [(result["teacher"] or {}).get("command_limit_audit") or {} for result in results]
    limiter = {
        "checked_layout_count": sum(audit.get("checked") is True for audit in command_audits),
        "passed_layout_count": sum(audit.get("checked") is True and audit.get("passed") is True for audit in command_audits),
        "maximum_adjustment_rad": max((float((result["teacher"] or {}).get("maximum_limiter_adjustment_rad") or 0.)
                                       for result in results), default=0.),
        "velocity_violation_samples": sum(audit.get("velocity_violation_samples", 0) for audit in command_audits),
        "anchor_violation_samples": sum(audit.get("anchor_violation_samples", 0) for audit in command_audits),
        "frozen_command_changed_layout_count": sum(audit.get("frozen_command_changed") is True for audit in command_audits),
    }
    report = {
        "schema_version": 1, "protocol": "green_right_teacher_acceptance_suite_v1", "status": "passed" if passed else "failed",
        "passed": passed, "smoke": smoke, "focus": focus, "scope": manifest["scope"],
        "focus_layout_ids": [row["layout_id"] for row in selected] if focus else [],
        "formal_gate_evaluated": not (smoke or focus), "formal_acceptance_passed": None if smoke or focus else passed,
        "smoke_counts_toward_formal_acceptance": False, "summary": summary, "limiter": limiter, "layouts": results,
        "raw_paths": [record["raw_path"] for record in records],
        "manifest": str(manifest_path), "manifest_provenance": str(provenance_path),
        "pairing_report": str(destination / "new_pairing/raw_pairing.json") if pairing_error is None else None,
        "pairing_summary": pairing_report["summary"], "pairing_error": pairing_error,
        "pairing_errors": pairing_report["errors"], "action_contract_sha256": contract.contract_sha256,
        "chunk_consistency_scope": "diagnostic of unmodified 8-step future labels; independent of the physical lift/hold gate",
        "training_started": False, "training_eligible": False, "split": "diagnostic",
        "evidence_complete": all((row["evidence"] or {}).get("complete") is True for row in results),
        "scores_csv": str(destination / "teacher_scores.csv"),
    }
    _write_scores(destination / "teacher_scores.csv", results)
    write_new_json(output_path, report)
    return report
