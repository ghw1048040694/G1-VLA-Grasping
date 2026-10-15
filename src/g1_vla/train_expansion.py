"""Admit a fixed same-source coverage expansion without claiming session identity.

Only candidate payloads and the original train/validation payloads are opened.
Frozen test identities are metadata exclusions; their arrays, pixels and metrics
are never materialized. All source-native row, camera and complete-chunk rules
remain unchanged, and the original dataset is never rewritten.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
import traceback

import numpy as np

from .artifacts import sha256_file
from .external_data import SOURCE_CAMERA_KEYS, SOURCE_JOINT_NAMES, _feature_names, _formatted_path, _parquet_columns
from .external_sources import verify_download_receipt
from . import grasp_data as native
from . import trajectory_coverage as coverage
from .runtime.experiment import canonical_hash, write_new_json

BACKEND = "grasp_square_train_expansion_v1"
CANDIDATE_IDS = [12, 38, 62, 73, 90, 98, 129, 147]
ORIGINAL_TRAIN_IDS = [20, 21, 75, 76, 125, 126]
VALIDATION_IDS = [180, 181, 182]
TEST_IDS = [280, 281, 282]
ARTIFACT_KEYS = ("arrays", "rgb", "decoded_receipt")


class TrainExpansionError(ValueError):
    """An expansion differs from the predeclared source or frozen validation."""


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(config, value):
    return native._resolve(Path(config["paths"]["root"]).resolve(), value)


def validate_train_expansion_settings(config):
    settings = copy.deepcopy(config["train_expansion"])
    if settings.get("backend") != BACKEND or settings.get("candidate_episode_ids") != CANDIDATE_IDS:
        raise TrainExpansionError("the eight metadata-selected candidates must remain fixed and ordered")
    if settings.get("allow_same_source_coverage") is not True or settings.get("require_independent_sessions") is not False:
        raise TrainExpansionError("this expansion admits disclosed same-source coverage, never verified independent sessions")
    if config["source_native"]["split_episode_ids"] != {
            "train": ORIGINAL_TRAIN_IDS, "validation": VALIDATION_IDS, "test": TEST_IDS}:
        raise TrainExpansionError("the original train/validation/test declaration must remain frozen")
    for key in ("reference_coverage_report_path", "reference_residual_report_path", "session_evidence_path",
                "candidate_build_dir", "build_dir", "audit_report_path"):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise TrainExpansionError(key + " must identify a declared immutable project path")
        _resolve(config, settings[key])
    for prefix in ("reference_coverage_report", "reference_residual_report", "session_evidence"):
        digest = settings.get(prefix + "_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[a-f0-9]{64}", digest) is None:
            raise TrainExpansionError(prefix + " must have a full SHA256")
    for key in ("max_total_rows", "max_candidate_derived_bytes", "max_derived_bytes"):
        if isinstance(settings.get(key), bool) or not isinstance(settings.get(key), int) or settings[key] < 1:
            raise TrainExpansionError(key + " must be a positive integer")
    if settings["max_candidate_derived_bytes"] > settings["max_derived_bytes"]:
        raise TrainExpansionError("candidate derived budget exceeds the complete expanded budget")
    folders = [_resolve(config, settings[key]) for key in ("candidate_build_dir", "build_dir")]
    immutable = [_resolve(config, config["source_native"][key]) for key in ("build_dir", "video_dir")]
    immutable += [_resolve(config, config["external_data"]["local_dir"])]
    for index, folder in enumerate(folders):
        for other in immutable + folders[:index]:
            if folder.is_relative_to(other) or other.is_relative_to(folder):
                raise TrainExpansionError("new output directories must be separate from immutable inputs and each other")
    return settings


def verify_candidate_groups(rows, info, original_manifest):
    """Direct all-camera exclusions are enforced; sharding is not a session ID."""
    indexed = {row["episode_index"]: row for row in rows}
    if len(indexed) != len(rows):
        raise TrainExpansionError("official metadata contains duplicate episode identities")
    frozen = set(VALIDATION_IDS + TEST_IDS)
    held_paths = {_formatted_path(info, "video_path", indexed[identity], camera)
                  for identity in frozen for camera in SOURCE_CAMERA_KEYS}
    original_ids = set(original_manifest["episode_ids"])
    prior = set(original_manifest["prior_diagnostic_episodes_excluded"])
    result = []
    seen_intervals = []
    for identity in CANDIDATE_IDS:
        if identity not in indexed or identity in original_ids or identity in prior:
            raise TrainExpansionError("a candidate is absent or reuses an existing/discovery episode")
        row = indexed[identity]
        paths = {camera: _formatted_path(info, "video_path", row, camera) for camera in SOURCE_CAMERA_KEYS}
        if held_paths.intersection(paths.values()):
            raise TrainExpansionError("candidate {} shares a frozen held-out source camera file".format(identity))
        start, end = row["dataset_from_index"], row["dataset_to_index"]
        for previous_start, previous_end in seen_intervals:
            if max(start, previous_start) < min(end, previous_end):
                raise TrainExpansionError("candidate source global-row intervals overlap")
        seen_intervals.append((start, end))
        result.append({"episode_id": identity, "row_count": row["length"],
            "source_tasks": row.get("tasks", []), "all_camera_video_relative_paths": paths,
            "source_global_index_interval": [start, end], "direct_frozen_video_files_disjoint": True,
            "capture_session_id": None, "capture_session_identity_verified": False,
            "admission_class": "same_source_coverage_only"})
    # Compare intervals against every existing selected episode using metadata,
    # without materializing any held-out state/action/image summary columns.
    for item in result:
        start, end = item["source_global_index_interval"]
        for identity in original_ids:
            row = indexed[identity]
            if max(start, row["dataset_from_index"]) < min(end, row["dataset_to_index"]):
                raise TrainExpansionError("candidate source rows overlap an existing selected episode")
    return result


def _base(config, settings):
    manifest_path = _resolve(config, config["grasp_act"]["dataset_manifest_path"])
    root = manifest_path.parent
    manifest = native.load_grasp_manifest(root, expected_settings=config["source_native"],
                                         expected_source=config["external_data"], verify_files=False)
    references = {}
    for prefix in ("reference_coverage_report", "reference_residual_report", "session_evidence"):
        path = _resolve(config, settings[prefix + "_path"])
        if sha256_file(path) != settings[prefix + "_sha256"]:
            raise TrainExpansionError(prefix + " differs from its predeclared hash")
        references[prefix] = _read(path)
    cov, residual = references["reference_coverage_report"], references["reference_residual_report"]
    preview = [identity for group in cov.get("expansion", {}).get("bounded_metadata_only_preview", [])
               for identity in group["metadata_audit_preview_episode_ids"]]
    if (cov.get("passed") is not True or cov.get("test_payloads_opened") is not False
            or cov.get("test_metrics_read") is not False or preview != CANDIDATE_IDS):
        raise TrainExpansionError("candidate identities differ from the completed metadata-only coverage preview")
    if (residual.get("passed") is not True or residual.get("test_payloads_opened") is not False
            or residual.get("test_used_for_selection") is not False):
        raise TrainExpansionError("the frozen residual baseline is not a completed train/validation-only report")
    manifest_sha, statistics_sha = sha256_file(manifest_path), manifest["statistics"]["sha256"]
    for report in (cov, residual):
        if report.get("bindings", {}).get("manifest_sha256") != manifest_sha:
            raise TrainExpansionError("the reference report does not bind the original dataset")
    if residual.get("bindings", {}).get("statistics_sha256") != statistics_sha:
        raise TrainExpansionError("the residual report does not bind the original train statistics")
    # Session evidence can describe a multi-session source while lacking the
    # episode-to-session mapping required for an independent-data claim.
    evidence = references["session_evidence"]
    if (evidence.get("repo_id") != native.OFFICIAL_REPO or evidence.get("revision") != native.OFFICIAL_REVISION
            or evidence.get("candidate_episode_ids") != CANDIDATE_IDS
            or evidence.get("same_source_coverage_only") is not True
            or evidence.get("capture_session_independence_verified") is not False
            or evidence.get("episode_session_mapping_available") is not False
            or evidence.get("source_public_multi_session_statement") is not True
            or any(evidence.get("candidate_sessions", {}).get(str(identity), {}).get("session_id") is not None
                   or evidence.get("candidate_sessions", {}).get(str(identity), {}).get("verified") is not False
                   for identity in CANDIDATE_IDS)):
        raise TrainExpansionError("session-source evidence does not bind the disclosed unknown per-episode sessions")
    binding = {"module_sha256": sha256_file(__file__), "settings_sha256": canonical_hash(settings),
        "original_manifest_path": str(manifest_path), "original_manifest_sha256": manifest_sha,
        "original_statistics_sha256": statistics_sha,
        "prior_residual_report_sha256": settings["reference_residual_report_sha256"],
        "reference_coverage_report_sha256": settings["reference_coverage_report_sha256"],
        "session_evidence_sha256": settings["session_evidence_sha256"],
        "source_revision": manifest["source_revision"]}
    return root, manifest, references, binding


def _stage(config, run_dir, filename):
    settings = validate_train_expansion_settings(config)
    target = Path(run_dir) / "reports" / filename
    if not Path(run_dir).is_dir() or target.exists():
        raise TrainExpansionError("an existing run and a new immutable report are required")
    target.parent.mkdir(exist_ok=True)
    report = {"schema_version": 1, "passed": False, "status": "failed", "settings": settings,
        "training_started": False, "optimizer_steps": 0, "test_payloads_opened": False,
        "test_metrics_read": False, "labels_modified": False, "closed_loop_control": False,
        "independent_session_data_admitted": False, "admission_class": "same_source_coverage_only"}
    return settings, target, report


def _array_summary(state, action):
    speeds = np.max(np.abs(np.diff(state.astype(np.float64), axis=0)) * 30.0, axis=1)
    return {"state_bytes_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
        "action_bytes_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
        "same_row_hold_mae_rad": float(np.abs(action.astype(np.float64) - state).mean()),
        "state_range_rad_by_joint": np.ptp(state.astype(np.float64), axis=0).tolist(),
        "action_range_rad_by_joint": np.ptp(action.astype(np.float64), axis=0).tolist(),
        "moving_interval_fraction": float((speeds >= .1).mean()),
        "stationary_interval_fraction": float((speeds <= .02).mean()),
        "speed_definition": "max over 28 joints abs(q[t]-q[t-1])*30; first row excluded"}


def _write_episode(build, identity, metadata, columns, video_verified, videos, settings):
    prefix = "videos/" + native.CAMERA
    start, end = float(metadata[prefix + "/from_timestamp"]), float(metadata[prefix + "/to_timestamp"])
    count = len(columns["timestamp"])
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or abs(end-start-count/30.) > settings["timestamp_tolerance_s"]:
        raise TrainExpansionError("candidate camera boundaries differ from complete recorded rows")
    base = build / "episodes" / "episode_{:06d}".format(identity)
    image_path, array_path, receipt_path = base.with_suffix(".rgb.npy"), base.with_suffix(".source.npz"), base.with_suffix(".decoded.json")
    video_name = _formatted_path(settings["source_info"], "video_path", metadata, native.CAMERA)
    decoded = native.decode_episode_rgb(native._inside(videos, video_name), start + columns["timestamp"].astype(np.float64),
        image_path, width=settings["image_width"], height=settings["image_height"], source_width=640, source_height=480,
        fps=30., tolerance_s=settings["timestamp_tolerance_s"])
    decoded.update(episode_id=identity, camera_key=native.CAMERA, source_video=video_verified[video_name],
        source_video_relative_path=video_name, source_video_from_timestamp_s=start, source_video_to_timestamp_s=end)
    write_new_json(receipt_path, decoded)
    with array_path.open("xb") as handle:
        np.savez(handle, source_state=columns["observation.state"], source_action=columns["action"],
            source_timestamp_s=columns["timestamp"], source_frame_index=columns["frame_index"],
            source_episode_index=columns["episode_index"], source_global_index=columns["index"],
            source_task_index=columns["task_index"], source_state_joint_names=np.asarray(SOURCE_JOINT_NAMES),
            source_action_joint_names=np.asarray(SOURCE_JOINT_NAMES),
            decoded_source_pts_s=np.asarray([item["decoded_source_pts_s"] for item in decoded["frames"]]),
            decoded_source_pts_ticks=np.asarray([item["decoded_pts_ticks"] for item in decoded["frames"]], dtype=np.int64))
    return {"episode_id": identity, "split": "train", "row_count": count,
        "sample_count": count-settings["chunk_steps"]+1, "excluded_tail_observation_count": settings["chunk_steps"]-1,
        "source_tasks": metadata.get("tasks", []), "source_task_indices": sorted(set(map(int, columns["task_index"]))),
        "data_relative_path": _formatted_path(settings["source_info"], "data_path", metadata),
        "all_camera_video_relative_paths": {key: _formatted_path(settings["source_info"], "video_path", metadata, key) for key in SOURCE_CAMERA_KEYS},
        "arrays": native._file_record(array_path, build), "rgb": native._file_record(image_path, build),
        "decoded_receipt": native._file_record(receipt_path, build), "paired_rgb_frames": decoded["paired_frame_count"],
        "maximum_absolute_pts_error_s": decoded["maximum_absolute_pts_error_s"],
        "payload_audit": _array_summary(columns["observation.state"], columns["action"])}


def _reject_exact_duplicate_trajectories(root, manifest, candidate_items):
    seen = {}
    for item in manifest["episodes"]:
        if item["split"] != "train":
            continue
        path = native._verify_artifact(root, item["arrays"])
        with np.load(path, allow_pickle=False) as saved:
            state, action = saved["source_state"], saved["source_action"]
        summary = _array_summary(state, action)
        pair = (item["row_count"], summary["state_bytes_sha256"], summary["action_bytes_sha256"])
        seen[pair] = item["episode_id"]
    for item in candidate_items:
        summary = item["payload_audit"]
        pair = (item["row_count"], summary["state_bytes_sha256"], summary["action_bytes_sha256"])
        if pair in seen:
            raise TrainExpansionError("candidate {} exactly duplicates state/action trajectory {}".format(item["episode_id"], seen[pair]))
        seen[pair] = item["episode_id"]


def audit_train_expansion(config, run_dir):
    """Decode all eight candidates; any protocol failure blocks the whole set."""
    settings, target, report = _stage(config, run_dir, "train_expansion_audit.json")
    started, built, build = time.perf_counter(), [], _resolve(config, settings["candidate_build_dir"])
    created = False
    try:
        original_root, original, references, binding = _base(config, settings)
        report["bindings"] = binding
        if build.exists():
            raise FileExistsError("candidate audit output already exists: " + str(build))
        info, rows, metadata_receipt = coverage._metadata(config, original)
        if (tuple(_feature_names(info["features"]["observation.state"], "observation.state")) != SOURCE_JOINT_NAMES
                or tuple(_feature_names(info["features"]["action"], "action")) != SOURCE_JOINT_NAMES
                or set(key for key, value in info["features"].items() if value.get("dtype") == "video") != set(SOURCE_CAMERA_KEYS)):
            raise TrainExpansionError("candidate source joint order or camera identities changed")
        camera = info["features"][native.CAMERA].get("info", {})
        if (camera.get("video.width"), camera.get("video.height"), camera.get("video.fps"), camera.get("video.codec")) != (640, 480, 30., "av1"):
            raise TrainExpansionError("original left eye metadata differs from the pinned source")
        report["candidate_metadata"] = verify_candidate_groups(rows, info, original)
        report["source_metadata"] = metadata_receipt
        indexed = {row["episode_index"]: row for row in rows}
        total = sum(indexed[identity]["length"] for identity in CANDIDATE_IDS)
        if total + sum(item["row_count"] for item in original["episodes"] if item["split"] != "test") > settings["max_total_rows"]:
            raise TrainExpansionError("expanded complete row total exceeds the declared bound")
        planned = total*(config["source_native"]["image_width"]*config["source_native"]["image_height"]*3+2048)+1048576
        if planned > settings["max_candidate_derived_bytes"]:
            raise TrainExpansionError("candidate complete RGB exceeds the predeclared byte budget")
        source = _resolve(config, config["external_data"]["local_dir"])
        videos = _resolve(config, config["source_native"]["video_dir"])
        data_names = sorted({_formatted_path(info, "data_path", indexed[identity]) for identity in CANDIDATE_IDS})
        video_names = sorted({_formatted_path(info, "video_path", indexed[identity], native.CAMERA) for identity in CANDIDATE_IDS})
        source_verified = verify_download_receipt(source, native.OFFICIAL_REPO, native.OFFICIAL_REVISION, data_names)
        video_verified = verify_download_receipt(videos, native.OFFICIAL_REPO, native.OFFICIAL_REVISION, video_names)
        before = {"source": sha256_file(source/"provenance.json"), "video": sha256_file(videos/"provenance.json")}
        build.mkdir(parents=True, exist_ok=False)
        created = True
        (build/"episodes").mkdir()
        native_settings = dict(config["source_native"], source_info=info)
        for identity in CANDIDATE_IDS:
            metadata = indexed[identity]
            columns = native.validate_episode_rows(_parquet_columns(native._inside(source, _formatted_path(info, "data_path", metadata)), identity),
                                                  metadata, identity, native_settings)
            built.append(_write_episode(build, identity, metadata, columns, video_verified, videos, native_settings))
            if native._derived_bytes(build) > settings["max_candidate_derived_bytes"]:
                raise TrainExpansionError("actual candidate output exceeds the declared byte budget")
        _reject_exact_duplicate_trajectories(original_root, original, built)
        verify_download_receipt(source, native.OFFICIAL_REPO, native.OFFICIAL_REVISION, data_names)
        verify_download_receipt(videos, native.OFFICIAL_REPO, native.OFFICIAL_REVISION, video_names)
        if before != {"source": sha256_file(source/"provenance.json"), "video": sha256_file(videos/"provenance.json")}:
            raise TrainExpansionError("immutable source receipts changed while decoding candidates")
        candidate_manifest = {"schema_version": 1, "backend": BACKEND, "passed": True,
            "settings_sha256": canonical_hash(settings), "bindings": binding, "episode_ids": CANDIDATE_IDS,
            "episodes": built, "source_revision": native.OFFICIAL_REVISION,
            "source_repo_id": native.OFFICIAL_REPO, "verified_source_files": source_verified,
            "verified_video_files": video_verified, "source_provenance_sha256": before["source"],
            "video_provenance_sha256": before["video"], "training_started": False,
            "independent_session_data_admitted": False, "admission_class": "same_source_coverage_only"}
        write_new_json(build/"candidate_manifest.json", candidate_manifest)
        if native._derived_bytes(build) > settings["max_candidate_derived_bytes"]:
            raise TrainExpansionError("candidate manifest exceeds the declared output budget")
        report.update(passed=True, status="passed_complete_candidate_protocol_same_source_coverage",
            candidate_manifest_path=str(build/"candidate_manifest.json"), candidate_manifest_sha256=sha256_file(build/"candidate_manifest.json"),
            candidate_episode_ids=CANDIDATE_IDS, candidate_row_count=total,
            candidate_complete_anchor_count=sum(item["sample_count"] for item in built),
            candidate_bytes=native._derived_bytes(build), episodes=built,
            all_camera_direct_frozen_file_disjoint=True, exact_state_action_duplicate_trajectories_found=False,
            capture_session_mapping_available=False, scene_operator_independence_verified=False,
            session_evidence={"path": settings["session_evidence_path"], "sha256": settings["session_evidence_sha256"],
                "interpretation": "Public multi-session descriptions do not identify per-episode sessions; all eight remain same-source coverage."},
            source_caches_unchanged=True, all_candidates_required_no_performance_filtering=True,
            source_label_protocol_preserved=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
        if created and not (build/"failed.json").exists():
            write_new_json(build/"failed.json", {"passed": False, "error": report["error"], "completed_episode_ids": [item["episode_id"] for item in built]})
    finally:
        report["total_elapsed_s"] = time.perf_counter()-started
        write_new_json(target, report)
    return report


def _copy_item(source, destination, item):
    copied = copy.deepcopy(item)
    for key in ARTIFACT_KEYS:
        path = native._verify_artifact(source, item[key])
        output = native._inside(destination, item[key]["relative_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as src, output.open("xb") as dst:
            shutil.copyfileobj(src, dst, 1<<20)
        if native._file_record(output, destination) != item[key]:
            raise TrainExpansionError("copied payload differs from immutable input bytes")
    return copied


def build_train_expansion(config, run_dir):
    """Copy admitted train/validation bytes and recompute only train statistics."""
    settings, target, report = _stage(config, run_dir, "train_expansion_build.json")
    started, copied, build = time.perf_counter(), [], _resolve(config, settings["build_dir"])
    created = False
    try:
        original_root, original, references, binding = _base(config, settings)
        audit_path = _resolve(config, settings["audit_report_path"])
        audit = _read(audit_path)
        if (audit.get("passed") is not True or audit.get("settings") != settings or audit.get("bindings") != binding
                or audit.get("candidate_episode_ids") != CANDIDATE_IDS or audit.get("test_payloads_opened") is not False
                or audit.get("all_candidates_required_no_performance_filtering") is not True):
            raise TrainExpansionError("the complete candidate audit differs from the declared build")
        candidate_root = _resolve(config, settings["candidate_build_dir"])
        candidate_path = candidate_root/"candidate_manifest.json"
        if sha256_file(candidate_path) != audit["candidate_manifest_sha256"] or (candidate_root/"failed.json").exists():
            raise TrainExpansionError("candidate derived manifest changed or is a failed partial build")
        candidates = _read(candidate_path)
        if (candidates.get("passed") is not True or candidates.get("bindings") != binding
                or candidates.get("episode_ids") != CANDIDATE_IDS):
            raise TrainExpansionError("candidate manifest differs from the complete audit admission")
        if build.exists():
            raise FileExistsError("expanded dataset already exists: " + str(build))
        build.mkdir(parents=True, exist_ok=False)
        created = True
        # Original train and frozen validation are copied as whole byte-identical
        # artifacts; test records are not copied or verified as model payloads.
        for item in original["episodes"]:
            if item["split"] == "train":
                copied.append(_copy_item(original_root, build, item))
        for item in candidates["episodes"]:
            copied.append(_copy_item(candidate_root, build, item))
        for item in original["episodes"]:
            if item["split"] == "validation":
                copied.append(_copy_item(original_root, build, item))
        statistics = native.train_only_statistics(build, copied, native.CAMERA)
        write_new_json(build/"train_stats.json", statistics)
        manifest = {key: copy.deepcopy(value) for key, value in original.items() if key not in (
            "episodes", "statistics", "settings", "settings_sha256", "episode_ids", "split_episode_ids", "file_groups",
            "split_sample_counts", "total_rows", "total_paired_rgb_frames", "planned_derived_bytes_upper_bound",
            "actual_derived_bytes_before_manifest", "verified_video_files", "verified_source_files")}
        split = {"train": ORIGINAL_TRAIN_IDS+CANDIDATE_IDS, "validation": VALIDATION_IDS}
        paths = {name: sorted({path for item in copied if item["split"] == name for path in item["all_camera_video_relative_paths"].values()})
                 for name in split}
        if set(paths["train"]).intersection(paths["validation"]):
            raise TrainExpansionError("expanded train and frozen validation share a source camera storage file")
        validation = {str(item["episode_id"]): {key: item[key] for key in ARTIFACT_KEYS}
                      for item in original["episodes"] if item["split"] == "validation"}
        manifest.update(backend=BACKEND, episode_ids=[item["episode_id"] for item in copied], episodes=copied,
            split_episode_ids=split, statistics=native._file_record(build/"train_stats.json", build),
            settings=settings, settings_sha256=canonical_hash(settings), bindings=binding,
            original_manifest_sha256=binding["original_manifest_sha256"], original_statistics_sha256=binding["original_statistics_sha256"],
            prior_residual_report_sha256=binding["prior_residual_report_sha256"], source_native_data_revision=native.OFFICIAL_REVISION,
            validation_payloads_unchanged=validation, metadata_excluded_test_episode_ids=TEST_IDS,
            file_groups={"checked_camera_keys": list(SOURCE_CAMERA_KEYS), "source_video_paths_by_split": paths,
                "all_camera_storage_files_disjoint": True, "capture_sessions_independent_verified": False,
                "layout_or_operator_independence_verified": False},
            total_rows=sum(item["row_count"] for item in copied), total_paired_rgb_frames=sum(item["paired_rgb_frames"] for item in copied),
            split_sample_counts={name: sum(item["sample_count"] for item in copied if item["split"] == name) for name in split},
            candidate_audit_path=str(audit_path), candidate_audit_sha256=sha256_file(audit_path), candidate_manifest_sha256=sha256_file(candidate_path),
            verified_source_files={**original["verified_source_files"], **candidates["verified_source_files"]},
            verified_video_files={**original["verified_video_files"], **candidates["verified_video_files"]},
            admission_class="same_source_coverage_only", independent_session_data_admitted=False,
            test_payloads_opened=False, test_metrics_read=False, current_simulator_adapter_ready=False,
            training_started=False, source_native_training_eligible=True, source_labels_unchanged=True,
            normalization_intervention="recomputed from all rows of the expanded 14-episode train set; no validation statistics",
            expanded_train_episode_count=14, frozen_validation_episode_count=3)
        if manifest["total_rows"] > settings["max_total_rows"] or native._derived_bytes(build) > settings["max_derived_bytes"]:
            raise TrainExpansionError("expanded output exceeds declared row or byte budget")
        write_new_json(build/"manifest.json", manifest)
        if native._derived_bytes(build) > settings["max_derived_bytes"]:
            raise TrainExpansionError("final expanded manifest exceeds byte budget")
        load_expanded_manifest(build, verify_files=True)
        report.update(passed=True, status="passed_immutable_expanded_train_frozen_validation_build", bindings=binding,
            manifest_path=str(build/"manifest.json"), manifest_sha256=sha256_file(build/"manifest.json"),
            statistics_sha256=manifest["statistics"]["sha256"], split_episode_ids=split,
            total_rows=manifest["total_rows"], split_sample_counts=manifest["split_sample_counts"],
            validation_payloads_unchanged=validation, candidate_audit_sha256=sha256_file(audit_path),
            actual_derived_bytes=native._derived_bytes(build), original_dataset_unchanged=True,
            expanded_train_statistics_recomputed=True, data_expansion_training_eligible=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
        if created and not (build/"failed.json").exists():
            write_new_json(build/"failed.json", {"passed": False, "error": report["error"], "completed_episode_ids": [item["episode_id"] for item in copied]})
    finally:
        report["total_elapsed_s"] = time.perf_counter()-started
        write_new_json(target, report)
    return report


def load_expanded_manifest(build_dir, *, verify_files=True):
    """Validate the train/validation-only dataset; never invoke a test loader."""
    root = Path(build_dir).resolve()
    if (root/"failed.json").exists():
        raise TrainExpansionError("a failed expanded build is not admitted")
    manifest = _read(native._inside(root, "manifest.json"))
    train_ids, all_ids = ORIGINAL_TRAIN_IDS+CANDIDATE_IDS, ORIGINAL_TRAIN_IDS+CANDIDATE_IDS+VALIDATION_IDS
    if (manifest.get("backend") != BACKEND or manifest.get("passed") is not True
            or manifest.get("episode_ids") != all_ids
            or manifest.get("split_episode_ids") != {"train": train_ids, "validation": VALIDATION_IDS}
            or manifest.get("metadata_excluded_test_episode_ids") != TEST_IDS
            or manifest.get("source_revision") != native.OFFICIAL_REVISION or manifest.get("source_repo_id") != native.OFFICIAL_REPO
            or manifest.get("source_fps") != 30 or manifest.get("state_dim") != 28 or manifest.get("action_dim") != 28
            or manifest.get("source_state_joint_names") != list(SOURCE_JOINT_NAMES)
            or manifest.get("source_action_joint_names") != list(SOURCE_JOINT_NAMES)
            or manifest.get("camera_key") != native.CAMERA or manifest.get("chunk_steps") != 16
            or manifest.get("image_shape_hwc") != [240, 320, 3]
            or manifest.get("source_image_shape_hwc") != [480, 640, 3]
            or manifest.get("split_unit") != "source_video_file_group"
            or manifest.get("current_simulator_adapter_ready") is not False
            or manifest.get("closed_loop_success_demonstrated") is not False
            or manifest.get("sensor_exposure_clock_verified") is not False
            or manifest.get("motor_application_clock_verified") is not False
            or manifest.get("original_state_action_float32_preserved") is not True
            or manifest.get("image_transform", {}).get("name") != native.TRANSFORM
            or manifest.get("label_alignment") != "unchanged_recorded_row_action_t_through_t_plus_chunk_minus_1"
            or manifest.get("source_native_training_eligible") is not True or manifest.get("complete_chunks_only") is not True
            or manifest.get("independent_session_data_admitted") is not False
            or manifest.get("admission_class") != "same_source_coverage_only"
            or any(manifest.get(key) is not False for key in ("labels_shifted", "actions_clipped", "waist_fabricated", "time_resampled", "test_payloads_opened", "test_metrics_read"))):
        raise TrainExpansionError("expanded source protocol, split, image transform or independence claim differs")
    if canonical_hash(manifest.get("settings", {})) != manifest.get("settings_sha256"):
        raise TrainExpansionError("expanded settings differ from their saved identity")
    audit_path = Path(manifest.get("candidate_audit_path", ""))
    if not audit_path.is_absolute() or not audit_path.is_file() or audit_path.is_symlink():
        raise TrainExpansionError("expanded dataset requires its immutable candidate audit report")
    audit = _read(audit_path)
    if (sha256_file(audit_path) != manifest.get("candidate_audit_sha256")
            or audit.get("passed") is not True or audit.get("bindings") != manifest.get("bindings")
            or audit.get("settings") != manifest.get("settings")
            or audit.get("candidate_episode_ids") != CANDIDATE_IDS
            or audit.get("admission_class") != "same_source_coverage_only"
            or audit.get("independent_session_data_admitted") is not False
            or audit.get("test_payloads_opened") is not False
            or audit.get("all_camera_direct_frozen_file_disjoint") is not True
            or audit.get("source_label_protocol_preserved") is not True
            or manifest.get("bindings", {}).get("module_sha256") != sha256_file(__file__)):
        raise TrainExpansionError("expanded manifest admission differs from the immutable candidate audit")
    episodes = manifest.get("episodes", [])
    if [item.get("episode_id") for item in episodes] != all_ids:
        raise TrainExpansionError("expanded episode records differ from the fixed admission")
    paths, seen, total = {"train": set(), "validation": set()}, set(), 0
    for item in episodes:
        split = "train" if item["episode_id"] in train_ids else "validation"
        count = item.get("row_count")
        if (isinstance(count, bool) or not isinstance(count, int) or not 16 <= count <= 3000
                or item.get("split") != split or item.get("sample_count") != count-15
                or item.get("paired_rgb_frames") != count or item.get("excluded_tail_observation_count") != 15
                or set(item.get("all_camera_video_relative_paths", {})) != set(SOURCE_CAMERA_KEYS)):
            raise TrainExpansionError("expanded episode rows, complete chunks or camera identities differ")
        total += count
        paths[split].update(item["all_camera_video_relative_paths"].values())
        for key in ARTIFACT_KEYS:
            identity = item[key]["relative_path"]
            native._inside(root, identity)
            if identity in seen:
                raise TrainExpansionError("expanded episodes share a derived artifact path")
            seen.add(identity)
            if verify_files:
                native._verify_artifact(root, item[key])
        if split == "validation" and manifest.get("validation_payloads_unchanged", {}).get(str(item["episode_id"])) != {key: item[key] for key in ARTIFACT_KEYS}:
            raise TrainExpansionError("frozen validation payload identities changed")
    if (paths["train"].intersection(paths["validation"])
            or manifest.get("file_groups", {}).get("source_video_paths_by_split") != {name: sorted(value) for name, value in paths.items()}
            or manifest.get("file_groups", {}).get("all_camera_storage_files_disjoint") is not True
            or manifest.get("file_groups", {}).get("capture_sessions_independent_verified") is not False
            or manifest.get("total_rows") != total or manifest.get("total_paired_rgb_frames") != total
            or total > manifest["settings"]["max_total_rows"]
            or manifest.get("split_sample_counts") != {name: sum(item["sample_count"] for item in episodes if item["split"] == name) for name in paths}):
        raise TrainExpansionError("expanded all-camera exclusions or row/sample totals differ")
    statistics_path = native._verify_artifact(root, manifest["statistics"]) if verify_files else native._inside(root, manifest["statistics"]["relative_path"])
    statistics = _read(statistics_path)
    if (statistics.get("training_episode_ids") != train_ids or statistics.get("statistics_include_held_out") is not False
            or statistics.get("validation_or_test_used") is not False or statistics.get("std_minimum") != 1e-3):
        raise TrainExpansionError("expanded normalization is not admitted train-only statistics")
    rows = sum(item["row_count"] for item in episodes if item["split"] == "train")
    for name, shape in (("observation.state", (28,)), ("action", (28,)), (native.CAMERA, (3, 1, 1))):
        count = rows*320*240 if name == native.CAMERA else rows
        feature = statistics.get("features", {}).get(name, {})
        if feature.get("count") != count:
            raise TrainExpansionError("expanded train statistics use the wrong row or pixel population")
        for suffix in ("mean", "std", "unclamped_std"):
            values = np.asarray(feature.get(suffix))
            if values.shape != shape or not np.isfinite(values).all() or (suffix == "std" and np.any(values < 1e-3)):
                raise TrainExpansionError("expanded normalization values are invalid")
    if verify_files:
        actual = native.train_only_statistics(root, episodes, native.CAMERA)
        for name in ("observation.state", "action", native.CAMERA):
            for suffix in ("mean", "std", "unclamped_std"):
                if not np.allclose(statistics["features"][name][suffix], actual["features"][name][suffix], rtol=1e-12, atol=1e-12):
                    raise TrainExpansionError("expanded saved statistics differ from actual train-only rows/pixels")
    return manifest
