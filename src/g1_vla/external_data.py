"""Bounded, diagnostic audit of an official LeRobot sample.

Source columns are retained verbatim.  A named joint correspondence is not a
validated hardware-to-simulator adapter, and a LeRobot row timestamp is not a
measurement of camera exposure or motor application time.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import sha256_file
from .contracts import read_joint_order
from .runtime.experiment import write_new_json


OFFICIAL_REPO = "unitreerobotics/G1_Dex3_GraspSquare_Dataset"
SOURCE_JOINT_NAMES = (
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
)
SOURCE_TARGET_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint", "right_hand_middle_1_joint",
)
JOINT_CORRESPONDENCE = dict(zip(SOURCE_JOINT_NAMES, SOURCE_TARGET_NAMES))
SOURCE_CAMERA_KEYS = (
    "observation.images.cam_left_high", "observation.images.cam_right_high",
    "observation.images.cam_left_wrist", "observation.images.cam_right_wrist",
)
REQUIRED_COLUMNS = ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index")


class ExternalDataError(ValueError):
    """A source identity, layout or bounded sample is invalid."""


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _source_path(base: Path, relative: str) -> Path:
    path = _resolve(base, relative)
    if not path.is_relative_to(base.resolve()):
        raise ExternalDataError("dataset path escapes the source directory")
    return path


def _feature_names(feature: Mapping[str, Any], key: str) -> tuple[str, ...]:
    if feature.get("shape") != [28]:
        raise ExternalDataError("{}.shape must be exactly [28]".format(key))
    names = feature.get("names")
    if isinstance(names, list) and len(names) == 1 and isinstance(names[0], list):
        names = names[0]
    elif isinstance(names, dict) and len(names) == 1:
        names = next(iter(names.values()))
    if not isinstance(names, (list, tuple)) or len(names) != 28 or len(set(names)) != 28:
        raise ExternalDataError("{} must declare 28 unique joint identities".format(key))
    if set(names) != set(SOURCE_JOINT_NAMES):
        raise ExternalDataError("{} contains unsupported official joint identities".format(key))
    return tuple(names)


def joint_mapping(info: Mapping[str, Any], target_names: Sequence[str]) -> dict:
    """Build indices from declared names; never infer identity from a column position."""
    features = info.get("features", {})
    state = _feature_names(features.get("observation.state", {}), "observation.state")
    action = _feature_names(features.get("action", {}), "action")
    target_names = tuple(target_names)
    if len(target_names) != 31 or len(set(target_names)) != 31:
        raise ExternalDataError("current target metadata must have 31 unique joints")
    rows = []
    for source_name in action:
        target = JOINT_CORRESPONDENCE[source_name]
        if target not in target_names:
            raise ExternalDataError("current target metadata lacks {}".format(target))
        rows.append({"source_name": source_name, "state_index": state.index(source_name),
                     "action_index": action.index(source_name), "target_name": target,
                     "target_index": target_names.index(target), "sign": None, "offset_rad": None,
                     "candidate_sign": 1.0, "candidate_offset_rad": 0.0, "candidate_units": "radian",
                     "candidate_basis": "named motor identity; source/current model geometry evidence may support the candidate, collector zero calibration is not supplied",
                     "status": "named_identity_only_sign_zero_and_units_require_verification"})
    return {"source_state_names": list(state), "source_action_names": list(action),
            "target_names": list(target_names), "correspondence": rows,
            "missing_source_joints": [name for name in target_names if name not in SOURCE_TARGET_NAMES],
            "fabricated_values": False, "deployable_adapter": False}


def camera_mapping(info: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    features = info.get("features", {})
    source = [key for key, feature in features.items() if feature.get("dtype") == "video"]
    if set(source) != set(SOURCE_CAMERA_KEYS):
        raise ExternalDataError("source must declare the four official binocular/wrist cameras")
    result = []
    for key in SOURCE_CAMERA_KEYS:
        feature = features[key]
        camera = feature.get("info", {})
        wrist = "_wrist" in key
        side = "left" if "left" in key else "right"
        result.append({"source_key": key, "source_role": side + ("_monocular_wrist" if wrist else "_binocular_head_eye"),
                       "source_shape": feature.get("shape"), "source_video": camera,
                       "current_candidate": "observation.images." + side + "_wrist" if wrist else "observation.images.head",
                       "exact_match": False,
                       "reason": "camera calibration and optical poses are not verified" if wrist else
                       "current head is one synthetic camera; a stereo eye cannot be relabelled as its exact equivalent"})
    observation = config["observation"]
    return {"source": result, "current_camera_ids": observation["camera_ids"],
            "current_camera_keys": observation["camera_keys"],
            "current_width": observation["image_width"], "current_height": observation["image_height"],
            "calibration_verified": False, "images_relabelled": False}


def _array(columns: Mapping[str, Any], key: str, *, vector: bool = False) -> np.ndarray:
    try:
        value = np.asarray(columns[key])
    except (KeyError, ValueError, TypeError) as exc:
        raise ExternalDataError("cannot read source column {}".format(key)) from exc
    if vector and (value.ndim != 2 or value.shape[1] != 28):
        raise ExternalDataError("{} must contain full 28D rows".format(key))
    if not vector:
        value = value.reshape(-1)
    if value.dtype.kind not in "fiu":
        raise ExternalDataError("{} must be numeric".format(key))
    return value


def _distribution(values: np.ndarray) -> dict:
    finite = np.isfinite(values)
    selected = values[finite]
    return {"count": int(values.size), "nonfinite_count": int((~finite).sum()),
            "minimum": float(selected.min()) if selected.size else None,
            "maximum": float(selected.max()) if selected.size else None,
            "median": float(np.median(selected)) if selected.size else None,
            "p95": float(np.percentile(selected, 95)) if selected.size else None,
            "rms": float(np.sqrt(np.mean(np.square(selected)))) if selected.size else None}


def read_xml_joint_ranges(path: Path) -> dict[str, tuple[float, float]]:
    """Read explicit radian ranges without loading MuJoCo or altering a scene."""
    xml = ET.parse(path).getroot()
    compiler = xml.find("compiler")
    if compiler is None or compiler.get("angle") != "radian":
        raise ExternalDataError("current XML must explicitly declare radian angles")
    ranges = {}
    for joint in xml.iter("joint"):
        name, raw = joint.get("name"), joint.get("range")
        if name is not None and raw is not None:
            low, high = map(float, raw.split())
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ExternalDataError("invalid current joint range for {}".format(name))
            if name in ranges:
                raise ExternalDataError("duplicate current XML joint {}".format(name))
            ranges[name] = (low, high)
    return ranges


def inspect_episode(columns: Mapping[str, Any], *, episode_id: int, fps: float,
                    tolerance_s: float, max_rows: int, mapping: Mapping[str, Any],
                    joint_ranges: Mapping[str, tuple[float, float]] | None = None,
                    current_speed_limits: Sequence[float] | None = None) -> tuple[dict, dict]:
    """Audit actual raw rows and return unchanged source arrays for a sample NPZ."""
    arrays = {key: _array(columns, key, vector=key in ("observation.state", "action")) for key in REQUIRED_COLUMNS}
    count = arrays["timestamp"].size
    if count < 2 or count > max_rows:
        raise ExternalDataError("episode row count {} is outside [2, {}]".format(count, max_rows))
    if any(value.shape[0] != count for value in arrays.values()):
        raise ExternalDataError("source columns have inconsistent row counts")
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(tolerance_s) or tolerance_s <= 0:
        raise ExternalDataError("source fps and timestamp tolerance must be finite and positive")
    for key in ("frame_index", "episode_index", "index", "task_index"):
        if arrays[key].dtype.kind not in "iu":
            raise ExternalDataError("{} must have an integer dtype".format(key))
    if not np.all(arrays["episode_index"] == episode_id):
        raise ExternalDataError("data shard has rows from the wrong episode")
    frames, times = arrays["frame_index"], arrays["timestamp"].astype(np.float64)
    interval = np.diff(times)
    expected_times = frames.astype(np.float64) / fps
    errors = []
    if not np.isfinite(times).all():
        errors.append("nonfinite source timestamps")
    if not np.array_equal(frames, np.arange(count, dtype=frames.dtype)):
        errors.append("frame indices do not form the complete zero-based contiguous episode")
    if not np.array_equal(np.diff(arrays["index"]), np.ones(count - 1, dtype=np.int64)):
        errors.append("global indices are not contiguous within the selected episode")
    if not np.all(interval > 0):
        errors.append("source timestamps are not strictly increasing")
    if not np.all(np.abs(times - expected_times) <= tolerance_s):
        errors.append("timestamps differ from frame_index/source_fps beyond configured tolerance")
    state, action = arrays["observation.state"], arrays["action"]
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        errors.append("nonfinite state/action values")
    state_reordered = state[:, [row["state_index"] for row in mapping["correspondence"]]]
    diagnostics = []
    with np.errstate(divide="ignore", invalid="ignore"):
        speeds = np.abs(np.diff(action.astype(np.float64), axis=0)) / interval[:, None]
    for source_index, row in enumerate(mapping["correspondence"]):
        item = {"source_name": row["source_name"], "target_name": row["target_name"],
                "state": _distribution(state_reordered[:, source_index]),
                "action": _distribution(action[:, source_index]),
                "sampled_absolute_command_speed_source_units_per_s": _distribution(speeds[:, source_index])}
        if joint_ranges and row["target_name"] in joint_ranges:
            low, high = joint_ranges[row["target_name"]]
            item["identity_sign_zero_range_hypothesis"] = {
                "range_rad": [low, high],
                "state_outside_count": int(((state_reordered[:, source_index] < low) | (state_reordered[:, source_index] > high)).sum()),
                "action_outside_count": int(((action[:, source_index] < low) | (action[:, source_index] > high)).sum()),
                "verified_conversion": False,
            }
        if current_speed_limits is not None:
            cap = float(current_speed_limits if isinstance(current_speed_limits, (float, int)) else current_speed_limits[row["target_index"]])
            item["current_speed_limit_hypothesis"] = {"limit_rad_s": cap,
                "sampled_intervals_above_count": int((speeds[:, source_index] > cap + 1e-8).sum()),
                "verified_conversion": False}
        diagnostics.append(item)
    residuals = {
        "same_row_action_minus_state": _distribution(action.astype(np.float64) - state_reordered),
        "action_t_minus_state_t_plus_1": _distribution(action[:-1].astype(np.float64) - state_reordered[1:]),
        "action_t_plus_1_minus_state_t": _distribution(action[1:].astype(np.float64) - state_reordered[:-1]),
        "interpretation": "correlation diagnostics only; no residual establishes observation-before-applied-command acquisition order",
    }
    report = {"episode_id": episode_id, "row_count": count, "status": "passed" if not errors else "failed",
              "errors": errors, "timestamp_start_s": float(times[0]) if np.isfinite(times[0]) else None,
              "timestamp_end_s": float(times[-1]) if np.isfinite(times[-1]) else None,
              "global_index_start": int(arrays["index"][0]), "global_index_end": int(arrays["index"][-1]),
              "task_indices": sorted(set(map(int, arrays["task_index"]))),
              "timing": {"source_fps": fps, "interval_s": _distribution(interval),
                         "frame_index_time_error_s": _distribution(np.abs(times - expected_times)),
                         "actual_hardware_timestamps_provided": False, "motor_application_timestamp_provided": False,
                         "camera_exposure_timestamp_provided": False, "observation_action_order_verified": False},
              "joint_diagnostics": diagnostics, "action_state_residuals": residuals,
              "training_eligible": False, "resampled": False, "labels_shifted": False,
              "actions_clipped": False, "waist_values_fabricated": False}
    saved = {"source_state": state, "source_action": action,
             "source_timestamp_s": arrays["timestamp"], "source_frame_index": frames,
             "source_episode_index": arrays["episode_index"], "source_global_index": arrays["index"],
             "source_task_index": arrays["task_index"],
             "source_state_joint_names": np.asarray(mapping["source_state_names"]),
             "source_action_joint_names": np.asarray(mapping["source_action_names"]),
             "source_fps": np.asarray(fps), "training_eligible": np.asarray(False)}
    return report, saved


def _parquet_rows(path: Path, *, filters=None, columns=None) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ExternalDataError("official sample audit requires pyarrow in its isolated environment") from exc
    return pq.read_table(path, filters=filters, columns=columns).to_pylist()


def _parquet_columns(path: Path, episode_id: int) -> dict[str, np.ndarray]:
    """Keep the Arrow numeric dtypes instead of passing raw rows through Python floats."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ExternalDataError("official sample audit requires pyarrow in its isolated environment") from exc
    table = pq.read_table(path, filters=[("episode_index", "=", episode_id)], columns=list(REQUIRED_COLUMNS))
    result = {}
    for key in REQUIRED_COLUMNS:
        values = table[key].to_numpy(zero_copy_only=False)
        if key in ("observation.state", "action"):
            values = np.stack(values) if values.size else np.empty((0, 28), dtype=np.float32)
        result[key] = values
    return result


def _episode_rows(source: Path, info: Mapping[str, Any], episode_id: int,
                  verified_files: Mapping[str, Any]) -> tuple[dict, list[Path]]:
    if info.get("codebase_version") == "v3.0":
        paths = sorted((source / "meta/episodes").glob("chunk-*/file-*.parquet"))
        if not paths or len(paths) > 32:
            raise ExternalDataError("a bounded v3 episode metadata shard is required")
        rows = []
        for path in paths:
            if str(path.relative_to(source)) not in verified_files:
                raise ExternalDataError("episode metadata shard is absent from pinned verified source receipt")
            rows.extend(_parquet_rows(path, filters=[("episode_index", "=", episode_id)]))
        if len(rows) != 1:
            raise ExternalDataError("episode {} does not have one unique metadata row".format(episode_id))
        return rows[0], paths
    raise ExternalDataError("this audit requires the pinned official LeRobot v3.0 layout")


def _formatted_path(info: Mapping[str, Any], key: str, metadata: Mapping[str, Any], camera: str | None = None) -> str:
    prefix = "data" if camera is None else "videos/" + camera
    values = {"chunk_index": int(metadata[prefix + "/chunk_index"]),
              "file_index": int(metadata[prefix + "/file_index"])}
    if camera is not None:
        values["video_key"] = camera
    try:
        return info[key].format(**values)
    except (KeyError, ValueError, AttributeError) as exc:
        raise ExternalDataError("source metadata has an invalid {} template".format(key)) from exc


def _video_audit(config: Mapping[str, Any], info: Mapping[str, Any], metadata: Mapping[str, Any],
                 arrays: Mapping[str, np.ndarray], *, budget: Any, output_dir: Path) -> list[dict]:
    selected = config["external_data"]
    result = []
    for camera in SOURCE_CAMERA_KEYS:
        prefix = "videos/" + camera
        path = _formatted_path(info, "video_path", metadata, camera)
        start = float(metadata[prefix + "/from_timestamp"])
        end = float(metadata[prefix + "/to_timestamp"])
        item = {"camera_key": camera, "source_video_path": path,
                "episode_video_from_timestamp_s": start, "episode_video_to_timestamp_s": end,
                "declared_video_duration_s": end - start,
                "declared_episode_duration_s": len(arrays["source_timestamp_s"]) / float(info["fps"]),
                "timestamp_basis": "episode video offset plus actual source parquet timestamp",
                "sensor_exposure_alignment_verified": False}
        sample_count = int(selected.get("video", {}).get("samples_per_episode", 3))
        positions = np.unique(np.linspace(0, len(arrays["source_timestamp_s"]) - 1, sample_count, dtype=np.int64))
        expected = (start + arrays["source_timestamp_s"][positions].astype(np.float64)).tolist()
        item["source_frame_indices_sampled"] = arrays["source_frame_index"][positions].tolist()
        item["expected_source_pts_s"] = expected
        if selected.get("probe_videos", True):
            try:
                from .video_audit import probe_video_samples
                item["probe"] = probe_video_samples(config, selected["repo_id"], selected["revision"], path, expected,
                    budget=budget, output_dir=output_dir,
                    sample_ids=["episode_{:06d}_frame_{:06d}_{}".format(int(metadata["episode_index"]), int(frame), camera.rsplit(".", 1)[-1])
                                for frame in arrays["source_frame_index"][positions]],
                    expected_width=int(info["features"][camera]["info"]["video.width"]),
                    expected_height=int(info["features"][camera]["info"]["video.height"]),
                    expected_fps=float(info["features"][camera]["info"]["video.fps"]),
                    expected_codec=info["features"][camera]["info"]["video.codec"])
            except Exception as exc:
                item["probe"] = {"status": "failed", "error": "{}: {}".format(type(exc).__name__, str(exc)),
                                 "full_video_hashed": False}
        else:
            item["probe"] = {"status": "not_requested", "full_video_hashed": False}
        result.append(item)
    return result


def audit_official_sample(config: Mapping[str, Any], run_dir: str | Path) -> dict:
    """Audit only selected episodes under one config and save exclusive evidence."""
    root = Path(config["paths"]["root"]).resolve()
    selected = config["external_data"]
    if selected["repo_id"] != OFFICIAL_REPO:
        raise ExternalDataError("the official sample audit supports only the declared Unitree GraspSquare repository")
    if re.fullmatch(r"[0-9a-f]{40}", selected["revision"]) is None:
        raise ExternalDataError("a pinned 40-character dataset revision is required")
    episodes = selected["episode_ids"]
    if (not isinstance(episodes, list) or not 1 <= len(episodes) <= 10 or
            any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in episodes) or
            len(set(episodes)) != len(episodes)):
        raise ExternalDataError("select one to ten distinct nonnegative episode indices")
    source = _resolve(root, selected["local_dir"])
    from .external_sources import DATA_FILES, verify_download_receipt
    verified_files = verify_download_receipt(source, selected["repo_id"], selected["revision"], DATA_FILES)
    provenance_sha = sha256_file(source / "provenance.json")
    info_path = source / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info["fps"])
    if not math.isclose(fps, float(selected["source_fps"]), rel_tol=0, abs_tol=1e-9):
        raise ExternalDataError("configured source cadence differs from official metadata")
    current_metadata = _resolve(root, config["action"]["joint_metadata_path"])
    names = read_joint_order(json.loads(current_metadata.read_text(encoding="utf-8")))
    mapping = joint_mapping(info, names)
    cameras = camera_mapping(info, config)
    scene = _resolve(root, config["scene"]["asset_path"])
    ranges = read_xml_joint_ranges(scene)
    run = Path(run_dir).resolve()
    sample_dir = run / "official_samples"
    sample_dir.mkdir(parents=True, exist_ok=False)
    hashes = {str(info_path.relative_to(source)): sha256_file(info_path)}
    episode_reports = []
    budget = None
    if selected.get("probe_videos", True):
        from .video_audit import RangeReadBudget
        budget = RangeReadBudget(int(selected["video"]["max_read_bytes"]))
    for episode_id in episodes:
        try:
            metadata, metadata_paths = _episode_rows(source, info, episode_id, verified_files)
            relative = _formatted_path(info, "data_path", metadata)
            if relative not in verified_files:
                raise ExternalDataError("episode data shard is absent from pinned verified source receipt")
            path = _source_path(source, relative)
            columns = _parquet_columns(path, episode_id)
            report, arrays = inspect_episode(columns, episode_id=episode_id, fps=fps,
                tolerance_s=float(selected["timestamp_tolerance_s"]), max_rows=int(selected["max_episode_rows"]),
                mapping=mapping, joint_ranges=ranges, current_speed_limits=config["control"]["max_velocity_rad_s"])
            if report["row_count"] != int(metadata["length"]):
                report["errors"].append("parquet row count differs from episode metadata length")
                report["status"] = "failed"
            if (report["global_index_start"] != int(metadata["dataset_from_index"]) or
                    report["global_index_end"] + 1 != int(metadata["dataset_to_index"])):
                report["errors"].append("parquet global indices differ from metadata dataset interval")
                report["status"] = "failed"
            report["episode_metadata"] = {key: value for key, value in metadata.items()
                if key in ("episode_index", "length", "tasks", "dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index")}
            report["source_data_path"] = relative
            report["videos"] = _video_audit(config, info, metadata, arrays, budget=budget, output_dir=sample_dir)
            filename = "episode_{:06d}.npz".format(episode_id)
            with (sample_dir / filename).open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            report["saved_sample"] = str((sample_dir / filename).relative_to(run))
            report["saved_sample_sha256"] = sha256_file(sample_dir / filename)
            for raw_path in [path, *metadata_paths]:
                key = str(raw_path.relative_to(source))
                if key not in hashes:
                    hashes[key] = sha256_file(raw_path)
            episode_reports.append(report)
        except (ExternalDataError, OSError, KeyError, ValueError, TypeError) as exc:
            episode_reports.append({"episode_id": episode_id, "status": "failed",
                                    "error": "{}: {}".format(type(exc).__name__, str(exc)), "training_eligible": False})
    table_passed = sum(item["status"] == "passed" for item in episode_reports)
    video_probes = [video["probe"] for item in episode_reports for video in item.get("videos", [])]
    video_passed = sum(probe.get("passed") is True for probe in video_probes)
    evidence_path = selected.get("semantic_evidence_path")
    semantics = None
    if evidence_path is not None:
        path = _resolve(root, evidence_path)
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ExternalDataError("semantic evidence exceeds the bounded metadata read limit")
        semantics = {"path": str(path), "sha256": sha256_file(path),
                     "evidence": json.loads(path.read_text(encoding="utf-8"))}
    blockers = ["Source has no waist state or waist command for three current action joints.",
                "Same joint names do not verify sign, zero calibration, or exact motor-to-MuJoCo conventions.",
                "Four physical cameras differ from three current simulated views; calibration is absent.",
                "Source action columns are intended absolute motor q targets; current policy outputs one-anchor relative target chunks.",
                "30 Hz source sampled rows cannot be relabelled as current 500/33 Hz next-applied commands.",
                "LeRobot row and video PTS do not prove hardware/camera exposure and applied motor command ordering.",
                "Task is dual-arm red/yellow/blue block stacking; current task is green single-right-hand lift.",
                "Contact/tactile, joint velocities and simulator finger-site observations are unavailable in source columns."]
    if table_passed != len(episodes):
        blockers.append("One or more selected episodes failed structural or numeric checks.")
    if selected.get("probe_videos", True) and video_passed != len(episodes) * 4:
        blockers.append("One or more bounded camera PTS/decode probes did not pass.")
    result = {"schema_version": 1, "stage": "external-audit", "status": "passed" if table_passed == len(episodes) else "failed",
              "repo_id": selected["repo_id"], "revision": selected["revision"], "local_dir": str(source),
              "source_codebase_version": info.get("codebase_version"), "source_fps": fps,
              "current_control_hz": config["control"]["control_hz"],
              "action_mapping": {"source_candidate_semantics": "intended_absolute_motor_q_target",
                                 "exact_hardware_applied_target_verified": False,
                                 "current_protocol": config["action"]["protocol"], "source_actions_reencoded": False},
              "timing_mapping": {"source_period_s": 1.0 / fps,
                                 "current_period_s": 1.0 / float(config["control"]["control_hz"]),
                                 "source_frames_per_current_interval": fps / float(config["control"]["control_hz"]),
                                 "source_clock_is_current_clock": False, "resampling_performed": False},
              "selected_episode_ids": episodes, "episodes_audited": len(episode_reports),
              "episodes_structurally_passed": table_passed, "bounded_video_probes_passed": video_passed,
              "bounded_video_probes_total": len(video_probes), "full_videos_downloaded_by_audit": False,
              "full_video_content_hashes_claimed": False, "source_file_sha256": hashes,
              "source_receipt_verified": True, "source_provenance_sha256": provenance_sha,
              "verified_source_files": verified_files,
              "current_joint_metadata_sha256": sha256_file(current_metadata), "current_scene_sha256": sha256_file(scene),
              "joint_mapping": mapping, "camera_mapping": cameras, "semantic_evidence": semantics,
              "episodes": episode_reports, "training_eligible": False, "training_started": False,
              "current_adapter_ready": False, "training_blockers": blockers,
              "source_native_training_review": {
                  "status": "candidate_requires_separate_source_native_validation",
                  "source_observation_action_protocol_preserved": True,
                  "current_teacher_gate_applies_to": "current green-right-hand simulator workflow only",
                  "current_teacher_failure_invalidates_official_source": False,
                  "source_native_training_started": False,
                  "note": "A future source-native pipeline may use the official protocol after its own validation; this current-task audit does not approve that training pipeline.",
              },
              "audit_scope": "small frozen diagnostic sample; no model training, scene playback, source resampling or action relabelling"}
    write_new_json(run / "reports/external_data_audit.json", result)
    return result


def _finite_close(actual: Any, expected: float, name: str, *, tolerance: float = 1e-9) -> float:
    if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual):
        raise ExternalDataError(name + " must be finite")
    if abs(float(actual) - expected) > tolerance:
        raise ExternalDataError(name + " differs from the pinned source plan")
    return float(actual)


def _run_evidence_path(original_run: Path, value: str, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ExternalDataError(name + " has no evidence path")
    raw = Path(value)
    path = raw if raw.is_absolute() else original_run / raw
    if path.is_symlink() or not path.resolve().is_relative_to(original_run.resolve()) or not path.is_file():
        raise ExternalDataError(name + " must be original regular evidence inside its original run")
    return path.resolve()


def _verify_png_sample(sample: Mapping[str, Any], original_run: Path) -> dict:
    """Re-decode saved pixels; a report's RGB digest alone is insufficient."""
    import hashlib
    from PIL import Image

    path = _run_evidence_path(original_run, sample.get("png_path"), "PNG")
    png_sha = sha256_file(path)
    if png_sha != sample.get("png_sha256"):
        raise ExternalDataError("saved PNG hash differs from its original report")
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != (640, 480) or image.mode != "RGB":
            raise ExternalDataError("saved source PNG must be exactly 640x480 RGB")
        image.load()
        rgb_sha = hashlib.sha256(image.tobytes()).hexdigest()
    if rgb_sha != sample.get("rgb_sha256"):
        raise ExternalDataError("saved PNG pixels differ from the original decoded RGB digest")
    return {"png_path": str(path), "png_sha256": png_sha, "rgb_sha256": rgb_sha,
            "width": 640, "height": 480, "pixel_format": "rgb24"}


def _reconciliation_plan(config: Mapping[str, Any], source: Path, info: Mapping[str, Any],
                         verified: Mapping[str, Any]) -> tuple[dict, dict]:
    settings = config["external_data"]
    if settings["video"]["samples_per_episode"] != 3:
        raise ExternalDataError("reconciliation requires the unchanged predeclared three-frame plan")
    plan, source_arrays = {}, {}
    for episode_id in settings["episode_ids"]:
        metadata, metadata_paths = _episode_rows(source, info, episode_id, verified)
        data_name = _formatted_path(info, "data_path", metadata)
        if data_name not in verified:
            raise ExternalDataError("reconciliation data shard is absent from the pinned receipt")
        arrays = _parquet_columns(_source_path(source, data_name), episode_id)
        count = len(arrays["timestamp"])
        if count != int(metadata["length"]) or count < 3 or count > settings["max_episode_rows"]:
            raise ExternalDataError("reconciliation source row count differs from the declared episode")
        if not np.array_equal(arrays["frame_index"], np.arange(count)):
            raise ExternalDataError("reconciliation source frame indices are incomplete")
        if not np.all(np.abs(arrays["timestamp"] - arrays["frame_index"] / settings["source_fps"]) <= settings["timestamp_tolerance_s"]):
            raise ExternalDataError("reconciliation source timestamps differ from declared source cadence")
        positions = np.linspace(0, count - 1, 3, dtype=np.int64)
        cameras = {}
        for camera in SOURCE_CAMERA_KEYS:
            prefix = "videos/" + camera
            start, end = float(metadata[prefix + "/from_timestamp"]), float(metadata[prefix + "/to_timestamp"])
            cameras[camera] = {"source_video_path": _formatted_path(info, "video_path", metadata, camera),
                "episode_video_from_timestamp_s": start, "episode_video_to_timestamp_s": end,
                "source_frame_indices_sampled": arrays["frame_index"][positions].tolist(),
                "expected_source_pts_s": (start + arrays["timestamp"][positions].astype(np.float64)).tolist()}
        plan[episode_id] = {"row_count": count, "source_data_path": data_name,
                           "metadata_paths": [str(path.relative_to(source)) for path in metadata_paths], "cameras": cameras}
        source_arrays[episode_id] = arrays
    return plan, source_arrays


def _verify_saved_npz(episode: Mapping[str, Any], original_run: Path,
                      source_arrays: Mapping[str, np.ndarray], info: Mapping[str, Any]) -> dict:
    path = _run_evidence_path(original_run, episode.get("saved_sample"), "source NPZ")
    digest = sha256_file(path)
    if digest != episode.get("saved_sample_sha256"):
        raise ExternalDataError("original source NPZ hash differs from its audit report")
    keys = {"source_state": "observation.state", "source_action": "action", "source_timestamp_s": "timestamp",
            "source_frame_index": "frame_index", "source_episode_index": "episode_index",
            "source_global_index": "index", "source_task_index": "task_index"}
    with np.load(path, allow_pickle=False) as sample:
        for saved_key, column in keys.items():
            value = sample[saved_key]
            if value.dtype != source_arrays[column].dtype or not np.array_equal(value, source_arrays[column]):
                raise ExternalDataError("original source NPZ changed raw source values or dtype: " + saved_key)
        if sample["source_state"].shape[1] != 28 or sample["source_action"].shape[1] != 28:
            raise ExternalDataError("original source NPZ has fabricated action dimensions")
        if bool(sample["training_eligible"].item()):
            raise ExternalDataError("original source NPZ was promoted for training")
        _finite_close(float(sample["source_fps"].item()), float(info["fps"]), "NPZ source fps")
        for key, feature in (("source_state_joint_names", "observation.state"), ("source_action_joint_names", "action")):
            if tuple(sample[key].tolist()) != _feature_names(info["features"][feature], feature):
                raise ExternalDataError("original source NPZ joint identities differ from pinned metadata")
    return {"path": str(path), "sha256": digest, "raw_source_values_and_dtype_verified": True}


def _receipt_integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExternalDataError(name + " must be an integer at least " + str(minimum))
    return value


def _verify_range_receipts(probe: Mapping[str, Any], settings: Mapping[str, Any]) -> dict:
    """Validate original transport records, without claiming stored MP4 bytes.

    Range bodies were hashed during the original read but are not stored here.
    This checks their recorded protocol and accounting, not a full-file digest.
    """
    receipts = probe.get("http_range_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise ExternalDataError("passed video probe lacks original HTTP Range receipts")
    source_size = _receipt_integer(probe.get("source_size_bytes"), "source video size", minimum=1)
    total_read = 0
    complete_count = 0
    etags = set()
    for receipt in receipts:
        if not isinstance(receipt, dict) or receipt.get("http_status") != 206 or receipt.get("full_file_checksum") is not False:
            raise ExternalDataError("original range receipt must record HTTP 206 without a full-file checksum claim")
        raw_range = receipt.get("response_content_range")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", raw_range) if isinstance(raw_range, str) else None
        if match is None:
            raise ExternalDataError("original range receipt has no valid Content-Range")
        content_start, content_end, content_total = map(int, match.groups())
        start = _receipt_integer(receipt.get("start_byte"), "range start")
        requested_end = _receipt_integer(receipt.get("requested_end_byte"), "requested range end")
        downloaded = _receipt_integer(receipt.get("downloaded_bytes"), "range downloaded bytes")
        declared_size = _receipt_integer(receipt.get("source_size_bytes"), "range source size", minimum=1)
        if (start != content_start or content_total != source_size or declared_size != source_size or
                content_end < start or content_end >= source_size or requested_end < start or
                content_end != min(requested_end, source_size - 1) or downloaded > content_end - start + 1):
            raise ExternalDataError("original range receipt byte interval disagrees with Content-Range or source size")
        if receipt.get("end_byte") != start + downloaded - 1:
            raise ExternalDataError("original range receipt end does not match actually downloaded bytes")
        if not isinstance(receipt.get("complete"), bool):
            raise ExternalDataError("original range receipt lacks a completion flag")
        if receipt["complete"] is True:
            if downloaded != content_end - start + 1 or downloaded == 0:
                raise ExternalDataError("complete range receipt did not receive the complete declared response body")
            complete_count += 1
        if re.fullmatch(r"[0-9a-f]{64}", receipt.get("sha256", "")) is None:
            raise ExternalDataError("original range receipt has no valid SHA256 digest")
        if receipt.get("etag") is not None:
            if not isinstance(receipt["etag"], str) or not receipt["etag"]:
                raise ExternalDataError("original range ETag must be a nonempty string")
            etags.add(receipt["etag"])
        total_read += downloaded
    if complete_count == 0 or len(etags) > 1:
        raise ExternalDataError("passed video requires a complete HTTP 206 range and consistent source ETag")
    read = _receipt_integer(probe.get("read_bytes"), "camera HTTP read bytes")
    used = _receipt_integer(probe.get("budget_used_bytes"), "shared used budget")
    maximum = _receipt_integer(probe.get("budget_max_read_bytes"), "shared maximum budget", minimum=1)
    if (read != total_read or used < read or used > maximum or maximum != settings["video"]["max_read_bytes"]):
        raise ExternalDataError("original range receipts disagree with camera/shared read budget accounting")
    return {"recorded_http_range_protocol_verified": True, "complete_http_206_receipts": complete_count,
            "total_http_receipts": len(receipts), "recorded_downloaded_bytes": total_read,
            "original_source_size_bytes": source_size, "shared_budget_used_bytes": used,
            "shared_budget_max_read_bytes": maximum, "original_etag": next(iter(etags)) if etags else None,
            "original_range_bodies_rehashed": False, "full_video_integrity_verified": False}


def _verified_video_triplet(video: Mapping[str, Any], planned: Mapping[str, Any], *, original_run: Path,
                            settings: Mapping[str, Any]) -> dict | None:
    """Reject changed plans; incomplete probes supply no candidate evidence."""
    for key in ("source_video_path", "source_frame_indices_sampled"):
        if video.get(key) != planned[key]:
            raise ExternalDataError("original video " + key + " differs from the source plan")
    for key in ("episode_video_from_timestamp_s", "episode_video_to_timestamp_s"):
        _finite_close(video.get(key), planned[key], key)
    if video.get("expected_source_pts_s") != planned["expected_source_pts_s"]:
        raise ExternalDataError("original video requested PTS plan differs from pinned rows")
    probe = video.get("probe", {})
    if probe.get("passed") is not True:
        return None
    if probe.get("status") != "sampled_format_and_pts_passed":
        raise ExternalDataError("video passed flag contradicts its original probe status")
    if probe.get("repo_id") != settings["repo_id"] or probe.get("revision") != settings["revision"] or probe.get("video_path") != planned["source_video_path"]:
        raise ExternalDataError("passed video probe identity differs from its parent/source plan")
    if probe.get("errors") or probe.get("full_video_sha256") is not None or probe.get("sensor_clock_synchronization_proven") is not False:
        raise ExternalDataError("passed video probe has errors or unsupported full-video/sensor-sync claims")
    transport = _verify_range_receipts(probe, settings)
    stream = probe.get("stream", {})
    if (stream.get("width"), stream.get("height")) != (640, 480):
        raise ExternalDataError("passed source video resolution differs")
    _finite_close(stream.get("average_fps"), float(settings["source_fps"]), "video stream fps", tolerance=1e-6)
    if (stream.get("codec_id") != 225 or stream.get("expected_codec_id") != 225 or
            stream.get("encoded_codec") != "av1" or stream.get("encoded_codec_verified_by_id") is not True):
        raise ExternalDataError("passed source video must prove AV1 codec ID 225")
    time_base = stream.get("time_base_s")
    if not isinstance(time_base, (int, float)) or isinstance(time_base, bool) or not math.isfinite(time_base) or time_base <= 0:
        raise ExternalDataError("passed source video lacks a finite positive PTS time base")
    samples = probe.get("samples", [])
    if len(samples) != 3:
        raise ExternalDataError("passed video probe does not contain the complete three samples")
    evidence = []
    limit = 0.5 / float(settings["source_fps"]) + 1e-6
    for sample, requested, frame in zip(samples, planned["expected_source_pts_s"], planned["source_frame_indices_sampled"]):
        _finite_close(sample.get("requested_source_pts_s"), requested, "requested sample PTS")
        decoded = sample.get("decoded_source_pts_s")
        if not isinstance(decoded, (int, float)) or isinstance(decoded, bool) or not math.isfinite(decoded):
            raise ExternalDataError("decoded sample PTS must be finite")
        error = abs(float(decoded) - requested)
        if error > limit:
            raise ExternalDataError("decoded sample PTS exceeds the declared half-source-frame tolerance")
        _finite_close(sample.get("pts_error_s"), error, "sample PTS error")
        allowed = sample.get("allowed_pts_error_s")
        if not isinstance(allowed, (int, float)) or isinstance(allowed, bool) or not math.isfinite(allowed) or allowed > limit or error > allowed:
            raise ExternalDataError("sample PTS tolerance differs from the declared half-source-frame gate")
        ticks = sample.get("decoded_pts_ticks")
        if not isinstance(ticks, int) or isinstance(ticks, bool) or ticks < 0:
            raise ExternalDataError("decoded source PTS ticks must be a nonnegative integer")
        _finite_close(float(decoded), ticks * time_base, "decoded PTS/time-base consistency")
        if (sample.get("width"), sample.get("height"), sample.get("pixel_format")) != (640, 480, "rgb24"):
            raise ExternalDataError("sample pixel metadata differs from original camera dimensions")
        pixels = _verify_png_sample(sample, original_run)
        evidence.append({"source_frame_index": frame, "requested_source_pts_s": requested,
                         "decoded_source_pts_s": float(decoded), "decoded_pts_ticks": ticks,
                         "pts_error_s": error, **pixels})
    return {"source_video_path": planned["source_video_path"], "stream": stream, "samples": evidence,
            "original_probe_status": probe.get("status"), "original_http_range_receipts": probe["http_range_receipts"],
            "original_transport_record_validation": transport}


def reconcile_official_audits(config: Mapping[str, Any], run_dir: str | Path) -> dict:
    """Join compatible successful camera evidence without retrying any network I/O.

    All declared runs remain in the report with their actual pass counts.  This
    certifies cumulative sample coverage, never a single complete audit run.
    """
    from .external_sources import DATA_FILES, verify_download_receipt

    root = Path(config["paths"]["root"]).resolve()
    settings = config["external_data"]
    source = _resolve(root, settings["local_dir"])
    verified = verify_download_receipt(source, settings["repo_id"], settings["revision"], DATA_FILES)
    info = json.loads((source / "meta/info.json").read_text())
    _finite_close(float(info["fps"]), float(settings["source_fps"]), "current source fps")
    paths = settings["audit_report_paths"]
    if (not isinstance(paths, list) or not 1 <= len(paths) <= 16 or
            len(paths) != len(set(paths)) or any(not isinstance(path, str) or not path for path in paths)):
        raise ExternalDataError("declare one to sixteen distinct original audit report paths")
    plan, arrays = _reconciliation_plan(config, source, info, verified)
    candidates = {(episode_id, camera): [] for episode_id in settings["episode_ids"] for camera in SOURCE_CAMERA_KEYS}
    original_reports, errors = [], []
    required_hashes = {"meta/info.json"}
    for episode in plan.values():
        required_hashes.add(episode["source_data_path"])
        required_hashes.update(episode["metadata_paths"])
    for value in paths:
        report_path = _resolve(root, value)
        entry = {"path": str(report_path), "usable": False}
        try:
            if report_path.is_symlink() or not report_path.is_file() or report_path.stat().st_size > 16 * 1024 * 1024:
                raise ExternalDataError("original report must be a bounded regular JSON file")
            if report_path.name != "external_data_audit.json" or report_path.parent.name != "reports":
                raise ExternalDataError("original audit report must remain inside its original run/reports directory")
            original_run = report_path.parent.parent.resolve()
            prior = json.loads(report_path.read_text())
            entry.update(sha256=sha256_file(report_path), original_run=str(original_run),
                         original_source_receipt_verified=prior.get("source_receipt_verified", False),
                         original_status=prior.get("status"), original_camera_passed=prior.get("bounded_video_probes_passed"),
                         original_camera_total=prior.get("bounded_video_probes_total"))
            if ((prior.get("repo_id"), prior.get("revision")) != (settings["repo_id"], settings["revision"]) or
                    prior.get("selected_episode_ids") != settings["episode_ids"]):
                raise ExternalDataError("original report repo/revision/episode plan differs from current pinned source")
            _finite_close(prior.get("source_fps"), float(info["fps"]), "original source fps")
            prior_hashes = prior.get("source_file_sha256", {})
            if not required_hashes.issubset(prior_hashes) or any(
                    name not in verified or digest != verified[name]["sha256"] for name, digest in prior_hashes.items()):
                raise ExternalDataError("original report source file hashes differ from current verified pinned files")
            if (prior.get("training_started") is not False or prior.get("training_eligible") is not False or
                    prior.get("current_adapter_ready") is not False):
                raise ExternalDataError("original audit contains unsupported training or current-adapter claims")
            episode_rows = prior.get("episodes", [])
            if [row.get("episode_id") for row in episode_rows] != settings["episode_ids"]:
                raise ExternalDataError("original report episode rows differ from the unchanged sample plan")
            verified_npz, actual_passed = [], 0
            pending = []
            for episode in episode_rows:
                episode_id = episode["episode_id"]
                planned = plan[episode_id]
                if (episode.get("status") != "passed" or episode.get("errors") or episode.get("row_count") != planned["row_count"] or
                        episode.get("source_data_path") != planned["source_data_path"]):
                    raise ExternalDataError("original table audit did not pass the pinned selected episode")
                verified_npz.append(_verify_saved_npz(episode, original_run, arrays[episode_id], info))
                videos = episode.get("videos", [])
                if len(videos) != 4 or {video.get("camera_key") for video in videos} != set(SOURCE_CAMERA_KEYS):
                    raise ExternalDataError("original report does not preserve all four planned cameras")
                for video in videos:
                    camera = video["camera_key"]
                    candidate = _verified_video_triplet(video, planned["cameras"][camera], original_run=original_run, settings=settings)
                    if candidate is not None:
                        actual_passed += 1
                        candidate.update(original_report_path=str(report_path), original_report_sha256=entry["sha256"],
                                         original_run=str(original_run))
                        pending.append(((episode_id, camera), candidate))
            if actual_passed != prior.get("bounded_video_probes_passed") or prior.get("bounded_video_probes_total") != len(candidates):
                raise ExternalDataError("original reported camera counts differ from its verified complete probes")
            for key, candidate in pending:
                candidates[key].append(candidate)
            entry.update(usable=True, source_hashes_reverified_against_current_receipt=True,
                         verified_table_episodes=len(verified_npz), verified_source_npz=verified_npz,
                         verified_complete_camera_triplets=actual_passed)
        except (ExternalDataError, OSError, KeyError, ValueError, TypeError, ImportError) as exc:
            entry["error"] = "{}: {}".format(type(exc).__name__, str(exc))
            errors.append(entry["error"])
        original_reports.append(entry)
    joined = []
    for (episode_id, camera), choices in candidates.items():
        if not choices:
            errors.append("Missing complete original camera triplet: episode {} {}".format(episode_id, camera))
            continue
        selected = choices[0]
        signature = [(sample["rgb_sha256"], sample["decoded_pts_ticks"], sample["decoded_source_pts_s"]) for sample in selected["samples"]]
        if any([(sample["rgb_sha256"], sample["decoded_pts_ticks"], sample["decoded_source_pts_s"]) for sample in other["samples"]] != signature for other in choices[1:]):
            errors.append("Conflicting verified RGB/PTS for episode {} {}".format(episode_id, camera))
            continue
        joined.append({"episode_id": episode_id, "camera_key": camera, **selected,
                       "matching_complete_original_reports": [choice["original_report_path"] for choice in choices],
                       "selection_rule": "first complete verified triplet in predeclared report path order"})
    result = {"schema_version": 1, "stage": "external-reconcile", "status": "passed" if not errors else "failed",
              "repo_id": settings["repo_id"], "revision": settings["revision"], "selected_episode_ids": settings["episode_ids"],
              "source_fps": info["fps"], "source_receipt_verified": True,
              "source_provenance_sha256": sha256_file(source / "provenance.json"), "verified_source_files": verified,
              "original_reports": original_reports, "consolidated_camera_triplets": joined,
              "consolidated_camera_triplets_passed": len(joined), "consolidated_camera_triplets_total": len(candidates),
              "consolidated_frame_samples_passed": sum(len(camera["samples"]) for camera in joined),
              "consolidated_frame_samples_total": len(candidates) * 3, "errors": errors,
              "single_run_full_camera_success_claimed": False, "network_requests_performed": False,
              "original_reports_modified": False, "current_adapter_ready": False,
              "sensor_clock_synchronization_proven": False, "training_eligible": False, "training_started": False,
              "full_video_content_hashes_claimed": False,
              "scope": "Cumulative matching camera-format/PTS/RGB evidence from declared immutable audit runs; no physics, adapter promotion or hardware-clock certification."}
    write_new_json(Path(run_dir) / "reports/external_reconciliation.json", result)
    return result
