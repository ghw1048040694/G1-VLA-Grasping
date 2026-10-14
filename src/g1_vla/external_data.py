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
