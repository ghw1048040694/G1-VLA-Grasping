"""Immutable train/validation diagnostics for the recorded GraspSquare protocol.

Lag comparisons describe the recorded arrays.  They neither identify a hardware
clock nor authorize shifting the original labels.  Test payloads and metrics are
deliberately outside this stage's read scope.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import traceback
from fractions import Fraction
from pathlib import Path

import numpy as np

from .artifacts import sha256_file
from .external_data import SOURCE_JOINT_NAMES, _episode_rows, _feature_names, _formatted_path, _parquet_columns
from .external_sources import verify_download_receipt
from .grasp_data import GraspSquareDataset, load_grasp_manifest, _inside, _verify_artifact


class NativeDiagnosticsError(ValueError):
    """A diagnostic input differs from its immutable, source-native identity."""


ARRAY_COLUMNS = {
    "observation.state": "source_state", "action": "source_action",
    "timestamp": "source_timestamp_s", "frame_index": "source_frame_index",
    "episode_index": "source_episode_index", "index": "source_global_index",
    "task_index": "source_task_index",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _resolve(config, value):
    root = Path(config["paths"]["root"]).resolve()
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise NativeDiagnosticsError("diagnostic paths must stay in the declared project root")
    return path.resolve()


def _array_digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "maximum": None}
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
            "maximum": float(values.max())}


def _correlation(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    left, right = left - left.mean(axis=0), right - right.mean(axis=0)
    denominator = np.sqrt(np.square(left).sum(axis=0) * np.square(right).sum(axis=0))
    numerator = (left * right).sum(axis=0)
    return [float(numerator[index] / value) if value > 0 else None
            for index, value in enumerate(denominator)]


def angle_metrics(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 28:
        raise NativeDiagnosticsError("angle comparisons need matching row/28 arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise NativeDiagnosticsError("angle comparisons cannot contain nonfinite values")
    error = prediction.astype(np.float64) - target.astype(np.float64)
    result = {"row_count": len(error), "scalar_count": int(error.size), "units": "source_recorded_radian"}
    if not len(error):
        return dict(result, all_28=None, arms_14=None, hands_14=None, mae_by_joint=None)
    for name, selected in (("all_28", error), ("arms_14", error[:, :14]), ("hands_14", error[:, 14:])):
        result[name] = {"mae": float(np.abs(selected).mean()), "rmse": float(np.sqrt(np.square(selected).mean()))}
    result["mae_by_joint"] = np.abs(error).mean(axis=0).tolist()
    return result


def lag_diagnostics(episodes, lag_frames):
    """Use the identical interior anchor population at every declared lag."""
    lags = list(lag_frames)
    if lags != [-3, -2, -1, 0, 1, 2, 3]:
        raise NativeDiagnosticsError("this diagnostic predeclares the full -3 through +3 lag grid")
    result = {"definition": "action[t + lag] compared with state[t]; positive lag means a later recorded action",
              "boundary_policy": "identical interior anchors at every lag; no cross-episode pairing",
              "labels_modified": False, "hardware_timing_identified": False, "by_lag": {}}
    for lag in lags:
        states, actions, state_velocity, action_velocity = [], [], [], []
        for episode in episodes:
            state, action = episode["arrays"]["source_state"], episode["arrays"]["source_action"]
            # t >= 4 also permits the backward velocity at every negative lag.
            anchors = np.arange(4, len(state) - 3)
            if not len(anchors):
                raise NativeDiagnosticsError("lag diagnostics require at least eight source rows per episode")
            states.append(state[anchors]); actions.append(action[anchors + lag])
            state_velocity.append((state[anchors] - state[anchors - 1]) * 30.0)
            action_velocity.append((action[anchors + lag] - action[anchors + lag - 1]) * 30.0)
        states, actions = np.concatenate(states), np.concatenate(actions)
        result["by_lag"][str(lag)] = {**angle_metrics(states, actions),
            "position_correlation_by_joint": _correlation(states, actions),
            "velocity_correlation_by_joint": _correlation(np.concatenate(state_velocity), np.concatenate(action_velocity))}
    result["minimum_mae_lag_diagnostic_only"] = min(lags, key=lambda lag: result["by_lag"][str(lag)]["all_28"]["mae"])
    result["interpretation_limit"] = "Smooth recorded trajectories and controller tracking can produce a lag preference without establishing label misalignment or hardware time."
    return result


def horizon_diagnostics(episodes, action_mean, chunk_steps, stationary_speed, moving_speed):
    """Compare short-horizon predictability with fixed, disjoint speed strata."""
    if not 0 <= stationary_speed < moving_speed or chunk_steps < 1:
        raise NativeDiagnosticsError("speed thresholds must be ordered and the horizon positive")
    all_states, all_actions, all_speeds = [], [], []
    for episode in episodes:
        state, action = episode["arrays"]["source_state"], episode["arrays"]["source_action"]
        count = len(state) - chunk_steps + 1
        speed = np.empty(len(state), dtype=np.float64)
        speed[0] = np.nan  # No earlier row exists; do not fabricate a zero velocity.
        speed[1:] = np.max(np.abs(np.diff(state.astype(np.float64), axis=0)) * 30.0, axis=1)
        all_states.append(state[:count]); all_speeds.append(speed[:count])
        all_actions.append(np.stack([action[h:h + count] for h in range(chunk_steps)], axis=1))
    states, actions, speeds = np.concatenate(all_states), np.concatenate(all_actions), np.concatenate(all_speeds)
    masks = {"all": np.ones(len(states), dtype=bool),
             "stationary": speeds <= stationary_speed, "moving": speeds >= moving_speed,
             "intermediate": (speeds > stationary_speed) & (speeds < moving_speed),
             "unknown_first_row": ~np.isfinite(speeds)}
    result = {"anchor_count": len(states), "all_horizons_share_anchor_population": True,
              "speed_definition": "max over all 28 joints of abs(state[t] - state[t-1]) * 30; first row unknown",
              "stationary_max_speed_rad_s": stationary_speed, "moving_min_speed_rad_s": moving_speed,
              "speed_summary_rad_s": _summary(speeds[np.isfinite(speeds)]),
              "stratum_anchor_counts": {key: int(mask.sum()) for key, mask in masks.items()}, "by_horizon": []}
    for horizon in range(chunk_steps):
        target = actions[:, horizon, :]
        entry = {"future_row_offset": horizon, "future_time_s": horizon / 30.0, "strata": {}}
        for key, mask in masks.items():
            reference = target[mask]
            entry["strata"][key] = {"anchor_count": int(mask.sum()),
                "state_hold": angle_metrics(states[mask], reference),
                "train_mean": angle_metrics(np.broadcast_to(action_mean, reference.shape), reference)}
        result["by_horizon"].append(entry)
    return result


def replay_training_coverage(episodes, training_report, seed, batch_size, chunk_steps, loss_window):
    """Reconstruct the stage4 NumPy draws without claiming an actual saved trace."""
    steps = training_report.get("steps", [])
    count = training_report.get("optimizer_steps")
    if (isinstance(count, bool) or not isinstance(count, int) or count < 1 or len(steps) != count or
            [item.get("step") for item in steps] != list(range(1, count + 1))):
        raise NativeDiagnosticsError("training receipt must have one contiguous finite loss row per optimizer step")
    samples = [(item["episode_id"], frame) for item in episodes
               for frame in range(item["row_count"] - chunk_steps + 1)]
    if not samples or batch_size < 1 or loss_window < 1:
        raise NativeDiagnosticsError("coverage needs train anchors and positive batch/window sizes")
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.integers(0, len(samples), size=batch_size) for _ in range(count)])
    per_episode = {}
    for episode in episodes:
        episode_id, rows = episode["episode_id"], episode["row_count"]
        visits = np.zeros(rows - chunk_steps + 1, dtype=np.int64)
        labels = np.zeros(rows, dtype=np.int64)
        for selected in draws:
            selected_episode, frame = samples[int(selected)]
            if selected_episode == episode_id:
                visits[frame] += 1; labels[frame:frame + chunk_steps] += 1
        unique_frames = np.flatnonzero(visits)
        bins = np.linspace(0, len(visits), 5, dtype=np.int64)
        per_episode[str(episode_id)] = {"anchor_count": len(visits), "draw_count": int(visits.sum()),
            "unique_anchor_count": int((visits > 0).sum()), "unique_anchor_fraction": float((visits > 0).mean()),
            "unique_anchor_frame_indices": unique_frames.tolist(), "anchor_visit_count_by_frame": visits.tolist(),
            "action_label_row_coverage_count": int((labels > 0).sum()),
            "action_label_row_coverage_fraction": float((labels > 0).mean()),
            "action_label_visits_by_source_frame": labels.tolist(),
            "quarter_frame_bin_edges": bins.tolist(),
            "quarter_draw_counts": [int(visits[bins[i]:bins[i + 1]].sum()) for i in range(4)]}
    windows = {}
    for name, rows in (("first", steps[:loss_window]), ("last", steps[-loss_window:])):
        values = {"total_loss": [row["loss"] for row in rows]}
        for key in ("l1_loss", "kld_loss"):
            values[key] = [row["loss_components"][key] for row in rows]
        if any(not np.isfinite(value).all() for value in values.values()):
            raise NativeDiagnosticsError("training loss evidence contains nonfinite components")
        windows[name] = {"start_step": rows[0]["step"], "end_step": rows[-1]["step"],
                         **{key: _summary(value) for key, value in values.items()}}
    validation = [{"step": row["step"], "policy_mae": row["metrics"]["policy"]["all_28"]["mae"],
                   "state_hold_mae": row["metrics"]["state_hold"]["all_28"]["mae"],
                   "train_mean_mae": row["metrics"]["train_mean"]["all_28"]["mae"]}
                  for row in training_report.get("validation", [])]
    return {"seed": seed, "batch_size": batch_size, "optimizer_steps": count,
            "sampling": "numpy.random.default_rng(seed).integers(0, N, size=batch_size) once per step; with replacement",
            "evidence_kind": "deterministic reconstruction from saved seed/config/code, not a historically saved anchor trace",
            "training_anchor_count": len(samples), "anchor_draw_count": len(draws),
            "unique_anchor_count": len(np.unique(draws)), "unique_anchor_fraction": len(np.unique(draws)) / len(samples),
            "equivalent_anchor_epochs": len(draws) / len(samples),
            "sampled_global_anchor_indices_sha256": _array_digest(draws.astype(np.int64)),
            "per_episode": per_episode, "loss_windows": windows, "validation_curve": validation,
            "selected_step": training_report.get("selected_step"),
            "selection_reached_step_budget_boundary": training_report.get("selected_step") == count,
            "training_sufficient_proven": False,
            "loss_interpretation": "Training L1 uses sampled action-conditioned VAE posterior; inference uses zero latent. Total loss includes kl_weight times KL and is not action MAE."}


def _verify_episode(root, item, chunk_steps, tolerance):
    array_path, receipt_path = (_verify_artifact(root, item[key]) for key in ("arrays", "decoded_receipt"))
    rgb_path = _verify_artifact(root, item["rgb"])
    with np.load(array_path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in (*ARRAY_COLUMNS.values(), "decoded_source_pts_s", "decoded_source_pts_ticks",
                                                   "source_state_joint_names", "source_action_joint_names")}
    count = item["row_count"]
    for key in ("source_state", "source_action"):
        if arrays[key].shape != (count, 28) or arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise NativeDiagnosticsError("source state/action must retain all finite float32/28 rows")
    for key in ("source_state_joint_names", "source_action_joint_names"):
        if tuple(arrays[key].tolist()) != SOURCE_JOINT_NAMES:
            raise NativeDiagnosticsError("source named joint order differs from the recorded 28D protocol")
    times = arrays["source_timestamp_s"]
    if (times.shape != (count,) or times.dtype != np.float32 or not np.isfinite(times).all() or
            not np.all(np.diff(times.astype(np.float64)) > 0) or
            np.max(np.abs(times.astype(np.float64) - np.arange(count) / 30.0)) > tolerance):
        raise NativeDiagnosticsError("source row timestamps must retain contiguous original float32 30 Hz rows")
    for key in ("source_frame_index", "source_episode_index", "source_global_index", "source_task_index"):
        if arrays[key].shape != (count,) or arrays[key].dtype.kind not in "iu":
            raise NativeDiagnosticsError("source identities must retain integer row vectors")
    if (not np.array_equal(arrays["source_frame_index"], np.arange(count)) or
            not np.all(arrays["source_episode_index"] == item["episode_id"]) or
            not np.array_equal(np.diff(arrays["source_global_index"]), np.ones(count - 1, dtype=np.int64)) or
            item["sample_count"] != count - chunk_steps + 1):
        raise NativeDiagnosticsError("complete source row identities or chunk boundaries changed")
    receipt = _read(receipt_path)
    frames = receipt.get("frames", [])
    if (receipt.get("passed") is not True or receipt.get("episode_id") != item["episode_id"] or
            len(frames) != count or receipt.get("paired_frame_count") != count or
            receipt.get("frame_reuse_or_padding") is not False or receipt.get("resampled_in_time") is not False or
            receipt.get("sensor_exposure_clock_verified") is not False or
            receipt.get("stream", {}).get("fps") != 30.0 or receipt.get("stream", {}).get("codec_id") != 225):
        raise NativeDiagnosticsError("source decoded receipt does not certify complete unchanged recorded 30 Hz pairing")
    pts, ticks, errors = [], [], []
    origin = receipt["source_video_from_timestamp_s"]
    for index, frame in enumerate(frames):
        actual, tick = frame["decoded_source_pts_s"], frame["decoded_pts_ticks"]
        tick_time = float(tick * Fraction(frame["decoded_time_base"]))
        query = origin + float(times[index])
        error = abs(actual - query)
        if (frame["frame_index"] != index or abs(frame["query_source_pts_s"] - query) > 1e-12 or
                not math.isfinite(actual) or abs(actual - tick_time) > 1e-12 or error > tolerance or
                abs(frame["absolute_pts_error_s"] - error) > 1e-12 or
                any(re.fullmatch(r"[0-9a-f]{64}", frame.get(key, "")) is None
                    for key in ("original_rgb_sha256", "model_rgb_sha256"))):
            raise NativeDiagnosticsError("row/PTS receipt values differ from their declared source row")
        pts.append(actual); ticks.append(tick); errors.append(error)
    pts, ticks = np.asarray(pts), np.asarray(ticks)
    if (not np.array_equal(pts, arrays["decoded_source_pts_s"]) or
            not np.array_equal(ticks, arrays["decoded_source_pts_ticks"]) or
            not np.all(np.diff(ticks) > 0) or not np.all(np.diff(pts) > 0) or
            np.max(np.abs(np.diff(pts) - 1 / 30.0)) > tolerance):
        raise NativeDiagnosticsError("source PTS duplicates, gaps or derived NPZ mismatch")
    return {**item, "arrays": arrays, "decoded": receipt}, {"episode_id": item["episode_id"], "split": item["split"],
        "row_count": count, "sample_count": item["sample_count"],
        "source_npz_sha256": sha256_file(array_path), "decoded_receipt_sha256": sha256_file(receipt_path),
        "model_rgb_file_sha256": sha256_file(rgb_path), "complete_model_rgb_file_identity_verified": True,
        "state_dtype": str(arrays["source_state"].dtype), "action_dtype": str(arrays["source_action"].dtype),
        "row_timestamp_dtype": str(times.dtype), "state_action_joint_names_verified": True,
        "row_30hz_maximum_error_s": float(np.max(np.abs(times.astype(np.float64) - np.arange(count) / 30.0))),
        "pts_maximum_pairing_error_s": max(errors), "unique_row_frame_count": len(np.unique(arrays["source_frame_index"])),
        "unique_pts_count": len(np.unique(ticks)), "sensor_exposure_clock_verified": False,
        "motor_application_clock_verified": False}


def _verify_statistics(episodes, statistics):
    train = [episode for episode in episodes if episode["split"] == "train"]
    if (statistics.get("training_episode_ids") != [episode["episode_id"] for episode in train] or
            statistics.get("validation_or_test_used") is not False or statistics.get("statistics_include_held_out") is not False):
        raise NativeDiagnosticsError("statistics are not exclusively the predeclared training episodes")
    result = {}
    for name, column in (("observation.state", "source_state"), ("action", "source_action")):
        values = np.concatenate([episode["arrays"][column] for episode in train]).astype(np.float64)
        mean = sum(episode["arrays"][column].astype(np.float64).sum(axis=0) for episode in train) / len(values)
        squares = sum(np.square(episode["arrays"][column].astype(np.float64)).sum(axis=0) for episode in train)
        std = np.maximum(np.sqrt(np.maximum(squares / len(values) - np.square(mean), 0)), .001)
        saved = statistics["features"][name]
        if saved["count"] != len(values) or any(not np.allclose(np.asarray(saved[key]), actual, rtol=1e-12, atol=1e-12)
                                              for key, actual in (("mean", mean), ("std", std))):
            raise NativeDiagnosticsError("saved normalization differs from actual train-only source rows")
        raw = values.astype(np.float32)
        normalized = (raw - np.asarray(saved["mean"], dtype=np.float32)) / np.asarray(saved["std"], dtype=np.float32)
        restored = normalized * np.asarray(saved["std"], dtype=np.float32) + np.asarray(saved["mean"], dtype=np.float32)
        result[name] = {"count": len(values), "saved_train_only_moments_match": True,
                        "float32_normalize_inverse_max_abs_error": float(np.max(np.abs(restored.astype(np.float64) - values))),
                        "std_clamped_joint_indices": list(saved.get("clamped_dimension_indices", []))}
    return result


def _label_accessor_checks(root, manifest, episodes):
    result = []
    datasets = {split: GraspSquareDataset(root, split, manifest=manifest, verify=False) for split in ("train", "validation")}
    for episode in episodes:
        dataset = datasets[episode["split"]]
        rgb_path = _verify_artifact(root, episode["rgb"])
        rgb = np.load(rgb_path, mmap_mode="r", allow_pickle=False)
        if rgb.dtype != np.uint8 or rgb.shape != (episode["row_count"], *manifest["image_shape_hwc"]):
            raise NativeDiagnosticsError("verified model RGB does not retain the declared full uint8 frame population")
        selected = sorted(set((0, episode["sample_count"] // 2, episode["sample_count"] - 1)))
        index_by_identity = {identity: index for index, identity in enumerate(dataset.samples)}
        for frame in selected:
            sample = dataset[index_by_identity[(episode["episode_id"], frame)]]
            original = episode["arrays"]
            expected = original["source_action"][frame:frame + manifest["chunk_steps"]]
            expected_rgb = np.ascontiguousarray(rgb[frame].transpose(2, 0, 1), dtype=np.float32) / 255.0
            if (not np.array_equal(sample["observation.state"], original["source_state"][frame]) or
                    not np.array_equal(sample["action"], expected) or sample["action"].dtype != np.float32 or
                    sample["action_is_pad"].any()):
                raise NativeDiagnosticsError("actual dataset accessor shifts or changes the original row/future-chunk labels")
            if sample[manifest["camera_key"]].dtype != np.float32 or not np.array_equal(sample[manifest["camera_key"]], expected_rgb):
                raise NativeDiagnosticsError("actual dataset accessor image differs from verified RGB[t] transformed to float32 CHW/255")
            result.append({"episode_id": episode["episode_id"], "anchor_frame_index": frame,
                "first_action_source_frame_index": frame, "last_action_source_frame_index": frame + manifest["chunk_steps"] - 1,
                "exact_state_row_and_action_chunk_equal": True,
                "exact_image_source_frame_and_float32_chw_transform_equal": True,
                "model_rgb_frame_sha256": _array_digest(rgb[frame]),
                "chunk_float32_bytes_sha256": _array_digest(sample["action"])})
        del rgb
    return {"definition": "state[t], RGB[t], actions[t:t+chunk_steps]; row t is included and no +1 is applied",
            "horizon_offsets": list(range(manifest["chunk_steps"])), "actual_dataset_accessor_checks": result,
            "checked_anchor_count": len(result), "anchor_selection": "first, middle and final complete anchor of each train/validation episode",
            "dataset_module_sha256": sha256_file(Path(__file__).with_name("grasp_data.py")),
            "labels_shifted": False, "actions_clipped": False, "padding_used": False}


def _raw_parquet_checks(config, manifest, episodes):
    source = _resolve(config, config["external_data"]["local_dir"])
    metadata_names = [name for name in manifest["verified_source_files"] if name.startswith("meta/episodes/")]
    names = sorted({"meta/info.json", *metadata_names, *(episode["data_relative_path"] for episode in episodes)})
    verified = verify_download_receipt(source, manifest["source_repo_id"], manifest["source_revision"], names)
    if any(verified[name]["sha256"] != manifest["verified_source_files"][name]["sha256"] for name in names):
        raise NativeDiagnosticsError("current original parquet hashes differ from the dataset build's immutable source identity")
    info = _read(source / "meta/info.json")
    if info.get("fps") != 30.0 or info.get("codebase_version") != "v3.0" or any(
            _feature_names(info.get("features", {}).get(name, {}), name) != SOURCE_JOINT_NAMES
            for name in ("observation.state", "action")):
        raise NativeDiagnosticsError("original source metadata joint order or cadence differs from the native protocol")
    results = []
    for episode in episodes:
        metadata, _ = _episode_rows(source, info, episode["episode_id"], verified)
        decoded = episode["decoded"]
        prefix = "videos/" + manifest["camera_key"]
        if (metadata["length"] != episode["row_count"] or
                metadata["dataset_from_index"] != int(episode["arrays"]["source_global_index"][0]) or
                metadata["dataset_to_index"] != int(episode["arrays"]["source_global_index"][-1]) + 1 or
                _formatted_path(info, "data_path", metadata) != episode["data_relative_path"] or
                _formatted_path(info, "video_path", metadata, manifest["camera_key"]) != decoded["source_video_relative_path"] or
                abs(float(metadata[prefix + "/from_timestamp"]) - decoded["source_video_from_timestamp_s"]) > 1e-12 or
                abs(float(metadata[prefix + "/to_timestamp"]) - decoded["source_video_to_timestamp_s"]) > 1e-12):
            raise NativeDiagnosticsError("original source episode/video boundaries differ from derived rows and PTS origins")
        columns = _parquet_columns(_inside(source, episode["data_relative_path"]), episode["episode_id"])
        for name, derived in ARRAY_COLUMNS.items():
            if columns[name].dtype != episode["arrays"][derived].dtype or not np.array_equal(columns[name], episode["arrays"][derived]):
                raise NativeDiagnosticsError("derived source rows differ by value or dtype from original filtered Arrow columns")
        results.append({"episode_id": episode["episode_id"], "selected_row_count": episode["row_count"],
                        "columns_exact_value_and_dtype_equal": list(ARRAY_COLUMNS),
                        "source_parquet_relative_path": episode["data_relative_path"],
                        "original_metadata_global_and_video_boundaries_equal": True})
    return {"independently_rechecked": True, "verified_source_files": verified, "selected_episodes": results,
            "scope": "whole shared source files are hashed, only declared train/validation episode rows are materialized and analyzed",
            "test_episode_rows_materialized": False}


def run_native_diagnostics(config, run_dir):
    """Audit original rows, source labels and actual prior training coverage."""
    path = Path(run_dir) / "reports/native_diagnostics.json"
    if path.exists():
        raise NativeDiagnosticsError("diagnostic evidence already exists and cannot be overwritten")
    started = time.perf_counter()
    report = {"schema_version": 1, "passed": False, "status": "failed", "training_started": False,
              "optimizer_steps": 0, "test_payloads_read": False, "test_metrics_read": False,
              "closed_loop_control": False, "labels_modified": False}
    try:
        settings = config["native_diagnostics"]
        if (settings["lag_frames"] != [-3, -2, -1, 0, 1, 2, 3] or
                not 0 <= settings["stationary_speed_rad_s"] < settings["moving_speed_rad_s"] or
                settings["loss_window_steps"] < 1):
            raise NativeDiagnosticsError("diagnostic lag, speed or loss-window declaration is invalid")
        manifest_path = _resolve(config, config["grasp_act"]["dataset_manifest_path"])
        root = manifest_path.parent
        manifest = load_grasp_manifest(root, expected_settings=config["source_native"],
                                       expected_source=config["external_data"], verify_files=False)
        selected = [item for item in manifest["episodes"] if item["split"] in ("train", "validation")]
        if not selected or {item["split"] for item in selected} != {"train", "validation"}:
            raise NativeDiagnosticsError("both train and validation are required; test cannot enter diagnostics")
        report["bindings"] = {"manifest_sha256": sha256_file(manifest_path), "settings": settings,
                              "module_sha256": sha256_file(__file__)}
        episodes, audit = [], []
        for item in selected:
            episode, receipt = _verify_episode(root, item, manifest["chunk_steps"], config["source_native"]["timestamp_tolerance_s"])
            episodes.append(episode); audit.append(receipt)
        globals_ = np.concatenate([episode["arrays"]["source_global_index"] for episode in episodes])
        if len(np.unique(globals_)) != len(globals_):
            raise NativeDiagnosticsError("different selected episodes share global row identities")
        report["scope"] = {"split_names": ["train", "validation"], "episode_ids": [item["episode_id"] for item in episodes],
                           "row_count": len(globals_), "test_artifact_paths_opened": [],
                           "test_metadata_read_for_exclusion_only": True}
        report["source_rows_and_pts"] = audit
        statistics_path = _verify_artifact(root, manifest["statistics"])
        statistics = _read(statistics_path)
        report["bindings"]["statistics_sha256"] = sha256_file(statistics_path)
        report["normalization"] = _verify_statistics(episodes, statistics)
        report["label_contract"] = _label_accessor_checks(root, manifest, episodes)
        report["raw_parquet_parity"] = (_raw_parquet_checks(config, manifest, episodes) if settings["verify_raw_source"] else
                                         {"independently_rechecked": False, "reason": "explicitly disabled; builder receipts alone do not constitute this independent source check"})
        report["by_split"] = {}
        for split in ("train", "validation"):
            subset = [episode for episode in episodes if episode["split"] == split]
            report["by_split"][split] = {"lag": lag_diagnostics(subset, settings["lag_frames"]),
                "horizon_and_motion": horizon_diagnostics(subset, np.asarray(statistics["features"]["action"]["mean"]),
                    manifest["chunk_steps"], settings["stationary_speed_rad_s"], settings["moving_speed_rad_s"])}
        training_path = _resolve(config, settings["training_report_path"])
        training_report = _read(training_path)
        if (training_report.get("training_passed") is not True or training_report.get("test_used_for_selection") is not False or
                training_report.get("bindings", {}).get("manifest_sha256") != sha256_file(manifest_path) or
                training_report.get("bindings", {}).get("statistics_sha256") != sha256_file(statistics_path)):
            raise NativeDiagnosticsError("prior training receipt is failed or bound to different data/statistics")
        checkpoint_config_path = _resolve(config, str(Path(training_report["checkpoint"]["path"]) / "experiment_config.json"))
        checkpoint_config = _read(checkpoint_config_path)
        expected_config_sha = training_report["checkpoint"]["files"]["experiment_config.json"]["sha256"]
        actual_config_identity = hashlib.sha256(json.dumps(checkpoint_config, sort_keys=True, separators=(",", ":"),
                                                        allow_nan=False).encode()).hexdigest()
        if (sha256_file(checkpoint_config_path) != expected_config_sha or
                actual_config_identity != training_report["bindings"]["experiment_config_sha256"]):
            raise NativeDiagnosticsError("prior RNG seed and batch configuration differ from the training receipt")
        report["bindings"]["training_report_sha256"] = sha256_file(training_path)
        report["bindings"]["prior_experiment_config_sha256"] = actual_config_identity
        report["training_coverage"] = replay_training_coverage([episode for episode in episodes if episode["split"] == "train"],
            training_report, checkpoint_config["train"]["seed"], checkpoint_config["train"]["batch_size"],
            manifest["chunk_steps"], settings["loss_window_steps"])
        report["status"], report["passed"] = "passed_train_validation_recorded_protocol_diagnostics", True
        report["limitations"] = ["Original row alignment is verified; independent sensor exposure and motor application clocks remain unavailable.",
            "A preferred diagnostic lag does not establish a label bug and no labels were changed.",
            "Low anchor exposure and decreasing losses are measured; insufficient optimization is a hypothesis requiring controlled training comparisons.",
            "These recorded-action diagnostics do not establish physical grasp success, held-out test performance or a 31D simulator adapter."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        report["total_elapsed_s"] = time.perf_counter() - started
        _write(path, report)
    return report
