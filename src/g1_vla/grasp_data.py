"""Verified, source-native GraspSquare rows and RGB for offline imitation.

The 28 columns and recorded row alignment remain the official protocol.  This
backend neither manufactures a waist nor certifies camera exposure or motor
application clocks.  Whole video files are pinned evidence; a declared resize
produces model inputs without relabelling a real head eye as a simulator camera.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifacts import sha256_file
from .external_data import (
    OFFICIAL_REPO, REQUIRED_COLUMNS, SOURCE_CAMERA_KEYS, SOURCE_JOINT_NAMES,
    _episode_rows, _feature_names, _formatted_path, _parquet_columns, _parquet_rows,
)
from .external_sources import DATA_FILES, verify_download_receipt
from .runtime.experiment import canonical_hash, write_new_json


class GraspDataError(ValueError):
    """Source identity, independent grouping or complete paired frames failed."""


BACKEND = "grasp_square_native_v1"
OFFICIAL_REVISION = "c2b44ca4c5f6329d75143613f9f1eada0cf94c62"
CAMERA = "observation.images.cam_left_high"
TRANSFORM = "bilinear_resize_rgb_v1"
SPLITS = ("train", "validation", "test")


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_symlink():
        raise GraspDataError("source and derived directories cannot be symlinks")
    return (path if path.is_absolute() else root / path).resolve()


def _inside(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise GraspDataError("artifact identities must be relative paths")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
        raise GraspDataError("artifact path escapes its directory or is a symlink")
    return path


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GraspDataError("{} must be an integer at least {}".format(name, minimum))
    return value


def _ids(value: Any, name: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise GraspDataError(name + " must be a nonempty episode list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise GraspDataError(name + " requires nonnegative integer episode identities")
    if len(value) != len(set(value)):
        raise GraspDataError(name + " contains duplicate episodes")
    return list(value)


def validate_native_settings(settings: Mapping[str, Any], prior_episode_ids=()) -> dict:
    """Require a predeclared recording/file group split, never a frame split."""
    if settings.get("backend") != BACKEND or settings.get("camera_key") != CAMERA:
        raise GraspDataError("this backend admits only native GraspSquare and its original left head eye")
    if settings.get("image_transform") != TRANSFORM:
        raise GraspDataError("model RGB transform must be explicitly bilinear_resize_rgb_v1")
    if settings.get("split_unit") != "source_video_file_group":
        raise GraspDataError("split unit must be source_video_file_group")
    ids = _ids(settings.get("episode_ids"), "episode_ids")
    if set(ids).intersection(prior_episode_ids):
        raise GraspDataError("prior diagnostic episodes cannot enter this new train/held-out declaration")
    split = settings.get("split_episode_ids")
    if not isinstance(split, dict) or set(split) != set(SPLITS):
        raise GraspDataError("train, validation and test episode groups must all be explicitly declared")
    collected = []
    for key in SPLITS:
        collected.extend(_ids(split[key], key))
    if len(set(collected)) != len(collected) or set(collected) != set(ids):
        raise GraspDataError("split episodes overlap or differ from the predeclared selection")
    result = dict(settings)
    for key in ("chunk_steps", "image_width", "image_height", "max_episode_rows",
                "max_total_rows", "max_derived_bytes"):
        result[key] = _integer(settings.get(key), key)
    fps, tolerance = settings.get("source_fps"), settings.get("timestamp_tolerance_s")
    if (isinstance(fps, bool) or not isinstance(fps, (int, float)) or float(fps) != 30.0):
        raise GraspDataError("native source cadence must remain exactly 30 Hz")
    if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or
            not math.isfinite(tolerance) or not 0 < tolerance <= 1e-4):
        raise GraspDataError("timestamp tolerance must be positive and no greater than 1e-4 s")
    if result["chunk_steps"] > result["max_episode_rows"]:
        raise GraspDataError("chunk horizon exceeds the declared episode row bound")
    return result


def verify_file_groups(episodes: Mapping[int, Mapping[str, Any]], info: Mapping[str, Any],
                       split: Mapping[str, list[int]]) -> dict:
    """Check the storage identity of all four cameras, including unused eyes."""
    groups = {key: set() for key in SPLITS}
    for key in SPLITS:
        for episode_id in split[key]:
            for camera in SOURCE_CAMERA_KEYS:
                groups[key].add(_formatted_path(info, "video_path", episodes[episode_id], camera))
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            overlap = groups[left].intersection(groups[right])
            if overlap:
                raise GraspDataError("{} and {} share source video files: {}".format(left, right, sorted(overlap)))
    return {"split_unit": "source_video_file_group", "checked_camera_keys": list(SOURCE_CAMERA_KEYS),
            "source_video_paths_by_split": {key: sorted(value) for key, value in groups.items()},
            "all_camera_storage_files_disjoint": True, "capture_sessions_independent_verified": False,
            "layout_or_operator_independence_verified": False,
            "basis": "complete recorded episodes grouped by all four declared source video file identities"}


def validate_episode_rows(columns: Mapping[str, Any], metadata: Mapping[str, Any],
                          episode_id: int, settings: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Preserve Arrow float32 arrays; reject gaps, shifts and incomplete episodes."""
    arrays = {key: np.asarray(columns[key]) for key in REQUIRED_COLUMNS}
    count = len(arrays["timestamp"])
    if count < settings["chunk_steps"] or count > settings["max_episode_rows"]:
        raise GraspDataError("episode {} lacks a complete chunk or exceeds the row bound".format(episode_id))
    if int(metadata["length"]) != count or int(metadata["episode_index"]) != episode_id:
        raise GraspDataError("source episode metadata length or identity differs from actual rows")
    for key, values in arrays.items():
        if values.shape[0] != count or not np.isfinite(values).all():
            raise GraspDataError("source column {} has inconsistent rows or nonfinite values".format(key))
        if key in ("observation.state", "action"):
            if values.shape != (count, 28) or values.dtype != np.float32:
                raise GraspDataError("native state/action must retain full float32 28D source rows")
        elif key != "timestamp" and (values.ndim != 1 or values.dtype.kind not in "iu"):
            raise GraspDataError("source identity columns must be one-dimensional integer arrays")
    if arrays["timestamp"].ndim != 1 or arrays["timestamp"].dtype != np.float32:
        raise GraspDataError("source row timestamps must retain their one-dimensional float32 dtype")
    if not np.array_equal(arrays["frame_index"], np.arange(count)):
        raise GraspDataError("source episode frames are not the complete zero-based contiguous sequence")
    if not np.all(arrays["episode_index"] == episode_id):
        raise GraspDataError("source shard contains rows from another episode")
    start, end = int(metadata["dataset_from_index"]), int(metadata["dataset_to_index"])
    if end - start != count or not np.array_equal(arrays["index"], np.arange(start, end)):
        raise GraspDataError("source global indices differ from complete metadata episode boundaries")
    times = arrays["timestamp"].astype(np.float64)
    if not np.all(np.diff(times) > 0) or not np.all(np.abs(times - np.arange(count) / 30.0) <= settings["timestamp_tolerance_s"]):
        raise GraspDataError("source timestamps are discontinuous or differ from the original 30 Hz cadence")
    return arrays


def _file_record(path: Path, root: Path) -> dict:
    return {"relative_path": str(path.relative_to(root)), "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size}


def decode_episode_rgb(video_path: Path, query_times_s: np.ndarray, output_path: Path, *,
                       width: int, height: int, source_width: int, source_height: int,
                       fps: float, tolerance_s: float, av_module=None, image_module=None) -> dict:
    """Continuously pair every row to one distinct true RGB/PTS, without padding."""
    if av_module is None:
        import av as av_module
    if image_module is None:
        from PIL import Image as image_module
    targets = np.asarray(query_times_s, dtype=np.float64)
    if (targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all() or
            targets[0] < 0 or not np.all(np.diff(targets) > 0)):
        raise GraspDataError("every source row needs a strictly increasing finite video PTS target")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(str(output_path))
    frame_receipts, rgb_digest, resized_digest = [], hashlib.sha256(), hashlib.sha256()
    images = None
    with av_module.open(str(video_path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise GraspDataError("official source must contain exactly one video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        actual_fps = float(stream.average_rate) if stream.average_rate else None
        codec_id = int(stream.codec_context.codec.id)
        expected_codec_id = int(av_module.Codec("av1", "r").id)
        if (stream.width != source_width or stream.height != source_height or actual_fps is None or
                abs(actual_fps - fps) > 1e-6 or codec_id != expected_codec_id):
            raise GraspDataError("verified source video shape, cadence or encoded codec differs from official metadata")
        if stream.time_base is None or float(stream.time_base) <= 0:
            raise GraspDataError("source video has no valid stream time base")
        stream_info = {"width": stream.width, "height": stream.height, "fps": actual_fps,
                       "codec_id": codec_id, "expected_encoded_codec": "av1",
                       "decoder_name": stream.codec_context.name, "time_base": str(stream.time_base)}
        container.seek(int(targets[0] / float(stream.time_base)), stream=stream, backward=True, any_frame=False)
        images = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint8,
                                          shape=(len(targets), height, width, 3))
        cursor, previous_time, previous_ticks, preceding_frames = 0, None, None, 0
        try:
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None:
                    raise GraspDataError("decoded real RGB has no source PTS/time base")
                actual = float(frame.pts * frame.time_base)
                if not math.isfinite(actual):
                    raise GraspDataError("decoded source PTS is nonfinite")
                if actual < targets[cursor] - tolerance_s:
                    preceding_frames += 1
                    continue
                if abs(actual - targets[cursor]) > tolerance_s:
                    raise GraspDataError("a real source RGB frame is missing at recorded row {}".format(cursor))
                if previous_time is not None and (actual <= previous_time or int(frame.pts) <= previous_ticks or
                        abs(actual - previous_time - 1.0 / fps) > tolerance_s):
                    raise GraspDataError("source RGB PTS is duplicated, skipped or not contiguous at 30 Hz")
                rgb = frame.to_ndarray(format="rgb24")
                if rgb.shape != (source_height, source_width, 3) or rgb.dtype != np.uint8:
                    raise GraspDataError("decoded source RGB changed pixel shape or dtype")
                resized = np.asarray(image_module.fromarray(rgb, mode="RGB").resize(
                    (width, height), resample=image_module.Resampling.BILINEAR), dtype=np.uint8)
                if resized.shape != (height, width, 3):
                    raise GraspDataError("declared bilinear model image has an invalid shape")
                raw_bytes, resized_bytes = rgb.tobytes(), resized.tobytes()
                rgb_digest.update(raw_bytes)
                resized_digest.update(resized_bytes)
                images[cursor] = resized
                frame_receipts.append({"frame_index": cursor, "query_source_pts_s": float(targets[cursor]),
                    "decoded_source_pts_s": actual, "decoded_pts_ticks": int(frame.pts),
                    "decoded_time_base": str(frame.time_base), "absolute_pts_error_s": abs(actual - targets[cursor]),
                    "original_rgb_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "model_rgb_sha256": hashlib.sha256(resized_bytes).hexdigest()})
                previous_time, previous_ticks = actual, int(frame.pts)
                cursor += 1
                if cursor == len(targets):
                    break
            if cursor != len(targets):
                raise GraspDataError("source video ended before every row had a real matching RGB frame")
            images.flush()
        finally:
            if images is not None:
                images.flush()
                del images
    return {"schema_version": 1, "passed": True, "stream": stream_info, "paired_frame_count": len(frame_receipts),
            "preceding_keyframe_frames_decoded": preceding_frames, "frames": frame_receipts,
            "all_original_rgb_sha256": rgb_digest.hexdigest(), "all_model_rgb_sha256": resized_digest.hexdigest(),
            "maximum_absolute_pts_error_s": max(item["absolute_pts_error_s"] for item in frame_receipts),
            "rgb_is_source_real": True, "image_transform": TRANSFORM,
            "model_image_shape_hwc": [height, width, 3], "source_image_shape_hwc": [source_height, source_width, 3],
            "sensor_exposure_clock_verified": False, "resampled_in_time": False, "frame_reuse_or_padding": False}


def _moments(values: np.ndarray, sum_: np.ndarray, squares: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    numbers = np.asarray(values, dtype=np.float64)
    return sum_ + numbers.sum(axis=0), squares + np.square(numbers).sum(axis=0), len(numbers)


def train_only_statistics(root: Path, episodes: list[dict], camera_key: str, *, epsilon=1e-3) -> dict:
    """Streaming population statistics use only declared train episode rows."""
    accum = {key: [np.zeros(size), np.zeros(size), 0] for key, size in (("observation.state", 28), ("action", 28), (camera_key, 3))}
    train_ids = []
    for item in episodes:
        if item["split"] != "train":
            continue
        train_ids.append(item["episode_id"])
        with np.load(_inside(root, item["arrays"]["relative_path"]), allow_pickle=False) as source:
            for key, column in (("observation.state", "source_state"), ("action", "source_action")):
                acc = accum[key]
                new_sum, new_squares, count = _moments(source[column], acc[0], acc[1])
                accum[key] = [new_sum, new_squares, acc[2] + count]
        images = np.load(_inside(root, item["rgb"]["relative_path"]), mmap_mode="r", allow_pickle=False)
        for start in range(0, len(images), 16):
            pixels = np.asarray(images[start:start + 16], dtype=np.float64).reshape(-1, 3) / 255.0
            acc = accum[camera_key]
            new_sum, new_squares, count = _moments(pixels, acc[0], acc[1])
            accum[camera_key] = [new_sum, new_squares, acc[2] + count]
        del images
    features = {}
    for key, (sum_, squares, count) in accum.items():
        if count < 2:
            raise GraspDataError("training statistics need at least two real observations")
        mean = sum_ / count
        raw_std = np.sqrt(np.maximum(squares / count - np.square(mean), 0))
        std = np.maximum(raw_std, epsilon)
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise GraspDataError("train-only statistics are nonfinite")
        features[key] = {"mean": (mean.reshape(3, 1, 1) if key == camera_key else mean).tolist(),
                         "std": (std.reshape(3, 1, 1) if key == camera_key else std).tolist(),
                         "unclamped_std": (raw_std.reshape(3, 1, 1) if key == camera_key else raw_std).tolist(),
                         "count": int(count), "clamped_dimension_indices": np.flatnonzero(raw_std < epsilon).tolist()}
    return {"schema_version": 1, "training_episode_ids": train_ids, "validation_or_test_used": False,
            "features": features, "std_minimum": epsilon, "variance_estimator": "population_float64_streaming",
            "row_scope": "all original rows in train episodes; complete action chunks use the unchanged row labels",
            "rgb_value_range": [0.0, 1.0], "statistics_include_held_out": False}


def _derived_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def build_grasp_square_data(config: Mapping[str, Any], run_dir: str | Path) -> dict:
    """Build exclusively from complete pinned files, never alter source evidence."""
    root = Path(config["paths"]["root"]).resolve()
    external = config["external_data"]
    settings = validate_native_settings(config["source_native"], external["episode_ids"])
    if external.get("repo_id") != OFFICIAL_REPO or external.get("revision") != OFFICIAL_REVISION:
        raise GraspDataError("official source identity must be an exact pinned public revision")
    source, videos, build = (_resolve(root, external["local_dir"]), _resolve(root, settings["video_dir"]),
                             _resolve(root, settings["build_dir"]))
    if build.is_relative_to(source) or build.is_relative_to(videos) or source.is_relative_to(build) or videos.is_relative_to(build):
        raise GraspDataError("derived output must remain separate from both immutable source caches")
    if build.exists():
        raise FileExistsError("derived native dataset already exists: " + str(build))
    source_verified = verify_download_receipt(source, external["repo_id"], external["revision"], DATA_FILES)
    source_receipt_sha = sha256_file(source / "provenance.json")
    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0" or float(info.get("fps", 0)) != 30.0:
        raise GraspDataError("native backend requires the pinned source's complete LeRobot v3/30 Hz layout")
    features = info.get("features", {})
    if tuple(_feature_names(features.get("observation.state", {}), "observation.state")) != SOURCE_JOINT_NAMES or \
            tuple(_feature_names(features.get("action", {}), "action")) != SOURCE_JOINT_NAMES:
        raise GraspDataError("source-native state/action named joint order changed")
    if {key for key, value in features.items() if value.get("dtype") == "video"} != set(SOURCE_CAMERA_KEYS):
        raise GraspDataError("all four original source camera identities must remain declared")
    camera_info = features[CAMERA].get("info", {})
    if (camera_info.get("video.width"), camera_info.get("video.height"), camera_info.get("video.fps"),
            camera_info.get("video.codec")) != (640, 480, 30.0, "av1"):
        raise GraspDataError("original left head camera metadata differs from the pinned protocol")
    episodes = {episode_id: _episode_rows(source, info, episode_id, source_verified)[0]
                for episode_id in settings["episode_ids"]}
    groups = verify_file_groups(episodes, info, settings["split_episode_ids"])
    data_names = sorted({_formatted_path(info, "data_path", metadata) for metadata in episodes.values()})
    # Every read data shard must be bound to its pinned receipt, not only the
    # shared default path previously used for three diagnostic episodes.
    source_verified.update(verify_download_receipt(source, external["repo_id"], external["revision"], data_names))
    video_names = sorted({_formatted_path(info, "video_path", metadata, CAMERA) for metadata in episodes.values()})
    video_verified = verify_download_receipt(videos, external["repo_id"], external["revision"], video_names)
    video_receipt_sha = sha256_file(videos / "provenance.json")
    arrays = {}
    for episode_id, metadata in episodes.items():
        path = _inside(source, _formatted_path(info, "data_path", metadata))
        arrays[episode_id] = validate_episode_rows(_parquet_columns(path, episode_id), metadata, episode_id, settings)
    total_rows = sum(len(item["timestamp"]) for item in arrays.values())
    if total_rows > settings["max_total_rows"]:
        raise GraspDataError("native selected row total exceeds its declared bound")
    # Include row metadata and JSON frame receipts in a conservative preflight,
    # then enforce the actual sum again before committing an admitted manifest.
    planned_bytes = total_rows * (settings["image_width"] * settings["image_height"] * 3 + 2048) + 1048576
    if planned_bytes > settings["max_derived_bytes"]:
        raise GraspDataError("complete native RGB/rows exceed the declared derived byte budget")
    build.mkdir(parents=True, exist_ok=False)
    (build / "episodes").mkdir()
    built = []
    try:
        import PIL
        for episode_id in settings["episode_ids"]:
            metadata, columns = episodes[episode_id], arrays[episode_id]
            split = next(key for key in SPLITS if episode_id in settings["split_episode_ids"][key])
            prefix = "videos/" + CAMERA
            start, end = float(metadata[prefix + "/from_timestamp"]), float(metadata[prefix + "/to_timestamp"])
            count = len(columns["timestamp"])
            if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or
                    abs(end - start - count / 30.0) > settings["timestamp_tolerance_s"]):
                raise GraspDataError("episode video boundaries do not cover its original recorded rows")
            base = build / "episodes" / "episode_{:06d}".format(episode_id)
            image_path, array_path, receipt_path = (base.with_suffix(".rgb.npy"), base.with_suffix(".source.npz"),
                                                     base.with_suffix(".decoded.json"))
            video_name = _formatted_path(info, "video_path", metadata, CAMERA)
            decoded = decode_episode_rgb(_inside(videos, video_name), start + columns["timestamp"].astype(np.float64),
                image_path, width=settings["image_width"], height=settings["image_height"], source_width=640,
                source_height=480, fps=30.0, tolerance_s=settings["timestamp_tolerance_s"])
            decoded.update(episode_id=episode_id, camera_key=CAMERA, source_video=video_verified[video_name],
                           source_video_relative_path=video_name, source_video_from_timestamp_s=start,
                           source_video_to_timestamp_s=end)
            write_new_json(receipt_path, decoded)
            with array_path.open("xb") as handle:
                np.savez(handle, source_state=columns["observation.state"], source_action=columns["action"],
                         source_timestamp_s=columns["timestamp"], source_frame_index=columns["frame_index"],
                         source_episode_index=columns["episode_index"], source_global_index=columns["index"],
                         source_task_index=columns["task_index"], source_state_joint_names=np.asarray(SOURCE_JOINT_NAMES),
                         source_action_joint_names=np.asarray(SOURCE_JOINT_NAMES),
                         decoded_source_pts_s=np.asarray([item["decoded_source_pts_s"] for item in decoded["frames"]]),
                         decoded_source_pts_ticks=np.asarray([item["decoded_pts_ticks"] for item in decoded["frames"]], dtype=np.int64))
            built.append({"episode_id": episode_id, "split": split, "row_count": count,
                "sample_count": count - settings["chunk_steps"] + 1, "excluded_tail_observation_count": settings["chunk_steps"] - 1,
                "source_tasks": metadata.get("tasks", []), "source_task_indices": sorted(set(map(int, columns["task_index"]))),
                "data_relative_path": _formatted_path(info, "data_path", metadata),
                "all_camera_video_relative_paths": {key: _formatted_path(info, "video_path", metadata, key) for key in SOURCE_CAMERA_KEYS},
                "arrays": _file_record(array_path, build), "rgb": _file_record(image_path, build),
                "decoded_receipt": _file_record(receipt_path, build), "paired_rgb_frames": decoded["paired_frame_count"],
                "maximum_absolute_pts_error_s": decoded["maximum_absolute_pts_error_s"]})
            if _derived_bytes(build) > settings["max_derived_bytes"]:
                raise GraspDataError("derived byte budget exceeded while pairing complete real RGB rows")
        statistics_path = build / "train_stats.json"
        statistics = train_only_statistics(build, built, CAMERA)
        write_new_json(statistics_path, statistics)
        # Source files and downloader receipts remain unchanged throughout all
        # decoding and statistics, and their hashes can be independently rerun.
        verify_download_receipt(source, external["repo_id"], external["revision"], list(source_verified))
        verify_download_receipt(videos, external["repo_id"], external["revision"], video_names)
        if sha256_file(source / "provenance.json") != source_receipt_sha or sha256_file(videos / "provenance.json") != video_receipt_sha:
            raise GraspDataError("immutable source cache receipts changed during the native build")
        manifest = {"schema_version": 1, "backend": BACKEND, "passed": True, "source_repo_id": external["repo_id"],
            "source_revision": external["revision"], "source_fps": 30.0, "state_dim": 28, "action_dim": 28,
            "source_state_joint_names": list(SOURCE_JOINT_NAMES), "source_action_joint_names": list(SOURCE_JOINT_NAMES),
            "camera_key": CAMERA, "source_camera_role": "original_left_binocular_head_eye",
            "source_image_shape_hwc": [480, 640, 3],
            "image_shape_hwc": [settings["image_height"], settings["image_width"], 3],
            "image_transform": {"name": TRANSFORM, "library": "Pillow", "library_version": PIL.__version__,
                                "resample": "Resampling.BILINEAR", "input_mode": "RGB", "output_dtype": "uint8"},
            "chunk_steps": settings["chunk_steps"], "label_alignment": "unchanged_recorded_row_action_t_through_t_plus_chunk_minus_1",
            "split_unit": settings["split_unit"],
            "complete_chunks_only": True, "labels_shifted": False, "actions_clipped": False,
            "waist_fabricated": False, "time_resampled": False, "original_state_action_float32_preserved": True,
            "sensor_exposure_clock_verified": False, "motor_application_clock_verified": False,
            "current_simulator_adapter_ready": False, "closed_loop_success_demonstrated": False,
            "episode_ids": settings["episode_ids"], "split_episode_ids": settings["split_episode_ids"], "file_groups": groups,
            "prior_diagnostic_episodes_excluded": list(external["episode_ids"]), "episodes": built,
            "total_rows": total_rows, "total_paired_rgb_frames": sum(item["paired_rgb_frames"] for item in built),
            "split_sample_counts": {key: sum(item["sample_count"] for item in built if item["split"] == key) for key in SPLITS},
            "statistics": _file_record(statistics_path, build), "settings": settings,
            "settings_sha256": canonical_hash(settings), "verified_source_files": source_verified,
            "verified_video_files": video_verified, "source_provenance_sha256": source_receipt_sha,
            "video_provenance_sha256": video_receipt_sha, "source_task_metadata": _parquet_rows(source / "meta/tasks.parquet"),
            "task_identity_basis": "original parquet and episode metadata; dataset card prose is not a relabelled task",
            "source_native_training_eligible": True, "training_started": False,
            "planned_derived_bytes_upper_bound": planned_bytes, "actual_derived_bytes_before_manifest": _derived_bytes(build)}
        write_new_json(build / "manifest.json", manifest)
        if _derived_bytes(build) > settings["max_derived_bytes"]:
            raise GraspDataError("derived byte budget exceeded by the final native manifest")
        result = {"schema_version": 1, "passed": True, "status": "passed_source_native_complete_rows_rgb_and_file_groups",
                  "build_dir": str(build), "manifest_sha256": sha256_file(build / "manifest.json"),
                  "total_rows": total_rows, "total_paired_rgb_frames": manifest["total_paired_rgb_frames"],
                  "split_sample_counts": manifest["split_sample_counts"], "actual_derived_bytes": _derived_bytes(build),
                  "original_sources_unchanged": True, "training_started": False,
                  "source_native_training_eligible": True, "current_simulator_adapter_ready": False}
        write_new_json(Path(run_dir) / "reports/grasp_data_build.json", result)
        return result
    except Exception as exc:
        write_new_json(build / "failed.json", {"passed": False, "error": "{}: {}".format(type(exc).__name__, exc),
            "completed_episode_ids": [item["episode_id"] for item in built], "training_started": False,
            "partial_artifacts_preserved": True, "source_native_training_eligible": False})
        raise


def _verify_artifact(root: Path, record: Mapping[str, Any]) -> Path:
    path = _inside(root, record.get("relative_path"))
    if not path.is_file() or path.stat().st_size != record.get("size_bytes") or sha256_file(path) != record.get("sha256"):
        raise GraspDataError("derived artifact differs from its immutable manifest: " + str(path))
    return path


def load_grasp_manifest(build_dir: str | Path, *, expected_settings=None, expected_source=None, verify_files=True) -> dict:
    """Check derived admission and file hashes before any optimizer/evaluation."""
    root = Path(build_dir).resolve()
    if (root / "failed.json").exists():
        raise GraspDataError("partial failed native build is not admitted")
    manifest = json.loads(_inside(root, "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("passed") is not True or manifest.get("backend") != BACKEND or
            manifest.get("source_repo_id") != OFFICIAL_REPO or manifest.get("source_revision") != OFFICIAL_REVISION or
            manifest.get("state_dim") != 28 or manifest.get("action_dim") != 28 or manifest.get("source_fps") != 30.0 or
            manifest.get("camera_key") != CAMERA or manifest.get("label_alignment") != "unchanged_recorded_row_action_t_through_t_plus_chunk_minus_1" or
            manifest.get("complete_chunks_only") is not True or manifest.get("source_native_training_eligible") is not True or
            manifest.get("source_state_joint_names") != list(SOURCE_JOINT_NAMES) or manifest.get("source_action_joint_names") != list(SOURCE_JOINT_NAMES) or
            manifest.get("source_image_shape_hwc") != [480, 640, 3] or
            manifest.get("image_transform", {}).get("name") != TRANSFORM or manifest.get("split_unit") != "source_video_file_group" or
            any(manifest.get(key) is not False for key in ("labels_shifted", "actions_clipped", "waist_fabricated", "time_resampled"))):
        raise GraspDataError("derived native protocol or admission differs from the complete source backend")
    settings = validate_native_settings(manifest.get("settings", {}), manifest.get("prior_diagnostic_episodes_excluded", []))
    if canonical_hash(settings) != manifest.get("settings_sha256"):
        raise GraspDataError("derived settings differ from their saved identity")
    if expected_settings is not None and canonical_hash(dict(expected_settings)) != manifest["settings_sha256"]:
        raise GraspDataError("configured source-native settings differ from the admitted derived dataset")
    if expected_source is not None and (manifest["source_repo_id"], manifest["source_revision"]) != (expected_source.get("repo_id"), expected_source.get("revision")):
        raise GraspDataError("configured official source pin differs from the admitted native dataset")
    if (manifest.get("episode_ids") != settings["episode_ids"] or manifest.get("split_episode_ids") != settings["split_episode_ids"] or
            manifest.get("chunk_steps") != settings["chunk_steps"] or
            manifest.get("image_shape_hwc") != [settings["image_height"], settings["image_width"], 3]):
        raise GraspDataError("manifest summary differs from its predeclared selection or native model shape")
    episodes = manifest.get("episodes", [])
    if [item.get("episode_id") for item in episodes] != settings["episode_ids"]:
        raise GraspDataError("derived episodes differ from the immutable selection")
    groups = {key: set() for key in SPLITS}
    required_videos = set()
    total_rows, artifact_paths = 0, set()
    for item in episodes:
        row_count = _integer(item.get("row_count"), "episode row_count")
        if (item.get("split") not in SPLITS or item["episode_id"] not in settings["split_episode_ids"][item["split"]] or
                item.get("sample_count") != row_count - settings["chunk_steps"] + 1 or row_count < settings["chunk_steps"] or
                row_count > settings["max_episode_rows"] or item.get("paired_rgb_frames") != row_count or
                item.get("excluded_tail_observation_count") != settings["chunk_steps"] - 1):
            raise GraspDataError("derived complete chunk boundaries or paired RGB coverage differ")
        total_rows += row_count
        cameras = item.get("all_camera_video_relative_paths", {})
        if set(cameras) != set(SOURCE_CAMERA_KEYS):
            raise GraspDataError("all-camera source file grouping identities are missing")
        for key, video_path in cameras.items():
            if not isinstance(video_path, str) or re.fullmatch(re.escape("videos/" + key) + r"/chunk-[0-9]{3}/file-[0-9]{3}\.mp4", video_path) is None:
                raise GraspDataError("source camera file grouping contains an invalid native path")
            groups[item["split"]].add(video_path)
        required_videos.add(cameras[CAMERA])
        for key in ("arrays", "rgb", "decoded_receipt"):
            identity = item.get(key, {}).get("relative_path")
            _inside(root, identity)
            if identity in artifact_paths:
                raise GraspDataError("episodes share derived artifact file identities")
            artifact_paths.add(identity)
        if verify_files:
            for key in ("arrays", "rgb", "decoded_receipt"):
                _verify_artifact(root, item[key])
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1:]:
            if groups[left].intersection(groups[right]):
                raise GraspDataError("derived native held-out split shares source video storage groups")
    declared_groups = manifest.get("file_groups", {})
    if (declared_groups.get("source_video_paths_by_split") != {key: sorted(value) for key, value in groups.items()} or
            declared_groups.get("checked_camera_keys") != list(SOURCE_CAMERA_KEYS) or
            declared_groups.get("all_camera_storage_files_disjoint") is not True or
            declared_groups.get("capture_sessions_independent_verified") is not False):
        raise GraspDataError("derived all-camera file group evidence differs from admitted episode metadata")
    if (total_rows > settings["max_total_rows"] or manifest.get("total_rows") != total_rows or
            manifest.get("total_paired_rgb_frames") != total_rows or
            manifest.get("split_sample_counts") != {key: sum(item["sample_count"] for item in episodes if item["split"] == key) for key in SPLITS}):
        raise GraspDataError("native complete row/RGB/sample totals differ from their source episodes")
    if not required_videos.issubset(manifest.get("verified_video_files", {})):
        raise GraspDataError("derived native video files are absent from full pinned source verification")
    statistics_path = _verify_artifact(root, manifest["statistics"]) if verify_files else _inside(root, manifest["statistics"]["relative_path"])
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    if (statistics.get("training_episode_ids") != [item["episode_id"] for item in episodes if item["split"] == "train"] or
            statistics.get("validation_or_test_used") is not False or statistics.get("statistics_include_held_out") is not False or
            statistics.get("std_minimum") != 1e-3):
        raise GraspDataError("normalization statistics are not declared train-only")
    train_rows = sum(item["row_count"] for item in episodes if item["split"] == "train")
    for key, shape in (("observation.state", (28,)), ("action", (28,)), (CAMERA, (3, 1, 1))):
        expected_count = train_rows * settings["image_width"] * settings["image_height"] if key == CAMERA else train_rows
        if statistics.get("features", {}).get(key, {}).get("count") != expected_count:
            raise GraspDataError("train-only statistics include a different row or RGB population")
        for suffix in ("mean", "std"):
            values = np.asarray(statistics.get("features", {}).get(key, {}).get(suffix))
            if values.shape != shape or not np.isfinite(values).all() or (suffix == "std" and not np.all(values >= 1e-3)):
                raise GraspDataError("train-only normalization has invalid native dimensions or values")
    if verify_files:
        actual = train_only_statistics(root, episodes, CAMERA)
        for key in ("observation.state", "action", CAMERA):
            for suffix in ("mean", "std", "unclamped_std"):
                saved_values = np.asarray(statistics["features"][key].get(suffix))
                actual_values = np.asarray(actual["features"][key][suffix])
                if saved_values.shape != actual_values.shape or not np.allclose(saved_values, actual_values, rtol=1e-12, atol=1e-12):
                    raise GraspDataError("saved normalization differs from the actual train-only rows or RGB population")
    return manifest


class GraspSquareDataset:
    """Dependency-free array dataset; a trainer applies saved train-only stats."""

    def __init__(self, build_dir: str | Path, split: str, *, manifest=None, verify=True):
        if split not in SPLITS:
            raise GraspDataError("unknown source-native dataset split")
        self.root = Path(build_dir).resolve()
        self.manifest = load_grasp_manifest(self.root, verify_files=verify) if manifest is None else manifest
        if manifest is not None and verify:
            verified = load_grasp_manifest(self.root, verify_files=True)
            if canonical_hash(verified) != canonical_hash(manifest):
                raise GraspDataError("provided manifest differs from independently verified dataset")
        self.statistics = json.loads(_inside(self.root, self.manifest["statistics"]["relative_path"]).read_text(encoding="utf-8"))
        self.split, self.chunk_steps, self.camera_key = split, self.manifest["chunk_steps"], self.manifest["camera_key"]
        self.episodes = {item["episode_id"]: item for item in self.manifest["episodes"] if item["split"] == split}
        self.samples = [(item["episode_id"], frame) for item in self.manifest["episodes"] if item["split"] == split
                        for frame in range(item["sample_count"])]
        if not self.samples:
            raise GraspDataError("native split has no complete source action chunks")
        self._arrays, self._images = {}, {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        episode_id, frame = self.samples[index]
        item = self.episodes[episode_id]
        if episode_id not in self._arrays:
            with np.load(_inside(self.root, item["arrays"]["relative_path"]), allow_pickle=False) as saved:
                state, action = saved["source_state"], saved["source_action"]
                if state.shape != (item["row_count"], 28) or action.shape != state.shape or state.dtype != np.float32 or action.dtype != np.float32:
                    raise GraspDataError("derived native float32 rows differ from admitted dimensions")
                self._arrays[episode_id] = (state.copy(), action.copy())
            images = np.load(_inside(self.root, item["rgb"]["relative_path"]), mmap_mode="r", allow_pickle=False)
            if images.dtype != np.uint8 or images.shape != (item["row_count"], *self.manifest["image_shape_hwc"]):
                raise GraspDataError("derived real camera rows differ from admitted RGB dimensions")
            self._images[episode_id] = images
        state, action = self._arrays[episode_id]
        return {"observation.state": state[frame].copy(),
                self.camera_key: np.ascontiguousarray(self._images[episode_id][frame].transpose(2, 0, 1), dtype=np.float32) / 255.0,
                "action": action[frame:frame + self.chunk_steps].copy(),
                "action_is_pad": np.zeros(self.chunk_steps, dtype=np.bool_),
                "episode_index": np.int64(episode_id), "frame_index": np.int64(frame)}

    def get_eval_indices(self, stride: int) -> list[int]:
        stride = _integer(stride, "evaluation stride")
        return [index for index, (_, frame) in enumerate(self.samples) if frame % stride == 0]
