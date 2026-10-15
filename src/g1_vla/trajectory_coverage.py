"""Train-only exposure audit and metadata-only independent-data planning.

Video storage boundaries are not capture-session boundaries.  Both direct file
exclusion and transitive all-camera connectivity are reported so an adjacent
episode is never advertised as an independent scene or recording session.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from .artifacts import sha256_file
from .external_data import SOURCE_CAMERA_KEYS, SOURCE_JOINT_NAMES, _formatted_path, _parquet_rows
from .external_sources import verify_download_receipt
from .grasp_data import _verify_artifact, load_grasp_manifest
from .native_diagnostics import _resolve
from .runtime.experiment import write_new_json


BACKEND = "grasp_square_trajectory_coverage_v1"


class TrajectoryCoverageError(ValueError):
    """The declared training exposure or metadata-only scope is not preserved."""


def validate_trajectory_coverage_settings(config):
    settings = copy.deepcopy(config["trajectory_coverage"])
    if settings.get("backend") != BACKEND:
        raise TrajectoryCoverageError("unknown trajectory coverage backend")
    digest = settings.get("reference_training_report_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise TrajectoryCoverageError("the prior train/validation-only report needs a complete SHA256")
    if not isinstance(settings.get("reference_training_report_path"), str):
        raise TrajectoryCoverageError("a prior training report path must be declared")
    _resolve(config, settings["reference_training_report_path"])
    for key in ("stationary_speed_rad_s", "moving_speed_rad_s"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TrajectoryCoverageError("motion thresholds must be finite numbers")
    if not 0 <= settings["stationary_speed_rad_s"] < settings["moving_speed_rad_s"]:
        raise TrajectoryCoverageError("stationary and moving thresholds must be ordered")
    if settings.get("quarter_bins") != 4 or isinstance(settings.get("quarter_bins"), bool):
        raise TrajectoryCoverageError("the audit predeclares four equal anchor-frame bins")
    for key, maximum in (("expansion_group_limit", 4), ("expansion_episodes_per_group", 2)):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise TrajectoryCoverageError("metadata candidate bounds exceed four groups / two episodes")
    return settings


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None, "p95": None}
    return {"count": int(values.size), "minimum": float(values.min()), "maximum": float(values.max()),
            "mean": float(values.mean()), "p95": float(np.quantile(values, .95))}


def _range(values):
    values = np.asarray(values, dtype=np.float64)
    return {"row_count": len(values), "minimum_by_joint": values.min(axis=0).tolist(),
            "maximum_by_joint": values.max(axis=0).tolist(),
            "range_by_joint": np.ptp(values, axis=0).tolist()}


def exposure_coverage(episodes, identities, indices, *, chunk_steps, stationary_speed, moving_speed):
    """Count actual saved draws, distinct anchor observations and future label rows."""
    if not episodes or {item["split"] for item in episodes} != {"train"}:
        raise TrajectoryCoverageError("only training payloads can enter trajectory exposure")
    samples = [(item["episode_id"], frame) for item in episodes for frame in range(item["sample_count"])]
    identities, indices = np.asarray(identities), np.asarray(indices)
    if (indices.ndim != 2 or indices.dtype.kind not in "iu" or not indices.size or
            np.any(indices < 0) or np.any(indices >= len(samples)) or
            identities.shape != (*indices.shape, 2) or identities.dtype.kind not in "iu"):
        raise TrajectoryCoverageError("saved schedule dimensions or anchor bounds differ")
    expected = np.asarray(samples, dtype=np.int64)[indices]
    if not np.array_equal(expected, identities):
        raise TrajectoryCoverageError("saved dataset indices differ from episode/frame identities")
    by_episode, all_state, all_action, all_residual = {}, [], [], []
    strata_total = {key: {"available_anchor_count": 0, "draw_count": 0, "unique_anchor_count": 0}
                    for key in ("stationary", "intermediate", "moving", "unknown_first_row")}
    total_label_rows, covered_label_rows, total_label_visits = 0, 0, 0
    for episode in episodes:
        identity, count, anchors = episode["episode_id"], episode["row_count"], episode["sample_count"]
        state, action = episode["state"], episode["action"]
        if (anchors != count - chunk_steps + 1 or state.shape != (count, 28) or action.shape != state.shape or
                state.dtype != np.float32 or action.dtype != np.float32 or
                not np.isfinite(state).all() or not np.isfinite(action).all()):
            raise TrajectoryCoverageError("training arrays must preserve complete finite float32/28 source rows")
        draws = identities.reshape(-1, 2)
        frames = draws[draws[:, 0] == identity, 1]
        visits = np.bincount(frames, minlength=anchors).astype(np.int64)
        label_visits = np.zeros(count, dtype=np.int64)
        for frame, visits_at_frame in enumerate(visits):
            label_visits[frame:frame + chunk_steps] += visits_at_frame
        speeds = np.full(anchors, np.nan)
        speeds[1:] = np.max(np.abs(np.diff(state[:anchors].astype(np.float64), axis=0)) * 30., axis=1)
        masks = {"stationary": speeds <= stationary_speed,
                 "intermediate": (speeds > stationary_speed) & (speeds < moving_speed),
                 "moving": speeds >= moving_speed, "unknown_first_row": ~np.isfinite(speeds)}
        strata = {}
        for key, mask in masks.items():
            available, unique, visited = int(mask.sum()), int(((visits > 0) & mask).sum()), int(visits[mask].sum())
            strata[key] = {"available_anchor_count": available, "draw_count": visited,
                           "unique_anchor_count": unique, "unique_anchor_fraction": unique / available if available else None}
            for field in strata_total[key]:
                strata_total[key][field] += strata[key][field]
        edges = np.linspace(0, anchors, 5, dtype=np.int64)
        bins = [{"start_anchor_frame_inclusive": int(left), "end_anchor_frame_exclusive": int(right),
                 "available_anchor_count": int(right - left), "draw_count": int(visits[left:right].sum()),
                 "unique_anchor_count": int((visits[left:right] > 0).sum()),
                 "unique_anchor_fraction": float((visits[left:right] > 0).mean()) if right > left else None}
                for left, right in zip(edges[:-1], edges[1:])]
        by_episode[str(identity)] = {"source_row_count": count, "available_anchor_count": anchors,
            "draw_count": len(frames), "unique_anchor_count": int((visits > 0).sum()),
            "unique_anchor_fraction": float((visits > 0).mean()),
            "action_label_row_coverage_count": int((label_visits > 0).sum()),
            "action_label_row_coverage_fraction": float((label_visits > 0).mean()),
            "action_label_scalar_visits": int(label_visits.sum()) * 28,
            "quarter_anchor_bins": bins, "velocity_strata": strata,
            "speed_summary_rad_s": _summary(speeds[np.isfinite(speeds)]),
            "anchor_visit_histogram": {str(int(value)): int((visits == value).sum()) for value in np.unique(visits)},
            "unobserved_anchor_frame_indices": np.flatnonzero(visits == 0).tolist(),
            "unvisited_action_label_row_indices": np.flatnonzero(label_visits == 0).tolist()}
        total_label_rows += count; covered_label_rows += int((label_visits > 0).sum())
        total_label_visits += int(label_visits.sum())
        all_state.append(state); all_action.append(action)
        # Same complete anchor population for every future step, without padding.
        all_residual.append(np.stack([action[h:h + anchors] - state[:anchors] for h in range(chunk_steps)], axis=1))
    for value in strata_total.values():
        available = value["available_anchor_count"]
        value["unique_anchor_fraction"] = value["unique_anchor_count"] / available if available else None
    residual = np.concatenate(all_residual).astype(np.float64)
    return {"evidence_kind": "verified saved training_anchor_schedule.npz; not a seed-only reconstruction",
        "training_episode_count": len(episodes), "source_row_count": total_label_rows,
        "available_anchor_count": len(samples), "sample_draw_count": indices.size,
        "unique_anchor_count": len(np.unique(indices)),
        "unique_anchor_fraction": len(np.unique(indices)) / len(samples),
        "equivalent_anchor_epochs": indices.size / len(samples),
        "action_label_row_coverage_count": covered_label_rows,
        "action_label_row_coverage_fraction": covered_label_rows / total_label_rows,
        "action_label_row_visits": total_label_visits, "action_label_scalar_visits": total_label_visits * 28,
        "sample_identity_sha256": _digest(identities), "per_episode": by_episode,
        "speed_definition": "max over 28 joints abs(state[t]-state[t-1])*30; first row unknown",
        "stationary_speed_rad_s": stationary_speed, "moving_speed_rad_s": moving_speed,
        "velocity_strata": strata_total, "state_range_train_only": _range(np.concatenate(all_state)),
        "action_range_train_only": _range(np.concatenate(all_action)),
        "hold_residual_train_only": {"definition": "action[t+h]-state[t], all complete train anchors at every h",
            "all_28_mae_rad": float(np.abs(residual).mean()),
            "mae_rad_by_horizon": np.abs(residual).mean(axis=(0, 2)).tolist(),
            "minimum_by_joint": residual.min(axis=(0, 1)).tolist(),
            "maximum_by_joint": residual.max(axis=(0, 1)).tolist()},
        "sufficiency_claim": False,
        "interpretation": "Many overlapping future labels can be visited while half the anchor observations are unseen; neither is a count of independent recordings."}


def metadata_components(rows, info):
    """Compute transitive connectivity using shared files from any of four cameras."""
    ordered = sorted(rows, key=lambda item: item["episode_index"])
    identities = [item["episode_index"] for item in ordered]
    if len(set(identities)) != len(identities):
        raise TrajectoryCoverageError("official episode metadata contains duplicate identities")
    parent = {identity: identity for identity in identities}

    def find(identity):
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]; identity = parent[identity]
        return identity

    seen, paths = {}, {}
    for row in ordered:
        identity = row["episode_index"]
        paths[identity] = [_formatted_path(info, "video_path", row, camera) for camera in SOURCE_CAMERA_KEYS]
        for path in paths[identity]:
            if path in seen:
                left, right = find(identity), find(seen[path])
                parent[max(left, right)] = min(left, right)
            else:
                seen[path] = identity
    components = {}
    for identity in identities:
        components.setdefault(find(identity), []).append(identity)
    return paths, [{"component_id": "all-camera-component-{:06d}".format(min(values)),
                    "episode_ids": values, "source_video_paths": sorted({p for identity in values for p in paths[identity]})}
                   for values in sorted(components.values(), key=lambda values: min(values))]


def expansion_plan(rows, info, splits, discovery_ids, settings):
    """Propose metadata-only audits; never admit these candidates for training."""
    paths, components = metadata_components(rows, info)
    all_ids = set(paths)
    if any(set(values) - all_ids for values in splits.values()):
        raise TrajectoryCoverageError("frozen split identities are absent from official metadata")
    current = set().union(*map(set, splits.values()))
    held = set(splits["validation"]) | set(splits["test"])
    held_paths = {path for identity in held for path in paths[identity]}
    train_paths = {path for identity in splits["train"] for path in paths[identity]}
    direct_blocked = {identity for identity in all_ids if held_paths.intersection(paths[identity])}
    exclusion = current | set(discovery_ids)
    direct_candidates = sorted(all_ids - direct_blocked - exclusion)
    connected_blocked, component_rows = set(), []
    for component in components:
        intersects = held.intersection(component["episode_ids"])
        if intersects:
            connected_blocked.update(component["episode_ids"])
        component_rows.append({**component, "intersects_frozen_validation_or_test": bool(intersects),
                               "frozen_episode_ids": sorted(intersects)})
    independent_candidates = sorted(all_ids - connected_blocked - exclusion)
    metadata_by_id = {row["episode_index"]: row for row in rows}
    groups = {}
    for identity in direct_candidates:
        groups.setdefault(tuple(paths[identity]), []).append(identity)
    previews = []
    ordered_groups = sorted(groups.items(), key=lambda pair: min(pair[1]))
    group_count = min(len(ordered_groups), settings["expansion_group_limit"])
    group_positions = np.floor((np.arange(group_count) + .5) * len(ordered_groups) / group_count).astype(int) if group_count else []
    for position in group_positions:
        key, candidates = ordered_groups[int(position)]
        # Fixed equally spaced metadata quantiles, unrelated to losses or visual outcomes.
        size = min(len(candidates), settings["expansion_episodes_per_group"])
        positions = np.floor((np.arange(size) + .5) * len(candidates) / size).astype(int)
        selected = [candidates[int(index)] for index in positions]
        previews.append({"candidate_episode_ids": candidates, "metadata_audit_preview_episode_ids": selected,
            "source_video_paths": list(key), "shares_existing_train_video_storage": bool(train_paths.intersection(key)),
            "contains_storage_path_not_in_current_train": bool(set(key) - train_paths),
            "independent_from_held_out_by_transitive_files": all(identity in independent_candidates for identity in selected),
            "preview_source_row_count": sum(int(metadata_by_id[identity]["length"]) for identity in selected),
            "preview_tasks": sorted({task for identity in selected for task in metadata_by_id[identity]["tasks"]})})
    return {"selection_basis": "pinned original identity/length/video-offset metadata only; no model score, candidate pixels, states, actions or episode summary statistics",
        "frozen_split_episode_ids": splits, "prior_discovery_episode_ids_excluded": sorted(set(discovery_ids)),
        "all_four_camera_keys": list(SOURCE_CAMERA_KEYS), "official_episode_count": len(rows),
        "official_frame_count": sum(int(row["length"]) for row in rows),
        "direct_frozen_video_file_exclusion": {"blocked_episode_count": len(direct_blocked),
            "blocked_episode_ids": sorted(direct_blocked), "eligible_unused_episode_count": len(direct_candidates),
            "eligible_unused_episode_ids": direct_candidates, "candidate_video_combination_count": len(groups)},
        "transitive_all_camera_connectivity": {"component_count": len(components), "components": component_rows,
            "frozen_connected_blocked_episode_count": len(connected_blocked),
            "eligible_unused_episode_count": len(independent_candidates), "eligible_unused_episode_ids": independent_candidates},
        "bounded_metadata_only_preview": previews,
        "preview_selection_rule": "equally spaced rank midpoints of metadata-ordered video-file combinations, then equally spaced rank midpoints of eligible episode IDs within each selected combination",
        "preview_admitted_for_training": False, "candidate_payloads_downloaded": False,
        "capture_session_independence_verified": False, "scene_operator_independence_verified": False,
        "shared_parquet_policy": "A shared Parquet container is not an independent episode identity; use exact row filters and hashes. File disjointness is checked on all four camera videos, with transitive connectivity separately disclosed.",
        "recommended_expansion": [
            "Preserve the current six train, three validation and three old test episodes and the existing budget for this residual comparison; do not append candidates during this stage.",
            "Treat direct-file-safe official candidates as same-dataset coverage expansion. The transitive storage component is a sharding relationship, not proof that episodes share a real capture session. Current metadata cannot certify independent sessions, scenes or operators; do not reject all coverage expansion or advertise an independent benchmark on that basis.",
            "Request traceable session metadata from the public source, use another compatible source with such provenance, or separately collect data only when a real collector is available and its teacher passes acceptance. The current local teacher passed only 4 of 20 layouts and is not an admitted collection route. Before opening new payloads, register session_id, recording_date, operator_id, scene/layout_id, object_instance_id, task/outcome definitions, robot joint protocol, camera calibration, source clocks, timestamps and immutable source/video hashes.",
            "Predeclare complete-session train/validation/new-test groups and all-four-camera connectivity exclusion before decoding or using model outcomes. No frame split, overlapping clip, old test promotion or label shifts are permitted.",
            "First audit at least three independent sessions with two complete episodes per session across predeclared object/layout changes; reserve at least one whole session as a new locked test. Expand session count before claiming generalization; six episodes are a protocol pilot, not statistical proof.",
            "Admit each full episode only after original 28D joint order, actual RGB/PTS uniqueness, unshifted 30Hz labels, complete 16-step chunks and source identity checks pass. Recompute normalization from the expanded train split only and report state-only/RGB/hold with fixed optimizer draws and the expanded coverage denominator."
        ]}


def _metadata(config, manifest):
    source = _resolve(config, config["external_data"]["local_dir"])
    paths = sorted((source / "meta/episodes").glob("chunk-*/file-*.parquet"))
    if not paths or len(paths) > 32:
        raise TrajectoryCoverageError("bounded original episode metadata shards are required")
    names = ["meta/info.json"] + [str(path.relative_to(source)) for path in paths]
    verified = verify_download_receipt(source, config["external_data"]["repo_id"],
                                       config["external_data"]["revision"], names)
    if any(item["sha256"] != manifest["verified_source_files"][name]["sha256"] for name, item in verified.items()):
        raise TrajectoryCoverageError("original metadata identity differs from the admitted dataset")
    info = _read(source / "meta/info.json")
    if info.get("codebase_version") != "v3.0" or info.get("fps") != 30:
        raise TrajectoryCoverageError("the pinned v3/30Hz metadata protocol must remain unchanged")
    columns = ["episode_index", "length", "tasks", "dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index"]
    columns += ["videos/" + camera + "/" + key for camera in SOURCE_CAMERA_KEYS
                for key in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")]
    rows = [row for path in paths for row in _parquet_rows(path, columns=columns)]
    if len(rows) != info.get("total_episodes") or sum(int(row["length"]) for row in rows) != info.get("total_frames"):
        raise TrajectoryCoverageError("official episode/frame metadata totals differ")
    for row in rows:
        count = row["length"]
        if (isinstance(count, bool) or not isinstance(count, int) or count < 1 or
                row["dataset_to_index"] - row["dataset_from_index"] != count):
            raise TrajectoryCoverageError("metadata row boundaries are incomplete")
        for camera in SOURCE_CAMERA_KEYS:
            prefix = "videos/" + camera
            start, end = row[prefix + "/from_timestamp"], row[prefix + "/to_timestamp"]
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or abs((end - start) - count / 30.) > 1e-5:
                raise TrajectoryCoverageError("metadata video offsets differ from complete episode duration")
    return info, rows, {"verified_source_files": verified, "materialized_columns": columns,
                        "state_action_image_summary_statistics_materialized": False}


def run_trajectory_coverage(config, run_dir):
    """Write immutable train-only coverage and bounded metadata-only expansion evidence."""
    target = Path(run_dir) / "reports/trajectory_coverage.json"
    if target.exists():
        raise TrajectoryCoverageError("coverage evidence already exists and cannot be overwritten")
    started = time.perf_counter()
    report = {"schema_version": 1, "passed": False, "status": "failed", "training_started": False,
              "optimizer_steps": 0, "test_payloads_opened": False, "test_metrics_read": False,
              "validation_payloads_opened": False, "candidate_payloads_opened": False,
              "images_opened": False, "closed_loop_control": False, "labels_modified": False}
    try:
        settings = validate_trajectory_coverage_settings(config)
        manifest_path = _resolve(config, config["grasp_act"]["dataset_manifest_path"])
        root = manifest_path.parent
        manifest = load_grasp_manifest(root, expected_settings=config["source_native"],
                                       expected_source=config["external_data"], verify_files=False)
        reference_path = _resolve(config, settings["reference_training_report_path"])
        if sha256_file(reference_path) != settings["reference_training_report_sha256"]:
            raise TrajectoryCoverageError("prior training receipt hash changed")
        reference = _read(reference_path)
        if (reference.get("passed") is not True or reference.get("test_used_for_selection") is not False or
                reference.get("test_payloads_opened") is not False or
                reference.get("bindings", {}).get("manifest_sha256") != sha256_file(manifest_path)):
            raise TrajectoryCoverageError("reference must be passed train/validation-only and bound to this manifest")
        schedule_record = reference["anchor_schedule_artifact"]
        schedule_path = _resolve(config, schedule_record["path"])
        if (schedule_path.is_symlink() or schedule_path.stat().st_size != schedule_record["size_bytes"] or
                sha256_file(schedule_path) != schedule_record["sha256"]):
            raise TrajectoryCoverageError("saved training schedule differs from its immutable receipt")
        with np.load(schedule_path, allow_pickle=False) as saved:
            identities, indices = saved["sample_episode_frame"].copy(), saved["dataset_sample_indices"].copy()
        declared = reference["anchor_schedule"]
        if (indices.shape != (declared["steps"], declared["batch_size"]) or
                not np.array_equal(indices, np.random.default_rng(declared["seed"]).integers(
                    0, declared["available_train_anchor_count"], size=indices.shape)) or
                _digest(identities) != declared["sample_identity_sha256"]):
            raise TrajectoryCoverageError("saved training draws do not replay their fixed NumPy identity")
        selected, artifacts = [], {}
        for item in manifest["episodes"]:
            if item["split"] != "train":
                continue
            path = _verify_artifact(root, item["arrays"])
            artifacts[item["arrays"]["relative_path"]] = sha256_file(path)
            with np.load(path, allow_pickle=False) as saved:
                if (tuple(saved["source_state_joint_names"].tolist()) != SOURCE_JOINT_NAMES or
                        tuple(saved["source_action_joint_names"].tolist()) != SOURCE_JOINT_NAMES or
                        not np.array_equal(saved["source_frame_index"], np.arange(item["row_count"])) or
                        not np.all(saved["source_episode_index"] == item["episode_id"])):
                    raise TrajectoryCoverageError("train source identities or 28D joint order changed")
                selected.append({**item, "state": saved["source_state"].copy(), "action": saved["source_action"].copy()})
        coverage = exposure_coverage(selected, identities, indices, chunk_steps=manifest["chunk_steps"],
            stationary_speed=settings["stationary_speed_rad_s"], moving_speed=settings["moving_speed_rad_s"])
        for key, value in (("available_train_anchor_count", coverage["available_anchor_count"]),
                           ("sample_draw_count", coverage["sample_draw_count"]),
                           ("unique_train_anchor_count", coverage["unique_anchor_count"])):
            if declared[key] != value:
                raise TrajectoryCoverageError("saved reference coverage summary differs from actual schedule")
        info, rows, metadata_receipt = _metadata(config, manifest)
        report.update(settings=settings, bindings={"module_sha256": sha256_file(__file__),
            "manifest_sha256": sha256_file(manifest_path), "reference_training_report_sha256": sha256_file(reference_path),
            "anchor_schedule_sha256": sha256_file(schedule_path), "verified_train_array_sha256": artifacts},
            scope={"payload_split_names": ["train"], "payload_episode_ids": [item["episode_id"] for item in selected],
                   "official_metadata_only_episode_count": len(rows), "held_out_metadata_for_exclusion_only": True},
            training_exposure=coverage, source_metadata=metadata_receipt,
            expansion=expansion_plan(rows, info, manifest["split_episode_ids"],
                                     config["external_data"]["episode_ids"], settings))
        report["passed"], report["status"] = True, "passed_train_only_exposure_and_metadata_expansion_audit"
        report["limitations"] = ["Training exposure does not establish independent capture sessions or successful physical grasps.",
            "Candidate payloads and held-out data/metrics are not opened; metadata file grouping alone cannot certify scene, operator or clock independence.",
            "This stage proposes a separately registered data version; it neither expands the active dataset nor changes frozen normalization, splits or optimizer budget."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        report["total_elapsed_s"] = time.perf_counter() - started
        write_new_json(target, report)
    return report
