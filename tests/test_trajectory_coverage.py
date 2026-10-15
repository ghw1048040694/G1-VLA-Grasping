"""Training exposure denominators and metadata-only expansion boundaries."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from g1_vla.artifacts import sha256_file
from g1_vla.external_data import SOURCE_CAMERA_KEYS, SOURCE_JOINT_NAMES
from g1_vla.trajectory_coverage import (
    BACKEND, TrajectoryCoverageError, expansion_plan, exposure_coverage,
    metadata_components, run_trajectory_coverage, validate_trajectory_coverage_settings,
)


def episode(identity=1, count=24, chunk=4):
    state = np.broadcast_to(np.arange(count, dtype=np.float32)[:, None] * np.float32(.01), (count, 28)).copy()
    return {"episode_id": identity, "split": "train", "row_count": count, "sample_count": count - chunk + 1,
            "state": state, "action": state + np.float32(.02)}


def metadata(identity, files=None, length=24):
    files = files if files is not None else [identity] * 4
    row = {"episode_index": identity, "length": length, "tasks": ["recorded task"], "data/chunk_index": 0,
           "data/file_index": 0, "dataset_from_index": identity * length, "dataset_to_index": (identity + 1) * length}
    for camera, file in zip(SOURCE_CAMERA_KEYS, files):
        prefix = "videos/" + camera
        row.update({prefix + "/chunk_index": 0, prefix + "/file_index": file,
                    prefix + "/from_timestamp": 0., prefix + "/to_timestamp": length / 30.})
    return row


INFO = {"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}


def settings():
    return {"backend": BACKEND, "reference_training_report_path": "reference.json",
            "reference_training_report_sha256": "a" * 64,
            "stationary_speed_rad_s": .02, "moving_speed_rad_s": .1, "quarter_bins": 4,
            "expansion_group_limit": 4, "expansion_episodes_per_group": 2}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class TrajectoryCoverageTests(unittest.TestCase):
    def test_counts_anchor_observations_and_future_label_rows_separately(self):
        item = episode(count=12)
        indices = np.array([[0, 0], [8, 3]], dtype=np.int64)
        identities = np.array([[[1, 0], [1, 0]], [[1, 8], [1, 3]]], dtype=np.int64)
        result = exposure_coverage([item], identities, indices, chunk_steps=4, stationary_speed=.02, moving_speed=.1)
        self.assertEqual(result["sample_draw_count"], 4)
        self.assertEqual(result["unique_anchor_count"], 3)
        self.assertEqual(result["action_label_row_coverage_count"], 11)
        self.assertEqual(result["action_label_row_visits"], 16)
        self.assertEqual(result["action_label_scalar_visits"], 16 * 28)
        self.assertEqual(sum(x["draw_count"] for x in result["per_episode"]["1"]["quarter_anchor_bins"]), 4)
        self.assertFalse(result["sufficiency_claim"])

    def test_velocity_strata_partition_and_first_row_unknown(self):
        item = episode(count=12)
        item["state"][:3] = 0
        item["state"][3] = .002
        indices = np.array([[0, 1], [3, 4]], dtype=np.int64)
        identities = np.stack([np.ones_like(indices), indices], axis=-1)
        result = exposure_coverage([item], identities, indices, chunk_steps=4, stationary_speed=.02, moving_speed=.1)
        strata = result["velocity_strata"]
        self.assertEqual(sum(value["available_anchor_count"] for value in strata.values()), 9)
        self.assertEqual(strata["unknown_first_row"]["available_anchor_count"], 1)
        self.assertEqual(strata["stationary"]["available_anchor_count"], 2)
        self.assertEqual(strata["intermediate"]["available_anchor_count"], 1)
        self.assertEqual(sum(value["draw_count"] for value in strata.values()), 4)

    def test_exposure_rejects_nontrain_and_changed_saved_schedule(self):
        item = episode()
        ids = np.array([[[1, 0]]], dtype=np.int64)
        indices = np.array([[1]], dtype=np.int64)
        with self.assertRaisesRegex(TrajectoryCoverageError, "identities"):
            exposure_coverage([item], ids, indices, chunk_steps=4, stationary_speed=.02, moving_speed=.1)
        item["split"] = "test"
        with self.assertRaisesRegex(TrajectoryCoverageError, "only training"):
            exposure_coverage([item], ids, indices, chunk_steps=4, stationary_speed=.02, moving_speed=.1)

    def test_complete_horizons_use_equal_train_anchor_population(self):
        item = episode(count=12)
        indices = np.array([[0]], dtype=np.int64)
        result = exposure_coverage([item], np.array([[[1, 0]]]), indices,
                                   chunk_steps=4, stationary_speed=.02, moving_speed=.1)
        expected = [.02, .03, .04, .05]
        np.testing.assert_allclose(result["hold_residual_train_only"]["mae_rad_by_horizon"], expected, atol=1e-8)
        self.assertEqual(result["state_range_train_only"]["row_count"], 12)

    def test_connected_component_uses_transitive_shared_files_across_cameras(self):
        rows = [metadata(1, [1, 1, 1, 1]), metadata(2, [2, 1, 2, 2]),
                metadata(3, [3, 3, 2, 3]), metadata(4, [4, 4, 4, 4])]
        _, groups = metadata_components(rows, INFO)
        self.assertEqual([row["episode_ids"] for row in groups], [[1, 2, 3], [4]])
        with self.assertRaisesRegex(TrajectoryCoverageError, "duplicate"):
            metadata_components([rows[0], rows[0]], INFO)

    def test_direct_safe_candidate_can_be_transitively_connected_without_proving_session(self):
        rows = [metadata(1, [1, 1, 1, 1]), metadata(2, [2, 1, 2, 2]),
                metadata(3, [3, 3, 2, 3]), metadata(4), metadata(5)]
        result = expansion_plan(rows, INFO, {"train": [4], "validation": [3], "test": [5]}, [], settings())
        self.assertIn(1, result["transitive_all_camera_connectivity"]["components"][0]["episode_ids"])
        self.assertEqual(result["direct_frozen_video_file_exclusion"]["eligible_unused_episode_ids"], [1])
        self.assertEqual(result["transitive_all_camera_connectivity"]["eligible_unused_episode_ids"], [])
        self.assertFalse(result["bounded_metadata_only_preview"][0]["independent_from_held_out_by_transitive_files"])
        self.assertFalse(result["capture_session_independence_verified"])
        self.assertFalse(result["preview_admitted_for_training"])

    def test_expansion_excludes_any_frozen_camera_file_current_and_discovery(self):
        rows = [metadata(i) for i in range(12)]
        rows[7] = metadata(7, [7, 7, 3, 7])
        result = expansion_plan(rows, INFO, {"train": [1], "validation": [3], "test": [5]}, [0, 9], settings())
        candidates = result["direct_frozen_video_file_exclusion"]["eligible_unused_episode_ids"]
        self.assertEqual(candidates, [2, 4, 6, 8, 10, 11])
        preview = result["bounded_metadata_only_preview"]
        self.assertLessEqual(len(preview), 4)
        self.assertTrue(all(len(row["metadata_audit_preview_episode_ids"]) <= 2 for row in preview))
        self.assertFalse(result["candidate_payloads_downloaded"])

    def test_preview_quantiles_are_fixed_metadata_order_and_repeatable(self):
        rows = [metadata(i, [0] * 4) for i in range(10)] + [metadata(10), metadata(11)]
        split = {"train": [0], "validation": [10], "test": [11]}
        first = expansion_plan(rows, INFO, split, [], settings())
        second = expansion_plan(list(reversed(rows)), INFO, split, [], settings())
        self.assertEqual(first["bounded_metadata_only_preview"], second["bounded_metadata_only_preview"])
        self.assertEqual(first["bounded_metadata_only_preview"][0]["metadata_audit_preview_episode_ids"], [3, 7])

    def test_settings_reject_boolean_bounds_and_unbound_report(self):
        config = {"paths": {"root": "/tmp"}, "trajectory_coverage": settings()}
        self.assertEqual(validate_trajectory_coverage_settings(config)["quarter_bins"], 4)
        for key, value in (("reference_training_report_sha256", "short"), ("quarter_bins", True),
                           ("moving_speed_rad_s", float("nan")), ("expansion_group_limit", True)):
            broken = {**config, "trajectory_coverage": {**settings(), key: value}}
            with self.assertRaises(TrajectoryCoverageError):
                validate_trajectory_coverage_settings(broken)

    def test_failure_receipt_is_written_without_training_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"paths": {"root": str(root)}, "trajectory_coverage": {**settings(), "quarter_bins": 5}}
            result = run_trajectory_coverage(config, root / "run")
            self.assertFalse(result["passed"])
            self.assertFalse(result["training_started"])
            self.assertFalse(result["test_payloads_opened"])
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertTrue((root / "run/reports/trajectory_coverage.json").is_file())
            with self.assertRaisesRegex(TrajectoryCoverageError, "overwritten"):
                run_trajectory_coverage(config, root / "run")

    def test_runner_loads_only_train_npz_and_saved_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = episode()
            train_path = root / "data/train.npz"
            train_path.parent.mkdir()
            np.savez(train_path, source_state=item["state"], source_action=item["action"],
                     source_frame_index=np.arange(24), source_episode_index=np.full(24, 1),
                     source_state_joint_names=np.asarray(SOURCE_JOINT_NAMES),
                     source_action_joint_names=np.asarray(SOURCE_JOINT_NAMES))
            artifact = {"relative_path": "train.npz", "size_bytes": train_path.stat().st_size, "sha256": sha256_file(train_path)}
            train_record = {key: value for key, value in item.items() if key not in ("state", "action")}
            train_record["arrays"] = artifact
            manifest = {"episodes": [train_record, {"episode_id": 2, "split": "validation", "arrays": {"relative_path": "forbidden_val.npz"}},
                                     {"episode_id": 3, "split": "test", "arrays": {"relative_path": "forbidden_test.npz"}}],
                        "chunk_steps": 4, "split_episode_ids": {"train": [1], "validation": [2], "test": [3]}}
            write(root / "data/manifest.json", {})
            indices = np.random.default_rng(12).integers(0, 21, size=(4, 2))
            identities = np.stack([np.ones_like(indices), indices], axis=-1)
            schedule_path = root / "schedule.npz"
            np.savez(schedule_path, dataset_sample_indices=indices, sample_episode_frame=identities)
            reference = {"passed": True, "test_used_for_selection": False, "test_payloads_opened": False,
                "bindings": {"manifest_sha256": sha256_file(root / "data/manifest.json")},
                "anchor_schedule_artifact": {"path": str(schedule_path), "size_bytes": schedule_path.stat().st_size, "sha256": sha256_file(schedule_path)},
                "anchor_schedule": {"steps": 4, "batch_size": 2, "seed": 12, "available_train_anchor_count": 21,
                    "sample_identity_sha256": hashlib.sha256(identities.tobytes()).hexdigest(),
                    "sample_draw_count": 8, "unique_train_anchor_count": len(np.unique(indices))}}
            write(root / "reference.json", reference)
            config = {"paths": {"root": str(root)}, "trajectory_coverage": {**settings(), "reference_training_report_sha256": sha256_file(root / "reference.json")},
                "grasp_act": {"dataset_manifest_path": "data/manifest.json"}, "source_native": {}, "external_data": {"episode_ids": [0]}}
            original_load, opened = np.load, []
            def recorded_load(path, *args, **kwargs):
                opened.append(Path(path).name)
                return original_load(path, *args, **kwargs)
            rows = [metadata(i) for i in range(5)]
            with patch("g1_vla.trajectory_coverage.load_grasp_manifest", return_value=manifest), \
                 patch("g1_vla.trajectory_coverage._metadata", return_value=(INFO, rows, {})), \
                 patch("g1_vla.trajectory_coverage.np.load", side_effect=recorded_load):
                result = run_trajectory_coverage(config, root / "run")
            self.assertTrue(result["passed"], result.get("error"))
            self.assertEqual(opened, ["schedule.npz", "train.npz"])
            self.assertFalse(result["validation_payloads_opened"])
            self.assertFalse(result["candidate_payloads_opened"])
            self.assertEqual(result["scope"]["payload_episode_ids"], [1])


if __name__ == "__main__":
    unittest.main()
