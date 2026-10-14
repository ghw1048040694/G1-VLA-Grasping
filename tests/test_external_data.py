"""External samples stay original diagnostics until an adapter is established."""

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from g1_vla.external_data import (
    ExternalDataError, OFFICIAL_REPO, SOURCE_CAMERA_KEYS, SOURCE_JOINT_NAMES,
    SOURCE_TARGET_NAMES, audit_official_sample, camera_mapping, inspect_episode,
    joint_mapping, read_xml_joint_ranges,
)


PROJECT = Path(__file__).resolve().parents[1]
TARGET_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                *SOURCE_TARGET_NAMES[:7], *SOURCE_TARGET_NAMES[14:21],
                *SOURCE_TARGET_NAMES[7:14], *SOURCE_TARGET_NAMES[21:])


def metadata():
    features = {key: {"dtype": "float32", "shape": [28], "names": [list(SOURCE_JOINT_NAMES)]}
                for key in ("observation.state", "action")}
    features.update({key: {"dtype": "video", "shape": [3, 480, 640],
                         "info": {"video.width": 640, "video.height": 480, "video.fps": 30.0, "video.codec": "av1"}}
                     for key in SOURCE_CAMERA_KEYS})
    return {"codebase_version": "v3.0", "fps": 30.0,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features}


def columns(episode_id=0, rows=5):
    state = np.arange(rows * 28, dtype=np.float32).reshape(rows, 28) * np.float32(0.001)
    return {"observation.state": state, "action": state + np.float32(0.005),
            "timestamp": (np.arange(rows) / 30).astype(np.float32),
            "frame_index": np.arange(rows, dtype=np.int64),
            "episode_index": np.full(rows, episode_id, dtype=np.int64),
            "index": np.arange(episode_id * rows, episode_id * rows + rows, dtype=np.int64),
            "task_index": np.zeros(rows, dtype=np.int64)}


def episode_metadata(episode_id=0, rows=5):
    result = {"episode_index": episode_id, "length": rows,
              "dataset_from_index": episode_id * rows, "dataset_to_index": episode_id * rows + rows,
              "data/chunk_index": 0, "data/file_index": 0, "tasks": ["Stack red, yellow and blue cubes."]}
    for camera in SOURCE_CAMERA_KEYS:
        prefix = "videos/" + camera
        result.update({prefix + "/chunk_index": 0, prefix + "/file_index": 0,
                       prefix + "/from_timestamp": episode_id * rows / 30,
                       prefix + "/to_timestamp": (episode_id + 1) * rows / 30})
    return result


class ExternalDataTests(unittest.TestCase):
    def test_joint_names_define_mapping_and_missing_waist_stays_missing(self):
        info = metadata()
        info["features"]["action"]["names"][0].reverse()
        result = joint_mapping(info, TARGET_NAMES)
        self.assertEqual(result["missing_source_joints"], list(TARGET_NAMES[:3]))
        right_index = next(row for row in result["correspondence"] if row["source_name"] == "kRightHandIndex0")
        self.assertEqual(right_index["target_index"], TARGET_NAMES.index("right_hand_index_0_joint"))
        self.assertEqual(right_index["state_index"], 24)
        self.assertEqual(right_index["action_index"], 3)
        self.assertIsNone(right_index["sign"])
        self.assertFalse(result["deployable_adapter"])
        self.assertFalse(result["fabricated_values"])

    def test_duplicate_unknown_and_absent_named_identity_rejected(self):
        for replacement in ("kLeftShoulderPitch", "invented"):
            info = metadata()
            info["features"]["action"]["names"][0][-1] = replacement
            with self.assertRaises(ExternalDataError):
                joint_mapping(info, TARGET_NAMES)
        with self.assertRaisesRegex(ExternalDataError, "31 unique"):
            joint_mapping(metadata(), TARGET_NAMES[:-1])

    def test_binocular_head_does_not_become_an_exact_single_camera(self):
        config = json.loads((PROJECT / "configs/experiment.json").read_text())
        result = camera_mapping(metadata(), config)
        self.assertEqual(result["current_width"], 96)
        self.assertEqual(result["current_height"], 64)
        self.assertFalse(any(row["exact_match"] for row in result["source"]))
        head = [row for row in result["source"] if "head_eye" in row["source_role"]]
        self.assertEqual(len(head), 2)
        self.assertTrue(all(row["current_candidate"] == "observation.images.head" for row in head))
        self.assertFalse(result["images_relabelled"])

    def _inspect(self, source=None, **kwargs):
        return inspect_episode(columns() if source is None else source,
            episode_id=0, fps=30, tolerance_s=1e-4, max_rows=3000,
            mapping=joint_mapping(metadata(), TARGET_NAMES), **kwargs)

    def test_original_source_arrays_and_dtype_are_preserved_without_waist(self):
        source = columns()
        report, saved = self._inspect(source)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(saved["source_state"].shape, (5, 28))
        self.assertEqual(saved["source_action"].dtype, np.float32)
        np.testing.assert_array_equal(saved["source_state"], source["observation.state"])
        np.testing.assert_array_equal(saved["source_timestamp_s"], source["timestamp"])
        self.assertFalse(report["training_eligible"])
        self.assertFalse(report["waist_values_fabricated"])
        self.assertFalse(report["timing"]["observation_action_order_verified"])

    def test_numeric_correlations_do_not_certify_next_command_pairing(self):
        source = columns()
        source["action"] = source["observation.state"].copy()
        report, _ = self._inspect(source)
        self.assertEqual(report["action_state_residuals"]["same_row_action_minus_state"]["rms"], 0)
        self.assertFalse(report["timing"]["motor_application_timestamp_provided"])
        self.assertFalse(report["training_eligible"])

    def test_timestamp_drops_and_nonfinite_are_failed_diagnostics(self):
        for column, index, value in (("frame_index", 2, 3), ("timestamp", 2, 0.5), ("timestamp", 2, np.nan)):
            source = columns()
            source[column][index] = value
            with self.subTest(column=column, value=value):
                result, _ = self._inspect(source)
                self.assertEqual(result["status"], "failed")
        source = columns()
        source["action"][2, 5] = np.inf
        result, _ = self._inspect(source)
        self.assertEqual(result["status"], "failed")
        json.dumps(result, allow_nan=False)

    def test_bad_row_identity_size_shape_and_index_dtype_rejected(self):
        for key, replacement in (("episode_index", np.ones(5, dtype=np.int64)),
                                 ("frame_index", np.arange(5, dtype=np.float64)),
                                 ("action", np.zeros((5, 31))),
                                 ("index", np.zeros(4, dtype=np.int64))):
            source = columns()
            source[key] = replacement
            with self.subTest(key=key), self.assertRaises(ExternalDataError):
                self._inspect(source)
        with self.assertRaisesRegex(ExternalDataError, "row count"):
            self._inspect(columns(rows=3001))

    def test_ranges_and_speed_caps_are_explicitly_unverified_hypotheses(self):
        source = columns()
        source["action"][2, 0] = 4
        report, _ = self._inspect(source, joint_ranges={"left_shoulder_pitch_joint": (-3, 3)},
                                  current_speed_limits=[0.0] * 31)
        shoulder = report["joint_diagnostics"][0]
        self.assertEqual(shoulder["identity_sign_zero_range_hypothesis"]["action_outside_count"], 1)
        self.assertFalse(shoulder["identity_sign_zero_range_hypothesis"]["verified_conversion"])
        self.assertGreater(shoulder["current_speed_limit_hypothesis"]["sampled_intervals_above_count"], 0)
        self.assertFalse(report["actions_clipped"])
        self.assertEqual(source["action"][2, 0], 4)

    def test_xml_ranges_require_declared_radians_and_unique_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.xml"
            path.write_text('<mujoco><compiler angle="degree"/><worldbody><joint name="a" range="-1 1"/></worldbody></mujoco>')
            with self.assertRaisesRegex(ExternalDataError, "radian"):
                read_xml_joint_ranges(path)
            path.write_text('<mujoco><compiler angle="radian"/><worldbody><joint name="a" range="-1 1"/><joint name="a" range="-1 1"/></worldbody></mujoco>')
            with self.assertRaisesRegex(ExternalDataError, "duplicate"):
                read_xml_joint_ranges(path)

    def _fixture(self, root, *, probe=False):
        config = json.loads((PROJECT / "configs/experiment.json").read_text())
        config["paths"]["root"] = str(root)
        config["action"]["joint_metadata_path"] = "target.json"
        config["scene"]["asset_path"] = "scene.xml"
        config["external_data"] = {"repo_id": OFFICIAL_REPO, "revision": "a" * 40, "local_dir": "source",
            "episode_ids": [0, 1, 2], "source_fps": 30, "max_episode_rows": 3000,
            "timestamp_tolerance_s": 1e-4, "semantic_evidence_path": None, "probe_videos": probe,
            "video": {"backend": "pyav_http_range", "samples_per_episode": 3, "max_read_bytes": 1048576}}
        (root / "target.json").write_text(json.dumps({"features": {"action": {"shape": [31], "names": list(TARGET_NAMES)}}}))
        xml_joints = ''.join('<joint name="{}" range="-3 3"/>'.format(name) for name in TARGET_NAMES)
        (root / "scene.xml").write_text('<mujoco><compiler angle="radian"/><worldbody>' + xml_joints + '</worldbody></mujoco>')
        source = root / "source"
        (source / "meta/episodes/chunk-000").mkdir(parents=True)
        (source / "meta/info.json").write_text(json.dumps(metadata()))
        (source / "meta/episodes/chunk-000/file-000.parquet").write_bytes(b"fixture episode parquet")
        (source / "data/chunk-000").mkdir(parents=True)
        (source / "data/chunk-000/file-000.parquet").write_bytes(b"fixture data parquet")
        from g1_vla.external_sources import DATA_FILES
        for name in ("README.md", "meta/stats.json", "meta/tasks.parquet"):
            (source / name).write_bytes(b"bounded fixture auxiliary source")
        entries = []
        for name in DATA_FILES:
            path = source / name
            entries.append({"relative_path": name, "path": str(path), "size_bytes": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "expected_lfs_sha256": None, "full_file_verified": True})
        (source / "provenance.json").write_text(json.dumps({"repo_id": OFFICIAL_REPO, "revision": "a" * 40,
            "repo_type": "dataset", "anonymous": True, "training_started": False, "files": entries}))
        return config

    def _run_fixture(self, root, config):
        with patch("g1_vla.external_data._parquet_rows", side_effect=lambda path, **kw: [episode_metadata(kw["filters"][0][2])]), \
             patch("g1_vla.external_data._parquet_columns", side_effect=lambda path, episode_id: columns(episode_id)):
            return audit_official_sample(config, root / "run")

    def test_full_audit_keeps_diagnostic_artifacts_and_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root)
            original = copy.deepcopy(config)
            report = self._run_fixture(root, config)
            self.assertEqual(report["episodes_structurally_passed"], 3)
            self.assertEqual(report["bounded_video_probes_total"], 12)
            self.assertEqual(report["bounded_video_probes_passed"], 0)
            self.assertFalse(report["training_started"])
            self.assertFalse(report["current_adapter_ready"])
            self.assertTrue(report["source_receipt_verified"])
            self.assertIn("data/chunk-000/file-000.parquet", report["verified_source_files"])
            self.assertFalse(report["source_native_training_review"]["current_teacher_failure_invalidates_official_source"])
            self.assertEqual(config, original)
            self.assertEqual(len(list((root / "run/official_samples").glob("*.npz"))), 3)
            with np.load(root / "run/official_samples/episode_000000.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["source_action"].shape[1], 28)
                self.assertEqual(saved["source_action"].dtype, np.float32)
                self.assertFalse(saved["training_eligible"])
            self.assertTrue((root / "run/reports/external_data_audit.json").is_file())
            with self.assertRaises(FileExistsError):
                self._run_fixture(root, config)

    def test_video_probes_share_budget_and_use_real_episode_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, probe=True)
            with patch("g1_vla.video_audit.probe_video_samples", return_value={"status": "sampled_format_and_pts_passed", "passed": True}) as probe:
                report = self._run_fixture(root, config)
            self.assertEqual(report["bounded_video_probes_passed"], 12)
            budgets = [call.kwargs["budget"] for call in probe.call_args_list]
            self.assertTrue(all(budget is budgets[0] for budget in budgets))
            last_expected = probe.call_args_list[-1].args[4]
            source = columns(2)
            np.testing.assert_allclose(last_expected, 10 / 30 + source["timestamp"][[0, 2, 4]])
            self.assertFalse(report["episodes"][2]["videos"][0]["sensor_exposure_alignment_verified"])

    def test_video_status_label_alone_cannot_claim_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root, probe=True)
            with patch("g1_vla.video_audit.probe_video_samples", return_value={"status": "passed", "passed": False}):
                report = self._run_fixture(root, config)
            self.assertEqual(report["bounded_video_probes_passed"], 0)

    def test_floating_revision_and_nonofficial_repository_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root)
            for field, value in (("revision", "main"), ("repo_id", "other/dataset"), ("episode_ids", [0, 0])):
                changed = copy.deepcopy(config)
                changed["external_data"][field] = value
                with self.subTest(field=field), self.assertRaises(ExternalDataError):
                    audit_official_sample(changed, root / "run")

    def test_cache_hash_corruption_cannot_claim_pinned_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root)
            path = root / "source/data/chunk-000/file-000.parquet"
            original = path.read_bytes()
            path.write_bytes(original[:-1] + b"X")
            with self.assertRaisesRegex(ValueError, "hash|SHA"):
                audit_official_sample(config, root / "run")
            self.assertFalse((root / "run/official_samples").exists())

    def test_cache_receipt_revision_mismatch_cannot_claim_configured_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root)
            path = root / "source/provenance.json"
            receipt = json.loads(path.read_text())
            receipt["revision"] = "b" * 40
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "revision|identity|provenance"):
                audit_official_sample(config, root / "run")
            self.assertFalse((root / "run/official_samples").exists())


@unittest.skipUnless(importlib.util.find_spec("pyarrow") is not None, "pyarrow required for physical parquet dtype check")
class ActualParquetTests(unittest.TestCase):
    def test_nested_float32_source_columns_remain_float32(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from g1_vla.external_data import _parquet_columns

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            source = columns()
            data = {key: pa.array(value.tolist(), type=pa.list_(pa.float32())) if value.ndim == 2 else pa.array(value)
                    for key, value in source.items()}
            pq.write_table(pa.table(data), path)
            result = _parquet_columns(path, 0)
            self.assertEqual(result["action"].dtype, np.float32)
            self.assertEqual(result["timestamp"].dtype, np.float32)
            np.testing.assert_array_equal(result["action"], source["action"])


if __name__ == "__main__":
    unittest.main()
