"""Whole-candidate admission, immutable validation and no-test expansion scope."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from g1_vla import train_expansion as expansion
from g1_vla import grasp_data as native
from g1_vla.external_data import SOURCE_CAMERA_KEYS, SOURCE_JOINT_NAMES
from g1_vla.runtime.experiment import canonical_hash


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return digest(path)


def metadata(identity, group):
    item = {"episode_index": identity, "length": 16, "dataset_from_index": identity*20,
        "dataset_to_index": identity*20+16, "data/chunk_index": 0, "data/file_index": 0,
        "tasks": ["camera packaging"]}
    for camera in SOURCE_CAMERA_KEYS:
        prefix = "videos/"+camera
        item.update({prefix+"/chunk_index": 0, prefix+"/file_index": group,
                     prefix+"/from_timestamp": 0., prefix+"/to_timestamp": 16/30.})
    return item


def columns(identity):
    state = np.arange(16*28, dtype=np.float32).reshape(16, 28)*np.float32(.001)+np.float32(identity)
    return {"observation.state": state, "action": state+np.float32(.03),
        "timestamp": (np.arange(16)/30.).astype(np.float32), "frame_index": np.arange(16, dtype=np.int64),
        "episode_index": np.full(16, identity, dtype=np.int64), "index": np.arange(identity*20, identity*20+16, dtype=np.int64),
        "task_index": np.zeros(16, dtype=np.int64)}


class TrainExpansionTests(unittest.TestCase):
    def fixture(self, root):
        old, source, videos = root/"old", root/"source", root/"videos"
        for folder in (old, source, videos):
            folder.mkdir()
        (source/"provenance.json").write_text("source receipt")
        (videos/"provenance.json").write_text("video receipt")
        train, val, test = expansion.ORIGINAL_TRAIN_IDS, expansion.VALIDATION_IDS, expansion.TEST_IDS
        split = {"train": train, "validation": val, "test": test}
        records = []
        for name, identities in split.items():
            for identity in identities:
                base = old/"episodes"/("episode_%06d" % identity)
                base.parent.mkdir(exist_ok=True)
                array, rgb, receipt = base.with_suffix(".source.npz"), base.with_suffix(".rgb.npy"), base.with_suffix(".decoded.json")
                col = columns(identity)
                with array.open("xb") as handle:
                    np.savez(handle, source_state=col["observation.state"], source_action=col["action"])
                with rgb.open("xb") as handle:
                    np.save(handle, np.full((16, 240, 320, 3), identity % 255, dtype=np.uint8), allow_pickle=False)
                receipt.write_text(json.dumps({"episode_id": identity}))
                row = metadata(identity, 0 if name == "train" else 2 if name == "validation" else 3)
                records.append({"episode_id": identity, "split": name, "row_count": 16, "sample_count": 1,
                    "paired_rgb_frames": 16, "excluded_tail_observation_count": 15,
                    "all_camera_video_relative_paths": {camera: "videos/{}/chunk-000/file-{:03d}.mp4".format(camera, row["videos/"+camera+"/file_index"]) for camera in SOURCE_CAMERA_KEYS},
                    "arrays": native._file_record(array, old), "rgb": native._file_record(rgb, old),
                    "decoded_receipt": native._file_record(receipt, old)})
        stats = native.train_only_statistics(old, records, native.CAMERA)
        write_json(old/"train_stats.json", stats)
        manifest = {"passed": True, "backend": native.BACKEND, "episode_ids": train+val+test, "episodes": records,
            "split_episode_ids": split, "statistics": native._file_record(old/"train_stats.json", old),
            "source_repo_id": native.OFFICIAL_REPO, "source_revision": native.OFFICIAL_REVISION,
            "source_fps": 30., "state_dim": 28, "action_dim": 28, "source_state_joint_names": list(SOURCE_JOINT_NAMES),
            "source_action_joint_names": list(SOURCE_JOINT_NAMES), "camera_key": native.CAMERA,
            "chunk_steps": 16, "image_shape_hwc": [240, 320, 3], "image_transform": {"name": native.TRANSFORM},
            "label_alignment": "unchanged_recorded_row_action_t_through_t_plus_chunk_minus_1",
            "source_native_training_eligible": True, "complete_chunks_only": True, "labels_shifted": False,
            "source_image_shape_hwc": [480, 640, 3], "split_unit": "source_video_file_group",
            "current_simulator_adapter_ready": False, "closed_loop_success_demonstrated": False,
            "sensor_exposure_clock_verified": False, "motor_application_clock_verified": False,
            "original_state_action_float32_preserved": True,
            "actions_clipped": False, "waist_fabricated": False, "time_resampled": False,
            "prior_diagnostic_episodes_excluded": [0, 150, 300], "verified_source_files": {}, "verified_video_files": {}}
        manifest_sha = write_json(old/"manifest.json", manifest)
        coverage = {"passed": True, "test_payloads_opened": False, "test_metrics_read": False,
            "bindings": {"manifest_sha256": manifest_sha}, "expansion": {"bounded_metadata_only_preview": [
                {"metadata_audit_preview_episode_ids": expansion.CANDIDATE_IDS}]}}
        residual = {"passed": True, "test_payloads_opened": False, "test_used_for_selection": False,
            "bindings": {"manifest_sha256": manifest_sha, "statistics_sha256": manifest["statistics"]["sha256"]}}
        sessions = {"repo_id": native.OFFICIAL_REPO, "revision": native.OFFICIAL_REVISION,
            "candidate_episode_ids": expansion.CANDIDATE_IDS, "same_source_coverage_only": True,
            "capture_session_independence_verified": False, "episode_session_mapping_available": False,
            "source_public_multi_session_statement": True,
            "candidate_sessions": {str(identity): {"session_id": None, "verified": False} for identity in expansion.CANDIDATE_IDS}}
        settings = {"backend": expansion.BACKEND, "candidate_episode_ids": expansion.CANDIDATE_IDS,
            "reference_coverage_report_path": "coverage.json", "reference_coverage_report_sha256": write_json(root/"coverage.json", coverage),
            "reference_residual_report_path": "residual.json", "reference_residual_report_sha256": write_json(root/"residual.json", residual),
            "session_evidence_path": "sessions.json", "session_evidence_sha256": write_json(root/"sessions.json", sessions),
            "candidate_build_dir": "candidates", "build_dir": "expanded", "audit_report_path": "audit/reports/train_expansion_audit.json",
            "max_total_rows": 20000, "max_candidate_derived_bytes": 2684354560, "max_derived_bytes": 5368709120,
            "allow_same_source_coverage": True, "require_independent_sessions": False}
        native_settings = {"split_episode_ids": split, "build_dir": "old", "video_dir": "videos", "chunk_steps": 16,
            "image_width": 320, "image_height": 240, "timestamp_tolerance_s": 1e-4, "max_episode_rows": 3000}
        config = {"paths": {"root": str(root)}, "train_expansion": settings, "source_native": native_settings,
            "external_data": {"local_dir": "source", "repo_id": native.OFFICIAL_REPO, "revision": native.OFFICIAL_REVISION},
            "grasp_act": {"dataset_manifest_path": "old/manifest.json"}}
        features = {name: {"dtype": "float32", "shape": [28], "names": [list(SOURCE_JOINT_NAMES)]} for name in ("observation.state", "action")}
        features.update({camera: {"dtype": "video", "info": {"video.width": 640, "video.height": 480, "video.fps": 30., "video.codec": "av1"}} for camera in SOURCE_CAMERA_KEYS})
        info = {"features": features, "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
        rows = [metadata(item["episode_id"], 0 if item["split"] == "train" else 2 if item["split"] == "validation" else 3) for item in records]
        rows += [metadata(identity, 1) for identity in expansion.CANDIDATE_IDS]
        return config, manifest, info, rows

    def run_audit(self, root, config, manifest, info, rows, *, bad_dtype=False):
        (root/"audit").mkdir(exist_ok=True)
        def decoder(path, times, output, **kwargs):
            with output.open("xb") as handle:
                np.save(handle, np.full((len(times), 240, 320, 3), 100, dtype=np.uint8), allow_pickle=False)
            return {"passed": True, "paired_frame_count": len(times), "maximum_absolute_pts_error_s": 0.,
                "frames": [{"decoded_source_pts_s": float(value), "decoded_pts_ticks": index} for index, value in enumerate(times)]}
        def values(path, identity):
            result = columns(identity)
            if bad_dtype:
                result["action"] = result["action"].astype(np.float64)
            return result
        def receipt(folder, repo, revision, names):
            return {name: {"sha256": "1"*64, "size_bytes": 100} for name in names}
        with patch.object(native, "load_grasp_manifest", return_value=manifest), \
             patch.object(expansion.coverage, "_metadata", return_value=(info, rows, {"materialized_columns": ["episode_index"]})), \
             patch.object(expansion, "verify_download_receipt", side_effect=receipt), \
             patch.object(expansion, "_parquet_columns", side_effect=values), \
             patch.object(native, "decode_episode_rgb", side_effect=decoder):
            return expansion.audit_train_expansion(config, root/"audit")

    def test_predeclared_ids_heldout_scope_and_same_source_semantics_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, info, rows = self.fixture(Path(directory))
            self.assertEqual(expansion.validate_train_expansion_settings(config)["candidate_episode_ids"], expansion.CANDIDATE_IDS)
            for key, value in (("candidate_episode_ids", list(reversed(expansion.CANDIDATE_IDS))),
                               ("allow_same_source_coverage", False), ("require_independent_sessions", True),
                               ("candidate_build_dir", "old/inside"), ("max_total_rows", True)):
                bad = copy.deepcopy(config)
                bad["train_expansion"][key] = value
                with self.subTest(key=key), self.assertRaises(expansion.TrainExpansionError):
                    expansion.validate_train_expansion_settings(bad)
            bad = copy.deepcopy(config)
            bad["source_native"]["split_episode_ids"]["validation"] = [180, 182, 181]
            with self.assertRaisesRegex(expansion.TrainExpansionError, "frozen"):
                expansion.validate_train_expansion_settings(bad)

    def test_unused_camera_frozen_file_overlap_and_global_row_overlap_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config, manifest, info, rows = self.fixture(Path(directory))
            result = expansion.verify_candidate_groups(rows, info, manifest)
            self.assertEqual(len(result), 8)
            self.assertTrue(all(item["admission_class"] == "same_source_coverage_only" for item in result))
            altered = copy.deepcopy(rows)
            altered[-1]["videos/observation.images.cam_right_wrist/file_index"] = 2
            with self.assertRaisesRegex(expansion.TrainExpansionError, "shares a frozen"):
                expansion.verify_candidate_groups(altered, info, manifest)
            altered = copy.deepcopy(rows)
            altered[-1]["dataset_from_index"] = rows[0]["dataset_from_index"]
            altered[-1]["dataset_to_index"] = rows[0]["dataset_to_index"]
            with self.assertRaisesRegex(expansion.TrainExpansionError, "overlap"):
                expansion.verify_candidate_groups(altered, info, manifest)

    def test_complete_audit_and_build_preserve_validation_bytes_and_never_open_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, info, rows = self.fixture(root)
            # Any path opening a test payload is a hard test failure, including
            # array loads, verification and copy. Metadata rows remain allowed.
            forbidden = {str(root/"old"/item[key]["relative_path"]) for item in manifest["episodes"] if item["split"] == "test" for key in expansion.ARTIFACT_KEYS}
            original_open = Path.open
            opened = []
            def checked_open(path, *args, **kwargs):
                opened.append(str(path))
                if str(path) in forbidden:
                    raise AssertionError("test payload opened")
                return original_open(path, *args, **kwargs)
            before = {item["episode_id"]: {key: (root/"old"/item[key]["relative_path"]).read_bytes() for key in expansion.ARTIFACT_KEYS} for item in manifest["episodes"] if item["split"] == "validation"}
            with patch.object(Path, "open", checked_open):
                audited = self.run_audit(root, config, manifest, info, rows)
                self.assertTrue(audited["passed"], audited.get("error"))
                self.assertFalse(audited["independent_session_data_admitted"])
                (root/"buildrun").mkdir()
                with patch.object(native, "load_grasp_manifest", return_value=manifest):
                    built = expansion.build_train_expansion(config, root/"buildrun")
                self.assertTrue(built["passed"], built.get("error"))
                expanded = expansion.load_expanded_manifest(root/"expanded", verify_files=True)
                self.assertNotIn("test", expanded["split_episode_ids"])
                self.assertEqual(expanded["split_episode_ids"]["validation"], expansion.VALIDATION_IDS)
                self.assertEqual(len(expanded["split_episode_ids"]["train"]), 14)
                for item in expanded["episodes"]:
                    if item["split"] == "validation":
                        for key in expansion.ARTIFACT_KEYS:
                            self.assertEqual((root/"expanded"/item[key]["relative_path"]).read_bytes(), before[item["episode_id"]][key])
                dataset = native.GraspSquareDataset(root/"expanded", "train", manifest=expanded, verify=False)
                np.testing.assert_array_equal(dataset[6]["observation.state"], columns(12)["observation.state"][0])
                statistics = json.loads((root/"expanded/train_stats.json").read_text())
                self.assertEqual(statistics["training_episode_ids"], expansion.ORIGINAL_TRAIN_IDS+expansion.CANDIDATE_IDS)
                self.assertEqual(statistics["features"]["action"]["count"], 14*16)
                expected_actions = np.concatenate([columns(identity)["action"] for identity in expansion.ORIGINAL_TRAIN_IDS+expansion.CANDIDATE_IDS]).astype(np.float64)
                np.testing.assert_allclose(statistics["features"]["action"]["mean"], expected_actions.mean(axis=0), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(statistics["features"]["action"]["std"], expected_actions.std(axis=0), rtol=1e-12, atol=1e-12)
                self.assertFalse(statistics["validation_or_test_used"])
                # Self-asserting session independence and altered native image
                # semantics are rejected even before any model can be built.
                saved_manifest = (root/"expanded/manifest.json").read_bytes()
                for key, value in (("source_image_shape_hwc", [240, 320, 3]),
                                   ("sensor_exposure_clock_verified", True),
                                   ("independent_session_data_admitted", True),
                                   ("metadata_excluded_test_episode_ids", [])):
                    modified = copy.deepcopy(expanded)
                    modified[key] = value
                    write_json(root/"expanded/manifest.json", modified)
                    with self.subTest(key=key), self.assertRaises(expansion.TrainExpansionError):
                        expansion.load_expanded_manifest(root/"expanded", verify_files=False)
                (root/"expanded/manifest.json").write_bytes(saved_manifest)
                audited_bytes = (root/"audit/reports/train_expansion_audit.json").read_bytes()
                modified_audit = json.loads(audited_bytes)
                modified_audit["passed"] = False
                write_json(root/"audit/reports/train_expansion_audit.json", modified_audit)
                with self.assertRaisesRegex(expansion.TrainExpansionError, "immutable candidate audit"):
                    expansion.load_expanded_manifest(root/"expanded", verify_files=False)
                (root/"audit/reports/train_expansion_audit.json").write_bytes(audited_bytes)
            self.assertFalse(forbidden.intersection(opened))
            # Reusing a completed destination may fail, but cannot mark the
            # successful immutable dataset failed or alter its bytes.
            manifest_before = (root/"expanded/manifest.json").read_bytes()
            (root/"retry").mkdir()
            with patch.object(native, "load_grasp_manifest", return_value=manifest):
                retried = expansion.build_train_expansion(config, root/"retry")
            self.assertFalse(retried["passed"])
            self.assertFalse((root/"expanded/failed.json").exists())
            self.assertEqual((root/"expanded/manifest.json").read_bytes(), manifest_before)

    def test_candidate_dtype_failure_blocks_all_admission_and_preserves_failed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, info, rows = self.fixture(root)
            report = self.run_audit(root, config, manifest, info, rows, bad_dtype=True)
            self.assertFalse(report["passed"])
            self.assertIn("float32", report["error"]["message"])
            self.assertTrue((root/"candidates/failed.json").exists())
            self.assertFalse((root/"candidates/candidate_manifest.json").exists())
            self.assertFalse(report["training_started"])

    def test_session_hash_or_fabricated_episode_session_mapping_blocks_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, info, rows = self.fixture(root)
            evidence = json.loads((root/"sessions.json").read_text())
            evidence["candidate_sessions"]["12"] = {"session_id": "invented", "verified": True}
            config["train_expansion"]["session_evidence_sha256"] = write_json(root/"sessions.json", evidence)
            result = self.run_audit(root, config, manifest, info, rows)
            self.assertFalse(result["passed"])
            self.assertIn("unknown per-episode", result["error"]["message"])
            self.assertFalse((root/"candidates").exists())

    def test_reference_hash_failure_never_creates_candidate_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, info, rows = self.fixture(root)
            (root/"coverage.json").write_text("changed reference")
            result = self.run_audit(root, config, manifest, info, rows)
            self.assertFalse(result["passed"])
            self.assertIn("predeclared hash", result["error"]["message"])
            self.assertFalse((root/"candidates").exists())
            self.assertFalse(result["training_started"])

    def test_candidate_must_not_duplicate_an_existing_raw_state_action_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest, info, rows = self.fixture(root)
            item = next(item for item in manifest["episodes"] if item["split"] == "train")
            col = columns(item["episode_id"])
            summary = expansion._array_summary(col["observation.state"], col["action"])
            with self.assertRaisesRegex(expansion.TrainExpansionError, "exactly duplicates"):
                expansion._reject_exact_duplicate_trajectories(root/"old", manifest,
                    [{"episode_id": 12, "row_count": 16, "payload_audit": summary}])


if __name__ == "__main__":
    unittest.main()
