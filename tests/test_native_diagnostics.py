"""Diagnostic denominator, original-label and untouched-test-scope checks."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from g1_vla.artifacts import sha256_file
from g1_vla.external_data import SOURCE_JOINT_NAMES
from g1_vla.native_diagnostics import (
    NativeDiagnosticsError, _label_accessor_checks, _raw_parquet_checks, _verify_episode, angle_metrics,
    horizon_diagnostics, lag_diagnostics, replay_training_coverage, run_native_diagnostics,
)


def episode(identity=1, rows=40, split="train", chunk=16):
    state = np.broadcast_to(np.arange(rows, dtype=np.float32)[:, None] * np.float32(.01), (rows, 28)).copy()
    return {"episode_id": identity, "row_count": rows, "sample_count": rows - chunk + 1, "split": split,
            "arrays": {"source_state": state, "source_action": state + np.float32(.02)}}


def training_receipt(steps=4):
    return {"optimizer_steps": steps, "steps": [{"step": step, "loss": float(10 / step),
        "loss_components": {"l1_loss": float(1 / step), "kld_loss": float(.9 / step)}}
        for step in range(1, steps + 1)], "validation": [{"step": steps, "metrics": {
            "policy": {"all_28": {"mae": .5}}, "state_hold": {"all_28": {"mae": .1}},
            "train_mean": {"all_28": {"mae": .8}}}}], "selected_step": steps}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def file_record(root, path):
    return {"relative_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def save_episode(root, item):
    identity, count = item["episode_id"], item["row_count"]
    arrays = item["arrays"]
    times = (np.arange(count) / 30.).astype(np.float32)
    pts = np.arange(count) / 30.
    arrays.update(source_timestamp_s=times, source_frame_index=np.arange(count, dtype=np.int64),
                  source_episode_index=np.full(count, identity, dtype=np.int64),
                  source_global_index=np.arange(count, dtype=np.int64) + identity * 1000,
                  source_task_index=np.zeros(count, dtype=np.int64), decoded_source_pts_s=pts,
                  decoded_source_pts_ticks=np.arange(count, dtype=np.int64),
                  source_state_joint_names=np.asarray(SOURCE_JOINT_NAMES),
                  source_action_joint_names=np.asarray(SOURCE_JOINT_NAMES))
    array_path = root / ("episode_{}.npz".format(identity))
    np.savez(array_path, **arrays)
    receipt = {"passed": True, "episode_id": identity, "frames": [], "paired_frame_count": count,
               "frame_reuse_or_padding": False, "resampled_in_time": False,
               "sensor_exposure_clock_verified": False, "stream": {"fps": 30., "codec_id": 225},
               "source_video_from_timestamp_s": 0.}
    for frame in range(count):
        error = abs(float(pts[frame]) - float(times[frame]))
        receipt["frames"].append({"frame_index": frame, "query_source_pts_s": float(times[frame]),
            "decoded_source_pts_s": float(pts[frame]), "decoded_pts_ticks": frame,
            "decoded_time_base": "1/30", "absolute_pts_error_s": error,
            "original_rgb_sha256": "a" * 64, "model_rgb_sha256": "b" * 64})
    receipt_path = root / ("episode_{}.decoded.json".format(identity))
    write_json(receipt_path, receipt)
    image_path = root / ("episode_{}.rgb.npy".format(identity))
    images = np.broadcast_to(np.arange(count, dtype=np.uint8)[:, None, None, None], (count, 3, 4, 3)).copy()
    np.save(image_path, images, allow_pickle=False)
    return {**{key: value for key, value in item.items() if key != "arrays"},
            "arrays": file_record(root, array_path), "decoded_receipt": file_record(root, receipt_path),
            "rgb": file_record(root, image_path), "data_relative_path": "data/source.parquet"}


class NativeDiagnosticsTests(unittest.TestCase):
    def test_lags_use_identical_interior_population_and_sign(self):
        item = episode()
        result = lag_diagnostics([item], list(range(-3, 4)))
        self.assertEqual({entry["row_count"] for entry in result["by_lag"].values()}, {33})
        self.assertEqual(result["minimum_mae_lag_diagnostic_only"], -2)
        self.assertFalse(result["labels_modified"])
        self.assertFalse(result["hardware_timing_identified"])

    def test_lag_grid_cannot_drop_unfavorable_offsets(self):
        with self.assertRaises(NativeDiagnosticsError):
            lag_diagnostics([episode()], [0, 1])

    def test_constant_joint_correlation_is_null(self):
        item = episode()
        item["arrays"]["source_state"][:, 2] = 2
        item["arrays"]["source_action"][:, 2] = 3
        result = lag_diagnostics([item], list(range(-3, 4)))
        self.assertIsNone(result["by_lag"]["0"]["position_correlation_by_joint"][2])

    def test_horizons_keep_denominators_and_first_row_velocity_unknown(self):
        item = episode(rows=20)
        result = horizon_diagnostics([item], np.zeros(28), 16, .02, .1)
        self.assertEqual(result["anchor_count"], 5)
        self.assertEqual(result["stratum_anchor_counts"], {"all": 5, "stationary": 0,
                         "moving": 4, "intermediate": 0, "unknown_first_row": 1})
        self.assertEqual([entry["strata"]["all"]["state_hold"]["row_count"] for entry in result["by_horizon"]], [5] * 16)
        self.assertLess(result["by_horizon"][0]["strata"]["all"]["state_hold"]["all_28"]["mae"],
                        result["by_horizon"][-1]["strata"]["all"]["state_hold"]["all_28"]["mae"])
        self.assertIsNone(result["by_horizon"][0]["strata"]["stationary"]["state_hold"]["all_28"])

    def test_stationary_moving_and_intermediate_partition(self):
        item = episode(rows=20)
        item["arrays"]["source_state"][:3] = 0
        item["arrays"]["source_state"][3] = np.float32(.002)
        result = horizon_diagnostics([item], np.zeros(28), 16, .02, .1)
        counts = result["stratum_anchor_counts"]
        self.assertEqual(sum(counts[key] for key in counts if key != "all"), counts["all"])
        self.assertEqual(counts["stationary"], 2)
        self.assertEqual(counts["intermediate"], 1)

    def test_replayed_draws_match_direct_rng_and_count_labels_separately(self):
        items = [episode(1), episode(2)]
        result = replay_training_coverage(items, training_receipt(), 123, 2, 16, 2)
        direct = np.random.default_rng(123).integers(0, 50, size=8)
        self.assertEqual(result["anchor_draw_count"], 8)
        self.assertEqual(result["unique_anchor_count"], len(np.unique(direct)))
        self.assertAlmostEqual(result["equivalent_anchor_epochs"], 8 / 50)
        self.assertEqual(sum(row["draw_count"] for row in result["per_episode"].values()), 8)
        self.assertEqual(sum(sum(row["action_label_visits_by_source_frame"]) for row in result["per_episode"].values()), 8 * 16)
        self.assertFalse(result["training_sufficient_proven"])
        self.assertEqual(result["loss_windows"]["first"]["end_step"], 2)

    def test_replay_rejects_incomplete_training_receipt(self):
        receipt = training_receipt()
        receipt["steps"].pop()
        with self.assertRaises(NativeDiagnosticsError):
            replay_training_coverage([episode()], receipt, 123, 2, 16, 2)

    def test_empty_strata_do_not_emit_nan_and_invalid_angles_rejected(self):
        result = angle_metrics(np.empty((0, 28)), np.empty((0, 28)))
        self.assertIsNone(result["all_28"])
        json.dumps(result, allow_nan=False)
        with self.assertRaises(NativeDiagnosticsError):
            angle_metrics(np.zeros((2, 27)), np.zeros((2, 27)))

    def test_episode_rejects_changed_pts_even_with_new_artifact_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = save_episode(root, episode())
            checked, receipt = _verify_episode(root, record, 16, .0001)
            self.assertEqual(receipt["unique_pts_count"], 40)
            path = root / record["decoded_receipt"]["relative_path"]
            saved = json.loads(path.read_text())
            saved["frames"][2]["decoded_pts_ticks"] = 1
            write_json(path, saved)
            record["decoded_receipt"] = file_record(root, path)
            with self.assertRaisesRegex(NativeDiagnosticsError, "PTS receipt"):
                _verify_episode(root, record, 16, .0001)

    def test_episode_rejects_dtype_and_joint_order_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item = episode()
            item["arrays"]["source_action"] = item["arrays"]["source_action"].astype(np.float64)
            with self.assertRaisesRegex(NativeDiagnosticsError, "float32"):
                _verify_episode(root, save_episode(root, item), 16, .0001)

    def test_episode_rejects_mutated_rgb_file_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = save_episode(root, episode())
            images = np.load(root / record["rgb"]["relative_path"], mmap_mode="r+")
            images[0, 0, 0, 0] ^= np.uint8(1)
            images.flush()
            del images
            with self.assertRaisesRegex(ValueError, "immutable manifest"):
                _verify_episode(root, record, 16, .0001)

    def test_label_accessor_rejects_rgb_frame_shift(self):
        from g1_vla.grasp_data import GraspSquareDataset
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest = self.fixture(root)
            episodes = [_verify_episode(root, item, 16, .0001)[0]
                        for item in manifest["episodes"] if item["split"] != "test"]
            original_getitem = GraspSquareDataset.__getitem__
            def shifted_getitem(dataset, index):
                sample = original_getitem(dataset, index)
                episode_id, frame = dataset.samples[index]
                sample[dataset.camera_key] = np.ascontiguousarray(
                    dataset._images[episode_id][frame + 1].transpose(2, 0, 1), dtype=np.float32) / 255.0
                return sample
            with patch.object(GraspSquareDataset, "__getitem__", shifted_getitem):
                with self.assertRaisesRegex(NativeDiagnosticsError, "verified RGB\\[t\\]"):
                    _label_accessor_checks(root, manifest, episodes)

    def test_original_parquet_values_and_dtype_are_independently_checked(self):
        item = episode()
        item["data_relative_path"] = "data/source.parquet"
        arrays = item["arrays"]
        for name in ("source_timestamp_s", "source_frame_index", "source_episode_index", "source_global_index", "source_task_index"):
            arrays[name] = np.arange(40)
        from g1_vla.native_diagnostics import ARRAY_COLUMNS
        columns = {name: arrays[key].copy() for name, key in ARRAY_COLUMNS.items()}
        columns["action"][0, 0] += 1
        camera = "observation.images.cam_left_high"
        verified = {name: {"sha256": "x"} for name in ("data/source.parquet", "meta/info.json", "meta/episodes/chunk-000/file-000.parquet")}
        manifest = {"source_repo_id": "x", "source_revision": "y", "verified_source_files": verified, "camera_key": camera}
        item["decoded"] = {"source_video_relative_path": "videos/source.mp4", "source_video_from_timestamp_s": 0., "source_video_to_timestamp_s": 40 / 30.}
        info = {"fps": 30., "codebase_version": "v3.0", "data_path": "data/source.parquet", "video_path": "videos/source.mp4",
                "features": {name: {"shape": [28], "names": list(SOURCE_JOINT_NAMES)} for name in ("observation.state", "action")}}
        metadata = {"length": 40, "dataset_from_index": 0, "dataset_to_index": 40, "data/chunk_index": 0, "data/file_index": 0,
                    "videos/" + camera + "/chunk_index": 0, "videos/" + camera + "/file_index": 0,
                    "videos/" + camera + "/from_timestamp": 0., "videos/" + camera + "/to_timestamp": 40 / 30.}
        with tempfile.TemporaryDirectory() as directory, patch("g1_vla.native_diagnostics.verify_download_receipt", return_value=verified), \
                patch("g1_vla.native_diagnostics._parquet_columns", return_value=columns), \
                patch("g1_vla.native_diagnostics._episode_rows", return_value=(metadata, [])):
            write_json(Path(directory) / "meta/info.json", info)
            with self.assertRaisesRegex(NativeDiagnosticsError, "original filtered Arrow"):
                _raw_parquet_checks({"paths": {"root": directory}, "external_data": {"local_dir": "."}}, manifest, [item])

    def fixture(self, root):
        items = [episode(1), episode(2, split="validation")]
        records = [save_episode(root, item) for item in items]
        train = items[0]["arrays"]
        features = {}
        for name, column in (("observation.state", "source_state"), ("action", "source_action")):
            values = train[column].astype(np.float64)
            mean = values.sum(axis=0) / len(values)
            std = np.maximum(np.sqrt(np.maximum(np.square(values).sum(axis=0) / len(values) - mean ** 2, 0)), .001)
            features[name] = {"mean": mean.tolist(), "std": std.tolist(), "count": len(values)}
        statistics = {"training_episode_ids": [1], "validation_or_test_used": False,
                      "statistics_include_held_out": False, "features": features}
        statistics_path = root / "train_stats.json"
        write_json(statistics_path, statistics)
        manifest = {"episodes": [*records, {"episode_id": 3, "split": "test", "arrays": {"relative_path": "DO_NOT_OPEN_TEST.npz"}}],
            "statistics": file_record(root, statistics_path), "chunk_steps": 16,
            "camera_key": "observation.images.cam_left_high", "image_shape_hwc": [3, 4, 3]}
        manifest_path = root / "manifest.json"
        write_json(manifest_path, manifest)
        saved_config = {"train": {"seed": 123, "batch_size": 2}}
        saved_config_path = root / "checkpoint/experiment_config.json"
        write_json(saved_config_path, saved_config)
        training = training_receipt()
        training.update(training_passed=True, test_used_for_selection=False,
            bindings={"manifest_sha256": sha256_file(manifest_path), "statistics_sha256": sha256_file(statistics_path),
                      "experiment_config_sha256": hashlib.sha256(json.dumps(saved_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()},
            checkpoint={"path": str(saved_config_path.parent), "files": {"experiment_config.json": {"sha256": sha256_file(saved_config_path)}}})
        write_json(root / "training.json", training)
        config = {"paths": {"root": str(root)}, "grasp_act": {"dataset_manifest_path": "manifest.json"},
                  "source_native": {"timestamp_tolerance_s": .0001}, "external_data": {}, "native_diagnostics": {
                    "training_report_path": "training.json", "lag_frames": list(range(-3, 4)),
                    "stationary_speed_rad_s": .02, "moving_speed_rad_s": .1,
                    "loss_window_steps": 2, "verify_raw_source": False}}
        return config, manifest

    def test_full_stage_opens_only_train_validation_payloads_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest = self.fixture(root)
            with patch("g1_vla.native_diagnostics.load_grasp_manifest", return_value=manifest):
                result = run_native_diagnostics(config, root / "run")
                self.assertTrue(result["passed"], result.get("error"))
                self.assertEqual(result["scope"]["episode_ids"], [1, 2])
                self.assertEqual(result["label_contract"]["checked_anchor_count"], 6)
                self.assertFalse(result["test_payloads_read"])
                self.assertFalse(result["raw_parquet_parity"]["independently_rechecked"])
                self.assertFalse((root / "DO_NOT_OPEN_TEST.npz").exists())
                json.dumps(result, allow_nan=False)
                with self.assertRaisesRegex(NativeDiagnosticsError, "overwritten"):
                    run_native_diagnostics(config, root / "run")

    def test_failure_is_preserved_without_claiming_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, manifest = self.fixture(root)
            (root / "episode_1.npz").write_bytes(b"mutated")
            with patch("g1_vla.native_diagnostics.load_grasp_manifest", return_value=manifest):
                result = run_native_diagnostics(config, root / "run")
            self.assertFalse(result["passed"])
            self.assertEqual(result["status"], "failed")
            self.assertTrue((root / "run/reports/native_diagnostics.json").is_file())


if __name__ == "__main__":
    unittest.main()
