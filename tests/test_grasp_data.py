"""Native pairing, disjoint complete episodes and train-only provenance."""

import copy
import hashlib
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from g1_vla.external_data import OFFICIAL_REPO, SOURCE_CAMERA_KEYS, SOURCE_JOINT_NAMES
from g1_vla.external_sources import DATA_FILES
from g1_vla.grasp_data import (
    BACKEND, CAMERA, OFFICIAL_REVISION, GraspDataError, GraspSquareDataset,
    build_grasp_square_data, decode_episode_rgb, load_grasp_manifest,
    validate_episode_rows, validate_native_settings, verify_file_groups,
)
from g1_vla.runtime.experiment import canonical_hash


def info():
    features = {key: {"dtype": "float32", "shape": [28], "names": [list(SOURCE_JOINT_NAMES)]}
                for key in ("observation.state", "action")}
    features.update({key: {"dtype": "video", "shape": [480, 640, 3],
                          "info": {"video.width": 640, "video.height": 480,
                                   "video.fps": 30.0, "video.codec": "av1"}} for key in SOURCE_CAMERA_KEYS})
    return {"codebase_version": "v3.0", "fps": 30.0, "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}


def settings():
    return {"backend": BACKEND, "episode_ids": [1, 2, 3],
            "split_episode_ids": {"train": [1], "validation": [2], "test": [3]},
            "split_unit": "source_video_file_group", "camera_key": CAMERA,
            "source_fps": 30.0, "chunk_steps": 3, "image_width": 4, "image_height": 3,
            "image_transform": "bilinear_resize_rgb_v1", "build_dir": "derived", "video_dir": "videos",
            "timestamp_tolerance_s": 1e-4, "max_episode_rows": 3000, "max_total_rows": 36000,
            "max_derived_bytes": 3 * 1024 ** 3, "max_video_download_bytes": 3 * 1024 ** 3}


def metadata(episode_id, rows=5):
    result = {"episode_index": episode_id, "length": rows, "dataset_from_index": episode_id * 10,
              "dataset_to_index": episode_id * 10 + rows, "data/chunk_index": 0, "data/file_index": 0,
              "tasks": ["camera packaging"]}
    for camera in SOURCE_CAMERA_KEYS:
        prefix = "videos/" + camera
        result.update({prefix + "/chunk_index": 0, prefix + "/file_index": episode_id,
                       prefix + "/from_timestamp": 0.0, prefix + "/to_timestamp": rows / 30.0})
    return result


def columns(episode_id, rows=5):
    offset = {1: 1.0, 2: 100.0, 3: 200.0}.get(episode_id, 1.0)
    state = np.arange(rows * 28, dtype=np.float32).reshape(rows, 28) * np.float32(.01) + np.float32(offset)
    return {"observation.state": state, "action": state + np.float32(10),
            "timestamp": (np.arange(rows) / 30.0).astype(np.float32), "frame_index": np.arange(rows, dtype=np.int64),
            "episode_index": np.full(rows, episode_id, dtype=np.int64),
            "index": np.arange(episode_id * 10, episode_id * 10 + rows, dtype=np.int64),
            "task_index": np.zeros(rows, dtype=np.int64)}


def write_receipt(directory, files):
    entries = []
    for name in files:
        path = directory / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(("verified fixture " + name).encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"relative_path": name, "path": str(path), "size_bytes": path.stat().st_size,
                        "sha256": digest, "expected_lfs_sha256": digest, "full_file_verified": True})
    (directory / "provenance.json").write_text(json.dumps({"repo_id": OFFICIAL_REPO, "revision": OFFICIAL_REVISION,
        "repo_type": "dataset", "anonymous": True, "training_started": False, "files": entries}))


class FakeFrame:
    def __init__(self, ticks, color=20):
        self.pts, self.time_base, self.color = ticks, Fraction(1, 30), color

    def to_ndarray(self, format):
        if format != "rgb24":
            raise AssertionError("must decode actual RGB")
        return np.full((480, 640, 3), self.color, dtype=np.uint8)


class FakeContainer:
    def __init__(self, frames):
        self.frames, self.seeks = frames, []
        self.streams = SimpleNamespace(video=[SimpleNamespace(width=640, height=480,
            average_rate=Fraction(30), time_base=Fraction(1, 30), thread_type=None,
            codec_context=SimpleNamespace(codec=SimpleNamespace(id=225), name="libdav1d"))])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def seek(self, ticks, **kwargs):
        self.seeks.append((ticks, kwargs))

    def decode(self, stream):
        return iter(self.frames)


class GraspDataTests(unittest.TestCase):
    def fixture(self, root):
        source, videos = root / "source", root / "videos"
        source.mkdir()
        (source / "meta").mkdir()
        (source / "meta/info.json").write_text(json.dumps(info()))
        write_receipt(source, DATA_FILES)
        write_receipt(videos, ["videos/{}/chunk-000/file-{:03d}.mp4".format(CAMERA, episode_id) for episode_id in (1, 2, 3)])
        return {"paths": {"root": str(root)}, "external_data": {"repo_id": OFFICIAL_REPO,
            "revision": OFFICIAL_REVISION, "local_dir": "source", "episode_ids": [0, 150, 300]},
            "source_native": settings()}

    def build(self, root, config):
        def decoder(path, targets, image_path, **kwargs):
            episode_id = int(path.stem.rsplit("-", 1)[1])
            color = {1: 20, 2: 210, 3: 250}[episode_id]
            images = np.full((len(targets), kwargs["height"], kwargs["width"], 3), color, dtype=np.uint8)
            with image_path.open("xb") as handle:
                np.save(handle, images, allow_pickle=False)
            return {"schema_version": 1, "passed": True, "paired_frame_count": len(targets),
                "maximum_absolute_pts_error_s": 0.0,
                "frames": [{"frame_index": index, "decoded_source_pts_s": float(target), "decoded_pts_ticks": index}
                           for index, target in enumerate(targets)]}
        with patch("g1_vla.grasp_data._episode_rows", side_effect=lambda source, details, episode_id, verified: (metadata(episode_id), [])), \
                patch("g1_vla.grasp_data._parquet_columns", side_effect=lambda path, episode_id: columns(episode_id)), \
                patch("g1_vla.grasp_data._parquet_rows", return_value=[{"task_index": 0, "task": "camera packaging"}]), \
                patch("g1_vla.grasp_data.decode_episode_rgb", side_effect=decoder):
            return build_grasp_square_data(config, root / "run")

    def test_predeclared_split_rejects_overlap_prior_audit_and_unknown_unit(self):
        for change in ({"split_episode_ids": {"train": [1], "validation": [1], "test": [3]}},
                       {"split_unit": "frame"}, {"source_fps": 15}, {"timestamp_tolerance_s": .02}):
            with self.subTest(change=change), self.assertRaises(GraspDataError):
                validate_native_settings({**settings(), **change})
        with self.assertRaisesRegex(GraspDataError, "prior diagnostic"):
            validate_native_settings(settings(), [1])

    def test_all_camera_grouping_rejects_unused_right_wrist_shared_across_split(self):
        episodes = {episode_id: metadata(episode_id) for episode_id in (1, 2, 3)}
        accepted = verify_file_groups(episodes, info(), settings()["split_episode_ids"])
        self.assertTrue(accepted["all_camera_storage_files_disjoint"])
        self.assertFalse(accepted["capture_sessions_independent_verified"])
        episodes[2]["videos/observation.images.cam_right_wrist/file_index"] = 1
        with self.assertRaisesRegex(GraspDataError, "share source video"):
            verify_file_groups(episodes, info(), settings()["split_episode_ids"])

    def test_native_episode_refuses_dtype_changes_frame_gaps_and_truncated_boundary(self):
        for key, replacement in (("action", columns(1)["action"].astype(np.float64)),
                                 ("frame_index", np.array([0, 1, 3, 4, 5])),
                                 ("timestamp", np.array([0, .04, .07, .10, .13], dtype=np.float32)),
                                 ("episode_index", np.full(5, 2))):
            values = columns(1)
            values[key] = replacement
            with self.subTest(key=key), self.assertRaises(GraspDataError):
                validate_episode_rows(values, metadata(1), 1, settings())
        with self.assertRaisesRegex(GraspDataError, "length"):
            validate_episode_rows(columns(1), metadata(1, 6), 1, settings())

    def test_every_true_rgb_is_paired_with_distinct_pts_and_declared_resize(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.npy"
            container = FakeContainer([FakeFrame(index, color=index * 10) for index in range(5)])
            av = SimpleNamespace(open=lambda *args, **kwargs: container, Codec=lambda *args: SimpleNamespace(id=225))
            receipt = decode_episode_rgb(Path("source.mp4"), np.arange(5) / 30, path, width=4, height=3,
                source_width=640, source_height=480, fps=30, tolerance_s=1e-4, av_module=av, image_module=Image)
            self.assertEqual(receipt["paired_frame_count"], 5)
            self.assertEqual([item["decoded_pts_ticks"] for item in receipt["frames"]], list(range(5)))
            self.assertEqual(receipt["maximum_absolute_pts_error_s"], 0)
            self.assertFalse(receipt["sensor_exposure_clock_verified"])
            self.assertFalse(receipt["frame_reuse_or_padding"])
            saved = np.load(path, allow_pickle=False)
            self.assertEqual(saved.shape, (5, 3, 4, 3))
            self.assertEqual(saved[4, 0, 0, 0], 40)
            self.assertEqual(len(container.seeks), 1)
            with self.assertRaises(FileExistsError):
                decode_episode_rgb(Path("source.mp4"), np.arange(5) / 30, path, width=4, height=3,
                    source_width=640, source_height=480, fps=30, tolerance_s=1e-4, av_module=av, image_module=Image)

    def test_missing_pts_or_frame_and_changed_codec_are_rejected(self):
        cases = ([FakeFrame(0), FakeFrame(2), FakeFrame(3)], [FakeFrame(0), FakeFrame(None)])
        for frames in cases:
            with self.subTest(frames=frames), tempfile.TemporaryDirectory() as directory:
                container = FakeContainer(frames)
                av = SimpleNamespace(open=lambda *args, **kwargs: container, Codec=lambda *args: SimpleNamespace(id=225))
                with self.assertRaises(GraspDataError):
                    decode_episode_rgb(Path("source.mp4"), np.arange(4) / 30, Path(directory) / "rgb.npy",
                        width=4, height=3, source_width=640, source_height=480, fps=30,
                        tolerance_s=1e-4, av_module=av, image_module=Image)
        with tempfile.TemporaryDirectory() as directory:
            container = FakeContainer([FakeFrame(index) for index in range(4)])
            container.streams.video[0].codec_context.codec.id = 27
            av = SimpleNamespace(open=lambda *args, **kwargs: container, Codec=lambda *args: SimpleNamespace(id=225))
            with self.assertRaisesRegex(GraspDataError, "codec"):
                decode_episode_rgb(Path("source.mp4"), np.arange(4) / 30, Path(directory) / "rgb.npy",
                    width=4, height=3, source_width=640, source_height=480, fps=30,
                    tolerance_s=1e-4, av_module=av, image_module=Image)

    def test_complete_build_preserves_native_labels_and_statistics_are_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            original = copy.deepcopy(config)
            before_source = (root / "source/provenance.json").read_bytes()
            result = self.build(root, config)
            self.assertTrue(result["passed"])
            self.assertEqual(result["total_rows"], 15)
            self.assertEqual(result["total_paired_rgb_frames"], 15)
            self.assertEqual(result["split_sample_counts"], {"train": 3, "validation": 3, "test": 3})
            self.assertEqual(config, original)
            self.assertEqual((root / "source/provenance.json").read_bytes(), before_source)
            manifest = load_grasp_manifest(root / "derived", expected_settings=config["source_native"], expected_source=config["external_data"])
            self.assertEqual(manifest["source_task_metadata"], [{"task_index": 0, "task": "camera packaging"}])
            self.assertEqual(manifest["episodes"][0]["source_tasks"], ["camera packaging"])
            self.assertFalse(manifest["current_simulator_adapter_ready"])
            dataset = GraspSquareDataset(root / "derived", "train", manifest=manifest, verify=False)
            self.assertEqual(dataset.samples, [(1, 0), (1, 1), (1, 2)])
            self.assertEqual(dataset.get_eval_indices(2), [0, 2])
            sample = dataset[2]
            np.testing.assert_array_equal(sample["action"], columns(1)["action"][2:5])
            np.testing.assert_array_equal(sample["observation.state"], columns(1)["observation.state"][2])
            self.assertEqual(sample[CAMERA].shape, (3, 3, 4))
            np.testing.assert_allclose(sample[CAMERA], 20 / 255, rtol=1e-6)
            self.assertFalse(sample["action_is_pad"].any())
            stats = dataset.statistics
            self.assertEqual(stats["training_episode_ids"], [1])
            np.testing.assert_allclose(stats["features"]["observation.state"]["mean"], columns(1)["observation.state"].mean(axis=0), rtol=1e-6)
            np.testing.assert_allclose(stats["features"][CAMERA]["mean"], np.full((3, 1, 1), 20 / 255), rtol=1e-6)
            self.assertFalse(stats["validation_or_test_used"])
            with self.assertRaises(FileExistsError):
                self.build(root, config)

    def test_source_hash_and_derived_bytes_are_checked_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            (root / "source/data/chunk-000/file-000.parquet").write_bytes(b"wrong source")
            with self.assertRaises(ValueError):
                self.build(root, config)
            self.assertFalse((root / "derived").exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            config["source_native"]["max_derived_bytes"] = 1
            with self.assertRaisesRegex(GraspDataError, "byte budget"):
                self.build(root, config)
            self.assertFalse((root / "derived").exists())

    def test_partial_decode_failure_stays_unadmitted_and_preserves_failure_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            with patch("g1_vla.grasp_data._episode_rows", side_effect=lambda source, details, episode_id, verified: (metadata(episode_id), [])), \
                    patch("g1_vla.grasp_data._parquet_columns", side_effect=lambda path, episode_id: columns(episode_id)), \
                    patch("g1_vla.grasp_data.decode_episode_rgb", side_effect=GraspDataError("missing real RGB")):
                with self.assertRaisesRegex(GraspDataError, "missing real RGB"):
                    build_grasp_square_data(config, root / "run")
            self.assertTrue((root / "derived/failed.json").is_file())
            self.assertFalse((root / "derived/manifest.json").exists())
            with self.assertRaisesRegex(GraspDataError, "partial failed"):
                load_grasp_manifest(root / "derived")

    def test_manifest_forged_joint_order_or_unused_camera_group_cannot_bypass_data_hashes(self):
        for field in ("joint_order", "file_group", "source_revision", "statistics_population", "statistics_mean", "summary"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = self.fixture(root)
                self.build(root, config)
                path = root / "derived/manifest.json"
                manifest = json.loads(path.read_text())
                if field == "joint_order":
                    manifest["source_action_joint_names"].reverse()
                elif field == "file_group":
                    camera = "observation.images.cam_right_wrist"
                    manifest["episodes"][1]["all_camera_video_relative_paths"][camera] = manifest["episodes"][0]["all_camera_video_relative_paths"][camera]
                elif field == "source_revision":
                    manifest["source_revision"] = "a" * 40
                elif field == "summary":
                    manifest["total_paired_rgb_frames"] = 3
                else:
                    stats_path = root / "derived/train_stats.json"
                    stats = json.loads(stats_path.read_text())
                    if field == "statistics_population":
                        stats["features"]["action"]["count"] = 15
                    else:
                        stats["features"]["action"]["mean"][0] = 1000.0
                    stats_path.write_text(json.dumps(stats))
                    manifest["statistics"]["sha256"] = hashlib.sha256(stats_path.read_bytes()).hexdigest()
                    manifest["statistics"]["size_bytes"] = stats_path.stat().st_size
                path.write_text(json.dumps(manifest))
                with self.assertRaises(GraspDataError):
                    load_grasp_manifest(root / "derived", expected_settings=config["source_native"], expected_source=config["external_data"])

    def test_derived_file_tampering_and_config_selection_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.fixture(root)
            self.build(root, config)
            changed = {**config["source_native"], "chunk_steps": 2}
            with self.assertRaisesRegex(GraspDataError, "configured source-native"):
                load_grasp_manifest(root / "derived", expected_settings=changed)
            manifest = load_grasp_manifest(root / "derived")
            rgb_path = root / "derived" / manifest["episodes"][0]["rgb"]["relative_path"]
            with rgb_path.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(GraspDataError, "immutable manifest"):
                load_grasp_manifest(root / "derived")


if __name__ == "__main__":
    unittest.main()
