"""Evidence must include failed phases and preserve real sensor timing."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from g1_vla.teacher_evidence import _write_video, export_teacher_evidence, phase_scores


class TeacherEvidenceTests(unittest.TestCase):
    def test_unvisited_phase_is_not_a_success(self):
        scores = phase_scores({"phases": [{"phase": "home", "completed": True},
                                          {"phase": "contact", "completed": False}]})
        self.assertTrue(scores["home"]["completed"])
        self.assertTrue(scores["contact"]["entered"])
        self.assertFalse(scores["contact"]["completed"])
        self.assertFalse(scores["hold"]["entered"])
        self.assertFalse(scores["hold"]["completed"])

    def fixture(self, directory, *, gap=False, missing_camera=False):
        root = Path(directory)
        keys = ["observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"]
        config = {"control": {"physics_dt_s": .002, "physics_substeps": 33},
                  "observation": {"camera_keys": keys}}
        times = np.array([0., .066, .132 + (.066 if gap else 0.)])
        arrays = {"observation_time_s": times, "command_start_time_s": times,
                  "command_end_time_s": times + .066, "camera_keys": np.asarray(keys[:2] if missing_camera else keys)}
        for index in range(len(arrays["camera_keys"])):
            arrays["images_camera_{}".format(index)] = np.full((3, 64, 96, 3), index * 70, dtype=np.uint8)
        raw = root / "episode.npz"
        np.savez_compressed(raw, **arrays)
        return raw, config

    def test_failure_video_keeps_all_precommand_frames_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, config = self.fixture(directory)
            calls = []

            def writer(path, frames, fps):
                calls.append((path, frames.copy(), fps))
                return {"path": str(path), "frame_count": len(frames), "decode_verified": True}

            report = {"passed": False, "failure": "contact timeout", "phases": []}
            with patch("g1_vla.teacher_evidence._write_video", side_effect=writer):
                result = export_teacher_evidence(report, raw, directory, config)
            self.assertTrue(result["complete"])
            self.assertFalse(result["physical_passed"])
            self.assertEqual(result["failure"], "contact timeout")
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(len(row[1]) == 3 for row in calls))
            self.assertAlmostEqual(calls[0][2], 500 / 33)
            self.assertEqual(result["last_observation_time_s"], .132)
            self.assertAlmostEqual(result["final_control_end_s"], .198)

    def test_gaps_and_missing_camera_cannot_be_hidden_by_constant_fps(self):
        for mode in ("gap", "missing_camera"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                raw, config = self.fixture(directory, **{mode: True})
                with patch("g1_vla.teacher_evidence._write_video") as writer:
                    with self.assertRaises(ValueError):
                        export_teacher_evidence({}, raw, directory, config)
                    writer.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("imageio_ffmpeg") and importlib.util.find_spec("imageio"),
                         "video encoder dependencies unavailable")
    def test_actual_encoder_decodes_all_frames_without_resizing_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sensor.mp4"
            frames = np.zeros((4, 64, 96, 3), dtype=np.uint8)
            frames[:, :, :, 1] = np.arange(4, dtype=np.uint8)[:, None, None] * 70
            result = _write_video(path, frames, 500 / 33)
            self.assertTrue(result["decode_verified"])
            self.assertEqual(result["frame_count"], 4)
            self.assertEqual((result["width"], result["height"]), (96, 64))
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                _write_video(path, frames, 500 / 33)
            self.assertEqual(path.read_bytes(), before)
