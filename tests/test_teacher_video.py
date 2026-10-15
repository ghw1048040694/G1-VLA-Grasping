"""Diagnostic video is on-step evidence and cannot substitute/fabricate states."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from g1_vla.teacher_video import (
    CAMERAS, HEIGHT, WIDTH, TeacherDiagnosticVideo, _parse_decoded_timing,
)


class FakeWriter:
    def __init__(self, owner, path):
        self.owner, self.path = owner, path
        self.frames = []

    def send(self, pixels):
        if pixels is not None:
            self.frames.append(pixels.tobytes())

    def close(self):
        if self.frames:
            Path(self.path).write_bytes(b"mock-video-only" + b"".join(self.frames))


class FakeFFmpeg:
    def __init__(self):
        self.writers, self.options = {}, []
        self.bad_dimensions = False
        self.drop_decoded_frame = False

    def write_frames(self, path, size, **kwargs):
        self.options.append((size, kwargs))
        result = FakeWriter(self, path)
        self.writers[path] = result
        return result

    def read_frames(self, path, **kwargs):
        yield {"size": (WIDTH + 1, HEIGHT) if self.bad_dimensions else (WIDTH, HEIGHT), "fps": 15.15}
        frames = self.writers[path].frames
        yield from frames[:-1] if self.drop_decoded_frame else frames


class FakeRenderer:
    def __init__(self, model, *, height, width):
        assert (height, width) == (HEIGHT, WIDTH)
        self.closed = False
        self.fail_camera = None

    def update_scene(self, data, *, camera):
        self.data, self.camera = data, camera
        if self.fail_camera == camera:
            raise RuntimeError("diagnostic render unavailable")

    def render(self):
        value = round(self.data.time * 1000) + CAMERAS.index(self.camera)
        return np.full((HEIGHT, WIDTH, 3), value % 256, dtype=np.uint8)

    def close(self):
        self.closed = True


class TeacherVideoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.ffmpeg = FakeFFmpeg()
        self.modules = patch.dict("sys.modules", {"imageio_ffmpeg": self.ffmpeg})
        self.modules.start()
        self.timing = patch.object(TeacherDiagnosticVideo, "_verify_timing",
                                   return_value={"passed": True, "timebase": "33/500"})
        self.timing.start()
        self.config = {"control": {"physics_dt_s": .002, "physics_substeps": 33},
                       "observation": {"image_width": 96, "image_height": 64}}
        self.api = SimpleNamespace(Renderer=FakeRenderer)

    def tearDown(self):
        self.timing.stop()
        self.modules.stop()
        self.temporary.cleanup()

    def make(self):
        return TeacherDiagnosticVideo(object(), self.config, self.root, mujoco_api=self.api)

    def test_live_precommand_inputs_are_preserved_and_failed_phase_retained(self):
        recorder = self.make()
        config_before = copy.deepcopy(self.config)
        data = SimpleNamespace(time=0., qpos=np.array([1., 2.]), qvel=np.array([3., 4.]))
        for index, phase in enumerate(("home", "contact", "contact_timeout")):
            data.time = index * .066
            before = (data.time, data.qpos.copy(), data.qvel.copy())
            recorder.capture(data, phase)
            self.assertEqual(data.time, before[0])
            np.testing.assert_array_equal(data.qpos, before[1])
            np.testing.assert_array_equal(data.qvel, before[2])
        report = recorder.finish()
        self.assertTrue(report["complete"])
        self.assertEqual(report["fps_fraction"], "500/33")
        self.assertEqual(self.config, config_before)
        self.assertEqual(report["capture_attempt_count"], 3)
        self.assertEqual(report["capture_attempts"][-1]["phase"], "contact_timeout")
        self.assertTrue(recorder.renderer.closed)
        for camera in CAMERAS:
            item = report["cameras"][camera]
            self.assertEqual(item["independent_decode"]["decoded_frame_count"], 3)
            self.assertTrue(Path(item["path"]).exists())
            frame = np.full((HEIGHT, WIDTH, 3), CAMERAS.index(camera), dtype=np.uint8)
            self.assertEqual(item["frames"][0]["input_pixel_sha256"], hashlib.sha256(frame.tobytes()).hexdigest())
        for size, options in self.ffmpeg.options:
            self.assertEqual(size, (320, 240))
            self.assertEqual(options["input_params"], ["-r", "500/33"])
            self.assertEqual(options["output_params"][:2], ["-vsync", "0"])
            self.assertEqual(options["macro_block_size"], 1)
        self.assertFalse(report["physical_score_modified"])
        self.assertFalse(report["policy_observation_modified"])

    def test_duplicate_or_missing_timestamp_does_not_create_a_frame(self):
        for bad_time in (0., .132):
            with self.subTest(bad_time=bad_time), tempfile.TemporaryDirectory() as directory:
                recorder = TeacherDiagnosticVideo(object(), self.config, directory, mujoco_api=self.api)
                recorder.capture(SimpleNamespace(time=0.), "home")
                recorder.capture(SimpleNamespace(time=bad_time), "contact")
                report = recorder.finish()
                self.assertFalse(report["complete"])
                self.assertEqual(report["capture_attempt_count"], 2)
                self.assertTrue(any(e["operation"] == "capture_timestamp" for e in report["errors"]))
                self.assertTrue(all(c["input_frame_count"] == 1 for c in report["cameras"].values()))

    def test_single_camera_failure_keeps_other_actual_frames_and_explicit_error(self):
        recorder = self.make()
        recorder.capture(SimpleNamespace(time=0.), "home")
        recorder.renderer.fail_camera = CAMERAS[0]
        recorder.capture(SimpleNamespace(time=.066), "contact")
        recorder.capture(SimpleNamespace(time=.132), "grasp")
        report = recorder.finish()
        self.assertFalse(report["complete"])
        self.assertEqual(report["cameras"][CAMERAS[0]]["input_frame_count"], 1)
        self.assertEqual(report["cameras"][CAMERAS[1]]["input_frame_count"], 3)
        self.assertTrue(any(e["camera"] == CAMERAS[0] for e in report["errors"]))

    def test_empty_or_nonfinite_capture_is_reported_and_never_scored(self):
        recorder = self.make()
        recorder.capture(SimpleNamespace(time=float("nan")), "failure")
        report = recorder.finish()
        self.assertFalse(report["complete"])
        self.assertTrue(report["errors"])
        self.assertTrue((self.root / "diagnostic_video.json").exists())

    def test_decoded_frame_count_and_dimensions_must_match(self):
        for flag in ("drop_decoded_frame", "bad_dimensions"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                setattr(self.ffmpeg, flag, True)
                recorder = TeacherDiagnosticVideo(object(), self.config, directory, mujoco_api=self.api)
                recorder.capture(SimpleNamespace(time=0.), "contact")
                result = recorder.finish()
                self.assertFalse(result["complete"])
                self.assertTrue(all(not x["published"] for x in result["cameras"].values()))
                self.assertTrue(all(x["retained_staging_path"] for x in result["cameras"].values()))
                setattr(self.ffmpeg, flag, False)

    def test_existing_manifest_and_existing_movie_cannot_be_overwritten(self):
        manifest = self.root / "diagnostic_video.json"
        manifest.write_text("earlier immutable receipt")
        recorder = self.make()
        self.assertFalse(recorder.finish()["complete"])
        self.assertEqual(manifest.read_text(), "earlier immutable receipt")
        with tempfile.TemporaryDirectory() as directory:
            recorder = TeacherDiagnosticVideo(object(), self.config, directory, mujoco_api=self.api)
            recorder.capture(SimpleNamespace(time=0.), "contact")
            movie = recorder.video_dir / (CAMERAS[0] + ".mp4")
            movie.write_bytes(b"earlier immutable movie")
            self.assertFalse(recorder.finish()["complete"])
            self.assertEqual(movie.read_bytes(), b"earlier immutable movie")

    def test_finish_is_idempotent_and_late_capture_does_not_throw_into_physics(self):
        recorder = self.make()
        recorder.capture(SimpleNamespace(time=0.), "contact")
        report = recorder.finish()
        original = (self.root / "diagnostic_video.json").read_bytes()
        self.assertIs(recorder.finish(), report)
        self.assertFalse(recorder.capture(SimpleNamespace(time=.066), "late")["complete"])
        self.assertEqual((self.root / "diagnostic_video.json").read_bytes(), original)

    def test_manifest_records_actual_input_times_not_rounded_decoder_fps(self):
        recorder = self.make()
        for index in range(3):
            recorder.capture(SimpleNamespace(time=10. + index * .066), "contact")
        recorder.finish()
        saved = json.loads((self.root / "diagnostic_video.json").read_text())
        self.assertEqual([r["timestamp_s"] for r in saved["capture_attempts"]], [10., 10.066, 10.132])
        self.assertEqual(saved["fps_fraction"], "500/33")


class ExactTimingTests(unittest.TestCase):
    def source(self, pts=(0, 1, 2), duration=1, timebase="33/500"):
        return "#tb 0: " + timebase + "\n#dimensions 0: 320x240\n" + "\n".join(
            "0, {0}, {0}, {1}, 230400, hash".format(p, duration) for p in pts)

    def test_exact_rational_frame_pts_pass(self):
        self.assertTrue(_parse_decoded_timing(self.source(), 3, .066)["passed"])

    def test_rounded_fps_duplicate_or_skipped_pts_cannot_pass(self):
        for source in (self.source(timebase="20/303"), self.source(pts=(0, 0, 1)),
                       self.source(pts=(0, 1, 3)), self.source(duration=2)):
            with self.subTest(source=source), self.assertRaises(ValueError):
                _parse_decoded_timing(source, 3, .066)


if __name__ == "__main__":
    unittest.main()
