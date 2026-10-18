"""Declared native RGB packaging preserves camera and episode identities."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

from g1_vla.rl_teacher import _diagnostic_video_delivery, _package_eval_videos


def rgb_png(value):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    pixels = (b"\0" + bytes([value, value, value]) * 640) * 480
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 640, 480, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b""))


class DeclaredVideoPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.counter = 0
        self.commands, self.encoded = [], {}
        self.probe_overrides = {}
        self.runner = patch("g1_vla.rl_teacher.subprocess.run", side_effect=self.fake_run)
        self.runner.start()

    def tearDown(self):
        self.runner.stop()
        self.temporary.cleanup()

    def fake_run(self, command, **kwargs):
        self.commands.append(command)
        target = Path(command[-1])
        if command[0] == "ffmpeg":
            if target.exists():
                raise subprocess.CalledProcessError(1, command)
            first = int(command[command.index("-start_number") + 1])
            source = Path(command[command.index("-i") + 1]).parent
            count = (int(command[command.index("-frames:v") + 1]) if "-frames:v" in command else
                     len(list(source.glob("frame_*.png"))) - first)
            target.write_bytes(b"test-container-only:" + str(target).encode())
            self.encoded[str(target)] = {"nb_read_frames": str(count), "r_frame_rate": "500/33",
                                         "width": 640, "height": 480}
            return SimpleNamespace(returncode=0)
        self.assertEqual(command[0], "ffprobe")
        return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": [
            {**self.encoded[str(target)], **self.probe_overrides}]}))

    def fixture(self):
        self.counter += 1
        base = self.root / str(self.counter)
        output, run_dir = base / "native", base / "run"
        environment = output / "environment_8"
        (run_dir / "logs").mkdir(parents=True)
        declaration = {"version": "g1_declared_multienv_native_rgb_v1", "env_ids": [0, 3]}
        profiles = [{"profile": {"name": "approach"}, "control_steps": 3,
                     "first_video_frame": 0, "last_video_frame_exclusive": 3,
                     "scored_episode_ids": [[0, 2], [3, 2]], "excluded_episode_ids": [[0, 3]]},
                    {"profile": {"name": "lift"}, "control_steps": 3,
                     "first_video_frame": 3, "last_video_frame_exclusive": 6,
                     "scored_episode_ids": [[0, 4], [3, 5]], "excluded_episode_ids": [[3, 4]]}]
        report = {"stage": "rl-diagnose", "video_recording": declaration,
                  "video_evidence_complete": True, "profiles": profiles}
        status = {"video_recording": declaration, "frames": 6, "failure": None,
                  "all_control_frames_recorded": True, "cameras": []}
        rows = {}
        for env in declaration["env_ids"]:
            relative = "video/env_" + str(env)
            frames = environment / relative / "rgb_frames"
            frames.mkdir(parents=True)
            status["cameras"].append({"camera_env_id": env, "frame_directory": relative + "/rgb_frames",
                "frame_ledger": relative + "/video_frames.jsonl", "frames": 6, "enabled": True,
                "failure": None, "expected_control_frames": 6, "all_control_frames_recorded": True})
            header = {"event": "camera_header", "camera_env_id": env, "video_recording": declaration,
                      "width": 640, "height": 480, "fps_numerator": 500, "fps_denominator": 33}
            entries = [header]
            episodes = [2, 2, 3, 4, 4, 4] if env == 0 else [2, 2, 2, 4, 5, 5]
            controls = [1, 2, 1, 1, 2, 3] if env == 0 else [1, 2, 3, 1, 1, 2]
            for index in range(6):
                path = frames / "frame_{:06d}.png".format(index)
                payload = rgb_png(env * 20 + index)
                path.write_bytes(payload)
                native = 99 + index * 33
                entries.append({"event": "camera_frame", "camera_env_id": env, "frame_index": index,
                    "path": path.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                    "native_physics_step_count": native, "actual_physics_dt_s": .002, "sim_time_s": native * .002,
                    "episode_id": episodes[index], "episode_control_steps": controls[index],
                    "camera_frame_counter": index + 1, "pre_reset": True,
                    "profile_name": "approach" if index < 3 else "lift"})
            rows[env] = entries
        fixture = SimpleNamespace(output=output, run_dir=run_dir, environment=environment,
                                  declaration=declaration, report=report, status=status, rows=rows)
        self.save(fixture)
        return fixture

    def save(self, fixture):
        (fixture.output / "native_report.json").write_text(json.dumps(fixture.report))
        (fixture.environment / "evaluation_video_status.json").write_text(json.dumps(fixture.status))
        for env, entries in fixture.rows.items():
            (fixture.environment / ("video/env_" + str(env)) / "video_frames.jsonl").write_text(
                "".join(json.dumps(entry) + "\n" for entry in entries))

    def package(self, fixture):
        return _package_eval_videos(fixture.output, fixture.run_dir)

    def test_distinct_camera_files_and_episode_quota_pairs_survive_asynchronous_resets(self):
        fixture = self.fixture()
        before = {path: path.read_bytes() for path in fixture.environment.rglob("*.png")}
        videos = self.package(fixture)
        self.assertEqual([video["camera_env_id"] for video in videos], [0, 3])
        self.assertTrue(all(video["encoded"] and video["complete"] for video in videos), videos)
        self.assertEqual(Path(videos[1]["path"]).name, "evaluation_env3.mp4")
        self.assertEqual(Path(videos[1]["profile_clips"][1]["path"]).name, "evaluation_env3_lift.mp4")
        env0_segments = videos[0]["profile_clips"][0]["segments"]
        env3_segments = videos[1]["profile_clips"][1]["segments"]
        self.assertEqual([(s["episode_id"], s["quota_membership"]) for s in env0_segments],
                         [(2, "scored"), (3, "quota_excluded")])
        self.assertEqual([(s["episode_id"], s["quota_membership"]) for s in env3_segments],
                         [(4, "quota_excluded"), (5, "scored")])
        self.assertEqual(env3_segments[1]["first_source_frame"], 4)
        self.assertEqual(env3_segments[1]["first_native_physics_step"], 231)
        self.assertEqual([video["decoded_frames"] for video in videos], [6, 6])
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertTrue(_diagnostic_video_delivery(videos, fixture.report,
                                                  [profile["profile"] for profile in fixture.report["profiles"]]))
        self.assertEqual(len([command for command in self.commands if command[0] == "ffmpeg"]), 6)

    def test_same_episode_number_in_different_environments_does_not_share_score_membership(self):
        fixture = self.fixture()
        fixture.report["profiles"][0]["scored_episode_ids"] = [[0, 2]]
        fixture.report["profiles"][0]["excluded_episode_ids"] = [[0, 3], [3, 2]]
        self.save(fixture)
        videos = self.package(fixture)
        self.assertEqual(videos[0]["profile_clips"][0]["segments"][0]["quota_membership"], "scored")
        self.assertEqual(videos[1]["profile_clips"][0]["segments"][0]["quota_membership"], "quota_excluded")

    def test_unfinished_after_quota_episode_is_explicitly_unscored_not_claimed_completed(self):
        fixture = self.fixture()
        fixture.report["profiles"][0]["excluded_episode_ids"] = []
        self.save(fixture)
        video = self.package(fixture)[0]
        self.assertEqual(video["profile_clips"][0]["segments"][1]["quota_membership"], "unscored_or_incomplete")
        self.assertFalse(video["profile_clips"][0]["segments"][1]["scored_for_profile"])

    def test_missing_misordered_duplicate_or_out_of_range_camera_declaration_fails(self):
        for mutation in ("missing", "order", "duplicate", "out_of_range", "boolean", "extra_key", "report"):
            with self.subTest(mutation=mutation):
                fixture = self.fixture()
                if mutation == "missing": fixture.status["cameras"].pop()
                elif mutation == "order": fixture.status["cameras"].reverse()
                elif mutation == "duplicate": fixture.declaration["env_ids"] = [0, 0]
                elif mutation == "out_of_range": fixture.declaration["env_ids"] = [0, 8]
                elif mutation == "boolean": fixture.declaration["env_ids"] = [False, 3]
                elif mutation == "extra_key": fixture.declaration["width"] = 640
                else: fixture.report["video_recording"] = {"version": fixture.declaration["version"], "env_ids": [0]}
                self.save(fixture)
                result = self.package(fixture)
                self.assertTrue(result and all(not video["encoded"] and "error" in video for video in result), result)

    def test_wrong_camera_header_source_identity_dimensions_and_path_are_rejected(self):
        for mutation in ("header_env", "header_version", "header_dimensions", "frame_env", "frame_path", "status_path"):
            with self.subTest(mutation=mutation):
                fixture = self.fixture()
                entries = fixture.rows[3]
                if mutation == "header_env": entries[0]["camera_env_id"] = 0
                elif mutation == "header_version": entries[0]["video_recording"] = None
                elif mutation == "header_dimensions": entries[0]["width"] = 96
                elif mutation == "frame_env": entries[2]["camera_env_id"] = 0
                elif mutation == "frame_path": entries[2]["path"] = "../frame_000001.png"
                else: fixture.status["cameras"][1]["frame_directory"] = "video/env_0/rgb_frames"
                self.save(fixture)
                result = self.package(fixture)[1]
                self.assertFalse(result["encoded"], result)
                self.assertIn("error", result)

    def test_missing_duplicate_or_corrupted_original_png_is_retained_as_failure(self):
        for mutation in ("missing", "index", "bytes", "sha", "dimensions", "duplicate"):
            with self.subTest(mutation=mutation):
                fixture = self.fixture()
                row = fixture.rows[3][2]
                path = fixture.environment / "video/env_3/rgb_frames" / row["path"]
                if mutation == "missing": path.unlink()
                elif mutation == "index": row["frame_index"] = 2
                elif mutation == "bytes": row["bytes"] += 1
                elif mutation == "sha": row["sha256"] = "0" * 64
                elif mutation == "dimensions":
                    payload = path.read_bytes()
                    payload = payload[:16] + struct.pack(">II", 96, 64) + payload[24:]
                    path.write_bytes(payload)
                    row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
                else: fixture.rows[3].append(copy.deepcopy(row))
                self.save(fixture)
                result = self.package(fixture)[1]
                self.assertFalse(result["encoded"], result)
                self.assertIn("error", result)

    def test_native_cadence_current_episode_pre_reset_and_dt_are_required(self):
        for mutation in ("native", "dt", "time", "pre_reset", "control", "counter", "episode"):
            with self.subTest(mutation=mutation):
                fixture = self.fixture()
                row = fixture.rows[3][2]
                if mutation == "native": row["native_physics_step_count"] += 1
                elif mutation == "dt": row["actual_physics_dt_s"] = .003
                elif mutation == "time": row["sim_time_s"] = .1
                elif mutation == "pre_reset": row["pre_reset"] = False
                elif mutation == "control": row["episode_control_steps"] = 1
                elif mutation == "counter": row["camera_frame_counter"] = 1
                else: row["episode_id"] = 1
                self.save(fixture)
                result = self.package(fixture)[1]
                self.assertFalse(result["encoded"], result)
                self.assertIn("error", result)

    def test_all_cameras_must_share_actual_native_frame_set_times(self):
        fixture = self.fixture()
        for row in fixture.rows[3][1:]:
            row["native_physics_step_count"] += 33
            row["sim_time_s"] += .066
        self.save(fixture)
        result = self.package(fixture)[1]
        self.assertFalse(result["encoded"])
        self.assertIn("frame-set", result["error"])

    def test_global_frame_count_is_per_set_not_sum_of_cameras(self):
        fixture = self.fixture()
        fixture.status["frames"] = 12
        self.save(fixture)
        self.assertTrue(all(not video["encoded"] for video in self.package(fixture)))

    def test_partial_capture_can_encode_intact_original_prefix_but_cannot_pass_delivery(self):
        fixture = self.fixture()
        fixture.status["failure"] = "camera 3 unavailable"
        fixture.status["cameras"][1].update(failure="camera unavailable", all_control_frames_recorded=False)
        self.save(fixture)
        videos = self.package(fixture)
        self.assertTrue(all(video["encoded"] for video in videos))
        self.assertTrue(all(not video["complete"] and "error" in video for video in videos))
        self.assertFalse(_diagnostic_video_delivery(videos, fixture.report,
                                                   [profile["profile"] for profile in fixture.report["profiles"]]))

    def test_profile_ranges_and_quota_identity_must_match_original_frame_ledger(self):
        for mutation in ("profile", "range", "missing_quota", "wrong_env_episode", "overlap"):
            with self.subTest(mutation=mutation):
                fixture = self.fixture()
                if mutation == "profile": fixture.rows[3][2]["profile_name"] = "lift"
                elif mutation == "range": fixture.report["profiles"][1]["first_video_frame"] = 2
                elif mutation == "missing_quota": fixture.report["profiles"][0].pop("scored_episode_ids")
                elif mutation == "wrong_env_episode": fixture.report["profiles"][0]["scored_episode_ids"].append([3, 3])
                else: fixture.report["profiles"][0]["excluded_episode_ids"].append([3, 2])
                self.save(fixture)
                videos = self.package(fixture)
                self.assertFalse(videos[1]["complete"], videos[1])
                self.assertIn("error", videos[1])

    def test_probe_detects_dropped_frames_wrong_fps_and_wrong_dimensions(self):
        for overrides in ({"nb_read_frames": "5"}, {"r_frame_rate": "30/1"}, {"width": 96}):
            with self.subTest(overrides=overrides):
                fixture = self.fixture()
                self.probe_overrides = overrides
                videos = self.package(fixture)
                self.assertTrue(all(not video["encoded"] and not video["complete"] for video in videos))

    def test_delivery_rejects_missing_or_relabelled_camera_or_episode_segment(self):
        fixture = self.fixture()
        videos = self.package(fixture)
        expected = [profile["profile"] for profile in fixture.report["profiles"]]
        for mutation in ("missing", "label", "incomplete", "segment_env", "segment_range", "quota", "profile"):
            with self.subTest(mutation=mutation):
                broken = copy.deepcopy(videos)
                if mutation == "missing": broken.pop()
                elif mutation == "label": broken[1]["camera_env_id"] = 0
                elif mutation == "incomplete": broken[1]["complete"] = False
                elif mutation == "segment_env": broken[1]["profile_clips"][0]["segments"][0]["camera_env_id"] = 0
                elif mutation == "segment_range": broken[1]["profile_clips"][0]["segments"][0]["first_source_frame"] = 1
                elif mutation == "quota": broken[1]["profile_clips"][0]["segments"][0]["scored_for_profile"] = False
                else: broken[1]["profile_clips"].pop()
                self.assertFalse(_diagnostic_video_delivery(broken, fixture.report, expected))

    def test_legacy_layout_row_fields_filenames_commands_and_delivery_remain_unchanged(self):
        fixture = self.fixture()
        for path in fixture.environment.rglob("*.png"):
            path.unlink()
        shutil.rmtree(fixture.environment / "video")
        (fixture.environment / "evaluation_video_status.json").write_text(json.dumps({"frames": 3}))
        frames = fixture.environment / "rgb_frames"
        frames.mkdir()
        for index in range(3):
            (frames / "frame_{:06d}.png".format(index)).write_bytes(rgb_png(index))
        fixture.report.pop("video_recording")
        fixture.report["profiles"] = fixture.report["profiles"][:1]
        (fixture.output / "native_report.json").write_text(json.dumps(fixture.report))
        videos = self.package(fixture)
        self.assertEqual(len(videos), 1)
        self.assertEqual(set(videos[0]), {"frames", "source", "simulation_fps", "frame_directory", "encoded", "path",
            "sha256", "bytes", "decoded_frames", "encoded_fps", "profile_clips"})
        self.assertEqual(Path(videos[0]["path"]), fixture.environment / "evaluation_env0.mp4")
        self.assertNotIn("-frames:v", self.commands[0])
        self.assertEqual(self.commands[1][self.commands[1].index("-show_entries") + 1],
                         "stream=nb_read_frames,r_frame_rate")
        report = json.loads((fixture.run_dir / "reports/native_eval_videos.json").read_text())
        self.assertEqual(report["scope"], "env0 actual RGB at native control times; episode boundaries remain in frame ledger")
        self.assertTrue(_diagnostic_video_delivery(videos, fixture.report, [{"name": "approach"}]))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "actual ffmpeg/ffprobe required")
    def test_real_ffmpeg_encodes_both_native_camera_fixtures_without_changing_source_pngs(self):
        fixture = self.fixture()
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in fixture.environment.rglob("*.png")}
        self.runner.stop()
        videos = self.package(fixture)
        self.assertTrue(all(video["complete"] and video["decoded_frames"] == 6 for video in videos), videos)
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before})
        self.assertTrue(_diagnostic_video_delivery(videos, fixture.report, [{"name": "approach"}, {"name": "lift"}]))


if __name__ == "__main__":
    unittest.main()
