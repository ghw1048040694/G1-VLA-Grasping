"""Actual on-step diagnostic cameras, independent of policy observations.

This recorder never advances or reconstructs the simulator.  Its caller supplies
the live data immediately before a control command, including failed episodes.
Video failures are explicit evidence failures and do not change physical scores.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path


CAMERAS = ("head_camera", "right_wrist_camera")
WIDTH, HEIGHT = 320, 240


def _parse_decoded_timing(text, expected_count, interval_s):
    """Use decoder PTS/timebase, never its rounded human-readable fps."""
    timebase, dimensions, records = None, None, []
    for line in text.splitlines():
        if line.startswith("#tb 0:"):
            timebase = Fraction(line.split(":", 1)[1].strip())
        elif line.startswith("#dimensions 0:"):
            dimensions = tuple(int(value) for value in line.split(":", 1)[1].strip().split("x"))
        elif line.strip() and not line.startswith("#"):
            fields = [value.strip() for value in line.split(",")]
            if len(fields) < 6 or int(fields[0]) != 0:
                raise ValueError("unexpected decoded frame timing record")
            records.append((int(fields[2]), int(fields[3])))
    if timebase is None or dimensions != (WIDTH, HEIGHT) or len(records) != expected_count:
        raise ValueError("decoded timing dimensions/count/timebase missing or incorrect")
    if not records:
        raise ValueError("decoded movie contains no actual frames")
    for index, (pts, duration) in enumerate(records):
        if not math.isclose(float(pts * timebase), index * interval_s, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("decoded video PTS disagree with actual capture cadence")
        if not math.isclose(float(duration * timebase), interval_s, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("decoded frame duration disagrees with actual capture cadence")
    return {"passed": True, "timebase": str(timebase),
            "pts": [record[0] for record in records],
            "duration_ticks": [record[1] for record in records],
            "verified_frame_count": len(records), "width": WIDTH, "height": HEIGHT}


def _sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


class TeacherDiagnosticVideo:
    """Stream two genuine RGB views; publish each movie at most once."""

    def __init__(self, model, config, run_dir, *, mujoco_api=None):
        self.run_dir = Path(run_dir).resolve()
        self.video_dir = self.run_dir / "diagnostic_videos"
        self.manifest_path = self.run_dir / "diagnostic_video.json"
        self.renderer = None
        self.writers = {}
        self.camera_rows = {camera: [] for camera in CAMERAS}
        self.attempts, self.errors = [], []
        self.disabled_cameras = set()
        self.interval_s = self.fps = None
        self.fps_fraction = None
        self._finished = None
        self._ready = False
        self._cadence_failed = False
        try:
            import imageio_ffmpeg
            import numpy as np
            from .control import _api
            self._np, self._ffmpeg = np, imageio_ffmpeg
            self.run_dir.mkdir(parents=True, exist_ok=True)
            if self.manifest_path.exists():
                raise FileExistsError("diagnostic manifest already exists")
            self.video_dir.mkdir(exist_ok=False)
            control = config["control"]
            substeps = control["physics_substeps"]
            dt = control["physics_dt_s"]
            if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps <= 0:
                raise ValueError("positive integer physics_substeps required")
            if isinstance(dt, bool) or not math.isfinite(float(dt)) or float(dt) <= 0:
                raise ValueError("positive finite physics_dt_s required")
            interval = Fraction(str(dt)) * substeps
            rate = 1 / interval
            self.interval_s, self.fps = float(interval), float(rate)
            self.fps_fraction = "{}/{}".format(rate.numerator, rate.denominator)
            api = _api(mujoco_api)
            self.renderer = api.Renderer(model, height=HEIGHT, width=WIDTH)
            for camera in CAMERAS:
                staging = self.video_dir / ("capture_" + camera + ".mp4")
                # imageio-ffmpeg otherwise rounds its input rate to two decimals.
                # The later input -r overrides that rounded value exactly.
                writer = imageio_ffmpeg.write_frames(
                    str(staging), (WIDTH, HEIGHT), fps=self.fps,
                    codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv444p",
                    quality=8, macro_block_size=1,
                    input_params=["-r", self.fps_fraction],
                    output_params=["-vsync", "0", "-movflags", "+faststart"],
                )
                self.writers[camera] = writer
                writer.send(None)
            self._ready = True
        except Exception as exc:
            self._error("initialize", exc)

    def _error(self, operation, error, *, camera=None, frame_index=None):
        self.errors.append({"operation": operation, "camera": camera,
                            "frame_index": frame_index,
                            "error": "{}: {}".format(type(error).__name__, error)})

    def capture(self, data, phase):
        """Capture the supplied live pre-command instant; never fill missing times."""
        if self._finished is not None:
            return {"complete": False, "error": "diagnostic recorder is already finished"}
        index = len(self.attempts)
        row = {"frame_index": index, "timestamp_s": None, "phase": str(phase),
               "sampling": "live_before_control_command", "cameras": {}}
        self.attempts.append(row)
        try:
            timestamp = float(data.time)
            if not math.isfinite(timestamp):
                raise ValueError("capture timestamp is not finite")
            row["timestamp_s"] = timestamp
            if index and self.interval_s is not None:
                previous = self.attempts[index - 1]["timestamp_s"]
                if previous is None or not math.isclose(
                        timestamp - previous, self.interval_s,
                        rel_tol=1e-7, abs_tol=1e-8):
                    self._cadence_failed = True
                    raise ValueError("nonuniform or repeated live timestamp; no frames fabricated")
            if not self._ready or self._cadence_failed:
                return row
            for camera in CAMERAS:
                if camera in self.disabled_cameras:
                    continue
                try:
                    self.renderer.update_scene(data, camera=camera)
                    pixels = self._np.asarray(self.renderer.render())
                    if pixels.dtype != self._np.uint8 or pixels.shape != (HEIGHT, WIDTH, 3):
                        raise ValueError("diagnostic renderer must return 240x320 uint8 RGB")
                    pixels = self._np.ascontiguousarray(pixels)
                    pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
                    self.writers[camera].send(pixels)
                    record = {"frame_index": index, "timestamp_s": timestamp,
                              "phase": str(phase), "input_pixel_sha256": pixel_hash}
                    self.camera_rows[camera].append(record)
                    row["cameras"][camera] = record
                except Exception as exc:
                    self.disabled_cameras.add(camera)
                    self._error("capture", exc, camera=camera, frame_index=index)
        except Exception as exc:
            self._error("capture_timestamp", exc, frame_index=index)
        return row

    def _verify_movie(self, path, expected_count):
        decoder = self._ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        try:
            metadata = next(decoder)
            dimensions = tuple(metadata["size"])
            count, hashes = 0, []
            for pixels in decoder:
                if len(pixels) != WIDTH * HEIGHT * 3:
                    raise ValueError("decoded frame byte count is incorrect")
                hashes.append(hashlib.sha256(pixels).hexdigest())
                count += 1
            if dimensions != (WIDTH, HEIGHT) or count != expected_count:
                raise ValueError("decoded dimensions/count disagree with live inputs")
            timing = self._verify_timing(path, expected_count)
            return {"passed": True, "decoded_frame_count": count,
                    "decoded_width": dimensions[0], "decoded_height": dimensions[1],
                    "decoder_reported_fps_rounded": metadata.get("fps"),
                    "exact_decoded_timing": timing,
                    "decoded_rgb_sha256_by_frame": hashes,
                    "lossy_codec": True, "input_and_decoded_hashes_expected_equal": False}
        finally:
            decoder.close()

    def _verify_timing(self, path, expected_count):
        result = subprocess.run(
            [self._ffmpeg.get_ffmpeg_exe(), "-v", "error", "-copyts", "-i", str(path),
             "-map", "0:v:0", "-c:v", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "0",
             "-f", "framemd5", "-"],
            capture_output=True, text=True, check=True, timeout=60,
        )
        return _parse_decoded_timing(result.stdout, expected_count, self.interval_s)

    def finish(self):
        """Close even a failed episode, decode every stored frame, save a receipt."""
        if self._finished is not None:
            return self._finished
        for camera, writer in self.writers.items():
            try:
                writer.close()
            except Exception as exc:
                self._error("close_encoder", exc, camera=camera)
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception as exc:
                self._error("close_renderer", exc)
        cameras = {}
        for camera in CAMERAS:
            staging = self.video_dir / ("capture_" + camera + ".mp4")
            final = self.video_dir / (camera + ".mp4")
            rows = self.camera_rows[camera]
            item = {"camera": camera, "path": str(final),
                    "input_frame_count": len(rows), "frames": rows,
                    "independent_decode": {"passed": False}}
            cameras[camera] = item
            try:
                if not rows:
                    raise ValueError("no actual live frames were captured")
                item["independent_decode"] = self._verify_movie(staging, len(rows))
                item["bytes"] = staging.stat().st_size
                item["sha256"] = _sha256_file(staging)
                # Atomic no-clobber publication. Never replace an earlier movie.
                os.link(staging, final)
                staging.unlink()
                item["published"] = True
            except Exception as exc:
                item["published"] = False
                item["retained_staging_path"] = str(staging) if staging.exists() else None
                self._error("verify_or_publish", exc, camera=camera)
        complete = bool(self.attempts) and not self.errors and all(
            item.get("published") and item["input_frame_count"] == len(self.attempts)
            and item["independent_decode"]["passed"] for item in cameras.values())
        report = {"schema_version": 1, "protocol": "live_teacher_diagnostic_video_v1",
                  "complete": bool(complete), "physical_score_modified": False,
                  "policy_observation_modified": False, "simulation_advanced_by_recorder": False,
                  "sampling": "live_before_control_command", "replay": False,
                  "interpolated_or_repeated_frames": False,
                  "width": WIDTH, "height": HEIGHT, "fps": self.fps,
                  "fps_fraction": self.fps_fraction, "control_interval_s": self.interval_s,
                  "capture_attempt_count": len(self.attempts), "capture_attempts": self.attempts,
                  "last_frame_is_pre_command_not_final_post_state": True,
                  "cameras": cameras, "errors": self.errors,
                  "manifest_path": str(self.manifest_path)}
        try:
            with self.manifest_path.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:
            self._error("write_manifest", exc)
            report["complete"] = False
        self._finished = report
        return report
