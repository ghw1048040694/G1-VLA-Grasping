"""Actual teacher camera videos and phase scores, without changing physics.

Called by the existing teacher stages after each episode. Videos contain only
the recorded pre-command sensor frames, at the recorded control cadence.
These are diagnostic evidence; they cannot make an episode pass a gate.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from fractions import Fraction

from .artifacts import sha256_file
from .runtime.experiment import write_new_json


PHASES = ("home", "high_approach", "pregrasp", "contact", "grasp", "lift", "hold")


def phase_scores(report):
    """Keep unvisited phases and the failed phase visible in the denominator."""
    records = {row["phase"]: row for row in report.get("phases", [])}
    result = {}
    for phase in PHASES:
        row = records.get(phase)
        result[phase] = {
            "entered": row is not None,
            "completed": bool(row and row.get("completed") is True),
            "start_s": row.get("start_s") if row else None,
            "end_s": row.get("end_s") if row else None,
            "actual_contact": row.get("actual_contact") if row else None,
            "tracking": row.get("tracking") if row else None,
        }
    return result


def _write_video(path, frames, fps):
    import imageio.v2 as imageio
    import imageio_ffmpeg

    path = Path(path)
    temporary = path.with_name(path.stem + ".partial.mp4")
    if path.exists() or temporary.exists():
        raise ValueError("teacher videos are immutable; choose a new run")
    # yuv444p and macro_block_size=1 retain the sensor dimensions, including
    # small diagnostic cameras. There is no resize, interpolation or replay.
    rate = str(Fraction(fps).limit_denominator(1000000))
    with imageio.get_writer(str(temporary), fps=fps, codec="libx264", quality=8,
                            pixelformat="yuv444p", macro_block_size=1,
                            input_params=["-r", rate]) as writer:
        for frame in frames:
            writer.append_data(frame)
    decoder = imageio_ffmpeg.read_frames(str(temporary), pix_fmt="rgb24")
    metadata = next(decoder)
    decoded_count = sum(1 for _ in decoder)
    height, width = frames.shape[1:3]
    if decoded_count != len(frames) or list(metadata["size"]) != [width, height]:
        raise ValueError("encoded teacher video frame count/dimensions differ from the raw sensor")
    timing = probe_video_timing(temporary, len(frames), fps)
    # link is exclusive, unlike rename on POSIX; never replace past evidence.
    os.link(temporary, path)
    temporary.unlink()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "frame_count": decoded_count, "width": width, "height": height,
            "fps": fps, "input_frame_rate": rate,
            "decoded_fps_rounded_by_ffmpeg": metadata.get("fps"), "decode_verified": True,
            "presentation_timing": timing}


def probe_video_timing(path, count, fps):
    """Decode presentation timestamps; rounded FFmpeg FPS text is insufficient."""
    import imageio_ffmpeg

    process = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-i", str(path),
                              "-map", "0:v:0", "-vf", "showinfo", "-f", "null", "-"],
                             capture_output=True, text=True, timeout=60)
    if process.returncode:
        raise ValueError("teacher video timestamp decode failed: " + process.stderr[-1000:])
    base = re.search(r"config in time_base: (\d+)/(\d+)", process.stderr)
    stamps = re.findall(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)", process.stderr)
    if base is None or len(stamps) != count:
        raise ValueError("teacher video timestamp evidence is incomplete")
    timebase = Fraction(int(base[1]), int(base[2]))
    rate = Fraction(fps).limit_denominator(1000000)
    pts = [int(stamp) for _, stamp in stamps]
    if any(int(index) != i or stamp * timebase != Fraction(i) / rate
           for i, ((index, _), stamp) in enumerate(zip(stamps, pts))):
        raise ValueError("teacher video presentation times differ from the recorded cadence")
    return {"verified": True, "time_base": str(timebase), "pts": pts,
            "frame_rate": str(rate), "frame_count": count,
            "method": "decode every frame with bundled FFmpeg showinfo; exact rational PTS comparison"}


def export_teacher_evidence(report, raw_path, episode_dir, config):
    """Export every recorded frame and retain the actual failure, even if short."""
    import numpy as np

    raw_path, episode_dir = Path(raw_path), Path(episode_dir)
    evidence = {
        "schema_version": 1, "scope": "teacher diagnostic camera and phase evidence",
        "raw_path": str(raw_path), "raw_sha256": sha256_file(raw_path),
        "physical_passed": report.get("passed") is True, "failure": report.get("failure"),
        "phases": phase_scores(report), "closure_start_event": report.get("closure_start_event"),
        "videos": [], "training_started": False,
        "frame_semantics": "actual pre-command sensor observation; final post-command state has no invented RGB frame",
        "physical_result_unmodified": True,
    }
    with np.load(raw_path, allow_pickle=False) as raw:
        times = raw["observation_time_s"]
        starts, ends = raw["command_start_time_s"], raw["command_end_time_s"]
        period = config["control"]["physics_dt_s"] * config["control"]["physics_substeps"]
        if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
                or not np.array_equal(times, starts)
                or not np.allclose(ends - starts, period, rtol=0, atol=1e-8)
                or not np.allclose(np.diff(times), period, rtol=0, atol=1e-8)):
            raise ValueError("teacher video requires exact contiguous recorded control timestamps")
        keys = raw["camera_keys"].tolist()
        if set(keys) != set(config["observation"]["camera_keys"]):
            raise ValueError("teacher video camera identities differ from the unified observation contract")
        evidence.update(observation_count=len(times), first_observation_time_s=float(times[0]),
                        last_observation_time_s=float(times[-1]), final_control_end_s=float(ends[-1]),
                        control_period_s=period)
        video_dir = episode_dir / "videos"
        video_dir.mkdir(exist_ok=False)
        for index, key in enumerate(keys):
            frames = raw["images_camera_{}".format(index)]
            if (frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[0] != len(times)
                    or frames.shape[-1] != 3):
                raise ValueError("teacher RGB frames must be recorded uint8 THWC arrays")
            video = _write_video(video_dir / "camera_{}.mp4".format(index), frames, 1. / period)
            video.update(camera_key=key, raw_array_key="images_camera_{}".format(index))
            evidence["videos"].append(video)
    evidence["complete"] = len(evidence["videos"]) == len(config["observation"]["camera_keys"])
    write_new_json(episode_dir / "teacher_evidence.json", evidence)
    return evidence
