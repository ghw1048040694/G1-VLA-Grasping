"""Unified WSL-to-Windows dispatcher for immutable privileged teacher runs.

The Windows installation supplies Isaac/CUDA. Project sources, configuration
and assets come from this project and are verified before the native process.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import time

from .rl_protocol import canonical_hash, file_hash, verify_rl_assets
from .runtime.experiment import write_new_json


def windows_to_wsl(value):
    path = PureWindowsPath(value)
    if not path.is_absolute() or len(path.drive) != 2 or not path.drive.endswith(":"):
        raise ValueError("the native runtime requires an absolute Windows drive path")
    if any(part in ("..",) for part in path.parts):
        raise ValueError("parent traversal is not permitted in a native runtime path")
    return Path("/mnt") / path.drive[0].lower() / Path(*path.parts[1:])


def _snapshot_release(config, run_dir, assets, diagnostic=False):
    from .config import load_config

    root = Path(config["paths"]["root"]).resolve()
    raw_config = root / "configs/experiment.json"
    if load_config(raw_config) != config:
        raise ValueError("native RL runs require the unchanged central configs/experiment.json")
    files = list((root / "src/g1_vla").rglob("*.py")) + list((root / "configs").glob("*.json"))
    files += [root / "pyproject.toml", root / "resources/g1_action31_train_root_stub/meta/info.json"]
    asset_root = Path(config["rl_teacher"]["asset_manifest_path"]).parent
    files += [asset_root / item["path"] for item in assets["manifest"]["files"]]
    files += [Path(config["rl_teacher"]["asset_manifest_path"])]
    if diagnostic:
        from .rl_probe_protocol import reference_snapshot_files
        files += reference_snapshot_files(config)
    selected = config["rl_teacher"]
    if selected["checkpoint_path"]:
        checkpoint = Path(selected["checkpoint_path"])
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        if file_hash(checkpoint) != selected["checkpoint_sha256"]:
            raise ValueError("selected RL checkpoint does not match its explicit SHA256")
        files.append(checkpoint)
    records = []
    for file in sorted(set(files)):
        relative = file.resolve().relative_to(root)
        if not file.is_file() or file.is_symlink():
            raise ValueError("the RL release accepts regular project files only")
        records.append({"path": relative.as_posix(), "bytes": file.stat().st_size,
                        "sha256": file_hash(file)})
    declaration = {"schema_version": 1, "files": records,
                   "source_config_sha256": file_hash(raw_config)}
    digest = canonical_hash(declaration)
    runtime = PureWindowsPath(config["rl_teacher"]["runtime_windows_root"])
    win_release = runtime / "releases" / digest[:20]
    destination = windows_to_wsl(str(win_release))
    manifest_path = destination / "release_manifest.json"
    if destination.exists():
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != {**declaration, "sha256": digest}:
            raise ValueError("an existing native release has a different immutable manifest")
    else:
        destination.mkdir(parents=True, exist_ok=False)
        for row in records:
            target = destination / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / row["path"], target)
        write_new_json(manifest_path, {**declaration, "sha256": digest})
    for row in records:
        target = destination / row["path"]
        if target.stat().st_size != row["bytes"] or file_hash(target) != row["sha256"]:
            raise ValueError("native source release integrity check failed: " + row["path"])
    write_new_json(Path(run_dir) / "manifests/native_release.json", {
        **declaration, "sha256": digest, "windows_release_path": str(win_release),
        "source_asset_manifest_sha256": assets["manifest_sha256"],
        "source_seed_sha256": assets["seed_sha256"],
        "native_installation_reused": True, "old_ACT_payload_exported": False,
        "selected_RL_checkpoint_exported": bool(selected["checkpoint_path"])})
    return digest, win_release, destination


def _snapshot_runtime_assets(config, assets):
    """Keep converter paths stable when only an explicit checkpoint is selected."""
    root = Path(config["paths"]["root"]).resolve()
    environment_sha = file_hash(root / "src/g1_vla/rl_environment.py")
    win_root = (PureWindowsPath(config["rl_teacher"]["runtime_windows_root"]) / "assets"
                / assets["manifest_sha256"] / environment_sha[:20])
    destination = windows_to_wsl(str(win_root))
    source = Path(config["rl_teacher"]["asset_manifest_path"]).parent
    rows = assets["manifest"]["files"]
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows + [{"path": "asset_manifest.json", "bytes": (source / "asset_manifest.json").stat().st_size,
                        "sha256": assets["manifest_sha256"]}]:
        target = destination / row["path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / row["path"], target)
        if target.is_symlink() or target.stat().st_size != row["bytes"] or file_hash(target) != row["sha256"]:
            raise ValueError("shared immutable native asset integrity failure: " + row["path"])
    return win_root


def _smoke_admission(config, release_sha256):
    root = Path(config["artifacts"]["runs_dir"]) / "runs"
    for path in sorted(root.glob("*/reports/rl_teacher.json"), reverse=True):
        try:
            row = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if row.get("stage") == "rl-smoke" and row.get("passed") is True and row.get("release_sha256") == release_sha256:
            return {"report_path": str(path), "sha256": file_hash(path),
                    "release_sha256": release_sha256}
    raise ValueError("RL training requires a passed matching source/config/asset smoke run")


def _native_script(stage, config, win_release, win_output, run_id, release_sha256, win_assets, entrypoint):
    return "\n".join([
        "import os,sys,json,pathlib,hashlib",
        "root=pathlib.Path(" + repr(str(win_release)) + ")",
        "os.chdir(root)",
        "sys.path.insert(0,str(root/'src'))",
        "os.environ['GIT_PYTHON_GIT_EXECUTABLE']=" + repr(config["rl_teacher"]["git_executable"]),
        "output=pathlib.Path(" + repr(str(win_output)) + ")",
        "output.mkdir(parents=True,exist_ok=False)",
        "(output/'process_pid.json').write_text(json.dumps({'pid':os.getpid(),'run_id':" + repr(run_id)
            + ",'release_sha256':" + repr(release_sha256) + "}),encoding='utf-8')",
        "from g1_vla.config import load_config",
        "from g1_vla.rl_protocol import verify_rl_assets",
        "cfg=load_config(root/'configs/experiment.json')",
        "asset_root=pathlib.Path(" + repr(str(win_assets)) + ")",
        "cfg['rl_teacher']['asset_path']=str(asset_root/" + repr(entrypoint) + ")",
        "cfg['rl_teacher']['seed_state_path']=str(asset_root/'seed_state.json')",
        "cfg['rl_teacher']['asset_manifest_path']=str(asset_root/'asset_manifest.json')",
        "verify_rl_assets(cfg)",
        "from g1_vla.rl_diagnostics import run_native_diagnostics" if stage == "rl-diagnose" else
            "from g1_vla.rl_ppo import run_native_stage",
        "run_native_diagnostics(cfg,output)" if stage == "rl-diagnose" else
            "run_native_stage(" + repr(stage) + ",cfg,output)",
    ])


def _package_eval_videos(output, run_dir):
    """Encode actual native RGB frames, preserving their original frame ledger."""
    videos = []
    for frames in sorted(Path(output).glob("environment_*/rgb_frames")):
        paths = sorted(frames.glob("frame_*.png"))
        if not paths:
            continue
        indices = [int(path.stem.split("_")[-1]) for path in paths]
        row = {"frames": len(paths), "source": "actual native Camera RGB",
               "simulation_fps": "500/33", "frame_directory": str(frames),
               "encoded": False}
        try:
            if indices != list(range(indices[0], indices[0] + len(indices))):
                raise ValueError("RGB evidence contains missing frame indices")
            target = frames.parent / "evaluation_env0.mp4"
            with (Path(run_dir) / "logs/native_video_encoding.log").open("ab") as log:
                subprocess.run(["ffmpeg", "-nostdin", "-n", "-v", "error", "-framerate", "500/33",
                    "-start_number", str(indices[0]), "-i", str(frames / "frame_%06d.png"),
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(target)],
                    stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate",
                "-of", "json", str(target)], capture_output=True, text=True, check=True, timeout=15)
            stream = json.loads(probe.stdout)["streams"][0]
            if int(stream["nb_read_frames"]) != len(paths):
                raise ValueError("encoded video frame count differs from actual RGB captures")
            row.update(encoded=True, path=str(target), sha256=file_hash(target),
                       bytes=target.stat().st_size, decoded_frames=int(stream["nb_read_frames"]),
                       encoded_fps=stream["r_frame_rate"])
            native_report = Path(output) / "native_report.json"
            report = json.loads(native_report.read_text()) if native_report.is_file() else {}
            if report.get("stage") == "rl-diagnose":
                row["profile_clips"] = []
                for profile in report.get("profiles", []):
                    name = profile["profile"]["name"]
                    if not name or any(c not in "abcdefghijklmnopqrstuvwxyz_0123456789" for c in name):
                        raise ValueError("invalid diagnostic video profile name")
                    first, last = profile["first_video_frame"], profile["last_video_frame_exclusive"]
                    if not 0 <= first < last <= len(paths):
                        raise ValueError("diagnostic video frame range is invalid")
                    clip = frames.parent / ("evaluation_env0_" + name + ".mp4")
                    with (Path(run_dir) / "logs/native_video_encoding.log").open("ab") as log:
                        subprocess.run(["ffmpeg", "-nostdin", "-n", "-v", "error", "-framerate", "500/33",
                            "-start_number", str(first), "-i", str(frames / "frame_%06d.png"),
                            "-frames:v", str(last-first), "-c:v", "libx264", "-crf", "20",
                            "-pix_fmt", "yuv420p", str(clip)], stdout=log, stderr=subprocess.STDOUT,
                            check=True, timeout=60)
                    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_frames", "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "json",
                        str(clip)], capture_output=True, text=True, check=True, timeout=15)
                    clip_stream = json.loads(probe.stdout)["streams"][0]
                    if int(clip_stream["nb_read_frames"]) != last-first:
                        raise ValueError("diagnostic profile video has incomplete frames")
                    row["profile_clips"].append({"profile": name, "path": str(clip),
                        "first_source_frame": first, "last_source_frame_exclusive": last,
                        "decoded_frames": int(clip_stream["nb_read_frames"]),
                        "encoded_fps": clip_stream["r_frame_rate"], "sha256": file_hash(clip),
                        "bytes": clip.stat().st_size, "raw_RGB_source_frames_unmodified": True,
                        "encoding": "lossy H.264 CRF20; unmodified camera PNG source, no overlays"})
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            row["error"] = str(exc)
        videos.append(row)
    write_new_json(Path(run_dir) / "reports/native_eval_videos.json", {"videos": videos,
        "scope": "env0 actual RGB at native control times; episode boundaries remain in frame ledger"})
    return videos


def _diagnostic_video_delivery(videos, native, expected_profiles=None):
    """Require every declared physical profile's actual complete RGB clip."""
    if expected_profiles is None:
        from .rl_probe_protocol import RL_PROBE_DEFAULTS
        expected_profiles = RL_PROBE_DEFAULTS["profiles"]
    if native.get("video_evidence_complete") is not True or len(videos) != 1:
        return False
    video = videos[0]
    profiles = native.get("profiles", [])
    clips = video.get("profile_clips", [])
    if (video.get("encoded") is not True or not expected_profiles or
            len(clips) != len(expected_profiles) or len(profiles) != len(expected_profiles) or
            video.get("encoded_fps") != "500/33"):
        return False
    previous = 0
    for clip, profile, expected in zip(clips, profiles, expected_profiles):
        first, last = profile["first_video_frame"], profile["last_video_frame_exclusive"]
        if (profile["profile"] != expected or clip.get("profile") != expected["name"] or
                clip.get("first_source_frame") != first or clip.get("last_source_frame_exclusive") != last or
                first != previous or last-first != profile["control_steps"] or
                clip.get("decoded_frames") != last-first or clip.get("encoded_fps") != "500/33"):
            return False
        previous = last
    return video.get("decoded_frames") == previous


def run_rl_teacher_stage(stage, config, run_dir):
    if stage not in ("rl-audit", "rl-smoke", "rl-train", "rl-eval", "rl-diagnose"):
        raise ValueError("unsupported privileged RL stage")
    assets = verify_rl_assets(config)
    run_dir = Path(run_dir)
    if config["train"]["enabled"] or config["wm"]["enabled"]:
        raise ValueError("this RL route must not simultaneously enable ACT or world-model training")
    if os.name == "nt":
        raise ValueError("invoke the unified project entry from WSL; native execution is its frozen child")
    cmd = Path("/mnt/c/Windows/System32/cmd.exe")
    if not cmd.is_file():
        raise ValueError("the declared Windows Isaac runtime is unavailable from this host")
    if stage == "rl-diagnose":
        from .rl_probe_protocol import verify_probe_reference
        verify_probe_reference(config)
        if config["rl_diagnostics"]["enabled"] is not True:
            raise ValueError("rl-diagnose requires an explicit enabled diagnostic configuration")
    if stage == "rl-diagnose":
        release_sha, win_release, local_release = _snapshot_release(config, run_dir, assets, diagnostic=True)
    else:
        release_sha, win_release, local_release = _snapshot_release(config, run_dir, assets)
    win_assets = _snapshot_runtime_assets(config, assets)
    admission = _smoke_admission(config, release_sha) if stage == "rl-train" else None
    if stage == "rl-eval":
        selected = config["rl_teacher"]
        if not selected["checkpoint_path"] or not selected["checkpoint_sha256"]:
            raise ValueError("RL evaluation requires an explicitly selected checkpoint and SHA256")
    run_id = run_dir.name
    runtime = PureWindowsPath(config["rl_teacher"]["runtime_windows_root"])
    win_output = runtime / "runs" / run_id
    local_output = windows_to_wsl(str(win_output))
    if local_output.exists():
        raise ValueError("native RL runs are immutable; choose a new run id")
    script = _native_script(stage, config, win_release, win_output, run_id, release_sha, win_assets, assets["entrypoint"])
    script_path = run_dir / "logs/native_invocation.txt"
    script_path.write_text(script, encoding="utf-8")
    started = datetime.now(timezone.utc).isoformat()
    begin = time.monotonic()
    native_python = str(PureWindowsPath(config["rl_teacher"]["isaac_python_bat"]))
    argument = "call " + native_python + " -X g1_rl_run_id=" + run_id + " -"
    command = [str(cmd), "/d", "/c", argument]
    with (run_dir / "logs/native_stdout.log").open("wb") as stream:
        process = subprocess.Popen(command, cwd=local_release, stdin=subprocess.PIPE,
            stdout=stream, stderr=subprocess.STDOUT)
        timed_out = False
        try:
            timeout = (config["rl_diagnostics"]["max_wall_time_s"] + 180 if stage == "rl-diagnose" else
                       config["rl_teacher"]["max_wall_time_s"] + 180 if stage == "rl-train" else 900)
            process.communicate(script.encode("utf-8"), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            pid_file = local_output / "process_pid.json"
            if pid_file.is_file():
                owner = json.loads(pid_file.read_text())
                if owner.get("run_id") == run_id and owner.get("release_sha256") == release_sha:
                    subprocess.run(["/mnt/c/Windows/System32/taskkill.exe", "/PID", str(int(owner["pid"])),
                        "/T", "/F"], stdout=stream, stderr=subprocess.STDOUT, timeout=30)
            process.kill()
            process.communicate(timeout=30)
    native_report_path = local_output / "native_report.json"
    native = json.loads(native_report_path.read_text(encoding="utf-8")) if native_report_path.is_file() else {
        "passed": False, "training_started": None, "optimizer_steps": None,
        "error": "native process ended without its authoritative native_report.json; optimization status unknown"}
    videos = _package_eval_videos(local_output, run_dir) if stage in ("rl-eval", "rl-diagnose") else []
    if local_output.exists():
        shutil.copytree(local_output, run_dir / "native", dirs_exist_ok=False)
    result = {"stage": stage, "passed": process.returncode == 0 and not timed_out and native.get("passed") is True,
        "native_return_code": process.returncode, "timed_out": timed_out, "started_utc": started,
        "wall_time_s": time.monotonic() - begin, "release_sha256": release_sha,
        "source_asset_manifest_sha256": assets["manifest_sha256"], "source_seed_sha256": assets["seed_sha256"],
        "native_output_windows_path": str(win_output), "native_report": native,
        "native_shared_asset_path": str(win_assets),
        "videos": videos,
        "smoke_admission": admission, "training_started": native.get("training_started", False),
        "optimizer_steps": native.get("optimizer_steps", 0), "formal_original20_evaluated": False,
        "original_mujoco_teacher_modified": False, "old_ACT_test_accessed": False,
        "scope": config["rl_teacher"]["scope"]}
    if stage == "rl-diagnose":
        from .rl_probe_protocol import validate_probe_settings
        declared = validate_probe_settings(config["rl_diagnostics"])
        result["video_evidence_complete"] = _diagnostic_video_delivery(videos, native, declared["profiles"])
        result["declared_profile_plan"] = declared["profile_plan"]
        result["four_way_comparison_complete"] = (declared["profile_plan"] == "four_way_v1" and
                                                   result["video_evidence_complete"])
        result["source_identical_trained_eval"] = False
        result["passed"] = result["passed"] and result["video_evidence_complete"]
    write_new_json(run_dir / "reports/rl_teacher.json", result)
    return result
