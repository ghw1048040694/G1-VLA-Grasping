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


def _snapshot_release(config, run_dir, assets):
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


def _native_script(stage, config, win_release, win_output, run_id, release_sha256, win_assets):
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
        "cfg['rl_teacher']['asset_path']=str(asset_root/'robot.xml')",
        "cfg['rl_teacher']['seed_state_path']=str(asset_root/'seed_state.json')",
        "cfg['rl_teacher']['asset_manifest_path']=str(asset_root/'asset_manifest.json')",
        "verify_rl_assets(cfg)",
        "from g1_vla.rl_ppo import run_native_stage",
        "run_native_stage(" + repr(stage) + ",cfg,output)",
    ])


def run_rl_teacher_stage(stage, config, run_dir):
    if stage not in ("rl-audit", "rl-smoke", "rl-train", "rl-eval"):
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
    script = _native_script(stage, config, win_release, win_output, run_id, release_sha, win_assets)
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
            process.communicate(script.encode("utf-8"), timeout=(config["rl_teacher"]["max_wall_time_s"] + 180
                if stage == "rl-train" else 900))
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
    if local_output.exists():
        shutil.copytree(local_output, run_dir / "native", dirs_exist_ok=False)
    result = {"stage": stage, "passed": process.returncode == 0 and not timed_out and native.get("passed") is True,
        "native_return_code": process.returncode, "timed_out": timed_out, "started_utc": started,
        "wall_time_s": time.monotonic() - begin, "release_sha256": release_sha,
        "source_asset_manifest_sha256": assets["manifest_sha256"], "source_seed_sha256": assets["seed_sha256"],
        "native_output_windows_path": str(win_output), "native_report": native,
        "native_shared_asset_path": str(win_assets),
        "smoke_admission": admission, "training_started": native.get("training_started", False),
        "optimizer_steps": native.get("optimizer_steps", 0), "formal_original20_evaluated": False,
        "original_mujoco_teacher_modified": False, "old_ACT_test_accessed": False,
        "scope": config["rl_teacher"]["scope"]}
    write_new_json(run_dir / "reports/rl_teacher.json", result)
    return result
