"""Immutable records for explicit, single-stage G1 workflows."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_state(root: Path) -> dict:
    result = {}
    for key, arguments in (("revision", ["rev-parse", "HEAD"]), ("branch", ["branch", "--show-current"]), ("status", ["status", "--short"])):
        try:
            command = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, text=True, timeout=5)
            result[key] = command.stdout.strip() if command.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result[key] = None
    result["dirty"] = bool(result.get("status"))
    return result


def source_fingerprint(root: Path) -> dict:
    """Record the actual preparation code even while the worktree is dirty."""
    hashes = {}
    paths = list((root / "src/g1_vla").rglob("*.py"))
    paths += [root / "pyproject.toml", root / "configs/experiment.json"]
    for path in sorted(paths):
        if path.is_file():
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"files": hashes, "sha256": canonical_hash(hashes)}


def write_new_json(path: Path, value: Any) -> None:
    """Never overwrite an existing evidence file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


class ExperimentRun:
    def __init__(self, project_root: Path, runtime_root: Path, stage: str, config: Mapping[str, Any], command: list, run_id: str | None = None):
        self.project_root = Path(project_root).resolve()
        self.runtime_root = Path(runtime_root).resolve()
        self.stage = stage
        self.config = dict(config)
        self.command = list(command)
        self.run_id = run_id or ("G1-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
        if not self.run_id or Path(self.run_id).name != self.run_id or self.run_id in (".", ".."):
            raise ValueError("run_id must be a single directory name")
        self.path = self.runtime_root / "runs" / self.run_id

    def begin(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=False)
        for name in ("logs", "manifests", "reports"):
            (self.path / name).mkdir()
        write_new_json(self.path / "effective_config.json", self.config)
        write_new_json(self.path / "provenance.json", {
            "schema_version": 1,
            "stage": self.stage,
            "command": self.command,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "effective_config_sha256": canonical_hash(self.config),
            "git": git_state(self.project_root),
            "source": source_fingerprint(self.project_root),
            "python": platform.python_version(),
            "project_root": str(self.project_root),
            "training_started": False,
        })
        write_new_json(self.path / "started.json", {"stage": self.stage, "status": "running"})
        return self.path

    def finish(self, result: Mapping[str, Any], success: bool) -> dict:
        training = result.get("training_started", False)
        report = {**dict(result), "run_id": self.run_id, "run_dir": str(self.path),
                  "training_started": None if training is None else bool(training)}
        write_new_json(self.path / "summary.json", report)
        write_new_json(self.path / "finished.json", {
            "status": "success" if success else "failed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "training_started": report["training_started"],
        })
        return report
