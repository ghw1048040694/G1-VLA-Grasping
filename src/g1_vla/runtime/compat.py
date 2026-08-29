"""Single compatibility boundary for the existing G1 Python scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .run_context import RunContext, workspace_root


COMPATIBILITY_SCRIPTS = {
    "collect": Path("scripts/collect_g1_language_pick_place_dataset.py"),
    "audit": Path("scripts/audit_g1_execution_contract.py"),
    "train-vla": Path("scripts/train_g1_continuous_dagger_vla.py"),
    "train-world-model": Path("scripts/train_g1_task_value_world_model.py"),
    "eval": Path("scripts/evaluate_g1_policy_intervention_wm_mpc.py"),
    "play": Path("scripts/play_g1_language_episode.py"),
}


def run_compatibility_script(kind: str, forwarded_args: Sequence[str], context: RunContext) -> dict:
    relative_script = COMPATIBILITY_SCRIPTS[kind]
    script = workspace_root() / relative_script
    if not script.is_file():
        raise FileNotFoundError(f"G1 compatibility entry point is missing: {script}")
    args = list(forwarded_args)
    if args[:1] == ["--"]:
        args = args[1:]
    command = [sys.executable, str(script), *args]
    stdout_path = context.path / "logs" / "stdout.log"
    stderr_path = context.path / "logs" / "stderr.log"
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.run(command, cwd=workspace_root(), env=environment, stdout=stdout, stderr=stderr, check=False)
    return {
        "schema": "g1_vla_compat_run_v1",
        "workflow": kind,
        "compatibility_script": str(relative_script),
        "forwarded_args": args,
        "exit_code": process.returncode,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "business_logic_migrated": False,
    }


__all__ = ["COMPATIBILITY_SCRIPTS", "run_compatibility_script"]
