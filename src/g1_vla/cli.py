"""Unified lifecycle CLI for the paused G1 VLA project."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from . import __version__
from .runtime import RunContext, default_runtime_root, workspace_root
from .runtime.compat import COMPATIBILITY_SCRIPTS, run_compatibility_script


def _runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-root", type=Path, default=default_runtime_root())
    parser.add_argument("--run-id", help="Explicit run directory name; defaults to a timestamped ID")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without writing artifacts")
    parser.add_argument("--json", action="store_true", help="Print compact JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="g1-vla", description="G1 VLA lifecycle CLI; the project is currently paused.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("doctor", "Check environment, assets, checkpoints, and workflow scripts"),
        ("prepare", "Validate the paused G1 workspace without launching a simulator"),
        ("smoke", "Parse the warehouse asset and verify the lightweight project contract"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        _runtime_args(command)
    for name in COMPATIBILITY_SCRIPTS:
        command = subparsers.add_parser(name, help=f"Run {name} through the recorded compatibility entry point")
        _runtime_args(command)
        command.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments forwarded to the existing G1 script")
    report = subparsers.add_parser("report", help="Summarize completed G1 unified runs")
    _runtime_args(report)
    report.add_argument("--limit", type=int, default=20)
    return parser


def _print(payload: Any, compact: bool = False) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=None if compact else 2, default=str))


def _context(args: argparse.Namespace, kind: str, config: dict[str, Any]) -> RunContext | None:
    if args.dry_run:
        return None
    return RunContext.create(kind, [sys.executable, *sys.argv], runtime_root=args.runtime_root, run_id=args.run_id, config=config)


def _finish(context: RunContext | None, success: bool, summary: dict[str, Any], error: Exception | None = None, *, compact: bool = False) -> int:
    output = dict(summary)
    if context is not None:
        output.update({"run_id": context.run_id, "run_dir": str(context.path)})
        context.write_summary(output)
        context.finish(success, error=None if error is None else f"{type(error).__name__}: {error}")
    _print(output, compact)
    if error is not None:
        print(f"g1-vla: {type(error).__name__}: {error}", file=sys.stderr)
    return 0 if success else 1


def _dependency(name: str) -> dict[str, Any]:
    found = importlib.util.find_spec(name) is not None
    result: dict[str, Any] = {"installed": found}
    if found:
        try:
            module = __import__(name)
            result["version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - diagnostics only
            result["import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _doctor_report() -> dict[str, Any]:
    root = workspace_root()
    asset = root / "resources/g1_warehouse_pregrasp.xml"
    scripts = {name: (root / path).is_file() for name, path in COMPATIBILITY_SCRIPTS.items()}
    checks = {
        "python": {"version": platform.python_version(), "required": ">=3.10"},
        "package": {"version": __version__, "root": str(root)},
        "dependencies": {"torch": _dependency("torch"), "genesis": _dependency("genesis"), "mujoco": _dependency("mujoco")},
        "asset": {"path": str(asset), "present": asset.is_file()},
        "reference_checkpoint": {"path": str(root / "resources/checkpoints/G1LANG-423b_action_token_vla_best.pt"), "present": (root / "resources/checkpoints/G1LANG-423b_action_token_vla_best.pt").is_file()},
        "compatibility_scripts": scripts,
        "paused": True,
    }
    ready = sys.version_info >= (3, 10) and checks["asset"]["present"] and all(scripts.values())
    return {"schema": "g1_vla_doctor_v1", "ready_for_lightweight_smoke": bool(ready), "checks": checks}


def command_doctor(args: argparse.Namespace) -> int:
    try:
        report = _doctor_report()
        report["dry_run"] = bool(args.dry_run)
        if args.dry_run:
            return _finish(None, bool(report["ready_for_lightweight_smoke"]), report, compact=args.json)
        context = _context(args, "doctor", {})
        return _finish(context, bool(report["ready_for_lightweight_smoke"]), report, compact=args.json)
    except Exception as exc:
        return _finish(None, False, {"schema": "g1_vla_doctor_v1"}, exc, compact=getattr(args, "json", False))


def command_smoke(args: argparse.Namespace) -> int:
    context: RunContext | None = None
    try:
        asset = workspace_root() / "resources/g1_warehouse_pregrasp.xml"
        root = ET.parse(asset).getroot()
        worldbody = root.find("worldbody")
        report = {
            "schema": "g1_vla_lightweight_smoke_v1",
            "asset": str(asset),
            "asset_parse": True,
            "has_worldbody": worldbody is not None,
            "body_count": len(root.findall(".//body")),
            "geom_count": len(root.findall(".//geom")),
            "simulator_launched": False,
            "paused": True,
        }
        success = bool(report["has_worldbody"] and report["body_count"] > 0 and report["geom_count"] > 0)
        if args.dry_run:
            report["dry_run"] = True
            return _finish(None, success, report, compact=args.json)
        context = _context(args, "smoke", {})
        return _finish(context, success, report, compact=args.json)
    except Exception as exc:
        return _finish(context, False, {"schema": "g1_vla_lightweight_smoke_v1"}, exc, compact=getattr(args, "json", False))


def command_prepare(args: argparse.Namespace) -> int:
    report = _doctor_report()
    result = {
        "schema": "g1_vla_prepare_v1",
        "ready_for_lightweight_smoke": report["ready_for_lightweight_smoke"],
        "simulator_launched": False,
        "next": "g1-vla smoke" if report["ready_for_lightweight_smoke"] else "repair the workspace contract",
        "paused": True,
    }
    if args.dry_run:
        result["dry_run"] = True
        return _finish(None, bool(result["ready_for_lightweight_smoke"]), result, compact=args.json)
    context = _context(args, "prepare", {})
    return _finish(context, bool(result["ready_for_lightweight_smoke"]), result, compact=args.json)


def command_compatibility(args: argparse.Namespace) -> int:
    context: RunContext | None = None
    try:
        config = {"forwarded_args": list(args.extra), "compatibility_script": str(COMPATIBILITY_SCRIPTS[args.command]), "paused": True}
        if args.dry_run:
            return _finish(None, True, {"schema": "g1_vla_compat_plan_v1", "workflow": args.command, "config": config, "dry_run": True}, compact=args.json)
        context = _context(args, args.command, config)
        report = run_compatibility_script(args.command, args.extra, context)
        return _finish(context, report["exit_code"] == 0, report, compact=args.json)
    except Exception as exc:
        return _finish(context, False, {"schema": "g1_vla_compat_run_v1", "workflow": args.command}, exc, compact=getattr(args, "json", False))


def command_report(args: argparse.Namespace) -> int:
    context: RunContext | None = None
    try:
        runs_root = args.runtime_root.expanduser().resolve() / "runs"
        rows = []
        for path in sorted(runs_root.glob("*/summary.json"), key=lambda value: value.stat().st_mtime, reverse=True) if runs_root.is_dir() else ():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if payload.get("schema") == "g1_vla_report_v1":
                continue
            rows.append(payload)
            if len(rows) >= max(1, args.limit):
                break
        report = {"schema": "g1_vla_report_v1", "runs_root": str(runs_root), "count": len(rows), "runs": rows, "paused": True}
        if args.dry_run:
            report["dry_run"] = True
            return _finish(None, True, report, compact=args.json)
        context = _context(args, "report", {"limit": args.limit})
        return _finish(context, True, report, compact=args.json)
    except Exception as exc:
        return _finish(context, False, {"schema": "g1_vla_report_v1"}, exc, compact=getattr(args, "json", False))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args, unknown = parser.parse_known_args(raw_args)
    if unknown:
        if args.command in COMPATIBILITY_SCRIPTS:
            command_index = raw_args.index(args.command)
            tail = raw_args[command_index + 1 :]
            forwarded: list[str] = []
            runtime_value_options = {"--runtime-root", "--run-id"}
            index = 0
            while index < len(tail):
                token = tail[index]
                if token == "--dry-run":
                    args.dry_run = True
                    index += 1
                    continue
                if token == "--json":
                    args.json = True
                    index += 1
                    continue
                if token in runtime_value_options:
                    if index + 1 >= len(tail):
                        parser.error(f"{token} requires a value")
                    setattr(args, token[2:].replace("-", "_"), Path(tail[index + 1]) if token == "--runtime-root" else tail[index + 1])
                    index += 2
                    continue
                if any(token.startswith(option + "=") for option in runtime_value_options):
                    option, value = token.split("=", 1)
                    setattr(args, option[2:].replace("-", "_"), Path(value) if option == "--runtime-root" else value)
                    index += 1
                    continue
                forwarded.append(token)
                index += 1
            args.extra = forwarded
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if args.command == "doctor":
        return command_doctor(args)
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "smoke":
        return command_smoke(args)
    if args.command in COMPATIBILITY_SCRIPTS:
        return command_compatibility(args)
    if args.command == "report":
        return command_report(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
