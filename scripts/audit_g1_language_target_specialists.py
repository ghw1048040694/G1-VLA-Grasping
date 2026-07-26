#!/usr/bin/env python3
"""Audit the final target-specific SmolVLA checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


TARGETS = ("red_triangle", "yellow_rod", "green_cube")
REQUIRED_FILES = (
    "pretrained_model/config.json",
    "pretrained_model/model.safetensors",
    "pretrained_model/train_config.json",
    "training_state/optimizer_param_groups.json",
    "training_state/optimizer_state.safetensors",
    "training_state/rng_state.safetensors",
    "training_state/scheduler_state.json",
    "training_state/training_step.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=5000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checkpoints(values: list[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Checkpoint must use TARGET=PATH syntax: {value}")
        target, raw_path = value.split("=", 1)
        if target not in TARGETS or target in checkpoints:
            raise ValueError(f"Invalid or duplicate target: {target}")
        checkpoints[target] = Path(raw_path)
    missing = sorted(set(TARGETS) - set(checkpoints))
    if missing:
        raise ValueError(f"Missing target checkpoints: {missing}")
    return checkpoints


def tensor_audit(path: Path) -> dict:
    all_finite = True
    tensor_count = 0
    parameter_count = 0
    keys = []
    shapes = {}
    with safe_open(path, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            tensor = checkpoint.get_tensor(key)
            keys.append(key)
            shapes[key] = list(tensor.shape)
            tensor_count += 1
            parameter_count += tensor.numel()
            all_finite = all_finite and bool(torch.isfinite(tensor).all())
    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "all_tensors_finite": all_finite,
        "keys": keys,
        "shapes": shapes,
    }


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    source_model = args.source / "model.safetensors"
    if not source_model.is_file():
        raise FileNotFoundError(source_model)
    source_hash = sha256(source_model)
    source_tensors = tensor_audit(source_model)

    reports = {}
    reference_keys: list[str] | None = None
    reference_shapes: dict | None = None
    for target in TARGETS:
        checkpoint = checkpoints[target]
        missing_files = [name for name in REQUIRED_FILES if not (checkpoint / name).is_file()]
        step_path = checkpoint / "training_state/training_step.json"
        step = None
        if step_path.is_file():
            step = int(json.loads(step_path.read_text(encoding="utf-8"))["step"])
        model_path = checkpoint / "pretrained_model/model.safetensors"
        model_hash = sha256(model_path) if model_path.is_file() else None
        tensors = tensor_audit(model_path) if model_path.is_file() else None
        if tensors is not None and reference_keys is None:
            reference_keys = tensors["keys"]
            reference_shapes = tensors["shapes"]
        structure_matches = bool(
            tensors is not None
            and tensors["keys"] == source_tensors["keys"]
            and tensors["shapes"] == source_tensors["shapes"]
            and tensors["keys"] == reference_keys
            and tensors["shapes"] == reference_shapes
        )
        passed = bool(
            not missing_files
            and step == args.expected_step
            and model_hash != source_hash
            and tensors is not None
            and tensors["all_tensors_finite"]
            and structure_matches
        )
        reports[target] = {
            "checkpoint": str(checkpoint),
            "training_step": step,
            "expected_step": args.expected_step,
            "missing_required_files": missing_files,
            "model_sha256": model_hash,
            "differs_from_source_sha256": model_hash != source_hash,
            "tensor_count": None if tensors is None else tensors["tensor_count"],
            "parameter_count": None if tensors is None else tensors["parameter_count"],
            "all_tensors_finite": bool(tensors and tensors["all_tensors_finite"]),
            "structure_matches_source_and_specialists": structure_matches,
            "audit_passed": passed,
        }

    audit_passed = source_tensors["all_tensors_finite"] and all(
        report["audit_passed"] for report in reports.values()
    )
    report = {
        "experiment": "G1FINAL-01-target-specialists-checkpoint-audit",
        "source_checkpoint": str(args.source),
        "source_model_sha256": source_hash,
        "source_all_tensors_finite": source_tensors["all_tensors_finite"],
        "specialists": reports,
        "audit_passed": audit_passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not audit_passed:
        raise RuntimeError("Target specialist checkpoint audit failed")


if __name__ == "__main__":
    main()
