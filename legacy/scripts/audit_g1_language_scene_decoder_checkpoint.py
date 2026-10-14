#!/usr/bin/env python3
"""Audit the freeze boundary of a G1 scene action decoder checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


PREFIX = "model.language_action_adapter."
EXPECTED_NEW_KEYS = {
    f"{PREFIX}scene_input_proj.weight",
    f"{PREFIX}scene_input_proj.bias",
    f"{PREFIX}scene_action_decoder.weight",
    f"{PREFIX}scene_action_decoder.bias",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=39000)
    return parser.parse_args()


def model_path(checkpoint: Path) -> Path:
    path = checkpoint / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    args = parse_args()
    source_path = model_path(args.source)
    trained_path = model_path(args.trained)

    with (
        safe_open(source_path, framework="pt", device="cpu") as source,
        safe_open(trained_path, framework="pt", device="cpu") as trained,
    ):
        source_keys = set(source.keys())
        trained_keys = set(trained.keys())
        missing_keys = sorted(source_keys - trained_keys)
        new_keys = trained_keys - source_keys
        unexpected_new_keys = sorted(new_keys - EXPECTED_NEW_KEYS)
        missing_new_keys = sorted(EXPECTED_NEW_KEYS - new_keys)
        changed_common_keys = []
        for key in sorted(source_keys & trained_keys):
            if not torch.equal(source.get_tensor(key), trained.get_tensor(key)):
                changed_common_keys.append(key)

        new_tensor_stats = {}
        for key in sorted(EXPECTED_NEW_KEYS & trained_keys):
            tensor = trained.get_tensor(key).float()
            new_tensor_stats[key] = {
                "shape": list(tensor.shape),
                "finite": bool(torch.isfinite(tensor).all()),
                "nonzero_count": int(torch.count_nonzero(tensor)),
                "l2_norm": float(tensor.norm()),
                "max_abs": float(tensor.abs().max()),
            }

    step_path = args.trained.parent / "training_state" / "training_step.json"
    step_report = json.loads(step_path.read_text(encoding="utf-8"))
    training_step = int(step_report["step"])
    contract_path = args.trained / "triplet_curriculum_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_trainable_names = sorted(key for key in EXPECTED_NEW_KEYS)
    contract_trainable_names = sorted(contract.get("trainable_parameter_names", []))

    checkpoint_passed = (
        not missing_keys
        and not unexpected_new_keys
        and not missing_new_keys
        and not changed_common_keys
        and training_step == args.expected_step
        and contract.get("scene_decoder_only") is True
        and contract_trainable_names == expected_trainable_names
        and all(
            stats["finite"] and stats["nonzero_count"] > 0
            for stats in new_tensor_stats.values()
        )
    )
    report = {
        "experiment": "G1LANG-scene-action-decoder-checkpoint-audit",
        "source_checkpoint": str(args.source),
        "trained_checkpoint": str(args.trained),
        "source_tensor_count": len(source_keys),
        "trained_tensor_count": len(trained_keys),
        "common_tensor_count": len(source_keys & trained_keys),
        "changed_common_tensor_count": len(changed_common_keys),
        "changed_common_keys": changed_common_keys,
        "missing_source_keys": missing_keys,
        "new_keys": sorted(new_keys),
        "unexpected_new_keys": unexpected_new_keys,
        "missing_new_keys": missing_new_keys,
        "new_tensor_stats": new_tensor_stats,
        "training_step": training_step,
        "expected_step": args.expected_step,
        "contract_scene_decoder_only": contract.get("scene_decoder_only"),
        "contract_trainable_parameter_names": contract_trainable_names,
        "audit_passed": checkpoint_passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not checkpoint_passed:
        raise RuntimeError("Scene decoder checkpoint audit failed")


if __name__ == "__main__":
    main()
