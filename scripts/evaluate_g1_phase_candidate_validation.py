#!/usr/bin/env python3
"""Measure phase-3 contact-palm error on held-out randomized RGB states.

This is deliberately an offline eligibility check, not a grasp-success claim.
The succeeding MuJoCo rollout must still prove physical hand-object contact.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from g1_phase_candidate_contact_vla import (
    PHASE_COUNT,
    TARGET_COUNT,
    encode_tasks,
    load_checkpoint,
)


OBJECT_NAMES = ("red_triangle", "yellow_rod", "green_cube")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def describe(errors_m: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(errors_m)),
        "mean_mm": float(errors_m.mean() * 1000.0),
        "median_mm": float(np.median(errors_m) * 1000.0),
        "p95_mm": float(np.percentile(errors_m, 95) * 1000.0),
        "max_mm": float(errors_m.max() * 1000.0),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    cache = torch.load(args.cache_dir / "val.pt", map_location="cpu", weights_only=False)
    model, vocab, stats, _ = load_checkpoint(args.checkpoint, device)
    if cache.get("action_contract") != "active_target_passive_palm_residual":
        raise ValueError("Expected camera-localized active-palm residual labels")
    if not bool((cache["phase"] == PHASE_COUNT - 1).all()):
        raise ValueError("This evaluator only accepts phase-3 physical-contact labels")

    token_ids, token_mask = encode_tasks(
        [cache["tasks"][int(index)] for index in cache["task_index"]], vocab
    )
    state = (cache["state"].to(device) - stats["state_mean"]) / stats["state_std"]
    output = model(
        state,
        cache["visual_object_position"].to(device),
        cache["palm_position"].to(device),
        cache["color_features"].to(device),
        token_ids.to(device),
        token_mask.to(device),
    )
    rows = torch.arange(len(cache["target"]), device=device)
    target = cache["target"].to(device)
    phase = cache["phase"].to(device)
    predicted = output["action_candidates"][rows, target, phase]
    predicted = predicted * stats["action_std"].to(device) + stats["action_mean"].to(device)
    expected = cache["action"].to(device)

    visual = cache["visual_object_position"].to(device).reshape(-1, TARGET_COUNT, 3)
    arms = cache["active_arm"].to(device)
    active_columns = 3 + arms[:, None] * 3 + torch.arange(3, device=device)[None]
    predicted_residual = predicted.gather(1, active_columns)
    expected_residual = expected.gather(1, active_columns)
    target_position = visual[rows, target]
    predicted_site = predicted_residual + target_position
    expected_site = expected_residual + target_position
    errors_m = torch.linalg.vector_norm(predicted_site - expected_site, dim=-1).cpu().numpy()

    per_object: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(OBJECT_NAMES):
        per_object[name] = describe(errors_m[cache["target"].numpy() == index])
    phase_accuracy = float(
        (output["phase_logits"].argmax(-1) == phase).float().mean().item()
    )
    active_arm_accuracy = float(
        (output["active_arm_logits"].argmax(-1) == arms).float().mean().item()
    )
    result = {
        "experiment": "G1LANG-heldout-randomized-phase3-contact-eligibility",
        "checkpoint": str(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "input_contract": "held-out randomized head-camera RGB geometry + language + state31 + measured palms",
        "labels": "MuJoCo-verified physical hand-object contact sites",
        "active_palm_site_error": describe(errors_m),
        "per_object_active_palm_site_error": per_object,
        "phase_accuracy": phase_accuracy,
        "active_arm_accuracy": active_arm_accuracy,
        "offline_eligibility_threshold": {
            "mean_mm_max": 10.0,
            "p95_mm_max": 20.0,
            "max_mm_max": 30.0,
            "note": "Passing is necessary before MuJoCo contact collection, but does not prove a grasp.",
        },
    }
    metric = result["active_palm_site_error"]
    result["eligible_for_strict_mujoco_contact_gate"] = bool(
        metric["mean_mm"] <= 10.0
        and metric["p95_mm"] <= 20.0
        and metric["max_mm"] <= 30.0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
