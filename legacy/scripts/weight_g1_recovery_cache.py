#!/usr/bin/env python3
"""Increase the training weight of a verified recovery-cache suffix.

The input cache keeps its clean expert prefix followed by unique recovery
samples. This utility repeats only that suffix; validation data is never
duplicated. It is intended for unique, physically audited recovery samples,
not failed rollout labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-in", type=Path, required=True)
    parser.add_argument("--cache-out", type=Path, required=True)
    parser.add_argument("--base-samples", type=int, required=True)
    parser.add_argument(
        "--recovery-repeat",
        type=int,
        required=True,
        help="Total copies of each unique recovery sample in the output cache.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_samples < 1 or args.recovery_repeat < 1:
        raise ValueError("base samples and recovery repeat must be positive")
    source_summary = json.loads((args.cache_in / "summary.json").read_text())
    source = torch.load(args.cache_in / "train.pt", map_location="cpu", weights_only=False)
    total = len(source["state"])
    if not 0 < args.base_samples < total:
        raise ValueError(f"base_samples={args.base_samples} is invalid for {total} rows")
    unique_count = total - args.base_samples
    merged = dict(source)
    weighted_keys = []
    for key, value in source.items():
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or len(value) != total:
            continue
        recovery = value[args.base_samples :]
        repeat_shape = (args.recovery_repeat,) + (1,) * (value.ndim - 1)
        merged[key] = torch.cat((value[: args.base_samples], recovery.repeat(repeat_shape)), dim=0)
        weighted_keys.append(key)
    expected = args.base_samples + unique_count * args.recovery_repeat
    if len(merged["state"]) != expected:
        raise RuntimeError("Weighted state count does not match expectation")
    if any(len(merged[key]) != expected for key in weighted_keys):
        raise RuntimeError("A weighted tensor has inconsistent leading dimension")
    args.cache_out.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.cache_out / "train.pt")
    shutil.copy2(args.cache_in / "val.pt", args.cache_out / "val.pt")
    summary = {
        "experiment": "G1LANG_weighted_verified_recovery_cache",
        "cache_in": str(args.cache_in),
        "source_experiment": source_summary.get("experiment"),
        "base_samples": args.base_samples,
        "unique_recovery_samples": unique_count,
        "recovery_repeat": args.recovery_repeat,
        "weighted_recovery_samples": unique_count * args.recovery_repeat,
        "merged_samples": expected,
        "weighted_tensor_keys": weighted_keys,
        "input_contract": source_summary.get("input_contract"),
        "label_contract": source_summary.get("label_contract"),
        "privileged_positions_at_inference": False,
    }
    (args.cache_out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
