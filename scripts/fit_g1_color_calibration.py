#!/usr/bin/env python3
"""Fit a fixed-camera color-centroid to world-coordinate calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def design(features: np.ndarray, object_index: int) -> np.ndarray:
    z = features[:, 3 * object_index : 3 * object_index + 3]
    x, y, area = z.T
    return np.column_stack((np.ones(len(z)), x, y, area, x * x, y * y, x * y))


def fit(cache: dict, max_frame: int) -> tuple[np.ndarray, dict]:
    frame = cache["frame_index"].numpy()
    selected = frame < max_frame
    features = cache["color_features"].float().numpy()[selected]
    target = cache["object_position"].float().numpy().reshape(-1, 3, 3)[selected]
    coefficients = np.zeros((3, 7, 3), dtype=np.float32)
    errors = []
    for object_index in range(3):
        matrix = design(features, object_index)
        coefficients[object_index] = np.linalg.lstsq(
            matrix, target[:, object_index], rcond=None
        )[0]
        errors.append(np.linalg.norm(matrix @ coefficients[object_index] - target[:, object_index], axis=1))
    error = np.stack(errors, axis=1)
    summary = {
        "samples": int(selected.sum()),
        "max_frame_exclusive": int(max_frame),
        "mean_error_m": float(error.mean()),
        "p95_error_m": float(np.quantile(error, 0.95)),
        "mean_error_by_object_m": error.mean(0).tolist(),
    }
    return coefficients, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frame", type=int, default=20)
    args = parser.parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    coefficients, summary = fit(cache, args.max_frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, coefficients=coefficients)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
