#!/usr/bin/env python3
"""Summarize a 200-iteration G1 motion-tracking run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = {
    "reward": "Train/mean_reward",
    "episode_length": "Train/mean_episode_length",
    "upper_body_error_m": "Env/upper_body_diff_norm",
    "lower_body_error_m": "Env/lower_body_diff_norm",
    "vr_3point_error_m": "Env/vr_3point_diff_norm",
    "joint_error_norm_rad": "Env/joint_pos_diff_norm",
    "termination_threshold_m": "Env/terminate_when_motion_far_threshold",
    "steps_per_second": "Perf/total_fps",
}


def trailing_mean(values: list[float], end: int, width: int = 20) -> float:
    return mean(values[max(0, end - width) : end])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-timesteps", type=int, required=True)
    parser.add_argument("--selected-checkpoint", type=int, required=True)
    parser.add_argument("--selection-reason", required=True)
    args = parser.parse_args()

    accumulator = EventAccumulator(str(args.event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    values = {
        name: [event.value for event in accumulator.Scalars(tag)]
        for name, tag in TAGS.items()
    }

    lengths = {len(series) for series in values.values()}
    if lengths != {200}:
        raise RuntimeError(f"Expected 200 values for every metric, got {sorted(lengths)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "iteration_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", *TAGS])
        writer.writeheader()
        for iteration in range(200):
            writer.writerow(
                {"iteration": iteration, **{name: series[iteration] for name, series in values.items()}}
            )

    checkpoints = {}
    for checkpoint in (50, 100, 150, 200):
        checkpoints[str(checkpoint)] = {
            name: trailing_mean(series, checkpoint) for name, series in values.items()
        }

    summary = {
        "iterations": 200,
        "total_timesteps": args.total_timesteps,
        "trailing_window_iterations": 20,
        "checkpoint_trailing_means": checkpoints,
        "selected_checkpoint": args.selected_checkpoint,
        "selection_reason": args.selection_reason,
    }
    summary_path = args.output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {metrics_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
