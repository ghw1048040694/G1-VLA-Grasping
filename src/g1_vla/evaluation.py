"""Shared event metrics. Simulator truth is confined to this evaluator."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def evaluate_trace(samples: Sequence[Mapping], *, lift_height_m: float = 0.1, hold_duration_s: float = 0.5, task: str = "grasp_lift") -> dict:
    if not samples:
        raise ValueError("A complete, non-empty trace is required")
    if task not in ("grasp_lift", "pick_place"):
        raise ValueError("Unknown task")
    if not math.isfinite(lift_height_m) or not math.isfinite(hold_duration_s) or min(lift_height_m, hold_duration_s) <= 0:
        raise ValueError("Event thresholds must be finite and positive")
    required = {"time_s", "target_lift_m", "target_contact", "assisted", "numerically_stable", "joint_safe"}
    previous_time = None
    grasp_start = placed_start = None
    best_hold = best_placed_hold = 0.0
    contact = False
    assisted = unstable = unsafe = wrong_object = False
    max_lift = float("-inf")
    for row in samples:
        if not required <= set(row):
            raise ValueError("Trace is missing explicit physical evidence: " + ", ".join(sorted(required - set(row))))
        timestamp, height = row["time_s"], row["target_lift_m"]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (timestamp, height)):
            raise ValueError("Trace time/height must be finite numbers")
        if previous_time is not None and timestamp <= previous_time:
            raise ValueError("Trace timestamps must increase strictly")
        previous_time = timestamp
        for field in ("target_contact", "assisted", "numerically_stable", "joint_safe"):
            if type(row[field]) is not bool:
                raise ValueError(field + " must be an explicit boolean")
        contact |= row["target_contact"]
        assisted |= row["assisted"]
        unstable |= not row["numerically_stable"]
        unsafe |= not row["joint_safe"]
        wrong_object |= row.get("wrong_object_grasp", False) is True
        max_lift = max(max_lift, height)
        holding = row["target_contact"] and height >= lift_height_m
        grasp_start = (timestamp if grasp_start is None else grasp_start) if holding else None
        if grasp_start is not None:
            best_hold = max(best_hold, timestamp - grasp_start)
        if task == "pick_place":
            for field in ("target_in_box", "released", "placement_stable"):
                if type(row.get(field)) is not bool:
                    raise ValueError("Pick/place trace requires boolean " + field)
            placed = row["target_in_box"] and row["released"] and row["placement_stable"]
            placed_start = (timestamp if placed_start is None else placed_start) if placed else None
            if placed_start is not None:
                best_placed_hold = max(best_placed_hold, timestamp - placed_start)
    physical_grasp = contact and max_lift >= lift_height_m and best_hold >= hold_duration_s
    physical_success = physical_grasp and (task == "grasp_lift" or best_placed_hold >= hold_duration_s)
    strict = physical_success and not any((assisted, unstable, unsafe, wrong_object))
    return {"schema_version": 1, "task": task, "samples": len(samples), "contact": contact,
            "max_lift_m": max_lift, "hold_seconds": best_hold, "placed_hold_seconds": best_placed_hold,
            "assisted": assisted, "numerically_stable": not unstable, "joint_safe": not unsafe,
            "wrong_object_grasp": wrong_object, "physical_grasp": physical_grasp,
            "physical_success": physical_success, "strict_success": strict}


def summarize_results(results: Sequence[Mapping]) -> dict:
    if not results:
        raise ValueError("An empty suite cannot establish a success rate")
    if any(type(row.get("strict_success")) is not bool for row in results):
        raise ValueError("Results must contain explicit strict-success decisions")
    return {"episodes": len(results), "strict_successes": sum(row["strict_success"] for row in results),
            "strict_success_rate": sum(row["strict_success"] for row in results) / len(results)}
