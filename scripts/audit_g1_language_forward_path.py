#!/usr/bin/env python3
"""Audit language conditioning between SmolVLA training and inference paths."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from evaluate_g1_language_smolvla_heldout import load_policy, make_dataset, seeded_noise


TARGETS = ("red_triangle", "yellow_rod", "green_cube")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--train-repo-id", default="local/g1_language_paired_train")
    parser.add_argument("--val-repo-id", default="local/g1_language_paired_val")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((left.float() - right.float()) ** 2)))


def pairwise(values: list[torch.Tensor]) -> list[float]:
    return [rmse(values[left], values[right]) for left, right in ((0, 1), (0, 2), (1, 2))]


def batchify(item: dict, task: str, device: torch.device) -> dict:
    batch = {
        key: value.unsqueeze(0).to(device)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in item.items()
    }
    batch["task"] = [task]
    return batch


def finite_or_fail(report: dict) -> None:
    def walk(value, path="report"):
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"Non-finite value at {path}: {value}")

    walk(report)


def run_audit(args: argparse.Namespace) -> dict:
    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    train_meta = LeRobotDatasetMetadata(args.train_repo_id, root=args.train_root)
    dataset = make_dataset(args, args.checkpoint)
    scene_count = dataset.num_episodes // len(TARGETS)
    if not 0 <= args.scene_index < scene_count:
        raise ValueError(f"scene-index must be in [0, {scene_count})")

    episode_indices = [args.scene_index * len(TARGETS) + index for index in range(3)]
    starts = [int(dataset.episode_data_index["from"][index]) for index in episode_indices]
    items = [dataset[start + args.frame_index] for start in starts]
    target_indices = [int(item["complementary_info.target_object_index"]) for item in items]
    if target_indices != list(range(len(TARGETS))):
        raise ValueError(f"Target order is {target_indices}, expected [0, 1, 2]")
    tasks = [item["task"] for item in items]
    policy = load_policy(args.checkpoint, train_meta, device_name)
    policy.eval()

    action_outputs: list[torch.Tensor] = []
    action_inputs: list[torch.Tensor] = []

    def capture_input(_module, inputs):
        action_inputs.append(inputs[0].detach().float().cpu())

    def capture_output(_module, _inputs, output):
        action_outputs.append(output.detach().float().cpu())

    input_handle = policy.model.action_out_proj.register_forward_pre_hook(capture_input)
    output_handle = policy.model.action_out_proj.register_forward_hook(capture_output)
    try:
        noise = seeded_noise(policy, 1, args.seed, device_name)
        time = torch.full((1,), args.time, dtype=torch.float32, device=device)
        conditions = []
        posteriors = []
        train_losses = []
        train_velocity = []
        with torch.no_grad():
            for item, task in zip(items, tasks, strict=True):
                batch = batchify(item, task, device)
                action_outputs.clear()
                action_inputs.clear()
                objective, output = policy.forward(batch, noise=noise, time=time)
                condition = policy.model._language_action_condition.detach().float().cpu()[0]
                posterior = policy.model._language_action_target_logits.softmax(-1).detach().float().cpu()[0]
                conditions.append(condition)
                posteriors.append(posterior)
                train_losses.append(float(objective))
                if not action_outputs:
                    raise RuntimeError("action_out_proj hook did not run during training forward")
                train_velocity.append(action_outputs[-1][0])

        training_pairwise_velocity = pairwise(train_velocity)
        training_pairwise_condition = pairwise(conditions)
        training_pairwise_posterior = pairwise(posteriors)

        # Verify the exact t=1 training output against the first inference denoise step.
        consistency_batch = batchify(items[0], tasks[0], device)
        with torch.no_grad():
            action_outputs.clear()
            action_inputs.clear()
            policy.forward(consistency_batch, noise=noise, time=torch.ones((1,), device=device))
            if not action_outputs:
                raise RuntimeError("action_out_proj hook did not run at t=1")
            t1_training_velocity = action_outputs[-1].clone()
            action_outputs.clear()
            prediction = policy.predict_action_chunk(
                {key: value for key, value in consistency_batch.items() if key.startswith("observation.") or key == "task"},
                noise=noise,
            )
            if not action_outputs:
                raise RuntimeError("action_out_proj hook did not run during inference")
            t1_inference_velocity = action_outputs[0]
        t1_consistency_rmse = rmse(t1_training_velocity, t1_inference_velocity)

        # A separate backward pass audits gradient connectivity without changing weights.
        policy.zero_grad(set_to_none=True)
        gradient_batch = batchify(items[0], tasks[0], device)
        objective, _ = policy.forward(gradient_batch, noise=noise, time=time)
        objective.backward()
        adapter_gradients = {}
        for name, parameter in policy.model.language_action_adapter.named_parameters():
            adapter_gradients[name] = None if parameter.grad is None else float(parameter.grad.detach().float().norm())
        policy.zero_grad(set_to_none=True)
    finally:
        input_handle.remove()
        output_handle.remove()

    report = {
        "experiment": "G1LANG-32A-forward-path-audit",
        "checkpoint": str(args.checkpoint),
        "device": device_name,
        "scene_index": args.scene_index,
        "frame_index": args.frame_index,
        "tasks": tasks,
        "target_order": target_indices,
        "flow_noise_seed": args.seed,
        "flow_time": args.time,
        "training_objective_per_target": train_losses,
        "pairwise_condition_rmse": training_pairwise_condition,
        "pairwise_target_posterior_rmse": training_pairwise_posterior,
        "pairwise_final_flow_velocity_rmse": training_pairwise_velocity,
        "t1_training_vs_first_inference_velocity_rmse": t1_consistency_rmse,
        "first_inference_action_chunk_shape": list(prediction.shape),
        "adapter_gradient_norms": adapter_gradients,
        "invariants": {
            "target_posterior_is_distinct": max(training_pairwise_posterior) > 1e-4,
            "final_flow_velocity_is_distinct": max(training_pairwise_velocity) > 1e-4,
            "training_inference_t1_consistent": t1_consistency_rmse <= 1e-5,
            "all_adapter_gradients_finite": all(
                value is None or math.isfinite(value) for value in adapter_gradients.values()
            ),
        },
        "reference": "Exact scene triplet uses one frame/image/state; only language task changes.",
    }
    finite_or_fail(report)
    return report


def main() -> None:
    args = parse_args()
    report = run_audit(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
