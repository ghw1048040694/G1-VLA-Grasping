#!/usr/bin/env python3
"""Run LeRobot training with fresh optimizer state and finite-update retries."""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.constants import TRAINING_STATE_DIR, TRAINING_STEP
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.utils import has_method


lerobot_train = importlib.import_module("lerobot.scripts.train")
MAX_RETRIES = int(os.environ.get("G1_NONFINITE_MAX_RETRIES", "3"))
MAX_SKIPPED_BATCHES = int(os.environ.get("G1_NONFINITE_MAX_SKIPPED_BATCHES", "25"))
FINITE_UPDATES = 0
SKIPPED_BATCHES = 0


def bounded_flow_inputs(policy: PreTrainedPolicy, batch: Any, device: torch.device):
    """Use bounded deterministic flow inputs for recovery continuation only."""
    batch_size = batch["action"].shape[0]
    shape = (batch_size, policy.config.chunk_size, policy.config.max_action_dim)
    noise = torch.zeros(shape, dtype=torch.float32, device=device)
    time = torch.full((batch_size,), 0.5, dtype=torch.float32, device=device)
    return noise, time


def make_policy_with_stable_dtype(*args, **kwargs):
    """Keep the frozen VLM in BF16 but run trainable expert layers in FP32."""
    policy = _ORIGINAL_MAKE_POLICY(*args, **kwargs)
    if os.environ.get("G1_FORCE_EXPERT_FP32", "0").lower() not in {"1", "true", "yes"}:
        return policy

    model = getattr(policy, "model", None)
    vlm_with_expert = getattr(model, "vlm_with_expert", None)
    if vlm_with_expert is None:
        raise RuntimeError("G1_FORCE_EXPERT_FP32 requires a SmolVLA policy")

    vlm_with_expert.lm_expert.float()
    for module_name in (
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
        "state_proj",
    ):
        module = getattr(model, module_name, None)
        if module is not None:
            module.float()
    logging.warning(
        "G1 stable continuation: frozen VLM remains BF16; trainable expert and "
        "action/state projections use FP32"
    )
    return policy


_ORIGINAL_MAKE_POLICY = lerobot_train.make_policy


def load_step_with_fresh_optimizer(checkpoint_dir, optimizer, scheduler):
    step_path = Path(checkpoint_dir) / TRAINING_STATE_DIR / TRAINING_STEP
    step = int(json.loads(step_path.read_text(encoding="utf-8"))["step"])
    logging.warning(
        "G1 stable continuation: keeping model step %d while resetting optimizer, "
        "scheduler, and RNG state",
        step,
    )
    return step, optimizer, scheduler


def update_policy_with_finite_retry(
    train_metrics,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
):
    global FINITE_UPDATES, SKIPPED_BATCHES
    started = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    last_reason = "unknown"

    for attempt in range(MAX_RETRIES + 1):
        optimizer.zero_grad()
        flow_inputs = None
        if os.environ.get("G1_BOUNDED_FLOW", "0").lower() in {"1", "true", "yes"}:
            flow_inputs = bounded_flow_inputs(policy, batch, device)
        with torch.autocast(device_type=device.type) if use_amp else nullcontext():
            if flow_inputs is None:
                loss, output_dict = policy.forward(batch)
            else:
                loss, output_dict = policy.forward(batch, noise=flow_inputs[0], time=flow_inputs[1])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if not bool(torch.isfinite(loss.detach()).all()):
            last_reason = "non-finite loss"
            logging.warning(
                "Skipping %s before optimizer step (attempt %d/%d)",
                last_reason,
                attempt + 1,
                MAX_RETRIES + 1,
            )
            continue

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            grad_clip_norm,
            error_if_nonfinite=False,
            foreach=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if not bool(torch.isfinite(grad_norm.detach()).all()):
            last_reason = "non-finite gradient norm"
            logging.warning(
                "Skipping %s before optimizer step (attempt %d/%d)",
                last_reason,
                attempt + 1,
                MAX_RETRIES + 1,
            )
            optimizer.zero_grad()
            if grad_scaler.is_enabled():
                grad_scaler.update()
            continue

        with lock if lock is not None else nullcontext():
            grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        if has_method(policy, "update"):
            policy.update()

        train_metrics.loss = loss.item()
        train_metrics.grad_norm = grad_norm.item()
        train_metrics.lr = optimizer.param_groups[0]["lr"]
        train_metrics.update_s = time.perf_counter() - started
        if output_dict is None:
            output_dict = {}
        output_dict["g1_nonfinite_retry_count"] = attempt
        output_dict["g1_nonfinite_batch_skipped"] = False
        FINITE_UPDATES += 1
        return train_metrics, output_dict

    optimizer.zero_grad()
    SKIPPED_BATCHES += 1
    if SKIPPED_BATCHES > MAX_SKIPPED_BATCHES:
        raise FloatingPointError(
            f"Exceeded {MAX_SKIPPED_BATCHES} skipped batches after repeated "
            f"non-finite updates: {last_reason}"
        )
    logging.warning(
        "Skipping the entire batch without an optimizer update (%d/%d skipped batches)",
        SKIPPED_BATCHES,
        MAX_SKIPPED_BATCHES,
    )
    train_metrics.loss = float("nan")
    train_metrics.grad_norm = float("nan")
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - started
    return train_metrics, {
        "g1_nonfinite_retry_count": MAX_RETRIES + 1,
        "g1_nonfinite_batch_skipped": True,
    }


def write_contract() -> None:
    raw_path = os.environ.get("G1_CONTINUATION_CONTRACT_PATH")
    if not raw_path:
        return
    path = Path(raw_path)
    report = {
        "experiment": "G1FINAL-target-specialist-stable-continuation",
        "target": os.environ["G1_CONTINUATION_TARGET"],
        "source_checkpoint": os.environ["G1_CONTINUATION_SOURCE"],
        "source_step": int(os.environ["G1_CONTINUATION_SOURCE_STEP"]),
        "continuation_updates": int(os.environ["G1_CONTINUATION_UPDATES"]),
        "checkpoint_attempt_step": int(os.environ["G1_CONTINUATION_TARGET_STEP"]),
        "finite_optimizer_updates": FINITE_UPDATES,
        "skipped_nonfinite_batches": SKIPPED_BATCHES,
        "effective_cumulative_updates": (
            int(os.environ["G1_CONTINUATION_SOURCE_STEP"]) + FINITE_UPDATES
        ),
        "minimum_required_effective_cumulative_updates": 10000,
        "optimizer_state": "reset_after_non-finite seamless-resume attempt",
        "peak_learning_rate": float(os.environ["G1_CONTINUATION_PEAK_LR"]),
        "scheduler_warmup_steps": int(os.environ["G1_CONTINUATION_WARMUP"]),
        "finite_update_max_retries": MAX_RETRIES,
        "maximum_skipped_nonfinite_batches": MAX_SKIPPED_BATCHES,
        "trainable_dtype": "float32"
        if os.environ.get("G1_FORCE_EXPERT_FP32", "0").lower() in {"1", "true", "yes"}
        else "checkpoint_default",
        "train_state_proj": os.environ.get("G1_TRAIN_STATE_PROJ", "checkpoint_default"),
        "flow_input_mode": "bounded_zero_noise_t05"
        if os.environ.get("G1_BOUNDED_FLOW", "0").lower() in {"1", "true", "yes"}
        else "random_default",
        "parameter_sweep": False,
        "completed": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if MAX_RETRIES < 0 or MAX_SKIPPED_BATCHES < 0:
        raise ValueError("Non-finite retry and skipped-batch limits must be non-negative")
    lerobot_train.load_training_state = load_step_with_fresh_optimizer
    lerobot_train.update_policy = update_policy_with_finite_retry
    lerobot_train.make_policy = make_policy_with_stable_dtype
    lerobot_train.main()
    minimum_updates = int(os.environ.get("G1_MINIMUM_FINITE_UPDATES", "0"))
    if FINITE_UPDATES < minimum_updates:
        raise RuntimeError(
            f"Only {FINITE_UPDATES} finite optimizer updates completed; "
            f"required {minimum_updates}"
        )
    write_contract()


if __name__ == "__main__":
    main()
