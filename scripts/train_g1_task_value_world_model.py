#!/usr/bin/env python3
"""Train the full-chunk task-value world model for G1 candidate selection."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from g1_task_value_world_model import (
    G1TaskValueWorldEnsemble,
    physical_outputs,
    save_checkpoint,
    score_task_value_candidates,
)


REQUIRED_KEYS = (
    "state",
    "velocity",
    "objects",
    "sites",
    "tactile",
    "action_chunk",
    "target",
    "arm",
    "decision_id",
    "horizon_final_distance_m",
    "horizon_min_distance_m",
    "horizon_contact",
    "horizon_target_contact_fraction",
    "horizon_target_contact_last",
    "horizon_target_lift_m",
    "horizon_hand_object_comotion_m",
    "horizon_target_bin_xy_distance_m",
    "horizon_stable_grasp",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-no-stable-positive",
        action="store_true",
        help="Train for diagnostics even if stable-grasp positives are absent.",
    )
    return parser.parse_args()


class BranchDataset(Dataset):
    def __init__(self, cache: dict[str, torch.Tensor]) -> None:
        self.cache = cache
        self.rows = int(cache["decision_id"].numel())

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: value[index]
            for key, value in self.cache.items()
            if isinstance(value, torch.Tensor)
        }


def require_cache_contract(cache: dict, split: str) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in cache]
    if missing:
        raise KeyError(f"{split}.pt is missing task-value labels: {missing}")
    if cache["action_chunk"].ndim != 2 or cache["action_chunk"].shape[-1] % 31:
        raise ValueError(f"{split}.pt action_chunk must be flattened [N, H*31]")


def normalizers(cache: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    fields = ("state", "velocity", "objects", "sites", "action_chunk")
    stats = {}
    for name in fields:
        value = cache[name].float()
        stats[f"{name}_mean"] = value.mean(0)
        stats[f"{name}_std"] = value.std(0).clamp_min(1e-3)
    return stats


def z(value: torch.Tensor, stats: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return (value - stats[f"{name}_mean"]) / stats[f"{name}_std"]


def visible_distance(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    count = int(batch["target"].numel())
    objects = batch["objects"].float().reshape(count, 3, 3)
    sites = batch["sites"].float().reshape(count, 2, -1, 3)
    rows = torch.arange(count, device=objects.device)
    active = sites[rows, batch["arm"].long()]
    target = objects[rows, batch["target"].long()]
    return torch.linalg.vector_norm(active - target[:, None], dim=-1).min(-1).values[:, None]


def model_outputs(model, batch: dict[str, torch.Tensor], stats: dict[str, torch.Tensor]):
    return model(
        z(batch["state"], stats, "state"),
        z(batch["velocity"], stats, "velocity"),
        z(batch["objects"], stats, "objects"),
        z(batch["sites"], stats, "sites"),
        batch["tactile"].reshape(-1, 1).float(),
        z(batch["action_chunk"], stats, "action_chunk"),
        F.one_hot(batch["target"].long(), 3).float(),
        F.one_hot(batch["arm"].long(), 2).float(),
        visible_distance(batch),
    )


def label_distribution(cache: dict[str, torch.Tensor]) -> dict:
    rows = int(cache["decision_id"].numel())
    stable = cache["horizon_stable_grasp"].float()
    contact = cache["horizon_contact"].float()
    sustained = cache["horizon_target_contact_fraction"].float() >= 0.75
    lift = cache["horizon_target_lift_m"].float()
    by_target = {}
    for target in range(3):
        mask = cache["target"].long() == target
        by_target[str(target)] = {
            "rows": int(mask.sum()),
            "stable_grasp_positive": int(stable[mask].sum()) if mask.any() else 0,
            "contact_positive": int(contact[mask].sum()) if mask.any() else 0,
            "lift_ge_15mm": int((lift[mask] >= 0.015).sum()) if mask.any() else 0,
        }
    return {
        "rows": rows,
        "stable_grasp_positive": int(stable.sum()),
        "stable_grasp_rate": float(stable.mean()) if rows else 0.0,
        "contact_positive": int(contact.sum()),
        "sustained_contact_positive": int(sustained.sum()),
        "lift_ge_15mm": int((lift >= 0.015).sum()),
        "mean_lift_mm": 1e3 * float(lift.mean()) if rows else 0.0,
        "max_lift_mm": 1e3 * float(lift.max()) if rows else 0.0,
        "by_target": by_target,
    }


def bce_with_distribution(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    positive = labels.mean()
    pos_weight = ((1.0 - positive) / positive.clamp_min(0.01)).clamp(1.0, 50.0)
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def member_loss(raw: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    physical = physical_outputs(raw)
    final_distance = F.smooth_l1_loss(
        physical["final_distance_m"],
        batch["horizon_final_distance_m"].float(),
        beta=0.010,
    )
    min_distance = F.smooth_l1_loss(
        physical["min_distance_m"],
        batch["horizon_min_distance_m"].float(),
        beta=0.010,
    )
    contact = bce_with_distribution(raw["contact_logit"], batch["horizon_contact"])
    contact_last = bce_with_distribution(
        raw["contact_last_logit"], batch["horizon_target_contact_last"]
    )
    stable = bce_with_distribution(raw["stable_grasp_logit"], batch["horizon_stable_grasp"])
    contact_fraction = F.mse_loss(
        physical["contact_fraction"],
        batch["horizon_target_contact_fraction"].float(),
    )
    lift = F.smooth_l1_loss(
        physical["lift_m"],
        batch["horizon_target_lift_m"].float(),
        beta=0.005,
    )
    comotion = F.smooth_l1_loss(
        physical["hand_object_comotion_m"],
        batch["horizon_hand_object_comotion_m"].float(),
        beta=0.015,
    )
    bin_xy = F.smooth_l1_loss(
        physical["bin_xy_distance_m"],
        batch["horizon_target_bin_xy_distance_m"].float(),
        beta=0.030,
    )
    return (
        18.0 * final_distance
        + 22.0 * min_distance
        + 0.25 * contact
        + 0.25 * contact_last
        + 0.40 * stable
        + 0.50 * contact_fraction
        + 18.0 * lift
        + 4.0 * comotion
        + 2.0 * bin_xy
    )


def percentile(values: np.ndarray, point: float) -> float:
    return float(np.percentile(values, point)) if len(values) else float("nan")


@torch.no_grad()
def evaluate(model, loader, stats, device: torch.device) -> dict:
    model.eval()
    final_errors = []
    min_errors = []
    lift_errors = []
    stable_truth = []
    stable_prob = []
    contact_truth = []
    contact_prob = []
    decision_ids = []
    costs = []
    truth_stable = []
    truth_lift = []
    for raw in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
        outputs = model_outputs(model, batch, stats)
        physical = [physical_outputs(item) for item in outputs]
        mean = {
            key: torch.stack([item[key] for item in physical]).mean(0)
            for key in physical[0]
        }
        cost, _ = score_task_value_candidates(outputs, batch["action_chunk"])
        final_errors.append((mean["final_distance_m"] - batch["horizon_final_distance_m"]).abs().cpu())
        min_errors.append((mean["min_distance_m"] - batch["horizon_min_distance_m"]).abs().cpu())
        lift_errors.append((mean["lift_m"] - batch["horizon_target_lift_m"]).abs().cpu())
        stable_truth.append(batch["horizon_stable_grasp"].float().cpu())
        stable_prob.append(mean["stable_grasp_prob"].cpu())
        contact_truth.append(batch["horizon_contact"].float().cpu())
        contact_prob.append(mean["contact_prob"].cpu())
        decision_ids.append(batch["decision_id"].cpu())
        costs.append(cost.cpu())
        truth_stable.append(batch["horizon_stable_grasp"].float().cpu())
        truth_lift.append(batch["horizon_target_lift_m"].float().cpu())
    final_error = torch.cat(final_errors).numpy()
    min_error = torch.cat(min_errors).numpy()
    lift_error = torch.cat(lift_errors).numpy()
    stable_y = torch.cat(stable_truth).numpy()
    stable_p = torch.cat(stable_prob).numpy()
    contact_y = torch.cat(contact_truth).numpy()
    contact_p = torch.cat(contact_prob).numpy()
    decision = torch.cat(decision_ids).numpy()
    cost_value = torch.cat(costs).numpy()
    stable_value = torch.cat(truth_stable).numpy()
    lift_value = torch.cat(truth_lift).numpy()
    selected_stable = []
    oracle_stable = []
    lift_regrets = []
    for decision_id in np.unique(decision):
        rows = decision == decision_id
        if rows.sum() < 2:
            continue
        row_indices = np.flatnonzero(rows)
        selected = row_indices[np.argmin(cost_value[rows])]
        oracle = row_indices[np.lexsort((-lift_value[rows], -stable_value[rows]))[0]]
        selected_stable.append(float(stable_value[selected]))
        oracle_stable.append(float(stable_value[oracle]))
        lift_regrets.append(float(lift_value[oracle] - lift_value[selected]))
    stable_pred = stable_p >= 0.5
    contact_pred = contact_p >= 0.5
    return {
        "final_distance_error": {
            "count": int(len(final_error)),
            "mean_mm": 1e3 * float(final_error.mean()),
            "p95_mm": 1e3 * percentile(final_error, 95),
            "max_mm": 1e3 * float(final_error.max()),
        },
        "min_distance_error": {
            "mean_mm": 1e3 * float(min_error.mean()),
            "p95_mm": 1e3 * percentile(min_error, 95),
        },
        "lift_error": {
            "mean_mm": 1e3 * float(lift_error.mean()),
            "p95_mm": 1e3 * percentile(lift_error, 95),
        },
        "contact_accuracy": float((contact_pred == (contact_y >= 0.5)).mean()),
        "stable_grasp_accuracy": float((stable_pred == (stable_y >= 0.5)).mean()),
        "stable_grasp_positive_rate": float(stable_y.mean()) if len(stable_y) else 0.0,
        "candidate_task_value_ranking": {
            "decisions": int(len(selected_stable)),
            "selected_stable_rate": float(np.mean(selected_stable)) if selected_stable else 0.0,
            "oracle_stable_rate": float(np.mean(oracle_stable)) if oracle_stable else 0.0,
            "mean_lift_regret_mm": 1e3 * float(np.mean(lift_regrets)) if lift_regrets else float("nan"),
            "p95_lift_regret_mm": 1e3 * percentile(np.asarray(lift_regrets), 95) if lift_regrets else float("nan"),
        },
    }


def passes(report: dict, validation_distribution: dict) -> bool:
    ranking = report["candidate_task_value_ranking"]
    return bool(
        validation_distribution["stable_grasp_positive"] > 0
        and ranking["oracle_stable_rate"] >= 0.25
        and ranking["selected_stable_rate"] >= 0.20
        and report["final_distance_error"]["mean_mm"] <= 35.0
        and report["lift_error"]["mean_mm"] <= 12.0
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed training: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_cache = torch.load(args.cache_dir / "train.pt", map_location="cpu", weights_only=False)
    validation_cache = torch.load(args.cache_dir / "validation.pt", map_location="cpu", weights_only=False)
    require_cache_contract(train_cache, "train")
    require_cache_contract(validation_cache, "validation")
    train_distribution = label_distribution(train_cache)
    validation_distribution = label_distribution(validation_cache)
    data_gate = {
        "train": train_distribution,
        "validation": validation_distribution,
        "passed": bool(
            train_distribution["stable_grasp_positive"] > 0
            and validation_distribution["stable_grasp_positive"] > 0
        ),
        "reason": "requires real stable-grasp positives before claiming a trainable task-value selector",
    }
    (args.output_dir / "data_gate.json").write_text(json.dumps(data_gate, indent=2) + "\n")
    if not data_gate["passed"] and not args.allow_no_stable_positive:
        summary = {
            "experiment": "G1LANG-task-value-world-model-data-gate",
            "cache_dir": str(args.cache_dir),
            "data_gate": data_gate,
            "trained": False,
            "eligible_for_closed_loop_selector": False,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        raise SystemExit(2)
    stats_cpu = normalizers(train_cache)
    stats = {key: value.to(device) for key, value in stats_cpu.items()}
    train_loader = DataLoader(
        BranchDataset(train_cache),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        BranchDataset(validation_cache),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    action_chunk = int(train_cache["action_chunk"].shape[-1] // 31)
    model = G1TaskValueWorldEnsemble(
        args.members,
        int(train_cache["sites"].shape[-1]),
        31,
        action_chunk,
        args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    best = math.inf
    best_state = None
    history = []
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        rows = 0
        for raw in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
            outputs = model_outputs(model, batch, stats)
            losses = []
            for output in outputs:
                mask = torch.rand(batch["target"].shape[0], generator=generator, device=device) < 0.84
                if not bool(mask.any()):
                    mask[torch.randint(batch["target"].shape[0], (1,), generator=generator, device=device)] = True
                masked_batch = {key: value[mask] for key, value in batch.items()}
                masked_output = {
                    key: value[mask]
                    if isinstance(value, torch.Tensor) and value.shape[:1] == batch["target"].shape[:1]
                    else value
                    for key, value in output.items()
                }
                losses.append(member_loss(masked_output, masked_batch))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * int(batch["target"].shape[0])
            rows += int(batch["target"].shape[0])
        report = evaluate(model, validation_loader, stats, device)
        report.update(epoch=epoch, train_loss=total / max(rows, 1))
        history.append(report)
        print("G1_TASK_VALUE_WORLD_EPOCH=" + json.dumps(report), flush=True)
        ranking = report["candidate_task_value_ranking"]
        score = (
            report["final_distance_error"]["mean_mm"]
            + 0.5 * report["lift_error"]["mean_mm"]
            + 40.0 * max(0.0, ranking["oracle_stable_rate"] - ranking["selected_stable_rate"])
        )
        if score < best:
            best = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            save_checkpoint(args.output_dir / "best.pt", model, stats_cpu, data_gate)
    if best_state is None:
        raise RuntimeError("No task-value world-model checkpoint was selected")
    model.load_state_dict(best_state)
    final = evaluate(model, validation_loader, stats, device)
    eligible = passes(final, validation_distribution)
    summary = {
        "experiment": "G1LANG-full-chunk-task-value-world-model",
        "cache_dir": str(args.cache_dir),
        "members": args.members,
        "hidden_dim": args.hidden_dim,
        "action_chunk": action_chunk,
        "data_gate": data_gate,
        "best_validation_score": best,
        "history": history,
        "final": final,
        "eligible_for_closed_loop_selector": eligible,
        "privileged_positions_at_inference": False,
        "note": "closed-loop use still requires physically feasible VLA candidate chunks",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
