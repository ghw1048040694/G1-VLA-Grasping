#!/usr/bin/env python3
"""Train and independently calibrate the policy-intervention world model."""

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

from g1_policy_intervention_world_model import PolicyInterventionWorldEnsemble, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=20260747)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class InterventionDataset(Dataset):
    def __init__(self, cache: dict) -> None:
        self.cache = cache
        self.rows = len(cache["decision_id"])

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.cache.items() if isinstance(value, torch.Tensor)}


def normalizers(cache: dict) -> dict[str, torch.Tensor]:
    basic = ("state", "velocity", "objects", "sites", "action_delta")
    output = {}
    for name in basic:
        value = cache[name].float()
        output[f"{name}_mean"] = value.mean(0)
        output[f"{name}_std"] = value.std(0).clamp_min(1e-3)
    output["site_delta_std"] = (cache["next_sites"].float() - cache["sites"].float()).std(0).clamp_min(1e-3)
    output["object_delta_std"] = (cache["next_objects_true"].float() - cache["objects"].float()).std(0).clamp_min(2e-3)
    return output


def z(value: torch.Tensor, stats: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    return (value - stats[f"{name}_mean"]) / stats[f"{name}_std"]


def visible_distance(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Current distance from deployment-visible VLA pose and measured sites."""
    count = len(batch["target"])
    objects = batch["objects"].float().reshape(count, 3, 3)
    sites = batch["sites"].float().reshape(count, 2, -1, 3)
    rows = torch.arange(count, device=objects.device)
    active = sites[rows, batch["arm"].long()]
    target = objects[rows, batch["target"].long()]
    return torch.linalg.vector_norm(active - target[:, None], dim=-1).min(-1).values[:, None]


def centered_correction(
    correction: torch.Tensor, decision_id: torch.Tensor
) -> torch.Tensor:
    """Keep every decision anchored to its directly observed geometry.

    Candidate actions share the same real observation.  Centering therefore
    removes a scene-wide distance bias while retaining the action-dependent
    ordering that MPC needs to choose a candidate.
    """
    ids = decision_id.long()
    unique, inverse = torch.unique(ids, sorted=False, return_inverse=True)
    totals = torch.zeros(len(unique), dtype=correction.dtype, device=correction.device)
    counts = torch.zeros(len(unique), dtype=correction.dtype, device=correction.device)
    totals.scatter_add_(0, inverse, correction)
    counts.scatter_add_(0, inverse, torch.ones_like(correction))
    return correction - totals[inverse] / counts[inverse].clamp_min(1.0)


def model_outputs(model, batch: dict[str, torch.Tensor], stats: dict[str, torch.Tensor]):
    return model(
        z(batch["state"], stats, "state"),
        z(batch["velocity"], stats, "velocity"),
        z(batch["objects"], stats, "objects"),
        z(batch["sites"], stats, "sites"),
        batch["tactile"].reshape(-1, 1).float(),
        z(batch["action_delta"], stats, "action_delta"),
        F.one_hot(batch["target"].long(), 3).float(),
        F.one_hot(batch["arm"].long(), 2).float(),
        visible_distance(batch),
    )


def predicted_distance(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return an anchored distance plus bounded successor geometry in metres."""
    next_sites = batch["sites"].float() + output["site_delta_m"]
    next_objects = batch["objects"].float() + output["object_delta_m"]
    correction = centered_correction(
        output["distance_correction_m"].squeeze(-1), batch["decision_id"]
    )
    distance = visible_distance(batch).squeeze(-1) + correction
    return distance.clamp_min(0.0), next_sites, next_objects


def member_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    stats: dict[str, torch.Tensor],
) -> torch.Tensor:
    site_target = batch["next_sites"].float() - batch["sites"].float()
    object_target = batch["next_objects_true"].float() - batch["objects"].float()
    site = F.smooth_l1_loss(
        output["site_delta_m"] / stats["site_delta_std"],
        site_target / stats["site_delta_std"],
        beta=1.0,
        reduction="none",
    ).mean(-1)
    objects = F.smooth_l1_loss(
        output["object_delta_m"] / stats["object_delta_std"],
        object_target / stats["object_delta_std"],
        beta=1.0,
        reduction="none",
    ).mean(-1)
    distance, _, _ = predicted_distance(output, batch)
    distance_loss = F.smooth_l1_loss(
        distance,
        batch["next_target_distance_m"].float(),
        beta=0.006,
        reduction="none",
    )
    positive = batch["next_contact"].float().mean()
    pos_weight = ((1.0 - positive) / positive.clamp_min(0.01)).clamp(1.0, 25.0)
    contact = F.binary_cross_entropy_with_logits(
        output["contact_logit"].squeeze(-1),
        batch["next_contact"].float(),
        pos_weight=pos_weight,
        reduction="none",
    )
    # Distance is the safety-critical objective. Geometry losses keep that
    # scalar grounded in an independently inspectable successor prediction.
    return site + 0.30 * objects + 35.0 * distance_loss + 0.10 * contact


def percentile(values: np.ndarray, point: float) -> float:
    return float(np.percentile(values, point)) if len(values) else float("nan")


@torch.no_grad()
def evaluate(model, loader, stats, device: torch.device) -> dict:
    model.eval()
    errors = []
    predicted_values = []
    truth_values = []
    site_square = 0.0
    site_count = 0
    contact_ok = contact_total = 0
    decision_ids = []
    for raw in loader:
        batch = {key: value.to(device) for key, value in raw.items()}
        outputs = model_outputs(model, batch, stats)
        distances = torch.stack([predicted_distance(item, batch)[0] for item in outputs])
        mean_distance = distances.mean(0)
        mean_sites = torch.stack([predicted_distance(item, batch)[1] for item in outputs]).mean(0)
        errors.append((mean_distance - batch["next_target_distance_m"]).abs().cpu())
        predicted_values.append(mean_distance.cpu())
        truth_values.append(batch["next_target_distance_m"].cpu())
        site_square += float((mean_sites - batch["next_sites"]).square().sum())
        site_count += mean_sites.numel()
        contact_logits = torch.stack([item["contact_logit"].squeeze(-1) for item in outputs]).mean(0)
        contact_ok += int(((contact_logits.sigmoid() >= 0.5) == (batch["next_contact"] >= 0.5)).sum())
        contact_total += len(contact_logits)
        decision_ids.append(batch["decision_id"].cpu())
    error = torch.cat(errors).numpy()
    prediction = torch.cat(predicted_values).numpy()
    truth = torch.cat(truth_values).numpy()
    decision = torch.cat(decision_ids).numpy()
    regrets = []
    top1 = []
    for item in np.unique(decision):
        rows = decision == item
        if rows.sum() < 2:
            continue
        selected = np.flatnonzero(rows)[np.argmin(prediction[rows])]
        oracle = np.flatnonzero(rows)[np.argmin(truth[rows])]
        regrets.append(float(truth[selected] - truth[oracle]))
        top1.append(int(selected == oracle))
    large = truth >= 0.20
    report = {
        "distance_error": {
            "count": int(len(error)),
            "mean_mm": 1e3 * float(error.mean()),
            "median_mm": 1e3 * percentile(error, 50),
            "p95_mm": 1e3 * percentile(error, 95),
            "max_mm": 1e3 * float(error.max()),
        },
        "site_successor_rmse_mm": 1e3 * math.sqrt(site_square / max(site_count, 1)),
        "contact_accuracy": contact_ok / max(contact_total, 1),
        "candidate_ranking": {
            "decisions": len(regrets),
            "top1_exact_rate": float(np.mean(top1)) if top1 else 0.0,
            "mean_regret_mm": 1e3 * float(np.mean(regrets)) if regrets else float("nan"),
            "p95_regret_mm": 1e3 * percentile(np.asarray(regrets), 95) if regrets else float("nan"),
        },
        "far_state_anti_hallucination": {
            "count_actual_distance_ge_200mm": int(large.sum()),
            "min_predicted_distance_mm": 1e3 * float(prediction[large].min()) if large.any() else float("nan"),
            "mean_predicted_distance_mm": 1e3 * float(prediction[large].mean()) if large.any() else float("nan"),
            "mean_actual_distance_mm": 1e3 * float(truth[large].mean()) if large.any() else float("nan"),
        },
    }
    return report


def passes(report: dict) -> bool:
    distance = report["distance_error"]
    rank = report["candidate_ranking"]
    far = report["far_state_anti_hallucination"]
    return bool(
        distance["mean_mm"] <= 20.0
        and distance["p95_mm"] <= 45.0
        and distance["max_mm"] <= 100.0
        and rank["mean_regret_mm"] <= 8.0
        and rank["p95_regret_mm"] <= 25.0
        and (far["count_actual_distance_ge_200mm"] == 0 or far["min_predicted_distance_mm"] >= 100.0)
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
    stats_cpu = normalizers(train_cache)
    stats = {key: value.to(device) for key, value in stats_cpu.items()}
    train_loader = DataLoader(
        InterventionDataset(train_cache),
        batch_size=args.batch_size,
        # Candidate rows are contiguous per decision. Preserving that grouping
        # lets the loss learn action-relative distances without reintroducing
        # a global visual-geometry bias.
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_loader = DataLoader(
        InterventionDataset(validation_cache),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model = PolicyInterventionWorldEnsemble(args.members, int(train_cache["sites"].shape[-1]), args.hidden_dim).to(device)
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
            # Bootstrap masks give the ensemble a meaningful disagreement on
            # candidate states without multiplying stored intervention data.
            losses = []
            for output in outputs:
                per_row = member_loss(output, batch, stats)
                mask = torch.rand(len(per_row), generator=generator, device=device) < 0.82
                losses.append(per_row[mask].mean() if mask.any() else per_row.mean())
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(batch["target"])
            rows += len(batch["target"])
        report = evaluate(model, validation_loader, stats, device)
        report.update(epoch=epoch, train_loss=total / max(rows, 1))
        history.append(report)
        print("POLICY_INTERVENTION_WORLD_EPOCH=" + json.dumps(report), flush=True)
        score = report["distance_error"]["mean_mm"] + 0.25 * report["distance_error"]["p95_mm"] + 0.5 * report["candidate_ranking"]["mean_regret_mm"]
        if score < best:
            best = score
            save_checkpoint(args.output_dir / "best.pt", model, stats_cpu)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No world-model checkpoint was selected")
    model.load_state_dict(best_state)
    final = evaluate(model, validation_loader, stats, device)
    calibration = {
        "experiment": "G1LANG-policy-visited-calibrated-action-token-world-model",
        "cache_dir": str(args.cache_dir),
        "heldout_split": "independent randomized scenes, never used in optimization",
        "input_contract": "VLA RGB-derived object pose + measured state31/qvel31/finger sites/tactile + executed candidate action + language target + VLA arm",
        "labels": "physical MuJoCo successor geometry and target distance",
        "offline_eligibility_threshold": {
            "mean_mm_max": 20.0,
            "p95_mm_max": 45.0,
            "max_mm_max": 100.0,
            "candidate_mean_regret_mm_max": 8.0,
            "candidate_p95_regret_mm_max": 25.0,
            "far_state_min_predicted_distance_mm_min": 100.0,
        },
        "final": final,
        "eligible_for_closed_loop_selector": passes(final),
    }
    (args.output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    summary = {
        "experiment": calibration["experiment"],
        "cache_dir": str(args.cache_dir),
        "members": args.members,
        "hidden_dim": args.hidden_dim,
        "best_validation_score": best,
        "history": history,
        "final": final,
        "eligible_for_closed_loop_selector": calibration["eligible_for_closed_loop_selector"],
        "privileged_positions_at_inference": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(calibration, indent=2))
    if not calibration["eligible_for_closed_loop_selector"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
