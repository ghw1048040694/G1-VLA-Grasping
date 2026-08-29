#!/usr/bin/env python3
"""Chunk-conditioned task-value world model for G1 action-token candidates.

This model scores a full VLA action chunk, not only the first control step.  It
is meant to answer the selector question that the one-step geometry model
cannot answer: does this candidate look like a physically useful grasp branch?
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


TASK_VALUE_OUTPUTS = (
    "final_distance_m",
    "min_distance_m",
    "contact_logit",
    "contact_last_logit",
    "contact_fraction_logit",
    "lift_m",
    "hand_object_comotion_m",
    "bin_xy_distance_m",
    "stable_grasp_logit",
)


class G1TaskValueWorldMember(nn.Module):
    """One ensemble member that predicts short-horizon task-value labels."""

    def __init__(
        self,
        contact_site_dim: int = 108,
        action_dim: int = 31,
        action_chunk: int = 8,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.contact_site_dim = int(contact_site_dim)
        self.action_dim = int(action_dim)
        self.action_chunk = int(action_chunk)
        self.hidden_dim = int(hidden_dim)
        action_chunk_dim = self.action_dim * self.action_chunk
        # state, velocity, RGB-derived objects, measured sites, tactile, full
        # candidate chunk, target one-hot, arm one-hot, and visible distance.
        input_dim = 31 + 31 + 9 + self.contact_site_dim + 1 + action_chunk_dim + 3 + 2 + 1
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(self.hidden_dim, len(TASK_VALUE_OUTPUTS))

    def forward(
        self,
        state: torch.Tensor,
        velocity: torch.Tensor,
        objects: torch.Tensor,
        contact_sites: torch.Tensor,
        tactile_contact: torch.Tensor,
        action_chunk: torch.Tensor,
        target_onehot: torch.Tensor,
        active_arm_onehot: torch.Tensor,
        visible_distance_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.backbone(
            torch.cat(
                (
                    state,
                    velocity,
                    objects,
                    contact_sites,
                    tactile_contact,
                    action_chunk,
                    target_onehot,
                    active_arm_onehot,
                    5.0 * visible_distance_m,
                ),
                dim=-1,
            )
        )
        raw = self.head(hidden)
        return {name: raw[:, index] for index, name in enumerate(TASK_VALUE_OUTPUTS)}


class G1TaskValueWorldEnsemble(nn.Module):
    def __init__(
        self,
        members: int = 3,
        contact_site_dim: int = 108,
        action_dim: int = 31,
        action_chunk: int = 8,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.members = int(members)
        self.contact_site_dim = int(contact_site_dim)
        self.action_dim = int(action_dim)
        self.action_chunk = int(action_chunk)
        self.hidden_dim = int(hidden_dim)
        self.models = nn.ModuleList(
            [
                G1TaskValueWorldMember(
                    self.contact_site_dim,
                    self.action_dim,
                    self.action_chunk,
                    self.hidden_dim,
                )
                for _ in range(self.members)
            ]
        )

    def forward(self, *args: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [model(*args) for model in self.models]


def positive_part(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(value, beta=4.0)


def physical_outputs(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map unconstrained network outputs into physical units/probabilities."""
    final_distance = positive_part(raw["final_distance_m"])
    min_distance = torch.minimum(positive_part(raw["min_distance_m"]), final_distance)
    return {
        "final_distance_m": final_distance,
        "min_distance_m": min_distance,
        "contact_prob": raw["contact_logit"].sigmoid(),
        "contact_last_prob": raw["contact_last_logit"].sigmoid(),
        "contact_fraction": raw["contact_fraction_logit"].sigmoid(),
        "lift_m": 0.080 * torch.tanh(raw["lift_m"]),
        "hand_object_comotion_m": positive_part(raw["hand_object_comotion_m"]),
        "bin_xy_distance_m": positive_part(raw["bin_xy_distance_m"]),
        "stable_grasp_prob": raw["stable_grasp_logit"].sigmoid(),
    }


def score_task_value_candidates(
    member_outputs: list[dict[str, torch.Tensor]],
    action_chunks: torch.Tensor,
    uncertainty_weight: float = 0.35,
    smoothness_weight_m: float = 0.010,
    rank_penalty_m: float = 0.00075,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return a grasp-aware cost in metres plus inspectable diagnostics.

    Lower is better.  The cost keeps approach distance, but explicitly rewards
    sustained contact, lift, and stable grasp probability while penalizing
    hand-object divergence and abrupt chunks.
    """
    physical = [physical_outputs(item) for item in member_outputs]
    stacked = {key: torch.stack([item[key] for item in physical]) for key in physical[0]}
    mean = {key: value.mean(0) for key, value in stacked.items()}
    std = torch.stack(
        [
            stacked["final_distance_m"].std(0, correction=0),
            stacked["min_distance_m"].std(0, correction=0),
            stacked["stable_grasp_prob"].std(0, correction=0),
        ]
    ).mean(0)
    if action_chunks.ndim == 2:
        steps = action_chunks.reshape(action_chunks.shape[0], -1, 31)
    else:
        steps = action_chunks
    velocity = steps[:, 1:] - steps[:, :-1]
    smoothness = velocity.abs().mean((1, 2)) if velocity.numel() else steps.abs().mean((1, 2))
    magnitude = steps.abs().mean((1, 2))
    rank_cost = torch.arange(
        steps.shape[0], dtype=steps.dtype, device=steps.device
    ) * float(rank_penalty_m)
    cost = (
        0.45 * mean["final_distance_m"]
        + 0.55 * mean["min_distance_m"]
        - 0.035 * mean["contact_fraction"]
        - 0.030 * mean["contact_last_prob"]
        - 0.060 * mean["stable_grasp_prob"]
        - 0.75 * mean["lift_m"].clamp_min(0.0)
        + 0.18 * mean["hand_object_comotion_m"]
        + float(uncertainty_weight) * std
        + float(smoothness_weight_m) * smoothness
        + 0.004 * magnitude
        + rank_cost
    )
    diagnostics = {
        **{f"mean_{key}": value for key, value in mean.items()},
        "uncertainty": std,
        "smoothness_rad": smoothness,
        "magnitude_rad": magnitude,
        "rank_cost_m": rank_cost,
        "cost_m": cost,
    }
    return cost, diagnostics


def save_checkpoint(
    path: Path,
    model: G1TaskValueWorldEnsemble,
    stats: dict[str, torch.Tensor],
    label_distribution: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "members": model.members,
                "contact_site_dim": model.contact_site_dim,
                "action_dim": model.action_dim,
                "action_chunk": model.action_chunk,
                "hidden_dim": model.hidden_dim,
            },
            "stats": {key: value.detach().cpu() for key, value in stats.items()},
            "label_distribution": label_distribution,
            "action_representation": "full_8_step_action_token_candidate_chunk",
            "input_contract": (
                "VLA RGB-derived object estimate + measured state31/qvel31/"
                "finger sites/tactile + full candidate action chunk + language "
                "target + VLA active arm"
            ),
            "label_contract": (
                "MuJoCo branch labels: final/min target distance, sustained "
                "selected-hand target contact, lift, hand-object co-motion, "
                "bin progress, and stable grasp"
            ),
            "privileged_positions_at_inference": False,
            "simulator_object_truth_at_inference": False,
        },
        path,
    )


def load_checkpoint(path: Path, device: str):
    value = torch.load(path, map_location=device, weights_only=False)
    model = G1TaskValueWorldEnsemble(**value["config"]).to(device)
    model.load_state_dict(value["model"])
    model.eval()
    return model, {key: item.to(device) for key, item in value["stats"].items()}, value.get("label_distribution", {})
