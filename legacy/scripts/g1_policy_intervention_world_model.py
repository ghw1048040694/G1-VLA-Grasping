#!/usr/bin/env python3
"""Calibrated one-step world model for action-token VLA interventions.

The model deliberately predicts bounded *residual* motion from measured hand
geometry and a VLA visual object estimate.  It cannot turn a 30 cm separation
into a millimetre separation by freely emitting unrelated absolute positions.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class PolicyInterventionWorldMember(nn.Module):
    """One ensemble member with physically bounded geometry residuals."""

    def __init__(self, contact_site_dim: int = 108, hidden_dim: int = 384):
        super().__init__()
        self.contact_site_dim = int(contact_site_dim)
        self.hidden_dim = int(hidden_dim)
        # state, velocity, visual objects, measured sites, tactile, action,
        # language target one-hot, and VLA-selected active-arm one-hot.
        input_dim = 31 + 31 + 9 + self.contact_site_dim + 1 + 31 + 3 + 2 + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.site_delta_head = nn.Linear(self.hidden_dim, self.contact_site_dim)
        self.object_delta_head = nn.Linear(self.hidden_dim, 9)
        self.distance_correction_head = nn.Linear(self.hidden_dim, 1)
        self.contact_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        velocity: torch.Tensor,
        objects: torch.Tensor,
        contact_sites: torch.Tensor,
        tactile_contact: torch.Tensor,
        action: torch.Tensor,
        target_onehot: torch.Tensor,
        active_arm_onehot: torch.Tensor,
        visible_distance_m: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.network(
            torch.cat(
                (
                    state,
                    velocity,
                    objects,
                    contact_sites,
                    tactile_contact,
                    action,
                    target_onehot,
                    active_arm_onehot,
                    # This is calculated from the same VLA object estimate
                    # and measured finger sites already present in the input.
                    # Scaling keeps it commensurate with normalized features.
                    5.0 * visible_distance_m,
                ),
                dim=-1,
            )
        )
        # One 15 Hz controller interval cannot move a fingertip or a free
        # object arbitrarily far. These caps are part of the deployment
        # contract and protect candidate selection under distribution shift.
        return {
            "site_delta_m": 0.050 * torch.tanh(self.site_delta_head(hidden)),
            "object_delta_m": 0.040 * torch.tanh(self.object_delta_head(hidden)),
            "distance_correction_m": 0.060 * torch.tanh(
                self.distance_correction_head(hidden)
            ),
            "contact_logit": self.contact_head(hidden),
        }


class PolicyInterventionWorldEnsemble(nn.Module):
    def __init__(
        self,
        members: int = 3,
        contact_site_dim: int = 108,
        hidden_dim: int = 384,
    ) -> None:
        super().__init__()
        self.members = int(members)
        self.contact_site_dim = int(contact_site_dim)
        self.hidden_dim = int(hidden_dim)
        self.models = nn.ModuleList(
            [
                PolicyInterventionWorldMember(self.contact_site_dim, self.hidden_dim)
                for _ in range(self.members)
            ]
        )

    def forward(self, *args: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [model(*args) for model in self.models]


def save_checkpoint(
    path: Path,
    model: PolicyInterventionWorldEnsemble,
    stats: dict[str, torch.Tensor],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "members": model.members,
                "contact_site_dim": model.contact_site_dim,
                "hidden_dim": model.hidden_dim,
            },
            "stats": {key: value.detach().cpu() for key, value in stats.items()},
            "action_representation": "executed_relative_joint_delta",
            "state_contract": (
                "action-token VLA visual object estimate + measured state31 + "
                "qvel31 + finger contact sites + tactile + candidate action + "
                "language target + VLA active arm"
            ),
            "geometry_contract": (
                "bounded residual next-site/object prediction plus direct "
                "physical hand-target distance calibration"
            ),
            "privileged_positions_at_inference": False,
            "simulator_object_truth_at_inference": False,
        },
        path,
    )


def load_checkpoint(path: Path, device: str):
    value = torch.load(path, map_location=device, weights_only=False)
    model = PolicyInterventionWorldEnsemble(**value["config"]).to(device)
    model.load_state_dict(value["model"])
    model.eval()
    return model, {key: item.to(device) for key, item in value["stats"].items()}
