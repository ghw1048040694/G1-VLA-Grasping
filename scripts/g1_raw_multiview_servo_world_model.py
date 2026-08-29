#!/usr/bin/env python3
"""Learned transition model used by the raw-RGB G1 CEM planner."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class RawServoWorldModel(nn.Module):
    def __init__(self, contact_site_dim: int = 108, hidden_dim: int = 512):
        super().__init__()
        self.contact_site_dim = int(contact_site_dim)
        self.hidden_dim = int(hidden_dim)
        input_dim = 31 + 31 + 9 + self.contact_site_dim + 1 + 31 + 3
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.state_head = nn.Linear(hidden_dim, 31)
        self.velocity_head = nn.Linear(hidden_dim, 31)
        self.object_head = nn.Linear(hidden_dim, 9)
        self.site_head = nn.Linear(hidden_dim, self.contact_site_dim)
        self.contact_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        velocity: torch.Tensor,
        objects: torch.Tensor,
        contact_sites: torch.Tensor,
        tactile_contact: torch.Tensor,
        action: torch.Tensor,
        target_onehot: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.network(torch.cat((state, velocity, objects, contact_sites, tactile_contact, action, target_onehot), dim=-1))
        return {
            "state": self.state_head(hidden),
            "velocity": self.velocity_head(hidden),
            "objects": self.object_head(hidden),
            "contact_sites": self.site_head(hidden),
            "contact_logit": self.contact_head(hidden),
        }


def save_checkpoint(path: Path, model: RawServoWorldModel, stats: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {"contact_site_dim": model.contact_site_dim, "hidden_dim": model.hidden_dim},
        "stats": {key: value.detach().cpu() for key, value in stats.items()},
        "action_representation": "relative_joint_delta",
        "state_contract": "VLA-estimated object pose + state31 + qvel31 + measured finger sites + tactile; no simulator object truth",
        "privileged_positions_at_inference": False,
    }, path)


def load_checkpoint(path: Path, device: str):
    value = torch.load(path, map_location=device, weights_only=False)
    model = RawServoWorldModel(**value["config"]).to(device)
    model.load_state_dict(value["model"])
    model.eval()
    return model, {key: item.to(device) for key, item in value["stats"].items()}
