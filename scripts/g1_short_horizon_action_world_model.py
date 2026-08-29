#!/usr/bin/env python3
"""Short-horizon, action-conditioned world model for G1 VLA MPC."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class ShortHorizonWorldMember(nn.Module):
    def __init__(self, action_dim: int, contact_site_dim: int = 108, hidden_dim: int = 384):
        super().__init__()
        self.action_dim = int(action_dim)
        self.contact_site_dim = int(contact_site_dim)
        self.hidden_dim = int(hidden_dim)
        input_dim = 31 + 31 + 9 + self.contact_site_dim + 1 + self.action_dim + 3 + 2 + 1
        self.network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
        )
        self.distance_correction_head = nn.Linear(self.hidden_dim, 1)
        self.contact_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        velocity: torch.Tensor,
        objects: torch.Tensor,
        sites: torch.Tensor,
        tactile: torch.Tensor,
        action_chunk: torch.Tensor,
        target: torch.Tensor,
        arm: torch.Tensor,
        visible_distance: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        hidden = self.network(torch.cat((state, velocity, objects, sites, tactile, action_chunk, target, arm, 5.0 * visible_distance), dim=-1))
        return {
            # Centered at candidate-set level by the planner/trainer. The cap
            # permits four-control-step differences without allowing absolute
            # distance hallucinations.
            "distance_correction_m": 0.100 * torch.tanh(self.distance_correction_head(hidden)),
            "contact_logit": self.contact_head(hidden),
        }


class ShortHorizonWorldEnsemble(nn.Module):
    def __init__(self, members: int, action_dim: int, contact_site_dim: int = 108, hidden_dim: int = 384):
        super().__init__()
        self.members = int(members)
        self.action_dim = int(action_dim)
        self.contact_site_dim = int(contact_site_dim)
        self.hidden_dim = int(hidden_dim)
        self.models = nn.ModuleList([ShortHorizonWorldMember(self.action_dim, self.contact_site_dim, self.hidden_dim) for _ in range(self.members)])

    def forward(self, *args: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        return [model(*args) for model in self.models]


def save_checkpoint(path: Path, model: ShortHorizonWorldEnsemble, stats: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {"members": model.members, "action_dim": model.action_dim, "contact_site_dim": model.contact_site_dim, "hidden_dim": model.hidden_dim},
        "stats": {key: value.detach().cpu() for key, value in stats.items()},
        "action_representation": "four_step_action_token_vla_chunk",
        "state_contract": "VLA RGB visual object estimate + measured state31/qvel31/finger sites/tactile + VLA action chunk + language target + VLA arm",
        "prediction_contract": "physical target distance and contact after a candidate action chunk; deployment executes first action then re-observes",
        "privileged_positions_at_inference": False,
    }, path)


def load_checkpoint(path: Path, device: str):
    value = torch.load(path, map_location=device, weights_only=False)
    model = ShortHorizonWorldEnsemble(**value["config"]).to(device)
    model.load_state_dict(value["model"])
    model.eval()
    return model, {key: item.to(device) for key, item in value["stats"].items()}
