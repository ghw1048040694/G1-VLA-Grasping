#!/usr/bin/env python3
"""Uncertainty ensemble transition model for temporal action-chunk MPC."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TemporalChunkWorldMember(nn.Module):
    def __init__(self, contact_site_dim: int = 108, hidden_dim: int = 512):
        super().__init__()
        self.contact_site_dim = int(contact_site_dim); self.hidden_dim = int(hidden_dim)
        input_dim = 31 + 31 + 9 + self.contact_site_dim + 1 + 31 + 3
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.state_head = nn.Linear(hidden_dim, 31); self.velocity_head = nn.Linear(hidden_dim, 31)
        self.object_head = nn.Linear(hidden_dim, 9); self.site_head = nn.Linear(hidden_dim, self.contact_site_dim)

    def forward(self, state, velocity, objects, contact_sites, tactile_contact, action, target_onehot):
        hidden = self.network(torch.cat((state, velocity, objects, contact_sites, tactile_contact, action, target_onehot), dim=-1))
        return {"state": self.state_head(hidden), "velocity": self.velocity_head(hidden), "objects": self.object_head(hidden), "contact_sites": self.site_head(hidden)}


class TemporalChunkWorldEnsemble(nn.Module):
    def __init__(self, members: int = 3, contact_site_dim: int = 108, hidden_dim: int = 512):
        super().__init__()
        self.members = int(members); self.contact_site_dim = int(contact_site_dim); self.hidden_dim = int(hidden_dim)
        self.models = nn.ModuleList([TemporalChunkWorldMember(contact_site_dim, hidden_dim) for _ in range(self.members)])

    def forward(self, *args):
        return [model(*args) for model in self.models]


def save_checkpoint(path: Path, model: TemporalChunkWorldEnsemble, stats: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": {"members": model.members, "contact_site_dim": model.contact_site_dim, "hidden_dim": model.hidden_dim},
        "stats": {key: value.detach().cpu() for key, value in stats.items()},
        "action_representation": "relative_joint_delta_chunk_prior",
        "state_contract": "temporal-VLA visual object pose + state31 + qvel31 + measured finger sites + tactile; no simulator object truth",
        "contact_reward_enabled": False,
        "privileged_positions_at_inference": False,
    }, path)


def load_checkpoint(path: Path, device: str):
    value = torch.load(path, map_location=device, weights_only=False)
    model = TemporalChunkWorldEnsemble(**value["config"]).to(device)
    model.load_state_dict(value["model"]); model.eval()
    return model, {key: item.to(device) for key, item in value["stats"].items()}
