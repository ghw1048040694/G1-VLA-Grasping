#!/usr/bin/env python3
"""Small camera-language-proprioception policy for local G1 contact actions.

The policy deliberately consumes only geometry derived from the live RGB head
camera, language tokens, and robot proprioception. Object ground truth is used
only while constructing supervised labels, never by this module at inference.
"""

from __future__ import annotations

import re

import torch
from torch import nn


TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def encode_tasks(tasks: list[str], vocab: dict[str, int], max_tokens: int = 16):
    ids = torch.zeros((len(tasks), max_tokens), dtype=torch.long)
    mask = torch.zeros((len(tasks), max_tokens), dtype=torch.bool)
    for row, task in enumerate(tasks):
        words = TOKEN_RE.findall(task.lower())[:max_tokens]
        values = [vocab.get(word, vocab[UNK_TOKEN]) for word in words]
        if values:
            ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
            mask[row, : len(values)] = True
    return ids, mask


def calibrated_object_positions(
    color_features: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    """Back-project RGB centroids using the fixed camera-only calibration."""
    if color_features.ndim != 2 or color_features.shape[1] != 9:
        raise ValueError("color_features must have shape [B,9]")
    if coefficients.shape != (3, 7, 3):
        raise ValueError("calibration coefficients must have shape [3,7,3]")
    values = []
    for index in range(3):
        x, y, area = color_features[:, 3 * index : 3 * index + 3].unbind(-1)
        design = torch.stack(
            (torch.ones_like(x), x, y, area, x * x, y * y, x * y), dim=-1
        )
        values.append(design @ coefficients[index])
    objects = torch.stack(values, dim=1)
    objects[..., 0] = objects[..., 0].clamp(0.28, 0.68)
    objects[..., 1] = objects[..., 1].clamp(-0.42, 0.28)
    objects[..., 2] = objects[..., 2].clamp(0.73, 1.05)
    return objects


class GeometricContactVLA(nn.Module):
    """Target-conditioned local Cartesian action policy.

    Per-target action heads avoid averaging the incompatible left/right grasp
    and finger profiles for the three object types. Target selection remains a
    learned language output and is part of the action computation.
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 384):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.text = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.text_context = nn.Sequential(
            nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU()
        )
        # state31 + objects9 + palms6 + raw colour cues9 + selected residual6
        self.geometry_context = nn.Sequential(
            nn.Linear(61, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.Linear(hidden_dim + 96, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.target_head = nn.Linear(96, 3)
        self.phase_head = nn.Linear(hidden_dim, 8)
        self.active_arm_head = nn.Linear(hidden_dim, 2)
        self.action_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim + 6, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 35),
                )
                for _ in range(3)
            ]
        )

    def forward(
        self,
        state: torch.Tensor,
        objects: torch.Tensor,
        palms: torch.Tensor,
        color_features: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        target_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if state.shape[-1] != 31 or objects.shape[-1] != 9 or palms.shape[-1] != 6:
            raise ValueError("Expected state31, objects9, and palms6 inputs")
        text = self.text(token_ids)
        mask = token_mask.to(dtype=text.dtype).unsqueeze(-1)
        text = (text * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        text = self.text_context(text)
        target_logits = self.target_head(text)
        target_probability = (
            target_override
            if target_override is not None
            else target_logits.softmax(dim=-1)
        )
        object_rows = objects.reshape(-1, 3, 3)
        selected = torch.sum(object_rows * target_probability[..., None], dim=1)
        selected_residual = (selected[:, None, :] - palms.reshape(-1, 2, 3)).reshape(
            -1, 6
        )
        geometry = torch.cat(
            (state, objects, palms, color_features, selected_residual), dim=-1
        )
        geometry = self.geometry_context(geometry)
        context = self.context(torch.cat((geometry, text), dim=-1))
        actions = torch.stack(
            [head(torch.cat((context, selected_residual), dim=-1)) for head in self.action_heads],
            dim=1,
        )
        action = torch.sum(actions * target_probability[:, :, None], dim=1)
        return {
            "action": action,
            "target_logits": target_logits,
            "phase_logits": self.phase_head(context),
            "active_arm_logits": self.active_arm_head(context),
            "selected_residual": selected_residual,
        }


def save_checkpoint(
    path,
    model: GeometricContactVLA,
    vocab: dict[str, int],
    stats: dict[str, torch.Tensor],
    calibration: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "stats": {key: value.detach().cpu() for key, value in stats.items()},
            "calibration": calibration.detach().cpu(),
            "config": {"vocab_size": len(vocab), "hidden_dim": model.hidden_dim},
        },
        path,
    )


def load_checkpoint(path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = GeometricContactVLA(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return (
        model,
        checkpoint["vocab"],
        {key: value.to(device) for key, value in checkpoint["stats"].items()},
        checkpoint["calibration"].to(device),
    )
