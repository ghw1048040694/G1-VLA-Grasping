#!/usr/bin/env python3
"""Causal arm-conditioned VLA that proposes local direct-joint candidates."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from g1_direct_joint_candidate_vla import (
    ACTION_DIM,
    CANDIDATE_COUNT,
    TARGET_COUNT,
    encode_tasks,
)


ARM_COUNT = 2


class LocalJointVLA(nn.Module):
    """Maps causal observation history to arm-specific joint candidates."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 384,
        vision_dim: int = 9,
        contact_site_dim: int = 6,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.vision_dim = int(vision_dim)
        self.contact_site_dim = int(contact_site_dim)
        if self.contact_site_dim <= 0 or self.contact_site_dim % 3:
            raise ValueError("contact_site_dim must be a positive multiple of three")
        self.text = nn.Embedding(self.vocab_size, 64, padding_idx=0)
        self.text_context = nn.Sequential(
            nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU()
        )
        # qpos31 + causal qvel31 + camera object geometry9 + sampled physical
        # finger collision geometry + tactile contact + synchronized RGB cues.
        self.geometry_context = nn.Sequential(
            nn.Linear(31 + 31 + 9 + self.contact_site_dim + 1 + self.vision_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        # This is learned causal memory, rather than a stage, timer, or
        # scripted task scheduler.
        self.temporal_context = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.target_context = nn.Sequential(
            nn.Linear(hidden_dim + 96 + self.contact_site_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.target_head = nn.Linear(96, TARGET_COUNT)
        self.active_arm_head = nn.Linear(hidden_dim, ARM_COUNT)
        self.arm_embedding = nn.Parameter(torch.randn(ARM_COUNT, 32) * 0.03)
        self.candidate_embedding = nn.Parameter(torch.randn(CANDIDATE_COUNT, 32) * 0.03)
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )

    def forward(
        self,
        state: torch.Tensor,
        velocity: torch.Tensor,
        objects: torch.Tensor,
        contact_sites: torch.Tensor,
        tactile_contact: torch.Tensor,
        color_features: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if (
            state.shape[-1] != 31
            or velocity.shape[-1] != 31
            or objects.shape[-1] != 9
            or contact_sites.shape[-1] != self.contact_site_dim
            or tactile_contact.shape[-1] != 1
            or color_features.shape[-1] != self.vision_dim
        ):
            raise ValueError(
                "Expected qpos31, qvel31, objects9, contact sites"
                f"{self.contact_site_dim}, tactile contact, and colour features{self.vision_dim}"
            )
        text = self.text(token_ids)
        mask = token_mask.to(dtype=text.dtype).unsqueeze(-1)
        text = (text * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        text = self.text_context(text)
        target_logits = self.target_head(text)
        geometry_input = torch.cat(
            (state, velocity, objects, contact_sites, tactile_contact, color_features), dim=-1
        )
        if history is None:
            geometry = self.geometry_context(geometry_input)
        else:
            if history.ndim != 3 or history.shape[0] != len(state):
                raise ValueError("Expected causal history shaped (batch, frames, features)")
            if history.shape[-1] != geometry_input.shape[-1]:
                raise ValueError("Causal history feature dimension disagrees with current input")
            encoded = self.geometry_context(history.reshape(-1, history.shape[-1]))
            encoded = encoded.reshape(len(state), history.shape[1], self.hidden_dim)
            _, hidden = self.temporal_context(encoded)
            geometry = hidden[-1]
        object_rows = objects.reshape(-1, TARGET_COUNT, 3)
        contact_rows = contact_sites.reshape(-1, self.contact_site_dim // 3, 3)
        residuals = (
            object_rows[:, :, None, :] - contact_rows[:, None, :, :]
        ).reshape(-1, TARGET_COUNT, self.contact_site_dim)
        context = self.target_context(
            torch.cat(
                (
                    geometry[:, None].expand(-1, TARGET_COUNT, -1),
                    text[:, None].expand(-1, TARGET_COUNT, -1),
                    residuals,
                ),
                dim=-1,
            )
        )
        action_context = torch.cat(
            (
                context[:, :, None, None].expand(
                    -1, -1, ARM_COUNT, CANDIDATE_COUNT, -1
                ),
                self.arm_embedding[None, None, :, None].expand(
                    len(state), TARGET_COUNT, -1, CANDIDATE_COUNT, -1
                ),
                self.candidate_embedding[None, None, None].expand(
                    len(state), TARGET_COUNT, ARM_COUNT, -1, -1
                ),
            ),
            dim=-1,
        )
        return {
            "action_candidates": self.action_decoder(action_context),
            "target_logits": target_logits,
            "active_arm_logits_by_target": self.active_arm_head(context),
        }


def save_checkpoint(
    path: Path,
    model: LocalJointVLA,
    vocab: dict[str, int],
    stats: dict[str, torch.Tensor],
    calibration: torch.Tensor,
    action_representation: str = "absolute_joint_position",
) -> None:
    if action_representation not in {"absolute_joint_position", "relative_joint_delta"}:
        raise ValueError(f"Unsupported action representation: {action_representation}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "stats": {key: value.detach().cpu() for key, value in stats.items()},
            "calibration": calibration.detach().cpu(),
            "action_representation": action_representation,
            "config": {
                "vocab_size": len(vocab),
                "hidden_dim": model.hidden_dim,
                "vision_dim": model.vision_dim,
                "contact_site_dim": model.contact_site_dim,
            },
        },
        path,
    )


def load_checkpoint(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LocalJointVLA(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.action_representation = checkpoint.get(
        "action_representation", "absolute_joint_position"
    )
    if model.action_representation not in {
        "absolute_joint_position",
        "relative_joint_delta",
    }:
        raise ValueError(
            f"Unsupported checkpoint action representation: {model.action_representation}"
        )
    model.eval()
    return model, checkpoint["vocab"], {key: value.to(device) for key, value in checkpoint["stats"].items()}, checkpoint["calibration"].to(device)
