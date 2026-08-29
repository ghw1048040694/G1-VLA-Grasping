#!/usr/bin/env python3
"""Stateless visual-language direct joint candidates for contact-gated MPC."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from g1_direct_joint_candidate_vla import (
    ACTION_DIM,
    CANDIDATE_COUNT,
    PAD_TOKEN,
    PHASE_COUNT,
    TARGET_COUNT,
    UNK_TOKEN,
    encode_tasks,
)


class ContactGatedJointVLA(nn.Module):
    """Propose phase candidates from the observed geometry, without a timer."""

    def __init__(self, vocab_size: int, hidden_dim: int = 384):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.text = nn.Embedding(self.vocab_size, 64, padding_idx=0)
        self.text_context = nn.Sequential(
            nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU()
        )
        # state31 + camera object geometry9 + measured palms6 + RGB cues9
        self.geometry_context = nn.Sequential(
            nn.Linear(55, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.target_context = nn.Sequential(
            nn.Linear(hidden_dim + 96 + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.target_head = nn.Linear(96, TARGET_COUNT)
        self.active_arm_head = nn.Linear(hidden_dim, 2)
        # This predicts the *next candidate stage* directly from observed state,
        # not a clock or an external hand-written stage scheduler.
        self.next_stage_head = nn.Linear(hidden_dim, CANDIDATE_COUNT)
        self.stage_embedding = nn.Parameter(torch.randn(CANDIDATE_COUNT, 32) * 0.03)
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )

    def forward(
        self,
        state: torch.Tensor,
        objects: torch.Tensor,
        palms: torch.Tensor,
        color_features: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state.shape[-1] != 31 or objects.shape[-1] != 9 or palms.shape[-1] != 6 or color_features.shape[-1] != 9:
            raise ValueError("Expected state31, objects9, palms6, and colour features9")
        text = self.text(token_ids)
        mask = token_mask.to(dtype=text.dtype).unsqueeze(-1)
        text = (text * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        text = self.text_context(text)
        target_logits = self.target_head(text)
        geometry = self.geometry_context(torch.cat((state, objects, palms, color_features), dim=-1))
        object_rows = objects.reshape(-1, TARGET_COUNT, 3)
        residuals = (object_rows[:, :, None, :] - palms.reshape(-1, 1, 2, 3)).reshape(-1, TARGET_COUNT, 6)
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
                context[:, :, None].expand(-1, -1, CANDIDATE_COUNT, -1),
                self.stage_embedding[None, None].expand(len(state), TARGET_COUNT, -1, -1),
            ),
            dim=-1,
        )
        return {
            "action_candidates": self.action_decoder(action_context),
            "target_logits": target_logits,
            "next_stage_logits_by_target": self.next_stage_head(context),
            "active_arm_logits_by_target": self.active_arm_head(context),
        }


def save_checkpoint(path: Path, model: ContactGatedJointVLA, vocab: dict[str, int], stats: dict[str, torch.Tensor], calibration: torch.Tensor) -> None:
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


def load_checkpoint(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ContactGatedJointVLA(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["vocab"], {key: value.to(device) for key, value in checkpoint["stats"].items()}, checkpoint["calibration"].to(device)
