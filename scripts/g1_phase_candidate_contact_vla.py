#!/usr/bin/env python3
"""Phase-specialized visual contact policy for G1.

The model accepts only camera-derived object geometry, language, and the 31-D
upper-body state.  It produces one learned Cartesian contact action for each
of the four approach phases.  Keeping those actions separate prevents a
deterministic regressor from averaging high-approach, pregrasp, and closing
hand poses into an unreachable waypoint.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn


TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TARGET_COUNT = 3
PHASE_COUNT = 4
ACTION_DIM = 35


def encode_tasks(tasks: list[str], vocab: dict[str, int], max_tokens: int = 16):
    ids = torch.zeros((len(tasks), max_tokens), dtype=torch.long)
    mask = torch.zeros((len(tasks), max_tokens), dtype=torch.bool)
    for row, task in enumerate(tasks):
        tokens = TOKEN_RE.findall(task.lower())[:max_tokens]
        values = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
        if values:
            ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
            mask[row, : len(values)] = True
    return ids, mask


class PhaseCandidateContactVLA(nn.Module):
    """Separate learned local action heads for target x approach phase."""

    def __init__(self, vocab_size: int, hidden_dim: int = 384):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.text = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.text_context = nn.Sequential(
            nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU()
        )
        # state31 + three camera-localized objects9 + palms6 + colour cues9
        self.geometry_context = nn.Sequential(
            nn.Linear(55, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Each target receives its own visual palm residual before actions are
        # decoded.  This avoids mixing mirrored left/right grasp geometries.
        self.target_context = nn.Sequential(
            nn.Linear(hidden_dim + 96 + 6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.target_head = nn.Linear(96, TARGET_COUNT)
        self.phase_head = nn.Linear(hidden_dim, PHASE_COUNT)
        self.active_arm_head = nn.Linear(hidden_dim, 2)
        self.phase_embedding = nn.Parameter(torch.randn(PHASE_COUNT, 32) * 0.03)
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )
        # Contact is a camera-geometry problem first.  The candidate target's
        # displacement from both measured palms is essential when random
        # layouts place the same colour on opposite sides of the robot.
        # This remains a learned decoder; no object truth enters inference.
        self.contact_visual_decoder = nn.Sequential(
            nn.Linear(30, hidden_dim),
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
        geometry = self.geometry_context(
            torch.cat((state, objects, palms, color_features), dim=-1)
        )
        object_rows = objects.reshape(-1, TARGET_COUNT, 3)
        residuals = (
            object_rows[:, :, None, :] - palms.reshape(-1, 1, 2, 3)
        ).reshape(-1, TARGET_COUNT, 6)
        target_context = self.target_context(
            torch.cat(
                (
                    geometry[:, None, :].expand(-1, TARGET_COUNT, -1),
                    text[:, None, :].expand(-1, TARGET_COUNT, -1),
                    residuals,
                ),
                dim=-1,
            )
        )
        selected_context = torch.sum(
            target_context * target_probability[:, :, None], dim=1
        )
        phase_context = torch.cat(
            (
                target_context[:, :, None, :].expand(-1, -1, PHASE_COUNT, -1),
                self.phase_embedding[None, None, :, :].expand(
                    len(state), TARGET_COUNT, -1, -1
                ),
            ),
            dim=-1,
        )
        candidates = self.action_decoder(phase_context)
        candidate_identity = torch.eye(
            TARGET_COUNT, dtype=objects.dtype, device=objects.device
        )[None].expand(len(objects), -1, -1)
        contact_context = torch.cat(
            (
                objects[:, None, :].expand(-1, TARGET_COUNT, -1),
                color_features[:, None, :].expand(-1, TARGET_COUNT, -1),
                object_rows,
                candidate_identity,
                residuals,
            ),
            dim=-1,
        )
        candidates[:, :, PHASE_COUNT - 1] = self.contact_visual_decoder(contact_context)
        return {
            "action_candidates": candidates,
            "target_logits": target_logits,
            "phase_logits": self.phase_head(selected_context),
            "active_arm_logits": self.active_arm_head(selected_context),
            "target_residuals": residuals,
        }


def save_checkpoint(
    path: Path,
    model: PhaseCandidateContactVLA,
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


def load_checkpoint(path: Path, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = PhaseCandidateContactVLA(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return (
        model,
        checkpoint["vocab"],
        {key: value.to(device) for key, value in checkpoint["stats"].items()},
        checkpoint["calibration"].to(device),
    )
