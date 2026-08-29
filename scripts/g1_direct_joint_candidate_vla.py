#!/usr/bin/env python3
"""Temporal VLA that proposes direct 31-D dynamic joint-action candidates."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn


TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TARGET_COUNT = 3
PHASE_COUNT = 8
CANDIDATE_COUNT = 4
ACTION_DIM = 31


def encode_tasks(tasks: list[str], vocab: dict[str, int], max_tokens: int = 16):
    ids = torch.zeros((len(tasks), max_tokens), dtype=torch.long)
    mask = torch.zeros((len(tasks), max_tokens), dtype=torch.bool)
    for row, task in enumerate(tasks):
        values = [vocab.get(token, vocab[UNK_TOKEN]) for token in TOKEN_RE.findall(task.lower())[:max_tokens]]
        if values:
            ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
            mask[row, : len(values)] = True
    return ids, mask


class DirectJointCandidateVLA(nn.Module):
    """RGB/language/proprioception VLA with target-specific temporal memory."""

    def __init__(self, vocab_size: int, hidden_dim: int = 384):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.text = nn.Embedding(self.vocab_size, 64, padding_idx=0)
        self.text_context = nn.Sequential(
            nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU()
        )
        # state31 + latched RGB object geometry9 + measured palms6 + RGB cues9
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
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.phase_head = nn.Linear(hidden_dim, PHASE_COUNT)
        self.active_arm_head = nn.Linear(hidden_dim, 2)
        self.candidate_embedding = nn.Parameter(
            torch.randn(CANDIDATE_COUNT, 32) * 0.03
        )
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
        hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        squeeze_time = state.ndim == 2
        if squeeze_time:
            state = state[:, None]
            palms = palms[:, None]
        if state.ndim != 3 or state.shape[-1] != 31:
            raise ValueError("Expected state shape (batch, time, 31)")
        if objects.shape != (len(state), 9) or palms.shape[-1] != 6 or color_features.shape != (len(state), 9):
            raise ValueError("Expected objects9, palms6, and colour features9")
        text = self.text(token_ids)
        text_mask = token_mask.to(dtype=text.dtype).unsqueeze(-1)
        text = (text * text_mask).sum(1) / text_mask.sum(1).clamp_min(1.0)
        text = self.text_context(text)
        target_logits = self.target_head(text)
        steps = state.shape[1]
        geometry = self.geometry_context(
            torch.cat(
                (
                    state,
                    objects[:, None].expand(-1, steps, -1),
                    palms,
                    color_features[:, None].expand(-1, steps, -1),
                ),
                dim=-1,
            )
        )
        object_rows = objects.reshape(-1, TARGET_COUNT, 3)
        palm_rows = palms.reshape(len(state), steps, 2, 3)
        residuals = (
            object_rows[:, None, :, None, :] - palm_rows[:, :, None, :, :]
        ).reshape(len(state), steps, TARGET_COUNT, 6)
        target_context = self.target_context(
            torch.cat(
                (
                    geometry[:, :, None].expand(-1, -1, TARGET_COUNT, -1),
                    text[:, None, None].expand(-1, steps, TARGET_COUNT, -1),
                    residuals,
                ),
                dim=-1,
            )
        )
        recurrent_input = target_context.permute(0, 2, 1, 3).reshape(
            len(state) * TARGET_COUNT, steps, self.hidden_dim
        )
        output, next_hidden = self.temporal(recurrent_input, hidden)
        output = output.reshape(len(state), TARGET_COUNT, steps, self.hidden_dim).permute(0, 2, 1, 3)
        output_for_heads = output
        action_context = torch.cat(
            (
                output[:, :, :, None].expand(-1, -1, -1, CANDIDATE_COUNT, -1),
                self.candidate_embedding[None, None, None].expand(
                    len(state), steps, TARGET_COUNT, -1, -1
                ),
            ),
            dim=-1,
        )
        candidates = self.action_decoder(action_context)
        if squeeze_time:
            candidates = candidates[:, 0]
            output_for_heads = output_for_heads[:, 0]
        return {
            "action_candidates": candidates,
            "target_logits": target_logits,
            "phase_logits_by_target": self.phase_head(output_for_heads),
            "active_arm_logits_by_target": self.active_arm_head(output_for_heads),
            "hidden": next_hidden,
        }


def save_checkpoint(
    path: Path,
    model: DirectJointCandidateVLA,
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
    model = DirectJointCandidateVLA(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return (
        model,
        checkpoint["vocab"],
        {key: value.to(device) for key, value in checkpoint["stats"].items()},
        checkpoint["calibration"].to(device),
    )
