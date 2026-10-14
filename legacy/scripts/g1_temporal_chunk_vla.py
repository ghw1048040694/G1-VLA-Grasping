#!/usr/bin/env python3
"""Temporal raw-RGB action-chunk VLA for the G1 upper-body task.

The model consumes a short causal observation history and predicts a planned
sequence of relative joint actions.  It is deliberately not a scripted phase
machine: language, pixels, robot state and tactile history are the only inputs.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from g1_raw_multiview_servo_vla import ACTION_DIM, ARM_COUNT, TARGET_COUNT, ViewEncoder


class TemporalChunkVLA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        contact_site_dim: int = 108,
        history_frames: int = 4,
        action_chunk: int = 8,
        hidden_dim: int = 384,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.contact_site_dim = int(contact_site_dim)
        self.history_frames = int(history_frames)
        self.action_chunk = int(action_chunk)
        self.hidden_dim = int(hidden_dim)
        self.encoder = ViewEncoder()
        self.view_embedding = nn.Parameter(torch.randn(3, 192) * 0.02)
        self.vision = nn.Sequential(nn.Linear(3 * 192, 320), nn.LayerNorm(320), nn.SiLU())
        self.text = nn.Embedding(self.vocab_size, 64, padding_idx=0)
        self.text_context = nn.Sequential(nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU())
        self.proprio = nn.Sequential(
            nn.Linear(31 + 31 + self.contact_site_dim + 1, 224), nn.LayerNorm(224), nn.SiLU()
        )
        self.temporal_input = nn.Sequential(
            nn.Linear(320 + 96 + 224, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.object_head = nn.Linear(hidden_dim, 9)
        self.object_log_std_head = nn.Linear(hidden_dim, 9)
        self.target_head = nn.Linear(96, TARGET_COUNT)
        self.arm_head = nn.Linear(hidden_dim, ARM_COUNT)
        self.action_context = nn.Sequential(
            nn.Linear(hidden_dim + 9 + ARM_COUNT, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.action_mean_head = nn.Linear(hidden_dim, self.action_chunk * ACTION_DIM)
        self.action_log_std_head = nn.Linear(hidden_dim, self.action_chunk * ACTION_DIM)

    def forward(
        self,
        images: torch.Tensor,
        state: torch.Tensor,
        velocity: torch.Tensor,
        contact_sites: torch.Tensor,
        tactile_contact: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 6 or images.shape[2:4] != (3, 3):
            raise ValueError(f"Expected images [B,T,3,3,H,W], got {tuple(images.shape)}")
        if images.shape[1] != self.history_frames:
            raise ValueError("Temporal image history length disagrees with checkpoint")
        if state.shape != (len(images), self.history_frames, 31):
            raise ValueError("Expected state history [B,T,31]")
        if velocity.shape != state.shape or contact_sites.shape != (len(images), self.history_frames, self.contact_site_dim):
            raise ValueError("Temporal proprioception dimensions disagree with checkpoint")
        if tactile_contact.shape != (len(images), self.history_frames, 1):
            raise ValueError("Expected tactile history [B,T,1]")
        batch, steps = images.shape[:2]
        pixels = images.reshape(batch * steps * 3, 3, images.shape[-2], images.shape[-1])
        pixels = torch.nn.functional.interpolate(pixels.float(), size=(64, 96), mode="bilinear", align_corners=False)
        view = self.encoder(pixels).reshape(batch, steps, 3, -1) + self.view_embedding[None, None]
        vision = self.vision(view.flatten(2))
        words = self.text(token_ids)
        mask = token_mask.to(words.dtype).unsqueeze(-1)
        text = self.text_context((words * mask).sum(1) / mask.sum(1).clamp_min(1.0))
        proprio = self.proprio(torch.cat((state, velocity, contact_sites, tactile_contact), dim=-1))
        temporal_input = self.temporal_input(torch.cat((vision, text[:, None].expand(-1, steps, -1), proprio), dim=-1))
        temporal_output, _ = self.temporal(temporal_input)
        context = temporal_output[:, -1]
        object_mean = self.object_head(context)
        object_log_std = self.object_log_std_head(context).clamp(-5.0, 0.0)
        arm_logits = self.arm_head(context)
        action_context = self.action_context(torch.cat((context, object_mean, arm_logits.softmax(-1)), dim=-1))
        return {
            "action_mean": self.action_mean_head(action_context).reshape(batch, self.action_chunk, ACTION_DIM),
            "action_log_std": self.action_log_std_head(action_context).reshape(batch, self.action_chunk, ACTION_DIM).clamp(-5.0, -1.0),
            "object_mean": object_mean,
            "object_log_std": object_log_std,
            "target_logits": self.target_head(text),
            "active_arm_logits": arm_logits,
        }


def save_checkpoint(path: Path, model: TemporalChunkVLA, vocab: dict[str, int], stats: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "stats": {key: value.detach().cpu() for key, value in stats.items()},
            "config": {
                "vocab_size": model.vocab_size,
                "contact_site_dim": model.contact_site_dim,
                "history_frames": model.history_frames,
                "action_chunk": model.action_chunk,
                "hidden_dim": model.hidden_dim,
            },
            "action_representation": "causal_relative_joint_delta_chunk",
            "input_contract": "four-frame raw RGB x3 + language + state31 + qvel31 + measured finger sites + tactile",
            "privileged_positions_at_inference": False,
        },
        path,
    )


def load_checkpoint(path: Path, device: str):
    value = torch.load(path, map_location=device, weights_only=False)
    model = TemporalChunkVLA(**value["config"]).to(device)
    model.load_state_dict(value["model"])
    model.eval()
    return model, value["vocab"], {key: item.to(device) for key, item in value["stats"].items()}
