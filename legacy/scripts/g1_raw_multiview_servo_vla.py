#!/usr/bin/env python3
"""Raw multi-view visual-servo action-prior VLA for the G1 upper body."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


ACTION_DIM = 31
TARGET_COUNT = 3
ARM_COUNT = 2


class ViewEncoder(nn.Module):
    """Small trainable spatial encoder suited to the rendered task cameras."""

    def __init__(self, output_dim: int = 192):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(12, 96), nn.SiLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GroupNorm(16, 128), nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(), nn.Linear(128 * 3 * 4, output_dim), nn.SiLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class RawMultiviewServoVLA(nn.Module):
    """Images, language and proprioception produce an action distribution."""

    def __init__(self, vocab_size: int, contact_site_dim: int = 108, hidden_dim: int = 384):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.contact_site_dim = int(contact_site_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = ViewEncoder()
        self.view_embedding = nn.Parameter(torch.randn(3, 192) * 0.02)
        self.vision = nn.Sequential(nn.Linear(3 * 192, 384), nn.LayerNorm(384), nn.SiLU())
        self.text = nn.Embedding(self.vocab_size, 64, padding_idx=0)
        self.text_context = nn.Sequential(nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU())
        self.proprio = nn.Sequential(
            nn.Linear(31 + 31 + self.contact_site_dim + 1, 256),
            nn.LayerNorm(256), nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.Linear(384 + 96 + 256, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
        )
        self.object_head = nn.Linear(self.hidden_dim, 9)
        self.object_log_std_head = nn.Linear(self.hidden_dim, 9)
        self.target_head = nn.Linear(96, TARGET_COUNT)
        self.arm_head = nn.Linear(self.hidden_dim, ARM_COUNT)
        self.action_context = nn.Sequential(
            nn.Linear(self.hidden_dim + 9 + 2, self.hidden_dim), nn.SiLU(),
        )
        self.action_mean_head = nn.Linear(self.hidden_dim, ACTION_DIM)
        self.action_log_std_head = nn.Linear(self.hidden_dim, ACTION_DIM)

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
        if images.ndim != 5 or images.shape[1:3] != (3, 3):
            raise ValueError(f"Expected images [B,3,3,H,W], got {tuple(images.shape)}")
        if state.shape[-1] != 31 or velocity.shape[-1] != 31 or contact_sites.shape[-1] != self.contact_site_dim:
            raise ValueError("Raw VLA observation dimensions disagree with checkpoint contract")
        batch = len(state)
        pixels = images.reshape(batch * 3, 3, images.shape[-2], images.shape[-1])
        pixels = torch.nn.functional.interpolate(pixels.float(), size=(64, 96), mode="bilinear", align_corners=False)
        view = self.encoder(pixels).reshape(batch, 3, -1) + self.view_embedding[None]
        vision = self.vision(view.flatten(1))
        words = self.text(token_ids)
        mask = token_mask.to(words.dtype).unsqueeze(-1)
        text = self.text_context((words * mask).sum(1) / mask.sum(1).clamp_min(1.0))
        proprio = self.proprio(torch.cat((state, velocity, contact_sites, tactile_contact), dim=-1))
        context = self.context(torch.cat((vision, text, proprio), dim=-1))
        object_mean = self.object_head(context)
        object_log_std = self.object_log_std_head(context).clamp(-5.0, 0.0)
        target_logits = self.target_head(text)
        arm_logits = self.arm_head(context)
        action_context = self.action_context(torch.cat((context, object_mean, arm_logits.softmax(-1)), dim=-1))
        return {
            "action_mean": self.action_mean_head(action_context),
            "action_log_std": self.action_log_std_head(action_context).clamp(-5.0, -1.0),
            "object_mean": object_mean,
            "object_log_std": object_log_std,
            "target_logits": target_logits,
            "active_arm_logits": arm_logits,
        }


def save_checkpoint(path: Path, model: RawMultiviewServoVLA, vocab: dict[str, int], stats: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "vocab": vocab,
        "stats": {key: value.detach().cpu() for key, value in stats.items()},
        "config": {"vocab_size": model.vocab_size, "contact_site_dim": model.contact_site_dim, "hidden_dim": model.hidden_dim},
        "action_representation": "relative_joint_delta",
        "input_contract": "raw synchronized RGB x3 + language + state31 + qvel31 + measured finger sites + tactile",
        "privileged_positions_at_inference": False,
    }, path)


def load_checkpoint(path: Path, device: str):
    value = torch.load(path, map_location=device, weights_only=False)
    model = RawMultiviewServoVLA(**value["config"]).to(device)
    model.load_state_dict(value["model"])
    model.eval()
    return model, value["vocab"], {key: item.to(device) for key, item in value["stats"].items()}
