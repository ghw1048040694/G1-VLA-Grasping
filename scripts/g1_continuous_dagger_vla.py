#!/usr/bin/env python3
"""Continuous action-chunk VLA trained with visited-state expert corrections."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from g1_raw_multiview_servo_vla import ACTION_DIM, ARM_COUNT, TARGET_COUNT, ViewEncoder

PHASE_COUNT = 8
HAND_INDICES = (10, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 29, 30)
ARM_INDICES = tuple(index for index in range(ACTION_DIM) if index not in HAND_INDICES)


class ContinuousDaggerVLA(nn.Module):
    def __init__(self, vocab_size: int, contact_site_dim: int = 108,
                 history_frames: int = 4, action_chunk: int = 8, hidden_dim: int = 384,
                 factorized_actions: bool = False):
        super().__init__()
        self.vocab_size, self.contact_site_dim = int(vocab_size), int(contact_site_dim)
        self.history_frames, self.action_chunk, self.hidden_dim = int(history_frames), int(action_chunk), int(hidden_dim)
        self.factorized_actions = bool(factorized_actions)
        self.encoder = ViewEncoder()
        self.view_embedding = nn.Parameter(torch.randn(3, 192) * 0.02)
        self.vision = nn.Sequential(nn.Linear(576, 320), nn.LayerNorm(320), nn.SiLU())
        self.text = nn.Embedding(self.vocab_size, 64, padding_idx=0)
        self.text_context = nn.Sequential(nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU())
        self.proprio = nn.Sequential(nn.Linear(31 + 31 + self.contact_site_dim + 1, 224), nn.LayerNorm(224), nn.SiLU())
        self.fusion = nn.Sequential(nn.Linear(320 + 96 + 224, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.temporal = nn.GRU(hidden_dim, hidden_dim, 2, batch_first=True)
        self.target_head = nn.Linear(96, TARGET_COUNT)
        self.arm_head = nn.Linear(hidden_dim, ARM_COUNT)
        self.phase_head = nn.Linear(hidden_dim, PHASE_COUNT)
        self.distance_head = nn.Sequential(nn.Linear(hidden_dim, 96), nn.SiLU(), nn.Linear(96, 1))
        self.contact_head = nn.Linear(hidden_dim, 1)
        def action_decoder(size): return nn.Sequential(
            nn.Linear(hidden_dim + ARM_COUNT + PHASE_COUNT, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.action_chunk * size),
        )
        if self.factorized_actions:
            self.arm_action_head = action_decoder(len(ARM_INDICES))
            self.hand_action_head = action_decoder(len(HAND_INDICES))
        else:
            self.action_head = action_decoder(ACTION_DIM)

    def forward(self, images, state, velocity, sites, tactile, ids, mask):
        batch, steps = images.shape[:2]
        if images.shape[1] != self.history_frames or images.shape[2:4] != (3, 3):
            raise ValueError(f"Expected [B,{self.history_frames},3,3,H,W], got {tuple(images.shape)}")
        pixels = torch.nn.functional.interpolate(
            images.reshape(batch * steps * 3, 3, *images.shape[-2:]).float(),
            size=(64, 96), mode="bilinear", align_corners=False,
        )
        view = self.encoder(pixels).reshape(batch, steps, 3, -1) + self.view_embedding[None, None]
        vision = self.vision(view.flatten(2))
        words = self.text(ids); text_mask = mask.to(words.dtype).unsqueeze(-1)
        text = self.text_context((words * text_mask).sum(1) / text_mask.sum(1).clamp_min(1))
        proprio = self.proprio(torch.cat((state, velocity, sites, tactile), -1))
        sequence = self.fusion(torch.cat((vision, text[:, None].expand(-1, steps, -1), proprio), -1))
        context = self.temporal(sequence)[0][:, -1]
        arm_logits, phase_logits = self.arm_head(context), self.phase_head(context)
        action_context = torch.cat((context, arm_logits.softmax(-1), phase_logits.softmax(-1)), -1)
        if self.factorized_actions:
            action = context.new_zeros(batch, self.action_chunk, ACTION_DIM)
            action[:, :, ARM_INDICES] = self.arm_action_head(action_context).reshape(batch, self.action_chunk, len(ARM_INDICES))
            action[:, :, HAND_INDICES] = self.hand_action_head(action_context).reshape(batch, self.action_chunk, len(HAND_INDICES))
        else:
            action = self.action_head(action_context).reshape(batch, self.action_chunk, ACTION_DIM)
        return {
            "action": action,
            "target_logits": self.target_head(text), "active_arm_logits": arm_logits,
            "phase_logits": phase_logits, "distance": self.distance_head(context),
            "contact_logits": self.contact_head(context),
        }


def save_checkpoint(path: Path, model: ContinuousDaggerVLA, vocab, stats, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "vocab": vocab,
        "stats": {key: value.detach().cpu() for key, value in stats.items()},
        "config": {key: getattr(model, key) for key in ("vocab_size", "contact_site_dim", "history_frames", "action_chunk", "hidden_dim", "factorized_actions")},
        "action_representation": "continuous_relative_joint_delta_chunk",
        "input_contract": "causal raw RGB x3 + language + state31 + qvel31 + finger sites + tactile",
        "privileged_positions_at_inference": False, "training_metadata": metadata or {},
    }, path)


def load_checkpoint(path: Path, device: str):
    value = torch.load(path, map_location=device, weights_only=False)
    model = ContinuousDaggerVLA(**value["config"]).to(device); model.load_state_dict(value["model"]); model.eval()
    return model, value["vocab"], {key: item.to(device) for key, item in value["stats"].items()}, value
