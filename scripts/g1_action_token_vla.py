#!/usr/bin/env python3
"""Multi-view VLA that proposes discrete expert action chunks instead of a mean action."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from g1_raw_multiview_servo_vla import ACTION_DIM, ARM_COUNT, TARGET_COUNT, ViewEncoder


class ActionTokenVLA(nn.Module):
    """A visual-language action proposal model with no privileged scene state."""

    def __init__(
        self,
        vocab_size: int,
        token_count: int,
        contact_site_dim: int = 108,
        history_frames: int = 4,
        action_chunk: int = 8,
        hidden_dim: int = 384,
        use_temporal_history: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.token_count = int(token_count)
        self.contact_site_dim = int(contact_site_dim)
        self.history_frames = int(history_frames)
        self.action_chunk = int(action_chunk)
        self.hidden_dim = int(hidden_dim)
        self.use_temporal_history = bool(use_temporal_history)
        self.encoder = ViewEncoder()
        self.view_embedding = nn.Parameter(torch.randn(3, 192) * 0.02)
        # This backbone intentionally matches G1LANG-412's validated raw-RGB
        # visual-state model.  Only the output action representation is new.
        self.vision = nn.Sequential(nn.Linear(576, 384), nn.LayerNorm(384), nn.SiLU())
        self.text = nn.Embedding(self.vocab_size, 64, padding_idx=0)
        self.text_context = nn.Sequential(
            nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU()
        )
        self.proprio = nn.Sequential(
            nn.Linear(31 + 31 + self.contact_site_dim + 1, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.context = nn.Sequential(
            nn.Linear(384 + 96 + 256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.temporal = (
            nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
            if self.use_temporal_history
            else None
        )
        self.temporal_scale = (
            nn.Parameter(torch.tensor(0.05)) if self.use_temporal_history else None
        )
        self.object_head = nn.Linear(self.hidden_dim, 9)
        self.target_head = nn.Linear(96, TARGET_COUNT)
        self.arm_head = nn.Linear(self.hidden_dim, ARM_COUNT)
        self.token_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 9 + ARM_COUNT, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.token_count),
        )
        self.token_embedding = nn.Embedding(self.token_count, 64)
        self.action_residual = nn.Sequential(
            nn.Linear(self.hidden_dim + 9 + ARM_COUNT + 64, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.action_chunk * ACTION_DIM),
        )

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
        if images.ndim != 6 or images.shape[1] != self.history_frames or images.shape[2:4] != (3, 3):
            raise ValueError(f"Expected [batch,{self.history_frames},3,3,height,width] RGB history")
        batch, steps = images.shape[:2]
        if state.shape != (batch, steps, 31) or velocity.shape != state.shape:
            raise ValueError("Unexpected state or velocity shape")
        if contact_sites.shape != (batch, steps, self.contact_site_dim):
            raise ValueError("Unexpected finger contact-site shape")
        if not self.use_temporal_history:
            images = images[:, -1:]
            state, velocity = state[:, -1:], velocity[:, -1:]
            contact_sites = contact_sites[:, -1:]
            tactile_contact = tactile_contact[:, -1:]
            steps = 1
        pixels = torch.nn.functional.interpolate(
            images.reshape(batch * steps * 3, 3, *images.shape[-2:]).float(),
            size=(64, 96),
            mode="bilinear",
            align_corners=False,
        )
        view = self.encoder(pixels).reshape(batch, steps, 3, -1)
        view = view + self.view_embedding[None, None]
        vision = self.vision(view.flatten(2))
        words = self.text(token_ids)
        word_mask = token_mask.to(words.dtype).unsqueeze(-1)
        text = self.text_context((words * word_mask).sum(1) / word_mask.sum(1).clamp_min(1.0))
        proprio = self.proprio(
            torch.cat((state, velocity, contact_sites, tactile_contact), dim=-1)
        )
        text_steps = text[:, None].expand(-1, steps, -1)
        context = self.context(torch.cat((vision, text_steps, proprio), dim=-1))
        if self.temporal is not None:
            temporal = self.temporal(context)[0][:, -1]
            context = context[:, -1] + self.temporal_scale * temporal
        else:
            context = context[:, -1]
        object_mean = self.object_head(context)
        arm_logits = self.arm_head(context)
        proposal_context = torch.cat((context, object_mean, arm_logits.softmax(-1)), dim=-1)
        token_logits = self.token_head(proposal_context)
        tokens = self.token_embedding.weight[None].expand(batch, -1, -1)
        residual_input = torch.cat(
            (proposal_context[:, None].expand(-1, self.token_count, -1), tokens), dim=-1
        )
        # The fixed token gives multimodal structure; this bounded learned
        # residual adapts it to the currently observed object layout.
        candidate_residual = .04 * torch.tanh(self.action_residual(residual_input))
        return {
            "action_token_logits": token_logits,
            "candidate_action_residual": candidate_residual.reshape(
                batch, self.token_count, self.action_chunk, ACTION_DIM
            ),
            "object_mean": object_mean,
            "target_logits": self.target_head(text),
            "active_arm_logits": arm_logits,
        }


class LegacyTemporalActionTokenVLA(nn.Module):
    """Loader-compatible model for the archived G1LANG-423 action-token VLA.

    The archived checkpoint predates per-token learned residuals.  Its action
    candidates are the discrete codebook chunks directly, so this wrapper
    returns a zero residual tensor under the current output contract.
    """

    def __init__(
        self,
        vocab_size: int,
        token_count: int,
        contact_site_dim: int = 108,
        history_frames: int = 4,
        action_chunk: int = 8,
        hidden_dim: int = 384,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.token_count = int(token_count)
        self.contact_site_dim = int(contact_site_dim)
        self.history_frames = int(history_frames)
        self.action_chunk = int(action_chunk)
        self.hidden_dim = int(hidden_dim)
        self.encoder = ViewEncoder()
        self.view_embedding = nn.Parameter(torch.randn(3, 192) * 0.02)
        self.vision = nn.Sequential(nn.Linear(576, 320), nn.LayerNorm(320), nn.SiLU())
        self.text = nn.Embedding(self.vocab_size, 64, padding_idx=0)
        self.text_context = nn.Sequential(
            nn.Linear(64, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU()
        )
        self.proprio = nn.Sequential(
            nn.Linear(31 + 31 + self.contact_site_dim + 1, 224),
            nn.LayerNorm(224),
            nn.SiLU(),
        )
        self.temporal_input = nn.Sequential(
            nn.Linear(320 + 96 + 224, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.temporal = nn.GRU(
            self.hidden_dim, self.hidden_dim, num_layers=2, batch_first=True
        )
        self.object_head = nn.Linear(self.hidden_dim, 9)
        self.target_head = nn.Linear(96, TARGET_COUNT)
        self.arm_head = nn.Linear(self.hidden_dim, ARM_COUNT)
        self.token_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 9 + ARM_COUNT, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.token_count),
        )

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
        if images.ndim != 6 or images.shape[1] != self.history_frames or images.shape[2:4] != (3, 3):
            raise ValueError(f"Expected [batch,{self.history_frames},3,3,height,width] RGB history")
        batch, steps = images.shape[:2]
        if state.shape != (batch, steps, 31) or velocity.shape != state.shape:
            raise ValueError("Unexpected state or velocity shape")
        if contact_sites.shape != (batch, steps, self.contact_site_dim):
            raise ValueError("Unexpected finger contact-site shape")
        pixels = torch.nn.functional.interpolate(
            images.reshape(batch * steps * 3, 3, *images.shape[-2:]).float(),
            size=(64, 96),
            mode="bilinear",
            align_corners=False,
        )
        view = (
            self.encoder(pixels).reshape(batch, steps, 3, -1)
            + self.view_embedding[None, None]
        )
        vision = self.vision(view.flatten(2))
        words = self.text(token_ids)
        word_mask = token_mask.to(words.dtype).unsqueeze(-1)
        text = self.text_context((words * word_mask).sum(1) / word_mask.sum(1).clamp_min(1.0))
        proprio = self.proprio(torch.cat((state, velocity, contact_sites, tactile_contact), dim=-1))
        temporal_input = self.temporal_input(
            torch.cat((vision, text[:, None].expand(-1, steps, -1), proprio), dim=-1)
        )
        context = self.temporal(temporal_input)[0][:, -1]
        object_mean = self.object_head(context)
        arm_logits = self.arm_head(context)
        token_logits = self.token_head(
            torch.cat((context, object_mean, arm_logits.softmax(-1)), dim=-1)
        )
        return {
            "action_token_logits": token_logits,
            "candidate_action_residual": torch.zeros(
                batch,
                self.token_count,
                self.action_chunk,
                ACTION_DIM,
                dtype=context.dtype,
                device=context.device,
            ),
            "object_mean": object_mean,
            "target_logits": self.target_head(text),
            "active_arm_logits": arm_logits,
        }


def save_checkpoint(
    path: Path,
    model: ActionTokenVLA,
    vocab: dict[str, int],
    stats: dict[str, torch.Tensor],
    action_codebook: torch.Tensor,
    future_offsets: tuple[int, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "stats": {key: value.detach().cpu() for key, value in stats.items()},
            "action_codebook": action_codebook.detach().cpu(),
            "future_offsets": list(future_offsets),
            "config": {
                "vocab_size": model.vocab_size,
                "token_count": model.token_count,
                "contact_site_dim": model.contact_site_dim,
                "history_frames": model.history_frames,
                "action_chunk": model.action_chunk,
                "hidden_dim": model.hidden_dim,
                "use_temporal_history": model.use_temporal_history,
            },
            "action_representation": "discrete_strict_expert_relative_joint_action_chunks",
            "input_contract": "causal temporal raw RGB x3 + language + state31 + qvel31 + measured finger sites + tactile",
            "privileged_positions_at_inference": False,
        },
        path,
    )


def load_checkpoint(path: Path, device: str):
    value = torch.load(path, map_location=device, weights_only=False)
    model_state = value["model"]
    if "temporal_input.0.weight" in model_state and "action_residual.0.weight" not in model_state:
        model = LegacyTemporalActionTokenVLA(**value["config"]).to(device)
    else:
        config = dict(value["config"])
        config.setdefault("use_temporal_history", "temporal.weight_ih_l0" in model_state)
        model = ActionTokenVLA(**config).to(device)
    model.load_state_dict(value["model"])
    model.eval()
    return (
        model,
        value["vocab"],
        {key: item.to(device) for key, item in value["stats"].items()},
        value["action_codebook"].to(device),
        tuple(int(item) for item in value["future_offsets"]),
    )
