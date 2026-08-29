"""Compact multimodal G1 policy used for the learned multi-object baseline.

The policy keeps the input/output contract explicit: three RGB views, tokenized
language, and the 31-dimensional upper-body state produce a future action
chunk. Auxiliary object-position, target, and phase heads make visual grounding
observable during training without feeding privileged positions at inference.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


CAMERA_KEYS = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
OBJECT_NAMES = ("red_triangle", "yellow_rod", "green_cube")
TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def build_vocab(tasks: list[str]) -> dict[str, int]:
    words = sorted({word for task in tasks for word in TOKEN_RE.findall(task.lower())})
    tokens = [PAD_TOKEN, UNK_TOKEN, *words]
    return {token: index for index, token in enumerate(tokens)}


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


class FrozenResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        if image.shape[-2:] != (128, 128):
            image = torch.nn.functional.interpolate(
                image, size=(128, 128), mode="bilinear", align_corners=False
            )
        image = (image - self.mean) / self.std
        return self.features(image).flatten(1)


class SpatialFrozenResNet18(nn.Module):
    """Frozen ResNet18 features retaining the 4x4 spatial grid."""

    output_dim = 512 * 4 * 4

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        if image.shape[-2:] != (128, 128):
            image = torch.nn.functional.interpolate(
                image, size=(128, 128), mode="bilinear", align_corners=False
            )
        image = (image - self.mean) / self.std
        return self.features(image).flatten(1)


class CompactG1VLA(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        chunk_horizon: int = 20,
        state_dim: int = 31,
        action_dim: int = 31,
        use_target_conditioning: bool = False,
        use_target_mixture: bool = False,
        action_is_delta: bool = False,
        use_phase_action_conditioning: bool = True,
        use_active_arm_conditioning: bool = False,
        use_color_features: bool = False,
        spatial_vision: bool = False,
        use_temporal_history: bool = False,
        history_length: int = 4,
        use_palm_conditioning: bool = False,
        use_palm_pose_conditioning: bool = False,
        use_visual_grasp_residual: bool = False,
    ):
        super().__init__()
        self.chunk_horizon = chunk_horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_target_conditioning = use_target_conditioning
        self.use_target_mixture = use_target_mixture
        self.action_is_delta = action_is_delta
        self.use_phase_action_conditioning = use_phase_action_conditioning
        self.use_active_arm_conditioning = use_active_arm_conditioning
        self.use_color_features = use_color_features
        self.spatial_vision = spatial_vision
        self.use_temporal_history = use_temporal_history
        self.history_length = max(1, int(history_length))
        self.use_palm_conditioning = use_palm_conditioning
        self.use_palm_pose_conditioning = use_palm_pose_conditioning
        self.use_visual_grasp_residual = use_visual_grasp_residual
        self.vision = SpatialFrozenResNet18() if spatial_vision else FrozenResNet18()
        vision_dim = SpatialFrozenResNet18.output_dim if spatial_vision else 512
        self.text = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.visual_fusion = nn.Sequential(
            nn.Linear(vision_dim * len(CAMERA_KEYS), 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        self.state_fusion = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.GELU()
        )
        color_dim = 64 if use_color_features else 0
        if use_color_features:
            self.color_fusion = nn.Sequential(
                nn.Linear(9, 64), nn.LayerNorm(64), nn.GELU()
            )
        history_dim = 0
        if use_temporal_history:
            self.history_state = nn.Sequential(
                nn.Linear(state_dim, 64), nn.LayerNorm(64), nn.GELU()
            )
            self.history_color = (
                nn.Sequential(nn.Linear(9, 32), nn.LayerNorm(32), nn.GELU())
                if use_color_features
                else None
            )
            temporal_input_dim = 64 + (32 if use_color_features else 0)
            self.temporal_gru = nn.GRU(
                temporal_input_dim, 128, batch_first=True
            )
            history_dim = 128
        self.context = nn.Sequential(
            nn.Linear(256 + 128 + 128 + color_dim + history_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.object_head = nn.Linear(512, 9)
        self.palm_head = nn.Linear(512, 6)
        self.palm_orientation_head = nn.Linear(512, 12)
        self.target_head = nn.Linear(128, 3)
        self.phase_head = nn.Linear(512, 8)
        self.active_arm_head = nn.Linear(512, 2)
        self.active_arm_action_embedding = nn.Embedding(2, 32)
        if use_target_conditioning:
            self.target_action_embedding = nn.Embedding(3, 96)
            self.phase_action_embedding = nn.Embedding(8, 64)
            action_condition_dim = (
                96
                + (64 if use_phase_action_conditioning else 0)
                + (32 if use_active_arm_conditioning else 0)
            )
        else:
            action_condition_dim = 0
        phase_action_dim = 8 if use_phase_action_conditioning else 0
        active_arm_action_dim = 2 if use_active_arm_conditioning else 0
        action_input_dim = 512 + 9 + 3 + phase_action_dim + active_arm_action_dim + action_condition_dim
        if use_palm_conditioning:
            self.palm_action_embedding = nn.Sequential(
                nn.Linear(6, 64), nn.LayerNorm(64), nn.GELU()
            )
            action_input_dim += 6 + 64
        if use_palm_pose_conditioning:
            self.palm_orientation_action_embedding = nn.Sequential(
                nn.Linear(12, 64), nn.LayerNorm(64), nn.GELU()
            )
            action_input_dim += 12 + 64
        if use_visual_grasp_residual:
            self.visual_grasp_residual_fusion = nn.Sequential(
                nn.Linear(6, 64), nn.LayerNorm(64), nn.GELU()
            )
            action_input_dim += 6 + 64
        def make_action_head():
            return nn.Sequential(
                nn.Linear(action_input_dim, 768),
                nn.LayerNorm(768),
                nn.GELU(),
                nn.Linear(768, chunk_horizon * action_dim),
            )
        if use_target_mixture:
            self.action_heads = nn.ModuleList([make_action_head() for _ in range(3)])
        else:
            self.action_head = make_action_head()

    def forward(
        self,
        vision_embeddings: torch.Tensor,
        state: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        target_override: torch.Tensor | None = None,
        phase_override: torch.Tensor | None = None,
        active_arm_override: torch.Tensor | None = None,
        color_features: torch.Tensor | None = None,
        history_state: torch.Tensor | None = None,
        history_color_features: torch.Tensor | None = None,
        object_position_override: torch.Tensor | None = None,
        palm_position_override: torch.Tensor | None = None,
        palm_orientation_override: torch.Tensor | None = None,
        visual_grasp_residual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        visual = self.visual_fusion(vision_embeddings.flatten(1))
        text_embeddings = self.text(token_ids)
        mask = token_mask.to(dtype=text_embeddings.dtype).unsqueeze(-1)
        text_features = (text_embeddings * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        state_features = self.state_fusion(state)
        context_features = (visual, text_features, state_features)
        if self.use_color_features:
            if color_features is None:
                color_features = torch.zeros(
                    (state.shape[0], 9), device=state.device, dtype=state.dtype
                )
            context_features = (*context_features, self.color_fusion(color_features))
        if self.use_temporal_history:
            if history_state is None:
                history_state = state[:, None, :].expand(-1, self.history_length, -1)
            if history_state.ndim != 3 or history_state.shape[1] != self.history_length:
                raise ValueError(
                    f"Expected history_state [B,{self.history_length},D], got {tuple(history_state.shape)}"
                )
            history_parts = [self.history_state(history_state)]
            if self.use_color_features:
                if history_color_features is None:
                    history_color_features = color_features[:, None, :].expand(
                        -1, self.history_length, -1
                    )
                if history_color_features.ndim != 3 or history_color_features.shape[1] != self.history_length:
                    raise ValueError(
                        "history_color_features must have shape "
                        f"[B,{self.history_length},9]"
                    )
                history_parts.append(self.history_color(history_color_features))
            temporal_input = torch.cat(history_parts, dim=-1)
            _, temporal_hidden = self.temporal_gru(temporal_input)
            context_features = (*context_features, temporal_hidden[-1])
        context = self.context(torch.cat(context_features, dim=-1))
        predicted_object_position = self.object_head(context)
        predicted_palm_position = self.palm_head(context)
        predicted_palm_orientation = self.palm_orientation_head(context)
        object_position = (
            object_position_override
            if object_position_override is not None
            else predicted_object_position
        )
        target_logits = self.target_head(text_features)
        phase_logits = self.phase_head(context)
        target_probabilities = (
            target_override
            if target_override is not None
            else target_logits.softmax(-1)
        )
        phase_probabilities = (
            phase_override
            if phase_override is not None
            else phase_logits.softmax(-1)
        )
        active_arm_logits = self.active_arm_head(context)
        active_arm_probabilities = (
            active_arm_override
            if active_arm_override is not None
            else active_arm_logits.softmax(-1)
        )
        action_features = (context, object_position, target_probabilities)
        palm_position = (
            palm_position_override
            if palm_position_override is not None
            else predicted_palm_position
        )
        if self.use_palm_conditioning:
            action_features = (
                *action_features,
                palm_position,
                self.palm_action_embedding(palm_position),
            )
        palm_orientation = (
            palm_orientation_override
            if palm_orientation_override is not None
            else predicted_palm_orientation
        )
        if self.use_palm_pose_conditioning:
            action_features = (
                *action_features,
                palm_orientation,
                self.palm_orientation_action_embedding(palm_orientation),
            )
        if self.use_visual_grasp_residual:
            if visual_grasp_residual is None:
                visual_grasp_residual = torch.zeros(
                    (state.shape[0], 6), device=state.device, dtype=state.dtype
                )
            if visual_grasp_residual.shape != (state.shape[0], 6):
                raise ValueError(
                    "visual_grasp_residual must have shape "
                    f"[B,6], got {tuple(visual_grasp_residual.shape)}"
                )
            action_features = (
                *action_features,
                visual_grasp_residual,
                self.visual_grasp_residual_fusion(visual_grasp_residual),
            )
        if self.use_phase_action_conditioning:
            action_features = (*action_features, phase_probabilities)
        if self.use_active_arm_conditioning:
            action_features = (*action_features, active_arm_probabilities)
        action_input = torch.cat(action_features, dim=-1)
        if self.use_target_conditioning:
            target_condition = target_probabilities @ self.target_action_embedding.weight
            conditions = [target_condition]
            if self.use_phase_action_conditioning:
                conditions.append(phase_probabilities @ self.phase_action_embedding.weight)
            if self.use_active_arm_conditioning:
                conditions.append(
                    active_arm_probabilities @ self.active_arm_action_embedding.weight
                )
            action_input = torch.cat((action_input, *conditions), dim=-1)
        if self.use_target_mixture:
            head_actions = torch.stack(
                [head(action_input) for head in self.action_heads], dim=1
            ).reshape(-1, 3, self.chunk_horizon, self.action_dim)
            action = (head_actions * target_probabilities[:, :, None, None]).sum(1)
        else:
            action = self.action_head(action_input).reshape(
                -1, self.chunk_horizon, self.action_dim
            )
        return {
            "action": action,
            "object_position": predicted_object_position,
            "action_object_position": object_position,
            "palm_position": predicted_palm_position,
            "action_palm_position": palm_position,
            "palm_orientation": predicted_palm_orientation,
            "action_palm_orientation": palm_orientation,
            "target_logits": target_logits,
            "phase_logits": phase_logits,
            "active_arm_logits": active_arm_logits,
        }


def save_checkpoint(
    path: Path,
    model: CompactG1VLA,
    vocab: dict[str, int],
    stats: dict[str, torch.Tensor],
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": vocab,
            "stats": {key: value.cpu() for key, value in stats.items()},
            "config": config,
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cuda"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = CompactG1VLA(**config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = {
        "active_arm_head.weight",
        "active_arm_head.bias",
        "active_arm_action_embedding.weight",
        "palm_head.weight",
        "palm_head.bias",
        "palm_orientation_head.weight",
        "palm_orientation_head.bias",
    }
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    stats = {key: value.to(device) for key, value in checkpoint["stats"].items()}
    return model, checkpoint["vocab"], stats
