"""Direct, zero-initialized language-to-action conditioning for SmolVLA.

The adapter is deliberately kept in the project tree so experiments do not
silently modify the checked-out LeRobot dependency.  It pools the valid input
token embeddings, projects them to the action-expert width, and adds the
result to every action suffix token.  The final projection starts at zero, so
attaching it preserves the source policy exactly until training changes it.
"""

from __future__ import annotations

import types

import torch
from torch import nn


ADAPTER_NAME = "language_action_adapter"


class LanguageActionAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_dim: int,
        chunk_size: int,
        bottleneck: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.bottleneck = bottleneck
        self.input_proj = nn.Linear(input_dim, bottleneck)
        self.output_proj = nn.Linear(bottleneck, output_dim)
        self.action_output_proj = nn.Linear(output_dim, action_dim)
        self.action_film_proj = nn.Linear(output_dim, output_dim * 2)
        self.target_classifier = nn.Linear(output_dim, 3)
        # Expand the learned language-to-target posterior over the full action
        # chunk. The zero initialization preserves the source policy on attach.
        self.target_action_chunk_proj = nn.Linear(3, chunk_size * action_dim)
        # A hidden-dependent basis lets the target posterior modulate the
        # action expert's state/time features instead of supplying only a
        # fixed trajectory offset.
        self.target_action_hidden_proj = nn.Linear(output_dim, 3 * action_dim)
        self.target_suffix_proj = nn.Linear(3, output_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.action_output_proj.weight)
        nn.init.zeros_(self.action_output_proj.bias)
        nn.init.zeros_(self.action_film_proj.weight)
        nn.init.zeros_(self.action_film_proj.bias)
        nn.init.zeros_(self.target_action_chunk_proj.weight)
        nn.init.zeros_(self.target_action_chunk_proj.bias)
        nn.init.zeros_(self.target_action_hidden_proj.weight)
        nn.init.zeros_(self.target_action_hidden_proj.bias)
        nn.init.zeros_(self.target_suffix_proj.weight)
        nn.init.zeros_(self.target_suffix_proj.bias)

    def forward(self, language_embedding: torch.Tensor) -> torch.Tensor:
        parameter_dtype = self.input_proj.weight.dtype
        language_embedding = language_embedding.to(dtype=parameter_dtype)
        return self.output_proj(torch.tanh(self.input_proj(language_embedding)))

    def project_action(self, condition: torch.Tensor) -> torch.Tensor:
        return self.action_output_proj(condition.to(dtype=self.action_output_proj.weight.dtype))

    def classify_target(self, condition: torch.Tensor) -> torch.Tensor:
        return self.target_classifier(condition.to(dtype=self.target_classifier.weight.dtype))

    def project_target_action_chunk(self, target_logits: torch.Tensor) -> torch.Tensor:
        target_probabilities = target_logits.softmax(dim=-1)
        projected = self.target_action_chunk_proj(
            target_probabilities.to(dtype=self.target_action_chunk_proj.weight.dtype)
        )
        return projected.reshape(
            projected.shape[0], self.chunk_size, self.action_dim
        )

    def project_target_action_hidden(
        self, hidden: torch.Tensor, target_logits: torch.Tensor
    ) -> torch.Tensor:
        target_probabilities = target_logits.softmax(dim=-1)
        basis = self.target_action_hidden_proj(
            hidden.to(dtype=self.target_action_hidden_proj.weight.dtype)
        ).reshape(hidden.shape[0], hidden.shape[1], 3, self.action_dim)
        return (
            basis
            * target_probabilities[:, None, :, None].to(dtype=basis.dtype)
        ).sum(dim=2)

    def project_target_suffix(self, target_logits: torch.Tensor) -> torch.Tensor:
        target_probabilities = target_logits.softmax(dim=-1)
        return self.target_suffix_proj(
            target_probabilities.to(dtype=self.target_suffix_proj.weight.dtype)
        )

    def modulate_action_hidden(
        self, hidden: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        film = self.action_film_proj(
            condition.to(dtype=self.action_film_proj.weight.dtype)
        )
        scale, shift = film.chunk(2, dim=-1)
        return hidden * (1.0 + scale[:, None, :].to(hidden.dtype)) + shift[
            :, None, :
        ].to(hidden.dtype)


def attach_language_action_adapter(policy, *, bottleneck: int = 256):
    """Attach the adapter and patch both training and inference model paths."""
    model = policy.model
    if hasattr(model, ADAPTER_NAME):
        return getattr(model, ADAPTER_NAME)

    vlm = model.vlm_with_expert
    hidden_dim = int(vlm.config.text_config.hidden_size)
    expert_dim = int(vlm.expert_hidden_size)
    adapter = LanguageActionAdapter(
        hidden_dim,
        expert_dim,
        int(model.config.max_action_dim),
        int(model.config.chunk_size),
        bottleneck,
    )
    # The VLM is loaded with device_map="auto"; newly attached modules stay on
    # CPU unless explicitly placed alongside the action expert.
    expert_device = next(model.action_in_proj.parameters()).device
    adapter = adapter.to(device=expert_device)
    setattr(model, ADAPTER_NAME, adapter)

    original_prefix = model.embed_prefix
    original_suffix = model.embed_suffix

    def embed_prefix_with_language_context(self, images, img_masks, lang_tokens, lang_masks, state=None):
        language_embeddings = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        mask = lang_masks.to(dtype=language_embeddings.dtype).unsqueeze(-1)
        pooled = (language_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        self._language_action_condition = self.language_action_adapter(pooled)
        self._language_action_target_logits = (
            self.language_action_adapter.classify_target(
                self._language_action_condition
            )
        )
        return self._language_action_original_prefix(images, img_masks, lang_tokens, lang_masks, state)

    def embed_suffix_with_language_context(self, noisy_actions, timestep):
        suffix_embs, pad_masks, att_masks = self._language_action_original_suffix(noisy_actions, timestep)
        condition = getattr(self, "_language_action_condition", None)
        if condition is not None:
            suffix_embs = suffix_embs + condition[:, None, :].to(dtype=suffix_embs.dtype)
        target_logits = getattr(model, "_language_action_target_logits", None)
        if target_logits is not None:
            target_suffix = model.language_action_adapter.project_target_suffix(
                target_logits
            )
            suffix_embs = suffix_embs + target_suffix[:, None, :].to(
                dtype=suffix_embs.dtype
            )
        return suffix_embs, pad_masks, att_masks

    model._language_action_original_prefix = original_prefix
    model._language_action_original_suffix = original_suffix
    model.embed_prefix = types.MethodType(embed_prefix_with_language_context, model)
    model.embed_suffix = types.MethodType(embed_suffix_with_language_context, model)

    def add_direct_language_action_residual(_module, _inputs, output):
        condition = getattr(model, "_language_action_condition", None)
        if condition is None:
            return output
        action_residual = model.language_action_adapter.project_action(condition)
        target_logits = getattr(model, "_language_action_target_logits", None)
        if target_logits is None:
            return output + action_residual[:, None, :].to(dtype=output.dtype)
        target_chunk_residual = model.language_action_adapter.project_target_action_chunk(
            target_logits
        )
        target_hidden_residual = model.language_action_adapter.project_target_action_hidden(
            _inputs[0], target_logits
        )
        return output + action_residual[:, None, :].to(
            dtype=output.dtype
        ) + target_chunk_residual.to(
            dtype=output.dtype, device=output.device
        ) + target_hidden_residual.to(dtype=output.dtype, device=output.device)

    model._language_action_output_hook = model.action_out_proj.register_forward_hook(
        add_direct_language_action_residual
    )

    def apply_language_action_film(_module, inputs):
        condition = getattr(model, "_language_action_condition", None)
        if condition is None:
            return inputs
        return (
            model.language_action_adapter.modulate_action_hidden(
                inputs[0], condition
            ),
        )

    model._language_action_film_hook = model.action_out_proj.register_forward_pre_hook(
        apply_language_action_film
    )
    return adapter


def adapter_parameter_names(policy) -> list[str]:
    return [name for name, _ in policy.named_parameters() if f"{ADAPTER_NAME}." in name]
