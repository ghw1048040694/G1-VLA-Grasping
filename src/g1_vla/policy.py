"""Continuous policy with no simulator or historical-script dependencies.

Parameter names/shapes preserve the historical continuous model. Metadata tools
remain importable without PyTorch; legacy weights require an explicit baseline
role and cannot silently become an audited rebuild checkpoint.
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from .contracts import (ActionContract, ContractError, canonical_sha256,
                        make_checkpoint_metadata as _make_metadata,
                        validate_checkpoint_metadata as _validate_metadata)

ACTION_DIM, ARM_COUNT, TARGET_COUNT, PHASE_COUNT = 31, 2, 3, 8
HAND_INDICES = (10, 11, 12, 13, 14, 15, 16, 24, 25, 26, 27, 28, 29, 30)
ARM_INDICES = tuple(index for index in range(ACTION_DIM) if index not in HAND_INDICES)
TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
MODEL_CONFIG_KEYS = ("vocab_size", "contact_site_dim", "history_frames", "action_chunk", "hidden_dim", "factorized_actions")
STATS_DIMS = {"state_mean": 31, "state_std": 31, "velocity_mean": 31, "velocity_std": 31,
              "contact_sites_mean": None, "contact_sites_std": None,
              "action_delta_mean": 31, "action_delta_std": 31}
_Module = nn.Module if nn is not None else object
CheckpointContractError = ContractError


def require_torch():
    if torch is None:
        raise RuntimeError("PyTorch is required for model execution; metadata audits remain available")
    return torch


def plain(value: Any) -> Any:
    if hasattr(value, "detach"):
        return plain(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ContractError("Metadata and normalization must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ContractError(f"Unsupported metadata value {type(value).__name__}")


def fingerprint(value: Any) -> str:
    return canonical_sha256(plain(value))


def validate_vocab(vocab: Mapping[str, int]) -> None:
    if not isinstance(vocab, Mapping) or vocab.get(PAD_TOKEN) != 0 or UNK_TOKEN not in vocab:
        raise ContractError("Vocabulary requires <pad>=0 and <unk>")
    values = list(vocab.values())
    if any(type(item) is not int for item in values) or sorted(values) != list(range(len(values))):
        raise ContractError("Vocabulary token IDs must be unique and contiguous")


def ordered_vocabulary(vocab: Mapping[str, int]) -> list[str]:
    validate_vocab(vocab)
    return sorted(vocab, key=vocab.__getitem__)


def validate_stats(stats: Mapping[str, Any], contact_site_dim: int | None = None) -> None:
    if not isinstance(stats, Mapping) or set(stats) != set(STATS_DIMS):
        raise ContractError("Normalization must contain exactly the eight continuous policy statistics")
    inferred = None
    for key, expected in STATS_DIMS.items():
        values = plain(stats[key])
        if not isinstance(values, list) or not values or any(type(item) not in (int, float) for item in values):
            raise ContractError(f"{key} must be a finite one-dimensional vector")
        if expected is not None and len(values) != expected:
            raise ContractError(f"{key} must contain {expected} values")
        if key.startswith("contact_sites"):
            if inferred is None:
                inferred = len(values)
            if len(values) != inferred or (contact_site_dim is not None and len(values) != contact_site_dim):
                raise ContractError("Contact-site normalization dimension disagrees with the model")
        if key.endswith("_std") and any(item <= 0 for item in values):
            raise ContractError(f"{key} must be strictly positive")


def validate_model_config(config: Mapping[str, Any], *, historical=False) -> None:
    required = set(MODEL_CONFIG_KEYS) - ({"factorized_actions"} if historical else set())
    if not isinstance(config, Mapping) or required - set(config) or set(config) - set(MODEL_CONFIG_KEYS):
        raise ContractError("Incomplete or unknown continuous model configuration")
    for key in set(MODEL_CONFIG_KEYS) - {"factorized_actions"}:
        if type(config[key]) is not int or config[key] <= 0:
            raise ContractError(f"Model {key} must be a positive integer")
    if "factorized_actions" in config and type(config["factorized_actions"]) is not bool:
        raise ContractError("Model factorized_actions must be a boolean")


def validate_model_metadata(config: Mapping[str, Any], metadata) -> None:
    validate_model_config(config)
    contract = contract_from_metadata(metadata)
    if contract.history_steps != config["history_frames"] or contract.chunk_steps != config["action_chunk"]:
        raise ContractError("Model temporal dimensions differ from the action contract")
    if metadata["observation_spec"]["finger_site_dimension"] != config["contact_site_dim"]:
        raise ContractError("Model contact-site dimension differs from the observation specification")


def contract_from_metadata(metadata: Mapping[str, Any]) -> ActionContract:
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("contract"), Mapping):
        raise ContractError("Checkpoint requires an explicit action contract")
    source = metadata["contract"]
    names = ("joint_names", "control_hz", "physics_dt_s", "max_anchor_delta_rad", "max_velocity_rad_s",
             "protocol", "history_steps", "chunk_steps", "execute_steps", "anchor",
             "body_gain_scale", "hand_gain_scale", "dynamics_profile", "scene_protocol")
    try:
        return ActionContract(**{key: source[key] for key in names})
    except (KeyError, TypeError) as exc:
        raise ContractError("Incomplete checkpoint action contract") from exc


def make_checkpoint_metadata(contract: ActionContract, vocab, stats, *, observation_spec) -> dict:
    validate_stats(stats)
    return _make_metadata(contract, ordered_vocabulary(vocab), plain(stats), observation_spec=observation_spec)


def validate_checkpoint_metadata(metadata, *, vocab, stats, expected_metadata=None) -> None:
    validate_stats(stats)
    contract = contract_from_metadata(metadata)
    _validate_metadata(dict(metadata), contract, ordered_vocabulary(vocab), plain(stats),
                       observation_spec=metadata.get("observation_spec"))
    if expected_metadata is not None:
        # Compare the entire envelope; vocab/stat/observation/control changes
        # cannot pass by merely preserving compatible tensor dimensions.
        if fingerprint(metadata) != fingerprint(expected_metadata):
            raise ContractError("Checkpoint action, vocabulary, normalization or observation contract mismatch")


def validate_data_provenance(provenance) -> None:
    if not isinstance(provenance, Mapping) or set(provenance) != {"manifest_sha256", "audit_sha256"}:
        raise ContractError("Checkpoint requires audited dataset provenance")
    for key, value in provenance.items():
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ContractError(f"{key} must be a SHA-256 digest")


def validate_warm_start(checkpoint: Mapping[str, Any], vocab, stats, expected_metadata) -> None:
    if not isinstance(expected_metadata, Mapping):
        raise ContractError("Warm-start requires the current audited action and observation metadata")
    if "metadata" not in checkpoint:
        raise ContractError("Historical checkpoints are regression baselines; training requires audited metadata")
    validate_checkpoint_metadata(checkpoint["metadata"], vocab=checkpoint.get("vocab"),
                                 stats=checkpoint.get("stats"), expected_metadata=expected_metadata)
    validate_data_provenance(checkpoint.get("data_provenance"))
    validate_vocab(vocab)
    validate_stats(stats)
    if fingerprint(vocab) != fingerprint(checkpoint["vocab"]):
        raise ContractError("Warm-start vocabulary token mapping mismatch")
    if fingerprint(stats) != fingerprint(checkpoint["stats"]):
        raise ContractError("Warm-start normalization mismatch; fixed statistics or verified weight conversion required")


def encode_tasks(tasks: list[str], vocab: dict[str, int], max_tokens: int = 16):
    require_torch()
    validate_vocab(vocab)
    if type(max_tokens) is not int or max_tokens <= 0 or any(not isinstance(task, str) for task in tasks):
        raise ValueError("Task encoding requires strings and a positive token length")
    ids = torch.zeros((len(tasks), max_tokens), dtype=torch.long)
    mask = torch.zeros((len(tasks), max_tokens), dtype=torch.bool)
    for row, task in enumerate(tasks):
        values = [vocab.get(token, vocab[UNK_TOKEN]) for token in TOKEN_RE.findall(task.lower())[:max_tokens]]
        if values:
            ids[row, :len(values)] = torch.tensor(values, dtype=torch.long)
            mask[row, :len(values)] = True
    return ids, mask


@dataclass(frozen=True)
class TensorObservation:
    """Observed tensors only; no simulator, object truth or target override."""
    images: Any
    state: Any
    velocity: Any
    sites: Any
    tactile: Any
    ids: Any
    mask: Any

    def model_args(self):
        require_torch()
        values = (self.images, self.state, self.velocity, self.sites, self.tactile, self.ids, self.mask)
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError("Policy observation fields must all be tensors")
        return values


def validate_tensor_observation(observation: TensorObservation, vocab_size: int) -> None:
    """Reject corrupt sensor values before the network can emit NaN actions."""
    values = observation.model_args()
    for name, value in zip(("images", "state", "velocity", "sites", "tactile"), values[:5]):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain finite floating-point values")
    if observation.images.numel() == 0 or ((observation.images < 0) | (observation.images > 1)).any():
        raise ValueError("Observed RGB must be nonempty and normalized to [0,1]")
    if ((observation.tactile < 0) | (observation.tactile > 1)).any():
        raise ValueError("Observed tactile values must lie in [0,1]")
    if observation.ids.dtype != torch.long or observation.mask.dtype != torch.bool:
        raise ValueError("Language requires int64 token IDs and a boolean mask")
    if observation.ids.numel() == 0 or ((observation.ids < 0) | (observation.ids >= vocab_size)).any():
        raise ValueError("Language token IDs lie outside the checkpoint vocabulary")


class ViewEncoder(_Module):
    def __init__(self, output_dim: int = 192):
        require_torch()
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(12, 96), nn.SiLU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GroupNorm(16, 128), nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(), nn.Linear(128 * 3 * 4, output_dim), nn.SiLU(),
        )

    def forward(self, images):
        return self.network(images)


class ContinuousDaggerVLA(_Module):
    def __init__(self, vocab_size: int, contact_site_dim: int = 108, history_frames: int = 4,
                 action_chunk: int = 8, hidden_dim: int = 384, factorized_actions: bool = False):
        require_torch()
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
        def action_decoder(size):
            return nn.Sequential(nn.Linear(hidden_dim + ARM_COUNT + PHASE_COUNT, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, self.action_chunk * size))
        if self.factorized_actions:
            self.arm_action_head = action_decoder(len(ARM_INDICES))
            self.hand_action_head = action_decoder(len(HAND_INDICES))
        else:
            self.action_head = action_decoder(ACTION_DIM)

    def forward_observation(self, observation: TensorObservation):
        if not isinstance(observation, TensorObservation):
            raise TypeError("Policy requires TensorObservation")
        return self(*observation.model_args())

    def forward(self, images, state, velocity, sites, tactile, ids, mask):
        if any(not isinstance(value, torch.Tensor) for value in (images, state, velocity, sites, tactile, ids, mask)):
            raise TypeError("Policy accepts observed tensors only")
        if images.ndim != 6 or images.shape[1] != self.history_frames or images.shape[2:4] != (3, 3):
            raise ValueError(f"Expected [B,{self.history_frames},3,3,H,W], got {tuple(images.shape)}")
        batch, steps = images.shape[:2]
        if state.shape != (batch, steps, 31) or velocity.shape != state.shape:
            raise ValueError("Expected synchronized state31 and velocity31 histories")
        if sites.shape != (batch, steps, self.contact_site_dim) or tactile.shape != (batch, steps, 1):
            raise ValueError("Unexpected observed contact-site or tactile history")
        if ids.ndim != 2 or len(ids) != batch or mask.shape != ids.shape:
            raise ValueError("Unexpected language tensor shape")
        validate_tensor_observation(TensorObservation(images, state, velocity, sites, tactile, ids, mask), self.vocab_size)
        pixels = torch.nn.functional.interpolate(images.reshape(batch * steps * 3, 3, *images.shape[-2:]).float(),
                                                size=(64, 96), mode="bilinear", align_corners=False)
        view = self.encoder(pixels).reshape(batch, steps, 3, -1) + self.view_embedding[None, None]
        vision = self.vision(view.flatten(2))
        words = self.text(ids)
        text_mask = mask.to(words.dtype).unsqueeze(-1)
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
        return {"action": action, "target_logits": self.target_head(text), "active_arm_logits": arm_logits,
                "phase_logits": phase_logits, "distance": self.distance_head(context),
                "contact_logits": self.contact_head(context)}


def save_checkpoint(path: Path, model: ContinuousDaggerVLA, vocab, stats, metadata,
                    *, data_provenance, training_state: Mapping[str, Any] | None = None) -> None:
    require_torch()
    validate_checkpoint_metadata(metadata, vocab=vocab, stats=stats)
    validate_data_provenance(data_provenance)
    validate_stats(stats, model.contact_site_dim)
    if len(vocab) != model.vocab_size:
        raise ContractError("Model embedding size differs from vocabulary")
    model_config = {key: getattr(model, key) for key in MODEL_CONFIG_KEYS}
    validate_model_metadata(model_config, metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"model": model.state_dict(), "vocab": dict(vocab),
             "stats": {key: item.detach().cpu() for key, item in stats.items()},
             "config": model_config,
             "metadata": copy.deepcopy(dict(metadata)), "data_provenance": dict(data_provenance),
             "training_state": training_state}
    with path.open("xb") as stream:
        torch.save(value, stream)


def load_checkpoint(path: Path, device: str = "cpu", *, expected_metadata=None, historical_baseline: bool = False):
    """The shared training/evaluation loader; legacy use must be explicit."""
    require_torch()
    value = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(value, Mapping) or any(key not in value for key in ("config", "model", "vocab", "stats")):
        raise ContractError("Invalid continuous checkpoint")
    config = value["config"]
    validate_model_config(config, historical="metadata" not in value)
    validate_vocab(value["vocab"])
    validate_stats(value["stats"], config.get("contact_site_dim", 108))
    if "metadata" in value:
        validate_checkpoint_metadata(value["metadata"], vocab=value["vocab"], stats=value["stats"],
                                     expected_metadata=expected_metadata)
        validate_data_provenance(value.get("data_provenance"))
        validate_model_metadata(config, value["metadata"])
    elif not historical_baseline:
        raise ContractError("Historical checkpoint requires historical_baseline=True; new protocol metadata is absent")
    elif expected_metadata is not None:
        raise ContractError("A historical baseline cannot satisfy a rebuild deployment contract")
    else:
        value = dict(value)
        value["load_role"] = "historical_regression_baseline"
    model = ContinuousDaggerVLA(**config).to(device)
    if len(value["vocab"]) != model.vocab_size:
        raise ContractError("Checkpoint vocabulary differs from model embedding size")
    model.load_state_dict(value["model"], strict=True)
    model.eval()
    return model, value["vocab"], {key: item.to(device) for key, item in value["stats"].items()}, value
