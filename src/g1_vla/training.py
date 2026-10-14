"""Auditable continuous training primitives and a conditional readiness gate.

There is deliberately no optimizer loop in this preparation revision. Dataset,
normalization and loss code are shared package functions, ready for the audited
builder. Historical caches must be rebuilt: commanded targets must be aligned
with the next real control command and never rewritten by teacher geometry.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Mapping

from .config import validate_config
from .contracts import ActionContract, ContractError
from .policy import (ARM_INDICES, HAND_INDICES, TensorObservation, encode_tasks, fingerprint,
                     plain, require_torch, validate_checkpoint_metadata,
                     validate_data_provenance, validate_stats, validate_warm_start)

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


class TrainingNotReady(RuntimeError):
    """Training remains disabled or the audited pipeline is incomplete."""


def _numpy():
    if np is None:
        raise RuntimeError("NumPy is required for dataset image loading and RNG restoration")
    return np


def _valid_lengths(cache) -> list[int]:
    require_torch()
    valid = cache["valid_frame"].bool()
    if valid.ndim != 2:
        raise ContractError("valid_frame must have shape [episodes,frames]")
    lengths = []
    for row in valid:
        length = int(row.sum())
        if not row[:length].all() or row[length:].any():
            raise ContractError("Episode valid frames must form one contiguous prefix")
        lengths.append(length)
    return lengths


def action_label(cache, episode: int, frame: int, contract: ActionContract):
    """Encode the next-command sequence against one observed planning anchor."""
    require_torch()
    anchor = cache["state"][episode, frame].float()
    targets = cache["command_target"][episode, frame:frame + contract.chunk_steps].float()
    encoded = contract.encode_chunk(anchor.tolist(), targets.tolist(), joint_names=cache["joint_names"])
    return torch.tensor(encoded, dtype=torch.float32)


class ContinuousDataset(Dataset):
    """Audited raw observations + offline labels; object truth is never an input.

    Required cache arrays: state, velocity, contact_sites, contact, valid_frame,
    command_target, target, active_arm, task_phase, target_distance. Images are
    uint8 [episode,frame,view,height,width,channel] in image_path. Explicit
    intervention_kind distinguishes direct_correction/waypoint_recovery/expert;
    source directory naming has no effect on labels or weighting.
    """
    def __init__(self, cache: Mapping[str, Any], contract: ActionContract, *, weighted=False, correction_boost=1.0):
        require_torch()
        _numpy()
        if not isinstance(contract, ActionContract):
            raise TypeError("Dataset requires the shared ActionContract")
        if correction_boost <= 0 or not math.isfinite(correction_boost):
            raise ValueError("Correction boost must be positive and finite")
        required = {"state", "velocity", "contact_sites", "contact", "valid_frame", "command_target",
                    "target", "active_arm", "task_phase", "target_distance", "image_path", "joint_names",
                    "tasks", "task_index", "vocab", "intervention_kind", "metadata", "stats", "data_provenance"}
        if required - set(cache):
            raise ContractError("Rebuild cache missing fields: " + ", ".join(sorted(required - set(cache))))
        contract.assert_joint_order(cache["joint_names"])
        validate_checkpoint_metadata(cache["metadata"], vocab=cache["vocab"], stats=cache["stats"])
        if fingerprint(cache["metadata"]["contract"]) != fingerprint(contract.metadata()):
            raise ContractError("Dataset action contract differs from execution contract")
        validate_data_provenance(cache["data_provenance"])
        state = cache["state"]
        if state.ndim != 3 or state.shape[-1] != 31:
            raise ContractError("state must have shape [episodes,frames,31]")
        for key in ("velocity", "command_target"):
            if cache[key].shape != state.shape:
                raise ContractError(key + " must match state shape")
        if cache["valid_frame"].shape != state.shape[:2]:
            raise ContractError("valid_frame shape must match episode/frame axes")
        if cache["contact_sites"].shape[:2] != state.shape[:2] or cache["contact"].shape != (*state.shape[:2], 1):
            raise ContractError("Contact observations must share episode/frame axes")
        if cache["target_distance"].shape != (*state.shape[:2], 1):
            raise ContractError("target_distance must be an offline scalar label per frame")
        self.cache, self.contract, self.images = cache, contract, None
        self.rows, weights = [], []
        self.tasks = [cache["tasks"][int(i)] for i in cache["task_index"].tolist()]
        self.ids, self.mask = encode_tasks(self.tasks, cache["vocab"])
        kinds = {"expert", "direct_correction", "waypoint_recovery"}
        for episode, length in enumerate(_valid_lengths(cache)):
            kind = cache["intervention_kind"][episode]
            if kind not in kinds:
                raise ContractError("Unknown intervention kind; startup waits cannot become corrections")
            correction = kind == "direct_correction"
            for frame in range(max(0, length - contract.chunk_steps + 1)):
                self.rows.append((episode, frame, correction))
                weight = float(cache["sample_weight"][episode, frame]) if weighted else 1.0
                weight *= correction_boost if correction else 1.0
                if not math.isfinite(weight) or weight <= 0:
                    raise ContractError("Sample weights must be finite and positive")
                weights.append(weight)
        self.weights = np.asarray(weights, dtype=np.float64)
        if not self.rows:
            raise ContractError("No complete continuous action chunks")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        episode, frame, correction = self.rows[index]
        if self.images is None:
            self.images = np.load(self.cache["image_path"], mmap_mode="r", allow_pickle=False)
            if self.images.ndim != 6 or self.images.shape[2] != 3 or self.images.shape[-1] != 3:
                raise ContractError("Images must have shape [episode,frame,3,height,width,3]")
        history = np.maximum(np.arange(frame - self.contract.history_steps + 1, frame + 1), 0)
        return {
            "images": torch.from_numpy(np.asarray(self.images[episode, history]).copy()).permute(0, 1, 4, 2, 3).float().div_(255),
            "state": self.cache["state"][episode, history].float(),
            "velocity": self.cache["velocity"][episode, history].float(),
            "sites": self.cache["contact_sites"][episode, history].float(),
            "tactile": self.cache["contact"][episode, history].float(),
            "action": action_label(self.cache, episode, frame, self.contract),
            "target": self.cache["target"][episode].long(),
            "arm": self.cache["active_arm"][episode, frame].long(),
            "phase": self.cache["task_phase"][episode, frame].long(),
            "distance": self.cache["target_distance"][episode, frame].float(),
            "contact": self.cache["contact"][episode, frame].float(),
            "ids": self.ids[episode], "mask": self.mask[episode],
            "correction": torch.tensor(float(correction)),
        }


# Keep the familiar core name available without keeping an experimental entrypoint.
Data = ContinuousDataset


def get_stats(cache, contract: ActionContract):
    """Compute statistics for fresh audited data only; never during warm-start."""
    require_torch()
    valid = cache["valid_frame"].bool()
    lengths = _valid_lengths(cache)
    result = {}
    for name in ("state", "velocity", "contact_sites"):
        values = cache[name][valid].float()
        if len(values) < 2 or not torch.isfinite(values).all():
            raise ContractError("Normalization requires at least two finite observed frames")
        result[name + "_mean"] = values.mean(0)
        result[name + "_std"] = values.std(0).clamp_min(1e-3)
    plans = [action_label(cache, episode, frame, contract)
             for episode, length in enumerate(lengths)
             for frame in range(max(0, length - contract.chunk_steps + 1))]
    if not plans:
        raise ContractError("No complete future command chunks for normalization")
    plans = torch.cat(plans, 0)
    if not torch.isfinite(plans).all():
        raise ContractError("Non-finite future commands")
    result["action_delta_mean"] = plans.mean(0)
    result["action_delta_std"] = plans.std(0).clamp_min(1e-3)
    validate_stats(result, int(cache["contact_sites"].shape[-1]))
    return result


def forward(model, batch, stats):
    """Shared normalized forward/loss; labels cannot cross the input boundary."""
    require_torch()
    functional = torch.nn.functional
    observation = TensorObservation(
        batch["images"], (batch["state"] - stats["state_mean"]) / stats["state_std"],
        (batch["velocity"] - stats["velocity_mean"]) / stats["velocity_std"],
        (batch["sites"] - stats["contact_sites_mean"]) / stats["contact_sites_std"],
        batch["tactile"], batch["ids"], batch["mask"])
    output = model.forward_observation(observation)
    truth = (batch["action"] - stats["action_delta_mean"]) / stats["action_delta_std"]
    horizon_weight = torch.linspace(1.5, .65, truth.shape[1], device=truth.device)
    arm_step = functional.smooth_l1_loss(output["action"][:, :, ARM_INDICES], truth[:, :, ARM_INDICES], reduction="none").mean(-1)
    hand_step = functional.smooth_l1_loss(output["action"][:, :, HAND_INDICES], truth[:, :, HAND_INDICES], reduction="none").mean(-1)
    action_loss = (arm_step * horizon_weight).mean() + .45 * (hand_step * horizon_weight).mean()
    # Anchor-relative targets are positions; summing across time has no endpoint
    # meaning. Supervise the final planned target directly.
    endpoint = functional.smooth_l1_loss(output["action"][:, -1, ARM_INDICES], truth[:, -1, ARM_INDICES])
    hand_endpoint = functional.smooth_l1_loss(output["action"][:, -1, HAND_INDICES], truth[:, -1, HAND_INDICES])
    valid = batch["phase"] >= 0
    phase_loss = (functional.cross_entropy(output["phase_logits"][valid], batch["phase"][valid])
                  if valid.any() else output["phase_logits"].sum() * 0)
    loss = (action_loss + .25 * endpoint + .08 * hand_endpoint
            + .20 * functional.cross_entropy(output["target_logits"], batch["target"])
            + .10 * functional.cross_entropy(output["active_arm_logits"], batch["arm"])
            + .18 * phase_loss + .20 * functional.smooth_l1_loss(output["distance"], batch["distance"] / .25)
            + .12 * functional.binary_cross_entropy_with_logits(output["contact_logits"], batch["contact"]))
    predicted = output["action"] * stats["action_delta_std"] + stats["action_delta_mean"]
    return loss, output, predicted


def evaluate(model, loader, stats, device):
    """Offline diagnostics only; these numbers do not prove physical success."""
    require_torch()
    model.eval()
    sq = first_sq = correction_sq = 0.0
    count = first_count = rows = target_ok = phase_ok = phase_n = correction_n = 0
    with torch.no_grad():
        for raw in loader:
            batch = {key: value.to(device) for key, value in raw.items()}
            _, output, predicted = forward(model, batch, stats)
            rows += len(batch["target"])
            sq += float((predicted - batch["action"]).square().sum())
            count += predicted.numel()
            first_sq += float((predicted[:, 0] - batch["action"][:, 0]).square().sum())
            first_count += predicted[:, 0].numel()
            target_ok += int((output["target_logits"].argmax(-1) == batch["target"]).sum())
            valid = batch["phase"] >= 0
            phase_ok += int((output["phase_logits"].argmax(-1)[valid] == batch["phase"][valid]).sum())
            phase_n += int(valid.sum())
            correction = batch["correction"] > .5
            if correction.any():
                correction_sq += float((predicted[correction, 0] - batch["action"][correction, 0]).square().sum())
                correction_n += predicted[correction, 0].numel()
    if rows == 0:
        raise ContractError("Cannot evaluate an empty dataset")
    return {"chunk_action_rmse_rad": math.sqrt(sq / count), "first_action_rmse_rad": math.sqrt(first_sq / first_count),
            "correction_first_action_rmse_rad": math.sqrt(correction_sq / correction_n) if correction_n else None,
            "target_accuracy": target_ok / rows, "phase_accuracy": phase_ok / phase_n if phase_n else None,
            "physical_success_verified": False}


def capture_training_state(optimizer, *, epoch: int, global_step: int) -> dict:
    require_torch()
    _numpy()
    return {"epoch": epoch, "global_step": global_step, "optimizer": optimizer.state_dict(),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}}


def restore_training_state(checkpoint, model, optimizer, *, vocab, stats, expected_metadata,
                           expected_data_provenance):
    """Resume weights, optimizer and every RNG; no partial weight-only resume."""
    require_torch()
    _numpy()
    validate_warm_start(checkpoint, vocab, stats, expected_metadata)
    validate_data_provenance(expected_data_provenance)
    if fingerprint(checkpoint["data_provenance"]) != fingerprint(expected_data_provenance):
        raise ContractError("Resume data provenance differs from the original run")
    state = checkpoint.get("training_state")
    if not isinstance(state, dict) or set(state) != {"epoch", "global_step", "optimizer", "rng"}:
        raise ContractError("Resume requires optimizer, epoch, global_step and RNG state")
    if any(type(state[key]) is not int or state[key] < 0 for key in ("epoch", "global_step")):
        raise ContractError("Resume counters must be non-negative integers")
    rng = state["rng"]
    if not isinstance(rng, dict) or set(rng) != {"python", "numpy", "torch", "cuda"}:
        raise ContractError("Resume requires Python, NumPy, CPU torch and CUDA RNG states")
    if rng["cuda"] and (not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count()):
        raise ContractError("Cannot restore the original CUDA RNG state on this runtime")
    # Validate RNGs with private generators before mutating global state.
    random.Random().setstate(rng["python"])
    np.random.RandomState().set_state(rng["numpy"])
    torch.Generator(device="cpu").set_state(rng["torch"].cpu())
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if rng["cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
    return {"epoch": state["epoch"], "global_step": state["global_step"]}


def training_readiness(config: Mapping[str, Any]) -> dict:
    """Inspect prerequisites without loading weights, data tensors or optimizer."""
    errors = []
    try:
        checked = validate_config(dict(config))
    except (ValueError, TypeError) as exc:
        return {"ready": False, "blockers": [str(exc)], "optimizer_created": False}
    try:
        contract = ActionContract.from_config(checked)
        contract_hash = contract.contract_sha256
    except (ValueError, OSError) as exc:
        contract_hash = None
        errors.append(str(exc))
    for name in ("batch_size", "epochs", "learning_rate"):
        if checked["train"][name] is None:
            errors.append("train." + name + " is unresolved")
    if checked["train"]["authorization"] != "conditional":
        errors.append("conditional training authorization is required")
    build_dir = Path(checked["data"]["build_dir"])
    requirements = ["manifest.json", "audit.json", "train.pt", "validation.pt"]
    for name in requirements:
        if not (build_dir / name).is_file():
            errors.append("audited dataset artifact missing: " + str(build_dir / name))
    # This preparation revision cannot begin optimization merely by editing a
    # boolean. The source-aligned builder and explicit execution stage remain.
    errors.append("the current action31 builder/optimizer runner has not been admitted; source-native public training requires its own audited backend and split")
    return {"ready": False, "blockers": errors, "action_contract_sha256": contract_hash,
            "required_dataset_artifacts": requirements, "optimizer_created": False,
            "warm_start_rule": "exact token mapping and fixed normalization; strict weights only",
            "resume_rule": "same metadata/data plus weights, optimizer, epoch and all RNG states"}


def train(config: Mapping[str, Any], run_dir: Path) -> dict:
    """Record prerequisites. This unadmitted runner never creates an optimizer."""
    readiness = training_readiness(config)
    enabled = config.get("train", {}).get("enabled")
    result = {"schema": "g1_vla_training_gate_v1", "status": "disabled" if enabled is False else "blocked",
              "training_started": False, "optimizer_created": False, "weights_updated": False,
              "reason": "train.enabled=false" if enabled is False else "Audited execution pipeline is not ready",
              "readiness": readiness}
    path = Path(run_dir) / "reports" / "training_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return result
