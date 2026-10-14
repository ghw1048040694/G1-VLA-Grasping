"""Bounded ACT training on the original GraspSquare 28-joint recorded protocol.

This is an offline behaviour-cloning baseline, with one real source camera and
train-only normalization. No policy is connected to a robot or simulator.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import time
import traceback

from .artifacts import sha256_file


CAMERA = "observation.images.cam_left_high"
REPO = "unitreerobotics/G1_Dex3_GraspSquare_Dataset"
REVISION = "c2b44ca4c5f6329d75143613f9f1eada0cf94c62"
SPLITS = {"train": [20, 21, 75, 76, 125, 126],
          "validation": [180, 181, 182], "test": [280, 281, 282]}


class GraspACTError(ValueError):
    """Source-native training or evaluation admission failed."""


def _read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _resolve(config, value):
    root = Path(config["paths"]["root"]).resolve()
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if path.is_symlink():
        raise GraspACTError("stage inputs and outputs must not be symlinks")
    path = path.resolve()
    if not path.is_relative_to(root):
        raise GraspACTError("stage path must remain inside the declared project root")
    return path


def validate_grasp_settings(config):
    settings, train = config["grasp_act"], config["train"]
    if settings.get("backend") != "lerobot_act_grasp_square" or train.get("backend") != "grasp_square_act":
        raise GraspACTError("the source-native GraspSquare ACT backend must be selected explicitly")
    if settings.get("device") != "cuda" or settings.get("chunk_size") != 16:
        raise GraspACTError("this admitted baseline requires CUDA and original 16-frame chunks")
    if train.get("epochs") is not None:
        raise GraspACTError("this training budget is step-bounded; epochs must remain null")
    if train.get("resume_checkpoint") or train.get("warm_start_checkpoint"):
        raise GraspACTError("this from-scratch native trial does not admit resume or warm-start checkpoints")
    for key in ("dim_model", "n_heads", "dim_feedforward", "n_encoder_layers", "n_decoder_layers",
                "n_vae_encoder_layers", "latent_dim", "max_steps", "eval_every_steps", "max_gpu_reserved_bytes",
                "eval_stride", "eval_batch_size"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise GraspACTError("{} must be a positive integer".format(key))
    if settings["dim_model"] % settings["n_heads"]:
        raise GraspACTError("ACT hidden dimension must be divisible by the number of attention heads")
    for key in ("weight_decay", "kl_weight", "max_train_wall_s"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise GraspACTError("{} must be finite and positive".format(key))
    if not isinstance(settings.get("dropout"), (float, int)) or isinstance(settings.get("dropout"), bool) or not 0 <= settings["dropout"] < 1:
        raise GraspACTError("dropout must be in [0, 1)")
    if settings["max_gpu_reserved_bytes"] > 7 * 1024 ** 3:
        raise GraspACTError("the source-native trial must not exceed the declared 7 GiB allocator cap")
    if isinstance(train.get("batch_size"), bool) or not isinstance(train.get("batch_size"), int) or train["batch_size"] <= 0:
        raise GraspACTError("train.batch_size must be a positive integer")
    lr = train.get("learning_rate")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise GraspACTError("train.learning_rate must be finite and positive")
    if isinstance(train.get("seed"), bool) or not isinstance(train.get("seed"), int) or train["seed"] < 0:
        raise GraspACTError("train.seed must be a non-negative integer")
    return copy.deepcopy(settings)


def validate_grasp_manifest(manifest, settings):
    """The native protocol is independent of the project's 31-joint controller."""
    if manifest.get("passed") is not True:
        raise GraspACTError("the complete source-native build must pass before model execution")
    from .external_data import SOURCE_JOINT_NAMES
    if (manifest.get("source_repo_id") != REPO or manifest.get("source_revision") != REVISION
            or manifest.get("source_state_joint_names") != list(SOURCE_JOINT_NAMES)
            or manifest.get("source_action_joint_names") != list(SOURCE_JOINT_NAMES)):
        raise GraspACTError("source identity and both original joint orders must match the pinned 28-joint protocol")
    if manifest.get("action_dim") != 28 or manifest.get("source_fps") != 30 or manifest.get("chunk_steps") != settings["chunk_size"]:
        raise GraspACTError("source state/action cadence or chunk protocol differs from the admitted baseline")
    if manifest.get("camera_key") != CAMERA or manifest.get("image_shape_hwc") != [240, 320, 3]:
        raise GraspACTError("the admitted single original camera and declared resize must be retained")
    if manifest.get("split_episode_ids") != SPLITS:
        raise GraspACTError("the predeclared independent source-video split changed")
    episodes = manifest.get("episodes", [])
    expected = {item for values in SPLITS.values() for item in values}
    actual = [episode.get("episode_id") for episode in episodes]
    if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
        raise GraspACTError("the manifest must contain all 12 unique predeclared episodes")
    for episode in episodes:
        split = episode.get("split")
        if split not in SPLITS or episode["episode_id"] not in SPLITS[split]:
            raise GraspACTError("an episode was assigned to a different declared split")
        if episode.get("row_count", 0) < settings["chunk_size"] or episode.get("sample_count", 0) != episode["row_count"] - settings["chunk_size"] + 1:
            raise GraspACTError("only complete recorded chunks inside each episode may be sampled")
    return manifest


def _data(config, settings):
    from .grasp_data import GraspSquareDataset, load_grasp_manifest
    manifest_path = _resolve(config, settings["dataset_manifest_path"])
    if manifest_path.name != "manifest.json":
        raise GraspACTError("dataset_manifest_path must name the verified build's manifest.json")
    manifest = load_grasp_manifest(manifest_path.parent, expected_settings=config["source_native"],
                                   expected_source=config["external_data"], verify_files=True)
    validate_grasp_manifest(manifest, settings)
    statistics_record = manifest["statistics"]
    statistics_path = manifest_path.parent / statistics_record["relative_path"]
    if statistics_path.is_symlink() or not statistics_path.resolve().is_relative_to(manifest_path.parent):
        raise GraspACTError("normalization statistics must belong to the verified build")
    if sha256_file(statistics_path) != statistics_record["sha256"]:
        raise GraspACTError("normalization statistics differ from the verified manifest")
    statistics = _read(statistics_path)
    binding = {"experiment_config_sha256": _digest(config), "manifest_sha256": sha256_file(manifest_path),
               "statistics_sha256": sha256_file(statistics_path), "module_sha256": sha256_file(__file__),
               "source_protocol": "grasp_square_recorded_row_28d_30hz_single_original_camera_v1",
               "camera_key": CAMERA, "state_dim": 28, "action_dim": 28,
               "image_shape_hwc": [240, 320, 3], "chunk_steps": settings["chunk_size"],
               "train_batch_size": config["train"]["batch_size"], "eval_batch_size": settings["eval_batch_size"]}
    datasets = {split: GraspSquareDataset(manifest_path.parent, split, manifest=manifest, verify=False)
                for split in SPLITS}
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise GraspACTError("every declared split must contain complete source-native samples")
    return manifest, statistics, binding, datasets


def validate_memory_admission(report, binding, settings):
    if report.get("status") != "passed_memory_probe" or report.get("memory_passed") is not True:
        raise GraspACTError("a successful actual forward/backward/optimizer memory probe is required")
    if report.get("bindings") != binding:
        raise GraspACTError("memory probe is not bound to the current full data, statistics, config and model code")
    if (report.get("probe_optimizer_steps") != 1 or report.get("temporary_parameters_discarded") is not True
            or report.get("training_started") is not False or report.get("finite_gradients") is not True
            or report.get("eval_batch_probe_passed") is not True):
        raise GraspACTError("memory evidence must include the exact temporary optimizer step and evaluation batch")
    reserved = report.get("gpu", {}).get("peak_reserved_bytes")
    if isinstance(reserved, bool) or not isinstance(reserved, int) or not 0 < reserved <= settings["max_gpu_reserved_bytes"]:
        raise GraspACTError("actual measured memory exceeds the current allocator budget")
    return report


def _runtime(config, settings, report):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
    if not torch.cuda.is_available():
        raise GraspACTError("the actual CUDA device is unavailable; CPU estimates do not admit training")
    torch.manual_seed(config["train"]["seed"])
    torch.cuda.manual_seed_all(config["train"]["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(settings["device"])
    index = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(index)
    torch.cuda.set_per_process_memory_fraction(min(1.0, settings["max_gpu_reserved_bytes"] / total), index)
    torch.cuda.reset_peak_memory_stats(index)
    report["gpu"] = {"name": torch.cuda.get_device_name(index), "device_index": index,
                     "total_bytes": total, "free_bytes_before_model_load": free,
                     "allocator_cap_bytes": settings["max_gpu_reserved_bytes"]}
    report["environment"] = {name: importlib.metadata.version(name) for name in
                             ("torch", "torchvision", "lerobot", "numpy", "safetensors")}
    report["environment"]["torch_cuda"] = torch.version.cuda
    report["runtime_sources"] = {name: sha256_file(inspect.getfile(value))
                                 for name, value in (("ACTConfig", ACTConfig), ("ACTPolicy", ACTPolicy))}
    architecture_keys = ("chunk_size", "dim_model", "n_heads", "dim_feedforward", "n_encoder_layers",
                         "n_decoder_layers", "n_vae_encoder_layers", "latent_dim", "dropout", "kl_weight")
    kwargs = {key: settings[key] for key in architecture_keys}
    act_config = ACTConfig(**kwargs, n_obs_steps=1, n_action_steps=settings["chunk_size"],
                           input_features={"observation.state": PolicyFeature(FeatureType.STATE, (28,)),
                                           CAMERA: PolicyFeature(FeatureType.VISUAL, (3, 240, 320))},
                           output_features={"action": PolicyFeature(FeatureType.ACTION, (28,))},
                           normalization_mapping={key: NormalizationMode.MEAN_STD for key in ("STATE", "VISUAL", "ACTION")},
                           pretrained_backbone_weights=None, vision_backbone="resnet18", use_vae=True,
                           device=settings["device"], optimizer_lr=config["train"]["learning_rate"],
                           optimizer_lr_backbone=config["train"]["learning_rate"],
                           optimizer_weight_decay=settings["weight_decay"])
    policy = ACTPolicy(act_config).to(device=device, dtype=torch.float32)
    # Canonical JSON removes dataclass tuple/list differences before comparing a
    # saved memory receipt with a later independently constructed architecture.
    report["architecture"] = json.loads(json.dumps(dataclasses.asdict(act_config), allow_nan=False))
    report["architecture"]["type"] = "act"
    report["model_initialization"] = {"public_checkpoint_loaded": False, "pretrained_backbone_weights": None,
                                      "all_parameters_randomly_initialized": True, "language_conditioned": False,
                                      "parameter_count": sum(item.numel() for item in policy.parameters()), "dtype": "float32"}
    return np, torch, device, policy


def _stats(statistics, np):
    train_ids = statistics.get("training_episode_ids")
    if (train_ids != SPLITS["train"] or statistics.get("validation_or_test_used") is not False
            or statistics.get("statistics_include_held_out") is not False or statistics.get("std_minimum") != 0.001):
        raise GraspACTError("normalization statistics must name only the predeclared training episodes")
    result = {}
    for key, shape in (("observation.state", (28,)), ("action", (28,)), (CAMERA, (3, 1, 1))):
        values = statistics.get("features", statistics).get(key)
        if not isinstance(values, dict):
            raise GraspACTError("normalization statistics lack {}".format(key))
        mean, std = np.asarray(values["mean"], dtype=np.float32), np.asarray(values["std"], dtype=np.float32)
        if mean.shape != shape or std.shape != shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < np.float32(0.001)):
            raise GraspACTError("train-only statistics must have finite matching shapes and positive standard deviations")
        result[key] = (mean, std)
    return result


def _batch(dataset, indices, statistics, np, torch, device, include_actions=True):
    samples = [dataset[int(index)] for index in indices]
    result = {}
    for key in ("observation.state", CAMERA) + (("action",) if include_actions else ()):
        raw = np.stack([sample[key] for sample in samples])
        mean, std = statistics[key]
        if not np.isfinite(raw).all():
            raise GraspACTError("a source-native batch contains nonfinite original observations or labels")
        result[key] = torch.from_numpy(np.asarray((raw - mean) / std, dtype=np.float32)).to(device)
    if include_actions:
        masks = np.stack([sample["action_is_pad"] for sample in samples])
        if masks.any():
            raise GraspACTError("only complete source chunks are admitted; padding is not permitted")
        result["action_is_pad"] = torch.from_numpy(masks.astype(np.bool_)).to(device)
    return result, samples


def _check_gpu(torch, settings):
    if torch.cuda.max_memory_reserved() > settings["max_gpu_reserved_bytes"]:
        raise GraspACTError("CUDA reserved memory exceeded the declared budget; no automatic batch reduction is allowed")


def _finite_gradients(policy, torch):
    gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
    return bool(gradients) and all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


def _gpu_finalize(torch, report):
    if torch is None:
        import sys
        torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and "gpu" in report:
        report["gpu"].update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                             free_bytes_after_stage=torch.cuda.mem_get_info()[0])


def _stage_report(config, run_dir, name):
    settings = validate_grasp_settings(config)
    directory = Path(run_dir)
    report_path = directory / "reports" / (name + ".json")
    if not directory.is_dir() or report_path.exists():
        raise GraspACTError("the stage requires an existing run and a new immutable report name")
    report_path.parent.mkdir(exist_ok=True)
    return settings, report_path, {"schema_version": 1, "status": "failed", "training_started": False,
                                  "optimizer_steps": 0, "closed_loop_control": False,
                                  "project_adapter_validated": False, "physical_success_evaluated": False,
                                  "language_conditioned": False, "settings": settings}


def run_grasp_act_memory(config, run_dir):
    """One temporary true training step, then a true configured validation batch."""
    settings, path, report = _stage_report(config, run_dir, "grasp_act_memory")
    started, torch = time.perf_counter(), None
    report.update(memory_passed=False, probe_optimizer_steps=0, temporary_parameters_discarded=False)
    try:
        manifest, statistics, binding, datasets = _data(config, settings)
        report["bindings"] = binding
        np, torch, device, policy = _runtime(config, settings, report)
        stats = _stats(statistics, np)
        batch, _ = _batch(datasets["train"], range(config["train"]["batch_size"]), stats, np, torch, device)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config["train"]["learning_rate"], weight_decay=settings["weight_decay"])
        report["optimizer_created"] = True
        policy.train()
        loss, loss_components = policy(batch)
        if not bool(torch.isfinite(loss)):
            raise GraspACTError("temporary memory-probe training loss is not finite")
        loss.backward()
        report["finite_gradients"] = _finite_gradients(policy, torch)
        if not report["finite_gradients"]:
            raise GraspACTError("temporary memory-probe gradients are missing or nonfinite")
        optimizer.step()
        torch.cuda.synchronize()
        report["probe_optimizer_steps"] = 1
        report["temporary_parameter_optimization_performed"] = True
        report["probe_loss"] = float(loss.detach().cpu())
        report["probe_loss_components"] = loss_components
        if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
            raise GraspACTError("temporary optimizer step produced nonfinite model parameters")
        del batch, loss
        policy.eval()
        indices = datasets["validation"].get_eval_indices(settings["eval_stride"])[:settings["eval_batch_size"]]
        if len(indices) != settings["eval_batch_size"]:
            raise GraspACTError("validation data cannot fill the configured evaluation memory-probe batch")
        batch, _ = _batch(datasets["validation"], indices, stats, np, torch, device, include_actions=False)
        with torch.inference_mode():
            prediction = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        report["eval_batch_probe_passed"] = bool(torch.isfinite(prediction).all()) and tuple(prediction.shape) == (settings["eval_batch_size"], settings["chunk_size"], 28)
        if not report["eval_batch_probe_passed"]:
            raise GraspACTError("evaluation memory-probe output is not a finite full source-native chunk")
        _check_gpu(torch, settings)
        report["status"], report["memory_passed"] = "passed_memory_probe", True
        del optimizer, policy, prediction, batch
        report["temporary_parameters_discarded"] = True
        report["limitations"] = ["A temporary forward/backward/AdamW step measured capacity; its parameters were discarded.",
                                 "The test split was not used, and no formal training checkpoint or control command was produced."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        _gpu_finalize(torch, report)
        report["total_elapsed_s"] = time.perf_counter() - started
        _write(path, report)
    return report


def compute_action_metrics(prediction, target, np):
    """Recorded-angle error, with the identical denominator for all baselines."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[2] != 28 or not prediction.size:
        raise GraspACTError("metrics require matching nonempty sample/chunk/28 arrays")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise GraspACTError("metrics cannot accept nonfinite predictions or recorded targets")
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    result = {"sample_count": int(target.shape[0]), "chunk_steps": int(target.shape[1]),
              "valid_scalar_count": int(target.size), "units": "source_recorded_radian"}
    for name, selected in (("all_28", difference), ("arms_14", difference[:, :, :14]),
                           ("hands_14", difference[:, :, 14:]), ("first_step_28", difference[:, :1, :])):
        result[name] = {"mae": float(np.abs(selected).mean()), "rmse": float(np.sqrt(np.square(selected).mean()))}
    result["mae_by_joint"] = np.abs(difference).mean(axis=(0, 1)).tolist()
    result["rmse_by_joint"] = np.sqrt(np.square(difference).mean(axis=(0, 1))).tolist()
    return result


def _evaluate(policy, dataset, statistics, settings, np, torch, device, *, deadline=None):
    stats = _stats(statistics, np)
    indices = dataset.get_eval_indices(settings["eval_stride"])
    if not indices:
        raise GraspACTError("the held-out split has no predeclared evaluation observations")
    predictions, targets, states, identities = [], [], [], []
    policy.eval()
    for offset in range(0, len(indices), settings["eval_batch_size"]):
        if deadline is not None and time.perf_counter() >= deadline:
            raise GraspACTError("the wall-clock budget expired before completing held-out validation")
        batch_indices = indices[offset:offset + settings["eval_batch_size"]]
        batch, samples = _batch(dataset, batch_indices, stats, np, torch, device, include_actions=False)
        with torch.inference_mode():
            predicted = policy.predict_action_chunk(batch).cpu().numpy()
        if predicted.shape != (len(samples), settings["chunk_size"], 28) or not np.isfinite(predicted).all():
            raise GraspACTError("the held-out prediction does not preserve the full finite source-native protocol")
        predicted = predicted * stats["action"][1] + stats["action"][0]
        predictions.extend(predicted)
        targets.extend(sample["action"] for sample in samples)
        states.extend(sample["observation.state"] for sample in samples)
        identities.extend(dataset.samples[int(index)] for index in batch_indices)
        _check_gpu(torch, settings)
    predictions, targets, states = np.asarray(predictions), np.asarray(targets), np.asarray(states)
    held = np.repeat(states[:, None, :], settings["chunk_size"], axis=1)
    constant = np.broadcast_to(stats["action"][0], targets.shape)
    def compared(values, reference):
        result = compute_action_metrics(values, reference, np)
        standardized_error = np.abs((values.astype(np.float64) - reference.astype(np.float64)) / stats["action"][1])
        result["normalized_l1"] = float(standardized_error.mean())
        result["normalized_l1_by_joint"] = standardized_error.mean(axis=(0, 1)).tolist()
        return result
    metrics = {"policy": compared(predictions, targets),
               "state_hold": compared(held, targets),
               "train_mean": compared(constant, targets),
               "selection_criterion": "policy.all_28.mae", "eval_stride_source_frames": settings["eval_stride"],
               "frame_selection": "each_episode_frame_index_modulo_stride_is_zero_with_complete_chunk"}
    ids = np.asarray(identities, dtype=np.int64)
    metrics["per_episode"] = {str(int(episode_id)): {
        "policy": compared(predictions[ids[:, 0] == episode_id], targets[ids[:, 0] == episode_id]),
        "state_hold": compared(held[ids[:, 0] == episode_id], targets[ids[:, 0] == episode_id]),
        "train_mean": compared(constant[ids[:, 0] == episode_id], targets[ids[:, 0] == episode_id])}
        for episode_id in np.unique(ids[:, 0])}
    metrics["anchor_independence_assumed"] = False
    if dataset.manifest["source_state_joint_names"] != dataset.manifest["source_action_joint_names"]:
        raise GraspACTError("state-hold baseline requires the verified identical source joint order")
    metrics["state_hold_joint_correspondence"] = list(range(28))
    metrics["source_action_joint_names"] = dataset.manifest["source_action_joint_names"]
    return metrics, {"predictions_source_actions": predictions, "target_source_recorded_actions": targets,
                     "observation_state": states, "sample_episode_frame": np.asarray(identities, dtype=np.int64)}


def _save_checkpoint(directory, config, manifest, statistics, report, best_state, best_validation):
    from safetensors.torch import save_file
    directory.mkdir(parents=True, exist_ok=False)
    save_file(best_state, str(directory / "model.safetensors"))
    _write(directory / "config.json", report["architecture"])
    _write(directory / "experiment_config.json", config)
    _write(directory / "manifest.json", manifest)
    _write(directory / "train_stats.json", statistics)
    _write(directory / "training_summary.json", {"optimizer_steps": report["optimizer_steps"],
                                                "selected_step": report["selected_step"],
                                                "validation": best_validation, "selection_split": "validation",
                                                "test_used_for_selection": False})
    files = {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
             for path in directory.iterdir() if path.is_file()}
    provenance = {"schema_version": 1, "passed": True, "bindings": report["bindings"], "files": files,
                  "source_repo_id": REPO, "source_revision": REVISION,
                  "protocol": "grasp_square_recorded_row_28d_30hz_single_original_camera_v1",
                  "normalization": "manual_featurewise_train_only_mean_std_no_added_epsilon",
                  "architecture_initialization": report["model_initialization"],
                  "selected_step": report["selected_step"], "test_used_for_selection": False,
                  "resumable_optimizer_state_saved": False, "physical_success_evaluated": False}
    _write(directory / "provenance.json", provenance)
    return {"path": str(directory), "provenance_sha256": sha256_file(directory / "provenance.json"),
            "files": files, "selected_step": report["selected_step"]}


def train_grasp_act(config, run_dir):
    settings, path, report = _stage_report(config, run_dir, "grasp_act_training")
    started, torch = time.perf_counter(), None
    report.update(training_passed=False, steps=[], validation=[], test_used_for_selection=False)
    try:
        if config["train"].get("enabled") is not True or config["train"].get("authorization") != "conditional":
            raise GraspACTError("conditional training must be enabled in the unified configuration")
        checkpoint_dir = _resolve(config, settings["checkpoint_path"])
        if checkpoint_dir.exists():
            raise GraspACTError("a checkpoint already exists; this trial never overwrites or resumes it")
        manifest, statistics, binding, datasets = _data(config, settings)
        report["bindings"] = binding
        memory_path = _resolve(config, settings["memory_report_path"])
        memory = validate_memory_admission(_read(memory_path), binding, settings)
        report["memory_report"] = {"path": str(memory_path), "sha256": sha256_file(memory_path)}
        np, torch, device, policy = _runtime(config, settings, report)
        if report["gpu"]["name"] != memory["gpu"]["name"] or report["gpu"]["total_bytes"] != memory["gpu"]["total_bytes"]:
            raise GraspACTError("training CUDA device differs from the admitted measured device")
        if (report["environment"] != memory.get("environment") or report["runtime_sources"] != memory.get("runtime_sources")
                or report["architecture"] != memory.get("architecture")):
            raise GraspACTError("training environment, actual ACT source or architecture differs from the measured admission")
        stats = _stats(statistics, np)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config["train"]["learning_rate"], weight_decay=settings["weight_decay"])
        report["optimizer_created"] = True
        rng = np.random.default_rng(config["train"]["seed"])
        deadline = started + settings["max_train_wall_s"]
        best_state, best_validation, best_mae = None, None, math.inf
        for step in range(1, settings["max_steps"] + 1):
            if time.perf_counter() >= deadline:
                report["stop_reason"] = "wall_clock_budget_reached"
                break
            policy.train()
            report["training_started"] = True
            indices = rng.integers(0, len(datasets["train"]), size=config["train"]["batch_size"])
            batch, _ = _batch(datasets["train"], indices, stats, np, torch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, components = policy(batch)
            if not bool(torch.isfinite(loss)):
                raise GraspACTError("training loss became nonfinite")
            loss.backward()
            if not _finite_gradients(policy, torch):
                raise GraspACTError("training gradients became missing or nonfinite")
            _check_gpu(torch, settings)
            optimizer.step()
            torch.cuda.synchronize()
            report["training_started"], report["optimizer_steps"] = True, step
            if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
                raise GraspACTError("training parameters became nonfinite after the optimizer step")
            _check_gpu(torch, settings)
            report["steps"].append({"step": step, "loss": float(loss.detach().cpu()),
                                     "loss_components": components, "elapsed_s": time.perf_counter() - started})
            del batch, loss
            if step % settings["eval_every_steps"] == 0 or step == settings["max_steps"]:
                metrics, _ = _evaluate(policy, datasets["validation"], statistics, settings, np, torch, device, deadline=deadline)
                report["validation"].append({"step": step, "metrics": metrics})
                mae = metrics["policy"]["all_28"]["mae"]
                if mae < best_mae:
                    best_mae, best_validation, report["selected_step"] = mae, metrics, step
                    best_state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in policy.state_dict().items()}
        if best_state is None:
            raise GraspACTError("the bounded trial did not complete a validation; no unselected checkpoint is published")
        report.setdefault("stop_reason", "configured_step_budget_completed")
        report["checkpoint"] = _save_checkpoint(checkpoint_dir, config, manifest, statistics, report, best_state, best_validation)
        report["status"], report["training_passed"] = "completed_bounded_training", True
        report["limitations"] = ["This is a small from-scratch ACT compute/pipeline trial, not a language-conditioned VLA model.",
                                 "Test observations and labels were not used for model inference, optimization, or checkpoint selection during training.",
                                 "Offline recorded-action error does not establish physical grasp success or a 31-joint control adapter."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        _gpu_finalize(torch, report)
        report["total_elapsed_s"] = time.perf_counter() - started
        _write(path, report)
    return report


def verify_grasp_checkpoint(directory, binding):
    directory = Path(directory)
    provenance = _read(directory / "provenance.json")
    if provenance.get("passed") is not True or provenance.get("bindings") != binding or provenance.get("test_used_for_selection") is not False:
        raise GraspACTError("the checkpoint lacks matching source/data/config/code provenance or used test for selection")
    required = {"model.safetensors", "config.json", "experiment_config.json", "manifest.json", "train_stats.json", "training_summary.json"}
    if set(provenance.get("files", {})) != required:
        raise GraspACTError("the complete independent checkpoint metadata and weights must be present")
    for name, record in provenance["files"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise GraspACTError("checkpoint full-file verification failed: {}".format(name))
    return provenance


def eval_grasp_act(config, run_dir):
    settings, path, report = _stage_report(config, run_dir, "grasp_act_evaluation")
    started, torch = time.perf_counter(), None
    report.update(evaluation_passed=False, held_out_evaluation=True, evaluated_split="test", test_used_for_selection=False)
    try:
        manifest, statistics, binding, datasets = _data(config, settings)
        report["bindings"] = binding
        checkpoint = _resolve(config, settings["checkpoint_path"])
        provenance = verify_grasp_checkpoint(checkpoint, binding)
        # An exclusive claim binds test evaluation to this selected checkpoint.
        # Failed attempts remain evidence; repeating test to tune hyperparameters is not automatic.
        claim = checkpoint / "test_evaluation_claim.json"
        if claim.exists():
            raise GraspACTError("this selected checkpoint already has a test-evaluation claim; automatic repeated test evaluation is forbidden")
        np, torch, device, policy = _runtime(config, settings, report)
        from safetensors.torch import load_file
        state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
        loaded = policy.load_state_dict(state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise GraspACTError("the complete source-native checkpoint did not load strictly")
        policy.requires_grad_(False).eval()
        report["model_initialization"] = {"checkpoint_loaded_strictly": True, "tensor_count": len(state),
                                          "pretrained_backbone_weights": None, "language_conditioned": False,
                                          "selected_step": provenance["selected_step"]}
        del state
        _write(claim, {"run_dir": str(run_dir), "checkpoint_provenance_sha256": sha256_file(checkpoint / "provenance.json"),
                       "selection_split": "validation", "test_used_for_selection": False})
        metrics, arrays = _evaluate(policy, datasets["test"], statistics, settings, np, torch, device)
        output = Path(run_dir) / "grasp_act_test_predictions.npz"
        with output.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report["metrics"] = metrics
        report["prediction_artifact"] = {"path": str(output), "sha256": sha256_file(output)}
        report["checkpoint"] = {"path": str(checkpoint), "provenance_sha256": sha256_file(checkpoint / "provenance.json")}
        report["test_claim"] = {"path": str(claim), "sha256": sha256_file(claim)}
        report["status"], report["evaluation_passed"] = "passed_held_out_recorded_action_evaluation", True
        report["limitations"] = ["The held-out split excludes source video files used by training and validation; recording-session identities are unavailable.",
                                 "Error is measured against recorded 28-joint source actions, never physical grasp success.",
                                 "No source-to-31-joint controller adapter, language conditioning or closed-loop rollout was used."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        _gpu_finalize(torch, report)
        report["total_elapsed_s"] = time.perf_counter() - started
        _write(path, report)
    return report
