"""Train/validation-only state-residual ACT with an exact state-hold origin.

The network predicts (recorded action - raw anchor q) / train action std.
The original train observation normalization remains unchanged.  Its zero
output decodes bit-for-bit to the raw observed anchor, including signed zero.
No source label, camera, split or original model module is rewritten here.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import time
import traceback

from .artifacts import sha256_file
from . import act_diagnostics as diagnostics
from . import grasp_act as native
from . import vision_pretraining as vision


BACKEND = "grasp_square_state_residual_v1"
VARIANTS = ("state_only_residual", "rgb_residual_scratch")
TARGET = "raw_action_minus_anchor_q_over_train_action_std_v1"
DECODE = "raw_anchor_q_plus_train_action_std_times_residual_zero_exact_v1"


class StateResidualError(ValueError):
    """A residual protocol, origin identity, or matched trial was not admitted."""


def validate_state_residual_settings(config):
    settings = copy.deepcopy(config["state_residual"])
    if settings.get("backend") != BACKEND or settings.get("variants") != list(VARIANTS):
        raise StateResidualError("the two predeclared state-only and scratch-RGB residual arms are required in order")
    if (settings.get("use_vae") is not False or settings.get("zero_head_init") is not True
            or settings.get("include_step_zero_in_selection") is not True
            or settings.get("target") != TARGET or settings.get("decode") != DECODE):
        raise StateResidualError("no-VAE, zero action head, exact hold decoder, and selectable step zero are mandatory")
    previous = vision.validate_visual_pretraining_settings(config)
    for name in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size", "max_gpu_reserved_bytes"):
        if isinstance(settings.get(name), bool) or not isinstance(settings.get(name), int) or settings[name] <= 0:
            raise StateResidualError(name + " must retain its positive integer type and value")
    for name in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size",
                 "max_train_wall_s", "max_gpu_reserved_bytes"):
        if settings.get(name) != previous[name] or isinstance(settings.get(name), bool):
            raise StateResidualError("the residual trial must retain the frozen visual comparison " + name)
    for name in ("reference_visual_report_path", "output_dir", "admission_report_path"):
        if not isinstance(settings.get(name), str) or not settings[name]:
            raise StateResidualError(name + " must name a declared project path")
        native._resolve(config, settings[name])
    digest = settings.get("reference_visual_report_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        raise StateResidualError("the frozen prior train/validation visual comparison must have its full SHA256")
    if (config.get("native_diagnostics", {}).get("stationary_speed_rad_s") != 0.02
            or config.get("native_diagnostics", {}).get("moving_speed_rad_s") != 0.1):
        raise StateResidualError("analysis must reuse the original 0.02/0.1 rad/s anchor-speed thresholds")
    return settings


def _residual_arrays(values, raw_state, action_std, np):
    values, state, std = np.asarray(values), np.asarray(raw_state), np.asarray(action_std)
    if (values.dtype != np.float32 or state.dtype != np.float32 or std.dtype != np.float32
            or values.ndim != 3 or values.shape[2] != 28 or values.shape[0] == 0 or values.shape[1] == 0
            or state.shape != (values.shape[0], 28) or std.shape != (28,)
            or not np.isfinite(values).all() or not np.isfinite(state).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise StateResidualError("residuals/actions, raw anchors and scale must be finite FP32 [B,H,28], [B,28] and positive [28]")
    return values, state, std


def residual_targets(raw_action, raw_state, action_std, np):
    """Scale an anchor-relative label; no action mean or reconstructed q is used."""
    action, state, std = _residual_arrays(raw_action, raw_state, action_std, np)
    result = np.asarray((action - state[:, None, :]) / std, dtype=np.float32)
    if not np.isfinite(result).all():
        raise StateResidualError("the scaled residual target overflowed FP32")
    return result


def decode_residual(normalized_residual, raw_state, action_std, np):
    """Use the direct observed q and preserve its exact bytes at residual zero."""
    residual, state, std = _residual_arrays(normalized_residual, raw_state, action_std, np)
    anchor = np.broadcast_to(state[:, None, :], residual.shape)
    result = np.where(residual == np.float32(0), anchor, anchor + std * residual)
    result = np.asarray(result, dtype=np.float32)
    if not np.isfinite(result).all():
        raise StateResidualError("the decoded raw residual prediction overflowed FP32")
    return result


def initialize_residual_head(policy, torch):
    """Zero exactly the final output projection while retaining all other tensors."""
    state = policy.state_dict()
    head_names = {"model.action_head.weight", "model.action_head.bias"}
    if (not head_names.issubset(state) or not hasattr(policy.model, "action_head")
            or tuple(policy.model.action_head.weight.shape)[0] != 28
            or tuple(policy.model.action_head.bias.shape) != (28,)):
        raise StateResidualError("the residual origin requires the original 28-output linear action head")
    non_head = {name: value.detach().cpu().contiguous().clone() for name, value in state.items() if name not in head_names}
    before = diagnostics._tensor_digest(non_head)
    with torch.no_grad():
        policy.model.action_head.weight.zero_()
        policy.model.action_head.bias.zero_()
    actual = policy.state_dict()
    if (any(bool(torch.count_nonzero(actual[name])) for name in head_names)
            or any(not torch.equal(actual[name].detach().cpu(), value) for name, value in non_head.items())):
        raise StateResidualError("zero-head initialization changed non-head values or retained nonzero output weights")
    return {"zero_final_action_head_verified": True, "zeroed_tensor_names": sorted(head_names),
            "non_head_values_unchanged_verified": True, "non_head_tensor_sha256": before,
            "all_initial_tensor_sha256": diagnostics._tensor_digest(actual),
            "zero_output_is_independent_of_features_dropout_and_latent": True}


def select_checkpoint(validation):
    """Select including step zero, preserving the earlier checkpoint on ties."""
    if not validation or validation[0].get("step") != 0:
        raise StateResidualError("selection must begin with the measured step-zero state-hold identity")
    prior, selected, trained = -1, None, None
    for item in validation:
        step = item.get("step")
        mae = item.get("metrics", {}).get("policy", {}).get("all_28", {}).get("mae")
        if (isinstance(step, bool) or not isinstance(step, int) or step <= prior
                or isinstance(mae, bool) or not isinstance(mae, (float, int))
                or not math.isfinite(mae) or mae < 0):
            raise StateResidualError("validation checkpoints must have increasing nonnegative steps and finite MAE")
        if selected is None or mae < selected["metrics"]["policy"]["all_28"]["mae"]:
            selected = item
        if step > 0 and (trained is None or mae < trained["metrics"]["policy"]["all_28"]["mae"]):
            trained = item
        prior = step
    return {"selected_step": selected["step"], "selected_mae": selected["metrics"]["policy"]["all_28"]["mae"],
            "best_trained_step": trained["step"] if trained else None,
            "best_trained_mae": trained["metrics"]["policy"]["all_28"]["mae"] if trained else None,
            "last_step": validation[-1]["step"], "last_mae": validation[-1]["metrics"]["policy"]["all_28"]["mae"],
            "step_zero_included": True, "ties_keep_earliest_step": True,
            "selection_split": "validation", "test_used_for_selection": False}


def _stage(config, run_dir, name):
    settings = validate_state_residual_settings(config)
    directory = Path(run_dir)
    path = directory / "reports" / (name + ".json")
    if not directory.is_dir() or path.exists():
        raise StateResidualError("the stage requires an existing run and a new immutable report")
    path.parent.mkdir(exist_ok=True)
    return settings, path, {"schema_version": 1, "status": "failed", "passed": False, "settings": settings,
                            "training_started": False, "optimizer_steps": 0, "test_payloads_opened": False,
                            "test_used_for_selection": False, "physical_success_evaluated": False,
                            "project_adapter_validated": False, "closed_loop_control": False}


def _verify_reference(config, settings, binding):
    path = native._resolve(config, settings["reference_visual_report_path"])
    if sha256_file(path) != settings["reference_visual_report_sha256"]:
        raise StateResidualError("the frozen stage6 train/validation report SHA256 changed")
    reference = native._read(path)
    if (reference.get("passed") is not True
            or reference.get("status") != "completed_train_validation_visual_pretraining_comparison"
            or reference.get("test_payloads_opened") is not False or reference.get("test_used_for_selection") is not False):
        raise StateResidualError("the reference must be the completed train/validation-only visual comparison")
    for name in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size",
                 "max_train_wall_s", "max_gpu_reserved_bytes"):
        if reference.get("settings", {}).get(name) != settings[name]:
            raise StateResidualError("the residual protocol differs from the prior trial " + name)
    for name in ("original_training_module_sha256", "manifest_sha256", "statistics_sha256",
                 "reference_provenance_sha256", "reference_model_sha256", "selected_artifact_sha256"):
        if reference.get("bindings", {}).get(name) != binding[name]:
            raise StateResidualError("the prior visual comparison differs in " + name)
    if reference.get("bindings", {}).get("visual_pretraining_module_sha256") != sha256_file(vision.__file__):
        raise StateResidualError("the previous no-VAE visual builder was modified after the reference run")
    if reference.get("bindings", {}).get("diagnostic_module_sha256") != sha256_file(diagnostics.__file__):
        raise StateResidualError("the previous diagnostic data/model helpers changed after the reference run")
    comparisons = {}
    for name in ("state_only", "rgb_scratch"):
        variant = reference.get("variants", {}).get(name, {})
        if (variant.get("status") != "completed_matched_budget_no_vae_training"
                or variant.get("optimizer_steps") != settings["max_steps"]
                or len(variant.get("steps", [])) != settings["max_steps"]
                or variant.get("initialization", {}).get("use_vae") is not False
                or [item.get("step") for item in variant.get("validation", [])]
                   != list(range(settings["eval_every_steps"], settings["max_steps"] + 1, settings["eval_every_steps"]))):
            raise StateResidualError("the prior absolute-action control must have the full no-VAE budget and validation plan")
        comparisons[name] = {"selected_step": variant["selected_step"],
                             "selected_train": variant["selected_prior_inference"]["train"]["metrics"],
                             "selected_validation": variant["selected_prior_inference"]["validation"]["metrics"],
                             "optimizer_steps": variant["optimizer_steps"], "reference_only_no_new_training": True}
    return reference, {"path": str(path), "sha256": sha256_file(path), "absolute_action_controls": comparisons,
                       "test_payloads_or_metrics_read": False}


def _prepare(config, declared, report):
    manifest, statistics, binding, datasets = diagnostics.load_diagnostic_data(config)
    binding = dict(binding, state_residual_module_sha256=sha256_file(__file__),
                   prior_no_vae_builder_sha256=sha256_file(vision.__file__))
    report["bindings"] = binding
    reference, report["absolute_action_reference"] = _verify_reference(config, declared, binding)
    settings = diagnostics._settings(config, declared)
    settings["motion_thresholds"] = {key: config["native_diagnostics"][key]
                                      for key in ("stationary_speed_rad_s", "moving_speed_rad_s")}
    np, torch, device, blueprint_policy = native._runtime(config, settings, report)
    diagnostics._verify_reference_runtime(config, config["act_diagnostics"], report)
    if any(report[name] != reference.get(name) for name in ("environment", "runtime_sources", "architecture")):
        raise StateResidualError("the runtime or original construction architecture differs from stage6")
    rng_state = {"cpu": torch.get_rng_state().clone(), "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]}
    blueprint = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in blueprint_policy.state_dict().items()}
    report["full_rgb_initial_blueprint_sha256"] = diagnostics._tensor_digest(blueprint)
    report["post_rgb_blueprint_rng_sha256"] = {"cpu": hashlib.sha256(rng_state["cpu"].numpy().tobytes()).hexdigest(),
                                              "cuda": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for value in rng_state["cuda"]]}
    if (report["full_rgb_initial_blueprint_sha256"] != reference.get("full_rgb_initial_blueprint_sha256")
            or report["post_rgb_blueprint_rng_sha256"] != reference.get("post_rgb_blueprint_rng_sha256")):
        raise StateResidualError("the shared original RGB initialization or post-construction RNG changed")
    del blueprint_policy
    torch.cuda.empty_cache()
    schedule, identities, schedule_report = diagnostics.anchor_schedule(datasets["train"], settings, config["train"]["batch_size"], declared["seed"], np)
    if schedule_report != reference.get("anchor_schedule"):
        raise StateResidualError("the exact 2000-step sampled anchor schedule differs from the absolute-action controls")
    report["anchor_schedule"] = schedule_report
    report["construction_blueprint_initialization"] = report.pop("model_initialization")
    report["model_initialization"] = {"formal_use_vae": False, "public_task_policy_loaded": False,
                                      "public_visual_backbone_loaded": False, "all_formal_heads_zero_initialized": True}
    report["architecture_scope"] = "Original untrained RGB construction blueprint; residual architectures are saved per variant."
    report["target_protocol"] = {"target": TARGET, "decode": DECODE, "anchor": "direct_raw_observation_state_at_sample",
                                  "scale": "existing_train_only_action_std", "action_mean_subtracted": False,
                                  "residual_statistics_fitted": False, "source_labels_modified": False,
                                  "normalized_state_inverted_for_anchor": False,
                                  "network_inference_receives_labels": False}
    return settings, manifest, statistics, datasets, binding, np, torch, device, blueprint, rng_state, schedule, identities


def _variant_policy(config, settings, variant, blueprint, np, torch, device, rng_state):
    if variant not in VARIANTS:
        raise StateResidualError("unknown state residual variant")
    mode = "state_only" if variant == "state_only_residual" else "rgb_scratch"
    policy, initialization = vision._variant_policy(config, settings, mode, blueprint, np, torch, device, rng_state)
    initialization["absolute_initialization_variant"] = initialization.pop("variant")
    initialization["variant"] = variant
    initialization["residual_head"] = initialize_residual_head(policy, torch)
    initialization["action_output_semantics"] = TARGET
    initialization["action_decoder"] = DECODE
    initialization["normalization_inside_policy"] = "original_policy_forward_does_not_apply_processor_normalization"
    torch.set_rng_state(rng_state["cpu"])
    torch.cuda.set_rng_state_all(rng_state["cuda"])
    return policy, initialization


def _training_batch(dataset, indices, stats, variant, np, torch, device):
    batch, samples = native._batch(dataset, indices, stats, np, torch, device, include_actions=False)
    if variant == "state_only_residual":
        del batch[native.CAMERA]
    state = np.stack([sample["observation.state"] for sample in samples])
    actions = np.stack([sample["action"] for sample in samples])
    batch["action"] = torch.from_numpy(residual_targets(actions, state, stats["action"][1], np)).to(device)
    masks = np.stack([sample["action_is_pad"] for sample in samples])
    if masks.any() or masks.shape != actions.shape[:2]:
        raise StateResidualError("residual training admits only the unchanged complete, unpadded source chunks")
    batch["action_is_pad"] = torch.from_numpy(masks.astype(np.bool_)).to(device)
    return batch, samples


def _evaluate(policy, dataset, statistics, settings, np, torch, device, *, variant, deadline=None, require_zero=False):
    indices = dataset.get_eval_indices(settings["eval_stride"])
    ids = np.asarray([dataset.samples[index] for index in indices], dtype=np.int64)
    stats = native._stats(statistics, np)
    predictions, residuals, targets, states = [], [], [], []
    policy.eval()
    for start in range(0, len(indices), settings["eval_batch_size"]):
        if deadline is not None and time.perf_counter() >= deadline:
            raise StateResidualError("the unchanged wall budget expired before complete validation")
        batch, samples = native._batch(dataset, indices[start:start + settings["eval_batch_size"]], stats, np, torch, device, include_actions=False)
        if variant == "state_only_residual":
            del batch[native.CAMERA]
        if "action" in batch or "action_is_pad" in batch:
            raise StateResidualError("residual inference must not receive labels or action masks")
        with torch.inference_mode():
            residual = policy.predict_action_chunk(batch).cpu().numpy()
        state = np.stack([sample["observation.state"] for sample in samples])
        if residual.shape != (len(samples), settings["chunk_size"], 28):
            raise StateResidualError("the residual network must predict one complete original action chunk")
        prediction = decode_residual(residual, state, stats["action"][1], np)
        if require_zero:
            held = np.repeat(state[:, None, :], settings["chunk_size"], axis=1)
            if np.any(residual != 0) or prediction.tobytes() != held.tobytes():
                raise StateResidualError("the zero-head actual network did not decode bit-for-bit to raw state-hold")
        residuals.extend(residual)
        predictions.extend(prediction)
        targets.extend(sample["action"] for sample in samples)
        states.extend(state)
        native._check_gpu(torch, settings)
    arrays = {"predictions_source_actions": np.asarray(predictions, dtype=np.float32),
              "predictions_scaled_state_residual": np.asarray(residuals, dtype=np.float32),
              "target_source_recorded_actions": np.asarray(targets, dtype=np.float32),
              "observation_state": np.asarray(states, dtype=np.float32), "sample_episode_frame": ids,
              "action_std": stats["action"][1]}
    metrics = diagnostics.diagnostic_metrics(arrays["predictions_source_actions"], arrays["target_source_recorded_actions"],
                                               arrays["observation_state"], stats["action"][0], stats["action"][1], ids, np)
    metrics.update(inference_latent="zero_latent_no_vae_no_labels", evaluated_split=dataset.split,
                   eval_stride_source_frames=settings["eval_stride"], state_only=variant == "state_only_residual",
                   camera_input_present=variant != "state_only_residual", target_protocol=TARGET,
                   raw_anchor_reconstructed_from_normalized_state=False, zero_origin_verified=require_zero)
    if require_zero and metrics["policy"] != metrics["state_hold"]:
        raise StateResidualError("measured zero-head policy and state-hold metrics differ")
    if "motion_thresholds" in settings:
        metrics["motion_strata"] = _motion_strata(dataset, arrays, statistics,
                                                  {"native_diagnostics": settings["motion_thresholds"]}, np)
    return metrics, arrays


def _motion_strata(dataset, arrays, statistics, config, np):
    """Analysis-only anchor speed uses the previous raw row, never a model input."""
    ids = arrays["sample_episode_frame"]
    known = ids[:, 1] > 0
    speeds = np.full(len(ids), np.nan, dtype=np.float64)
    for offset, (episode, frame) in enumerate(ids):
        if frame > 0:
            # _evaluate already loaded exactly these selected train/val episodes.
            # No call to a test dataset, future row, action or vision is needed.
            states = dataset._arrays[int(episode)][0]
            speeds[offset] = float(np.max(np.abs(states[int(frame)].astype(np.float64)
                                                - states[int(frame) - 1].astype(np.float64))) * 30.0)
    thresholds = config["native_diagnostics"]
    stationary, moving = thresholds["stationary_speed_rad_s"], thresholds["moving_speed_rad_s"]
    masks = {"stationary": known & (speeds <= stationary), "moving": known & (speeds >= moving),
             "intermediate": known & (speeds > stationary) & (speeds < moving), "unknown_first_row": ~known}
    arrays["anchor_max_joint_speed_rad_s"] = speeds
    arrays["anchor_speed_known"] = known
    action_mean, action_std = native._stats(statistics, np)["action"]
    result = {"speed_definition": "30*max_joint_abs(raw_q[t]-raw_q[t-1]); t=0 unknown",
              "stationary_max_speed_rad_s": stationary, "moving_min_speed_rad_s": moving,
              "uses_only_current_and_previous_observed_state": True, "used_as_model_input": False,
              "partition_anchor_count": len(ids), "strata": {}}
    for name, mask in masks.items():
        result["strata"][name] = {"anchor_count": int(mask.sum()), "metrics": None}
        if mask.any():
            result["strata"][name]["metrics"] = diagnostics.diagnostic_metrics(
                arrays["predictions_source_actions"][mask], arrays["target_source_recorded_actions"][mask],
                arrays["observation_state"][mask], action_mean, action_std, ids[mask], np)
    if sum(item["anchor_count"] for item in result["strata"].values()) != len(ids):
        raise StateResidualError("motion strata must partition the same evaluated anchor denominator exactly")
    return result


def _roundtrip_audit(datasets, statistics, np):
    std = native._stats(statistics, np)["action"][1]
    result = {}
    for split, dataset in datasets.items():
        maxima, count = [], 0
        for episode, metadata in dataset.episodes.items():
            # One accessor call per episode admits the original FP32 arrays;
            # the remaining audit never materializes every RGB frame again.
            first = next(index for index, identity in enumerate(dataset.samples) if identity[0] == episode)
            dataset[first]
            source_state, source_action = dataset._arrays[episode]
            for offset in range(0, metadata["sample_count"], 128):
                stop = min(offset + 128, metadata["sample_count"])
                states = source_state[offset:stop]
                actions = np.stack([source_action[frame:frame + dataset.chunk_steps] for frame in range(offset, stop)])
                encoded = residual_targets(actions, states, std, np)
                decoded = decode_residual(encoded, states, std, np)
                maximum = float(np.max(np.abs(decoded.astype(np.float64) - actions.astype(np.float64))))
                if maximum > 1e-6:
                    raise StateResidualError("the declared scaled residual roundtrip exceeds 1e-6 rad FP32 tolerance")
                zero = decode_residual(np.zeros_like(actions), states, std, np)
                if zero.tobytes() != np.broadcast_to(states[:, None, :], actions.shape).copy().tobytes():
                    raise StateResidualError("raw state-hold bytes were not preserved for every admitted anchor")
                maxima.append(maximum)
                count += stop - offset
        result[split] = {"complete_anchor_count": count, "max_encode_decode_error_rad": max(maxima),
                         "all_anchor_zero_residual_identity_bitwise": True, "roundtrip_tolerance_rad": 1e-6}
    return result


def run_residual_admission(config, run_dir):
    """Audit all raw anchors, actual step-zero inference, and discarded probes."""
    declared, path, report = _stage(config, run_dir, "state_residual_admission")
    started, torch = time.perf_counter(), None
    try:
        if config["train"].get("enabled") is not True or config["train"].get("authorization") != "conditional":
            raise StateResidualError("discarded residual optimizer probes require the unified conditional training admission")
        prepared = _prepare(config, declared, report)
        settings, _, statistics, datasets, _, np, torch, device, blueprint, rng, schedule, _ = prepared
        report.update(optimizer_created=False, formal_optimizer_created=False, variants={})
        report["roundtrip_audit"] = _roundtrip_audit(datasets, statistics, np)
        for variant in VARIANTS:
            policy, initialization = _variant_policy(config, settings, variant, blueprint, np, torch, device, rng)
            entry = {"initialization": initialization, "step_zero": {}}
            report["variants"][variant] = entry
            for split in ("train", "validation"):
                metrics, arrays = _evaluate(policy, datasets[split], statistics, settings, np, torch, device,
                                             variant=variant, require_zero=True)
                entry["step_zero"][split] = {"metrics": metrics, "predictions": diagnostics._save_arrays(
                    Path(run_dir) / (variant + "_" + split + "_step_zero.npz"), arrays, np)}
            del policy
            torch.cuda.empty_cache()
            entry["memory_probe"] = {}
            _probe_variant(config, settings, variant, blueprint, datasets, statistics, schedule,
                           np, torch, device, rng, entry["memory_probe"])
            if entry["memory_probe"]["initialization"] != initialization:
                raise StateResidualError("the temporary memory model differs from the measured zero-origin model")
        report.update(status="passed_exact_zero_state_hold_residual_admission", passed=True,
                      all_anchor_zero_residual_identity_bitwise=True,
                      actual_network_step_zero_train_validation_equals_state_hold=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        report["optimizer_created"] = any(entry.get("memory_probe", {}).get("optimizer_created", False)
                                           for entry in report.get("variants", {}).values())
        report["discarded_probe_optimizer_steps"] = sum(entry.get("memory_probe", {}).get("probe_optimizer_steps", 0)
                                                        for entry in report.get("variants", {}).values())
        report["optimizer_steps_including_discarded_probes"] = report["optimizer_steps"] + report["discarded_probe_optimizer_steps"]
        native._gpu_finalize(torch, report)
        report["total_elapsed_s"] = time.perf_counter() - started
        native._write(path, report)
    return report


def _verify_admission(config, declared, binding, report):
    path = native._resolve(config, declared["admission_report_path"])
    admission = native._read(path)
    if (admission.get("passed") is not True
            or admission.get("status") != "passed_exact_zero_state_hold_residual_admission"
            or admission.get("bindings") != binding or admission.get("settings") != declared
            or admission.get("formal_optimizer_created") is not False or admission.get("optimizer_steps") != 0
            or admission.get("discarded_probe_optimizer_steps") != len(VARIANTS)
            or admission.get("optimizer_steps_including_discarded_probes") != len(VARIANTS)
            or admission.get("test_payloads_opened") is not False or admission.get("test_used_for_selection") is not False
            or admission.get("all_anchor_zero_residual_identity_bitwise") is not True
            or admission.get("actual_network_step_zero_train_validation_equals_state_hold") is not True):
        raise StateResidualError("formal residual training requires the exact current config/code/data zero-origin admission")
    for name in ("environment", "runtime_sources", "architecture", "full_rgb_initial_blueprint_sha256", "post_rgb_blueprint_rng_sha256", "anchor_schedule"):
        if report[name] != admission.get(name):
            raise StateResidualError("the current residual runtime differs from its zero-origin admission " + name)
    for split, expected in (("train", 5651), ("validation", 2488)):
        audit = admission.get("roundtrip_audit", {}).get(split, {})
        error = audit.get("max_encode_decode_error_rad")
        if (audit.get("complete_anchor_count") != expected or audit.get("all_anchor_zero_residual_identity_bitwise") is not True
                or isinstance(error, bool) or not isinstance(error, (int, float))
                or not math.isfinite(error) or not 0 <= error <= 1e-6):
            raise StateResidualError("admission must audit every unchanged train/validation complete raw anchor")
    for variant in VARIANTS:
        entry = admission.get("variants", {}).get(variant, {})
        if entry.get("initialization", {}).get("residual_head", {}).get("zero_final_action_head_verified") is not True:
            raise StateResidualError("the admitted residual output head is not exactly zero")
        probe = entry.get("memory_probe", {})
        reserved = probe.get("peak_reserved_bytes")
        if (probe.get("passed") is not True or probe.get("probe_optimizer_steps") != 1
                or probe.get("formal_training_steps") != 0 or probe.get("temporary_parameters_discarded") is not True
                or probe.get("finite_gradients") is not True or probe.get("full_validation_batch_passed") is not True
                or probe.get("initialization") != entry["initialization"]
                or isinstance(reserved, bool) or not isinstance(reserved, int)
                or not 0 < reserved <= declared["max_gpu_reserved_bytes"]):
            raise StateResidualError("each residual arm requires its exact discarded optimizer/evaluation memory probe")
        for split in ("train", "validation"):
            item = entry.get("step_zero", {}).get(split, {})
            artifact = item.get("predictions", {})
            if sha256_file(artifact.get("path", "")) != artifact.get("sha256"):
                raise StateResidualError("an admitted step-zero prediction artifact changed")
            if (item.get("metrics", {}).get("zero_origin_verified") is not True
                    or item["metrics"]["policy"] != item["metrics"]["state_hold"]):
                raise StateResidualError("the admitted step-zero policy is not the exact state-hold control")
    report["zero_origin_admission"] = {"path": str(path), "sha256": sha256_file(path), "same_full_binding_verified": True,
                                       "formal_optimizer_steps": 0, "discarded_probe_optimizer_steps": len(VARIANTS),
                                       "prediction_files_rehashed": True}
    return admission


def _probe_variant(config, settings, variant, blueprint, datasets, statistics, schedule, np, torch, device, rng, receipt):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    receipt.update(passed=False, optimizer_created=False, probe_optimizer_steps=0,
                   temporary_parameters_discarded=False, formal_training_steps=0)
    policy, initialization = _variant_policy(config, settings, variant, blueprint, np, torch, device, rng)
    stats = native._stats(statistics, np)
    batch, _ = _training_batch(datasets["train"], schedule[0], stats, variant, np, torch, device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config["train"]["learning_rate"], weight_decay=settings["weight_decay"])
    receipt["optimizer_created"] = True
    policy.train()
    loss, components = policy(batch)
    if not bool(torch.isfinite(loss)) or "kld_loss" in components:
        raise StateResidualError("the residual memory probe has nonfinite loss or an active VAE")
    loss.backward()
    gradients = diagnostics._gradient_report(policy, torch)
    if not gradients["finite"]:
        raise StateResidualError("the residual memory probe gradients are missing or nonfinite")
    optimizer.step()
    receipt["probe_optimizer_steps"] = 1
    if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
        raise StateResidualError("the temporary residual optimizer produced nonfinite parameters")
    del batch, loss
    # One complete configured validation batch exercises both image inference
    # and direct-anchor decoding at the same shapes used in formal validation.
    indices = datasets["validation"].get_eval_indices(settings["eval_stride"])[:settings["eval_batch_size"]]
    if len(indices) != settings["eval_batch_size"]:
        raise StateResidualError("the full validation memory-probe batch is unavailable")
    batch, samples = native._batch(datasets["validation"], indices, stats, np, torch, device, include_actions=False)
    if variant == "state_only_residual":
        del batch[native.CAMERA]
    policy.eval()
    with torch.inference_mode():
        residual = policy.predict_action_chunk(batch).cpu().numpy()
    decoded = decode_residual(residual, np.stack([sample["observation.state"] for sample in samples]), stats["action"][1], np)
    torch.cuda.synchronize()
    if decoded.shape != (settings["eval_batch_size"], settings["chunk_size"], 28):
        raise StateResidualError("the temporary residual validation output shape changed")
    native._check_gpu(torch, settings)
    receipt.update(passed=True, finite_gradients=True, temporary_parameters_discarded=True,
                   full_validation_batch_passed=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), initialization=initialization,
                   probe_loss_components=components, first_backward_gradients=gradients,
                   zero_head_first_backward_blocks_upstream_data_gradient_expected=True,
                   adamw_weight_decay_can_still_update_zero_gradient_parameters=True)
    del optimizer, policy, residual, decoded, batch
    torch.cuda.empty_cache()


def _save_checkpoint(directory, config, manifest, statistics, binding, initialization, state, step, metrics):
    result = diagnostics._save_variant_checkpoint(directory, config, manifest, statistics, binding, initialization, state, step, metrics)
    # Decoder semantics are inside initialization/provenance and immutable full
    # configuration. A vanilla absolute ACT loader must not interpret this file.
    result.update(output_semantics=TARGET, decoder=DECODE, compatible_with_absolute_action_loader=False)
    return result


def train_state_residual(config, run_dir):
    """Run the fixed paired residual budgets; explicitly retain hold as step zero."""
    declared, path, report = _stage(config, run_dir, "state_residual_training")
    started, torch = time.perf_counter(), None
    report.update(variants={}, matched_anchor_schedule=True, optimizer_steps_including_discarded_probes=0)
    try:
        if config["train"].get("enabled") is not True or config["train"].get("authorization") != "conditional":
            raise StateResidualError("the unified conditional training admission must be enabled")
        output = native._resolve(config, declared["output_dir"])
        if output.exists():
            raise StateResidualError("residual variants never overwrite existing output or resume prior optimizer state")
        prepared = _prepare(config, declared, report)
        settings, manifest, statistics, datasets, binding, np, torch, device, blueprint, rng, schedule, identities = prepared
        admission = _verify_admission(config, declared, binding, report)
        report["prior_admission_probe_optimizer_steps"] = admission["discarded_probe_optimizer_steps"]
        report["discarded_probes_performed_this_run"] = 0
        output.mkdir(parents=True, exist_ok=False)
        report["anchor_schedule_artifact"] = diagnostics._save_arrays(output / "training_anchor_schedule.npz",
            {"sample_episode_frame": identities, "dataset_sample_indices": schedule}, np)
        report["checkpoint_selection"] = {"include_measured_step_zero": True, "criterion": "validation.policy.all_28.mae",
                                           "best_trained_and_last_step_reported_separately": True, "test_used": False}
        stats = native._stats(statistics, np)
        for variant in VARIANTS:
            entry = {"status": "started", "training_started": False, "optimizer_steps": 0, "steps": [], "validation": [],
                     "selection_split": "validation", "test_payloads_opened": False, "test_used_for_selection": False}
            report["variants"][variant] = entry
            (output / variant).mkdir(exist_ok=False)
            variant_started = time.perf_counter()
            deadline = variant_started + settings["max_train_wall_s"]
            entry["prior_admission_memory_probe"] = copy.deepcopy(admission["variants"][variant]["memory_probe"])
            entry["discarded_probes_performed_this_run"] = 0
            torch.cuda.reset_peak_memory_stats()
            policy, initialization = _variant_policy(config, settings, variant, blueprint, np, torch, device, rng)
            entry["initialization"] = initialization
            if (initialization != entry["prior_admission_memory_probe"]["initialization"]
                    or initialization != admission["variants"][variant]["initialization"]):
                raise StateResidualError("the formal residual initialization differs from its probe or zero-origin admission")
            metrics, _ = _evaluate(policy, datasets["validation"], statistics, settings, np, torch, device,
                                    variant=variant, deadline=deadline, require_zero=True)
            if metrics != admission["variants"][variant]["step_zero"]["validation"]["metrics"]:
                raise StateResidualError("formal step-zero validation differs from the admitted exact state-hold identity")
            entry["validation"].append({"step": 0, "metrics": metrics})
            best_mae, best_metrics, entry["selected_step"] = metrics["policy"]["all_28"]["mae"], metrics, 0
            best_state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in policy.state_dict().items()}
            trained_state, trained_mae, trained_metrics = None, math.inf, None
            optimizer = torch.optim.AdamW(policy.parameters(), lr=config["train"]["learning_rate"], weight_decay=settings["weight_decay"])
            # Step-zero evaluation is label-free and consumes no training RNG.
            # Reapply the declared post-blueprint state before the first formal
            # dropout draw, after probe rebuilding and origin verification.
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state_all(rng["cuda"])
            for offset, indices in enumerate(schedule):
                if time.perf_counter() >= deadline:
                    raise StateResidualError("a residual arm exhausted its unchanged wall budget before all 2000 steps")
                step = offset + 1
                policy.train()
                batch, _ = _training_batch(datasets["train"], indices, stats, variant, np, torch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, components = policy(batch)
                if not bool(torch.isfinite(loss)) or "kld_loss" in components:
                    raise StateResidualError("residual training loss is nonfinite or includes KL")
                loss.backward()
                gradients = diagnostics._gradient_report(policy, torch)
                if not gradients["finite"]:
                    raise StateResidualError("residual training gradients are missing or nonfinite")
                optimizer.step()
                report["training_started"], entry["training_started"] = True, True
                report["optimizer_steps"] += 1
                entry["optimizer_steps"] = step
                torch.cuda.synchronize()
                if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
                    raise StateResidualError("a formal residual optimizer produced nonfinite parameters")
                native._check_gpu(torch, settings)
                entry["steps"].append({"step": step, "loss": float(loss.detach().cpu()), "loss_components": components,
                                       "gradients": gradients, "elapsed_s": time.perf_counter() - variant_started})
                del batch, loss
                if step % settings["eval_every_steps"] == 0:
                    metrics, validation_arrays = _evaluate(policy, datasets["validation"], statistics, settings, np, torch, device,
                                                           variant=variant, deadline=deadline)
                    entry["validation"].append({"step": step, "metrics": metrics})
                    mae = metrics["policy"]["all_28"]["mae"]
                    if mae < best_mae:
                        best_mae, best_metrics, entry["selected_step"] = mae, metrics, step
                        best_state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in policy.state_dict().items()}
                    if mae < trained_mae:
                        trained_mae, trained_metrics, entry["best_trained_step"] = mae, metrics, step
                        trained_state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in policy.state_dict().items()}
                    if step == settings["max_steps"]:
                        entry["last_step_validation_predictions"] = diagnostics._save_arrays(
                            output / variant / "validation_last_step_predictions.npz", validation_arrays, np)
            selection = select_checkpoint(entry["validation"])
            if selection["selected_step"] != entry["selected_step"] or selection["best_trained_step"] != entry["best_trained_step"]:
                raise StateResidualError("online checkpoint tracking differs from independent step-zero-aware selection")
            entry["selection_summary"] = selection
            entry["last_step_validation"] = entry["validation"][-1]
            entry["checkpoint"] = _save_checkpoint(output / variant / "checkpoint_best", config, manifest, statistics, binding,
                                                    initialization, best_state, entry["selected_step"], best_metrics)
            entry["best_trained_checkpoint"] = _save_checkpoint(output / variant / "checkpoint_best_trained", config, manifest,
                statistics, binding, initialization, trained_state, entry["best_trained_step"], trained_metrics)
            entry["selected_prior_inference"] = {}
            policy.load_state_dict(best_state, strict=True)
            policy.requires_grad_(False).eval()
            for split in ("train", "validation"):
                metrics, arrays = _evaluate(policy, datasets[split], statistics, settings, np, torch, device, variant=variant,
                                            deadline=deadline, require_zero=entry["selected_step"] == 0)
                entry["selected_prior_inference"][split] = {"metrics": metrics, "predictions": diagnostics._save_arrays(
                    output / variant / (split + "_predictions.npz"), arrays, np)}
            entry["best_trained_prior_inference"] = {}
            policy.load_state_dict(trained_state, strict=True)
            for split in ("train", "validation"):
                metrics, arrays = _evaluate(policy, datasets[split], statistics, settings, np, torch, device,
                                            variant=variant, deadline=deadline)
                entry["best_trained_prior_inference"][split] = {"metrics": metrics, "predictions": diagnostics._save_arrays(
                    output / variant / (split + "_best_trained_predictions.npz"), arrays, np)}
            entry["loss_windows"] = diagnostics._loss_windows(entry["steps"])
            entry["gpu"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            entry["total_elapsed_s"] = time.perf_counter() - variant_started
            entry["status"] = "completed_matched_budget_state_residual_training"
            del optimizer, policy, best_state, trained_state
            torch.cuda.empty_cache()
        report["status"], report["passed"] = "completed_train_validation_state_residual_comparison", True
        report["limitations"] = ["All source labels, anchor q, input normalization, split identities, sample schedule and 2000-step budgets are unchanged.",
                                  "State hold is selectable as the measured zero-step checkpoint; best trained and final validation scores are also reported without hiding regressions.",
                                  "Residual heads are deliberately zero initialized, so the prediction parameterization and output-head origin both differ from previous absolute controls.",
                                  "Scratch RGB and state-only have different parameter counts and FLOPs; no new pretrained arm is introduced in this isolated residual comparison.",
                                  "One seed and three repeatedly used validation trajectories support development decisions, not independent-test or physical-success claims.",
                                  "This checkpoint predicts scaled residuals and requires its bound raw-anchor decoder; the original absolute-action deployment loader is incompatible."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        report["discarded_probe_optimizer_steps"] = 0
        report["optimizer_steps_including_discarded_probes"] = report["optimizer_steps"] + report["discarded_probe_optimizer_steps"]
        native._gpu_finalize(torch, report)
        report["total_elapsed_s"] = time.perf_counter() - started
        native._write(path, report)
    return report
