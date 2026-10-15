"""Matched normalization attribution on the admitted fourteen-train dataset.

The immutable dataset retains its actual fourteen-episode statistics.  Models
in this stage deliberately use the original six-train statistics instead,
with both origins recorded separately.  Frozen stage8 models are references;
no old model, held-out test, or expanded-statistics helper is run implicitly.
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
from . import expanded_residual as expanded
from . import grasp_act as native
from . import state_residual as residual
from . import vision_pretraining as vision


BACKEND = "grasp_square_normalization_attribution_v1"
MODEL_STATISTICS_SOURCE = "original_six_train_only_v1"
VARIANTS, TARGET, DECODE = residual.VARIANTS, residual.TARGET, residual.DECODE
ORIGINAL_TRAIN, CANDIDATES, VALIDATION, TRAIN = (
    expanded.ORIGINAL_TRAIN, expanded.CANDIDATES, expanded.VALIDATION, expanded.TRAIN)


class NormalizationAttributionError(ValueError):
    """Normalization treatment or its fixed reference failed admission."""


# These frozen helpers take statistics explicitly and require exactly the
# original six train IDs.  They cannot silently select expanded statistics.
_stats = native._stats
_variant_policy = residual._variant_policy
_training_batch = residual._training_batch
_roundtrip_audit = residual._roundtrip_audit
_probe_variant = residual._probe_variant
select_checkpoint = residual.select_checkpoint


def validate_normalization_settings(config):
    settings = copy.deepcopy(config["normalization_attribution"])
    previous = expanded.validate_expanded_residual_settings(config)
    if settings.get("backend") != BACKEND:
        raise NormalizationAttributionError("the declared normalization attribution backend is required")
    for key in ("variants", "use_vae", "zero_head_init", "include_step_zero_in_selection", "target", "decode",
                "seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size", "max_train_wall_s",
                "max_gpu_reserved_bytes", "data_manifest_path", "data_manifest_sha256",
                "reference_residual_report_path", "reference_residual_report_sha256"):
        if settings.get(key) != previous[key] or (isinstance(previous[key], bool) and settings.get(key) is not previous[key]):
            raise NormalizationAttributionError("normalization attribution changed the frozen comparison " + key)
        if key in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size", "max_gpu_reserved_bytes"):
            if isinstance(settings.get(key), bool) or not isinstance(settings.get(key), int):
                raise NormalizationAttributionError("integer comparison budgets must keep their declared type: " + key)
    if settings.get("model_statistics_source") != MODEL_STATISTICS_SOURCE:
        raise NormalizationAttributionError("the model treatment must explicitly use original-six train-only statistics")
    for key in ("reference_expanded_report_path", "reference_statistics_manifest_path", "admission_report_path", "output_dir"):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise NormalizationAttributionError(key + " must name an explicit project path")
        native._resolve(config, settings[key])
    for key in ("reference_expanded_report_sha256", "reference_statistics_manifest_sha256", "reference_statistics_sha256"):
        digest = settings.get(key)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise NormalizationAttributionError(key + " must contain a full lowercase SHA256")
    if native._resolve(config, settings["reference_statistics_manifest_path"]) != native._resolve(config, config["grasp_act"]["dataset_manifest_path"]):
        raise NormalizationAttributionError("original-six statistics must belong to the frozen original native manifest")
    return settings


def _stage(config, run_dir, name):
    settings = validate_normalization_settings(config)
    directory = Path(run_dir)
    path = directory / "reports" / (name + ".json")
    if not directory.is_dir() or path.exists():
        raise NormalizationAttributionError("the stage requires an existing run and a new immutable report")
    path.parent.mkdir(exist_ok=True)
    return settings, path, {"schema_version": 1, "status": "failed", "passed": False, "settings": settings,
        "training_started": False, "optimizer_steps": 0, "test_payloads_opened": False,
        "test_used_for_selection": False, "physical_success_evaluated": False,
        "project_adapter_validated": False, "closed_loop_control": False}


def _verify_expanded_reference(config, declared):
    path = native._resolve(config, declared["reference_expanded_report_path"])
    if sha256_file(path) != declared["reference_expanded_report_sha256"]:
        raise NormalizationAttributionError("the frozen fourteen-train reference report SHA256 changed")
    reference = native._read(path)
    if (reference.get("passed") is not True
            or reference.get("status") != "completed_expanded_train_validation_state_residual_comparison"
            or reference.get("test_payloads_opened") is not False or reference.get("test_used_for_selection") is not False
            or reference.get("optimizer_steps") != 2 * declared["max_steps"]):
        raise NormalizationAttributionError("the reference must be the complete train/validation-only expanded comparison")
    if reference.get("settings") != config["expanded_residual"]:
        raise NormalizationAttributionError("the frozen expanded configuration must remain unchanged")
    for key, module in (("original_training_module_sha256", native), ("diagnostic_module_sha256", diagnostics),
                        ("prior_no_vae_builder_sha256", vision), ("state_residual_module_sha256", residual),
                        ("expanded_residual_module_sha256", expanded)):
        if reference.get("bindings", {}).get(key) != sha256_file(module.__file__):
            raise NormalizationAttributionError("a frozen implementation differs from the expanded reference: " + key)
    if reference.get("bindings", {}).get("manifest_sha256") != declared["data_manifest_sha256"]:
        raise NormalizationAttributionError("the expanded reference used a different fourteen-train manifest")
    controls = {}
    for variant in VARIANTS:
        entry = reference.get("variants", {}).get(variant, {})
        if (entry.get("status") != "completed_matched_budget_state_residual_training"
                or entry.get("optimizer_steps") != declared["max_steps"]
                or len(entry.get("steps", [])) != declared["max_steps"]
                or [item.get("step") for item in entry.get("validation", [])]
                   != list(range(0, declared["max_steps"] + 1, declared["eval_every_steps"]))):
            raise NormalizationAttributionError("each expanded reference arm must retain its complete original budget")
        if select_checkpoint(entry["validation"]) != entry.get("selection_summary"):
            raise NormalizationAttributionError("the frozen expanded selection summary is inconsistent")
        for name in ("checkpoint", "best_trained_checkpoint"):
            checkpoint = entry[name]
            directory = Path(checkpoint["path"])
            if sha256_file(directory / "provenance.json") != checkpoint["provenance_sha256"]:
                raise NormalizationAttributionError("a frozen expanded checkpoint provenance changed")
            native.verify_grasp_checkpoint(directory, reference["bindings"])
        prior_config = native._read(Path(entry["checkpoint"]["path"]) / "experiment_config.json")
        for key in ("seed", "batch_size", "learning_rate"):
            if config["train"][key] != prior_config["train"][key]:
                raise NormalizationAttributionError("normalization attribution changed training " + key)
        if config["grasp_act"] != prior_config["grasp_act"]:
            raise NormalizationAttributionError("normalization attribution changed the original ACT construction")
        controls[variant] = {"selected_step": entry["selected_step"], "selection_summary": entry["selection_summary"],
            "selected_validation": entry["selected_prior_inference"]["validation"]["metrics"],
            "selected_train": entry["selected_prior_inference"]["train"]["metrics"],
            "best_trained_validation": entry["best_trained_prior_inference"]["validation"]["metrics"],
            "last_step_validation": entry["last_step_validation"], "optimizer_steps": entry["optimizer_steps"],
            "reference_only_no_new_training": True, "checkpoint_files_rehashed": True}
    return reference, {"path": str(path), "sha256": sha256_file(path), "controls": controls,
        "raw_validation_metric_comparison_only": True, "frozen_models_not_retrained_or_retuned": True,
        "test_payloads_or_metrics_read": False}


def _load_original_statistics(config, declared, stage7_reference, np):
    """Independently verify actual original-six rows/pixels and their saved fit."""
    manifest, statistics, binding, _ = diagnostics.load_diagnostic_data(config)
    path = native._resolve(config, declared["reference_statistics_manifest_path"])
    if (sha256_file(path) != declared["reference_statistics_manifest_sha256"]
            or binding["manifest_sha256"] != declared["reference_statistics_manifest_sha256"]
            or binding["statistics_sha256"] != declared["reference_statistics_sha256"]
            or binding["manifest_sha256"] != stage7_reference["bindings"]["manifest_sha256"]
            or binding["statistics_sha256"] != stage7_reference["bindings"]["statistics_sha256"]):
        raise NormalizationAttributionError("the model statistics differ from the frozen original-six manifest/statistics")
    _stats(statistics, np)
    if set(statistics["training_episode_ids"]) != set(ORIGINAL_TRAIN) or not set(ORIGINAL_TRAIN) < set(TRAIN):
        raise NormalizationAttributionError("model statistics must fit only the original-six subset of the admitted train pool")
    return statistics, {"manifest_path": str(path), "manifest_sha256": binding["manifest_sha256"],
        "statistics_sha256": binding["statistics_sha256"], "training_episode_ids": list(ORIGINAL_TRAIN),
        "actual_train_rows_and_pixels_recomputed": True, "validation_or_test_used": False,
        "selected_train_validation_artifacts_rehashed": binding["selected_artifact_sha256"]}


def _prepare(config, declared, report):
    import numpy as np
    stage7, report["frozen_original_residual_reference"] = expanded._verify_reference(config, config["expanded_residual"])
    reference, report["frozen_expanded_reference"] = _verify_expanded_reference(config, declared)
    manifest, data_statistics, binding, datasets = expanded.load_expanded_data(config, config["expanded_residual"], stage7)
    if (binding["manifest_sha256"] != reference["bindings"]["manifest_sha256"]
            or binding["statistics_sha256"] != reference["bindings"]["statistics_sha256"]
            or binding["selected_artifact_sha256"] != reference["bindings"]["selected_artifact_sha256"]):
        raise NormalizationAttributionError("the fourteen-train/frozen-validation data differ from the expanded reference")
    statistics, origin = _load_original_statistics(config, declared, stage7, np)
    binding["data_statistics_sha256"] = binding["statistics_sha256"]
    binding["statistics_sha256"] = origin["statistics_sha256"]
    binding.update(model_statistics_source=MODEL_STATISTICS_SOURCE,
        dataset_statistics_sha256=binding["data_statistics_sha256"],
        model_statistics_sha256=origin["statistics_sha256"], normalization_statistics_sha256=origin["statistics_sha256"],
        model_statistics_source_file_sha256=origin["statistics_sha256"],
        model_statistics_content_sha256=native._digest(statistics),
        model_statistics_manifest_sha256=origin["manifest_sha256"],
        model_statistics_training_episode_ids=list(ORIGINAL_TRAIN), dataset_training_episode_ids=list(TRAIN),
        model_statistics_are_train_subset=True, dataset_manifest_statistics_not_modified=True,
        reference_expanded_report_sha256=declared["reference_expanded_report_sha256"],
        normalization_attribution_module_sha256=sha256_file(__file__))
    report["bindings"], report["model_statistics_origin"] = binding, origin
    report["frozen_variant_initialization"] = {name: reference["variants"][name]["initialization"] for name in VARIANTS}
    old, new = _stats(statistics, np), expanded._stats(data_statistics, np)
    report["normalization_treatment"] = {"model_statistics_source": MODEL_STATISTICS_SOURCE,
        "dataset_training_episode_ids": list(TRAIN), "model_statistics_training_episode_ids": list(ORIGINAL_TRAIN),
        "dataset_statistics_sha256": binding["data_statistics_sha256"], "model_statistics_sha256": binding["statistics_sha256"],
        "dataset_manifest_retains_original_fourteen_train_statistics": True,
        "state_and_rgb_input_normalization_changed": True, "action_residual_scale_and_l1_weighting_changed": True,
        "action_mean_not_subtracted_from_residual": True, "training_membership_or_sampler_changed": False,
        "cross_fit_normalized_losses_not_comparable": True, "comparison_metric": "raw_action_MAE_rad_on_unchanged_validation",
        "fourteen_over_six_action_std_ratio": (new["action"][1] / old["action"][1]).tolist(),
        "unique_causal_decomposition_claimed": False,
        "interactions_unidentified_with_this_single_path": True}
    settings = diagnostics._settings(config, declared)
    settings["motion_thresholds"] = {key: config["native_diagnostics"][key]
        for key in ("stationary_speed_rad_s", "moving_speed_rad_s")}
    np, torch, device, blueprint_policy = native._runtime(config, settings, report)
    if any(report[name] != reference.get(name) for name in ("environment", "runtime_sources", "architecture")):
        raise NormalizationAttributionError("normalization attribution changed the frozen runtime or ACT architecture")
    rng = {"cpu": torch.get_rng_state().clone(), "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]}
    blueprint = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in blueprint_policy.state_dict().items()}
    report["full_rgb_initial_blueprint_sha256"] = diagnostics._tensor_digest(blueprint)
    report["post_rgb_blueprint_rng_sha256"] = {"cpu": hashlib.sha256(rng["cpu"].numpy().tobytes()).hexdigest(),
        "cuda": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for value in rng["cuda"]]}
    for key in ("full_rgb_initial_blueprint_sha256", "post_rgb_blueprint_rng_sha256"):
        if report[key] != reference.get(key):
            raise NormalizationAttributionError("the shared original initialization or dropout RNG changed: " + key)
    del blueprint_policy
    torch.cuda.empty_cache()
    schedule, identities, schedule_report = diagnostics.anchor_schedule(datasets["train"], settings,
        config["train"]["batch_size"], declared["seed"], np)
    if schedule_report != reference.get("anchor_schedule"):
        raise NormalizationAttributionError("the exact stage8 expanded-train anchor schedule changed")
    artifact = reference.get("anchor_schedule_artifact", {})
    schedule_path = Path(artifact.get("path", ""))
    if sha256_file(schedule_path) != artifact.get("sha256"):
        raise NormalizationAttributionError("the frozen expanded schedule artifact changed")
    with np.load(schedule_path, allow_pickle=False) as saved:
        if (set(saved.files) != {"sample_episode_frame", "dataset_sample_indices"}
                or not np.array_equal(saved["sample_episode_frame"], identities)
                or not np.array_equal(saved["dataset_sample_indices"], schedule)):
            raise NormalizationAttributionError("the exact sampler indices or episode/frame draws differ from stage8")
    report["anchor_schedule"] = schedule_report
    report["sampler_exact_stage8_replay"] = {"passed": True, "path": str(schedule_path),
        "sha256": artifact["sha256"], "dataset_indices_and_episode_frame_arrays_equal": True}
    report["construction_blueprint_initialization"] = report.pop("model_initialization")
    report["model_initialization"] = {"formal_use_vae": False, "public_task_policy_loaded": False,
        "public_visual_backbone_loaded": False, "all_formal_heads_zero_initialized": True,
        "previous_checkpoint_weights_resumed": False}
    report["architecture_scope"] = "Original untrained RGB construction blueprint; fresh no-VAE residual architectures are saved per variant."
    report["target_protocol"] = {"target": TARGET, "decode": DECODE, "anchor": "direct_raw_observation_state_at_sample",
        "scale": "original_six_train_only_action_std", "action_mean_subtracted": False,
        "residual_statistics_fitted": False, "source_labels_modified": False,
        "normalized_state_inverted_for_anchor": False, "network_inference_receives_labels": False}
    return settings, manifest, statistics, datasets, binding, np, torch, device, blueprint, rng, schedule, identities


def _evaluate(policy, dataset, statistics, settings, np, torch, device, *, variant, deadline=None, require_zero=False):
    metrics, arrays = residual._evaluate(policy, dataset, statistics, settings, np, torch, device,
        variant=variant, deadline=deadline, require_zero=require_zero)
    if dataset.split == "train":
        metrics["train_cohorts"] = expanded._train_cohorts(arrays, _stats(statistics, np), np)
    metrics["model_statistics_source"] = MODEL_STATISTICS_SOURCE
    return metrics, arrays


def verify_normalization_checkpoint(directory, binding):
    """Verify the deliberately different dataset and model statistics origins."""
    import numpy as np
    directory = Path(directory)
    provenance = native.verify_grasp_checkpoint(directory, binding)
    manifest = native._read(directory / "manifest.json")
    statistics = native._read(directory / "train_stats.json")
    config = native._read(directory / "experiment_config.json")
    _stats(statistics, np)
    if (binding.get("model_statistics_source") != MODEL_STATISTICS_SOURCE
            or binding.get("dataset_manifest_statistics_not_modified") is not True
            or binding.get("model_statistics_are_train_subset") is not True
            or binding.get("model_statistics_training_episode_ids") != list(ORIGINAL_TRAIN)
            or binding.get("dataset_training_episode_ids") != list(TRAIN)
            or set(manifest.get("split_episode_ids", {}).get("train", [])) != set(TRAIN)
            or manifest.get("split_episode_ids", {}).get("validation") != list(VALIDATION)
            or manifest.get("statistics", {}).get("sha256") != binding.get("dataset_statistics_sha256")
            or binding.get("data_statistics_sha256") != binding.get("dataset_statistics_sha256")
            or native._digest(statistics) != binding.get("model_statistics_content_sha256")
            or binding.get("model_statistics_source_file_sha256") != binding.get("model_statistics_sha256")
            or binding.get("statistics_sha256") != binding.get("model_statistics_sha256")
            or binding.get("normalization_statistics_sha256") != binding.get("model_statistics_sha256")
            or config.get("normalization_attribution", {}).get("model_statistics_source") != MODEL_STATISTICS_SOURCE):
        raise NormalizationAttributionError("checkpoint dataset statistics and original-six model statistics origins are not explicitly distinct and bound")
    return provenance


def _save_checkpoint(directory, config, manifest, statistics, binding, initialization, state, step, metrics):
    result = residual._save_checkpoint(directory, config, manifest, statistics, binding, initialization, state, step, metrics)
    verify_normalization_checkpoint(directory, binding)
    result.update(model_statistics_source=MODEL_STATISTICS_SOURCE,
        model_statistics_sha256=binding["model_statistics_sha256"], dataset_statistics_sha256=binding["dataset_statistics_sha256"],
        requires_explicit_original_six_statistics_decoder=True, compatible_with_expanded_statistics_loader=False)
    return result


def run_normalization_admission(config, run_dir):
    """Audit all raw anchors, actual step-zero inference, and discarded probes."""
    declared, path, report = _stage(config, run_dir, "normalization_admission")
    started, torch = time.perf_counter(), None
    try:
        if config["train"].get("enabled") is not True or config["train"].get("authorization") != "conditional":
            raise NormalizationAttributionError("discarded residual optimizer probes require the unified conditional training admission")
        prepared = _prepare(config, declared, report)
        settings, _, statistics, datasets, _, np, torch, device, blueprint, rng, schedule, _ = prepared
        report.update(optimizer_created=False, formal_optimizer_created=False, variants={})
        report["roundtrip_audit"] = _roundtrip_audit(datasets, statistics, np)
        for variant in VARIANTS:
            policy, initialization = _variant_policy(config, settings, variant, blueprint, np, torch, device, rng)
            if initialization != report["frozen_variant_initialization"][variant]:
                raise NormalizationAttributionError("the fresh zero-head model differs from its frozen stage8 initialization")
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
                raise NormalizationAttributionError("the temporary memory model differs from the measured zero-origin model")
        report.update(status="passed_normalization_exact_zero_state_hold_admission", passed=True,
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

def _verify_admission(config, declared, binding, report, datasets):
    path = native._resolve(config, declared["admission_report_path"])
    admission = native._read(path)
    if (admission.get("passed") is not True
            or admission.get("status") != "passed_normalization_exact_zero_state_hold_admission"
            or admission.get("bindings") != binding or admission.get("settings") != declared
            or admission.get("formal_optimizer_created") is not False or admission.get("optimizer_steps") != 0
            or admission.get("discarded_probe_optimizer_steps") != len(VARIANTS)
            or admission.get("optimizer_steps_including_discarded_probes") != len(VARIANTS)
            or admission.get("test_payloads_opened") is not False or admission.get("test_used_for_selection") is not False
            or admission.get("all_anchor_zero_residual_identity_bitwise") is not True
            or admission.get("actual_network_step_zero_train_validation_equals_state_hold") is not True):
        raise NormalizationAttributionError("formal residual training requires the exact current config/code/data zero-origin admission")
    for name in ("environment", "runtime_sources", "architecture", "full_rgb_initial_blueprint_sha256", "post_rgb_blueprint_rng_sha256", "anchor_schedule",
                 "frozen_variant_initialization", "normalization_treatment", "model_statistics_origin", "sampler_exact_stage8_replay"):
        if report[name] != admission.get(name):
            raise NormalizationAttributionError("the current residual runtime differs from its zero-origin admission " + name)
    for split in ("train", "validation"):
        expected = len(datasets[split])
        audit = admission.get("roundtrip_audit", {}).get(split, {})
        error = audit.get("max_encode_decode_error_rad")
        if (audit.get("complete_anchor_count") != expected or audit.get("all_anchor_zero_residual_identity_bitwise") is not True
                or isinstance(error, bool) or not isinstance(error, (int, float))
                or not math.isfinite(error) or not 0 <= error <= 1e-6):
            raise NormalizationAttributionError("admission must audit every admitted expanded-train/frozen-validation complete raw anchor")
    for variant in VARIANTS:
        entry = admission.get("variants", {}).get(variant, {})
        if entry.get("initialization", {}).get("residual_head", {}).get("zero_final_action_head_verified") is not True:
            raise NormalizationAttributionError("the admitted residual output head is not exactly zero")
        probe = entry.get("memory_probe", {})
        reserved = probe.get("peak_reserved_bytes")
        if (probe.get("passed") is not True or probe.get("probe_optimizer_steps") != 1
                or probe.get("formal_training_steps") != 0 or probe.get("temporary_parameters_discarded") is not True
                or probe.get("finite_gradients") is not True or probe.get("full_validation_batch_passed") is not True
                or probe.get("initialization") != entry["initialization"]
                or isinstance(reserved, bool) or not isinstance(reserved, int)
                or not 0 < reserved <= declared["max_gpu_reserved_bytes"]):
            raise NormalizationAttributionError("each residual arm requires its exact discarded optimizer/evaluation memory probe")
        for split in ("train", "validation"):
            item = entry.get("step_zero", {}).get(split, {})
            artifact = item.get("predictions", {})
            if sha256_file(artifact.get("path", "")) != artifact.get("sha256"):
                raise NormalizationAttributionError("an admitted step-zero prediction artifact changed")
            if (item.get("metrics", {}).get("zero_origin_verified") is not True
                    or item["metrics"]["policy"] != item["metrics"]["state_hold"]):
                raise NormalizationAttributionError("the admitted step-zero policy is not the exact state-hold control")
    report["zero_origin_admission"] = {"path": str(path), "sha256": sha256_file(path), "same_full_binding_verified": True,
                                       "formal_optimizer_steps": 0, "discarded_probe_optimizer_steps": len(VARIANTS),
                                       "prediction_files_rehashed": True}
    return admission

def train_normalization_attribution(config, run_dir):
    """Run the fixed paired residual budgets; explicitly retain hold as step zero."""
    declared, path, report = _stage(config, run_dir, "normalization_attribution")
    started, torch = time.perf_counter(), None
    report.update(variants={}, matched_anchor_schedule=True, optimizer_steps_including_discarded_probes=0)
    try:
        if config["train"].get("enabled") is not True or config["train"].get("authorization") != "conditional":
            raise NormalizationAttributionError("the unified conditional training admission must be enabled")
        output = native._resolve(config, declared["output_dir"])
        if output.exists():
            raise NormalizationAttributionError("residual variants never overwrite existing output or resume prior optimizer state")
        prepared = _prepare(config, declared, report)
        settings, manifest, statistics, datasets, binding, np, torch, device, blueprint, rng, schedule, identities = prepared
        admission = _verify_admission(config, declared, binding, report, datasets)
        report["prior_admission_probe_optimizer_steps"] = admission["discarded_probe_optimizer_steps"]
        report["discarded_probes_performed_this_run"] = 0
        output.mkdir(parents=True, exist_ok=False)
        report["anchor_schedule_artifact"] = diagnostics._save_arrays(output / "training_anchor_schedule.npz",
            {"sample_episode_frame": identities, "dataset_sample_indices": schedule}, np)
        report["checkpoint_selection"] = {"include_measured_step_zero": True, "criterion": "validation.policy.all_28.mae",
                                           "best_trained_and_last_step_reported_separately": True, "test_used": False}
        stats = _stats(statistics, np)
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
                raise NormalizationAttributionError("the formal residual initialization differs from its probe or zero-origin admission")
            metrics, _ = _evaluate(policy, datasets["validation"], statistics, settings, np, torch, device,
                                    variant=variant, deadline=deadline, require_zero=True)
            if metrics != admission["variants"][variant]["step_zero"]["validation"]["metrics"]:
                raise NormalizationAttributionError("formal step-zero validation differs from the admitted exact state-hold identity")
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
                    raise NormalizationAttributionError("a residual arm exhausted its unchanged wall budget before all 2000 steps")
                step = offset + 1
                policy.train()
                batch, _ = _training_batch(datasets["train"], indices, stats, variant, np, torch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, components = policy(batch)
                if not bool(torch.isfinite(loss)) or "kld_loss" in components:
                    raise NormalizationAttributionError("residual training loss is nonfinite or includes KL")
                loss.backward()
                gradients = diagnostics._gradient_report(policy, torch)
                if not gradients["finite"]:
                    raise NormalizationAttributionError("residual training gradients are missing or nonfinite")
                optimizer.step()
                report["training_started"], entry["training_started"] = True, True
                report["optimizer_steps"] += 1
                entry["optimizer_steps"] = step
                torch.cuda.synchronize()
                if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
                    raise NormalizationAttributionError("a formal residual optimizer produced nonfinite parameters")
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
                raise NormalizationAttributionError("online checkpoint tracking differs from independent step-zero-aware selection")
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
        report["status"], report["passed"] = "completed_old_statistics_expanded_train_normalization_comparison", True
        report["limitations"] = ["Only model normalization changes against the frozen fourteen-train reference: original-six state/RGB statistics and action residual scales replace fourteen-train statistics; training membership, anchors, initialization, budgets and raw validation bytes remain fixed.",
                                  "State hold is selectable as the measured zero-step checkpoint; best trained and final validation scores are also reported without hiding regressions.",
                                  "Residual heads are deliberately zero initialized, so the prediction parameterization and output-head origin both differ from previous absolute controls.",
                                  "Scratch RGB and state-only have different parameter counts and FLOPs; no new pretrained arm is introduced in this isolated residual comparison.",
                                  "One seed and three repeatedly used validation trajectories support development decisions, not independent-test or physical-success claims.",
                                  "This joint normalization treatment changes state inputs, RGB inputs and action residual scales/loss weighting; it does not separately identify their individual effects. Raw radian errors are the comparison metric.",
                                  "Storage-file separation does not verify collection session independence; candidate addition improves same-source coverage only.",
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
