"""Measured native ACT inference from pinned real observations, without control.

The compact public policy has a 29-state/20-action protocol. It is a compute
baseline on its original training dataset, not a GraspSquare adapter or a
closed-loop G1 result. Loading and measurements use the original LeRobot ACT
implementation and saved processor pipelines; no optimizer is constructed.
"""

import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
import traceback

from .artifacts import sha256_file


class OfflineInferenceError(ValueError):
    """Native evidence or runtime cannot support a real offline inference probe."""


CAMERA = "observation.images.global_view"
MODEL_REPO = "ambitiousmangosteen/g1-dex3-pick-act"
MODEL_REVISION = "a9a8c4e2bd523ebe5f3dc360188741aee438bbe3"
DATASET_REPO = "ambitiousmangosteen/g1-dex3-pick"
DATASET_REVISION = "b4e1bb305a7355fbb41ea82ff75dbf272a3893ad"


def _json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def _write_new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _inside(root, value):
    root = Path(root).resolve()
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if path.is_symlink():
        raise OfflineInferenceError("native evidence must not be a symlink")
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise OfflineInferenceError("native evidence must remain inside its declared root") from exc
    return path


def verify_native_provenance(directory, repo_id, revision, required_files):
    """Verify downloader receipts and full hashes before processing local inputs."""
    directory = Path(directory).resolve()
    receipt = _json(directory / "provenance.json")
    if receipt.get("repo_id") != repo_id or receipt.get("revision") != revision:
        raise OfflineInferenceError("native provenance does not match configured pinned identity")
    if receipt.get("training_started"):
        raise OfflineInferenceError("download receipt unexpectedly records training")
    entries = {}
    for item in receipt.get("files", []):
        name = item.get("relative_path")
        if not isinstance(name, str) or name in entries:
            raise OfflineInferenceError("native provenance has missing or duplicate relative paths")
        path = _inside(directory, name)
        if item.get("path") is not None and Path(item["path"]).resolve() != path:
            raise OfflineInferenceError("native provenance absolute and relative paths disagree")
        entries[name] = (path, item)
    verified = {}
    for name in required_files:
        if name not in entries:
            raise OfflineInferenceError("native provenance does not include required file: {}".format(name))
        path, item = entries[name]
        if not path.is_file() or not item.get("full_file_verified"):
            raise OfflineInferenceError("native provenance requires complete verified files")
        if path.stat().st_size != item.get("size_bytes"):
            raise OfflineInferenceError("native evidence file size differs from downloader receipt")
        digest = sha256_file(path)
        if digest != item.get("sha256") or (item.get("expected_lfs_sha256") and digest != item["expected_lfs_sha256"]):
            raise OfflineInferenceError("native evidence file hash differs from pinned receipt")
        verified[name] = {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}
    return verified


def validate_native_act_metadata(model, info, preprocessor, postprocessor, settings):
    """Reject dimensional padding, camera aliasing and processor substitutions."""
    if (settings.get("repo_id"), settings.get("revision")) != (MODEL_REPO, MODEL_REVISION):
        raise OfflineInferenceError("this native ACT backend supports only its audited pinned checkpoint")
    if (settings.get("dataset_repo_id"), settings.get("dataset_revision")) != (DATASET_REPO, DATASET_REVISION):
        raise OfflineInferenceError("this native ACT backend requires the checkpoint's audited original dataset")
    if settings.get("backend") != "lerobot_act_native":
        raise OfflineInferenceError("unsupported offline policy backend")
    if model.get("type") != "act" or model.get("n_obs_steps") != 1 or model.get("chunk_size") != 100:
        raise OfflineInferenceError("native ACT architecture or horizon changed")
    inputs, outputs = model.get("input_features", {}), model.get("output_features", {})
    expected_inputs = {"observation.state": {"type": "STATE", "shape": [29]},
                       CAMERA: {"type": "VISUAL", "shape": [3, 480, 640]}}
    if inputs != expected_inputs or outputs != {"action": {"type": "ACTION", "shape": [20]}}:
        raise OfflineInferenceError("native ACT input/output identities or shapes changed")
    if model.get("normalization_mapping") != {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}:
        raise OfflineInferenceError("native saved normalization must remain MEAN_STD")
    features = info.get("features", {})
    if info.get("codebase_version") != "v3.0" or info.get("fps") != 30:
        raise OfflineInferenceError("native data layout or cadence changed")
    for key, shape in (("observation.state", [29]), ("action", [20]), (CAMERA, [480, 640, 3])):
        if features.get(key, {}).get("shape") != shape:
            raise OfflineInferenceError("native dataset feature shape differs from checkpoint")
    state_names, action_names = features["observation.state"].get("names"), features["action"].get("names")
    if (not isinstance(state_names, list) or len(state_names) != 29 or len(set(state_names)) != 29
            or not isinstance(action_names, list) or len(action_names) != 20 or len(set(action_names)) != 20):
        raise OfflineInferenceError("native joint/action identities are missing or duplicated")
    if action_names[14:] != ["remote.lx", "remote.ly", "remote.rx", "remote.ry", "left_hand_open", "right_hand_open"]:
        raise OfflineInferenceError("native action tail is not the audited remote and hand-open protocol")
    if action_names[:14] != state_names[15:]:
        raise OfflineInferenceError("native arm identities do not agree between state and action")
    pre_steps, post_steps = preprocessor.get("steps", []), postprocessor.get("steps", [])
    if [step.get("registry_name") for step in pre_steps] != [
            "rename_observations_processor", "to_batch_processor", "device_processor", "normalizer_processor"]:
        raise OfflineInferenceError("native preprocessor steps changed")
    if [step.get("registry_name") for step in post_steps] != ["unnormalizer_processor", "device_processor"]:
        raise OfflineInferenceError("native postprocessor steps changed")
    if pre_steps[0].get("config", {}).get("rename_map") != {}:
        raise OfflineInferenceError("native camera/state renaming is not allowed")
    if pre_steps[2].get("config", {}).get("device") != settings.get("device"):
        raise OfflineInferenceError("native processor device does not match the configured probe")
    if post_steps[1].get("config", {}).get("device") != "cpu":
        raise OfflineInferenceError("native postprocessor must return CPU actions")
    expected = {**expected_inputs, "action": {"type": "ACTION", "shape": [20]}}
    for step, expected_features in ((pre_steps[3], expected), (post_steps[0], {"action": expected["action"]})):
        value = step.get("config", {})
        if value.get("features") != expected_features or value.get("norm_map") != model["normalization_mapping"]:
            raise OfflineInferenceError("native processor feature/normalization configuration differs from checkpoint")
        epsilon = value.get("eps")
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise OfflineInferenceError("native processor epsilon must be finite and positive")
        state_file = step.get("state_file")
        if not isinstance(state_file, str) or Path(state_file).name != state_file or not state_file.endswith(".safetensors"):
            raise OfflineInferenceError("native processor requires its local saved safetensors statistics")
    frames = settings.get("frame_indices")
    if (not isinstance(frames, list) or not frames or len(set(frames)) != len(frames)
            or any(isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 for frame in frames)):
        raise OfflineInferenceError("native frame selection must contain unique non-negative indices")
    for key, minimum in (("warmup_runs", 0), ("measured_runs", 1), ("episode_id", 0)):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise OfflineInferenceError("{} must be an integer at least {}".format(key, minimum))
    return {"state_names": state_names, "action_names": action_names, "camera_key": CAMERA,
            "camera_shape_hwc": [480, 640, 3], "fps": 30, "state_dim": 29, "action_dim": 20,
            "chunk_size": 100, "pre_stats_file": pre_steps[3]["state_file"],
            "post_stats_file": post_steps[0]["state_file"], "pre_eps": pre_steps[3]["config"]["eps"]}


def _array_sha(array):
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_native_rows(directory, info, settings, parquet):
    episodes = []
    for path in sorted((directory / "meta/episodes").glob("chunk-*/*.parquet")):
        episodes.extend(parquet.read_table(path).to_pylist())
    matching = [row for row in episodes if int(row["episode_index"]) == settings["episode_id"]]
    if len(matching) != 1:
        raise OfflineInferenceError("selected native episode has no unique metadata row")
    episode = matching[0]
    data_name = info["data_path"].format(chunk_index=int(episode["data/chunk_index"]),
                                          file_index=int(episode["data/file_index"]))
    video_name = info["video_path"].format(video_key=CAMERA,
                                            chunk_index=int(episode["videos/{}/chunk_index".format(CAMERA)]),
                                            file_index=int(episode["videos/{}/file_index".format(CAMERA)]))
    rows = [row for row in parquet.read_table(_inside(directory, data_name)).to_pylist()
            if int(row["episode_index"]) == settings["episode_id"]]
    rows.sort(key=lambda row: int(row["frame_index"]))
    if not rows or len(rows) != int(episode["length"]):
        raise OfflineInferenceError("native selected episode length differs from metadata")
    if [int(row["frame_index"]) for row in rows] != list(range(len(rows))):
        raise OfflineInferenceError("native episode frame indices are incomplete or duplicated")
    start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
    if end - start != len(rows) or [int(row["index"]) for row in rows] != list(range(start, end)):
        raise OfflineInferenceError("native global row indices differ from episode boundaries")
    times = [float(row["timestamp"]) for row in rows]
    if any(not math.isfinite(value) or abs(value - index / info["fps"]) > 1e-4 for index, value in enumerate(times)):
        raise OfflineInferenceError("native timestamps do not match the declared frame cadence")
    if any(frame >= len(rows) for frame in settings["frame_indices"]):
        raise OfflineInferenceError("selected native observation frame is outside its episode")
    offset = float(episode["videos/{}/from_timestamp".format(CAMERA)])
    if not math.isfinite(offset) or offset < 0:
        raise OfflineInferenceError("native video episode offset is invalid")
    queries = [offset + times[frame] for frame in settings["frame_indices"]]
    return rows, episode, data_name, video_name, queries


def decode_native_rgb(path, query_times_s, av, *, tolerance_s=1e-4):
    """Decode real RGB and match source PTS, never synthesize a camera frame."""
    wanted = sorted(enumerate(query_times_s), key=lambda item: item[1])
    if any(not math.isfinite(value) or value < 0 for _, value in wanted):
        raise OfflineInferenceError("native video query timestamps must be finite and non-negative")
    results, cursor = {}, 0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise OfflineInferenceError("native source has no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        codec = stream.codec_context.name
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise OfflineInferenceError("native video frame has no source PTS/time base")
            time_s = float(frame.pts * frame.time_base)
            while cursor < len(wanted) and abs(time_s - wanted[cursor][1]) <= tolerance_s:
                index, query = wanted[cursor]
                rgb = frame.to_ndarray(format="rgb24")
                if list(rgb.shape) != [480, 640, 3] or str(rgb.dtype) != "uint8":
                    raise OfflineInferenceError("native decoded camera shape or pixel dtype changed")
                results[index] = (rgb.copy(), {"source_pts": int(frame.pts),
                                              "source_time_base": str(frame.time_base),
                                              "source_time_s": time_s, "query_time_s": query,
                                              "absolute_error_s": abs(time_s - query), "codec": codec})
                cursor += 1
            if cursor == len(wanted):
                break
            if time_s > wanted[cursor][1] + tolerance_s:
                raise OfflineInferenceError("native camera PTS cannot be paired to the selected real observation")
    if len(results) != len(wanted):
        raise OfflineInferenceError("native video ended before all selected observations were decoded")
    return [results[index] for index in range(len(query_times_s))]


def _stats_validation(pre_stats, post_stats, protocol, torch):
    details = {}
    for key, shape in (("observation.state", (29,)), ("action", (20,)), (CAMERA, (3, 1, 1))):
        for suffix in ("mean", "std"):
            name = key + "." + suffix
            value = pre_stats.get(name)
            if value is None or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise OfflineInferenceError("native saved statistics shape/value mismatch: {}".format(name))
            if suffix == "std" and bool((value < 0).any()):
                raise OfflineInferenceError("native normalization standard deviations cannot be negative")
        details[key] = {"mean_sha256": _array_sha(pre_stats[key + ".mean"].cpu().numpy()),
                        "std_sha256": _array_sha(pre_stats[key + ".std"].cpu().numpy()),
                        "zero_std_indices": (pre_stats[key + ".std"].reshape(-1) == 0).nonzero().reshape(-1).tolist()}
    for suffix in ("mean", "std"):
        name = "action." + suffix
        if name not in post_stats or not torch.equal(pre_stats[name], post_stats[name]):
            raise OfflineInferenceError("native pre/post saved action statistics disagree")
    return details


def _parse_act_config(model_dir, act_config_type, pretrained_config_type):
    # The saved JSON includes "type": "act". The generic registry loader must
    # select that class before draccus decodes its fields; a concrete ACTConfig
    # loader treats the choice discriminator as an unexpected field.
    value = pretrained_config_type.from_pretrained(str(model_dir), local_files_only=True)
    if not isinstance(value, act_config_type):
        raise OfflineInferenceError("saved native configuration did not select the registered ACT class")
    return value


def run_native_act_inference(config, run_dir):
    """Run pinned native FP32 policy with real sensor inputs and immutable outputs."""
    directory = Path(run_dir)
    report_path, array_path = directory / "offline_inference.json", directory / "native_act_predictions.npz"
    if not directory.is_dir() or report_path.exists() or array_path.exists():
        raise OfflineInferenceError("offline inference requires an existing run with new output names")
    settings = copy.deepcopy(config["offline_policy"])
    report = {"schema_version": 1, "inference_protocol": "native_act_saved_processors_fp32_v1",
              "status": "failed", "role": "compute_baseline_on_original_training_data",
              "training_started": False, "optimizer_created": False, "closed_loop_control": False,
              "held_out_evaluation": False, "project_adapter_validated": False,
              "grasp_square_adapter_validated": False, "training_eligible": False,
              "settings": settings, "samples": []}
    torch = None
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        import pyarrow.parquet as parquet
        import av
        from safetensors.torch import load_file
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

        root = Path(config["paths"]["root"]).resolve()
        model_dir, dataset_dir = (_inside(root, settings[key]) for key in ("model_dir", "dataset_dir"))
        model_json, info = _json(model_dir / "config.json"), _json(dataset_dir / "meta/info.json")
        pre_json, post_json = _json(model_dir / "policy_preprocessor.json"), _json(model_dir / "policy_postprocessor.json")
        protocol = validate_native_act_metadata(model_json, info, pre_json, post_json, settings)
        model_files = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
                       protocol["pre_stats_file"], protocol["post_stats_file"]]
        report["model_files"] = verify_native_provenance(model_dir, settings["repo_id"], settings["revision"], model_files)
        episode_files = [str(path.relative_to(dataset_dir)) for path in sorted((dataset_dir / "meta/episodes").glob("chunk-*/*.parquet"))]
        report["source_files"] = verify_native_provenance(dataset_dir, settings["dataset_repo_id"], settings["dataset_revision"],
                                                         ["meta/info.json"] + episode_files)
        rows, episode, data_name, video_name, queries = _load_native_rows(dataset_dir, info, settings, parquet)
        report["source_files"].update(verify_native_provenance(dataset_dir, settings["dataset_repo_id"], settings["dataset_revision"],
                                                              [data_name, video_name]))
        frames = decode_native_rgb(_inside(dataset_dir, video_name), queries, av)
        pre_stats, post_stats = load_file(str(model_dir / protocol["pre_stats_file"])), load_file(str(model_dir / protocol["post_stats_file"]))
        report["normalization_statistics"] = _stats_validation(pre_stats, post_stats, protocol, torch)
        report["native_protocol"] = protocol
        report["episode_metadata"] = {key: value for key, value in episode.items()
                                      if not key.startswith("stats/")}
        report["environment"] = {name: importlib.metadata.version(name) for name in
                                 ("torch", "torchvision", "lerobot", "safetensors", "av", "pyarrow", "huggingface_hub")}
        report["environment"]["torch_cuda"] = torch.version.cuda
        device = torch.device(settings["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise OfflineInferenceError("this 8GB feasibility probe requires the configured real CUDA device")
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device_index)
        report["gpu"] = {"name": torch.cuda.get_device_name(device_index), "total_bytes": total,
                         "free_bytes_before_model_load": free, "device_index": device_index}
        torch.cuda.reset_peak_memory_stats(device_index)

        act_config = _parse_act_config(model_dir, ACTConfig, PreTrainedConfig)
        # These initialization weights are fully replaced by the public policy;
        # strict=True below ensures no randomly initialized backbone is retained.
        act_config.pretrained_backbone_weights = None
        act_config.device = settings["device"]
        policy = ACTPolicy(act_config)
        state_dict = load_file(str(model_dir / "model.safetensors"), device="cpu")
        loaded = policy.load_state_dict(state_dict, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise OfflineInferenceError("native full checkpoint did not load strictly")
        report["checkpoint_load"] = {"strict": True, "tensor_count": len(state_dict),
                                     "parameter_count": sum(value.numel() for value in policy.parameters()),
                                     "pretrained_backbone_initialization": None,
                                     "retained_random_initialization": False, "dtype": "float32"}
        del state_dict
        policy.requires_grad_(False).to(device=device, dtype=torch.float32).eval()
        preprocessor = PolicyProcessorPipeline.from_pretrained(str(model_dir), config_filename="policy_preprocessor.json",
                                                               local_files_only=True, token=False)
        postprocessor = PolicyProcessorPipeline.from_pretrained(str(model_dir), config_filename="policy_postprocessor.json",
                                                                local_files_only=True, token=False,
                                                                to_transition=policy_action_to_transition,
                                                                to_output=transition_to_policy_action)
        predictions, normalized_predictions, states, rgb_images, future_targets, future_valid = [], [], [], [], [], []
        for frame_index, (rgb, pts) in zip(settings["frame_indices"], frames):
            row = rows[frame_index]
            state = np.asarray(row["observation.state"], dtype=np.float32)
            if state.shape != (29,) or not np.isfinite(state).all():
                raise OfflineInferenceError("native row state is not a finite real 29-joint observation")
            raw_batch = {"observation.state": torch.from_numpy(state.copy()),
                         CAMERA: torch.from_numpy(np.transpose(rgb.copy(), (2, 0, 1))).to(torch.float32) / 255.0}
            input_sha = hashlib.sha256((_array_sha(state) + _array_sha(rgb)).encode()).hexdigest()
            measurements, first_prediction, repeated_max_error = [], None, 0.0
            normalized_saved = None
            for iteration in range(settings["warmup_runs"] + settings["measured_runs"]):
                torch.cuda.synchronize(device_index)
                wall_start = time.perf_counter()
                prepared = preprocessor(raw_batch)
                for key in raw_batch:
                    expected = ((raw_batch[key] - pre_stats[key + ".mean"]) /
                                (pre_stats[key + ".std"] + protocol["pre_eps"])).unsqueeze(0)
                    if not torch.allclose(prepared[key].cpu(), expected, atol=1e-6, rtol=1e-6):
                        raise OfflineInferenceError("original preprocessor does not match its saved normalization statistics")
                event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.inference_mode():
                    event_start.record()
                    normalized = policy.predict_action_chunk(prepared)
                    event_end.record()
                    prediction = postprocessor(normalized)
                torch.cuda.synchronize(device_index)
                elapsed_ms = (time.perf_counter() - wall_start) * 1000
                if tuple(prediction.shape) != (1, 100, 20) or not bool(torch.isfinite(prediction).all()):
                    raise OfflineInferenceError("native policy did not produce a finite full 100x20 action chunk")
                expected_action = normalized.detach().cpu() * post_stats["action.std"] + post_stats["action.mean"]
                if not torch.allclose(prediction.cpu(), expected_action, atol=1e-6, rtol=1e-6):
                    raise OfflineInferenceError("original postprocessor does not match saved action statistics")
                current = prediction.detach().cpu().numpy()[0].copy()
                if first_prediction is None:
                    first_prediction, normalized_saved = current, normalized.detach().cpu().numpy()[0].copy()
                repeated_max_error = max(repeated_max_error, float(np.max(np.abs(current - first_prediction))))
                if iteration >= settings["warmup_runs"]:
                    measurements.append({"wall_ms_preprocess_model_postprocess": elapsed_ms,
                                         "model_cuda_elapsed_ms": event_start.elapsed_time(event_end)})
            target = np.full((100, 20), np.nan, dtype=np.float32)
            valid = np.zeros(100, dtype=np.bool_)
            for offset, future_row in enumerate(rows[frame_index:frame_index + 100]):
                action = np.asarray(future_row["action"], dtype=np.float32)
                if action.shape != (20,) or not np.isfinite(action).all():
                    raise OfflineInferenceError("native future recorded action is not finite or has the wrong protocol")
                target[offset], valid[offset] = action, True
            difference = first_prediction[valid] - target[valid]
            sample = {"episode_id": settings["episode_id"], "frame_index": frame_index,
                      "source_global_index": int(row["index"]), "source_timestamp_s": float(row["timestamp"]),
                      "input_sha256": input_sha, "raw_state_sha256": _array_sha(state),
                      "raw_rgb_sha256": _array_sha(rgb), "video": pts,
                      "measurements": measurements, "same_input_max_absolute_repeat_error": repeated_max_error,
                      "native_future_valid_steps": int(valid.sum()),
                      "native_training_sample_mae_by_action_channel": np.mean(np.abs(difference), axis=0).tolist(),
                      "native_training_sample_mse_by_action_channel": np.mean(difference ** 2, axis=0).tolist(),
                      "finite_output": True, "normalization_processor_parity": True,
                      "prediction_sha256": _array_sha(first_prediction)}
            report["samples"].append(sample)
            predictions.append(first_prediction); normalized_predictions.append(normalized_saved)
            states.append(state); rgb_images.append(rgb); future_targets.append(target); future_valid.append(valid)
        with array_path.open("xb") as stream:
            np.savez_compressed(stream, prediction_native_actions=np.stack(predictions),
                                normalized_action_chunks=np.stack(normalized_predictions),
                                observation_state=np.stack(states), observation_rgb=np.stack(rgb_images),
                                native_future_recorded_actions=np.stack(future_targets),
                                native_future_valid_mask=np.stack(future_valid),
                                frame_indices=np.asarray(settings["frame_indices"], dtype=np.int64),
                                native_action_names=np.asarray(protocol["action_names"]),
                                native_state_names=np.asarray(protocol["state_names"]))
        report["prediction_artifact"] = {"path": str(array_path), "sha256": sha256_file(array_path)}
        all_wall = [item["wall_ms_preprocess_model_postprocess"] for sample in report["samples"] for item in sample["measurements"]]
        all_model = [item["model_cuda_elapsed_ms"] for sample in report["samples"] for item in sample["measurements"]]
        report["latency"] = {"measured_calls": len(all_wall), "warmup_calls": len(frames) * settings["warmup_runs"],
                             "wall_ms_median": float(np.median(all_wall)), "wall_ms_max": max(all_wall),
                             "model_cuda_ms_median": float(np.median(all_model)), "model_cuda_ms_max": max(all_model),
                             "video_io_included": False, "normalization_parity_checks_included_in_wall_time": True}
        report["status"] = "passed_native_compute_probe"
        report["inference_passed"] = True
        report["limitations"] = ["Selected real observations belong to the checkpoint's original training dataset; errors are diagnostic, not held-out accuracy.",
                                 "Only native 29-state/20-action single-camera inputs were used; no GraspSquare or project controller mapping is certified.",
                                 "Measured offline GPU capacity and latency do not establish real-time closed-loop control or grasp success.",
                                 "No optimizer, backward pass, training, simulated rollout or real robot command was executed."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc(limit=8)}
        report["inference_passed"] = False
    finally:
        if torch is not None and torch.cuda.is_available() and "gpu" in report:
            device_index = report["gpu"]["device_index"]
            report["gpu"].update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
                                   "peak_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
                                   "free_bytes_after_probe": torch.cuda.mem_get_info(device_index)[0]})
        report["total_elapsed_s"] = time.perf_counter() - started
        _write_new_json(report_path, report)
    return report
