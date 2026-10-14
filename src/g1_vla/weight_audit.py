"""Anonymous, revision-pinned metadata audit of public policy checkpoints.

This stage never downloads model tensors, imports remote code, allocates CUDA
memory, modifies dependencies, or trains. A storage/memory estimate is not an
inference result. Native policy protocols remain distinct from this project's
31-joint controller and the official 28-joint GraspSquare data.
"""

import copy
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class WeightAuditError(ValueError):
    """Checkpoint identity or metadata cannot support the requested audit."""


_SOURCE_N15 = "4af2b622892f7dcb5aae5a3fb70bcb02dc217b96"
_SOURCE_N16 = "9b37aa1ce69c73c6d165233fa88128283bba4508"
_SOURCE_UNIFOLM = "ff6c39aeb0454cfb95418c66aef40ca777f935c1"

# Verified on 2026-10-14. Revisions are deliberately never resolved from main.
CATALOG = (
    {
        "repo_id": "nvidia/GR00T-N1.7-3B",
        "revision": "2fc962b973bccdd5d8ce4f67cc63b264d6886495",
        "role": "vla_foundation",
        "runtime": {"python_min": "3.12", "torch_min": "2.7", "cuda_min": "12.6",
                    "native_quantization": False, "native_cpu_offload": False},
        "dependencies": [{"repo_id": "nvidia/Cosmos-Reason2-2B", "gated": True}],
        "protocol_notes": ["Base pretrain embodiment heads are not an audited GraspSquare policy."],
        "sources": ["https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md",
                    "https://github.com/NVIDIA/Isaac-GR00T#zero-shot-inference-base-model"],
    },
    {
        "repo_id": "nvidia/GR00T-N1.6-G1-PnPAppleToPlate",
        "revision": "897d0313a190f46a2cccaeb34077752a0db4b0de",
        "role": "g1_vla_task_policy",
        "runtime": {"python_min": "3.10", "torch_min": "2.7.1", "cuda_min": "12.8",
                    "native_quantization": False, "native_cpu_offload": False},
        "dependencies": [],
        "native_protocol": {"state_dim": 43, "camera_keys": ["ego_view"],
                            "action_semantics": "relative_arms_absolute_hands_whole_body_commands",
                            "action_horizon": 30},
        "protocol_notes": ["Whole-body locomotion outputs require the original policy wrapper.",
                           "Vendored Eagle processor; its Hub repository is not needed by this source loader."],
        "sources": ["https://github.com/NVIDIA/Isaac-GR00T/blob/{}/gr00t/policy/gr00t_policy.py".format(_SOURCE_N16),
                    "https://github.com/NVIDIA/Isaac-GR00T/blob/{}/pyproject.toml".format(_SOURCE_N16)],
    },
    {
        "repo_id": "nvidia/GR00T-N1.5-3B",
        "revision": "869830fc749c35f34771aa5209f923ac57e4564e",
        "role": "vla_foundation",
        "runtime": {"python_min": "3.10", "cuda_min": "12.4",
                    "native_quantization": False, "native_cpu_offload": False},
        "dependencies": [],
        "protocol_notes": ["Released heads are GR1, OXE_DROID and AGIBOT_GENIE1, not a trained G1 head.",
                           "Official policy loader moves the entire model to one device; quantization/offload need a separately validated adapter."],
        "sources": ["https://github.com/NVIDIA/Isaac-GR00T/blob/{}/gr00t/model/policy.py".format(_SOURCE_N15),
                    "https://github.com/NVIDIA/Isaac-GR00T/blob/{}/README.md".format(_SOURCE_N15)],
    },
    {
        "repo_id": "unitreerobotics/UnifoLM-VLA-Base",
        "revision": "06fee5014922ba6791cfe48d1e4aefac995dd8a2",
        "weight_files": ["checkpoints/pytorch_model.pt"],
        "role": "vla_foundation",
        "runtime": {"python_min": "3.10", "native_quantization": False,
                    "native_cpu_offload": False},
        "dependencies": [{"repo_id": "unitreerobotics/UnifoLM-VLM-Base", "gated": False,
                          "revision": "da121034eb5681c085a17001853a0267632a796f",
                          "bf16_parameter_lower_bound_bytes": 16584333312}],
        "protocol_notes": ["Official loader constructs the VLM on CUDA, then loads the full state dictionary; no native 4-bit/offload option.",
                           "Action dimensions and normalization statistics must agree before a G1 data adapter can be enabled."],
        "sources": ["https://github.com/unitreerobotics/unifolm-vla/blob/{}/src/unifolm_vla/model/modules/vlm/QWen2_5.py".format(_SOURCE_UNIFOLM),
                    "https://github.com/unitreerobotics/unifolm-vla/blob/{}/src/unifolm_vla/model/framework/base_framework.py".format(_SOURCE_UNIFOLM)],
    },
    {
        "repo_id": "Fichtl00/Unifolm-VLA_mix",
        "revision": "438922f41c39ef2a28d8634b6762fba06d9a5c58",
        "weight_files": ["final_model/pytorch_model.pt"],
        "role": "community_g1_vla_task_policy",
        "runtime": {"python_min": "3.10", "native_quantization": False,
                    "native_cpu_offload": False},
        "dependencies": [{"repo_id": "unitreerobotics/UnifoLM-VLM-Base", "gated": False,
                          "revision": "da121034eb5681c085a17001853a0267632a796f",
                          "bf16_parameter_lower_bound_bytes": 16584333312}],
        "protocol_notes": ["Select only final_model, not all intermediate checkpoints or optimizer files.",
                           "28-dimensional outputs alone do not establish camera, joint identity or timing compatibility."],
        "sources": ["https://github.com/unitreerobotics/unifolm-vla/blob/{}/src/unifolm_vla/model/modules/vlm/QWen2_5.py".format(_SOURCE_UNIFOLM)],
    },
    {
        "repo_id": "ambitiousmangosteen/g1-dex3-pick-act",
        "revision": "a9a8c4e2bd523ebe5f3dc360188741aee438bbe3",
        "weight_files": ["model.safetensors"],
        "role": "compact_g1_task_policy_compute_baseline",
        "runtime": {"python_min": "3.10", "native_quantization": False,
                    "native_cpu_offload": False},
        "dependencies": [],
        "native_protocol": {"state_dim": 29, "action_dim": 20, "action_horizon": 100,
                            "camera_keys": ["observation.images.global_view"],
                            "fps": 30, "action_semantics": "14_arm_joints_4_remote_axes_2_hand_open_flags"},
        "native_dataset": {"repo_id": "ambitiousmangosteen/g1-dex3-pick",
                           "revision": "b4e1bb305a7355fbb41ea82ff75dbf272a3893ad"},
        "protocol_notes": ["ACT is a task-specific imitation policy, not a language-conditioned VLA foundation model.",
                           "Native 29-state/20-action protocol does not expose independent Dex3 finger joints and cannot be padded into GraspSquare 28 actions or the project's 31 actions."],
        "sources": ["https://huggingface.co/ambitiousmangosteen/g1-dex3-pick-act/blob/a9a8c4e2bd523ebe5f3dc360188741aee438bbe3/config.json",
                    "https://huggingface.co/datasets/ambitiousmangosteen/g1-dex3-pick/blob/b4e1bb305a7355fbb41ea82ff75dbf272a3893ad/meta/info.json"],
    },
)


def _identity(candidate):
    if not isinstance(candidate, dict):
        raise WeightAuditError("candidate must be an object")
    repo, revision = candidate.get("repo_id"), candidate.get("revision")
    if (not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo)
            or any(part in (".", "..") for part in repo.split("/"))):
        raise WeightAuditError("checkpoint must have an explicit owner/repository identity")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise WeightAuditError("checkpoint revision must be a full lowercase immutable commit SHA")
    return repo, revision


def _get_bytes(url, max_bytes=5 * 1024 * 1024):
    # No authentication headers or ambient HF token are read or transmitted.
    request = Request(url, headers={"User-Agent": "g1-vla-public-metadata-audit/0.1"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                value = response.read(max_bytes + 1)
            break
        except HTTPError as exc:
            if exc.code != 429 and exc.code < 500:
                raise
            if attempt == 2:
                raise
        except URLError:
            if attempt == 2:
                raise
        time.sleep(0.5 * (attempt + 1))
    if len(value) > max_bytes:
        raise WeightAuditError("metadata exceeds the bounded read limit")
    return value


def fetch_checkpoint_metadata(candidate):
    """Read only public Hub API plus existing config/processor/index/license files."""
    repo, revision = _identity(candidate)
    base = "https://huggingface.co/{}".format(quote(repo, safe="/"))
    api_url = "https://huggingface.co/api/models/{}/revision/{}?blobs=true".format(repo, revision)
    raw = _get_bytes(api_url)
    hub = json.loads(raw)
    if hub.get("sha") != revision:
        raise WeightAuditError("Hub response does not match the pinned revision")
    if hub.get("private") or hub.get("gated"):
        return {"hub": hub, "documents": {}, "blocked": "gated_or_private_repository"}
    siblings = {item["rfilename"]: item for item in hub.get("siblings", [])}
    documents = {}
    # No checkpoint or optimizer extension can enter this metadata allowlist.
    for filename in ("config.json", "processor_config.json", "model.safetensors.index.json",
                     "experiment_cfg/metadata.json", "README.md", "LICENSE", "LICENSE.txt"):
        if filename not in siblings:
            continue
        content = _get_bytes("{}/resolve/{}/{}".format(base, revision, filename))
        documents[filename] = {"sha256": hashlib.sha256(content).hexdigest(),
                               "size_bytes": len(content), "url": "{}/blob/{}/{}".format(base, revision, filename)}
        if filename.endswith(".json"):
            documents[filename]["value"] = json.loads(content)
        elif filename.lower().startswith("license"):
            documents[filename]["text"] = content.decode("utf-8")
    return {"hub": hub, "documents": documents, "api_url": api_url,
            "hub_metadata_sha256": hashlib.sha256(raw).hexdigest()}


def _version(value):
    values = re.findall(r"\d+", str(value).split("+")[0])
    return tuple(int(item) for item in values[:3]) + (0,) * max(0, 3 - len(values))


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise WeightAuditError("{} must be a finite non-negative number".format(name))
    return float(value)


def _local_software():
    packages = ("torch", "transformers", "accelerate", "bitsandbytes", "safetensors", "huggingface_hub", "lerobot")
    result = {"python": ".".join(map(str, sys.version_info[:3])), "packages": {}}
    for package in packages:
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    result["torch"] = result["packages"]["torch"]
    result["cuda"] = None  # Querying software metadata must not create a CUDA context.
    return result


def _policy_config(documents):
    document = documents.get("config.json", {})
    return document.get("value", document) if isinstance(document, dict) else {}


def _native_protocol(candidate, documents):
    config = _policy_config(documents)
    protocol = copy.deepcopy(candidate.get("native_protocol", {}))
    if config.get("type") == "act":
        inputs, outputs = config.get("input_features", {}), config.get("output_features", {})
        protocol.update({"state_dim": inputs.get("observation.state", {}).get("shape", [None])[-1],
                         "action_dim": outputs.get("action", {}).get("shape", [None])[-1],
                         "camera_keys": [key for key, value in inputs.items() if value.get("type") == "VISUAL"],
                         "action_horizon": config.get("chunk_size"), "n_obs_steps": config.get("n_obs_steps"),
                         "camera_shapes_chw": {key: value.get("shape") for key, value in inputs.items() if value.get("type") == "VISUAL"}})
    else:
        for source, target in (("action_dim", "padded_action_dim"), ("max_action_dim", "padded_action_dim"),
                               ("action_horizon", "model_action_horizon")):
            if source in config:
                protocol[target] = config[source]
    return protocol


def _memory_estimate(candidate, hub, hardware, reserve_mib):
    files = {item["rfilename"]: item for item in hub.get("siblings", [])}
    selected = candidate.get("weight_files")
    if selected is None:
        selected = [name for name in files if name.endswith(".safetensors") and
                    not name.startswith(("checkpoint-", "policy_")) and "optimizer" not in name]
    details = []
    for name in selected:
        if name not in files:
            raise WeightAuditError("selected model weight is absent from pinned metadata: {}".format(name))
        item = files[name]
        size = item.get("size", item.get("lfs", {}).get("size"))
        if size is None:
            details.append({"path": name, "size_bytes": None})
        else:
            details.append({"path": name, "size_bytes": int(_number(size, "weight size"))})
    bytes_on_disk = sum(item["size_bytes"] for item in details) if details and all(item["size_bytes"] is not None for item in details) else None
    tensors = hub.get("safetensors") or {}
    parameter_types = tensors.get("parameters", {})
    # The dtype counts can disagree with total (observed in the N1.7 API). Keep
    # both and use stored dtype counts when available; do not hide inconsistency.
    parameters = sum(parameter_types.values()) if parameter_types else tensors.get("total")
    bf16_bytes = int(parameters) * 2 if parameters else None
    free_mib = hardware.get("gpu_free_mib")
    if free_mib is not None:
        free_mib = _number(free_mib, "gpu_free_mib")
    total_mib = hardware.get("gpu_total_mib")
    if total_mib is not None:
        total_mib = _number(total_mib, "gpu_total_mib")
    dependency_bytes = max([item.get("bf16_parameter_lower_bound_bytes", 0)
                            for item in candidate.get("dependencies", [])] or [0])
    # The base VLM is included in full policy checkpoints. Use the maximum floor
    # instead of adding it twice; the reference loader constructs it on CUDA.
    execution_floor_bytes = max(bf16_bytes or 0, dependency_bytes) or None
    lower_mib = execution_floor_bytes / 1024 ** 2 if execution_floor_bytes is not None else None
    # A .pt archive's file size is not a trustworthy parameter-count estimator.
    status = "unknown_parameter_memory" if lower_mib is None else "gpu_memory_not_measured"
    if lower_mib is not None and free_mib is not None:
        status = "estimated_room_for_native_probe" if lower_mib + reserve_mib <= free_mib else "insufficient_estimated_native_headroom"
    return {"weight_files": details, "download_bytes": bytes_on_disk,
            "stored_parameter_types": parameter_types, "reported_parameter_total": tensors.get("total"),
            "parameter_counts_consistent": not parameter_types or sum(parameter_types.values()) == tensors.get("total"),
            "bf16_parameter_lower_bound_bytes": bf16_bytes,
            "dependency_bf16_lower_bound_bytes": dependency_bytes or None,
            "reference_execution_parameter_floor_bytes": execution_floor_bytes,
            "bf16_parameter_lower_bound_mib": lower_mib,
            "gpu_total_mib": total_mib, "gpu_free_mib": free_mib,
            "activation_reserve_mib_heuristic": reserve_mib,
            "estimated_status": status, "runtime_measured": False,
            "notes": ["Parameter storage is a lower bound; activations, CUDA context, allocator and image count are additional.",
                      "The activation reserve is a declared heuristic, not an observed peak."]}


def audit_public_weights(root, config, *, hardware=None, metadata=None):
    """Return pinned feasibility evidence; caller persists it in its normal run.

    ``hardware`` accepts gpu_total_mib, gpu_free_mib, python, torch, cuda and
    packages. ``metadata`` may inject repo-id keyed fixtures for offline tests.
    No static check marks a model as having successfully produced an action.
    """
    del root  # This metadata stage needs no project payloads or output directory.
    settings = config.get("public_weights", {})
    candidates = settings.get("candidates", CATALOG)
    if not isinstance(candidates, (list, tuple)) or not candidates:
        raise WeightAuditError("public_weights.candidates must be non-empty")
    reserve = _number(settings.get("activation_reserve_mib", 1024), "activation_reserve_mib")
    machine = copy.deepcopy(hardware if hardware is not None else _local_software())
    catalog = {item["repo_id"]: item for item in CATALOG}
    reports = []
    seen = set()
    for configured in candidates:
        repo, revision = _identity(configured)
        if (repo, revision) in seen:
            raise WeightAuditError("duplicate checkpoint identity")
        seen.add((repo, revision))
        candidate = copy.deepcopy(catalog.get(repo, {}))
        if candidate and candidate["revision"] != revision:
            # Source/API assertions about one revision must not silently bless another.
            candidate = {}
        candidate.update(copy.deepcopy(configured))
        entry = {"repo_id": repo, "revision": revision, "role": candidate.get("role", "unreviewed"),
                 "sources": candidate.get("sources", []), "inference_passed": False,
                 "project_adapter_validated": False, "training_eligible": False}
        try:
            evidence = metadata[repo] if metadata is not None else fetch_checkpoint_metadata(candidate)
            hub, documents = evidence["hub"], evidence.get("documents", {})
            if hub.get("sha") != revision:
                raise WeightAuditError("metadata revision does not match configured checkpoint")
            runtime = candidate.get("runtime", {})
            blockers = []
            for field, minimum in (("python", runtime.get("python_min")),
                                   ("torch", runtime.get("torch_min")), ("cuda", runtime.get("cuda_min"))):
                observed = machine.get(field)
                if minimum and observed and _version(observed) < _version(minimum):
                    blockers.append("{}_{}_below_required_{}".format(field, observed, minimum))
                elif minimum and not observed:
                    blockers.append("{}_version_unverified".format(field))
            if hub.get("private") or hub.get("gated"):
                blockers.append("gated_or_private_checkpoint")
            blockers.extend("gated_dependency:{}".format(item["repo_id"]) for item in candidate.get("dependencies", []) if item.get("gated"))
            memory = _memory_estimate(candidate, hub, machine, reserve)
            if memory["estimated_status"] == "insufficient_estimated_native_headroom":
                blockers.append("insufficient_estimated_native_gpu_headroom")
            license_name = (hub.get("cardData") or {}).get("license")
            entry.update({"status": "blocked_for_native_probe" if blockers else "metadata_audited_runtime_unverified",
                          "blockers": blockers, "runtime": runtime, "dependencies": candidate.get("dependencies", []),
                          "license": license_name, "license_document_present": any(name.lower().startswith("license") for name in documents),
                          "memory": memory, "native_protocol": _native_protocol(candidate, documents),
                          "protocol_notes": candidate.get("protocol_notes", []),
                          "native_dataset": candidate.get("native_dataset"),
                          "metadata_documents": {name: {key: value for key, value in document.items() if key not in ("value", "text")}
                                                 for name, document in documents.items()},
                          "hub_metadata_sha256": evidence.get("hub_metadata_sha256"),
                          "api_url": evidence.get("api_url")})
        except (HTTPError, URLError, OSError, ValueError, KeyError) as exc:
            entry.update({"status": "metadata_unavailable", "error": str(exc)})
        reports.append(entry)
    return {"schema_version": 1, "audit_protocol": "public_weights_revision_metadata_v1",
            "status": "metadata_audited" if all(item["status"] != "metadata_unavailable" for item in reports) else "partial_metadata_audit",
            "hardware": machine, "candidates": reports,
            "model_tensors_downloaded": False, "remote_code_executed": False,
            "training_started": False, "inference_passed": False,
            "project_adapter_validated": False,
            "next_gate": "Run the selected policy on audited original observations with its saved processors and measure actual peak VRAM; project deployment and training require separate mapping and evaluation gates."}
