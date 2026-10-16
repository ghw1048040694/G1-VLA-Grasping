"""Pinned provenance for zero-optimizer control diagnostics of the old actor.

This is a separate transfer/diagnostic bridge.  It does not relax rl-eval's
source/config/environment checks and never imports Torch or a simulator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath


_RUN = ".runtime/rebuild/runs/G1-REBUILD-20261016-rl-train-03"
_REFERENCE_RUNNER_VERSION = "g1_rsl5_bounded_ppo_capacity512_nograd_v3"
_REFERENCE_PINS = {
    "reference_manifest_sha256": "5213179bbb8b7d86d5aeab461400b0a2b37d7bc4e516924240b9361fc1be2218",
    "reference_environment_audit_sha256": "3e872f071533c3ab4c2442cf7d144e319d0847e66bdaab7ac2fb8b5f938f0140",
    "reference_checkpoint_sha256": "0c130aa93c88c1f6710940e4cb509c7a649d670e0c6a3cd3f89abbe705de4fef",
    "reference_release_sha256": "9f1d51ff2156d4167ea2e069eab3a4333d37daa6466e0b3e52b68435a47e0c8c",
    "reference_source_tree_sha256": "6708889a39383ef82f084ac2ffb12809f3ef81ad666fd1ba4361e7fc868ef100",
    "reference_training_config_sha256": "673a383cd2491b611d51c8d5cbd06bbda4dda334e735c4b2f6e66b2bf282516b",
    "reference_policy_contract_sha256": "c2c60bef1a2afb22d6a6d704542e9beb42c2d6b9fe4af6449d66f14699ce0d0b",
}
_PROFILES = [
    {"name": "policy_legacy", "action_source": "frozen_actor", "control_guard_version": "legacy_scalar_v1"},
    {"name": "policy_projected", "action_source": "frozen_actor", "control_guard_version": "box_halfspace_projected_v2"},
    {"name": "guided_hand_actor_arm", "action_source": "guided_hand_actor_arm", "control_guard_version": "box_halfspace_projected_v2"},
    {"name": "guided_hand_pose_hold", "action_source": "guided_hand_pose_hold", "control_guard_version": "box_halfspace_projected_v2"},
]
RL_PROBE_DEFAULTS = {
    "enabled": False,
    "backend": "g1_control_physics_diagnostics_v1",
    "num_envs": 8,
    "episodes_per_env": 4,
    "max_control_steps": 364,
    "max_wall_time_s": 3600,
    "optimizer_steps": 0,
    "training_allowed": False,
    "profiles": copy.deepcopy(_PROFILES),
    "guided_hand_target_rad": None,
    "guided_closure_ramp_s": 2.0,
    "guided_preload_controls": 3,
    "guided_probe_height_m": .0025,
    "guided_probe_speed_m_s": .002,
    "guided_force_relief_step_rad": .005,
    "reference_manifest_path": _RUN + "/manifests/native_release.json",
    "reference_environment_audit_path": _RUN + "/native/environment_256/environment_audit.json",
    "reference_checkpoint_path": _RUN + "/native/checkpoints/checkpoint_update_0078_final.pt",
    **_REFERENCE_PINS,
}
_HOST_FIELDS = frozenset((
    "runtime_windows_root", "isaac_python_bat", "isaaclab_path", "git_executable", "asset_path",
    "seed_state_path", "asset_manifest_path", "checkpoint_path", "checkpoint_sha256", "num_envs",
    "smoke_env_counts", "smoke_steps", "max_iterations", "max_wall_time_s", "eval_interval",
    "eval_episodes", "eval_num_envs", "save_interval",
))


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_probe_settings(value):
    """Return an independent complete settings dict; never access files."""
    if not isinstance(value, dict):
        raise ValueError("rl_diagnostics must be an object")
    unknown = set(value) - set(RL_PROBE_DEFAULTS)
    if unknown:
        raise ValueError("Unknown rl_diagnostics keys: " + ", ".join(sorted(unknown)))
    result = copy.deepcopy(RL_PROBE_DEFAULTS)
    result.update(copy.deepcopy(value))
    if type(result["enabled"]) is not bool or result["training_allowed"] is not False:
        raise ValueError("Diagnostics enabled must be boolean and training_allowed must be false")
    for key, expected in (("backend", "g1_control_physics_diagnostics_v1"),
                          ("num_envs", 8), ("episodes_per_env", 4), ("max_control_steps", 364),
                          ("optimizer_steps", 0), ("guided_preload_controls", 3),
                          ("guided_probe_height_m", .0025), ("guided_probe_speed_m_s", .002),
                          ("guided_force_relief_step_rad", .005)):
        if result[key] != expected or (type(expected) is int and type(result[key]) is not int):
            raise ValueError("Diagnostic protocol pins " + key)
    if result["profiles"] != _PROFILES:
        raise ValueError("Diagnostics require the four declared action/guard profiles in order")
    for key, limit in (("max_wall_time_s", 3600), ("guided_closure_ramp_s", 6)):
        number = result[key]
        if type(number) not in (int, float) or not math.isfinite(number) or not 0 < number <= limit:
            raise ValueError(key + " must be finite, positive and within the unchanged horizon/budget")
    target = result["guided_hand_target_rad"]
    if target is not None:
        if (not isinstance(target, list) or len(target) != 7 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in target)):
            raise ValueError("guided_hand_target_rad requires seven finite numbers")
        result["guided_hand_target_rad"] = [float(x) for x in target]
    elif result["enabled"]:
        raise ValueError("Enabled diagnostics require an explicit seven-joint guided hand target")
    for key, expected in _REFERENCE_PINS.items():
        if result[key] != expected or not _is_sha(result[key]):
            raise ValueError("Diagnostics cannot replace the frozen source policy pin: " + key)
    for key in ("reference_manifest_path", "reference_environment_audit_path", "reference_checkpoint_path"):
        path = result[key]
        if not isinstance(path, str) or not path or ".." in path.replace("\\", "/").split("/"):
            raise ValueError(key + " must be an explicit path without parent traversal")
    return result


def _path(config, value):
    path = Path(value)
    if not path.is_absolute():
        try:
            path = Path(config["paths"]["root"]) / path
        except (KeyError, TypeError) as exc:
            raise ValueError("Reference paths require config.paths.root") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError("Diagnostic reference must be a regular existing file: " + str(path))
    return path.resolve()


def _read_pinned(config, settings, path_key, hash_key):
    path = _path(config, settings[path_key])
    if _file_hash(path) != settings[hash_key]:
        raise ValueError("Diagnostic reference bytes do not match " + hash_key)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("Invalid diagnostic reference JSON: " + str(path)) from exc
    if not isinstance(value, dict):
        raise ValueError("Diagnostic reference JSON must be an object")
    return path, value


def _reference_source_identity(manifest):
    declaration = {key: manifest.get(key) for key in ("schema_version", "files", "source_config_sha256")}
    if declaration["schema_version"] != 1 or not isinstance(declaration["files"], list):
        raise ValueError("Reference release declaration is malformed")
    if _canonical(declaration) != manifest.get("sha256"):
        raise ValueError("Reference release canonical declaration does not match its digest")
    seen, source = set(), {}
    for row in declaration["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ValueError("Malformed reference manifest file row")
        name = row["path"]
        if (not isinstance(name, str) or not name or "\\" in name or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts or name in seen or type(row["bytes"]) is not int
                or row["bytes"] < 0 or not _is_sha(row["sha256"])):
            raise ValueError("Invalid or duplicated reference file identity")
        seen.add(name)
        if name.startswith("src/g1_vla/") and name.endswith(".py"):
            source[name[len("src/g1_vla/"):].replace("/", "\\")] = row["sha256"]
    if len(source) != 51:
        raise ValueError("Frozen source policy must have the original 51 package source files")
    return _canonical(source)


def _semantic_training_identity(config):
    """Re-evaluate the old identity, including exact control/expert/metadata bytes."""
    try:
        teacher = {key: value for key, value in config["rl_teacher"].items() if key not in _HOST_FIELDS}
        action = copy.deepcopy(config.get("action"))
        if isinstance(action, dict):
            metadata = action.pop("joint_metadata_path", None)
            if metadata is not None:
                action["joint_metadata_sha256"] = _file_hash(_path(config, metadata))
        return _canonical({"rl_teacher": teacher, "action": action, "control": config.get("control"),
            "expert": config.get("expert"), "runner_version": _REFERENCE_RUNNER_VERSION})
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError("Cannot verify the unchanged source-policy training/config semantics") from exc


def _current_source_identity():
    root = Path(__file__).resolve().parent
    return _canonical({str(path.relative_to(root)): _file_hash(path) for path in sorted(root.rglob("*.py"))})


def verify_probe_reference(config, checkpoint_infos=None, current_audit=None):
    """Verify old provenance and, when supplied, the new diagnostic contract.

    Preflight without Torch metadata/current native audit never grants physical
    admission.  The native caller must check ready_for_physical_probe after
    passing both actual checkpoint infos and its current source-tagged audit.
    """
    settings = validate_probe_settings(config.get("rl_diagnostics", {}))
    manifest_path, manifest = _read_pinned(config, settings, "reference_manifest_path", "reference_manifest_sha256")
    audit_path, reference = _read_pinned(config, settings, "reference_environment_audit_path", "reference_environment_audit_sha256")
    checkpoint = _path(config, settings["reference_checkpoint_path"])
    if _file_hash(checkpoint) != settings["reference_checkpoint_sha256"]:
        raise ValueError("Source actor checkpoint bytes do not match the final78 pin")
    source_sha = _reference_source_identity(manifest)
    if manifest.get("sha256") != settings["reference_release_sha256"] or source_sha != settings["reference_source_tree_sha256"]:
        raise ValueError("Source actor release/source identity does not match the frozen policy")
    policy = reference.get("policy_contract")
    if (not isinstance(policy, dict) or _canonical(policy) != settings["reference_policy_contract_sha256"] or
            reference.get("policy_contract_sha256") != settings["reference_policy_contract_sha256"]):
        raise ValueError("Frozen environment policy contract is not the pinned original")
    identities = {row["sha256"] for row in manifest["files"]}
    if (manifest.get("source_asset_manifest_sha256") != policy.get("asset_manifest_sha256") or
            manifest.get("source_seed_sha256") != policy.get("seed_state_sha256") or
            any(policy.get(key) not in identities for key in
                ("asset_sha256", "asset_manifest_sha256", "seed_state_sha256"))):
        raise ValueError("Reference robot/asset manifest/seed identities are inconsistent")
    if (policy.get("observation_dim") != 81 or policy.get("action_dim") != 14 or
            policy.get("quaternion_order") != "xyzw" or policy.get("physics_dt_s") != .002 or
            policy.get("control_period_s") != .066 or len(policy.get("action_joint_names", [])) != 14 or
            len(set(policy.get("action_joint_names", []))) != 14):
        raise ValueError("Source actor observation/action/cadence ABI is invalid")
    semantic_sha = _semantic_training_identity(config)
    if semantic_sha != settings["reference_training_config_sha256"]:
        raise ValueError("Source-policy config/control/expert/asset semantics changed; a diagnostic guard is the only allowed change")
    metadata_verified = checkpoint_infos is not None
    if metadata_verified:
        if not isinstance(checkpoint_infos, dict):
            raise ValueError("checkpoint_infos must be the actual source checkpoint metadata")
        expected = {"runner_version": _REFERENCE_RUNNER_VERSION,
            "source_tree_sha256": source_sha, "training_config_sha256": semantic_sha,
            "policy_contract_sha256": settings["reference_policy_contract_sha256"],
            "completed_updates": 78, "optimizer_steps": 1248}
        if (any(checkpoint_infos.get(key) != value for key, value in expected.items()) or
                any(type(checkpoint_infos.get(key)) is not int for key in ("completed_updates", "optimizer_steps")) or
                checkpoint_infos.get("policy_contract") != policy):
            raise ValueError("Source checkpoint metadata does not match its original frozen identities")
    current_sha, guard, current_verified = _current_source_identity(), None, current_audit is not None
    if current_verified:
        if not isinstance(current_audit, dict) or current_audit.get("source_tree_sha256") != current_sha:
            raise ValueError("Current diagnostic source hash must truthfully match current package bytes")
        actual = copy.deepcopy(current_audit.get("policy_contract"))
        if not isinstance(actual, dict):
            raise ValueError("Current diagnostic policy contract is missing")
        guard = actual.pop("control_guard_version", None)
        if guard not in {profile["control_guard_version"] for profile in settings["profiles"]}:
            raise ValueError("Current diagnostic control_guard_version is not explicitly declared")
        if actual != policy:
            raise ValueError("Unexpected environment/observation/action/physics/PD/reset change in the diagnostic contract")
        if current_audit.get("policy_contract_sha256") != _canonical(current_audit["policy_contract"]):
            raise ValueError("Current diagnostic policy contract digest does not match its contents")
    return {"reference_verified": True, "checkpoint_metadata_verified": metadata_verified,
        "current_contract_verified": current_verified,
        "ready_for_physical_probe": settings["enabled"] and metadata_verified and current_verified,
        "source_identical_trained_eval": False, "optimizer_steps": 0, "training_allowed": False,
        "diagnostic_backend": settings["backend"], "reference_release_sha256": manifest["sha256"],
        "reference_source_tree_sha256": source_sha, "reference_training_config_sha256": semantic_sha,
        "reference_policy_contract_sha256": settings["reference_policy_contract_sha256"],
        "reference_checkpoint_sha256": settings["reference_checkpoint_sha256"],
        "current_source_tree_sha256": current_sha, "control_guard_version": guard,
        "reference_files": [str(manifest_path), str(audit_path), str(checkpoint)],
        "scope": "frozen actor counterfactual control diagnostics; no new training or original20 acceptance"}


def reference_snapshot_files(config):
    """Return only the three byte-verified source-policy provenance files."""
    result = verify_probe_reference(config)
    return list(dict.fromkeys(Path(value) for value in result["reference_files"]))
