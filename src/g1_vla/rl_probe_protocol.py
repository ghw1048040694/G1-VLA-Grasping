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
_PLANS = {"four_way_v1": ("g1_control_physics_diagnostics_v1", _PROFILES),
          "geometry_pose_only_v2": ("g1_control_physics_diagnostics_v2", [_PROFILES[3]]),
          "geometry_pose_fixed_cluster_v3": ("g1_control_physics_diagnostics_v3", [_PROFILES[3]])}
_OLD_PHYSICS = "physx_unitree_official_usd_fixed_root_kinematic_table_tgs_v3"
_FIXED_PHYSICS = "physx_unitree_official_usd_fixed_root_kinematic_table_tgs_v4_fixed_palm_adjacency"
_V3_CANDIDATE_SHA256 = "09ee1ae234ab143076c514273d1105b1ee3a939cb7c349a5e3a40901a765e81f"

_HAND_JOINT_ORDER = ["right_hand_" + x + "_joint" for x in
    ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]

RL_PROBE_DEFAULTS = {
    "enabled": False,
    "backend": "g1_control_physics_diagnostics_v1",
    "profile_plan": "four_way_v1",
    "candidate_path": None,
    "candidate_sha256": "",
    "candidate_id": "",
    "guided_target_lead_max_rad": None,
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
    plan = result["profile_plan"]
    if not isinstance(plan, str) or plan not in _PLANS:
        raise ValueError("Unknown declared diagnostic profile_plan")
    backend, profiles = _PLANS[plan]
    if result["backend"] != backend:
        raise ValueError("Diagnostic backend must match the declared profile_plan")
    if "profiles" not in value:
        result["profiles"] = copy.deepcopy(profiles)
    for key, expected in (("num_envs", 8), ("episodes_per_env", 4), ("max_control_steps", 364),
                          ("optimizer_steps", 0), ("guided_preload_controls", 3),
                          ("guided_probe_height_m", .0025), ("guided_probe_speed_m_s", .002),
                          ("guided_force_relief_step_rad", .005)):
        if result[key] != expected or (type(expected) is int and type(result[key]) is not int):
            raise ValueError("Diagnostic protocol pins " + key)
    if result["profiles"] != profiles:
        raise ValueError("Diagnostics require exactly the declared profile_plan action/guard profiles in order")
    lead = result["guided_target_lead_max_rad"]
    if ((plan == "four_way_v1" and lead is not None) or
            (plan in ("geometry_pose_only_v2", "geometry_pose_fixed_cluster_v3") and (type(lead) not in (int, float) or lead != .006))):
        raise ValueError("Diagnostic plan pins guided_target_lead_max_rad: None forv1, .006 forv2")
    if plan == "four_way_v1":
        if (result["candidate_path"] is not None or
                any(result[key] != "" for key in ("candidate_sha256", "candidate_id"))):
            raise ValueError("Legacy four_way_v1 cannot silently select a new candidate")
    else:
        path = result["candidate_path"]
        if (not isinstance(path, str) or not path or ".." in path.replace("\\", "/").split("/") or
                not _is_sha(result["candidate_sha256"]) or
                not isinstance(result["candidate_id"], str) or not result["candidate_id"].strip()):
            raise ValueError("Geometry diagnostic plan requires explicit candidate path/SHA256/ID")
        if plan == "geometry_pose_fixed_cluster_v3" and result["candidate_sha256"] != _V3_CANDIDATE_SHA256:
            raise ValueError("Fixed-cluster model comparison must preserve the pinned geometry candidate")
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




def _declared_fixed_filter_contract():
    """Independent fixed expected model declaration, never from current audit.

    Native factory must separately verify the actual55-joint pinned USD graph
    before applying this15-pair allowlist. This function does not apply physics.
    """
    contract = {
    "contract_version": "g1_right_wrist_fixed_component_adjacent_pairs_v1",
    "asset_sha256": "d23b9bbc8b59cf20fa1983f55874f855a76e038b9e22eae39542e62be5b4ba3d",
    "fixed_component": [
        "right_hand_camera_base_link",
        "right_hand_palm_link",
        "right_wrist_yaw_link"
    ],
    "revolute_adjacent_bodies": [
        "right_hand_index_0_link",
        "right_hand_middle_0_link",
        "right_hand_thumb_0_link",
        "right_wrist_pitch_link"
    ],
    "excluded_body_pairs": [
        [
            "right_hand_camera_base_link",
            "right_hand_index_0_link"
        ],
        [
            "right_hand_camera_base_link",
            "right_hand_middle_0_link"
        ],
        [
            "right_hand_camera_base_link",
            "right_hand_palm_link"
        ],
        [
            "right_hand_camera_base_link",
            "right_hand_thumb_0_link"
        ],
        [
            "right_hand_camera_base_link",
            "right_wrist_pitch_link"
        ],
        [
            "right_hand_camera_base_link",
            "right_wrist_yaw_link"
        ],
        [
            "right_hand_index_0_link",
            "right_hand_palm_link"
        ],
        [
            "right_hand_index_0_link",
            "right_wrist_yaw_link"
        ],
        [
            "right_hand_middle_0_link",
            "right_hand_palm_link"
        ],
        [
            "right_hand_middle_0_link",
            "right_wrist_yaw_link"
        ],
        [
            "right_hand_palm_link",
            "right_hand_thumb_0_link"
        ],
        [
            "right_hand_palm_link",
            "right_wrist_pitch_link"
        ],
        [
            "right_hand_palm_link",
            "right_wrist_yaw_link"
        ],
        [
            "right_hand_thumb_0_link",
            "right_wrist_yaw_link"
        ],
        [
            "right_wrist_pitch_link",
            "right_wrist_yaw_link"
        ]
    ]
}
    digest = _canonical(contract)
    if digest != "04dd990d50f4632012408b13747313a5504614e47a13d05d9540c31ef66fb770":
        raise ValueError("Pinned fixed-cluster collision declaration changed")
    return {"contract": contract,
        "source_graph_sha256": "1bc9a5d33856280a489137634f3ddcbb84196ef231b6dfa902acbd271eac600c",
        "sha256": digest}


def _finite_vector(value, length, description):
    if (not isinstance(value, list) or len(value) != length or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError(description + " requires finite numbers of the declared dimension")
    return [float(x) for x in value]


def _verify_geometry_candidate(config, settings, policy, reference):
    """Verify pinned offline evidence; never certify physical contact/support."""
    if settings["profile_plan"] == "four_way_v1":
        return None, None
    path, candidate = _read_pinned(config, settings, "candidate_path", "candidate_sha256")
    keys = {"schema_version", "candidate_id", "asset_sha256", "oldseed_sha256",
            "hand_joint_order", "target_rad", "geometry_evidence",
            "known_geometric_overlaps", "admission_limitations"}
    if (set(candidate) != keys or type(candidate["schema_version"]) is not int or
            candidate["schema_version"] != 1 or candidate["candidate_id"] != settings["candidate_id"] or
            candidate["asset_sha256"] != policy["asset_sha256"] or
            candidate["oldseed_sha256"] != policy["seed_state_sha256"] or
            candidate["hand_joint_order"] != _HAND_JOINT_ORDER):
        raise ValueError("Geometry candidate schema/ID/asset/oldseed/joint order mismatch")
    if (not isinstance(candidate["known_geometric_overlaps"], list) or
            not isinstance(candidate["admission_limitations"], list) or not candidate["admission_limitations"]):
        raise ValueError("Candidate known geometric overlaps and explicit admission limitations must be lists")
    target = _finite_vector(candidate["target_rad"], 7, "Candidate target_rad")
    if target != settings["guided_hand_target_rad"]:
        raise ValueError("Geometry candidate target must exactly equal guided_hand_target_rad")
    evidence = candidate["geometry_evidence"]
    expected = {"protocol", "qualification", "units", "hand_joint_limits_rad",
                "finger_surface_evidence", "source_detail"}
    if (not isinstance(evidence, dict) or set(evidence) != expected or
            evidence["protocol"] != "offline_native_usd_grasp_geometry_v1" or
            evidence["qualification"] != "candidate_only_not_physics_qualified" or
            evidence["units"] != "m_rad" or not isinstance(evidence["source_detail"], dict) or
            not evidence["source_detail"]):
        raise ValueError("Geometry candidate requires declared non-physical offline geometry evidence")
    # The file is trusted as a pinned audit artifact, not a substitute for the
    # native body's hard-limit/loaded-contact checks at physical execution.
    limits = evidence["hand_joint_limits_rad"]
    if not isinstance(limits, list) or len(limits) != 7:
        raise ValueError("Candidate geometry requires seven native joint bounds")
    schema = policy["native_joint_axis_frame_unit_schema"]
    for i, row in enumerate(limits):
        low, high = _finite_vector(row, 2, "Candidate native joint bounds")
        if not low <= target[i] <= high:
            raise ValueError("Candidate target is outside declared native joint bounds")
        native = schema.get(_HAND_JOINT_ORDER[i], {})
        expected_low, expected_high = native.get("lower_limit_degrees"), native.get("upper_limit_degrees")
        if (type(expected_low) not in (int, float) or type(expected_high) not in (int, float) or
                not math.isclose(low, math.radians(expected_low), rel_tol=0, abs_tol=1e-7) or
                not math.isclose(high, math.radians(expected_high), rel_tol=0, abs_tol=1e-7)):
            raise ValueError("Candidate bounds do not match the pinned native USD joint schema")
    rows = evidence["finger_surface_evidence"]
    if not isinstance(rows, dict) or set(rows) != {"thumb", "index", "middle"}:
        raise ValueError("Candidate needs thumb/index/middle surface evidence")
    for finger, distal in zip(("thumb", "index", "middle"), ("thumb_2", "index_1", "middle_1")):
        row = rows[finger]
        row_keys = {"contact_body", "minimum_distance_to_cube_m", "closest_point_world_m3",
                    "cube_closest_point_world_m3", "closing_direction_world3", "method"}
        if (not isinstance(row, dict) or set(row) != row_keys or
                row["contact_body"] != "right_hand_" + distal + "_link" or
                not isinstance(row["method"], str) or not row["method"].strip()):
            raise ValueError("Candidate finger/body/method geometry evidence mismatch")
        distance = row["minimum_distance_to_cube_m"]
        if type(distance) not in (int, float) or not math.isfinite(distance) or distance < 0:
            raise ValueError("Candidate surface distance must be finite nonnegative metres")
        point = _finite_vector(row["closest_point_world_m3"], 3, "Candidate finger closest point")
        cube_point = _finite_vector(row["cube_closest_point_world_m3"], 3, "Candidate cube closest point")
        _finite_vector(row["closing_direction_world3"], 3, "Candidate closing direction")
        if not math.isclose(math.dist(point, cube_point), distance, rel_tol=0, abs_tol=1e-6):
            raise ValueError("Candidate surface points and minimum distance are inconsistent")
    detail = evidence["source_detail"]
    if (detail.get("no_object_state_change") is not True or
            _finite_vector(detail.get("palm_offset_xyz_world_m"), 3, "Candidate palm XYZ offset") != [0., 0., 0.] or
            _finite_vector(detail.get("palm_offset_rpy_world_rad"), 3, "Candidate palm RPY offset") != [0., 0., 0.]):
        raise ValueError("Candidate must retain object state and zero static palm-pose offset")
    stages = detail.get("stage_keyframes")
    if not isinstance(stages, list) or len(stages) != 6:
        raise ValueError("Candidate must declare exactly six staged keyframes")
    names, duration = set(), 0.
    targets = []
    for index, frame in enumerate(stages):
        keys = {"name", "target_rad"} if index == 0 else {"name", "target_rad", "duration_s"}
        if (not isinstance(frame, dict) or set(frame) != keys or not isinstance(frame["name"], str) or
                not frame["name"].strip() or frame["name"] in names):
            raise ValueError("Candidate staged keyframe name/schema is invalid")
        names.add(frame["name"])
        q = _finite_vector(frame["target_rad"], 7, "Candidate staged target")
        if any(not low <= angle <= high for angle, (low, high) in zip(q, limits)):
            raise ValueError("Candidate staged target violates native joint bounds")
        targets.append(q)
        if index:
            seconds = frame["duration_s"]
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Candidate staged durations must be finite positive seconds")
            duration += seconds
    source = reference.get("initial_state", {}).get("joint_positions_rad", {})
    source_hand = [source.get(name) for name in _HAND_JOINT_ORDER]
    source_hand = _finite_vector(source_hand, 7, "Pinned oldseed hand state")
    if (stages[0]["name"] != "source" or
            any(abs(a-b) > 1e-4 for a, b in zip(targets[0], source_hand)) or
            targets[-1] != target or duration > 6.):
        raise ValueError("Candidate staged source/final/horizon must match oldseed/target/original6s")
    try:
        _canonical(candidate)
    except (ValueError, TypeError) as exc:
        raise ValueError("Candidate offline geometry evidence must be finite JSON") from exc
    return path, {"candidate_id": candidate["candidate_id"], "candidate_sha256": settings["candidate_sha256"],
        "candidate_path": str(path), "schema_and_provenance_verified": True,
        "geometry_recomputed_by_protocol": False, "physical_qualification": False,
        "evidence_protocol": evidence["protocol"],
        "known_geometric_overlaps": copy.deepcopy(candidate["known_geometric_overlaps"]),
        "admission_limitations": copy.deepcopy(candidate["admission_limitations"]),
        "self_collision_free_claimed": False, "stage_keyframes": copy.deepcopy(stages),
        "declared_stage_duration_s": duration, "dynamic_feasibility_verified": False}


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
    candidate_path, candidate = _verify_geometry_candidate(config, settings, policy, reference)
    current_semantic_sha = _semantic_training_identity(config)
    fixed_model = settings["profile_plan"] == "geometry_pose_fixed_cluster_v3"
    expected_physics = _FIXED_PHYSICS if fixed_model else _OLD_PHYSICS
    if policy.get("physics_protocol") != _OLD_PHYSICS or config["rl_teacher"].get("physics_protocol") != expected_physics:
        raise ValueError("Diagnostic plan must use its exact old-v3 or declared new-v4 physics model")
    # Preserve the actual new-v4 identity above. Only this exact approved
    # physics field is restored in a copy to verify all remaining old config.
    comparison_config = copy.deepcopy(config)
    if fixed_model:
        comparison_config["rl_teacher"]["physics_protocol"] = policy["physics_protocol"]
    semantic_sha = _semantic_training_identity(comparison_config)
    if semantic_sha != settings["reference_training_config_sha256"]:
        raise ValueError("Source-policy config/control/expert/asset semantics changed beyond the declared model difference")
    model_declaration = _declared_fixed_filter_contract() if fixed_model else None
    if model_declaration and model_declaration["contract"]["asset_sha256"] != policy["asset_sha256"]:
        raise ValueError("Fixed-cluster model declaration does not match the pinned original asset")
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
        if fixed_model:
            if (actual.get("physics_protocol") != _FIXED_PHYSICS or
                    current_audit.get("physics_version") != _FIXED_PHYSICS or
                    actual.get("fixed_cluster_collision_filter") != model_declaration):
                raise ValueError("Current fixed-cluster physics/filter declaration is not exactly the pinned model")
            # This is not an arbitrary pop/allow_source_mismatch: exact new
            # contract verified first, and every remaining field stays strict.
            del actual["fixed_cluster_collision_filter"]
            actual["physics_protocol"] = policy["physics_protocol"]
        if actual != policy:
            raise ValueError("Unexpected environment/observation/action/physics/PD/reset change in the diagnostic contract")
        if current_audit.get("policy_contract_sha256") != _canonical(current_audit["policy_contract"]):
            raise ValueError("Current diagnostic policy contract digest does not match its contents")
    return {"reference_verified": True, "checkpoint_metadata_verified": metadata_verified,
        "current_contract_verified": current_verified,
        "ready_for_physical_probe": settings["enabled"] and metadata_verified and current_verified,
        "source_identical_trained_eval": False, "optimizer_steps": 0, "training_allowed": False,
        "diagnostic_backend": settings["backend"], "profile_plan": settings["profile_plan"],
        "declared_profile_names": [row["name"] for row in settings["profiles"]],
        "four_way_comparison_complete": False,
        "four_way_comparison_declared": settings["profile_plan"] == "four_way_v1",
        "candidate_reference": candidate, "reference_release_sha256": manifest["sha256"],
        "reference_source_tree_sha256": source_sha, "reference_training_config_sha256": semantic_sha,
        "reference_policy_contract_sha256": settings["reference_policy_contract_sha256"],
        "reference_checkpoint_sha256": settings["reference_checkpoint_sha256"],
        "current_source_tree_sha256": current_sha, "control_guard_version": guard,
        "current_config_semantic_sha256": current_semantic_sha,
        "reference_rest_config_comparison_sha256": semantic_sha,
        "physics_model_comparison": {
            "old_checkpoint_physics_protocol": policy["physics_protocol"],
            "current_declared_physics_protocol": expected_physics,
            "old_policy_contract_sha256": settings["reference_policy_contract_sha256"],
            "model_changed": fixed_model, "fixed_cluster_collision_filter": model_declaration,
            "old_physics_restored_only_for_rest_config_comparison": fixed_model,
            "scope": "explicit new-model counterfactual; no old-trained-environment equivalence" if fixed_model else
                     "original physics retained; control-only counterfactual"},
        "reference_files": [str(manifest_path), str(audit_path), str(checkpoint)] +
            ([str(candidate_path)] if candidate_path is not None else []),
        "scope": "frozen actor counterfactual control diagnostics; no new training or original20 acceptance"}


def reference_snapshot_files(config):
    """Return old source-policy provenance plus an explicit v2 candidate file."""
    result = verify_probe_reference(config)
    return list(dict.fromkeys(Path(value) for value in result["reference_files"]))
