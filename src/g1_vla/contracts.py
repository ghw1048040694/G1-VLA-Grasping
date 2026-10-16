"""One observation/action boundary and checkpoint compatibility gate.

This module never imports MuJoCo, torch, or a policy. It defines the wire format
shared by future data builders, training, replay and evaluation integrations.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .config import OBSERVATION_FIELDS, validate_config, NOSLIP10_CONTACT_BACKEND, NOSLIP10_SOLVER


class ContractError(ValueError):
    """A policy input, action or checkpoint is incompatible with the contract."""


def canonical_sha256(value: Any) -> str:
    """Hash JSON values with preserved list order and no non-finite numbers."""
    try:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractError("metadata must contain only finite JSON values: {}".format(exc)) from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scene_protocol_for_contact_model(settings) -> str:
    """Keep the original identity, or bind the explicit candidate physics.

    Candidate contacts are a simulation hypothesis, not hardware calibration.
    Their complete declaration is hashed so cached labels/checkpoints cannot
    silently cross a different contact model under the same action identity.
    """
    original = "g1_strict_scene_force_transfer_v1"
    if settings is None:
        return original
    if not isinstance(settings, Mapping) or set(settings) != {"backend", "time_constant_s", "damping_ratio"}:
        raise ContractError("contact model requires its complete explicit declaration")
    time_constant, damping = settings["time_constant_s"], settings["damping_ratio"]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
           for v in (time_constant, damping)) or damping != 1.:
        raise ContractError("contact model requires a positive time constant and critical damping")
    if settings["backend"] == "source_default_v1":
        if time_constant != .02:
            raise ContractError("source default contact model has a 0.02 second reference")
        return original
    if settings["backend"] == NOSLIP10_CONTACT_BACKEND:
        if time_constant != .02:
            raise ContractError("Noslip version retains the original 0.02 second contact reference")
        declaration = {"backend": NOSLIP10_CONTACT_BACKEND, "solref": [.02, 1.],
                       "scope": "global_friction_postprocessor", "solver": NOSLIP10_SOLVER}
        return "g1_strict_scene_" + NOSLIP10_CONTACT_BACKEND + "_" + canonical_sha256(declaration)
    if settings["backend"] != "right_hand_green_compliance_v2":
        raise ContractError("unknown contact model protocol")
    declaration = {"backend": settings["backend"], "time_constant_s": float(time_constant),
                   "damping_ratio": float(damping)}
    return "g1_strict_scene_right_hand_green_compliance_v2_" + canonical_sha256(declaration)


def _joint_names(names: Any) -> Tuple[str, ...]:
    if not isinstance(names, (list, tuple)) or len(names) != 31:
        raise ContractError("joint order must contain exactly 31 names from source metadata")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ContractError("joint names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ContractError("joint names must be unique")
    return tuple(names)


def _names_from_feature(feature: Any, name: str) -> Tuple[str, ...]:
    if not isinstance(feature, dict):
        raise ContractError("metadata.features.{} must be an object".format(name))
    if feature.get("shape") != [31]:
        raise ContractError("metadata.features.{}.shape must equal [31]".format(name))
    names = feature.get("names")
    # LeRobot versions use either a list or a single named-axis dictionary.
    if isinstance(names, dict) and len(names) == 1:
        names = next(iter(names.values()))
    return _joint_names(names)


def read_joint_order(metadata: Mapping[str, Any]) -> Tuple[str, ...]:
    """Read the exact order from the real LeRobot action feature metadata."""
    if not isinstance(metadata, dict):
        raise ContractError("source joint metadata must be a JSON object")
    features = metadata.get("features")
    if not isinstance(features, dict):
        raise ContractError("source metadata must contain features.action")
    names = _names_from_feature(features.get("action"), "action")
    state = features.get("observation.state")
    if state is not None and _names_from_feature(state, "observation.state") != names:
        raise ContractError("source action and observation.state joint order differ")
    return names


def _vector(values: Sequence[float], size: int, name: str) -> Tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ContractError("{} must contain {} values".format(name, size))
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ContractError("{} must contain finite numbers".format(name))
        result.append(float(value))
    return tuple(result)


def _limits(value: Any, size: int, name: str, allow_zero: bool = False) -> Tuple[float, ...]:
    values = [value] * size if isinstance(value, (int, float)) and not isinstance(value, bool) else value
    result = _vector(values, size, name)
    if any(item < 0 if allow_zero else item <= 0 for item in result):
        raise ContractError("{} values must be {}".format(name, "non-negative" if allow_zero else "positive"))
    return result


@dataclass(frozen=True)
class ActionContract:
    joint_names: Tuple[str, ...]
    control_hz: float
    physics_dt_s: float
    max_anchor_delta_rad: Tuple[float, ...]
    max_velocity_rad_s: Tuple[float, ...]
    protocol: str = "anchor_relative_target_chunk"
    history_steps: int = 4
    chunk_steps: int = 8
    execute_steps: int = 4
    anchor: str = "qpos_at_plan"
    body_gain_scale: float = 5.0
    hand_gain_scale: float = 6.0
    dynamics_profile: str = "robot_regularized_v1"
    scene_protocol: str = "g1_strict_scene_force_transfer_v1"
    controller_protocol: str = "g1_pd_substep_feedback_v2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_names", _joint_names(self.joint_names))
        object.__setattr__(self, "max_anchor_delta_rad", _limits(
            self.max_anchor_delta_rad, 31, "max_anchor_delta_rad"))
        object.__setattr__(self, "max_velocity_rad_s", _limits(
            self.max_velocity_rad_s, 31, "max_velocity_rad_s", allow_zero=True))
        for name in ("control_hz", "physics_dt_s", "body_gain_scale", "hand_gain_scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ContractError("{} must be a finite positive number".format(name))
        physics_steps = 1.0 / (self.control_hz * self.physics_dt_s)
        if physics_steps < 1 or not math.isclose(physics_steps, round(physics_steps), rel_tol=1e-8, abs_tol=1e-8):
            raise ContractError("control period must contain an integer number of physics steps")
        expected = ("anchor_relative_target_chunk", 4, 8, 4, "qpos_at_plan")
        actual = (self.protocol, self.history_steps, self.chunk_steps, self.execute_steps, self.anchor)
        if actual != expected or any(isinstance(getattr(self, key), bool) for key in ("history_steps", "chunk_steps", "execute_steps")):
            raise ContractError("rebuild contract requires history=4, chunk=8, execute=4 and one qpos_at_plan anchor")
        candidate_prefixes = ("g1_strict_scene_right_hand_green_compliance_v2_",
                              "g1_strict_scene_" + NOSLIP10_CONTACT_BACKEND + "_")
        candidate_digest = next((self.scene_protocol[len(prefix):] for prefix in candidate_prefixes
                                if isinstance(self.scene_protocol, str) and self.scene_protocol.startswith(prefix)), "")
        valid_scene = self.scene_protocol == "g1_strict_scene_force_transfer_v1" or (
            len(candidate_digest) == 64 and all(character in "0123456789abcdef" for character in candidate_digest))
        if isinstance(self.scene_protocol, str) and self.scene_protocol.startswith(candidate_prefixes[1]):
            valid_scene = self.scene_protocol == scene_protocol_for_contact_model({
                "backend": NOSLIP10_CONTACT_BACKEND, "time_constant_s": .02, "damping_ratio": 1.})
        if (self.dynamics_profile != "robot_regularized_v1" or not valid_scene
                or self.controller_protocol != "g1_pd_substep_feedback_v2"):
            raise ContractError("execution dynamics/scene protocol differs from the rebuild controller")

    @classmethod
    def from_config(cls, config: Dict[str, Any], project_root: Optional[Path] = None,
                    metadata: Optional[Mapping[str, Any]] = None) -> "ActionContract":
        """Audit real source metadata and resolved physical settings.

        metadata injection is provided for unit tests/in-memory source audits;
        normal operation reads action.joint_metadata_path without a fallback.
        """
        config = validate_config(config)
        action, control = config["action"], config["control"]
        unresolved = [name for name in ("control_hz", "physics_dt_s", "max_anchor_delta_rad", "max_velocity_rad_s")
                      if control[name] is None]
        if unresolved:
            raise ContractError("control settings unresolved by source audit: {}".format(", ".join(unresolved)))
        if metadata is None:
            path = Path(action["joint_metadata_path"])
            if not path.is_absolute():
                root = Path(project_root) if project_root is not None else Path(config["paths"]["root"])
                if not root.is_absolute():
                    raise ContractError("load_config or explicit absolute project_root is required")
                path = root / path
            try:
                with path.open("r", encoding="utf-8") as stream:
                    metadata = json.load(stream)
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ContractError("cannot read joint metadata {}: {}".format(path, exc)) from exc
        names = read_joint_order(metadata)
        expected_hash = action["joint_order_sha256"]
        if expected_hash is not None and canonical_sha256(list(names)) != expected_hash:
            raise ContractError("source joint order SHA-256 does not match configuration")
        return cls(joint_names=names, control_hz=control["control_hz"],
                   physics_dt_s=control["physics_dt_s"],
                   max_anchor_delta_rad=control["max_anchor_delta_rad"],
                   max_velocity_rad_s=control["max_velocity_rad_s"],
                   protocol=action["protocol"], history_steps=action["history_steps"],
                   chunk_steps=action["chunk_steps"], execute_steps=action["execute_steps"],
                   anchor=action["anchor"], body_gain_scale=control["body_gain_scale"],
                   hand_gain_scale=control["hand_gain_scale"],
                   scene_protocol=scene_protocol_for_contact_model(config["scene"].get("contact_model")))

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)

    def assert_joint_order(self, names: Sequence[str]) -> None:
        if tuple(names) != self.joint_names:
            raise ContractError("joint order differs from the source action metadata")

    def _chunk(self, values: Sequence[Sequence[float]], name: str):
        if not isinstance(values, (list, tuple)) or len(values) != self.chunk_steps:
            raise ContractError("{} must contain exactly {} steps".format(name, self.chunk_steps))
        return tuple(_vector(step, self.action_dim, "{}[{}]".format(name, index))
                     for index, step in enumerate(values))

    def encode_chunk(self, anchor_qpos: Sequence[float], future_targets: Sequence[Sequence[float]],
                     *, joint_names: Sequence[str]):
        """target[t+k] - qpos[t], always relative to the same planning anchor."""
        self.assert_joint_order(joint_names)
        anchor = _vector(anchor_qpos, self.action_dim, "anchor_qpos")
        targets = self._chunk(future_targets, "future_targets")
        return tuple(tuple(target[j] - anchor[j] for j in range(self.action_dim)) for target in targets)

    def decode_chunk(self, anchor_qpos: Sequence[float], delta_chunk: Sequence[Sequence[float]],
                     *, joint_names: Sequence[str]):
        """qpos[t] + delta[k]; do not cumulatively sum the chunk."""
        self.assert_joint_order(joint_names)
        anchor = _vector(anchor_qpos, self.action_dim, "anchor_qpos")
        deltas = self._chunk(delta_chunk, "delta_chunk")
        return tuple(tuple(anchor[j] + delta[j] for j in range(self.action_dim)) for delta in deltas)

    def limit_targets(self, anchor_qpos: Sequence[float], targets: Sequence[Sequence[float]],
                      *, joint_names: Sequence[str], previous_target: Optional[Sequence[float]] = None):
        """Enforce anchor amplitude AND adjacent-command velocity independently.

        Intersect both valid intervals. If a prior command makes the constraints
        incompatible, reject instead of silently breaking either bound.
        """
        self.assert_joint_order(joint_names)
        anchor = _vector(anchor_qpos, self.action_dim, "anchor_qpos")
        chunk = self._chunk(targets, "targets")
        previous = anchor if previous_target is None else _vector(previous_target, self.action_dim, "previous_target")
        result = []
        for target in chunk:
            command = []
            for joint in range(self.action_dim):
                step_limit = self.max_velocity_rad_s[joint] / self.control_hz
                low = max(anchor[joint] - self.max_anchor_delta_rad[joint], previous[joint] - step_limit)
                high = min(anchor[joint] + self.max_anchor_delta_rad[joint], previous[joint] + step_limit)
                if low > high:
                    raise ContractError("amplitude and velocity bounds conflict for joint {}".format(self.joint_names[joint]))
                command.append(min(high, max(low, target[joint])))
            previous = tuple(command)
            result.append(previous)
        return tuple(result)

    def executable_targets(self, anchor_qpos: Sequence[float], delta_chunk: Sequence[Sequence[float]],
                           *, joint_names: Sequence[str], previous_target: Optional[Sequence[float]] = None):
        targets = self.decode_chunk(anchor_qpos, delta_chunk, joint_names=joint_names)
        limited = self.limit_targets(anchor_qpos, targets, joint_names=joint_names, previous_target=previous_target)
        return limited[:self.execute_steps]

    def metadata(self) -> Dict[str, Any]:
        return {
            "schema_version": 1, "protocol": self.protocol, "action_dim": self.action_dim,
            "joint_names": list(self.joint_names),
            "joint_order_sha256": canonical_sha256(list(self.joint_names)),
            "history_steps": self.history_steps, "chunk_steps": self.chunk_steps,
            "execute_steps": self.execute_steps, "anchor": self.anchor,
            "control_hz": float(self.control_hz), "physics_dt_s": float(self.physics_dt_s),
            "max_anchor_delta_rad": list(self.max_anchor_delta_rad),
            "max_velocity_rad_s": list(self.max_velocity_rad_s),
            "body_gain_scale": float(self.body_gain_scale),
            "hand_gain_scale": float(self.hand_gain_scale),
            "dynamics_profile": self.dynamics_profile, "scene_protocol": self.scene_protocol,
            "controller_protocol": self.controller_protocol,
        }

    @property
    def contract_sha256(self) -> str:
        return canonical_sha256(self.metadata())


CAMERA_BINDINGS = {
    "observation.images.head": "head_camera",
    "observation.images.left_wrist": "left_wrist_camera",
    "observation.images.right_wrist": "right_wrist_camera",
}
FINGER_CONTACT_LINKS = (
    "hand_thumb_1_link_contact", "hand_thumb_2_link_contact",
    "hand_middle_0_link_contact", "hand_middle_1_link_contact",
    "hand_index_0_link_contact", "hand_index_1_link_contact",
)
FINGER_CONTACT_GEOM_NAMES = tuple("{}_{}".format(side, link) for side in ("left", "right") for link in FINGER_CONTACT_LINKS)
TACTILE_PROTOCOL = "g1_hand_external_contact_binary_v1"
SENSOR_FRAME_FIELDS = frozenset(OBSERVATION_FIELDS - {"history"})


def observation_spec_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Describe every sensor convention that affects checkpoint compatibility."""
    config = validate_config(config)
    settings = config["observation"]
    result = {
        "schema_version": 1, "fields": list(settings["fields"]),
        "camera_keys": list(settings["camera_keys"]),
        "camera_ids": list(settings["camera_ids"] or []),
        "image_width": settings["image_width"], "image_height": settings["image_height"],
        "image_layout": "HWC_RGB", "image_dtype": "uint8", "image_value_range": [0, 255],
        "history_steps": config["action"]["history_steps"],
        "history_padding": "repeat_first_observed_frame", "history_clock": "actual_control_period",
        "joint_order_sha256": config["action"]["joint_order_sha256"],
        "finger_site_dimension": 108, "finger_site_geometry_order": list(FINGER_CONTACT_GEOM_NAMES),
        "finger_site_sample_order": ["negative_endpoint", "center", "positive_endpoint"],
        "finger_site_frame": "world_fixed_base_from_robot_kinematics", "finger_site_dtype": "float32",
        "tactile_protocol": TACTILE_PROTOCOL,
        "tactile_definition": "hand_contact_with_any_non_robot_body",
        "tactile_dimension": 1, "tactile_dtype": "float32", "tactile_value_range": [0, 1],
        "timestamp_unit": "seconds",
    }
    validate_observation_spec(result)
    return result


def validate_observation_spec(spec: Mapping[str, Any]) -> None:
    fixed = {
        "schema_version": 1, "image_layout": "HWC_RGB", "image_dtype": "uint8",
        "image_value_range": [0, 255], "history_steps": 4,
        "history_padding": "repeat_first_observed_frame", "history_clock": "actual_control_period",
        "finger_site_dimension": 108, "finger_site_geometry_order": list(FINGER_CONTACT_GEOM_NAMES),
        "finger_site_sample_order": ["negative_endpoint", "center", "positive_endpoint"],
        "finger_site_frame": "world_fixed_base_from_robot_kinematics", "finger_site_dtype": "float32",
        "tactile_protocol": TACTILE_PROTOCOL, "tactile_definition": "hand_contact_with_any_non_robot_body",
        "tactile_dimension": 1, "tactile_dtype": "float32", "tactile_value_range": [0, 1],
        "timestamp_unit": "seconds",
    }
    variable = {"fields", "camera_keys", "camera_ids", "image_width", "image_height", "joint_order_sha256"}
    if not isinstance(spec, dict) or set(spec) != set(fixed) | variable:
        raise ContractError("observation_spec must contain the complete sensor schema")
    for key, expected in fixed.items():
        if canonical_sha256(spec[key]) != canonical_sha256(expected):
            raise ContractError("observation_spec {} does not match the rebuild sensor protocol".format(key))
    fields = spec["fields"]
    if not isinstance(fields, list) or len(set(fields)) != len(fields) or set(fields) != OBSERVATION_FIELDS:
        raise ContractError("observation_spec fields must name the complete sensor whitelist")
    keys, ids = spec["camera_keys"], spec["camera_ids"]
    if not isinstance(keys, list) or len(keys) != 3 or set(keys) != set(CAMERA_BINDINGS):
        raise ContractError("observation_spec must name exactly three known camera keys")
    if not isinstance(ids, list) or ids != [CAMERA_BINDINGS[key] for key in keys]:
        raise ContractError("observation_spec camera identities do not match camera key order")
    for key in ("image_width", "image_height"):
        if isinstance(spec[key], bool) or not isinstance(spec[key], int) or spec[key] <= 0:
            raise ContractError("observation_spec {} must be a positive integer".format(key))
    digest = spec["joint_order_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ContractError("observation_spec must freeze the source joint order SHA-256")


def _plain_sensor(value):
    """Allow array/tensor conversion without importing their runtimes."""
    return value.tolist() if hasattr(value, "tolist") else value


def validate_observation(observation: Mapping[str, Any], *, observation_spec: Mapping[str, Any]) -> None:
    """Validate raw sensor frames and optional bounded sensor-only history.

    Images are three HWC uint8 RGB streams before policy normalization. This
    function accepts array/tensor tolist conversion; policy tensor shape checks
    remain mandatory after preprocessing. Arbitrary nested dictionaries are not
    observation containers.
    """
    validate_observation_spec(observation_spec)

    def frame(value, allow_history):
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise ContractError("observation must be a sensor frame dictionary")
        allowed = SENSOR_FRAME_FIELDS | ({"history"} if allow_history else set())
        unexpected = set(value) - allowed
        if unexpected:
            raise ContractError("observation contains unsupported fields: {}".format(", ".join(sorted(unexpected))))
        if not SENSOR_FRAME_FIELDS <= set(value):
            raise ContractError("observation is missing required sensor fields")
        _vector(_plain_sensor(value["joint_position"]), 31, "observation.joint_position")
        _vector(_plain_sensor(value["joint_velocity"]), 31, "observation.joint_velocity")
        _vector(_plain_sensor(value["finger_sites"]), 108, "observation.finger_sites")
        tactile = _vector(_plain_sensor(value["tactile"]), 1, "observation.tactile")
        if tactile[0] not in (0.0, 1.0):
            raise ContractError("observation.tactile must be a binary external hand contact")
        timestamp = value["timestamp_s"]
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
            raise ContractError("observation.timestamp_s must be finite non-negative seconds")
        if not isinstance(value["language"], str) or not value["language"].strip():
            raise ContractError("observation.language must be a non-empty string")
        images = value["images"]
        if not isinstance(images, dict) or set(images) != set(observation_spec["camera_keys"]):
            raise ContractError("observation.images must contain exactly the configured camera keys")
        for key in observation_spec["camera_keys"]:
            image = _plain_sensor(images[key])
            if not isinstance(image, (list, tuple)) or len(image) != observation_spec["image_height"]:
                raise ContractError("{} image height does not match observation_spec".format(key))
            for row in image:
                if not isinstance(row, (list, tuple)) or len(row) != observation_spec["image_width"]:
                    raise ContractError("{} image width does not match observation_spec".format(key))
                for pixel in row:
                    if not isinstance(pixel, (list, tuple)) or len(pixel) != 3:
                        raise ContractError("{} must contain HWC RGB pixels".format(key))
                    if any(isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255 for channel in pixel):
                        raise ContractError("{} must contain uint8 RGB channels in [0,255]".format(key))
        if "history" in value:
            history = value["history"]
            if not isinstance(history, (list, tuple)) or not 1 <= len(history) <= observation_spec["history_steps"]:
                raise ContractError("observation.history must contain one to four sensor frames")
            last_time = None
            for item in history:
                frame(item, False)
                if last_time is not None and item["timestamp_s"] < last_time:
                    raise ContractError("observation.history timestamps must be chronological")
                if item["timestamp_s"] > timestamp:
                    raise ContractError("observation.history cannot contain future sensor frames")
                if item["language"] != value["language"]:
                    raise ContractError("observation.history language must match the current episode")
                last_time = item["timestamp_s"]
    frame(observation, True)


def make_checkpoint_metadata(contract: ActionContract, vocabulary: Sequence[str],
                             normalization_stats: Mapping[str, Any], *,
                             observation_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the metadata envelope; this does not save or load model weights."""
    if not isinstance(vocabulary, (list, tuple)) or not vocabulary or any(not isinstance(item, str) for item in vocabulary):
        raise ContractError("checkpoint vocabulary must be a non-empty ordered string list")
    if len(set(vocabulary)) != len(vocabulary):
        raise ContractError("checkpoint vocabulary must not contain duplicate tokens")
    if not isinstance(normalization_stats, dict) or not normalization_stats:
        raise ContractError("normalization_stats must be a non-empty JSON object")
    validate_observation_spec(observation_spec)
    if observation_spec["joint_order_sha256"] != canonical_sha256(list(contract.joint_names)):
        raise ContractError("observation_spec joint order differs from action contract")
    fields = list(observation_spec["fields"])
    return {
        "schema_version": 1, "contract": contract.metadata(),
        "contract_sha256": contract.contract_sha256,
        "vocabulary": list(vocabulary), "vocabulary_sha256": canonical_sha256(list(vocabulary)),
        "normalization_stats_sha256": canonical_sha256(normalization_stats),
        "observation_fields": fields, "observation_fields_sha256": canonical_sha256(fields),
        "observation_spec": json.loads(json.dumps(observation_spec)),
        "observation_spec_sha256": canonical_sha256(observation_spec),
    }


def validate_checkpoint_metadata(metadata: Mapping[str, Any], contract: ActionContract,
                                 vocabulary: Sequence[str], normalization_stats: Mapping[str, Any],
                                 *, observation_spec: Mapping[str, Any]) -> None:
    """Fail before weight loading when contract, vocabulary or statistics differ."""
    expected = make_checkpoint_metadata(contract, vocabulary, normalization_stats, observation_spec=observation_spec)
    if not isinstance(metadata, dict) or set(metadata) != set(expected):
        raise ContractError("checkpoint metadata envelope has missing or unknown fields")
    for field, value in expected.items():
        if canonical_sha256(metadata[field]) != canonical_sha256(value):
            raise ContractError("checkpoint {} mismatch".format(field))
