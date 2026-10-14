"""Sensor-only G1 observations; simulator access ends at this adapter.

The FK sites describe the robot's own collision capsules. Tactile v1 reports
hand contact with any non-robot body, rather than privileged target-object
membership used by historical 672 evaluation. It is sensor-equivalent simulated
contact, not a validated hardware tactile sensor calibration.
"""

import copy
import math
from collections import deque
from typing import Mapping, Sequence

from .contracts import (
    FINGER_CONTACT_GEOM_NAMES, SENSOR_FRAME_FIELDS, TACTILE_PROTOCOL,
    ContractError, canonical_sha256, observation_spec_from_config,
    validate_observation, validate_observation_spec,
)
from .control import _api, action_joint_bindings, controlled_joints


class ObservationError(ContractError):
    """Observed sensor values or their geometry do not match the schema."""


class ObservationSpec:
    def __init__(self, metadata: Mapping):
        validate_observation_spec(metadata)
        self._metadata = copy.deepcopy(dict(metadata))

    @classmethod
    def from_config(cls, config):
        return cls(observation_spec_from_config(config))

    def metadata(self):
        return copy.deepcopy(self._metadata)

    @property
    def sha256(self):
        return canonical_sha256(self._metadata)


def _finite_vector(values, count, name):
    values = values.tolist() if hasattr(values, "tolist") else values
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ObservationError("{} must contain {} values".format(name, count))
    result = []
    for value in values:
        if isinstance(value, bool):
            raise ObservationError("{} must contain finite numeric values".format(name))
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ObservationError("{} must contain finite numeric values".format(name)) from exc
        if not math.isfinite(number):
            raise ObservationError("{} must contain finite numeric values".format(name))
        result.append(number)
    return tuple(result)


def capsule_samples(center: Sequence[float], rotation: Sequence[float], half_length: float):
    """Return negative endpoint, center, positive endpoint along local z."""
    center = _finite_vector(center, 3, "capsule center")
    rotation = _finite_vector(rotation, 9, "capsule rotation")
    if isinstance(half_length, bool) or not math.isfinite(float(half_length)) or half_length < 0:
        raise ObservationError("capsule half_length must be finite and non-negative")
    axis = (rotation[2], rotation[5], rotation[8])
    if not math.isclose(sum(value * value for value in axis), 1.0, abs_tol=1e-5):
        raise ObservationError("capsule local z axis must be normalized")
    return tuple(center[joint] + direction * float(half_length) * axis[joint]
                 for direction in (-1.0, 0.0, 1.0) for joint in range(3))


def finger_contact_geom_ids(model, *, mujoco_api=None):
    api = _api(mujoco_api)
    ids = tuple(api.mj_name2id(model, api.mjtObj.mjOBJ_GEOM, name) for name in FINGER_CONTACT_GEOM_NAMES)
    if min(ids) < 0 or len(set(ids)) != 12:
        raise ObservationError("scene is missing the twelve canonical finger contact capsules")
    for geom_id in ids:
        if hasattr(model, "geom_type") and int(model.geom_type[geom_id]) != int(api.mjtGeom.mjGEOM_CAPSULE):
            raise ObservationError("FK sampling requires physical capsule geometry")
    return ids


def finger_contact_sites(model, data, geom_ids):
    """Preserve 108D left/right, six-capsule, endpoint/center FK ordering."""
    if len(geom_ids) != 12 or len(set(geom_ids)) != 12:
        raise ObservationError("FK contact sites require twelve unique capsule ids in canonical order")
    result = []
    for geom_id in geom_ids:
        rotation = data.geom_xmat[geom_id]
        if hasattr(rotation, "reshape"):
            rotation = rotation.reshape(-1)
        result.extend(capsule_samples(data.geom_xpos[geom_id], rotation, float(model.geom_size[geom_id][1])))
    return result


def robot_body_ids(model, *, mujoco_api=None):
    """Find the pelvis subtree without consulting any task-object identity."""
    api = _api(mujoco_api)
    pelvis = api.mj_name2id(model, api.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis <= 0:
        raise ObservationError("robot kinematics require the pelvis body")
    result = set()
    for body_id in range(1, model.nbody):
        ancestor, seen = body_id, set()
        while ancestor > 0:
            if ancestor in seen:
                raise ObservationError("body parent hierarchy contains a cycle")
            seen.add(ancestor)
            if ancestor == pelvis:
                result.add(body_id)
                break
            ancestor = int(model.body_parentid[ancestor])
    return frozenset(result)


def hand_contact_geom_ids(model, *, mujoco_api=None):
    """All hand link/palm collision geoms, including historical convex hulls."""
    api = _api(mujoco_api)
    result = set()
    for geom_id in range(model.ngeom):
        body_name = api.mj_id2name(model, api.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        geom_name = api.mj_id2name(model, api.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if body_name.startswith(("left_hand_", "right_hand_")) or geom_name in ("left_palm_contact_pad", "right_palm_contact_pad"):
            result.add(geom_id)
    if not result:
        raise ObservationError("scene has no physical hand contact geometry")
    return frozenset(result)


def external_hand_contact(contacts, hand_geoms, robot_bodies, geom_body_ids):
    """Binary actual contact with ANY non-robot body; no target IDs accepted.

    Contacts are (geom1, geom2, distance) tuples; speculative positive-distance
    pairs are excluded. Self contacts and opposite-hand contacts are excluded.
    Collision masks still determine which contacts the simulator can generate.
    """
    for first, second, distance in contacts:
        if not math.isfinite(float(distance)):
            raise ObservationError("contact distance must be finite")
        if float(distance) > 0:
            continue
        if first in hand_geoms and int(geom_body_ids[second]) not in robot_bodies:
            return 1.0
        if second in hand_geoms and int(geom_body_ids[first]) not in robot_bodies:
            return 1.0
    return 0.0


class ObservationBuilder:
    """Translate internal simulator state into an explicit sensor frame."""

    def __init__(self, model, renderer, joint_names, spec: ObservationSpec, *, mujoco_api=None):
        self._model, self._renderer, self._api = model, renderer, _api(mujoco_api)
        self.spec = spec
        self._joint_names = tuple(joint_names)
        metadata = spec.metadata()
        if canonical_sha256(list(self._joint_names)) != metadata["joint_order_sha256"]:
            raise ObservationError("observation joint order differs from the sensor schema")
        self._bindings = action_joint_bindings(controlled_joints(model, mujoco_api=self._api), self._joint_names)
        self._finger_ids = finger_contact_geom_ids(model, mujoco_api=self._api)
        self._robot_bodies = robot_body_ids(model, mujoco_api=self._api)
        self._hand_geoms = hand_contact_geom_ids(model, mujoco_api=self._api)

    def observe(self, data, instruction: str):
        spec = self.spec.metadata()
        images = {}
        for key, camera in zip(spec["camera_keys"], spec["camera_ids"]):
            self._renderer.update_scene(data, camera=camera)
            images[key] = copy.deepcopy(self._renderer.render())
        contacts = [(int(data.contact[index].geom1), int(data.contact[index].geom2), float(data.contact[index].dist))
                    for index in range(data.ncon)]
        observation = {
            "images": images,
            "joint_position": [float(data.qpos[item["qpos_id"]]) for item in self._bindings],
            "joint_velocity": [float(data.qvel[item["qvel_id"]]) for item in self._bindings],
            "finger_sites": finger_contact_sites(self._model, data, self._finger_ids),
            "tactile": [external_hand_contact(contacts, self._hand_geoms, self._robot_bodies, self._model.geom_bodyid)],
            "language": instruction, "timestamp_s": float(data.time),
        }
        validate_observation(observation, observation_spec=spec)
        return observation


class SensorHistory:
    """Four causal sensor frames with explicit first-frame padding."""

    def __init__(self, spec: ObservationSpec):
        self.spec = spec
        self._frames = deque(maxlen=spec.metadata()["history_steps"])

    def reset(self):
        self._frames.clear()

    def append(self, observation):
        if "history" in observation:
            raise ObservationError("SensorHistory accepts individual sensor frames only")
        validate_observation(observation, observation_spec=self.spec.metadata())
        if self._frames:
            if observation["timestamp_s"] <= self._frames[-1]["timestamp_s"]:
                raise ObservationError("new history frames must have strictly increasing observation time")
            if observation["language"] != self._frames[-1]["language"]:
                raise ObservationError("reset history before changing the episode language")
        self._frames.append(copy.deepcopy(observation))

    def frames(self):
        if not self._frames:
            raise ObservationError("policy history needs at least one observed frame")
        rows = list(self._frames)
        count = self.spec.metadata()["history_steps"]
        return copy.deepcopy([rows[0]] * (count - len(rows)) + rows)

    def tensor_observation(self, vocabulary, normalization_stats, *, device="cpu"):
        """Preprocess the same schema for training/replay/inference tensor inputs.

        Torch is imported only on an explicit preprocessing call; this method
        does not load a checkpoint or call a model.
        """
        from .policy import TensorObservation, encode_tasks, require_torch
        require_torch()
        import torch
        frames, spec = self.frames(), self.spec.metadata()
        images = torch.stack([torch.stack([
            torch.as_tensor(frame["images"][key], device=device).permute(2, 0, 1).float() / 255.0
            for key in spec["camera_keys"]]) for frame in frames]).unsqueeze(0)

        def observed(field):
            return torch.tensor([frame[field] for frame in frames], dtype=torch.float32, device=device).unsqueeze(0)

        def normalize(tensor, prefix):
            mean = torch.as_tensor(normalization_stats[prefix + "_mean"], device=device, dtype=torch.float32)
            std = torch.as_tensor(normalization_stats[prefix + "_std"], device=device, dtype=torch.float32)
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
                raise ObservationError("normalization statistics must be finite with positive std")
            return (tensor - mean) / std
        ids, mask = encode_tasks([frames[-1]["language"]], vocabulary)
        return TensorObservation(images, normalize(observed("joint_position"), "state"),
                                 normalize(observed("joint_velocity"), "velocity"),
                                 normalize(observed("finger_sites"), "contact_sites"),
                                 observed("tactile"), ids.to(device), mask.to(device))
