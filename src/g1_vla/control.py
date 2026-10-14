"""Shared G1 joint mapping and PD execution, with optional MuJoCo binding.

Pure validation/math functions import without MuJoCo. PDController is the only
physics executor: it requires a resolved ActionContract before touching physics.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .contracts import ActionContract, ContractError


class ControlError(ContractError):
    """Joint mapping, fixed stance or execution settings are invalid."""


LOWER_BODY_JOINT_NAMES = tuple(
    "{}_{}_joint".format(side, joint)
    for side in ("left", "right")
    for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
)
FIXED_BASE_EQUALITY = "warehouse_fixed_base"


def _api(mujoco_api=None):
    if mujoco_api is not None:
        return mujoco_api
    try:
        import mujoco
    except ImportError as exc:
        raise ControlError("MuJoCo is required for model binding/execution, not for plan or inventory") from exc
    return mujoco


def _finite(value, name):
    if isinstance(value, bool):
        raise ControlError("{} must be a finite number".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ControlError("{} must be a finite number".format(name)) from exc
    if not math.isfinite(result):
        raise ControlError("{} must be a finite number".format(name))
    return result


def unitree_gains(name: str) -> Tuple[float, float]:
    """Preserve the historical Unitree gain profile before configured scales."""
    if "hand_" in name:
        return 1.5, 0.2
    if "wrist_" in name:
        return 40.0, 1.5
    weak_body = "ankle_pitch" in name or "shoulder_" in name or "elbow_" in name
    return (80.0, 3.0) if weak_body else (300.0, 3.0)


def pd_torque(name: str, target: float, qpos: float, qvel: float, bias: float,
              *, body_gain_scale: float, hand_gain_scale: float) -> float:
    """kp*scale*(target-q) - kd*sqrt(scale)*qdot + per-substep bias."""
    values = [_finite(value, label) for value, label in (
        (target, "target"), (qpos, "qpos"), (qvel, "qvel"), (bias, "bias"))]
    scale = _finite(hand_gain_scale if "hand_" in name else body_gain_scale, "gain_scale")
    if scale <= 0:
        raise ControlError("gain_scale must be positive")
    kp, kd = unitree_gains(name)
    return kp * scale * (values[0] - values[1]) - kd * math.sqrt(scale) * values[2] + values[3]


def controlled_joints(model, *, mujoco_api=None) -> Dict[str, Dict[str, int]]:
    """Bind by actuator/joint identity, never by a positional 43-to-31 slice."""
    api = _api(mujoco_api)
    result = {}
    for actuator_id in range(model.nu):
        name = api.mj_id2name(model, api.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if not name:
            raise ControlError("actuator {} has no name".format(actuator_id))
        joint_id = api.mj_name2id(model, api.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ControlError("actuator {} has no same-named joint".format(name))
        if name in result:
            raise ControlError("duplicate actuator name {}".format(name))
        if hasattr(model, "actuator_trnid") and int(model.actuator_trnid[actuator_id][0]) != joint_id:
            raise ControlError("actuator {} transmission does not drive its same-named joint".format(name))
        result[name] = {
            "actuator_id": actuator_id, "joint_id": joint_id,
            "qpos_id": int(model.jnt_qposadr[joint_id]),
            "qvel_id": int(model.jnt_dofadr[joint_id]),
        }
    return result


def action_joint_bindings(controlled: Mapping[str, Mapping[str, int]], joint_names: Sequence[str]):
    """Return 31 bindings in metadata order and assert the intended 43 actuators."""
    if len(joint_names) != 31 or len(set(joint_names)) != 31:
        raise ControlError("source action metadata must define 31 unique joints")
    expected = set(joint_names) | set(LOWER_BODY_JOINT_NAMES)
    if len(expected) != 43 or set(controlled) != expected:
        missing, unexpected = sorted(expected - set(controlled)), sorted(set(controlled) - expected)
        raise ControlError("43 joint mapping mismatch; missing={}, unexpected={}".format(missing, unexpected))
    return tuple(controlled[name] for name in joint_names)


def map_action_targets(action: Sequence[float], joint_names: Sequence[str],
                       controlled: Mapping[str, Mapping[str, int]], hold_targets: Mapping[str, float]):
    """Merge upper-body metadata31 targets with all twelve standing leg holds."""
    action_joint_bindings(controlled, joint_names)
    if len(action) != 31:
        raise ControlError("action must contain exactly 31 target positions")
    if set(hold_targets) != set(controlled):
        raise ControlError("hold_targets must name all 43 commanded joints")
    result = {name: _finite(value, "hold_target[{}]".format(name)) for name, value in hold_targets.items()}
    result.update({name: _finite(value, "action[{}]".format(name)) for name, value in zip(joint_names, action)})
    return result


def apply_regularized_dynamics(model, *, joint_names: Optional[Sequence[str]] = None, mujoco_api=None) -> None:
    """Migrate robot damping/armature; task free bodies are never regularized.

    Legacy helper also touched one DOF of each object freejoint. The shared
    implementation scopes this profile explicitly to commanded robot joints.
    """
    api = _api(mujoco_api)
    names = set(joint_names) if joint_names is not None else set(controlled_joints(model, mujoco_api=api))
    for joint_id in range(model.njnt):
        name = api.mj_id2name(model, api.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if name not in names or name == "floating_base_joint":
            continue
        dof_id = int(model.jnt_dofadr[joint_id])
        if "hand_" in name:
            damping, armature = 0.02, 0.0005
        elif "wrist_" in name:
            damping, armature = 0.1, 0.005
        else:
            damping, armature = 0.2, 0.01
        model.dof_damping[dof_id] = damping
        model.dof_armature[dof_id] = armature


@dataclass(frozen=True)
class ControlCadence:
    nominal_control_hz: float
    control_hz: float
    physics_dt_s: float
    physics_substeps: int

    @classmethod
    def from_config(cls, config):
        settings = config["control"]
        if any(settings.get(key) is None for key in (
                "nominal_control_hz", "control_hz", "physics_dt_s", "physics_substeps")):
            raise ControlError("control cadence must be explicitly resolved before execution")
        result = cls(settings["nominal_control_hz"], settings["control_hz"],
                     settings["physics_dt_s"], settings["physics_substeps"])
        if isinstance(result.physics_substeps, bool) or not isinstance(result.physics_substeps, int) or result.physics_substeps <= 0:
            raise ControlError("physics_substeps must be a positive integer")
        for name in ("nominal_control_hz", "control_hz", "physics_dt_s"):
            if _finite(getattr(result, name), name) <= 0:
                raise ControlError("{} must be positive".format(name))
        actual = 1.0 / (result.physics_dt_s * result.physics_substeps)
        if not math.isclose(result.control_hz, actual, rel_tol=1e-9, abs_tol=1e-9):
            raise ControlError("control_hz must describe actual physics_dt_s * physics_substeps")
        return result

    @property
    def period_s(self):
        return self.physics_dt_s * self.physics_substeps


def assert_fixed_stance(model, data, *, mujoco_api=None) -> None:
    api = _api(mujoco_api)
    index = api.mj_name2id(model, api.mjtObj.mjOBJ_EQUALITY, FIXED_BASE_EQUALITY)
    pelvis = api.mj_name2id(model, api.mjtObj.mjOBJ_BODY, "pelvis")
    if index < 0 or pelvis < 0:
        raise ControlError("fixed-stance execution requires a pelvis-to-world warehouse_fixed_base weld")
    if int(model.eq_type[index]) != int(api.mjtEq.mjEQ_WELD):
        raise ControlError("warehouse_fixed_base must be a weld")
    if int(model.eq_obj1id[index]) != pelvis or int(model.eq_obj2id[index]) != 0:
        raise ControlError("warehouse_fixed_base must weld the pelvis to the world")
    active = getattr(data, "eq_active", getattr(model, "eq_active0", None))
    if active is None or not bool(active[index]):
        raise ControlError("fixed-base weld must be active")
    if int(model.neq) != 1:
        raise ControlError("strict fixed-stance scene permits only its base weld equality")


class PDController:
    """Execute the first four limited targets; replan from an explicit anchor.

    The controller receives model/data internally. Neither is a policy input.
    Construction intentionally fails while the velocity limit is unresolved.
    """

    def __init__(self, model, data, config, *, contract: Optional[ActionContract] = None, mujoco_api=None):
        configured_contract = ActionContract.from_config(config)
        if contract is not None and contract.contract_sha256 != configured_contract.contract_sha256:
            raise ControlError("provided action contract differs from source-audited configuration")
        self.contract = configured_contract
        self.cadence = ControlCadence.from_config(config)
        if not math.isclose(self.contract.control_hz, self.cadence.control_hz, rel_tol=1e-9):
            raise ControlError("action contract and controller frequencies differ")
        self.api = _api(mujoco_api)
        self.model, self.data = model, data
        if not math.isclose(float(model.opt.timestep), self.cadence.physics_dt_s, rel_tol=1e-9):
            raise ControlError("model timestep differs from the explicit control cadence")
        assert_fixed_stance(model, data, mujoco_api=self.api)
        self.controlled = controlled_joints(model, mujoco_api=self.api)
        self.bindings = action_joint_bindings(self.controlled, self.contract.joint_names)
        self.hold_targets = {name: float(data.qpos[item["qpos_id"]]) for name, item in self.controlled.items()}
        self.previous_target = tuple(self.hold_targets[name] for name in self.contract.joint_names)
        self.body_gain_scale = config["control"]["body_gain_scale"]
        self.hand_gain_scale = config["control"]["hand_gain_scale"]
        apply_regularized_dynamics(model, joint_names=tuple(self.controlled), mujoco_api=self.api)

    def _validate_joint_ranges(self, targets):
        for target in targets:
            for name, value, item in zip(self.contract.joint_names, target, self.bindings):
                if bool(self.model.jnt_limited[item["joint_id"]]):
                    low, high = self.model.jnt_range[item["joint_id"]]
                    if not float(low) <= value <= float(high):
                        raise ControlError("limited command outside mechanical joint range for {}".format(name))

    def execute_chunk(self, anchor_qpos: Sequence[float], delta_chunk: Sequence[Sequence[float]],
                      *, joint_names: Sequence[str], observer=None, substep_observer=None):
        self.contract.assert_joint_order(joint_names)
        actual_anchor = tuple(float(self.data.qpos[item["qpos_id"]]) for item in self.bindings)
        if len(anchor_qpos) != 31 or any(not math.isclose(_finite(a, "anchor_qpos"), b, abs_tol=1e-7, rel_tol=1e-7)
                                        for a, b in zip(anchor_qpos, actual_anchor)):
            raise ControlError("planning anchor does not match the visited state before execution")
        assert_fixed_stance(self.model, self.data, mujoco_api=self.api)
        targets = self.contract.executable_targets(anchor_qpos, delta_chunk, joint_names=joint_names,
                                                   previous_target=self.previous_target)
        requested = self.contract.decode_chunk(anchor_qpos, delta_chunk, joint_names=joint_names)[:self.contract.execute_steps]
        return self._execute_targets(targets, requested, joint_names, observer, substep_observer)

    def execute_target(self, requested_target: Sequence[float], *, joint_names: Sequence[str],
                       observer=None, substep_observer=None):
        """An expert's one feedback step, through the same codec/limits/PD.

        The model still produces eight anchored targets and executes four.
        Expert recording observes every command interval rather than holding an
        unavailable future observation. It never bypasses the command limits.
        """
        self.contract.assert_joint_order(joint_names)
        anchor = tuple(float(self.data.qpos[item["qpos_id"]]) for item in self.bindings)
        requested = (tuple(requested_target),) * self.contract.chunk_steps
        encoded = self.contract.encode_chunk(anchor, requested, joint_names=joint_names)
        limited = self.contract.executable_targets(anchor, encoded, joint_names=joint_names,
                                                   previous_target=self.previous_target)
        return self._execute_targets(limited[:1], requested[:1], joint_names, observer, substep_observer)[0]

    def _execute_targets(self, targets, requested, joint_names, observer, substep_observer):
        assert_fixed_stance(self.model, self.data, mujoco_api=self.api)
        self._validate_joint_ranges(targets)
        for index, target in enumerate(targets):
            start_time = float(getattr(self.data, "time", 0.0))
            if observer is not None:
                observer(event="before", step_index=index, requested_target=requested[index],
                         executed_target=target, model=self.model, data=self.data, start_time_s=start_time)
            commanded = map_action_targets(target, joint_names, self.controlled, self.hold_targets)
            for substep in range(self.cadence.physics_substeps):
                for name, item in self.controlled.items():
                    torque = pd_torque(name, commanded[name], self.data.qpos[item["qpos_id"]],
                                       self.data.qvel[item["qvel_id"]], self.data.qfrc_bias[item["qvel_id"]],
                                       body_gain_scale=self.body_gain_scale, hand_gain_scale=self.hand_gain_scale)
                    self.data.ctrl[item["actuator_id"]] = torque
                self.api.mj_step(self.model, self.data)
                # mj_step integrates qpos/time after evaluating geometry. Refresh
                # FK/contact/sensors at the new state without advancing time.
                self.api.mj_forward(self.model, self.data)
                if any(not math.isfinite(float(value)) for values in (self.data.qpos, self.data.qvel, getattr(self.data, "qacc", ())) for value in values):
                    raise ControlError("MuJoCo became non-finite during physical execution")
                if substep_observer is not None:
                    substep_observer(substep_index=substep, control_step_index=index,
                                     model=self.model, data=self.data)
            self.previous_target = target
            if observer is not None:
                observer(event="after", step_index=index, requested_target=requested[index],
                         executed_target=target, model=self.model, data=self.data, start_time_s=start_time)
        return targets

    def metadata(self):
        return {
            "protocol": "g1_pd_substep_feedback_v2", "profile": "unitree_regularized_robot_only_v1",
            "post_step_state_sync": "mj_forward_after_each_mj_step_no_time_advance",
            "nominal_control_hz": self.cadence.nominal_control_hz,
            "actual_control_hz": self.cadence.control_hz,
            "physics_substeps": self.cadence.physics_substeps,
            "physics_dt_s": self.cadence.physics_dt_s,
            "body_gain_scale": self.body_gain_scale, "hand_gain_scale": self.hand_gain_scale,
            "lower_body_hold": list(LOWER_BODY_JOINT_NAMES), "fixed_base_equality": FIXED_BASE_EQUALITY,
            "legacy_equivalence": False,
            "changes_from_legacy": ["regularization excludes free task objects", "force bounds migrated explicitly in scene manifest"],
        }
