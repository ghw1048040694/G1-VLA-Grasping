"""Pure, pinned pre-contact control changes independent of acceptance geometry.

The static evidence identifies an authored, sampled geometry pre-screen.  It
does not certify a native manifold, continuous clearance, support, or dynamic
admission.  The caller verifies candidate file bytes before passing their SHA;
this module additionally pins the parsed candidate and original six frames.
It imports neither Torch nor a simulator, reads no files, and mutates no input.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math


CONTROL_PRESHAPE_VERSION = "g1_pinned_precontact_control_preshape_v1"
_CANDIDATE_SHA256 = "2daa38dafba243eae42cd1450d9a0030f7d0503e6073647309109f3cbc885a60"
_CANDIDATE_CANONICAL_SHA256 = "cf8f9964eaf7f30fc814757098730ffc1bac7ddba97b2297880378bde94fb92d"
_FRAMES_SHA256 = "3e107979a990e65a1996481bc237621db1787ecaf5541dbf0dceff577eafcc19"
_STATIC_SHA256 = "97f7afb5c1fed93bbd5a582530a576fde6c0072dd4daec54c671f9062df2b10a"
_ORIGINAL_TOLERANCE_RAD = .025
_HAND_JOINT_ORDER = tuple("right_hand_" + suffix + "_joint" for suffix in (
    "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"))
_NATIVE_HAND_LIMITS_RAD = (
    (-1.0471975511965976, 1.0471975511965976),
    (-1.0471975511965976, .6108651716189881),
    (-1.745329118836275, 0.0),
    (0.0, 1.5707963267948966),
    (0.0, 1.745329118836275),
    (0.0, 1.5707963267948966),
    (0.0, 1.745329118836275),
)
_FRAME1_OFFSET = [0, .01, 0, .006825593136154828, .006401722017027434,
                  .0028729982670930576, .008135523724513316]
_LATER_OFFSET = [0, .01, 0, .004337697881246272, .01578057645952391,
                 .0037687095397486714, .006497875709490897]
_CONTROL_PRESHAPE = {
    "version": CONTROL_PRESHAPE_VERSION,
    "acceptance_candidate_sha256": _CANDIDATE_SHA256,
    "acceptance_frames_sha256": _FRAMES_SHA256,
    "static_geometry_evidence_sha256": _STATIC_SHA256,
    "static_geometry_qualification": "sampled_authored_geometry_only_not_native_support_or_dynamic_admission",
    "palm_offset_world_m": [.0005, 0, 0],
    "palm_offset_schedule": "smoothstep_existing_first_stage_then_hold_v1",
    "palm_offset_freezes_on_first_fresh_physical_contact": True,
    "contact_latch_cadence_s": .002,
    "palm_reference_freeze_boundary": "next_macro_after_first_same_episode_native_loaded_point_v1",
    "force_mode_hand_position_springs_remain_zero": True,
    "execution_frame_hand_offsets_rad": [[0] * 7, _FRAME1_OFFSET,
                                          *[deepcopy(_LATER_OFFSET) for _ in range(4)]],
    "first_stage_duration_s": .9,
    "original_actual_geometry_tolerance_rad": _ORIGINAL_TOLERANCE_RAD,
    "all_environments_same_preshape": True,
    "original_reset_and_episode_budget_unchanged": True,
    "training_allowed": False,
}


def control_preshape_contract():
    """Return a fresh exact v1 declaration; it is not a physical qualification."""
    return deepcopy(_CONTROL_PRESHAPE)


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _finite_vector(value, size, label):
    if (not isinstance(value, list) or len(value) != size or
            any(not _finite_number(number) for number in value)):
        raise ValueError(label + " requires exactly " + str(size) + " finite numbers")
    return value


def validate_control_preshape(value):
    """Validate the exact optional v1 declaration without importing a backend."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(_CONTROL_PRESHAPE):
        raise ValueError("Control preshape requires the exact v1 declaration keys")
    _finite_vector(value["palm_offset_world_m"], 3, "Preshape world palm offset")
    offsets = value["execution_frame_hand_offsets_rad"]
    if not isinstance(offsets, list) or len(offsets) != 6:
        raise ValueError("Control preshape requires exactly six hand-frame offsets")
    for offset in offsets:
        _finite_vector(offset, 7, "Preshape hand-frame offset")
    for key in ("first_stage_duration_s", "original_actual_geometry_tolerance_rad", "contact_latch_cadence_s"):
        number = value[key]
        if not _finite_number(number) or number <= 0:
            raise ValueError("Control preshape requires finite positive " + key)
    for key in ("all_environments_same_preshape", "original_reset_and_episode_budget_unchanged", "training_allowed",
                "palm_offset_freezes_on_first_fresh_physical_contact", "force_mode_hand_position_springs_remain_zero"):
        if type(value[key]) is not bool:
            raise ValueError("Control preshape requires an explicit boolean " + key)
    if value != _CONTROL_PRESHAPE:
        raise ValueError("Control preshape differs from the pinned v1 geometry, timing, or provenance")
    return deepcopy(value)


def _canonical_sha256(value):
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Control preshape requires a finite JSON candidate") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _original_frames(candidate):
    if not isinstance(candidate, dict):
        raise ValueError("Control preshape requires a parsed geometry candidate")
    evidence = candidate.get("geometry_evidence")
    detail = evidence.get("source_detail") if isinstance(evidence, dict) else None
    frames = detail.get("stage_keyframes") if isinstance(detail, dict) else None
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("Control preshape requires finite original hand keyframes")
    values = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Original hand keyframes must be objects")
        values.append(_finite_vector(frame.get("hand_target_rad", frame.get("target_rad")),
                                     7, "Original hand keyframe"))
    return deepcopy(frames), values


def _envelope(values):
    return ([min(row[index] for row in values) for index in range(7)],
            [max(row[index] for row in values) for index in range(7)])


def _validate_pinned_candidate(candidate, candidate_sha256, frames, values):
    if candidate_sha256 != _CANDIDATE_SHA256:
        raise ValueError("Control preshape requires the pinned acceptance candidate file SHA256")
    if len(frames) != 6 or _canonical_sha256(frames) != _FRAMES_SHA256:
        raise ValueError("Control preshape requires the unchanged original six-frame canonical SHA256")
    if (candidate.get("schema_version") != 2 or
            candidate.get("candidate_id") != "opposed_face_effort_pinch_v2" or
            candidate.get("hand_joint_order") != list(_HAND_JOINT_ORDER)):
        raise ValueError("Control preshape candidate schema, identity, or joint order changed")
    evidence = candidate["geometry_evidence"]
    mechanical = evidence.get("hand_joint_limits_rad")
    domain = candidate.get("expected_geometry_q_domain")
    if (mechanical != [list(pair) for pair in _NATIVE_HAND_LIMITS_RAD] or
            not isinstance(domain, dict) or
            domain.get("native_joint_limits_rad") != mechanical):
        raise ValueError("Control preshape requires the unchanged native seven-hand mechanical limits")
    reset = candidate.get("reset_joint_positions_rad")
    if (not isinstance(reset, dict) or len(reset) != 43 or
            any(not _finite_number(number) for number in reset.values()) or
            [reset.get(name) for name in _HAND_JOINT_ORDER] != values[0] or
            candidate.get("source_hand_rad") != values[0] or
            candidate.get("target_rad") != values[-1]):
        raise ValueError("Control preshape must preserve the original authored reset, source, and acceptance target")
    for index, frame in enumerate(frames[1:]):
        duration = frame.get("duration_s")
        if type(duration) not in (int, float) or duration != (.9, .8, .6, .4, .6)[index]:
            raise ValueError("Control preshape must preserve every original stage duration")
    original_lower, original_upper = _envelope(values)
    if domain.get("lower_rad") != original_lower or domain.get("upper_rad") != original_upper:
        raise ValueError("Acceptance geometry envelope differs from the original six frames")
    if _canonical_sha256(candidate) != _CANDIDATE_CANONICAL_SHA256:
        raise ValueError("Control preshape parsed candidate differs from the pinned original resource")


def resolve_preshape_frames(declaration, candidate, candidate_sha256):
    """Return original score frames separately from declared execution frames.

    ``None`` preserves the input frame content and does not enforce the new v1
    candidate pin.  It returns the same six resolver keys, without declaration
    metadata.  Existing runtime ``None`` paths need not call this helper.

    Domain values here are pure Python audit values.  Runtime score consumers
    must retain their original Torch dtype and operation order on
    ``original_frames`` to preserve exact historical floating-point bounds.
    Execution references use their own min/max intersected with the immutable
    original score domain, with no added execution tolerance.
    """
    selected = validate_control_preshape(declaration)
    frames, original_values = _original_frames(candidate)
    if selected is not None:
        _validate_pinned_candidate(candidate, candidate_sha256, frames, original_values)
    original_lower, original_upper = _envelope(original_values)
    domain_lower = [value - _ORIGINAL_TOLERANCE_RAD for value in original_lower]
    domain_upper = [value + _ORIGINAL_TOLERANCE_RAD for value in original_upper]
    execution = deepcopy(frames)
    if selected is not None:
        offsets = selected["execution_frame_hand_offsets_rad"]
        for index, frame in enumerate(execution):
            key = "hand_target_rad" if "hand_target_rad" in frame else "target_rad"
            frame[key] = [original_values[index][joint] + offsets[index][joint] for joint in range(7)]
    execution_values = [frame.get("hand_target_rad", frame.get("target_rad")) for frame in execution]
    if selected is not None:
        for row in execution_values:
            for joint, number in enumerate(row):
                lower, upper = _NATIVE_HAND_LIMITS_RAD[joint]
                if (not math.isfinite(number) or not lower <= number <= upper or
                        not domain_lower[joint] <= number <= domain_upper[joint]):
                    raise ValueError("Execution preshape leaves original acceptance domain or native mechanical limits")
    execution_lower, execution_upper = _envelope(execution_values)
    reference_lower = [max(execution_lower[index], domain_lower[index]) for index in range(7)]
    reference_upper = [min(execution_upper[index], domain_upper[index]) for index in range(7)]
    if any(lower > upper for lower, upper in zip(reference_lower, reference_upper)):
        raise ValueError("Execution preshape has an empty original-domain reference intersection")
    return {
        "original_frames": frames,
        "execution_frames": execution,
        "original_domain_lower7_rad": domain_lower,
        "original_domain_upper7_rad": domain_upper,
        "execution_reference_lower7_rad": reference_lower,
        "execution_reference_upper7_rad": reference_upper,
    }
