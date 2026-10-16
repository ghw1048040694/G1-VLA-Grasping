"""Versioned seven-joint probe guards; no simulator or optimizer side effects.

The caller supplies the unchanged velocity/anchor/mechanical intersection and
probe cap. A first-order joint-space guard is not a physical overshoot proof;
the independent 2 ms object-height and contact gates remain necessary.
"""

from __future__ import annotations


PROBE_GUARD_MODES = ("legacy_scalar_v1", "box_halfspace_projected_v2")
PROJECTION_BISECTION_STEPS = 64
LEGACY_HEIGHT_ROUNDOFF_TOLERANCE_M = 1e-7
LEGACY_JOINT_ROUNDOFF_TOLERANCE_RAD = 1e-6


def apply_probe_guard(torch, desired_arm, previous_arm, current_arm, low_arm,
                      high_arm, point_jz, palm_z, start_palm_z, cap_m, *, mode):
    """Return targets and tensor evidence without mutating any input.

    Inputs are ``(N, 7)`` floating tensors, with ``(N,)`` palm heights and a
    nonnegative scalar or ``(N,)`` cap. The projected mode minimizes distance
    to ``desired_arm`` within the supplied box and linear height halfspace.
    An infeasible intersection holds ``previous_arm`` and requires the caller
    to mark that environment unsafe; a held target is not certified safe.

    ``factor`` always reports the legacy reference factor. Only legacy mode
    applies it; projected mode reports ``scalar_factor_applied=False`` and its
    actual correction/dual variable instead. Legacy violations are evidence,
    not silently repaired, so the historical controller remains reproducible.
    Legacy alone distinguishes fixed float32 roundoff tolerances from material
    bound violations; the projected mode and physical overshoot gates stay strict.
    """
    if mode not in PROBE_GUARD_MODES:
        raise ValueError("Unknown probe guard mode: " + str(mode))
    matrices = (desired_arm, previous_arm, current_arm, low_arm, high_arm, point_jz)
    if (not all(torch.is_tensor(value) for value in matrices) or
            desired_arm.ndim != 2 or desired_arm.shape[1] != 7):
        raise ValueError("Probe guard requires batched (N, 7) tensors")
    if desired_arm.dtype not in (torch.float32, torch.float64):
        raise ValueError("Probe guard requires float32 or float64")
    for value in matrices + (palm_z, start_palm_z):
        if (not torch.is_tensor(value) or value.device != desired_arm.device or
                value.dtype != desired_arm.dtype):
            raise ValueError("Probe tensors must share floating dtype and device")
    if any(value.shape != desired_arm.shape for value in matrices):
        raise ValueError("Probe joint tensor shapes differ")
    count = desired_arm.shape[0]
    if palm_z.shape != (count,) or start_palm_z.shape != (count,):
        raise ValueError("Probe heights must have shape (N,)")
    cap = torch.as_tensor(cap_m, dtype=desired_arm.dtype, device=desired_arm.device)
    if cap.ndim == 0:
        cap = cap.expand(count)
    if cap.shape != (count,):
        raise ValueError("Probe cap must be scalar or have shape (N,)")
    if not all(bool(torch.isfinite(value).all()) for value in matrices + (palm_z, start_palm_z, cap)):
        raise FloatingPointError("Nonfinite probe guard input")
    if bool((cap < 0).any()):
        raise ValueError("Probe cap must be nonnegative")

    with torch.no_grad():
        ceiling = start_palm_z + cap

        def prediction(target):
            return palm_z + (point_jz * (target-current_arm)).sum(dim=-1)

        previous_prediction = prediction(previous_arm)
        increment = desired_arm-previous_arm
        predicted_increment = (point_jz*increment).sum(dim=-1)
        allowance = (ceiling-previous_prediction).clamp_min(0)
        factor = torch.where(predicted_increment > allowance,
            allowance/predicted_increment.clamp_min(1e-12), torch.ones_like(allowance)).clamp(0, 1)
        box_feasible = (low_arm <= high_arm).all(dim=-1)
        clipped = torch.maximum(torch.minimum(desired_arm, high_arm), low_arm)
        minimum_corner = torch.where(point_jz > 0, low_arm,
            torch.where(point_jz < 0, high_arm, clipped))
        minimum_prediction = prediction(minimum_corner)
        feasible = box_feasible & (minimum_prediction <= ceiling)
        dual = torch.zeros_like(palm_z)

        if mode == "legacy_scalar_v1":
            target = previous_arm + increment*factor[:, None]
        else:
            # KKT form x(lambda)=clip(desired-lambda*Jz, low, high).
            # Saturating every nonzero component gives a feasible upper dual
            # bracket whenever the supplied box/halfspace intersection exists.
            nonzero = point_jz != 0
            denominator = torch.where(nonzero, point_jz, torch.ones_like(point_jz))
            saturation = torch.where(nonzero, (desired_arm-minimum_corner)/denominator,
                                     torch.zeros_like(point_jz)).clamp_min(0).amax(dim=-1)
            upper = saturation + torch.maximum(saturation.abs(), torch.ones_like(saturation)) * (
                8*torch.finfo(desired_arm.dtype).eps)
            lower = torch.zeros_like(upper)

            def candidate(value):
                return torch.maximum(torch.minimum(desired_arm-value[:, None]*point_jz, high_arm), low_arm)

            active = feasible & (prediction(clipped) > ceiling)
            if not bool(torch.isfinite(upper).all()):
                raise FloatingPointError("Nonfinite probe projection dual bracket")
            if bool((active & (prediction(candidate(upper)) > ceiling)).any()):
                raise FloatingPointError("Probe projection could not establish a feasible upper bracket")
            for _ in range(PROJECTION_BISECTION_STEPS):
                middle = lower + (upper-lower)*0.5
                above = prediction(candidate(middle)) > ceiling
                lower = torch.where(active & above, middle, lower)
                upper = torch.where(active & ~above, middle, upper)
            dual = torch.where(active, upper, dual)
            target = torch.where(active[:, None], candidate(upper), clipped)
            target = torch.where(feasible[:, None], target, previous_arm)

        predicted_post = prediction(target)
        lower_violation = (low_arm-target).clamp_min(0)
        upper_violation = (target-high_arm).clamp_min(0)
        height_violation = (predicted_post-ceiling).clamp_min(0)
        postcondition = (box_feasible & (lower_violation == 0).all(dim=-1) &
                         (upper_violation == 0).all(dim=-1) & (height_violation == 0))
        numerical_postcondition = (
            (lower_violation <= LEGACY_JOINT_ROUNDOFF_TOLERANCE_RAD).all(dim=-1) &
            (upper_violation <= LEGACY_JOINT_ROUNDOFF_TOLERANCE_RAD).all(dim=-1) &
            (height_violation <= LEGACY_HEIGHT_ROUNDOFF_TOLERANCE_M))
        requires_unsafe = (~box_feasible | ~numerical_postcondition if mode == "legacy_scalar_v1"
                           else ~feasible | ~postcondition)
        evidence = {"factor": factor, "scalar_factor_applied": torch.full_like(feasible, mode == "legacy_scalar_v1"),
            "feasible": feasible, "box_feasible": box_feasible, "requires_unsafe": requires_unsafe,
            "infeasible_previous_hold": ~feasible if mode != "legacy_scalar_v1" else torch.zeros_like(feasible),
            "predicted_pre": prediction(desired_arm), "predicted_previous": previous_prediction,
            "predicted_increment": predicted_increment, "allowance": allowance,
            "predicted_post": predicted_post, "minimum_box_prediction": minimum_prediction,
            "ceiling": ceiling, "dual_lambda": dual, "target_correction": target-desired_arm,
            "lower_limit_violations": lower_violation, "upper_limit_violations": upper_violation,
            "height_violation_m": height_violation, "postcondition_passed": postcondition,
            "numerical_postcondition_passed": numerical_postcondition,
            "legacy_roundoff_only": box_feasible & ~postcondition & numerical_postcondition & (mode == "legacy_scalar_v1"),
            "legacy_height_roundoff_tolerance_m": torch.full_like(palm_z, LEGACY_HEIGHT_ROUNDOFF_TOLERANCE_M),
            "legacy_joint_roundoff_tolerance_rad": torch.full_like(palm_z, LEGACY_JOINT_ROUNDOFF_TOLERANCE_RAD)}
        if not bool(torch.isfinite(target).all()) or not all(bool(torch.isfinite(value).all()) for value in evidence.values()):
            raise FloatingPointError("Nonfinite probe guard output or evidence")
        # Do not return an uncertified projected result under a feasible label.
        if mode != "legacy_scalar_v1" and bool((feasible & ~postcondition).any()):
            raise FloatingPointError("Probe projection failed its strict box/height postcondition")
        return target, evidence
