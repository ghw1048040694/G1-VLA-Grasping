"""Feedback-gated progress along an already audited joint-space polyline.

No simulator, clock, configuration, state clipping or joint-wise route changes
belong here.  A scalar arc-time uses each segment's maximum joint travel time.
The caller owns actual task/contact gates and must execute the returned target.
"""
from __future__ import annotations

import bisect
import copy
import math
from numbers import Real


_BOUND_EPS = 1e-10
_APPLIED_EPS = 1e-9


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(label + " must be a finite real number")
    return float(value)


def _vector(values, label, dimensions=None):
    try:
        result = tuple(_number(value, label) for value in values)
    except TypeError as exc:
        raise ValueError(label + " must be a numeric vector") from exc
    if not result or dimensions is not None and len(result) != dimensions:
        raise ValueError(label + " has an incorrect dimension")
    return result


def _inside(values, lower, upper):
    return all(low - _BOUND_EPS <= value <= high + _BOUND_EPS
               for value, low, high in zip(values, lower, upper))


def _same(first, second):
    return all(abs(a - b) <= _APPLIED_EPS for a, b in zip(first, second))


def _same_direction(first, second):
    """Only exactly collinear forward knots can be skipped without a validator."""
    scale = sum(value * value for value in first)
    if scale == 0:
        return True
    ratio = sum(a * b for a, b in zip(first, second)) / scale
    return ratio >= 0 and all(abs(b - ratio * a) <= 1e-12
                              for a, b in zip(first, second))


class JointPathProgress:
    """Propose bounded references; acknowledge execution on the next call.

    Inputs are explicitly bound active-joint vectors (right seven in the G1
    caller).  ``previous_applied`` must be the real preceding applied command,
    not a predicted state.  A proposed endpoint alone is not ``complete``:
    completion additionally needs its execution acknowledged and a fresh true
    ``tracking_ready`` from the caller.  ``end_reference_reached`` describes
    only reference progress and never authorizes closure or task success.

    A geometry callback receives (from_reference, to_reference, context) and
    returns bool or a dict with ``passed``.  Each crossed polyline piece and
    the actual applied-command-to-candidate and raw-state-to-candidate chords
    are checked before an
    atomic commit.  A rejection leaves both reference and progress untouched.
    """

    def __init__(self, q_waypoints, velocity_limits, period_s, lower, upper, *,
                 max_anchor_delta=.08, chord_validator=None):
        self._vmax = _vector(velocity_limits, "velocity_limits")
        self._dimensions = len(self._vmax)
        if any(value < 0 for value in self._vmax):
            raise ValueError("velocity_limits must be nonnegative")
        self._lower = _vector(lower, "lower", self._dimensions)
        self._upper = _vector(upper, "upper", self._dimensions)
        if any(a > b for a, b in zip(self._lower, self._upper)):
            raise ValueError("mechanical bounds are reversed")
        self._dt = _number(period_s, "period_s")
        self._anchor = _number(max_anchor_delta, "max_anchor_delta")
        if self._dt <= 0 or self._anchor <= 0:
            raise ValueError("period_s and max_anchor_delta must be positive")
        if chord_validator is not None and not callable(chord_validator):
            raise ValueError("chord_validator must be callable")
        self._validator = chord_validator
        points = [_vector(point, "q_waypoints", self._dimensions) for point in q_waypoints]
        if not points:
            raise ValueError("q_waypoints must contain at least one point")
        for point in points:
            if any(not low <= value <= high
                   for value, low, high in zip(point, self._lower, self._upper)):
                raise ValueError("waypoint lies outside mechanical bounds")
        self._points, self._times = [points[0]], [0.]
        self._original_waypoint_count = len(points)
        for point in points[1:]:
            delta = tuple(b - a for a, b in zip(self._points[-1], point))
            if not any(value != 0 for value in delta):
                continue
            if any(speed == 0 and change != 0 for change, speed in zip(delta, self._vmax)):
                raise ValueError("path changes a zero-velocity joint")
            duration = max(abs(change) / speed if speed else 0.
                           for change, speed in zip(delta, self._vmax))
            next_time = self._times[-1] + duration
            if not math.isfinite(next_time) or next_time <= self._times[-1]:
                raise ValueError("path arc-time cannot be represented finitely")
            self._points.append(point)
            self._times.append(next_time)
        self._points, self._times = tuple(self._points), tuple(self._times)
        self._u = self._executed_u = 0.
        self._reference = self._points[0]
        self._records = []

    @property
    def reference(self):
        return list(self._reference)

    @property
    def total_duration_s(self):
        """Minimum reference arc-time; not a claim of actual elapsed time."""
        return self._times[-1]

    @property
    def records(self):
        return copy.deepcopy(self._records)

    def _location(self, progress):
        if len(self._points) == 1:
            return 0, 1.
        if progress >= self.total_duration_s:
            return len(self._points) - 2, 1.
        index = max(0, bisect.bisect_right(self._times, progress) - 1)
        return index, ((progress - self._times[index]) /
                       (self._times[index + 1] - self._times[index]))

    def _at(self, progress):
        if progress <= 0:
            return self._points[0]
        if progress >= self.total_duration_s:
            return self._points[-1]
        index, fraction = self._location(progress)
        return tuple(a + fraction * (b - a)
                     for a, b in zip(self._points[index], self._points[index + 1]))

    def _pieces(self, start, end):
        cursor = start
        pieces = []
        while cursor < end:
            index, _ = self._location(cursor)
            stop = min(end, self._times[index + 1])
            if stop <= cursor:
                raise ValueError("nonincreasing polyline progress")
            pieces.append((cursor, stop, self._at(cursor), self._at(stop), index))
            cursor = stop
        return pieces

    def _bounded_progress(self, budget_end, lower, upper):
        """Stop at the first box exit, including paths which leave and return."""
        stop = self._u
        first_direction = None
        geometry_limited = False
        for start, end, first, last, index in self._pieces(self._u, budget_end):
            direction = tuple(b - a for a, b in zip(first, last))
            if self._validator is None:
                if first_direction is None:
                    first_direction = direction
                elif not _same_direction(first_direction, direction):
                    geometry_limited = True
                    break
            fraction = 1.
            for value, change, low, high in zip(first, direction, lower, upper):
                if change > 0:
                    fraction = min(fraction, (high - value) / change)
                elif change < 0:
                    fraction = min(fraction, (low - value) / change)
            fraction = max(0., min(1., fraction))
            stop = end if fraction == 1. else start + fraction * (end - start)
            if fraction < 1.:
                break
        return stop, geometry_limited

    def _validate(self, actual, previous, candidate, candidate_u):
        if self._validator is None:
            return True, {"passed": True, "mode": "audited_piece_or_collinear_forward_knots",
                          "callback_used": False, "checks": []}
        checks = []
        pieces = self._pieces(self._u, candidate_u)
        operations = [(first, last, {"kind": "polyline_piece", "segment_index": index,
                       "start_u_s": start, "end_u_s": end})
                      for start, end, first, last, index in pieces]
        operations.append((previous, candidate, {"kind": "applied_command_shortcut",
                           "start_u_s": self._u, "end_u_s": candidate_u,
                           "crossed_piece_count": len(pieces)}))
        operations.append((actual, candidate, {"kind": "actual_state_to_candidate",
                           "start_u_s": self._executed_u, "end_u_s": candidate_u,
                           "actual": list(actual), "previous_applied": list(previous)}))
        for first, last, context in operations:
            try:
                result = self._validator(list(first), list(last), copy.deepcopy(context))
                if isinstance(result, bool):
                    passed, detail = result, {"passed": result}
                elif isinstance(result, dict) and isinstance(result.get("passed"), bool):
                    passed, detail = result["passed"], copy.deepcopy(result)
                else:
                    raise ValueError("validator must return bool or dict with boolean passed")
            except Exception as exc:
                passed, detail = False, {"passed": False,
                    "error": "{}: {}".format(type(exc).__name__, exc)}
            checks.append({"context": context, "from_reference": list(first),
                           "to_reference": list(last), "result": detail})
            if not passed:
                return False, {"passed": False, "callback_used": True, "checks": checks}
        return True, {"passed": True, "callback_used": True, "checks": checks}

    def step(self, actual, previous_applied, tracking_ready, *, maximum_advance_s=None):
        """Advance at most one original period, optionally less path arc-time.

        An explicit cap never changes the physical period used for velocity,
        anchor or acknowledgment checks. Zero is a legal acknowledged hold;
        caller phase clocks continue independently. None preserves the original
        behavior and ledger schema exactly.
        """
        advance_budget = self._dt
        if maximum_advance_s is not None:
            advance_budget = _number(maximum_advance_s, "maximum_advance_s")
            if advance_budget < 0 or advance_budget > self._dt:
                raise ValueError("maximum_advance_s must lie in [0, period_s]")
        actual = _vector(actual, "actual", self._dimensions)
        previous = _vector(previous_applied, "previous_applied", self._dimensions)
        if not isinstance(tracking_ready, bool):
            raise ValueError("tracking_ready must be an observed boolean")
        old_u = self._u
        lower = tuple(max(lo, state - self._anchor, old - speed * self._dt)
                      for lo, state, old, speed in zip(self._lower, actual, previous, self._vmax))
        upper = tuple(min(hi, state + self._anchor, old + speed * self._dt)
                      for hi, state, old, speed in zip(self._upper, actual, previous, self._vmax))
        box_exists = all(lo <= hi for lo, hi in zip(lower, upper))
        hold_legal = box_exists and _inside(self._reference, lower, upper)
        applied_matches = _same(previous, self._reference)
        if applied_matches:
            self._executed_u = self._u
        validation = {"passed": None, "callback_used": False, "checks": []}
        reason, blocked, geometry_limited = None, False, False
        candidate_u, candidate = self._u, self._reference
        if not box_exists:
            reason, blocked = "empty_mechanical_velocity_anchor_intersection", True
        elif not applied_matches:
            reason, blocked = "previous_applied_does_not_match_current_reference", True
        elif not hold_legal:
            reason, blocked = "current_reference_outside_raw_actual_or_command_bounds", True
        elif not tracking_ready:
            reason, blocked = "waiting_for_actual_reference_tracking", True
        elif self._u < self.total_duration_s:
            budget_end = min(self._u + advance_budget, self.total_duration_s)
            candidate_u, geometry_limited = self._bounded_progress(budget_end, lower, upper)
            candidate = self._at(candidate_u)
            if candidate_u <= self._u:
                if maximum_advance_s is not None and advance_budget == 0:
                    reason = "maximum_advance_s_zero_hold"
                else:
                    reason, blocked = "no_continuous_feasible_progress", True
            else:
                passed, validation = self._validate(actual, previous, candidate, candidate_u)
                if not passed:
                    reason, blocked = "geometry_validator_rejected", True
                    candidate_u, candidate = self._u, self._reference
                elif not _inside(candidate, lower, upper):
                    reason, blocked = "candidate_failed_limit_recheck", True
                    candidate_u, candidate = self._u, self._reference
                else:
                    self._u, self._reference = candidate_u, tuple(candidate)
        end_reference = self._u >= self.total_duration_s
        complete = bool(end_reference and self._executed_u >= self.total_duration_s
                        and tracking_ready and applied_matches and not blocked)
        # If the existing point itself is illegal, provide no invented fallback.
        reference = list(self._reference) if hold_legal else None
        if not applied_matches:
            reference = None
        if not blocked:
            reference = list(self._reference)
        mechanical = reference is not None and _inside(reference, self._lower, self._upper)
        velocity = reference is not None and all(
            abs(value - old) <= speed * self._dt + _BOUND_EPS
            for value, old, speed in zip(reference, previous, self._vmax))
        anchor = reference is not None and all(
            abs(value - state) <= self._anchor + _BOUND_EPS
            for value, state in zip(reference, actual))
        segment, fraction = self._location(self._u)
        record = {"reference": reference, "u_s": self._u,
                  "u": self._u / self.total_duration_s if self.total_duration_s else 1.,
                  "path_time_s": self._u, "total_duration_s": self.total_duration_s,
                  "old_u_s": old_u, "executed_u_s": self._executed_u,
                  "advanced_u_s": self._u - old_u,
                  "segment_index": segment, "segment": segment, "segment_fraction": fraction,
                  "end_reference_reached": bool(end_reference), "complete": complete,
                  "blocked": blocked, "reason": reason, "tracking_ready": tracking_ready,
                  "actual": list(actual), "previous_applied": list(previous),
                  "period_s": self._dt, "velocity_limits": list(self._vmax),
                  "max_anchor_delta": self._anchor,
                  "limit_checks": {"passed": bool(mechanical and velocity and anchor),
                      "mechanical_bounds": bool(mechanical), "velocity_bounds": bool(velocity),
                      "raw_actual_anchor_bounds": bool(anchor), "box_exists": box_exists,
                      "current_reference_hold_legal": bool(hold_legal),
                      "previous_applied_matches_reference": bool(applied_matches),
                      "intersection_lower": list(lower), "intersection_upper": list(upper),
                      "geometry_limited_at_corner_without_callback": geometry_limited},
                  "validation": validation}
        if maximum_advance_s is not None:
            record["maximum_advance_s"] = advance_budget
        self._records.append(copy.deepcopy(record))
        return copy.deepcopy(record)
