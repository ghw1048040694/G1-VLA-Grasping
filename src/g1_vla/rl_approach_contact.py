"""Read-only native path-collision witness for the pinned G1 articulation.

This module creates contact *views*, never reporters, filters or physics state.
Call ``reset`` only after a real episode reset, then ``sample_completed`` once
for every completed 2 ms step, including the terminal step before reset.  A
handoff is not a reset.  Absence is certified only for the explicitly covered
normal-contact pairs; Cube support, friction and mesh separation are not proved.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

from .rl_collision_model import bilateral_fixed_cluster_contract


PROTOCOL = "g1_native_completed_2ms_approach_pair_witness_v1"
_ATOL_N, _RTOL, _NORMAL_TOL = 1e-5, 1e-4, 1e-3


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _attribute(prim, name):
    attr = prim.GetAttribute(name)
    return attr.Get() if attr and attr.IsValid() else None


def _targets(prim, name):
    rel = prim.GetRelationship(name)
    return [str(p) for p in rel.GetTargets()] if rel and rel.IsValid() else []


def approach_contact_routes(stage, num_envs):
    """Read every clone's real graph/inventory and exact existing 30 filters.

    Descendants of the shoulder-pitch joint include fixed camera/palm bodies.
    Only pairs already mechanically disabled by a joint or the verified 30-pair
    model are excluded. Nonadjacent moving/moving pairs remain covered in both
    directions. Including the complete Robot inventory in each filter row keeps
    the native route rectangular and makes omissions visible at binding time.
    """
    if type(num_envs) is not int or not 1 <= num_envs <= 8:
        raise ValueError("approach witness requires 1..8 exact environments")
    roots = ["/World/envs/env_"+str(e)+"/Robot" for e in range(num_envs)]
    bodies, graphs, articulations, filters = ({r: {} for r in roots},
        {r: [] for r in roots}, {}, {})
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/envs/"):
            continue
        apis = set(prim.GetAppliedSchemas())
        root = next((r for r in roots if path == r or path.startswith(r+"/")), None)
        kind = prim.GetTypeName()
        if root is None and "/Robot/" in path and "PhysicsRigidBodyAPI" in apis:
            raise ValueError("Robot rigid body outside declared exact environments")
        if kind.startswith("Physics") and kind.endswith("Joint") and kind not in (
                "PhysicsFixedJoint", "PhysicsRevoluteJoint"):
            raise ValueError("unsupported extra native joint type")
        if root is not None and "PhysicsRigidBodyAPI" in apis:
            name = path[len(root)+1:]
            if not name or "/" in name or name in bodies[root]:
                raise ValueError("duplicate or nested Robot rigid body")
            bodies[root][name] = path
        if kind in ("PhysicsFixedJoint", "PhysicsRevoluteJoint"):
            if root is None:
                raise ValueError("joint outside an exact declared Robot environment")
            graphs[root].append({"name": prim.GetName(), "type": kind, "path": path,
                "body0": _targets(prim, "physics:body0"),
                "body1": _targets(prim, "physics:body1"),
                "physics:jointEnabled": _attribute(prim, "physics:jointEnabled"),
                "physics:collisionEnabled": _attribute(prim, "physics:collisionEnabled")})
        if "PhysicsArticulationRootAPI" in apis:
            if root is None or root in articulations or _attribute(
                    prim, "physxArticulation:enabledSelfCollisions") is not True:
                raise ValueError("one self-colliding articulation per exact environment required")
            articulations[root] = path
        if "PhysicsFilteredPairsAPI" in apis:
            if path in filters:
                raise ValueError("duplicate filtered-pair prim")
            filters[path] = _targets(prim, "physics:filteredPairs")
    if set(articulations) != set(roots):
        raise ValueError("missing native articulation self-collision readback")
    expected_filters, sensors, filter_rows, routes, per_env = {}, [], [], [], []
    bilateral = bilateral_fixed_cluster_contract()
    disabled = {tuple(pair) for pair in bilateral["contract"]["excluded_body_pairs"]}
    for env, root in enumerate(roots):
        bilateral_fixed_cluster_contract(graphs[root])  # Full 55-joint hash, every clone.
        graph_bodies = {p.rsplit("/", 1)[-1] for row in graphs[root]
                        for key in ("body0", "body1") for p in row[key]}
        if set(bodies[root]) != graph_bodies:
            raise ValueError("actual Robot rigid-body inventory differs from full pinned graph")
        for a, b in sorted(disabled):
            expected_filters.setdefault(root+"/"+a, []).append(root+"/"+b)
        children = {}
        mechanical = set(disabled)
        for row in graphs[root]:
            a = row["body0"][0].rsplit("/", 1)[-1] if row["body0"] else None
            b = row["body1"][0].rsplit("/", 1)[-1] if row["body1"] else None
            if a is not None and b is not None:
                children.setdefault(a, []).append(b)
                # The pinned graph verifies collisionEnabled=False. No other
                # graph distance, name prefix or entire subtree is excluded.
                mechanical.add(tuple(sorted((a, b))))
        shoulder = next(row for row in graphs[root] if row["name"] == "right_shoulder_pitch_joint")
        pending = [shoulder["body1"][0].rsplit("/", 1)[-1]]
        moving = set()
        while pending:
            name = pending.pop()
            if name in moving:
                raise ValueError("cycle or repeated child in pinned moving graph")
            moving.add(name)
            pending.extend(children.get(name, ()))
        if not {"right_hand_camera_base_link", "right_hand_palm_link",
                "right_wrist_yaw_link", "right_shoulder_pitch_link"} <= moving:
            raise ValueError("moving chain omitted a fixed wrist attachment")
        names, moving = sorted(bodies[root]), sorted(moving)
        partners = [("Table", "/World/envs/env_"+str(env)+"/Table")]+[
            (name, bodies[root][name]) for name in names]
        covered, excluded = [], []
        for body in moving:
            sensor_index = len(sensors)
            sensors.append(bodies[root][body])
            filter_rows.append([p for _, p in partners])
            for column, (other, path) in enumerate(partners):
                reason = ("same_body" if body == other else
                    "verified_existing_joint_or_30pair_filter" if tuple(sorted((body, other))) in mechanical else None)
                row = {"env_id": env, "sensor_index": sensor_index, "filter_index": column,
                    "sensor_body": body, "sensor_path": bodies[root][body],
                    "partner_body": other, "partner_path": path,
                    "category": "table" if other == "Table" else
                        "moving_internal" if other in moving else "nonmoving_robot",
                    "covered": reason is None, "exclusion_reason": reason}
                routes.append(row)
                (covered if reason is None else excluded).append([body, other])
        per_env.append({"env_id": env, "moving_body_names": moving,
            "nonmoving_body_names": sorted(set(names)-set(moving)),
            "actual_body_inventory": bodies[root],
            "covered_ordered_pairs": covered, "excluded_ordered_pairs": excluded,
            "covered_pairs_sha256": _hash(covered), "excluded_pairs_sha256": _hash(excluded),
            "verified_55_joint_graph_sha256": bilateral["source_graph_sha256"]})
    if (set(filters) != set(expected_filters) or any(
            len(v) != len(set(v)) or sorted(v) != sorted(expected_filters[k])
            for k, v in filters.items())):
        raise ValueError("existing native filter relationships differ from exact bilateral30 model")
    result = {"protocol": PROTOCOL, "num_envs": num_envs,
        "sensor_paths": sensors, "filter_paths": filter_rows, "routes": routes,
        "per_env": per_env, "existing_30pair_filter_contract": bilateral,
        "existing_30pair_relationships_sha256": _hash(expected_filters),
        "mechanical_exclusions_are_observation_only": True,
        "cube_in_filtered_partners": False, "physics_or_reporting_writes": 0}
    result["source_binding_sha256"] = _hash(result)
    return result


class NativeApproachContactWitness:
    """Exact full-matrix observation plus per-episode irreversible clean history.

    Public results are JSON-safe, detached copies. ``available`` describes the
    current observation; ``per_env_clean`` also requires an unbroken sequence
    since the explicitly registered real reset. Unavailable data poisons history
    until that environment's next real reset, never until a handoff.
    """

    def __init__(self, torch, as_tensor, simulation_view, stage, num_envs,
                 physics_dt=.002, *, capacity_per_sensor=128, max_evidence_points=64):
        if type(num_envs) is not int or not 1 <= num_envs <= 8:
            raise ValueError("witness needs 1..8 exact environments")
        if (type(physics_dt) not in (int, float) or not math.isclose(float(physics_dt), .002, abs_tol=1e-12, rel_tol=0) or
                type(capacity_per_sensor) is not int or not 1 <= capacity_per_sensor <= 4096 or
                type(max_evidence_points) is not int or not 1 <= max_evidence_points <= 4096):
            raise ValueError("witness requires actual2ms and explicit bounded positive capacities")
        self.t, self.as_tensor, self.n, self.dt = torch, as_tensor, num_envs, float(physics_dt)
        self.max_evidence_points = max_evidence_points
        self.view, self.routes, self._last_native = None, None, None
        self._epoch = [None]*num_envs
        self._reset_native = [None]*num_envs
        self._clean = [False]*num_envs
        self._samples = [0]*num_envs
        self._first_failure = [None]*num_envs
        self._peak_force = [0.]*num_envs
        self._peak_depth = [0.]*num_envs
        self._worst_force = [None]*num_envs
        self._worst_depth = [None]*num_envs
        self.readback = {"protocol": PROTOCOL, "available": False,
            "physics_or_reporting_writes": 0, "cadence_s": self.dt,
            "friction_measured": False, "bounded_evidence_limit": max_evidence_points}
        try:
            binding = approach_contact_routes(stage, num_envs)
            self.routes = binding["routes"]
            sensors, filters = binding["sensor_paths"], binding["filter_paths"]
            reports = []
            for path in dict.fromkeys(sensors+[p for row in filters for p in row]):
                prim = stage.GetPrimAtPath(path)
                apis = set(prim.GetAppliedSchemas()) if prim.IsValid() else set()
                threshold = _attribute(prim, "physxContactReport:threshold") if prim.IsValid() else None
                if (not {"PhysicsRigidBodyAPI", "PhysxContactReportAPI"} <= apis or
                        isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or
                        not math.isfinite(threshold) or threshold != 0):
                    raise ValueError("preexisting rigid reporter with exact zero threshold required: "+path)
                reports.append({"path": path, "threshold_n": threshold})
            self.capacity = capacity_per_sensor*len(sensors)
            view = simulation_view.create_rigid_contact_view(sensors,
                filter_patterns=filters, max_contact_data_count=self.capacity)
            observed_sensors = list(view.sensor_paths)
            observed_filters = [list(p) if not isinstance(p, str) else [p] for p in view.filter_paths]
            if (type(view.sensor_count) is not int or type(view.filter_count) is not int or
                    type(view.max_contact_data_count) is not int or
                    view.sensor_count != len(sensors) or view.filter_count != len(filters[0]) or
                    view.max_contact_data_count != self.capacity or
                    observed_sensors != sensors or observed_filters != filters):
                raise ValueError("native API cannot verify exact per-sensor full-matrix routing/capacity")
            self.view = view
            self.readback.update(available=True, unavailable_reason=None, binding=binding,
                preexisting_contact_reports=reports, raw_capacity_global=self.capacity,
                source_binding_sha256=binding["source_binding_sha256"],
                normal_matrix_reconstruction={"atol_n": _ATOL_N, "rtol": _RTOL},
                unfiltered_net_residual_semantics="net includes allowed Cube/other unfiltered partners; residual is not required to vanish",
                coverage_limit="normal loaded-contact witness for declared pairs; no friction, disjoint geometry or unlisted partner absence proof")
        except Exception as exc:
            self.readback.update(available=False, unavailable_reason=type(exc).__name__+": "+str(exc))

    def summary(self):
        return copy.deepcopy({**self.readback, "registered_episode_ids": self._epoch,
            "episode_reset_native": self._reset_native, "last_completed_native": self._last_native,
            "per_env_history_clean": self._clean, "valid_completed_samples": self._samples,
            "first_failure": self._first_failure,
            "episode_peak_abs_signed_point_force_n": self._peak_force,
            "episode_maximum_penetration_m": self._peak_depth,
            "episode_peak_force_contact": self._worst_force,
            "episode_deepest_contact": self._worst_depth})

    def _integers(self, value, size, name):
        t = self.t
        if t.is_tensor(value):
            if tuple(value.shape) != (size,) or value.dtype not in (t.int32, t.int64, t.uint32):
                raise ValueError(name+" requires an exact integer vector")
            values = value.detach().cpu().tolist()
        elif type(value) in (list, tuple):
            values = list(value)
        else:
            raise ValueError(name+" requires an integer vector")
        if len(values) != size or any(type(x) is not int or x < 0 for x in values):
            raise ValueError(name+" requires distinct-range nonnegative integer values")
        return values

    def reset(self, env_ids, episode_ids, reset_native):
        """Register a true reset. Re-registering the same epoch cannot clear it.

        ``episode_ids`` is the selected vector, in ``env_ids`` order. The caller
        must call after the physical reset and before its first completed step.
        Only monotonically newer episodes at the current native boundary pass.
        """
        size = len(env_ids)
        ids = self._integers(env_ids, size, "reset env_ids")
        epochs = self._integers(episode_ids, size, "reset episode_ids")
        if (not ids or len(set(ids)) != size or any(e >= self.n for e in ids) or
                type(reset_native) is not int or reset_native < 0 or
                (self._last_native is not None and reset_native != self._last_native)):
            raise ValueError("real reset must name distinct envs at the current native boundary")
        if any(self._epoch[e] is not None and epoch <= self._epoch[e] for e, epoch in zip(ids, epochs)):
            raise ValueError("same/stale episode cannot clear path history at handoff")
        if self._last_native is None:
            self._last_native = reset_native
        for e, epoch in zip(ids, epochs):
            self._epoch[e], self._reset_native[e] = epoch, reset_native
            self._clean[e], self._samples[e], self._first_failure[e] = True, 0, None
            self._peak_force[e], self._peak_depth[e] = 0., 0.
            self._worst_force[e], self._worst_depth[e] = None, None

    def _poison(self, env, native, reason):
        self._clean[env] = False
        if self._first_failure[env] is None:
            self._first_failure[env] = {"native": native, "episode_id": self._epoch[env], "reason": reason}

    def _read(self):
        t, view = self.t, self.view
        if view is None:
            raise ValueError(self.readback["unavailable_reason"])
        # Clone immediately: the SDK reuses buffers across getter calls.
        net = self.as_tensor(view.get_net_contact_forces(dt=self.dt)).detach().clone()
        matrix = self.as_tensor(view.get_contact_force_matrix(dt=self.dt)).detach().clone()
        raw = tuple(self.as_tensor(x).detach().clone() for x in view.get_contact_data(dt=self.dt))
        if len(raw) != 6:
            raise ValueError("expected forces,points,normals,separation,COUNTS,STARTS")
        force, point, normal, separation, counts, starts = raw
        s, f, capacity = view.sensor_count, view.filter_count, self.capacity
        if (force.shape not in ((capacity,), (capacity, 1)) or
                separation.shape not in ((capacity,), (capacity, 1)) or
                tuple(point.shape) != (capacity, 3) or tuple(normal.shape) != (capacity, 3) or
                tuple(counts.shape) != (s, f) or tuple(starts.shape) != (s, f) or
                tuple(net.shape) != (s, 3) or tuple(matrix.shape) != (s, f, 3)):
            raise ValueError("native full-matrix/contact buffer shape mismatch")
        if counts.dtype not in (t.int32, t.int64, t.uint32) or starts.dtype not in (t.int32, t.int64, t.uint32):
            raise ValueError("COUNTS/STARTS must be integer native buffers")
        values = (net, matrix, force, point, normal, separation)
        if (not all(x.is_floating_point() and x.dtype == matrix.dtype and x.device == matrix.device for x in values) or
                counts.device != matrix.device or starts.device != matrix.device):
            raise ValueError("native floating buffers must share dtype/device")
        counts, starts = counts.to(t.int64), starts.to(t.int64)
        flat_count, flat_start = counts.flatten(), starts.flatten()
        if bool((flat_count < 0).any()) or int(flat_count.sum().item()) >= capacity:
            raise ValueError("negative count or saturated native raw buffer")
        # The SDK owns unused entries; a zero-count START is never dereferenced
        # or interpreted as a raw point, even if its unused storage is a sentinel.
        if bool(((flat_count > 0) & ((flat_start < 0) | (flat_start >= capacity) |
                (flat_start+flat_count > capacity))).any()):
            raise ValueError("native STARTS/COUNTS exceed buffer")
        buckets = t.repeat_interleave(t.arange(s*f, device=matrix.device), flat_count)
        offsets = t.repeat_interleave(t.cumsum(flat_count, 0)-flat_count, flat_count)
        indices = t.repeat_interleave(flat_start, flat_count)+t.arange(buckets.numel(), device=matrix.device)-offsets
        if indices.unique().numel() != indices.numel():
            raise ValueError("native point index overlaps different routed pair buckets")
        force, separation = force.reshape(-1)[indices], separation.reshape(-1)[indices]
        point, normal = point[indices], normal[indices]
        if not all(bool(t.isfinite(x).all()) for x in (net, matrix, force, separation, point, normal)):
            raise ValueError("nonfinite active native contact data")
        if bool(((t.linalg.vector_norm(normal, dim=-1)-1).abs() > _NORMAL_TOL).any()):
            raise ValueError("active contact normal is not unit length")
        reconstructed = t.zeros(s*f, 3, dtype=matrix.dtype, device=matrix.device)
        reconstructed.index_add_(0, buckets, force[:, None]*normal)
        error = (reconstructed.reshape(s, f, 3)-matrix).abs()
        if not t.allclose(reconstructed.reshape(s, f, 3), matrix, atol=_ATOL_N, rtol=_RTOL):
            raise ValueError("signed raw Fn*normal does not reconstruct every pair matrix")
        # Serialize only active points. No inactive buffer bytes become evidence.
        ledger = [{"route_index": b, "native_buffer_index": i, "signed_normal_force_n": fn,
                   "point_world_m": p, "normal_world": n, "separation_m": sep}
                  for b, i, fn, p, n, sep in zip(buckets.cpu().tolist(), indices.cpu().tolist(),
                    force.cpu().tolist(), point.cpu().tolist(), normal.cpu().tolist(), separation.cpu().tolist())]
        residual = net-matrix.sum(1)
        return ledger, {"active_raw_points": len(ledger), "capacity_global": capacity,
            "counts_starts_sha256": _hash({"counts": counts.cpu().tolist(), "starts": starts.cpu().tolist()}),
            "active_raw_ledger_sha256": _hash(ledger),
            "raw_matrix_maximum_abs_error_n": float(error.amax().item()),
            "unfiltered_net_residual_world_n": residual.cpu().tolist(),
            "unfiltered_residual_not_a_collision_or_zero_gate": True}

    def sample_completed(self, native_step, episode_ids):
        """Observe exactly one completed native step, never advance physics."""
        available = [False]*self.n
        unavailable = [None]*self.n
        current_clean, collision = [False]*self.n, [False]*self.n
        maxima, depths = [0.]*self.n, [0.]*self.n
        worst, deepest = [None]*self.n, [None]*self.n
        sample_error, ledger, validation = None, [], None
        try:
            epochs = self._integers(episode_ids, self.n, "sample episode_ids")
            if type(native_step) is not int or native_step < 1:
                raise ValueError("sample must name a positive completed native step")
            if self._last_native is None or native_step != self._last_native+1:
                raise ValueError("missing/duplicate/out-of-order 2ms sample or no registered reset")
            ledger, validation = self._read()
            for e in range(self.n):
                available[e] = self._epoch[e] == epochs[e] and self._reset_native[e] is not None and native_step > self._reset_native[e]
                current_clean[e] = available[e]
                if not available[e]:
                    unavailable[e] = "unregistered/stale episode identity or reset-stale sample"
                    self._poison(e, native_step, unavailable[e])
            for item in ledger:
                route = self.routes[item["route_index"]]
                if not route["covered"]:
                    continue
                e, fn, sep = route["env_id"], abs(item["signed_normal_force_n"]), item["separation_m"]
                contact = {**item, "env_id": e, "episode_id": epochs[e], "native": native_step,
                    "sensor_body": route["sensor_body"], "partner_body": route["partner_body"],
                    "category": route["category"], "loaded_path_collision": fn > 0 and sep <= 0}
                depth = max(0., -sep)
                if fn > maxima[e]:
                    maxima[e], worst[e] = fn, contact
                if depth > depths[e]:
                    depths[e], deepest[e] = depth, contact
                if fn > 0 and sep <= 0:
                    collision[e], current_clean[e] = True, False
            for e in range(self.n):
                if available[e]:
                    self._samples[e] += 1
                    if collision[e]:
                        self._poison(e, native_step, "nonzero_signed_Fn_at_nonpositive_separation")
                    if maxima[e] > self._peak_force[e]:
                        self._peak_force[e], self._worst_force[e] = maxima[e], worst[e]
                    if depths[e] > self._peak_depth[e]:
                        self._peak_depth[e], self._worst_depth[e] = depths[e], deepest[e]
        except Exception as exc:
            sample_error = type(exc).__name__+": "+str(exc)
            available, current_clean = [False]*self.n, [False]*self.n
            unavailable = [sample_error]*self.n
            for e in range(self.n):
                self._poison(e, native_step if type(native_step) is int else None, sample_error)
        if type(native_step) is int and native_step >= 0:
            self._last_native = max(self._last_native if self._last_native is not None else native_step, native_step)
        points = []
        if sample_error is None:
            # Collision witnesses first; omitted valid points stay bound by the
            # complete active-ledger digest and explicit count, never implied zero.
            chosen = sorted(ledger, key=lambda x: (
                not (self.routes[x["route_index"]]["covered"] and abs(x["signed_normal_force_n"]) > 0 and x["separation_m"] <= 0),
                -abs(x["signed_normal_force_n"]), x["route_index"], x["native_buffer_index"]))[:self.max_evidence_points]
            points = [dict(x, env_id=self.routes[x["route_index"]]["env_id"],
                covered=self.routes[x["route_index"]]["covered"]) for x in chosen]
        result = {"protocol": PROTOCOL, "native_physics_step_count": (
                native_step if type(native_step) is int and native_step >= 0 else None),
            "measurement_dt_s": self.dt, "episode_ids": list(self._epoch),
            "source_binding_sha256": self.readback.get("source_binding_sha256"),
            "available": all(available), "per_env_available": available,
            "unavailable_reason": sample_error or ("some environment identities unavailable" if not all(available) else None),
            "per_env_unavailable_reason": unavailable, "per_env_current_clean": current_clean,
            "per_env_clean": [bool(ok and clean) for ok, clean in zip(available, self._clean)],
            "per_env_path_collision": collision, "per_env_max_abs_signed_point_force_n": maxima,
            "per_env_maximum_penetration_m": depths, "per_env_worst_force_contact": worst,
            "per_env_deepest_contact": deepest, "bounded_valid_points": points,
            "valid_points_omitted": max(0, len(ledger)-len(points)) if sample_error is None else None,
            "validation": validation, "first_failure": self._first_failure,
            "valid_completed_samples": self._samples, "physics_or_reporting_writes": 0,
            "collision_definition": "covered pair has abs(signed Fn)>0 and separation<=0; no force epsilon",
            "friction_or_support_certified": False}
        return copy.deepcopy(result)
