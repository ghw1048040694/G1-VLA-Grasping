"""Same-stage collision representation evidence; imports no simulator at load.

The cooking request can populate a cooking cache.  It does not step physics,
author USD, or prove that its returned representation is the shape already
attached to a live actor.  Raw callback coordinates and the independently
recorded transforms remain separate evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct


AUDIT_VERSION = "g1_same_stage_collision_representation_v1"
PAD_BODY_NAMES = ("right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link")
_ACTIVE_REQUESTS = set()
_MAX_HULLS = 1024
_MAX_VERTICES = 1_000_000
_MAX_INDICES = 6_000_000


def _finite_vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} requires {size} numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) for v in value):
        raise ValueError(f"{name} requires finite numbers")
    return [float(v) for v in value]


def pose_matrix_xyzw(position, quaternion):
    """Column-vector rigid matrix from a native body pose, without editing it."""
    p = _finite_vector(position, 3, "body position")
    q = _finite_vector(quaternion, 4, "body xyzw quaternion")
    norm = math.sqrt(sum(v*v for v in q))
    if abs(norm-1) > 1e-4:
        raise ValueError("body quaternion is not unit length")
    x, y, z, w = (v/norm for v in q)
    return [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), p[0]],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), p[1]],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), p[2]],
            [0., 0., 0., 1.]]


def _multiply(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _field(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def _index(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def serialize_convex_hulls(convexes):
    """Validate the exact SDK callback arrays; never reconstruct or simplify."""
    if convexes is None or len(convexes) == 0 or len(convexes) > _MAX_HULLS:
        raise ValueError("missing or excessive callback convex hulls")
    rows = []
    total_vertices = total_indices = 0
    for convex in convexes:
        vertices = [list(v) for v in _field(convex, "vertices")]
        total_vertices += len(vertices)
        if len(vertices) < 4 or total_vertices > _MAX_VERTICES:
            raise ValueError("invalid or excessive convex vertex count")
        vertices = [_finite_vector(v, 3, "callback vertex") for v in vertices]
        indices = [_index(v, "callback index") for v in _field(convex, "indices")]
        total_indices += len(indices)
        if not indices or total_indices > _MAX_INDICES or any(v < 0 or v >= len(vertices) for v in indices):
            raise ValueError("callback convex index is invalid or excessive")
        polygons = []
        for polygon in _field(convex, "polygons"):
            start = _index(_field(polygon, "index_base"), "polygon index_base")
            count = _index(_field(polygon, "num_vertices"), "polygon num_vertices")
            plane = _finite_vector(list(_field(polygon, "plane")), 4, "polygon plane")
            if start < 0 or count < 3 or start+count > len(indices) or sum(v*v for v in plane[:3]) < 1e-20:
                raise ValueError("callback polygon range or plane is invalid")
            polygons.append({"index_base": start, "num_vertices": count, "plane": plane})
        if len(polygons) < 4:
            raise ValueError("callback hull has fewer than four polygons")
        row = {"vertices": vertices, "indices": indices, "polygons": polygons}
        row["vertices_f32_sha256"] = hashlib.sha256(b"".join(
            struct.pack("<3f", *v) for v in vertices)).hexdigest()
        row["geometry_sha256"] = hashlib.sha256(_canonical_bytes(row)).hexdigest()
        rows.append(row)
    return rows


def request_collision_representation(cooking, *, stage_id, collision_prim_id, valid_result):
    """One synchronous SDK request, with no event-loop tick or fallback cook."""
    key = (stage_id, collision_prim_id)
    if key in _ACTIVE_REQUESTS:
        return {"passed": False, "error": "reentrant_collision_request", "callback_count": 0}, []
    _ACTIVE_REQUESTS.add(key)
    callbacks = []
    accepting = True

    def callback(result, convexes):
        if accepting:
            callbacks.append((result, convexes))

    evidence = {"stage_id": stage_id, "collision_prim_id": collision_prim_id,
                "run_asynchronously": False, "callback_count": 0, "passed": False}
    try:
        returned = cooking.request_convex_collision_representation(
            stage_id=stage_id, collision_prim_id=collision_prim_id,
            run_asynchronously=False, on_result=callback)
        accepting = False
        evidence["returned_type"] = type(returned).__name__
        evidence["callback_count"] = len(callbacks)
        if len(callbacks) != 1:
            raise ValueError("synchronous request did not produce exactly one callback before return")
        result, convexes = callbacks[0]
        evidence["callback_result"] = str(result)
        evidence["callback_result_matches_RESULT_VALID"] = result == valid_result
        if result != valid_result:
            raise ValueError("SDK callback result is not RESULT_VALID")
        hulls = serialize_convex_hulls(convexes)
        evidence.update(passed=True, hull_count=len(hulls),
                        vertex_count=sum(len(h["vertices"]) for h in hulls))
        return evidence, hulls
    except Exception as exc:
        evidence.update(error_type=type(exc).__name__, error=str(exc))
        return evidence, []
    finally:
        accepting = False
        _ACTIVE_REQUESTS.remove(key)


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("nonfinite USD property")
        return value
    if hasattr(value, "pathString"):
        return str(value)
    try:
        return [_json_value(v) for v in value]
    except TypeError:
        return str(value)


def _attribute(prim, name):
    attr = prim.GetAttribute(name)
    return {"valid": bool(attr), "authored": bool(attr and attr.HasAuthoredValueOpinion()),
            "value": _json_value(attr.Get()) if attr else None}


def _rigid_ancestor(prim, UsdPhysics):
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
        prim = prim.GetParent()
    return None


def _relative(cache, prim, ancestor):
    matrix, resets = cache.ComputeRelativeTransform(prim, ancestor)
    if resets:
        raise ValueError("resetXformStack prevents an unambiguous rigid-body-relative transform")
    # Gf matrices multiply row vectors. Store a transposed, explicitly labelled
    # column-vector matrix for multiplication with the native body pose.
    result = [[float(matrix[j][i]) for j in range(4)] for i in range(4)]
    if not all(math.isfinite(v) for row in result for v in row):
        raise ValueError("nonfinite collider/source relative transform")
    return result


def _collider_binding(stage, collider, body, native_world, Usd, UsdGeom, UsdPhysics, cache):
    if _rigid_ancestor(collider, UsdPhysics) != body:
        raise ValueError("collider's nearest RigidBodyAPI ancestor is not the supplied native body")
    schemas = list(collider.GetAppliedSchemas())
    merged = "PhysxMeshMergeCollisionAPI" in schemas
    collection_evidence = None
    if merged:
        collection = Usd.CollectionAPI.Get(collider, "collisionmeshes")
        if not collection:
            raise ValueError("MeshMerge collider has no collisionmeshes collection")
        paths = Usd.CollectionAPI.ComputeIncludedPaths(collection.ComputeMembershipQuery(), stage)
        meshes = [stage.GetPrimAtPath(path) for path in sorted(paths, key=str)
                  if stage.GetPrimAtPath(path).IsA(UsdGeom.Mesh)]
        collection_evidence = {"name": "collisionmeshes",
            "includes": [str(p) for p in collection.GetIncludesRel().GetTargets()],
            "excludes": [str(p) for p in collection.GetExcludesRel().GetTargets()],
            "expansion_rule": _json_value(collection.GetExpansionRuleAttr().Get()),
            "resolved_mesh_paths": [str(m.GetPath()) for m in meshes]}
    elif collider.IsA(UsdGeom.Mesh):
        meshes = [collider]
    else:
        raise ValueError("requested collider is neither MeshMerge nor a single Mesh")
    if not meshes:
        raise ValueError("collision geometry collection resolved to no Mesh")
    sources = []
    for mesh in meshes:
        if _rigid_ancestor(mesh, UsdPhysics) != body:
            raise ValueError("MeshMerge source is outside the supplied moving rigid body")
        points = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
        if points is None or len(points) == 0:
            raise ValueError("source mesh has no points")
        digest = hashlib.sha256()
        for point in points:
            digest.update(struct.pack("<3f", *_finite_vector(list(point), 3, "source point")))
        sources.append({"path": str(mesh.GetPath()), "point_count": len(points),
            "points_f32_sha256": digest.hexdigest(),
            "source_to_collider_column_matrix": _relative(cache, mesh, collider),
            "source_to_body_column_matrix": _relative(cache, mesh, body),
            "instance_proxy": bool(mesh.IsInstanceProxy())})
    relative = _relative(cache, collider, body)
    names = ("physics:collisionEnabled", "physics:approximation",
             "physxConvexHullCollision:hullVertexLimit", "physxConvexHullCollision:minThickness",
             "physxCollision:contactOffset", "physxCollision:restOffset")
    return {"collider_path": str(collider.GetPath()), "collider_type": collider.GetTypeName(),
        "collider_applied_schemas": schemas, "rigid_body_ancestor_path": str(body.GetPath()),
        "mesh_merge": merged, "mesh_merge_collection": collection_evidence, "source_meshes": sources,
        "composed_collider_attributes": {name: _attribute(collider, name) for name in names},
        "collider_to_body_column_matrix": relative,
        "body_world_from_native_pose_column_matrix": native_world,
        "collider_world_from_native_body_column_matrix": _multiply(native_world, relative),
        "transform_source": "native current body tensor pose times USD body-relative collider transform; USD world pose is not used"}


def _write_json(path, payload):
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode()+b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _stage_binding(stage, context, UsdUtils, UsdGeom):
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    context_id = int(context.get_stage_id())
    if stage != context.get_stage() or stage_id <= 0 or stage_id != context_id:
        raise ValueError("supplied stage is not the existing active USD context/cache stage")
    result = {"stage_id": stage_id, "active_context_stage_id": context_id,
        "root_layer_identifier": stage.GetRootLayer().identifier, "same_stage_verified": True,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage), "up_axis": str(UsdGeom.GetStageUpAxis(stage))}
    if result["meters_per_unit"] != 1.0:
        raise ValueError("native world-metre poses require this audit's metre-unit USD stage")
    return result


def _coverage(report, payload, bodies):
    expected = sorted(b["body_path"] for b in bodies)
    actual = sorted(b["body_path"] for b in report["bodies"])
    stage_id = report.get("stage_binding", {}).get("stage_id")
    bindings = []
    valid = (not report["errors"] and expected == actual and len(expected) == 3 and
             len(set(expected)) == 3 and report.get("stage_binding", {}).get("same_stage_verified") is True)
    successful_bodies = []
    for body in report["bodies"]:
        good = bool(body["colliders"]) and "error" not in body
        for collider in body["colliders"]:
            request = collider.get("request", {})
            good &= (request.get("passed") is True and collider.get("stage_id") == stage_id and
                     collider.get("collision_prim_id") == request.get("collision_prim_id") and
                     request.get("stage_id") == stage_id and
                     collider.get("rigid_body_ancestor_path") == body["body_path"])
            bindings.append((collider.get("binding_sha256"), collider.get("collider_path"),
                             body["body_path"], collider.get("stage_id"), collider.get("collision_prim_id")))
        valid &= good
        if good:
            successful_bodies.append(body["body_path"])
    exported = [(c.get("binding_sha256"), c.get("collider_path"), c.get("body_path"),
                 c.get("stage_id"), c.get("collision_prim_id")) for c in payload["colliders"]]
    exact = sorted(bindings) == sorted(exported) and len(set(bindings)) == len(bindings)
    nonempty = all(bool(c.get("hulls")) for c in payload["colliders"])
    return {"requested_body_count": len(expected), "successful_body_count": len(successful_bodies),
        "requested_body_paths": expected, "successful_body_paths": sorted(successful_bodies),
        "collider_count": len(bindings), "payload_collider_count": len(exported),
        "all_payload_bindings_exact": exact, "complete": bool(valid and exact and nonempty)}


def export_collision_geometry(stage, *, bodies, output_path):
    """Export the three named moving pads on the caller's existing active stage.

    ``bodies`` contains body_name/body_path/current_position_world_m/
    current_quaternion_xyzw from native tensors. No App, stage or cache entry
    is created. Failures are saved and returned; the caller owns admission.
    """
    path = Path(output_path)
    payload_path = path.with_name(path.stem+"_hulls.json")
    report = {"version": AUDIT_VERSION, "passed": False, "complete": False, "bodies": [], "errors": [],
        "request_may_cook_or_populate_cache": True, "physics_steps_requested": 0,
        "usd_properties_authored": 0, "attached_native_shape_identity_verified": False,
        "callback_vertex_coordinate_frame_certified": False,
        "representation_scope": "SDK requested same-stage collider representation; not an attached-shape/cache identity proof"}
    payload = {"version": AUDIT_VERSION, "coordinate_scope": "raw SDK callback coordinates; no implicit world/body frame claim", "colliders": []}
    try:
        if not isinstance(bodies, list) or not bodies or len(bodies) > 64:
            raise ValueError("bodies must be a nonempty bounded list")
        if len(bodies) != 3 or {b.get("body_name") for b in bodies} != set(PAD_BODY_NAMES):
            raise ValueError("this audit requires all three declared right distal pad bodies")
        paths = [b["body_path"] for b in bodies]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate native body path")
        import omni.usd
        from omni.physx import get_physx_cooking_interface
        from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
        from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

        context = omni.usd.get_context()
        report["stage_binding"] = _stage_binding(stage, context, UsdUtils, UsdGeom)
        stage_id = report["stage_binding"]["stage_id"]
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        cooking = get_physx_cooking_interface()
        for entry in bodies:
            pose_source = {name: entry.get(name) for name in (
                "env_id", "native_link_index", "native_articulation_prim_path")}
            body_record = {"body_name": entry.get("body_name"), "body_path": entry.get("body_path"),
                           **pose_source, "colliders": []}
            report["bodies"].append(body_record)
            try:
                body = stage.GetPrimAtPath(entry["body_path"])
                if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError("native body path is not an active RigidBodyAPI prim")
                if body.GetName() != entry["body_name"]:
                    raise ValueError("native body name/path mismatch")
                native_world = pose_matrix_xyzw(entry["current_position_world_m"], entry["current_quaternion_xyzw"])
                body_record["native_pose_xyzw"] = list(entry["current_position_world_m"])+list(entry["current_quaternion_xyzw"])
                colliders = sorted((p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                                    if p.HasAPI(UsdPhysics.CollisionAPI)), key=lambda p: str(p.GetPath()))
                if len(colliders) > 64:
                    raise ValueError("excessive collider count for a requested native body")
                body_record["collider_candidate_paths"] = [str(p.GetPath()) for p in colliders]
                if not colliders:
                    raise ValueError("native body has no CollisionAPI descendant")
                for collider in colliders:
                    binding = _collider_binding(stage, collider, body, native_world, Usd, UsdGeom, UsdPhysics, cache)
                    prim_id = int(PhysicsSchemaTools.sdfPathToInt(str(collider.GetPath())))
                    binding["stage_id"] = stage_id
                    binding["collision_prim_id"] = prim_id
                    binding["native_body_pose_source"] = pose_source
                    binding["native_pose_source_verification_scope"] = "native tensor metadata mapping supplied and verified by environment adapter; module independently checks USD body ancestry"
                    binding["binding_sha256"] = hashlib.sha256(_canonical_bytes(binding)).hexdigest()
                    request, hulls = request_collision_representation(cooking, stage_id=stage_id,
                        collision_prim_id=prim_id, valid_result=PhysxCollisionRepresentationResult.RESULT_VALID)
                    binding["request"] = request
                    body_record["colliders"].append(binding)
                    if not request["passed"]:
                        raise ValueError("collision representation request failed: "+request.get("error", "unknown"))
                    payload["colliders"].append({"binding_sha256": binding["binding_sha256"],
                        "body_path": entry["body_path"], "collider_path": str(collider.GetPath()),
                        "stage_id": stage_id, "collision_prim_id": prim_id, "hulls": hulls})
            except Exception as exc:
                body_record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                report["errors"].append({"body_path": entry.get("body_path"), **body_record["error"]})
    except Exception as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    try:
        report["coverage"] = _coverage(report, payload, bodies)
        report["passed"] = report["complete"] = report["coverage"]["complete"]
    except Exception as exc:
        report["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    try:
        report["geometry_payload"] = _write_json(payload_path, payload)
        written = _write_json(path, report)
    except Exception as exc:
        return {"version": AUDIT_VERSION, "passed": False, "complete": False, "error_type": type(exc).__name__,
                "error": str(exc), "report_path": str(path), "attached_native_shape_identity_verified": False}
    return {"version": AUDIT_VERSION, "passed": report["passed"], "complete": report["complete"], "report": written,
        "geometry_payload": report["geometry_payload"], "body_count": len(report["bodies"]),
        "collider_count": sum(len(b["colliders"]) for b in report["bodies"]),
        "coverage": report.get("coverage"),
        "errors": report["errors"][:64], "attached_native_shape_identity_verified": False,
        "callback_vertex_coordinate_frame_certified": False, "request_may_cook_or_populate_cache": True}
