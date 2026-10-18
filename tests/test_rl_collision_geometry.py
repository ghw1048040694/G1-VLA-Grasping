"""Pure evidence validation; these tests never create an App or physics scene."""

import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
from types import ModuleType, SimpleNamespace

from g1_vla import rl_collision_geometry as geometry


def tetrahedron():
    return {"vertices": [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            "indices": [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
            "polygons": [{"index_base": i*3, "num_vertices": 3, "plane": p}
                         for i, p in enumerate([[0., 0., -1., 0.], [0., -1., 0., 0.],
                                                [-1., 0., 0., 0.], [1., 1., 1., -1.]])]}


class CollisionGeometryPureTests(unittest.TestCase):
    def test_native_xyzw_pose_matrix_uses_column_vectors(self):
        s = math.sqrt(.5)
        m = geometry.pose_matrix_xyzw([2., 3., 4.], [0., 0., s, s])
        transformed = [sum(m[i][j]*[1., 0., 0., 1.][j] for j in range(4)) for i in range(4)]
        for actual, expected in zip(transformed, [2., 4., 4., 1.]):
            self.assertAlmostEqual(actual, expected, places=12)

    def test_pose_requires_finite_unit_quaternion_and_native_dimensions(self):
        for p, q in [([0., 0.], [0., 0., 0., 1.]),
                     ([float("nan"), 0., 0.], [0., 0., 0., 1.]),
                     ([0., 0., 0.], [0., 0., 0., 2.]),
                     ([False, 0., 0.], [0., 0., 0., 1.])]:
            with self.subTest(position=p, quaternion=q), self.assertRaises(ValueError):
                geometry.pose_matrix_xyzw(p, q)

    def test_same_stage_identity_requires_context_and_existing_cache_id(self):
        stage = SimpleNamespace(GetRootLayer=lambda: SimpleNamespace(identifier="liveStage"))
        context = SimpleNamespace(get_stage=lambda: stage, get_stage_id=lambda: 19)
        usdutils = SimpleNamespace(StageCache=SimpleNamespace(Get=lambda: SimpleNamespace(
            GetId=lambda s: SimpleNamespace(ToLongInt=lambda: 19))))
        usdgeom = SimpleNamespace(GetStageMetersPerUnit=lambda s: 1., GetStageUpAxis=lambda s: "Z")
        result = geometry._stage_binding(stage, context, usdutils, usdgeom)
        self.assertEqual(result["stage_id"], 19)
        self.assertTrue(result["same_stage_verified"])
        for bad in (SimpleNamespace(get_stage=lambda: object(), get_stage_id=lambda: 19),
                    SimpleNamespace(get_stage=lambda: stage, get_stage_id=lambda: 20)):
            with self.assertRaises(ValueError):
                geometry._stage_binding(stage, bad, usdutils, usdgeom)
        usdgeom.GetStageMetersPerUnit = lambda s: .01
        with self.assertRaises(ValueError):
            geometry._stage_binding(stage, context, usdutils, usdgeom)

    def test_gf_row_transform_is_transposed_before_native_body_world_composition(self):
        rows = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [1., 0., 0., 1.]]
        cache = SimpleNamespace(ComputeRelativeTransform=lambda a, b: (rows, False))
        relative = geometry._relative(cache, object(), object())
        s = math.sqrt(.5)
        body = geometry.pose_matrix_xyzw([2., 3., 4.], [0., 0., s, s])
        world = geometry._multiply(body, relative)
        for actual, expected in zip([world[i][3] for i in range(3)], [2., 4., 4.]):
            self.assertAlmostEqual(actual, expected, places=12)
        cache.ComputeRelativeTransform = lambda a, b: (rows, True)
        with self.assertRaises(ValueError):
            geometry._relative(cache, object(), object())

    def test_callback_arrays_preserved_with_stable_hash_not_reconstructed(self):
        original = tetrahedron()
        rows = geometry.serialize_convex_hulls([original])
        self.assertEqual(rows[0]["vertices"], original["vertices"])
        self.assertEqual(rows[0]["indices"], original["indices"])
        self.assertEqual(rows[0]["polygons"], original["polygons"])
        self.assertEqual(rows, geometry.serialize_convex_hulls([copy.deepcopy(original)]))
        self.assertEqual(len(rows[0]["vertices_f32_sha256"]), 64)
        original["vertices"][0][0] = 100.
        self.assertEqual(rows[0]["vertices"][0][0], 0.)

    def test_malformed_hull_never_becomes_valid_evidence(self):
        bad = []
        a = tetrahedron(); a["vertices"][0][0] = float("inf"); bad.append(a)
        a = tetrahedron(); a["indices"][0] = 4; bad.append(a)
        a = tetrahedron(); a["indices"][0] = True; bad.append(a)
        a = tetrahedron(); a["polygons"][0]["index_base"] = 100; bad.append(a)
        a = tetrahedron(); a["polygons"][0]["plane"] = [0., 0., 0., 1.]; bad.append(a)
        for hull in bad:
            with self.subTest(hull=hull), self.assertRaises(ValueError):
                geometry.serialize_convex_hulls([hull])
        with self.assertRaises(ValueError):
            geometry.serialize_convex_hulls([])

    def test_sync_request_uses_exact_stage_and_collider_and_copies_callback(self):
        class Cooking:
            def request_convex_collision_representation(self, **kwargs):
                self.request = kwargs
                kwargs["on_result"]("VALID", [tetrahedron()])
        cooking = Cooking()
        report, hulls = geometry.request_collision_representation(
            cooking, stage_id=19, collision_prim_id=42, valid_result="VALID")
        self.assertTrue(report["passed"])
        self.assertEqual(report["callback_count"], 1)
        self.assertEqual(cooking.request["stage_id"], 19)
        self.assertEqual(cooking.request["collision_prim_id"], 42)
        self.assertIs(cooking.request["run_asynchronously"], False)
        self.assertEqual(len(hulls), 1)

    def test_missing_invalid_or_repeated_callbacks_fail_closed(self):
        for results in ([], ["FAILED"], ["VALID", "VALID"]):
            class Cooking:
                def request_convex_collision_representation(self, **kwargs):
                    for result in results:
                        kwargs["on_result"](result, [tetrahedron()])
            with self.subTest(results=results):
                report, hulls = geometry.request_collision_representation(
                    Cooking(), stage_id=19, collision_prim_id=42, valid_result="VALID")
                self.assertFalse(report["passed"])
                self.assertEqual(hulls, [])
                self.assertEqual(report["callback_count"], len(results))

    def test_reentrant_same_stage_collider_is_rejected_and_guard_is_released(self):
        class Cooking:
            def request_convex_collision_representation(self, **kwargs):
                inner, _ = geometry.request_collision_representation(
                    self, stage_id=19, collision_prim_id=42, valid_result="VALID")
                self.inner = inner
                kwargs["on_result"]("VALID", [tetrahedron()])
        cooking = Cooking()
        report, _ = geometry.request_collision_representation(
            cooking, stage_id=19, collision_prim_id=42, valid_result="VALID")
        self.assertTrue(report["passed"])
        self.assertEqual(cooking.inner["error"], "reentrant_collision_request")
        self.assertFalse(geometry._ACTIVE_REQUESTS)

    def test_sdk_exception_and_bad_geometry_leave_no_cached_fake_hull(self):
        class Cooking:
            def request_convex_collision_representation(self, **kwargs):
                raise RuntimeError("native request rejected")
        report, hulls = geometry.request_collision_representation(
            Cooking(), stage_id=19, collision_prim_id=42, valid_result="VALID")
        self.assertFalse(report["passed"])
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertFalse(hulls)
        self.assertFalse(geometry._ACTIVE_REQUESTS)

    def test_complete_requires_all_three_bodies_and_exact_stage_collider_payload_binding(self):
        bodies = [{"body_name": name, "body_path": "/World/Robot/"+name} for name in geometry.PAD_BODY_NAMES]
        report = {"errors": [], "stage_binding": {"stage_id": 19, "same_stage_verified": True}, "bodies": []}
        payload = {"colliders": []}
        for i, body in enumerate(bodies):
            collider = {"binding_sha256": str(i)*64, "collider_path": body["body_path"]+"/collisions",
                        "rigid_body_ancestor_path": body["body_path"], "stage_id": 19, "collision_prim_id": i+1,
                        "request": {"passed": True, "stage_id": 19, "collision_prim_id": i+1}}
            report["bodies"].append({**body, "colliders": [collider]})
            payload["colliders"].append({**collider, "body_path": body["body_path"], "hulls": [tetrahedron()]})
        self.assertTrue(geometry._coverage(report, payload, bodies)["complete"])
        for fault in ["missing_body", "wrong_payload_id", "failed_request", "wrong_rigid_ancestor"]:
            bad_report, bad_payload = copy.deepcopy(report), copy.deepcopy(payload)
            if fault == "missing_body":
                bad_report["bodies"].pop()
            elif fault == "wrong_payload_id":
                bad_payload["colliders"][0]["collision_prim_id"] = 100
            elif fault == "failed_request":
                bad_report["bodies"][0]["colliders"][0]["request"]["passed"] = False
            else:
                bad_report["bodies"][0]["colliders"][0]["rigid_body_ancestor_path"] = "/WrongBody"
            with self.subTest(fault=fault):
                self.assertFalse(geometry._coverage(bad_report, bad_payload, bodies)["complete"])

    def test_missing_backend_is_saved_as_failure_without_claiming_attached_shape(self):
        with TemporaryDirectory() as directory, mock.patch.dict("sys.modules", {"omni.usd": None}):
            path = Path(directory)/"audit.json"
            result = geometry.export_collision_geometry(None, bodies=[{
                "body_name": name, "body_path": "/World/"+name,
                "current_position_world_m": [0., 0., 0.],
                "current_quaternion_xyzw": [0., 0., 0., 1.]} for name in geometry.PAD_BODY_NAMES], output_path=path)
            self.assertFalse(result["passed"])
            self.assertFalse(result["complete"])
            saved = json.loads(path.read_text())
            self.assertTrue(saved["errors"])
            self.assertFalse(saved["attached_native_shape_identity_verified"])
            self.assertFalse(saved["callback_vertex_coordinate_frame_certified"])
            self.assertEqual(saved["physics_steps_requested"], 0)
            payload = json.loads((Path(directory)/"audit_hulls.json").read_text())
            self.assertEqual(payload["colliders"], [])


class BoxSceneQuery:
    """Synchronous test-only API oracle, not a simulation or production fallback."""
    def __init__(self, world, body_path, *, foreign=False, mutate=None):
        self.world, self.body_path = world, body_path
        self.foreign, self.mutate = foreign, mutate
        self.calls = []

    def raycast_all(self, origin, direction, distance, callback, both_sides):
        self.calls.append((origin, direction, distance, both_sides))
        local_origin = geometry._world_to_body(self.world, list(origin), point=True)
        local_direction = geometry._world_to_body(self.world, list(direction), point=False)
        axis = max(range(3), key=lambda i: abs(local_direction[i]))
        side = -1. if local_direction[axis] > 0. else 1.
        t = (side*.01-local_origin[axis])/local_direction[axis]
        p = [origin[i]+t*direction[i] for i in range(3)]
        local_normal = [0., 0., 0.]; local_normal[axis] = side
        normal = geometry._rotate_vector(self.world, local_normal)
        hit = SimpleNamespace(rigid_body=self.body_path, collision=self.body_path+"/collisions",
            position=p, normal=normal, distance=t, face_index=axis, material="/Material", protoIndex=0xFFFFFFFF)
        if self.foreign:
            other = copy.copy(hit); other.rigid_body = "/Other"; other.collision = "/Other/collisions"
            callback(other)
        if self.mutate:
            self.mutate(hit)
        callback(hit)
        return True


class CollisionSurfacePureTests(unittest.TestCase):
    def setUp(self):
        s = math.sqrt(.5)
        self.world = geometry.pose_matrix_xyzw([.2, -.3, .8], [0., 0., s, s])
        self.body = "/World/Robot/right_hand_thumb_2_link"
        self.rays = geometry.make_surface_rays([-.01]*3, [.01]*3, self.world)

    def query(self, backend):
        return geometry.query_collision_surface(backend, body_path=self.body,
            collider_paths=[self.body+"/collisions"], native_world=self.world, rays=self.rays)

    def test_six_direction_rays_are_bounded_and_transform_with_native_pose(self):
        self.assertEqual(len(self.rays), 150)
        self.assertEqual({side: sum(r["side"] == side for r in self.rays)
                          for side in {r["side"] for r in self.rays}},
                         {axis+sign: 25 for axis in "xyz" for sign in "-+"})
        for ray in self.rays:
            self.assertAlmostEqual(sum(v*v for v in ray["direction_world"]), 1.)
            self.assertAlmostEqual(ray["distance_m"], .03)
            recovered = geometry._world_to_body(self.world, ray["origin_world_m"], point=True)
            for actual, expected in zip(recovered, ray["origin_body_m"]):
                self.assertAlmostEqual(actual, expected)
        for low, high in [([0.]*3, [0.]*3), ([0.]*3, [2.]*3), ([0., 0., float("nan")], [1.]*3)]:
            with self.assertRaises(ValueError):
                geometry.make_surface_rays(low, high, self.world)

    def test_live_hits_use_sdk_fields_filter_both_paths_and_recover_body_coordinates(self):
        backend = BoxSceneQuery(self.world, self.body, foreign=True)
        result = self.query(backend)
        self.assertTrue(result["coverage"]["surface_complete"])
        self.assertEqual(result["coverage"]["bound_hit_count"], 150)
        self.assertEqual(result["coverage"]["foreign_hit_count"], 150)
        self.assertEqual(result["errors"], [])
        for row in result["rays"]:
            hit = row["hits"][0]
            self.assertEqual(hit["rigid_body_path"], self.body)
            self.assertEqual(hit["collider_path"], self.body+"/collisions")
            self.assertAlmostEqual(max(abs(v) for v in hit["position_body_m"]), .01)
            self.assertAlmostEqual(sum(v*v for v in hit["normal_body"]), 1.)
        self.assertTrue(all(call[3] is False for call in backend.calls))

    def test_wrong_rigid_body_or_wrong_collider_cannot_complete(self):
        for field, value in [("rigid_body", "/WrongBody"), ("collision", self.body+"/visualMesh")]:
            result = self.query(BoxSceneQuery(self.world, self.body,
                mutate=lambda hit, f=field, v=value: setattr(hit, f, v)))
            self.assertFalse(result["coverage"]["surface_complete"])
            self.assertEqual(result["coverage"]["bound_hit_count"], 0)
            self.assertEqual(result["coverage"]["foreign_hit_count"], 150)

    def test_nonfinite_off_ray_or_nonunit_native_hits_fail_closed(self):
        mutations = [lambda h: setattr(h, "distance", float("nan")),
                     lambda h: setattr(h, "normal", [0., 0., 0.]),
                     lambda h: setattr(h, "position", [h.position[0]+.001, h.position[1], h.position[2]])]
        for mutation in mutations:
            result = self.query(BoxSceneQuery(self.world, self.body, mutate=mutation))
            self.assertFalse(result["coverage"]["surface_complete"])
            self.assertTrue(result["errors"])

    def test_no_hits_and_coplanar_or_duplicate_points_are_incomplete(self):
        class Empty:
            def raycast_all(self, *args):
                return False
        result = self.query(Empty())
        self.assertFalse(result["coverage"]["surface_complete"])
        self.assertFalse(result["errors"])
        all_sides = [a+s for a in "xyz" for s in "-+"]
        for points in ([[0., 0., 0.]]*30, [[i*.0001, j*.0001, 0.] for i in range(5) for j in range(5)]):
            self.assertFalse(geometry.surface_point_coverage(points, all_sides)["surface_complete"])
        box_points = [[x, y, z] for x in [-.01, 0., .01] for y in [-.01, 0., .01] for z in [-.01, .01]]
        self.assertTrue(geometry.surface_point_coverage(box_points, all_sides)["surface_complete"])
        self.assertFalse(geometry.surface_point_coverage(box_points, all_sides[:-1])["surface_complete"])

    def test_callback_after_return_is_not_allowed_to_attach_to_next_ray(self):
        class Late(BoxSceneQuery):
            previous = None
            def raycast_all(self, origin, direction, distance, callback, both_sides):
                if self.previous:
                    self.previous(SimpleNamespace())
                self.previous = callback
                return super().raycast_all(origin, direction, distance, callback, both_sides)
        result = self.query(Late(self.world, self.body))
        self.assertFalse(result["coverage"]["surface_complete"])
        self.assertEqual(result["errors"][0]["ray_index"], 0)
        self.assertEqual(result["errors"][0]["error"], "callback_after_synchronous_return")

    def test_surface_completeness_requires_three_exact_payload_bindings(self):
        bodies = [{"body_path": "/"+name} for name in geometry.PAD_BODY_NAMES]
        report = {"errors": [], "stage_binding": {"same_stage_verified": True},
            "bodies": [{**body, "binding_sha256": str(i), "coverage": {"surface_complete": True}}
                       for i, body in enumerate(bodies)]}
        payload = {"bodies": [{**body, "binding_sha256": str(i)} for i, body in enumerate(bodies)]}
        self.assertTrue(geometry._surface_coverage(report, payload, bodies)["surface_complete"])
        payload["bodies"][0]["binding_sha256"] = "Wrong"
        self.assertFalse(geometry._surface_coverage(report, payload, bodies)["surface_complete"])
        payload["bodies"].pop()
        self.assertFalse(geometry._surface_coverage(report, payload, bodies)["surface_complete"])

    def test_scene_query_support_must_be_explicit_and_never_fakes_full_hulls(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"audit.json"
            result = geometry.export_collision_surface_audit(None, bodies=[], output_path=path)
            self.assertFalse(result["complete"])
            self.assertFalse(result["surface_complete"])
            self.assertFalse(result["full_convex_hull_complete"])
            saved = json.loads(path.read_text())
            self.assertIn("explicit enabled", saved["errors"][0]["message"])
            self.assertEqual(saved["physics_steps_requested"], 0)
            self.assertEqual(saved["app_updates_requested"], 0)
            self.assertFalse(saved["request_may_cook_or_populate_cache"])
            self.assertTrue((Path(directory)/"audit_surface_samples.json").is_file())

    def test_export_preserves_three_native_metadata_bindings_without_calling_cook_or_step(self):
        bodies = [{"body_name": name, "body_path": "/World/Robot/"+name,
            "current_position_world_m": [0., 0., 0.], "current_quaternion_xyzw": [0., 0., 0., 1.],
            "env_id": 0, "native_link_index": i+20, "native_articulation_prim_path": "/World/Robot"}
            for i, name in enumerate(geometry.PAD_BODY_NAMES)]
        body_prims, collider_prims = {}, {}
        for entry in bodies:
            body_prims[entry["body_path"]] = SimpleNamespace(HasAPI=lambda api: True, GetName=lambda n=entry["body_name"]: n)
            collider_prims[entry["body_path"]] = SimpleNamespace(HasAPI=lambda api: True, GetPath=lambda p=entry["body_path"]+"/collisions": p)
        class AllBodies:
            def raycast_all(self, origin, direction, distance, callback, both_sides):
                for entry in bodies:
                    BoxSceneQuery(geometry.pose_matrix_xyzw([0., 0., 0.], [0., 0., 0., 1.]), entry["body_path"]).raycast_all(
                        origin, direction, distance, callback, both_sides)
                return True
        omni = ModuleType("omni"); omni.__path__ = []
        omni.usd = ModuleType("omni.usd"); omni.usd.get_context = lambda: None
        omni.timeline = ModuleType("omni.timeline")
        omni.timeline.get_timeline_interface = lambda: SimpleNamespace(is_stopped=lambda: False)
        physx = ModuleType("omni.physx"); physx.get_physx_scene_query_interface = lambda: AllBodies()
        carb = ModuleType("carb"); carb.Float3 = lambda *v: list(v)
        pxr = ModuleType("pxr")
        pxr.Usd = SimpleNamespace(PrimRange=lambda body, _: [collider_prims[next(path for path, b in body_prims.items() if b is body)]],
            TraverseInstanceProxies=lambda: None, TimeCode=SimpleNamespace(Default=lambda: None))
        pxr.UsdGeom = SimpleNamespace(XformCache=lambda _: None)
        pxr.UsdPhysics = SimpleNamespace(RigidBodyAPI="rigid", CollisionAPI="collision")
        pxr.UsdUtils = None
        modules = {"omni": omni, "omni.usd": omni.usd, "omni.timeline": omni.timeline,
                   "omni.physx": physx, "carb": carb, "pxr": pxr}
        def binding(stage, collider, body, *args):
            path = collider.GetPath(); ancestor = path.rsplit("/", 1)[0]
            return {"collider_path": path, "rigid_body_ancestor_path": ancestor, "source_meshes": []}
        with TemporaryDirectory() as directory, mock.patch.dict("sys.modules", modules), \
             mock.patch.object(geometry, "_stage_binding", return_value={"stage_id": 19, "same_stage_verified": True}), \
             mock.patch.object(geometry, "_collider_binding", side_effect=binding), \
             mock.patch.object(geometry, "_source_body_bounds", return_value={"minimum_body_m": [-.01]*3, "maximum_body_m": [.01]*3}):
            path = Path(directory)/"audit.json"
            result = geometry.export_collision_surface_audit(SimpleNamespace(GetPrimAtPath=lambda p: body_prims[p]),
                bodies=bodies, output_path=path, scene_query_support={"enabled": True, "source": "mock config evidence"})
            self.assertTrue(result["complete"], result["errors"])
            self.assertEqual(result["route"], "live_surface_queries_v2")
            self.assertFalse(result["full_convex_hull_complete"])
            self.assertFalse(result["attached_native_shape_identity_verified"])
            saved = json.loads(path.read_text())
            self.assertEqual([b["native_link_index"] for b in saved["bodies"]], [20, 21, 22])
            self.assertEqual([b["env_id"] for b in saved["bodies"]], [0]*3)
            payload = json.loads((path.parent/"audit_surface_samples.json").read_text())
            self.assertEqual(len(payload["bodies"]), 3)
            self.assertTrue(all(len(b["rays"]) == 150 for b in payload["bodies"]))


if __name__ == "__main__":
    unittest.main()
