"""Pure evidence validation; these tests never create an App or physics scene."""

import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
