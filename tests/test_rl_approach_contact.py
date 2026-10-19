"""No App/physics creation: adversarial native-view fixtures for path evidence."""

import copy
import json
from types import SimpleNamespace
import unittest

from g1_vla.rl_approach_contact import NativeApproachContactWitness, approach_contact_routes
from g1_vla.rl_collision_model import bilateral_fixed_cluster_contract


# Official 55-joint fixture, independent of descendant/exclusion implementation.
def official_graph():
    result = [("root_joint", "PhysicsFixedJoint", None, "pelvis")]
    result += [(n+"_joint", "PhysicsFixedJoint", a, b) for n, a, b in (
        ("imu_in_pelvis", "pelvis", "imu_in_pelvis"),
        ("pelvis_contour", "pelvis", "pelvis_contour_link"),
        ("imu_in_torso", "torso_link", "imu_in_torso"),
        ("logo", "torso_link", "logo_link"), ("head", "torso_link", "head_link"),
        ("d435", "torso_link", "d435_link"), ("mid360", "torso_link", "mid360_link"))]
    for side in ("left", "right"):
        chain = ["pelvis"]+[side+"_"+n+"_link" for n in
            ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
        result += [(b.replace("_link", "_joint"), "PhysicsRevoluteJoint", a, b)
                   for a, b in zip(chain, chain[1:])]
        chain = ["torso_link"]+[side+"_"+n+"_link" for n in
            ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
        result += [(b.replace("_link", "_joint"), "PhysicsRevoluteJoint", a, b)
                   for a, b in zip(chain, chain[1:])]
        result += [(side+"_hand_"+name+"_joint", "PhysicsFixedJoint", side+"_wrist_yaw_link",
                    side+"_hand_"+name+"_link") for name in ("camera_base", "palm")]
        for finger, count in (("thumb", 3), ("index", 2), ("middle", 2)):
            chain = [side+"_hand_palm_link"]+[side+"_hand_"+finger+"_"+str(i)+"_link" for i in range(count)]
            result += [(b.replace("_link", "_joint"), "PhysicsRevoluteJoint", a, b)
                       for a, b in zip(chain, chain[1:])]
    chain = ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"]
    result += [(name, "PhysicsRevoluteJoint", a, b) for name, a, b in
               zip(("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"), chain, chain[1:])]
    return result


class Prim:
    def __init__(self, path, kind="Xform", apis=(), attrs=None, rels=None):
        self.path, self.kind, self.apis = path, kind, list(apis)
        self.attrs, self.rels = dict(attrs or {}), dict(rels or {})
    def GetPath(self): return self.path
    def GetName(self): return self.path.rsplit("/", 1)[-1]
    def GetTypeName(self): return self.kind
    def IsValid(self): return True
    def GetAppliedSchemas(self): return list(self.apis)
    def GetAttribute(self, name):
        return SimpleNamespace(IsValid=lambda: name in self.attrs, Get=lambda: self.attrs.get(name))
    def GetRelationship(self, name):
        return SimpleNamespace(IsValid=lambda: name in self.rels, GetTargets=lambda: list(self.rels.get(name, [])))


class Stage:
    def __init__(self, num_envs=2):
        self.prims = {}
        for e in range(num_envs):
            root = "/World/envs/env_"+str(e)+"/Robot"
            names = {b for _, _, a, b in official_graph() if b} | {a for _, _, a, b in official_graph() if a}
            for name in sorted(names):
                p = root+"/"+name
                self.prims[p] = Prim(p, apis=("PhysicsRigidBodyAPI", "PhysxContactReportAPI"),
                    attrs={"physxContactReport:threshold": 0.})
            p = "/World/envs/env_"+str(e)+"/Table"
            self.prims[p] = Prim(p, apis=("PhysicsRigidBodyAPI", "PhysxContactReportAPI"),
                                attrs={"physxContactReport:threshold": 0.})
            for name, kind, a, b in official_graph():
                p = root+("/" if name == "root_joint" else "/joints/")+name
                self.prims[p] = Prim(p, kind, attrs={"physics:jointEnabled": True,
                    "physics:collisionEnabled": False}, rels={"physics:body0": [root+"/"+a] if a else [],
                    "physics:body1": [root+"/"+b]})
            p = self.prims[root+"/root_joint"]
            p.apis.append("PhysicsArticulationRootAPI")
            p.attrs["physxArticulation:enabledSelfCollisions"] = True
            for a, b in bilateral_fixed_cluster_contract()["contract"]["excluded_body_pairs"]:
                p = self.prims[root+"/"+a]
                if "PhysicsFilteredPairsAPI" not in p.apis:
                    p.apis.append("PhysicsFilteredPairsAPI")
                p.rels.setdefault("physics:filteredPairs", []).append(root+"/"+b)
    def Traverse(self): return iter(self.prims.values())
    def GetPrimAtPath(self, path): return self.prims[path]


class RouteTests(unittest.TestCase):
    def test_actual_full_graph_covers_camera_legs_head_table_and_nonadjacent_fingers(self):
        value = approach_contact_routes(Stage(), 2)
        self.assertEqual(len(value["sensor_paths"]), 32)
        self.assertEqual(len(value["filter_paths"][0]), 56)
        self.assertEqual(len(value["routes"]), 32*56)
        e = value["per_env"][0]
        self.assertEqual(len(e["moving_body_names"]), 16)
        for other in ("right_hand_camera_base_link", "right_hand_palm_link", "right_wrist_yaw_link"):
            self.assertIn(other, e["moving_body_names"])
        covered = e["covered_ordered_pairs"]
        for other in ("Table", "head_link", "left_hand_index_1_link", "left_ankle_roll_link", "torso_link"):
            self.assertIn(["right_hand_camera_base_link", other], covered)
        self.assertIn(["right_hand_index_1_link", "right_hand_thumb_2_link"], covered)
        self.assertNotIn(["right_hand_thumb_0_link", "right_hand_camera_base_link"], covered)
        self.assertEqual(value["physics_or_reporting_writes"], 0)
        json.dumps(value, allow_nan=False)

    def test_missing_attachment_extra_body_crossenv_joint_and_extra_filter_fail_closed(self):
        mutations = [
            lambda s: s.prims.pop("/World/envs/env_0/Robot/right_hand_camera_base_link"),
            lambda s: s.prims.update({"/World/envs/env_0/Robot/hidden": Prim(
                "/World/envs/env_0/Robot/hidden", apis=("PhysicsRigidBodyAPI",))}),
            lambda s: s.prims["/World/envs/env_0/Robot/joints/right_hand_palm_joint"].rels.update(
                {"physics:body1": ["/World/envs/env_1/Robot/right_hand_palm_link"]}),
            lambda s: s.prims["/World/envs/env_0/Robot/right_elbow_link"].apis.append("PhysicsFilteredPairsAPI"),
            lambda s: s.prims["/World/envs/env_0/Robot/root_joint"].attrs.update(
                {"physxArticulation:enabledSelfCollisions": False}),
        ]
        for mutate in mutations:
            stage = Stage(); mutate(stage)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                approach_contact_routes(stage, 2)


class View:
    def __init__(self, t, sensors, filters, capacity):
        self.t, self.sensor_paths, self.filter_paths = t, list(sensors), copy.deepcopy(filters)
        self.sensor_count, self.filter_count, self.max_contact_data_count = len(sensors), len(filters[0]), capacity
        self.calls = 0
        self.clear()
    def clear(self):
        t, s, f, c = self.t, self.sensor_count, self.filter_count, self.max_contact_data_count
        self.net, self.matrix = t.zeros(s, 3), t.zeros(s, f, 3)
        self.force, self.sep, self.point, self.normal = t.zeros(c), t.zeros(c), t.zeros(c, 3), t.zeros(c, 3)
        self.counts, self.starts = t.zeros(s, f, dtype=t.int32), t.zeros(s, f, dtype=t.int32)
        self.used = 0
    def add(self, env, body, partner, *, force=1., separation=-.0001, normal=(1., 0., 0.)):
        sensor = self.sensor_paths.index("/World/envs/env_"+str(env)+"/Robot/"+body)
        prefix = "/World/envs/env_"+str(env)+("/" if partner == "Table" else "/Robot/")
        column = self.filter_paths[sensor].index(prefix+partner)
        index = self.used; self.used += 1
        if not self.counts[sensor, column]: self.starts[sensor, column] = index
        self.counts[sensor, column] += 1
        self.force[index], self.sep[index] = force, separation
        self.normal[index] = self.t.tensor(normal)
        self.point[index] = self.t.tensor([.2, .3, .8])
        self.matrix[sensor, column] += force*self.t.tensor(normal)
        self.net[sensor] += force*self.t.tensor(normal)
        return sensor, column, index
    def get_net_contact_forces(self, dt): return self.net
    def get_contact_force_matrix(self, dt): return self.matrix
    def get_contact_data(self, dt):
        self.calls += 1
        return self.force, self.point, self.normal, self.sep, self.counts, self.starts


class WitnessTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t, self.stage, self.views, self.create_calls = torch, Stage(), [], []
        def create(sensors, *, filter_patterns, max_contact_data_count):
            self.create_calls.append((copy.deepcopy(sensors), copy.deepcopy(filter_patterns), max_contact_data_count))
            view = View(torch, sensors, filter_patterns, max_contact_data_count)
            self.views.append(view)
            return view
        self.sim = SimpleNamespace(create_rigid_contact_view=create)
        self.w = NativeApproachContactWitness(torch, lambda x:x, self.sim, self.stage, 2)
        self.v = self.views[0]
        self.w.reset([0, 1], [3, 7], 0)
    def sample(self, step=1, episodes=(3, 7)):
        return self.w.sample_completed(step, list(episodes))
    def test_empty_exact_full_matrix_is_current_and_history_clean(self):
        row = self.sample()
        self.assertTrue(row["available"])
        self.assertEqual(row["per_env_clean"], [True, True])
        self.assertEqual(row["valid_completed_samples"], [1, 1])
        self.assertEqual(len(self.create_calls), 1)
        self.assertEqual(len(set(self.create_calls[0][0])), 32)
        self.assertEqual(row["physics_or_reporting_writes"], 0)
        json.dumps(row, allow_nan=False)

    def test_signed_nonzero_force_even_tiny_or_cancelling_cannot_hide_loaded_collision(self):
        for fn in (1., -1., 1e-20):
            with self.subTest(force=fn):
                self.v.clear(); self.v.add(0, "right_hand_camera_base_link", "Table", force=fn, separation=0.)
                # Fresh observer so history of earlier subtests is irrelevant.
                w = NativeApproachContactWitness(self.t, lambda x:x, self.sim, self.stage, 2)
                v = self.views[-1]; v.add(0, "right_hand_camera_base_link", "Table", force=fn, separation=0.)
                w.reset([0, 1], [3, 7], 0); row = w.sample_completed(1, [3, 7])
                self.assertTrue(row["available"]); self.assertEqual(row["per_env_clean"], [False, True])
        self.v.clear()
        self.v.add(0, "right_hand_index_1_link", "right_hand_thumb_2_link", force=1.)
        self.v.add(0, "right_hand_index_1_link", "right_hand_thumb_2_link", force=-1.)
        self.assertEqual(float(self.v.matrix.abs().sum()), 0.)
        row = self.sample(); self.assertEqual(row["per_env_path_collision"], [True, False])

    def test_camera_torso_leftarm_leg_and_nonadjacent_internal_pairs_are_covered(self):
        targets = ("torso_link", "left_elbow_link", "right_ankle_roll_link", "head_link", "right_elbow_link")
        for step, partner in enumerate(targets, 1):
            self.v.clear(); self.v.add(1, "right_hand_camera_base_link", partner)
            row = self.sample(step)
            self.assertTrue(row["available"])
            self.assertEqual(row["per_env_path_collision"], [False, True])
            self.assertEqual(row["per_env_worst_force_contact"][1]["partner_body"], partner)

    def test_only_verified_joint_or_exact_filter_pairs_are_excluded(self):
        self.v.add(0, "right_shoulder_pitch_link", "torso_link")
        self.v.add(0, "right_hand_thumb_0_link", "right_hand_camera_base_link")
        row = self.sample()
        self.assertEqual(row["per_env_clean"], [True, True])
        self.assertTrue(all(not p["covered"] for p in row["bounded_valid_points"]))

    def test_positive_gap_force_is_reported_but_not_loaded_path_contact(self):
        self.v.add(0, "right_wrist_yaw_link", "Table", separation=.0001, force=.3)
        row = self.sample()
        self.assertEqual(row["per_env_clean"], [True, True])
        self.assertFalse(row["per_env_worst_force_contact"][0]["loaded_path_collision"])
        self.assertAlmostEqual(row["per_env_max_abs_signed_point_force_n"][0], .3, places=6)

    def test_allowed_unfiltered_cube_force_does_not_need_zero_net_residual(self):
        self.v.net[0, 2] = .785
        row = self.sample()
        self.assertTrue(row["available"]); self.assertEqual(row["per_env_clean"], [True, True])
        self.assertAlmostEqual(row["validation"]["unfiltered_net_residual_world_n"][0][2], .785, places=6)

    def test_collision_history_survives_handoff_and_selective_true_reset(self):
        self.v.add(0, "right_hand_index_1_link", "Table")
        first = self.sample(); self.v.clear()
        row = self.sample(2)
        self.assertEqual(row["per_env_current_clean"], [True, True])
        self.assertEqual(row["per_env_clean"], [False, True])
        self.assertEqual(row["first_failure"][0]["native"], 1)
        with self.assertRaises(ValueError): self.w.reset([0], [3], 2)
        self.w.reset([0], [4], 2)
        row = self.sample(3, (4, 7))
        self.assertEqual(row["per_env_clean"], [True, True])
        self.assertEqual(row["valid_completed_samples"], [1, 3])
        self.assertIsNotNone(first["first_failure"][0])  # Detached prior evidence.

    def test_missing_duplicate_and_out_of_order_steps_poison_history(self):
        self.assertTrue(self.sample()["available"])
        duplicate = self.sample()
        self.assertFalse(duplicate["available"])
        self.assertEqual(duplicate["per_env_clean"], [False, False])
        self.assertEqual(self.sample(2)["per_env_clean"], [False, False])
        self.assertFalse(self.sample(4)["available"])
        self.assertFalse(self.sample(3)["available"])
        self.assertEqual(self.w.summary()["last_completed_native"], 4)

    def test_unregistered_epoch_is_unavailable_not_a_silent_reset(self):
        row = self.sample(1, (4, 7))
        self.assertEqual(row["per_env_available"], [False, True])
        self.assertEqual(row["per_env_clean"], [False, True])
        self.assertIsNotNone(row["per_env_unavailable_reason"][0])
        self.assertIsNone(row["per_env_unavailable_reason"][1])
        self.assertEqual(self.w.summary()["registered_episode_ids"], [3, 7])

    def test_missing_or_censored_reporter_never_creates_a_view_or_writes(self):
        for change in (lambda p:p.apis.remove("PhysxContactReportAPI"),
                       lambda p:p.attrs.update({"physxContactReport:threshold": 1.}),
                       lambda p:p.attrs.update({"physxContactReport:threshold": float("nan")})):
            stage = Stage(); change(stage.prims["/World/envs/env_1/Robot/head_link"])
            before = len(self.views)
            w = NativeApproachContactWitness(self.t, lambda x:x, self.sim, stage, 2)
            self.assertFalse(w.summary()["available"]); self.assertEqual(len(self.views), before)
            w.reset([0, 1], [3, 7], 0)
            self.assertEqual(w.sample_completed(1, [3, 7])["per_env_clean"], [False, False])

    def test_wrong_routing_wrong_env_and_capacity_are_unavailable(self):
        original = self.sim.create_rigid_contact_view
        for mutation in (lambda v:v.filter_paths[0].reverse(),
                         lambda v:v.filter_paths[0].__setitem__(0, "/World/envs/env_1/Table"),
                         lambda v:setattr(v, "max_contact_data_count", 1)):
            def create(*args, **kwargs):
                view = original(*args, **kwargs); mutation(view); return view
            sim = SimpleNamespace(create_rigid_contact_view=create)
            w = NativeApproachContactWitness(self.t, lambda x:x, sim, self.stage, 2)
            self.assertFalse(w.summary()["available"])
            self.assertIn("routing/capacity", w.summary()["unavailable_reason"])

    def test_nonfinite_active_buffers_and_nonunit_normals_fail_closed(self):
        for name in ("force", "sep", "point", "normal", "matrix", "net"):
            self.v.clear(); self.v.add(0, "right_hand_index_1_link", "Table")
            getattr(self.v, name).fill_(float("nan"))
            row = self.sample(self.w.summary()["last_completed_native"]+1)
            self.assertFalse(row["available"], name); self.assertEqual(row["bounded_valid_points"], [])
        self.v.clear(); self.v.add(0, "right_hand_index_1_link", "Table", normal=(2., 0., 0.))
        self.assertIn("unit", self.sample(7)["unavailable_reason"])

    def test_unused_nan_buffer_is_not_invented_active_evidence_and_old_rows_are_detached(self):
        self.v.add(0, "right_hand_index_1_link", "Table")
        self.v.point[100:] = float("nan")
        row = self.sample(); self.assertTrue(row["available"])
        self.v.force.fill_(99.)
        self.assertEqual(row["bounded_valid_points"][0]["signed_normal_force_n"], 1.)

    def test_bad_counts_starts_overlap_saturation_and_reconstruction_reject(self):
        def overflow(v):
            v.counts[0, 0] = 1
            v.starts[0, 0] = v.max_contact_data_count
        cases = (lambda v:setattr(v, "counts", v.counts.float()),
                 lambda v:v.counts.__setitem__((0, 0), -1),
                 overflow,
                 lambda v:v.counts.__setitem__((0, 0), v.max_contact_data_count),
                 lambda v:v.matrix.add_(1.))
        for i, mutate in enumerate(cases, 1):
            self.v.clear(); mutate(self.v)
            row = self.sample(i)
            self.assertFalse(row["available"]); self.assertEqual(row["per_env_clean"], [False, False])
        self.v.clear(); s, f, _ = self.v.add(0, "right_hand_index_1_link", "Table")
        self.v.counts[s, f+1] = 1; self.v.starts[s, f+1] = 0
        self.assertIn("overlap", self.sample(6)["unavailable_reason"])

    def test_unused_start_sentinel_is_never_dereferenced(self):
        self.v.starts.fill_(-1)
        row = self.sample()
        self.assertTrue(row["available"])
        self.assertEqual(row["bounded_valid_points"], [])

    def test_invalid_clock_output_is_still_json_safe_and_irreversibly_unavailable(self):
        row = self.w.sample_completed(float("nan"), [3, 7])
        self.assertFalse(row["available"])
        self.assertIsNone(row["native_physics_step_count"])
        json.dumps(row, allow_nan=False)
        self.assertEqual(self.sample()["per_env_clean"], [False, False])

    def test_valid_unsigned_indices_and_bounded_evidence_digest(self):
        w = NativeApproachContactWitness(self.t, lambda x:x, self.sim, self.stage, 2, max_evidence_points=1)
        v = self.views[-1]
        v.add(0, "right_hand_index_1_link", "Table", force=.2)
        v.add(0, "right_hand_thumb_2_link", "Table", force=.4)
        v.counts, v.starts = v.counts.to(self.t.uint32), v.starts.to(self.t.uint32)
        w.reset([0, 1], [3, 7], 0); row = w.sample_completed(1, [3, 7])
        self.assertTrue(row["available"])
        self.assertEqual(len(row["bounded_valid_points"]), 1)
        self.assertEqual(row["valid_points_omitted"], 1)
        self.assertEqual(row["validation"]["active_raw_points"], 2)
        self.assertEqual(len(row["validation"]["active_raw_ledger_sha256"]), 64)
        self.assertAlmostEqual(row["bounded_valid_points"][0]["signed_normal_force_n"], .4, places=6)


if __name__ == "__main__":
    unittest.main()
