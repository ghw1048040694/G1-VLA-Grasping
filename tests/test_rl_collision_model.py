"""Read-only fake-USD checks of pinned bilateral mechanical adjacency."""
from copy import deepcopy
import unittest
from g1_vla.rl_collision_model import (
    OFFICIAL_ASSET_SHA256, CollisionModelError, bilateral_fixed_cluster_contract,
    author_bilateral_fixed_cluster_filters, readback_bilateral_fixed_cluster_filters,
)

FIXTURE = [('d435_joint', 'PhysicsFixedJoint', 'torso_link', 'd435_link'), ('head_joint', 'PhysicsFixedJoint', 'torso_link', 'head_link'), ('imu_in_pelvis_joint', 'PhysicsFixedJoint', 'pelvis', 'imu_in_pelvis'), ('imu_in_torso_joint', 'PhysicsFixedJoint', 'torso_link', 'imu_in_torso'), ('left_ankle_pitch_joint', 'PhysicsRevoluteJoint', 'left_knee_link', 'left_ankle_pitch_link'), ('left_ankle_roll_joint', 'PhysicsRevoluteJoint', 'left_ankle_pitch_link', 'left_ankle_roll_link'), ('left_elbow_joint', 'PhysicsRevoluteJoint', 'left_shoulder_yaw_link', 'left_elbow_link'), ('left_hand_camera_base_joint', 'PhysicsFixedJoint', 'left_wrist_yaw_link', 'left_hand_camera_base_link'), ('left_hand_index_0_joint', 'PhysicsRevoluteJoint', 'left_hand_palm_link', 'left_hand_index_0_link'), ('left_hand_index_1_joint', 'PhysicsRevoluteJoint', 'left_hand_index_0_link', 'left_hand_index_1_link'), ('left_hand_middle_0_joint', 'PhysicsRevoluteJoint', 'left_hand_palm_link', 'left_hand_middle_0_link'), ('left_hand_middle_1_joint', 'PhysicsRevoluteJoint', 'left_hand_middle_0_link', 'left_hand_middle_1_link'), ('left_hand_palm_joint', 'PhysicsFixedJoint', 'left_wrist_yaw_link', 'left_hand_palm_link'), ('left_hand_thumb_0_joint', 'PhysicsRevoluteJoint', 'left_hand_palm_link', 'left_hand_thumb_0_link'), ('left_hand_thumb_1_joint', 'PhysicsRevoluteJoint', 'left_hand_thumb_0_link', 'left_hand_thumb_1_link'), ('left_hand_thumb_2_joint', 'PhysicsRevoluteJoint', 'left_hand_thumb_1_link', 'left_hand_thumb_2_link'), ('left_hip_pitch_joint', 'PhysicsRevoluteJoint', 'pelvis', 'left_hip_pitch_link'), ('left_hip_roll_joint', 'PhysicsRevoluteJoint', 'left_hip_pitch_link', 'left_hip_roll_link'), ('left_hip_yaw_joint', 'PhysicsRevoluteJoint', 'left_hip_roll_link', 'left_hip_yaw_link'), ('left_knee_joint', 'PhysicsRevoluteJoint', 'left_hip_yaw_link', 'left_knee_link'), ('left_shoulder_pitch_joint', 'PhysicsRevoluteJoint', 'torso_link', 'left_shoulder_pitch_link'), ('left_shoulder_roll_joint', 'PhysicsRevoluteJoint', 'left_shoulder_pitch_link', 'left_shoulder_roll_link'), ('left_shoulder_yaw_joint', 'PhysicsRevoluteJoint', 'left_shoulder_roll_link', 'left_shoulder_yaw_link'), ('left_wrist_pitch_joint', 'PhysicsRevoluteJoint', 'left_wrist_roll_link', 'left_wrist_pitch_link'), ('left_wrist_roll_joint', 'PhysicsRevoluteJoint', 'left_elbow_link', 'left_wrist_roll_link'), ('left_wrist_yaw_joint', 'PhysicsRevoluteJoint', 'left_wrist_pitch_link', 'left_wrist_yaw_link'), ('logo_joint', 'PhysicsFixedJoint', 'torso_link', 'logo_link'), ('mid360_joint', 'PhysicsFixedJoint', 'torso_link', 'mid360_link'), ('pelvis_contour_joint', 'PhysicsFixedJoint', 'pelvis', 'pelvis_contour_link'), ('right_ankle_pitch_joint', 'PhysicsRevoluteJoint', 'right_knee_link', 'right_ankle_pitch_link'), ('right_ankle_roll_joint', 'PhysicsRevoluteJoint', 'right_ankle_pitch_link', 'right_ankle_roll_link'), ('right_elbow_joint', 'PhysicsRevoluteJoint', 'right_shoulder_yaw_link', 'right_elbow_link'), ('right_hand_camera_base_joint', 'PhysicsFixedJoint', 'right_wrist_yaw_link', 'right_hand_camera_base_link'), ('right_hand_index_0_joint', 'PhysicsRevoluteJoint', 'right_hand_palm_link', 'right_hand_index_0_link'), ('right_hand_index_1_joint', 'PhysicsRevoluteJoint', 'right_hand_index_0_link', 'right_hand_index_1_link'), ('right_hand_middle_0_joint', 'PhysicsRevoluteJoint', 'right_hand_palm_link', 'right_hand_middle_0_link'), ('right_hand_middle_1_joint', 'PhysicsRevoluteJoint', 'right_hand_middle_0_link', 'right_hand_middle_1_link'), ('right_hand_palm_joint', 'PhysicsFixedJoint', 'right_wrist_yaw_link', 'right_hand_palm_link'), ('right_hand_thumb_0_joint', 'PhysicsRevoluteJoint', 'right_hand_palm_link', 'right_hand_thumb_0_link'), ('right_hand_thumb_1_joint', 'PhysicsRevoluteJoint', 'right_hand_thumb_0_link', 'right_hand_thumb_1_link'), ('right_hand_thumb_2_joint', 'PhysicsRevoluteJoint', 'right_hand_thumb_1_link', 'right_hand_thumb_2_link'), ('right_hip_pitch_joint', 'PhysicsRevoluteJoint', 'pelvis', 'right_hip_pitch_link'), ('right_hip_roll_joint', 'PhysicsRevoluteJoint', 'right_hip_pitch_link', 'right_hip_roll_link'), ('right_hip_yaw_joint', 'PhysicsRevoluteJoint', 'right_hip_roll_link', 'right_hip_yaw_link'), ('right_knee_joint', 'PhysicsRevoluteJoint', 'right_hip_yaw_link', 'right_knee_link'), ('right_shoulder_pitch_joint', 'PhysicsRevoluteJoint', 'torso_link', 'right_shoulder_pitch_link'), ('right_shoulder_roll_joint', 'PhysicsRevoluteJoint', 'right_shoulder_pitch_link', 'right_shoulder_roll_link'), ('right_shoulder_yaw_joint', 'PhysicsRevoluteJoint', 'right_shoulder_roll_link', 'right_shoulder_yaw_link'), ('right_wrist_pitch_joint', 'PhysicsRevoluteJoint', 'right_wrist_roll_link', 'right_wrist_pitch_link'), ('right_wrist_roll_joint', 'PhysicsRevoluteJoint', 'right_elbow_link', 'right_wrist_roll_link'), ('right_wrist_yaw_joint', 'PhysicsRevoluteJoint', 'right_wrist_pitch_link', 'right_wrist_yaw_link'), ('waist_pitch_joint', 'PhysicsRevoluteJoint', 'waist_roll_link', 'torso_link'), ('waist_roll_joint', 'PhysicsRevoluteJoint', 'waist_yaw_link', 'waist_roll_link'), ('waist_yaw_joint', 'PhysicsRevoluteJoint', 'pelvis', 'waist_yaw_link'), ('root_joint', 'PhysicsFixedJoint', None, 'pelvis')]

class Attribute:
    def __init__(self, value): self.value = value
    def Get(self): return self.value

class Relationship:
    def __init__(self, prim, key): self.prim, self.key = prim, key
    def GetTargets(self): return list(self.prim.rels.get(self.key, []))
    def SetTargets(self, targets):
        if self.prim.fail_set: return False
        self.prim.rels[self.key] = list(targets)
        return True

class Prim:
    def __init__(self, path, kind="Xform", *, valid=True):
        self.path, self.kind, self.valid = path, kind, valid
        self.apis, self.attrs, self.rels = set(), {}, {}
        self.proxy = self.fail_set = False
    def GetPath(self): return self.path
    def GetName(self): return self.path.rsplit("/", 1)[-1]
    def GetTypeName(self): return self.kind
    def HasAPI(self, api): return api in self.apis
    def IsA(self, api): return api is Joint and self.kind in ("PhysicsFixedJoint", "PhysicsRevoluteJoint")
    def GetAttribute(self, name): return Attribute(self.attrs.get(name))
    def IsValid(self): return self.valid
    def IsInstanceProxy(self): return self.proxy

class Joint:
    def __init__(self, prim): self.prim = prim
    def GetBody0Rel(self): return Relationship(self.prim, "body0")
    def GetBody1Rel(self): return Relationship(self.prim, "body1")
    def GetJointEnabledAttr(self): return Attribute(self.prim.attrs["jointEnabled"])
    def GetCollisionEnabledAttr(self): return Attribute(self.prim.attrs["collisionEnabled"])

class RigidBodyAPI: pass
class ArticulationRootAPI: pass
class FilteredPairsAPI:
    def __init__(self, prim): self.prim = prim
    @classmethod
    def Apply(cls, prim):
        prim.apis.add(cls)
        return cls(prim)
    def GetFilteredPairsRel(self): return Relationship(self.prim, "filteredPairs")
    def CreateFilteredPairsRel(self): return Relationship(self.prim, "filteredPairs")

class USD:
    Joint, RigidBodyAPI, ArticulationRootAPI, FilteredPairsAPI = Joint, RigidBodyAPI, ArticulationRootAPI, FilteredPairsAPI

class Stage:
    def __init__(self, count=2):
        self.prims = {}
        for i in range(count):
            root = "/World/envs/env_"+str(i)+"/Robot"
            for name, kind, b0, b1 in FIXTURE:
                path = root+("/" if name == "root_joint" else "/joints/")+name
                joint = Prim(path, kind)
                joint.rels.update(body0=[root+"/"+b0] if b0 else [], body1=[root+"/"+b1] if b1 else [])
                joint.attrs.update(jointEnabled=True, collisionEnabled=False)
                if name == "root_joint":
                    joint.apis.add(ArticulationRootAPI)
                    joint.attrs["physxArticulation:enabledSelfCollisions"] = True
                self.prims[path] = joint
                for body in (b0, b1):
                    if body:
                        body_prim = self.prims.setdefault(root+"/"+body, Prim(root+"/"+body))
                        body_prim.apis.add(RigidBodyAPI)
        for i in range(count):
            self.prims["/World/envs/env_"+str(i)+"/Cube"] = Prim("/World/envs/env_"+str(i)+"/Cube")
    def Traverse(self): return list(self.prims.values())
    def GetPrimAtPath(self, path): return self.prims.get(path, Prim(path, valid=False))

class BilateralCollisionModelTests(unittest.TestCase):
    def setUp(self): self.stage = Stage(2)
    def author(self): return author_bilateral_fixed_cluster_filters(self.stage, USD, 2, OFFICIAL_ASSET_SHA256)
    def filters(self): return [x for x in self.stage.prims.values() if x.HasAPI(FilteredPairsAPI)]
    def test_declared_30_pairs_are_unique_and_only_mechanical_adjacency(self):
        d = bilateral_fixed_cluster_contract()
        p = d["contract"]["excluded_body_pairs"]
        self.assertEqual(len(p), 30)
        self.assertEqual(len(set(map(tuple, p))), 30)
        self.assertEqual(sum(x[0].startswith("right_") for x in p), 15)
        self.assertFalse(any("Cube" in x or "Table" in x for pair in p for x in pair))
        self.assertFalse(any("thumb_2" in x or "index_1" in x or "middle_1" in x for pair in p for x in pair))
    def test_right15_exactly_matches_unchanged_old_contract(self):
        from g1_vla.rl_environment import fixed_palm_adjacency_filter_contract
        old = fixed_palm_adjacency_filter_contract()
        new = bilateral_fixed_cluster_contract()
        self.assertEqual([p for p in new["contract"]["excluded_body_pairs"] if p[0].startswith("right_")], old["contract"]["excluded_body_pairs"])
        self.assertEqual(new["contract"]["original_right_contract_sha256"], old["sha256"])
    def test_declaration_is_fresh(self):
        d = bilateral_fixed_cluster_contract()
        d["contract"]["excluded_body_pairs"].clear()
        self.assertEqual(len(bilateral_fixed_cluster_contract()["contract"]["excluded_body_pairs"]), 30)
    def test_two_native_graphs_author60_and_readback(self):
        result = self.author()
        self.assertTrue(result["available"])
        self.assertEqual(result["excluded_pair_count_total"], 60)
        self.assertEqual(result["source_graph_verified_env_count"], 2)
        self.assertEqual(len(result["global_self_collision_readback"]), 2)
        again = readback_bilateral_fixed_cluster_filters(self.stage, USD, 2)
        self.assertEqual(result["relationships_sha256"], again["relationships_sha256"])
    def test_eight_native_graphs_have_exact240_pairs(self):
        stage = Stage(8)
        result = author_bilateral_fixed_cluster_filters(stage, USD, 8, OFFICIAL_ASSET_SHA256)
        self.assertEqual(result["excluded_pair_count_total"], 240)
        self.assertEqual(len(result["global_self_collision_readback"]), 8)
    def test_wrong_asset_fails_before_authoring(self):
        with self.assertRaises(CollisionModelError): author_bilateral_fixed_cluster_filters(self.stage, USD, 2, "bad")
        self.assertFalse(self.filters())
    def test_each_clone_graph_is_verified_before_any_authoring(self):
        self.stage.prims.pop("/World/envs/env_1/Robot/joints/left_hand_thumb_2_joint")
        with self.assertRaises(CollisionModelError): self.author()
        self.assertFalse(self.filters())
    def test_endpoint_graph_change_is_rejected(self):
        p = self.stage.GetPrimAtPath("/World/envs/env_0/Robot/joints/left_hand_thumb_0_joint")
        p.rels["body0"] = ["/World/envs/env_0/Robot/left_hand_thumb_2_link"]
        with self.assertRaises(CollisionModelError): self.author()
        self.assertFalse(self.filters())
    def test_cross_environment_joint_is_rejected(self):
        p = self.stage.GetPrimAtPath("/World/envs/env_0/Robot/joints/left_hand_thumb_0_joint")
        p.rels["body0"] = ["/World/envs/env_1/Robot/left_hand_palm_link"]
        with self.assertRaises(CollisionModelError): self.author()
    def test_joint_flags_must_be_exact_booleans_and_unchanged(self):
        for key, value in (("jointEnabled", False), ("jointEnabled", 1), ("collisionEnabled", True)):
            stage = Stage(2)
            stage.GetPrimAtPath("/World/envs/env_1/Robot/joints/left_hand_thumb_0_joint").attrs[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(CollisionModelError):
                author_bilateral_fixed_cluster_filters(stage, USD, 2, OFFICIAL_ASSET_SHA256)
    def test_global_self_collision_must_remain_true(self):
        self.stage.GetPrimAtPath("/World/envs/env_0/Robot/root_joint").attrs["physxArticulation:enabledSelfCollisions"] = False
        with self.assertRaises(CollisionModelError): self.author()
        self.assertFalse(self.filters())
    def test_extra_environment_cannot_be_silently_unfiltered(self):
        with self.assertRaises(CollisionModelError): author_bilateral_fixed_cluster_filters(Stage(3), USD, 2, OFFICIAL_ASSET_SHA256)
    def test_refuses_old_or_unknown_preexisting_filters(self):
        p = self.stage.GetPrimAtPath("/World/envs/env_0/Robot/right_hand_palm_link")
        FilteredPairsAPI.Apply(p)
        with self.assertRaises(CollisionModelError): self.author()
    def test_writable_rigid_endpoints_preflight(self):
        for proxy in (True, False):
            stage = Stage(2)
            p = stage.GetPrimAtPath("/World/envs/env_1/Robot/left_hand_thumb_0_link")
            if proxy: p.proxy = True
            else: p.apis.remove(RigidBodyAPI)
            with self.subTest(proxy=proxy), self.assertRaises(CollisionModelError):
                author_bilateral_fixed_cluster_filters(stage, USD, 2, OFFICIAL_ASSET_SHA256)
            self.assertFalse(any(x.HasAPI(FilteredPairsAPI) for x in stage.prims.values()))
    def test_missing_duplicate_crossenv_and_distal_targets_rejected(self):
        for case in ("missing", "duplicate", "crossenv", "distal"):
            stage = Stage(2)
            author_bilateral_fixed_cluster_filters(stage, USD, 2, OFFICIAL_ASSET_SHA256)
            p = stage.GetPrimAtPath("/World/envs/env_0/Robot/left_hand_camera_base_link")
            targets = p.rels["filteredPairs"]
            if case == "missing": targets.pop()
            elif case == "duplicate": targets.append(targets[0])
            elif case == "crossenv": targets[0] = targets[0].replace("env_0", "env_1")
            else: targets.append("/World/envs/env_0/Robot/left_hand_thumb_2_link")
            with self.subTest(case=case), self.assertRaises(CollisionModelError): readback_bilateral_fixed_cluster_filters(stage, USD, 2)
    def test_cube_reverse_filter_cannot_hide(self):
        self.author()
        p = self.stage.GetPrimAtPath("/World/envs/env_0/Cube")
        FilteredPairsAPI.Apply(p).CreateFilteredPairsRel().SetTargets(["/World/envs/env_0/Robot/left_hand_camera_base_link"])
        with self.assertRaises(CollisionModelError): readback_bilateral_fixed_cluster_filters(self.stage, USD, 2)
    def test_readback_rechecks_full_graph_not_only_relations(self):
        self.author()
        self.stage.GetPrimAtPath("/World/envs/env_1/Robot/joints/left_hand_thumb_0_joint").attrs["jointEnabled"] = False
        with self.assertRaises(CollisionModelError): readback_bilateral_fixed_cluster_filters(self.stage, USD, 2)
    def test_failed_write_is_not_reported_available(self):
        self.stage.GetPrimAtPath("/World/envs/env_0/Robot/left_hand_camera_base_link").fail_set = True
        with self.assertRaises(CollisionModelError): self.author()
    def test_invalid_environment_counts_rejected(self):
        for count in (0, True, 2.0, 513):
            with self.subTest(count=count), self.assertRaises(CollisionModelError):
                author_bilateral_fixed_cluster_filters(self.stage, USD, count, OFFICIAL_ASSET_SHA256)

if __name__ == "__main__": unittest.main()
