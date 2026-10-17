"""Pinned bilateral mechanical-adjacency exclusions for a new model version.

This module has no simulator/config imports. USD operations use the stage and
UsdPhysics module supplied by the caller, after clone and before physics init.
It neither changes the source USD nor treats a filtered overlap as disjoint.
"""

from __future__ import annotations

import hashlib
import json


OFFICIAL_ASSET_SHA256 = "d23b9bbc8b59cf20fa1983f55874f855a76e038b9e22eae39542e62be5b4ba3d"
OFFICIAL_JOINT_GRAPH_SHA256 = "1bc9a5d33856280a489137634f3ddcbb84196ef231b6dfa902acbd271eac600c"
ORIGINAL_RIGHT_CONTRACT_SHA256 = "04dd990d50f4632012408b13747313a5504614e47a13d05d9540c31ef66fb770"
BILATERAL_CONTRACT_VERSION = "g1_bilateral_wrist_fixed_component_adjacent_pairs_v1"


class CollisionModelError(ValueError):
    """The native graph or exclusion declaration differs from its pin."""


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _component(side):
    return [side + "_hand_camera_base_link", side + "_hand_palm_link", side + "_wrist_yaw_link"]


def _neighbors(side):
    return [side + "_hand_index_0_link", side + "_hand_middle_0_link",
            side + "_hand_thumb_0_link", side + "_wrist_pitch_link"]


def _pairs(component, neighbors):
    return [list(pair) for pair in sorted({tuple(sorted((a, b)))
            for a in component for b in component + neighbors if a != b})]


def _verify_joint_topology(rows):
    if type(rows) not in (tuple, list) or len(rows) != 55:
        raise CollisionModelError("bilateral model requires the complete official 55-joint graph")
    normalized, roots, names = [], set(), set()
    for row in rows:
        if type(row) is not dict:
            raise CollisionModelError("joint topology rows must be objects")
        name, kind, path = row.get("name"), row.get("type"), row.get("path")
        if (type(name) is not str or not name or "/" in name or name in names or
                kind not in ("PhysicsFixedJoint", "PhysicsRevoluteJoint")):
            raise CollisionModelError("duplicate or unsupported official joint")
        endpoints = []
        for key in ("body0", "body1"):
            value = row.get(key)
            if type(value) is not list or len(value) > 1 or any(type(p) is not str for p in value):
                raise CollisionModelError("joint endpoints must be single absolute body paths")
            endpoints.extend(value)
        if not endpoints or not endpoints[0].startswith("/"):
            raise CollisionModelError("joint has no absolute body endpoint")
        root = endpoints[0].rsplit("/", 1)[0]
        if (not root or ".." in root.split("/") or type(path) is not str or
                not path.startswith(root + "/") or not path.endswith("/" + name) or
                ".." in path.split("/")):
            raise CollisionModelError("joint path does not belong to its body robot root")
        roots.add(root)
        names.add(name)
        for key, expected in (("physics:jointEnabled", True), ("physics:collisionEnabled", False)):
            value = row.get(key)
            if isinstance(value, dict):
                value = value.get("value")
            if type(value) is not bool or value is not expected:
                raise CollisionModelError("official joint enabled/collision flags changed: " + name)
        item = {"name": name, "type": kind}
        for key in ("body0", "body1"):
            if not row[key]:
                item[key] = None
                continue
            endpoint = row[key][0]
            if not endpoint.startswith(root + "/"):
                raise CollisionModelError("joint endpoint crosses robot/environment roots")
            leaf = endpoint[len(root) + 1:]
            if not leaf or "/" in leaf or leaf in (".", ".."):
                raise CollisionModelError("joint endpoint is not an exact top-level body")
            item[key] = leaf
        normalized.append(item)
    normalized.sort(key=lambda row: row["name"])
    if len(roots) != 1 or _hash(normalized) != OFFICIAL_JOINT_GRAPH_SHA256:
        raise CollisionModelError("actual official joint graph differs from its pinned topology")
    for side in ("left", "right"):
        reached = {side + "_wrist_yaw_link"}
        while True:
            expanded = reached | {other for row in normalized if row["type"] == "PhysicsFixedJoint"
                                  for own, other in ((row["body0"], row["body1"]), (row["body1"], row["body0"]))
                                  if own in reached and other is not None}
            if expanded == reached:
                break
            reached = expanded
        neighbors = {other for row in normalized if row["type"] == "PhysicsRevoluteJoint"
                     for own, other in ((row["body0"], row["body1"]), (row["body1"], row["body0"]))
                     if own in reached and other is not None and other not in reached}
        if sorted(reached) != _component(side) or sorted(neighbors) != _neighbors(side):
            raise CollisionModelError("bilateral fixed component or direct adjacency differs")


def bilateral_fixed_cluster_contract(joint_topology=None):
    """Return a new declaration, optionally verifying one full native graph.

    ``None`` is for protocol declaration only. Native author/readback always
    verify every clone's complete graph; prefix substitution is not evidence.
    """
    if joint_topology is not None:
        _verify_joint_topology(joint_topology)
    right = {"contract_version": "g1_right_wrist_fixed_component_adjacent_pairs_v1",
             "asset_sha256": OFFICIAL_ASSET_SHA256,
             "fixed_component": _component("right"),
             "revolute_adjacent_bodies": _neighbors("right"),
             "excluded_body_pairs": _pairs(_component("right"), _neighbors("right"))}
    if _hash(right) != ORIGINAL_RIGHT_CONTRACT_SHA256:
        raise CollisionModelError("new model must preserve the original right15 exactly")
    contract = {
        "contract_version": BILATERAL_CONTRACT_VERSION,
        "asset_sha256": OFFICIAL_ASSET_SHA256,
        "fixed_components": {side: _component(side) for side in ("left", "right")},
        "revolute_adjacent_bodies": {side: _neighbors(side) for side in ("left", "right")},
        "excluded_body_pairs": sorted(_pairs(_component("left"), _neighbors("left")) +
                                      right["excluded_body_pairs"]),
        "original_right_contract_sha256": ORIGINAL_RIGHT_CONTRACT_SHA256,
    }
    if len(contract["excluded_body_pairs"]) != 30:
        raise CollisionModelError("bilateral model requires exactly30 distinct pairs")
    return {"contract": contract, "source_graph_sha256": OFFICIAL_JOINT_GRAPH_SHA256,
            "sha256": _hash(contract)}


def _roots(num_envs):
    if type(num_envs) is not int or not 1 <= num_envs <= 512:
        raise CollisionModelError("bilateral filtering needs1..512 declared environments")
    return ["/World/envs/env_" + str(i) + "/Robot" for i in range(num_envs)]


def _expected_relationships(num_envs):
    relationships = {}
    for root in _roots(num_envs):
        for left, right in bilateral_fixed_cluster_contract()["contract"]["excluded_body_pairs"]:
            relationships.setdefault(root + "/" + left, []).append(root + "/" + right)
    return {path: sorted(targets) for path, targets in sorted(relationships.items())}


def _native_graphs_and_self_collision(stage, usd_physics, num_envs):
    roots = _roots(num_envs)
    topologies, self_collisions = {root: [] for root in roots}, {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/envs/"):
            continue
        root = next((root for root in roots if path == root or path.startswith(root + "/")), None)
        if prim.IsA(usd_physics.Joint):
            if root is None:
                raise CollisionModelError("unexpected extra environment joint/articulation")
            joint = usd_physics.Joint(prim)
            topologies[root].append({"name": prim.GetName(), "type": prim.GetTypeName(), "path": path,
                "body0": [str(p) for p in joint.GetBody0Rel().GetTargets()],
                "body1": [str(p) for p in joint.GetBody1Rel().GetTargets()],
                "physics:jointEnabled": joint.GetJointEnabledAttr().Get(),
                "physics:collisionEnabled": joint.GetCollisionEnabledAttr().Get()})
        if prim.HasAPI(usd_physics.ArticulationRootAPI):
            if root is None:
                raise CollisionModelError("unexpected extra environment articulation root")
            value = prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get()
            if type(value) is not bool or value is not True or root in self_collisions:
                raise CollisionModelError("one globally self-colliding articulation is required per environment")
            self_collisions[root] = {"prim_path": path, "enabled_self_collisions": True}
    if set(self_collisions) != set(roots):
        raise CollisionModelError("missing native self-collision readback for an environment")
    for rows in topologies.values():
        bilateral_fixed_cluster_contract(rows)
    return self_collisions


def readback_bilateral_fixed_cluster_filters(stage, usd_physics, num_envs):
    """Verify every graph, all exact relations, and global self-collision flags."""
    self_collisions = _native_graphs_and_self_collision(stage, usd_physics, num_envs)
    actual = {}
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/World/envs/") and prim.HasAPI(usd_physics.FilteredPairsAPI):
            actual[str(prim.GetPath())] = [str(p) for p in usd_physics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()]
    expected = _expected_relationships(num_envs)
    if set(actual) != set(expected):
        raise CollisionModelError("bilateral filtered-body relationship allowlist mismatch")
    for path, targets in actual.items():
        if len(set(targets)) != len(targets) or sorted(targets) != expected[path]:
            raise CollisionModelError("bilateral filter has missing, duplicate or unexpected targets: " + path)
        for body in [path] + targets:
            prim = stage.GetPrimAtPath(body)
            if not prim.IsValid() or not prim.HasAPI(usd_physics.RigidBodyAPI) or prim.IsInstanceProxy():
                raise CollisionModelError("filter endpoint is not an exact non-proxy rigid body: " + body)
    return {"available": True, "num_envs": num_envs, "excluded_pair_count_per_env": 30,
            "excluded_pair_count_total": 30 * num_envs, "relationships": expected,
            "relationships_sha256": _hash(expected),
            "global_self_collision_readback": self_collisions,
            "source_graph_verified_env_count": num_envs,
            "source_graph_sha256": OFFICIAL_JOINT_GRAPH_SHA256,
            "model_override_version": BILATERAL_CONTRACT_VERSION,
            "filter_contract": bilateral_fixed_cluster_contract(),
            "interpretation": "explicit bilateral mechanical-adjacency exclusions; not disjoint geometry or proven grasp"}


def author_bilateral_fixed_cluster_filters(stage, usd_physics, num_envs, asset_sha256):
    """Author only the new30pair model after clone, before native physics init.

    Preflight every graph, body and old relationship before authoring anything.
    The old right-only author must not run first on this model path. Repeated
    verification uses readback, not a second mutation through this function.
    """
    if asset_sha256 != OFFICIAL_ASSET_SHA256:
        raise CollisionModelError("bilateral model requires the pinned official asset")
    expected = _expected_relationships(num_envs)
    _native_graphs_and_self_collision(stage, usd_physics, num_envs)
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith("/World/envs/") and prim.HasAPI(usd_physics.FilteredPairsAPI):
            raise CollisionModelError("bilateral author refuses preexisting filtered-pair APIs")
    for path, targets in expected.items():
        for body in [path] + targets:
            prim = stage.GetPrimAtPath(body)
            if not prim.IsValid() or not prim.HasAPI(usd_physics.RigidBodyAPI) or prim.IsInstanceProxy():
                raise CollisionModelError("filter endpoint is not a writable exact rigid body: " + body)
    for path, targets in expected.items():
        api = usd_physics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path))
        if not api or not api.CreateFilteredPairsRel().SetTargets(targets):
            raise CollisionModelError("failed to author bilateral exclusion: " + path)
    result = readback_bilateral_fixed_cluster_filters(stage, usd_physics, num_envs)
    result.update(authoring_phase="after_clone_before_physics_initialization",
                  source_asset_bytes_unchanged=True)
    return result
