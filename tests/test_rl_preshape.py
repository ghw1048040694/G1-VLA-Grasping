"""Pure preshape contract, provenance, and immutable acceptance-frame checks."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

from g1_vla.rl_preshape import (
    CONTROL_PRESHAPE_VERSION,
    control_preshape_contract,
    resolve_preshape_frames,
    validate_control_preshape,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT / "resources/rl_geometry_candidates/opposed_face_effort_pinch_v2.json"
CANDIDATE_SHA = "2daa38dafba243eae42cd1450d9a0030f7d0503e6073647309109f3cbc885a60"
FRAMES_SHA = "3e107979a990e65a1996481bc237621db1787ecaf5541dbf0dceff577eafcc19"
STATIC_SHA = "97f7afb5c1fed93bbd5a582530a576fde6c0072dd4daec54c671f9062df2b10a"
RESOLVER_KEYS = {
    "original_frames", "execution_frames", "original_domain_lower7_rad", "original_domain_upper7_rad",
    "execution_reference_lower7_rad", "execution_reference_upper7_rad",
}


def candidate():
    return json.loads(CANDIDATE_PATH.read_text())


def original_frames(value):
    return value["geometry_evidence"]["source_detail"]["stage_keyframes"]


class ControlPreshapeContractTests(unittest.TestCase):
    def test_factory_pins_exact_source_frames_static_evidence_and_semantics(self):
        selected = control_preshape_contract()
        self.assertEqual(selected["version"], CONTROL_PRESHAPE_VERSION)
        self.assertEqual(selected["acceptance_candidate_sha256"], CANDIDATE_SHA)
        self.assertEqual(selected["acceptance_frames_sha256"], FRAMES_SHA)
        self.assertEqual(selected["static_geometry_evidence_sha256"], STATIC_SHA)
        self.assertEqual(selected["palm_offset_world_m"], [.0005, 0, 0])
        self.assertEqual(selected["first_stage_duration_s"], .9)
        self.assertEqual(selected["original_actual_geometry_tolerance_rad"], .025)
        self.assertEqual(selected["palm_offset_schedule"], "smoothstep_existing_first_stage_then_hold_v1")
        self.assertEqual(selected["static_geometry_qualification"],
                         "sampled_authored_geometry_only_not_native_support_or_dynamic_admission")
        self.assertIs(selected["training_allowed"], False)
        self.assertIs(selected["all_environments_same_preshape"], True)
        self.assertIs(selected["original_reset_and_episode_budget_unchanged"], True)
        self.assertIs(selected["palm_offset_freezes_on_first_fresh_physical_contact"], True)
        self.assertEqual(selected["contact_latch_cadence_s"], .002)
        self.assertEqual(selected["palm_reference_freeze_boundary"],
                         "next_macro_after_first_same_episode_native_loaded_point_v1")
        self.assertIs(selected["force_mode_hand_position_springs_remain_zero"], True)

    def test_factory_and_validation_return_independent_deep_copies(self):
        selected = control_preshape_contract()
        validated = validate_control_preshape(selected)
        self.assertEqual(validated, selected)
        validated["execution_frame_hand_offsets_rad"][2][1] = -1
        self.assertEqual(selected["execution_frame_hand_offsets_rad"][2][1], .01)
        selected["palm_offset_world_m"][0] = 9
        fresh = control_preshape_contract()
        self.assertEqual(fresh["palm_offset_world_m"][0], .0005)
        fresh["execution_frame_hand_offsets_rad"][2][3] = 9
        self.assertNotEqual(fresh["execution_frame_hand_offsets_rad"][3][3], 9)

    def test_none_is_optional_without_replacement_declaration(self):
        self.assertIsNone(validate_control_preshape(None))

    def test_exact_schema_and_pins_reject_undeclared_changes(self):
        for key in control_preshape_contract():
            changed = control_preshape_contract()
            del changed[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_control_preshape(changed)
        for changed in ([], False, "v1", {}, {**control_preshape_contract(), "extra": 0}):
            with self.subTest(value=changed), self.assertRaises(ValueError):
                validate_control_preshape(changed)
        for key in ("version", "acceptance_candidate_sha256", "acceptance_frames_sha256",
                    "static_geometry_evidence_sha256", "static_geometry_qualification", "palm_offset_schedule",
                    "palm_reference_freeze_boundary"):
            changed = control_preshape_contract()
            changed[key] += "_changed"
            with self.subTest(changed=key), self.assertRaises(ValueError):
                validate_control_preshape(changed)

    def test_finite_dimensions_booleans_and_exact_geometry_are_enforced(self):
        invalid_vectors = ([.0005, 0], [.0005, 0, 0, 0], (.0005, 0, 0),
                           [.0005, False, 0], [.0005, float("nan"), 0], [.0005, float("inf"), 0],
                           [.0006, 0, 0], [.0005, 10 ** 400, 0])
        for vector in invalid_vectors:
            changed = control_preshape_contract()
            changed["palm_offset_world_m"] = vector
            with self.subTest(palm=vector), self.assertRaises(ValueError):
                validate_control_preshape(changed)
        for frames in ([], [[0] * 7] * 5, [[0] * 7] * 7, [[0] * 6] * 6):
            changed = control_preshape_contract()
            changed["execution_frame_hand_offsets_rad"] = frames
            with self.subTest(offsets=frames), self.assertRaises(ValueError):
                validate_control_preshape(changed)
        for value in (True, float("nan"), float("inf"), -.9, 0, 1.0, 10 ** 400):
            changed = control_preshape_contract()
            changed["first_stage_duration_s"] = value
            with self.subTest(duration=value), self.assertRaises(ValueError):
                validate_control_preshape(changed)
        for value in (False, float("nan"), float("inf"), -.025, 0, .03):
            changed = control_preshape_contract()
            changed["original_actual_geometry_tolerance_rad"] = value
            with self.subTest(tolerance=value), self.assertRaises(ValueError):
                validate_control_preshape(changed)
        for value in (True, float("nan"), float("inf"), -.002, 0, .066, 10 ** 400):
            changed = control_preshape_contract()
            changed["contact_latch_cadence_s"] = value
            with self.subTest(latch_cadence=value), self.assertRaises(ValueError):
                validate_control_preshape(changed)
        for value in (True, float("nan"), float("inf"), .010000000000001):
            changed = control_preshape_contract()
            changed["execution_frame_hand_offsets_rad"][1][1] = value
            with self.subTest(offset=value), self.assertRaises(ValueError):
                validate_control_preshape(changed)
        for key in ("all_environments_same_preshape", "original_reset_and_episode_budget_unchanged", "training_allowed",
                    "palm_offset_freezes_on_first_fresh_physical_contact", "force_mode_hand_position_springs_remain_zero"):
            for value in (0, 1, None, "true"):
                changed = control_preshape_contract()
                changed[key] = value
                with self.subTest(boolean=key, value=value), self.assertRaises(ValueError):
                    validate_control_preshape(changed)

    def test_pure_module_imports_no_torch_simulator_or_file_runtime(self):
        source = (ROOT / "src/g1_vla/rl_preshape.py").read_text()
        tree = ast.parse(source)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module)
        self.assertEqual(imports, {"__future__", "copy", "hashlib", "json", "math"})


class ResolvePreshapeFramesTests(unittest.TestCase):
    def test_resource_file_and_full_original_frames_are_pinned(self):
        self.assertEqual(hashlib.sha256(CANDIDATE_PATH.read_bytes()).hexdigest(), CANDIDATE_SHA)
        frames = original_frames(candidate())
        canonical = json.dumps(frames, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), FRAMES_SHA)

    def test_none_keeps_original_frames_and_original_envelope_without_metadata(self):
        value = candidate()
        result = resolve_preshape_frames(None, value, "unused_in_old_none_path")
        frames = original_frames(value)
        self.assertEqual(set(result), RESOLVER_KEYS)
        self.assertEqual(result["original_frames"], frames)
        self.assertEqual(result["execution_frames"], frames)
        values = [frame["hand_target_rad"] for frame in frames]
        for joint in range(7):
            minimum = min(row[joint] for row in values)
            maximum = max(row[joint] for row in values)
            self.assertEqual(result["original_domain_lower7_rad"][joint], minimum - .025)
            self.assertEqual(result["original_domain_upper7_rad"][joint], maximum + .025)
            self.assertEqual(result["execution_reference_lower7_rad"][joint], minimum)
            self.assertEqual(result["execution_reference_upper7_rad"][joint], maximum)

    def test_execution_changes_only_declared_hand_values_retains_source_and_durations(self):
        value = candidate()
        selected = control_preshape_contract()
        result = resolve_preshape_frames(selected, value, CANDIDATE_SHA)
        frames = original_frames(value)
        self.assertEqual(set(result), RESOLVER_KEYS)
        self.assertEqual(result["original_frames"], frames)
        self.assertEqual(result["execution_frames"][0], frames[0])
        self.assertEqual([frame["duration_s"] for frame in result["execution_frames"][1:]], [.9, .8, .6, .4, .6])
        for index, frame in enumerate(result["execution_frames"]):
            self.assertEqual({key: x for key, x in frame.items() if key != "hand_target_rad"},
                             {key: x for key, x in frames[index].items() if key != "hand_target_rad"})
            self.assertEqual(frame["hand_target_rad"], [
                frames[index]["hand_target_rad"][joint] + selected["execution_frame_hand_offsets_rad"][index][joint]
                for joint in range(7)])

    def test_score_bounds_stay_original_while_control_references_use_execution(self):
        value = candidate()
        old = resolve_preshape_frames(None, value, CANDIDATE_SHA)
        new = resolve_preshape_frames(control_preshape_contract(), value, CANDIDATE_SHA)
        for key in ("original_domain_lower7_rad", "original_domain_upper7_rad"):
            self.assertEqual(new[key], old[key])
        self.assertNotEqual(new["execution_reference_lower7_rad"], old["execution_reference_lower7_rad"])
        self.assertNotEqual(new["execution_reference_upper7_rad"], old["execution_reference_upper7_rad"])
        values = [frame["hand_target_rad"] for frame in new["execution_frames"]]
        for joint in range(7):
            self.assertEqual(new["execution_reference_lower7_rad"][joint],
                             max(min(row[joint] for row in values), new["original_domain_lower7_rad"][joint]))
            self.assertEqual(new["execution_reference_upper7_rad"][joint],
                             min(max(row[joint] for row in values), new["original_domain_upper7_rad"][joint]))
            native_lower, native_upper = value["geometry_evidence"]["hand_joint_limits_rad"][joint]
            for row in values:
                self.assertLessEqual(native_lower, row[joint])
                self.assertLessEqual(row[joint], native_upper)
                self.assertLessEqual(new["original_domain_lower7_rad"][joint], row[joint])
                self.assertLessEqual(row[joint], new["original_domain_upper7_rad"][joint])
        # Thumb1's execution max is higher; that must not move the original score upper bound.
        self.assertEqual(new["original_domain_upper7_rad"][1], .015076029779142719)

    def test_all_outputs_are_independent_and_no_input_is_modified(self):
        value, selected = candidate(), control_preshape_contract()
        before_value, before_selected = deepcopy(value), deepcopy(selected)
        new = resolve_preshape_frames(selected, value, CANDIDATE_SHA)
        self.assertEqual(value, before_value)
        self.assertEqual(selected, before_selected)
        new["original_frames"][0]["hand_target_rad"][0] = 99
        self.assertNotEqual(new["execution_frames"][0]["hand_target_rad"][0], 99)
        self.assertEqual(value, before_value)
        old = resolve_preshape_frames(None, value, CANDIDATE_SHA)
        old["execution_frames"][0]["hand_target_rad"][0] = 99
        self.assertNotEqual(old["original_frames"][0]["hand_target_rad"][0], 99)
        self.assertEqual(value, before_value)

    def test_candidate_file_pin_and_full_frame_object_pin_cannot_be_bypassed(self):
        with self.assertRaisesRegex(ValueError, "file SHA256"):
            resolve_preshape_frames(control_preshape_contract(), candidate(), "0" * 64)
        for key, replacement in (("name", "changed"), ("duration_s", .91), ("nominal_authored_gap_m", .0039),
                                 ("hand_target_rad", [0] * 7)):
            value = candidate()
            original_frames(value)[1][key] = replacement
            with self.subTest(frame_field=key), self.assertRaisesRegex(ValueError, "six-frame canonical"):
                resolve_preshape_frames(control_preshape_contract(), value, CANDIDATE_SHA)
        value = candidate()
        original_frames(value).pop()
        with self.assertRaisesRegex(ValueError, "six-frame canonical"):
            resolve_preshape_frames(control_preshape_contract(), value, CANDIDATE_SHA)

    def test_reset_native_limits_domain_and_other_candidate_fields_cannot_change(self):
        mutations = [
            lambda x: x.update(candidate_id="other"),
            lambda x: x["hand_joint_order"].reverse(),
            lambda x: x["reset_joint_positions_rad"].update(right_hand_thumb_1_joint=-.017542513087391853),
            lambda x: x["source_hand_rad"].__setitem__(1, -.017542513087391853),
            lambda x: x["target_rad"].__setitem__(1, 0),
            lambda x: x["geometry_evidence"]["hand_joint_limits_rad"][1].__setitem__(1, 9),
            lambda x: x["expected_geometry_q_domain"]["lower_rad"].__setitem__(1, -9),
            lambda x: x["expected_geometry_q_domain"]["native_joint_limits_rad"][1].__setitem__(1, 9),
            lambda x: x.update(undeclared=True),
            lambda x: x["reset_root_pose_xyzw"].__setitem__(0, 0),
        ]
        for index, mutate in enumerate(mutations):
            value = candidate()
            mutate(value)
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                resolve_preshape_frames(control_preshape_contract(), value, CANDIDATE_SHA)

    def test_invalid_frame_types_and_nonfinite_values_fail_before_envelope_math(self):
        for value in (None, {}, {"geometry_evidence": []},
                      {"geometry_evidence": {"source_detail": {"stage_keyframes": []}}}):
            with self.subTest(candidate=value), self.assertRaises(ValueError):
                resolve_preshape_frames(None, value, "")
        for replacement in (None, [0] * 6, [0] * 8, [False] * 7, [float("nan")] * 7,
                            [float("inf")] * 7, [10 ** 400] * 7):
            value = candidate()
            original_frames(value)[0]["hand_target_rad"] = replacement
            with self.subTest(hand=replacement), self.assertRaises(ValueError):
                resolve_preshape_frames(None, value, "")


if __name__ == "__main__":
    unittest.main()
