"""A named wall-resource extension never changes physical or training budgets."""
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from g1_vla.rl_effort_protocol import (effort_grasp_contract,
    full_lift_effort_grasp_contract, full_lift_protocol_contract,
    validate_effort_grasp_contract, effort_evaluation_wall_time_limit_s,
    FULL_LIFT32_WALL7200_PROTOCOL)
from g1_vla.rl_probe_protocol import validate_probe_settings
from g1_vla.rl_protocol import RL_TEACHER_DEFAULTS, RL_PHYSICS, validate_rl_settings
from g1_vla.rl_ppo import Deadline, BudgetExpired, EvaluationQuota
from g1_vla.rl_diagnostics import _diagnostic_deadline


def diagnostic_config(protocol=FULL_LIFT32_WALL7200_PROTOCOL):
    config = json.loads((Path(__file__).resolve().parents[1]/"configs/experiment.json").read_text())
    settings = config["rl_diagnostics"]
    evaluation = full_lift_effort_grasp_contract(protocol)["evaluation"]
    settings["effort_grasp_controller"]["evaluation"] = evaluation
    settings.update({name: evaluation[name] for name in
        ("num_envs", "episodes_per_env", "max_control_steps", "max_wall_time_s")})
    settings["full_lift"] = full_lift_protocol_contract("g1_contact_qualified_full_lift_v4_native_arm_16s")
    # This fixture freezes the historical eight-env wall-budget protocol.
    settings.pop("approach_course", None)
    from g1_vla.rl_preshape import control_preshape_contract
    settings["control_preshape"] = control_preshape_contract()
    return config


class NamedWallResourceTests(unittest.TestCase):
    def test_new_factory_changes_only_resource_protocol_and_wall_default(self):
        old = full_lift_effort_grasp_contract("full_lift32_v2")
        new = full_lift_effort_grasp_contract(FULL_LIFT32_WALL7200_PROTOCOL)
        expected = copy.deepcopy(old)
        expected["evaluation"].update(protocol=FULL_LIFT32_WALL7200_PROTOCOL, max_wall_time_s=7200.)
        self.assertEqual(new, expected)
        evaluation = new["evaluation"]
        self.assertEqual((evaluation["num_envs"], evaluation["episodes_per_env"],
            evaluation["max_control_steps"], evaluation["nominal_episode_budget_s"]), (8, 4, 972, 16.))
        self.assertEqual(evaluation["validation_offsets"], "original_eight_offsets_unchanged")
        self.assertFalse(evaluation["randomization_expanded"])
        self.assertFalse(evaluation["training_allowed"])
        self.assertEqual(evaluation["optimizer_steps"], 0)
        self.assertEqual(validate_effort_grasp_contract(new), new)
        for value in [effort_grasp_contract()] + [full_lift_effort_grasp_contract(name) for name in
                ("full_lift8_v1", "full_lift32_v1", "full_lift8_v2", "full_lift32_v2")]:
            self.assertEqual(value["evaluation"]["max_wall_time_s"], 3600.)
            self.assertEqual(effort_evaluation_wall_time_limit_s(value), 3600.)
            value["evaluation"]["max_wall_time_s"] = 3601.
            with self.assertRaises(ValueError): validate_effort_grasp_contract(value)

    def test_both_validators_accept_only_the_explicit_extension_and_matching_walls(self):
        settings = diagnostic_config()["rl_diagnostics"]
        self.assertEqual(validate_probe_settings(settings)["max_wall_time_s"], 7200.)
        for seconds in (1, 1800., 3601., 7200.):
            candidate = copy.deepcopy(settings)
            candidate["max_wall_time_s"] = seconds
            candidate["effort_grasp_controller"]["evaluation"]["max_wall_time_s"] = seconds
            self.assertEqual(validate_probe_settings(candidate)["max_wall_time_s"], seconds)
        for outer, inner in ((3600., 7200.), (7200., 3600.), (7199., 7200.)):
            candidate = copy.deepcopy(settings)
            candidate["max_wall_time_s"] = outer
            candidate["effort_grasp_controller"]["evaluation"]["max_wall_time_s"] = inner
            with self.subTest(outer=outer, inner=inner), self.assertRaises(ValueError):
                validate_probe_settings(candidate)
        legacy = diagnostic_config("full_lift32_v2")["rl_diagnostics"]
        legacy["max_wall_time_s"] = 900.
        legacy["effort_grasp_controller"]["evaluation"]["max_wall_time_s"] = 1800.
        self.assertEqual(validate_probe_settings(legacy)["max_wall_time_s"], 900.,
            "Existing mismatched smaller wall declarations retain historical validation")

    def test_bad_numbers_unknown_names_and_physical_changes_are_rejected(self):
        settings = diagnostic_config()["rl_diagnostics"]
        for bad in (0, -.1, .5, 7200.001, 10800, True, math.inf, math.nan, "7200", None):
            candidate = copy.deepcopy(settings)
            candidate["max_wall_time_s"] = bad
            candidate["effort_grasp_controller"]["evaluation"]["max_wall_time_s"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_effort_grasp_contract(candidate["effort_grasp_controller"])
            with self.subTest(outer=bad), self.assertRaises(ValueError):
                validate_probe_settings(candidate)
        for key, bad in (("protocol", "full_lift32_v3_wall10800"), ("protocol", "full_lift32_v2"),
                ("episodes_per_env", 3), ("max_control_steps", 971),
                ("nominal_episode_budget_s", 20.), ("training_allowed", True),
                ("optimizer_steps", 1), ("validation_offsets", "new_offsets")):
            candidate = copy.deepcopy(settings)
            candidate["effort_grasp_controller"]["evaluation"][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                validate_probe_settings(candidate)
        candidate = copy.deepcopy(settings)
        candidate["effort_grasp_controller"]["acceptance"]["group_force_max_n"] = 1.6
        with self.assertRaises(ValueError): validate_probe_settings(candidate)
        candidate = copy.deepcopy(settings)
        candidate["full_lift"]["episode_length_s"] = 20.
        with self.assertRaises(ValueError): validate_probe_settings(candidate)

    def test_native_deadline_explicit_cap_and_monotonic_boundary(self):
        current = [10.]
        deadline = _diagnostic_deadline(diagnostic_config()["rl_diagnostics"], clock=lambda: current[0])
        self.assertEqual((deadline.seconds, deadline.maximum_seconds), (7200., 7200.))
        current[0] = 7209.999
        deadline.check()
        current[0] = 7210.
        with self.assertRaises(BudgetExpired): deadline.check()
        for bad in (7201., math.inf, math.nan, True, "7200"):
            with self.subTest(seconds=bad), self.assertRaises(ValueError):
                Deadline(bad, maximum_seconds=7200.)
        for bad in (10800., math.inf, math.nan, True, "7200"):
            with self.subTest(ceiling=bad), self.assertRaises(ValueError):
                Deadline(1., maximum_seconds=bad)
        for seconds in (3601., 7200.):
            with self.assertRaises(ValueError): Deadline(seconds)
        old = _diagnostic_deadline(diagnostic_config("full_lift32_v2")["rl_diagnostics"])
        self.assertEqual(old.maximum_seconds, 3600.)
        bad = diagnostic_config()["rl_diagnostics"]
        bad["max_wall_time_s"] = 3600.
        with self.assertRaises(ValueError): _diagnostic_deadline(bad)

    def test_training_budget_and_expired_partial_quota_cannot_inherit_extension(self):
        teacher = copy.deepcopy(RL_TEACHER_DEFAULTS)
        teacher["max_wall_time_s"] = 3601.
        with self.assertRaises(ValueError): validate_rl_settings(teacher)
        quota = EvaluationQuota(8, 32)
        quota.add([{"env_id":env, "episode_id":2, "success":True} for env in range(8)])
        current = [0.]
        deadline = _diagnostic_deadline(diagnostic_config()["rl_diagnostics"], clock=lambda:current[0])
        current[0] = 7200.
        with self.assertRaises(BudgetExpired): deadline.check()
        report = quota.report()
        self.assertFalse(report["target_completed"])
        self.assertEqual(report["per_env_quota"], dict.fromkeys(range(8), 4))
        self.assertEqual(report["completed_episodes"], 8)
        self.assertEqual(report["score_status"], "incomplete")
        self.assertIsNone(report["success_rate"])

    def test_real_launcher_call_uses_configured_wall_plus180_without_starting_a_child(self):
        from g1_vla.rl_teacher import run_rl_teacher_stage
        class StopBeforeNative(RuntimeError): pass
        original_is_file = Path.is_file
        for stage, protocol, expected_timeout in (("rl-diagnose", FULL_LIFT32_WALL7200_PROTOCOL, 7380.),
                ("rl-diagnose", "full_lift32_v2", 3780.), ("rl-train", FULL_LIFT32_WALL7200_PROTOCOL, 3780.),
                ("rl-eval", FULL_LIFT32_WALL7200_PROTOCOL, 900)):
            config = diagnostic_config(protocol)
            config["rl_teacher"]["max_wall_time_s"] = 3600.
            if stage != "rl-diagnose":
                config["rl_teacher"]["physics_protocol"] = RL_PHYSICS
                config["rl_teacher"].update(checkpoint_path="frozen.pt", checkpoint_sha256="a"*64)
            calls = []
            def communicate(payload, *, timeout):
                calls.append(timeout)
                raise StopBeforeNative()
            with self.subTest(stage=stage, protocol=protocol), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); run = root/"run"; (run/"logs").mkdir(parents=True)
                def is_file(path):
                    return True if str(path)=="/mnt/c/Windows/System32/cmd.exe" else original_is_file(path)
                with patch("g1_vla.rl_teacher.verify_rl_assets", return_value={"entrypoint":"unused"}), \
                        patch("g1_vla.rl_teacher._snapshot_release", return_value=("a"*64,"D:/unused",root)), \
                        patch("g1_vla.rl_teacher._snapshot_runtime_assets", return_value="D:/unused"), \
                        patch("g1_vla.rl_teacher._smoke_admission", return_value={}), \
                        patch("g1_vla.rl_teacher._native_script", return_value="no child execution"), \
                        patch("g1_vla.rl_teacher.windows_to_wsl", return_value=root/"nonexistent_output"), \
                        patch("g1_vla.rl_probe_protocol.verify_probe_reference", return_value={}), \
                        patch("pathlib.Path.is_file", new=is_file), \
                        patch("g1_vla.rl_teacher.subprocess.Popen", return_value=SimpleNamespace(communicate=communicate)) as popen:
                    with self.assertRaises(StopBeforeNative): run_rl_teacher_stage(stage, config, run)
                    self.assertEqual(popen.call_count, 1)
                self.assertEqual(calls, [expected_timeout])


if __name__ == "__main__":
    unittest.main()
