"""Contract, denominator and optimizer safety checks without an Isaac task."""

import copy
import contextlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from g1_vla import rl_ppo


def flat_config():
    return {
        "enabled": True, "seed": 101, "evaluation_seed": 102, "smoke_seed": 103,
        "device": "cuda:0", "num_envs": 32, "git_executable": "D:/tools/git.exe",
        "asset_path": "robot.xml", "smoke_env_counts": [1, 8, 32], "smoke_steps": 128,
        "eval_episodes": 32, "eval_num_envs": 8, "episode_length_s": 6.0,
        "physics_dt_s": 0.002, "decimation": 33, "checkpoint_path": None,
        "checkpoint_sha256": None, "hidden_dims": [128, 128], "steps_per_env": 32,
        "num_learning_epochs": 4, "num_mini_batches": 4, "max_iterations": 400,
        "eval_interval": 50, "init_noise_std": 0.5, "entropy_coef": 0.01,
        "learning_rate": 0.0003, "max_wall_time_s": 3600,
        "gamma": 0.99, "lam": 0.95, "clip_param": 0.2, "max_grad_norm": 1.0,
        "physics_protocol": "physx_test_v1", "hand_friction": 2.2,
        "seed_state_sha256": "a" * 64, "asset_manifest_sha256": "b" * 64,
    }


def outcome(env, episode, success=False):
    return {"env_id": env, "episode_id": episode, "success": success,
            "episode_return": 10000, "gates": {"support": success},
            "failure_reasons": [] if success else ["support_failed"]}


def measurement_audit():
    return {"qualification_supported": True, "contact_data_unavailable_reason": None,
            "table_contact_channel": {"supported": True, "table_kinematic": True,
                "disable_gravity": True, "positive_force_physics_samples": 1,
                "positive_force_env_samples": 1, "max_force_n": 0.7848, "unavailable_reason": None}}


class RunnerContractTests(unittest.TestCase):
    def test_import_has_no_torch_or_simulator_side_effects(self):
        source = str(Path(rl_ppo.__file__).resolve().parents[1])
        code = "import sys; import g1_vla.rl_ppo; assert 'torch' not in sys.modules; assert 'isaaclab' not in sys.modules"
        env = dict(os.environ, PYTHONPATH=source)
        subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True)

    def test_settings_are_a_copy_and_pin_rsl5_actor_critic(self):
        raw = flat_config()
        original = copy.deepcopy(raw)
        settings = rl_ppo.normalize_settings(raw)
        rl_ppo._validate_settings(settings)
        cfg = rl_ppo.build_rsl_config(settings)
        self.assertEqual(raw, original)
        self.assertNotIn("policy", cfg)
        self.assertEqual(cfg["actor"]["hidden_dims"], [128, 128])
        self.assertIsNone(cfg["critic"]["distribution_cfg"])
        self.assertIn("GaussianDistribution", cfg["actor"]["distribution_cfg"]["class_name"])
        self.assertEqual(cfg["num_steps_per_env"], 32)
        self.assertEqual(cfg["algorithm"]["num_learning_epochs"], 4)
        self.assertEqual(cfg["obs_groups"], {"actor": ["policy"], "critic": ["policy"]})
        cfg["actor"]["hidden_dims"][0] = 4
        self.assertEqual(raw["hidden_dims"], [128, 128])

    def test_budget_expansion_and_seed_reuse_are_rejected(self):
        for key, value in (("max_iterations", 401), ("max_wall_time_s", 3601),
                           ("num_envs", 513), ("steps_per_env", 64),
                           ("num_learning_epochs", 8), ("eval_interval", 100),
                           ("num_mini_batches", 8),
                           ("evaluation_seed", 101), ("learning_rate", float("nan"))):
            with self.subTest(key=key):
                raw = flat_config()
                raw[key] = value
                with self.assertRaises(ValueError):
                    rl_ppo._validate_settings(rl_ppo.normalize_settings(raw))

    def test_capacity512_declares_full_budget_without_increasing_network_or_optimizer_calls(self):
        raw = flat_config()
        raw.update(num_envs=512, smoke_env_counts=[1, 8, 32, 128, 256, 512])
        settings = rl_ppo.normalize_settings(raw)
        rl_ppo._validate_settings(settings)
        plan = settings["planned_interaction"]
        self.assertEqual(plan["training_environment_transitions"], 6_553_600)
        self.assertEqual(plan["training_control_steps"], 12_800)
        self.assertEqual(plan["optimizer_steps"], 6_400)
        self.assertFalse(plan["observed_execution"])
        self.assertEqual(rl_ppo.build_rsl_config(settings)["actor"]["hidden_dims"], [128, 128])
        for counts in ([1, 8, 32], [1, 8, 32, 512, 512], [1, 32, 8, 512],
                       [1, 8, 32, True, 512], [1, 8, 32, 513], [1, 8, 512]):
            with self.subTest(counts=counts):
                raw["smoke_env_counts"] = counts
                with self.assertRaises(ValueError):
                    rl_ppo._validate_settings(rl_ppo.normalize_settings(raw))
        raw.update(num_envs=128, smoke_env_counts=[1, 8, 32, 128, 512])
        rl_ppo._validate_settings(rl_ppo.normalize_settings(raw))

    def test_memory_reserves_stop_without_claiming_capacity(self):
        memory = {"cuda": True, "free_bytes": rl_ppo.MIN_DEVICE_FREE_BYTES,
                  "host_available_ram_bytes": rl_ppo.MIN_HOST_AVAILABLE_BYTES}
        self.assertEqual(rl_ppo.memory_capacity_issues(memory), [])
        for field in ("free_bytes", "host_available_ram_bytes"):
            with self.subTest(field=field):
                self.assertTrue(rl_ppo.memory_capacity_issues(dict(memory, **{field: memory[field] - 1})))
                self.assertTrue(rl_ppo.memory_capacity_issues(dict(memory, **{field: None})))
        with tempfile.TemporaryDirectory() as directory:
            profile = {"num_envs": 512, "passed": False}
            stream = rl_ppo.EvidenceStream(Path(directory) / "memory.jsonl")
            try:
                with mock.patch.object(rl_ppo, "_gpu_memory", return_value=dict(memory, free_bytes=1)):
                    with self.assertRaises(rl_ppo.CapacityUnavailable):
                        rl_ppo._record_profile_memory(profile, "after_build", None, "cuda:0", stream,
                                                      require_headroom=True)
                self.assertFalse(profile["passed"])
                self.assertEqual(profile["memory"]["after_build"]["free_bytes"], 1)
                self.assertEqual(stream.count, 1)
            finally:
                stream.close()

    def test_contact_equivalence_requires_33_actual_samples_and_nonzero_contact(self):
        passed = {"enabled": True, "status": "passed", "completed_physics_samples": 33,
                  "completed_nonzero_samples": 1, "first_failure": None}
        self.assertEqual(rl_ppo.verify_contact_collection_crosscheck({"contact_collection_crosscheck": passed}), passed)
        for key, value in (("enabled", False), ("status", "pending_nonzero_contact"),
                           ("completed_physics_samples", 32), ("completed_physics_samples", 34),
                           ("completed_nonzero_samples", 0), ("completed_nonzero_samples", True),
                           ("first_failure", {"difference": 1})):
            with self.subTest(key=key, value=value):
                with self.assertRaises(rl_ppo.NumericalFailure):
                    rl_ppo.verify_contact_collection_crosscheck({"contact_collection_crosscheck": dict(passed, **{key: value})})
        with self.assertRaises(rl_ppo.NumericalFailure):
            rl_ppo.verify_contact_collection_crosscheck({})

    def test_contact_admission_requires_available_measurements_and_positive_kinematic_table_witness(self):
        good = measurement_audit()
        self.assertEqual(rl_ppo.verify_contact_measurements(good, require_table_witness=True), good["table_contact_channel"])
        for key, value in (("qualification_supported", False), ("qualification_supported", None),
                           ("contact_data_unavailable_reason", "Table unsupported"), ("table_contact_channel", None)):
            with self.subTest(key=key):
                with self.assertRaises(rl_ppo.NumericalFailure):
                    rl_ppo.verify_contact_measurements(dict(good, **{key: value}), require_table_witness=True)
        for key in ("qualification_supported", "contact_data_unavailable_reason", "table_contact_channel"):
            changed = copy.deepcopy(good)
            del changed[key]
            with self.subTest(missing=key):
                with self.assertRaises(rl_ppo.NumericalFailure):
                    rl_ppo.verify_contact_measurements(changed, require_table_witness=True)
        for key, value in (("supported", False), ("table_kinematic", False), ("disable_gravity", False),
                           ("positive_force_physics_samples", 0), ("positive_force_physics_samples", True),
                           ("max_force_n", 0), ("max_force_n", -1), ("max_force_n", True),
                           ("max_force_n", float("nan")), ("max_force_n", float("inf")),
                           ("unavailable_reason", "backend unavailable")):
            changed = copy.deepcopy(good)
            changed["table_contact_channel"][key] = value
            with self.subTest(channel_key=key, value=value):
                with self.assertRaises(rl_ppo.NumericalFailure):
                    rl_ppo.verify_contact_measurements(changed, require_table_witness=True)
        changed = copy.deepcopy(good)
        changed["table_contact_channel"]["positive_force_physics_samples"] = 0
        changed["table_contact_channel"]["max_force_n"] = 0
        self.assertIsNone(rl_ppo.verify_contact_measurements(changed))

    def test_missing_contact_measurements_are_recorded_and_rejected_before_episode_consumption(self):
        audit = {"physics_samples": 33, "sim_time_s": 0.066, "reset_count": 1,
                 "qualification_supported": False, "contact_data_unavailable_reason": "native pair read failed"}
        env = types.SimpleNamespace()
        env.unwrapped = types.SimpleNamespace(step_audit=lambda: audit, pop_episode_outcomes=mock.Mock())
        streams = {"steps": types.SimpleNamespace(write=mock.Mock()),
                   "episodes": types.SimpleNamespace(write=mock.Mock())}
        with self.assertRaises(rl_ppo.NumericalFailure):
            rl_ppo._record_step(env, {}, streams, "training", 1, 1, None)
        streams["steps"].write.assert_called_once()
        self.assertEqual(streams["steps"].write.call_args.args[0]["environment"], audit)
        env.unwrapped.pop_episode_outcomes.assert_not_called()

    @contextlib.contextmanager
    def native_capacity_fakes(self, directory):
        state = {"built": [], "closed": [], "fail_build_count": None, "low_after_first": False,
                 "missing_table_witness_count": None}
        app = types.SimpleNamespace(close=mock.Mock())
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(reset_peak_memory_stats=mock.Mock(),
                                                                 empty_cache=mock.Mock()))
        def build(config, **kwargs):
            state["built"].append(kwargs)
            count = kwargs["num_envs"]
            if state["fail_build_count"] == count:
                raise MemoryError("capacity allocation failed")
            def step_audit():
                audit = measurement_audit()
                if state["missing_table_witness_count"] == count:
                    audit["table_contact_channel"].update(positive_force_physics_samples=0, max_force_n=0.0)
                return audit
            env = types.SimpleNamespace(num_envs=count, step_audit=step_audit, get_observations=lambda: {})
            env.unwrapped = env
            env.close = lambda: state["closed"].append(count)
            return env
        def memory(*args):
            return {"cuda": True, "free_bytes": 1 if state["low_after_first"] and state["closed"] else 2 * 1024**3,
                    "total_bytes": 8 * 1024**3, "peak_allocated_bytes": 100,
                    "host_available_ram_bytes": 3 * 1024**3}
        modules = {"isaaclab": types.ModuleType("isaaclab"),
            "isaaclab.app": types.SimpleNamespace(AppLauncher=lambda **kwargs: types.SimpleNamespace(app=app)),
            "torch": torch, "isaaclab_rl": types.ModuleType("isaaclab_rl"),
            "isaaclab_rl.rsl_rl": types.SimpleNamespace(RslRlVecEnvWrapper=lambda env, **kwargs: env),
            "g1_vla.rl_environment": types.SimpleNamespace(build_environment=build)}
        config = {"rl_teacher": flat_config()}
        git = Path(directory) / "git.exe"
        asset = Path(directory) / "robot.usd"
        git.write_bytes(b"fake native git executable")
        asset.write_bytes(b"fake frozen asset")
        config["rl_teacher"].update(git_executable=str(git), asset_path=str(asset))
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, modules))
            stack.enter_context(mock.patch.object(rl_ppo, "_seed"))
            stack.enter_context(mock.patch.object(rl_ppo, "_assert_finite"))
            stack.enter_context(mock.patch.object(rl_ppo, "_environment_audit", return_value={"policy_contract": {"fake": True}}))
            stack.enter_context(mock.patch.object(rl_ppo, "_gpu_memory", side_effect=memory))
            stack.enter_context(mock.patch.object(rl_ppo.importlib.metadata, "version", return_value="5.0.1"))
            stack.enter_context(mock.patch.object(rl_ppo, "_smoke", return_value={"optimizer_steps": 0}))
            yield state, config

    def test_native_capacity_stop_retains_lower_results_and_uses_actual_smoke_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.native_capacity_fakes(directory) as (state, config):
                state["low_after_first"] = True
                run = Path(directory) / "run"
                with self.assertRaises(rl_ppo.CapacityUnavailable):
                    rl_ppo.run_native_stage("rl-smoke", config, run)
            report = json.loads((run / "native_report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["status"], "capacity_stopped")
            self.assertEqual([row["num_envs"] for row in state["built"]], [1])
            self.assertEqual(state["built"][0]["train_seed"], 103)
            self.assertEqual(report["profiles"][0]["actual_environment_seed"], 103)
            self.assertTrue(report["profiles"][0]["passed"])
            self.assertFalse(report["profiles"][1]["passed"])
            self.assertIn("before_build", report["profiles"][1]["memory"])
            self.assertEqual(report["capacity_summary"]["completed_profiles"], [1])
            self.assertEqual(report["capacity_summary"]["unattempted_profiles"], [32])
            self.assertFalse(report["capacity_summary"]["requested_training_capacity_smoke_passed"])
            self.assertEqual(report["optimizer_steps"], 0)

    def test_native_build_oom_is_not_hidden_by_lower_smoke_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.native_capacity_fakes(directory) as (state, config):
                state["fail_build_count"] = 8
                run = Path(directory) / "run"
                with self.assertRaises(MemoryError):
                    rl_ppo.run_native_stage("rl-smoke", config, run)
            report = json.loads((run / "native_report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertTrue(report["failure"]["out_of_memory"])
            self.assertTrue(report["profiles"][1]["out_of_memory"])
            self.assertEqual(report["capacity_summary"]["highest_completed_num_envs"], 1)
            self.assertIn("at_failure", report["profiles"][1]["memory"])

    def test_larger_profile_without_table_load_cannot_pass_or_qualify_training(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.native_capacity_fakes(directory) as (state, config):
                state["missing_table_witness_count"] = 8
                run = Path(directory) / "run"
                with self.assertRaises(rl_ppo.NumericalFailure):
                    rl_ppo.run_native_stage("rl-smoke", config, run)
            report = json.loads((run / "native_report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertFalse(report["profiles"][1]["passed"])
            self.assertEqual(report["profiles"][1]["final_step_audit"]["table_contact_channel"]["positive_force_physics_samples"], 0)
            self.assertEqual(report["capacity_summary"]["completed_profiles"], [1])
            self.assertFalse(report["capacity_summary"]["requested_training_capacity_smoke_passed"])

    def test_identity_ignores_only_host_selection_and_execution_budgets(self):
        config = {"rl_teacher": flat_config(), "control": {"gain": 12}, "action": {"joint_names": ["a"]}}
        expected = rl_ppo.policy_training_identity(config)
        for key, value in (("asset_path", "D:/new/export/robot.xml"),
                           ("checkpoint_path", "D:/prior/weights.pt"),
                           ("checkpoint_sha256", "f" * 64), ("max_iterations", 200),
                           ("max_wall_time_s", 1800), ("num_envs", 8)):
            changed = copy.deepcopy(config)
            changed["rl_teacher"][key] = value
            self.assertEqual(rl_ppo.policy_training_identity(changed), expected)
        for key, value in (("hand_friction", 3.0), ("seed_state_sha256", "c" * 64),
                           ("asset_manifest_sha256", "d" * 64), ("physics_protocol", "physx_changed")):
            changed = copy.deepcopy(config)
            changed["rl_teacher"][key] = value
            self.assertNotEqual(rl_ppo.policy_training_identity(changed), expected)
        changed = copy.deepcopy(config)
        changed["control"]["gain"] = 13
        self.assertNotEqual(rl_ppo.policy_training_identity(changed), expected)

    def test_metadata_identity_binds_bytes_instead_of_export_host_path(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "host_a.json"
            second = Path(directory) / "host_b.json"
            first.write_text('{"joint_names":["a"]}')
            second.write_bytes(first.read_bytes())
            config = {"rl_teacher": flat_config(), "action": {"joint_metadata_path": str(first)}}
            before = rl_ppo.policy_training_identity(config)
            config["action"]["joint_metadata_path"] = str(second)
            self.assertEqual(before, rl_ppo.policy_training_identity(config))
            second.write_text('{"joint_names":["b"]}')
            self.assertNotEqual(before, rl_ppo.policy_training_identity(config))

    def test_deadline_is_monotonic_and_rejects_exact_boundary(self):
        current = [10.0]
        deadline = rl_ppo.Deadline(2, clock=lambda: current[0])
        current[0] = 11.999
        deadline.check()
        current[0] = 12.0
        with self.assertRaises(rl_ppo.BudgetExpired):
            deadline.check()
        for value in (0, -1, 3601, math.inf, math.nan):
            with self.assertRaises(ValueError):
                rl_ppo.Deadline(value)

    def test_new_evidence_never_overwrites_and_rejects_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            writer = rl_ppo.EvidenceStream(path)
            with self.assertRaises(rl_ppo.NumericalFailure):
                writer.write({"loss": math.nan})
            self.assertEqual(writer.count, 0)
            writer.write({"optimizer_steps": 0})
            writer.close()
            previous = path.read_bytes()
            with self.assertRaises(FileExistsError):
                rl_ppo.EvidenceStream(path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(json.loads(previous), {"optimizer_steps": 0})

    def test_control_cadence_and_reset_evidence_reject_missing_or_changed_time(self):
        before = {"physics_samples": 100, "sim_time_s": 0.2, "reset_count": 8}
        after = {"physics_samples": 133, "sim_time_s": 0.266, "reset_count": 9,
                 "actual_physics_dt_s": 0.002}
        self.assertEqual(rl_ppo.verify_step_timing(before, after)["reset_delta"], 1)
        for key, value in (("physics_samples", 132), ("sim_time_s", 0.265),
                           ("reset_count", 7), ("actual_physics_dt_s", 0.001)):
            changed = dict(after, **{key: value})
            with self.assertRaises(rl_ppo.NumericalFailure):
                rl_ppo.verify_step_timing(before, changed)
        with self.assertRaises(ValueError):
            rl_ppo.verify_step_timing({}, after)

    def test_pre_simulator_failure_preserves_false_training_status(self):
        config = {"rl_teacher": flat_config()}
        config["rl_teacher"]["git_executable"] = "/nonexistent/stage15-test/git.exe"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                rl_ppo.run_native_stage("rl-train", config, directory)
            report = json.loads((Path(directory) / "native_report.json").read_text())
            self.assertFalse(report["passed"])
            self.assertFalse(report["training_started"])
            self.assertTrue(report["training_requested"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertEqual(report["failure"]["type"], "FileNotFoundError")
            self.assertFalse(report["application_close_requested"])
            with self.assertRaises(FileExistsError):
                rl_ppo.run_native_stage("rl-train", config, directory)


class EvaluationDenominatorTests(unittest.TestCase):
    def test_reward_does_not_become_success(self):
        unavailable = outcome(0, 1)
        unavailable["success"] = None
        unavailable["unavailable_reason"] = "no contact normals"
        report = rl_ppo.summarize_outcomes([unavailable])
        self.assertIsNone(report["success_rate"])
        self.assertEqual(report["score_status"], "unavailable")
        self.assertEqual(report["successes"], 0)
        self.assertFalse(report["reward_used_as_success"])

    def test_duplicate_episode_cannot_inflate_denominator(self):
        with self.assertRaises(ValueError):
            rl_ppo.summarize_outcomes([outcome(0, 1, True), outcome(0, 1, True)])

    def test_fast_failures_cannot_replace_slow_offsets(self):
        quota = rl_ppo.EvaluationQuota(8, 32)
        quota.add([outcome(0, index) for index in range(1, 41)])
        self.assertFalse(quota.complete)
        report = quota.report()
        self.assertEqual(report["completed_episodes"], 4)
        self.assertIsNone(report["success_rate"])
        self.assertEqual(report["score_status"], "incomplete")
        quota.add([outcome(env, episode, True) for env in range(1, 8) for episode in range(1, 5)])
        self.assertTrue(quota.complete)
        report = quota.report()
        self.assertEqual(report["completed_episodes"], 32)
        self.assertEqual(report["successes"], 28)
        self.assertEqual(report["success_rate"], 28 / 32)
        self.assertEqual(report["per_env_completed"], dict.fromkeys(range(8), 4))
        self.assertEqual(len(report["excluded_episode_ids"]), 36)

    def test_unscored_simulated_environments_are_preserved_but_excluded(self):
        quota = rl_ppo.EvaluationQuota(2, 4)
        quota.add([outcome(30, 1, True), outcome(31, 1, True), outcome(0, 1), outcome(1, 1)])
        self.assertEqual(len(quota.rows), 2)
        self.assertEqual(quota.excluded, [[30, 1], [31, 1]])
        self.assertFalse(quota.complete)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Torch-only optimizer tests; no Isaac simulator needed")
class TorchOptimizerTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.manual_seed(5)
        self.algorithm = types.SimpleNamespace(actor=torch.nn.Linear(2, 1), critic=torch.nn.Linear(2, 1))
        params = list(self.algorithm.actor.parameters()) + list(self.algorithm.critic.parameters())
        self.algorithm.optimizer = torch.optim.Adam(params, lr=0.001)

    def loss(self, nonfinite=False):
        x = self.torch.tensor([[1.0, 2.0], [-1.0, 3.0]])
        loss = self.algorithm.actor(x).square().mean() + self.algorithm.critic(x).square().mean()
        return loss * float("nan") if nonfinite else loss

    def test_real_backward_optimizer_count_and_parameter_change(self):
        torch = self.torch
        before = rl_ppo.parameter_snapshot(self.algorithm.actor, torch)
        guard = rl_ppo.OptimizerAudit(self.algorithm, torch, rl_ppo.Deadline(60))
        try:
            self.loss().backward()
            torch.nn.utils.clip_grad_norm_(self.algorithm.actor.parameters(), 0.1)
            torch.nn.utils.clip_grad_norm_(self.algorithm.critic.parameters(), 0.1)
            self.algorithm.optimizer.step()
            records = guard.drain()
            self.assertEqual(guard.optimizer_steps, 1)
            self.assertEqual(len(records), 1)
            self.assertGreater(records[0]["raw_gradient_l2"], records[0]["clipped_gradient_l2"])
            change = rl_ppo.parameter_change(before, rl_ppo.parameter_snapshot(self.algorithm.actor, torch), torch)
            self.assertGreater(change["changed_elements"], 0)
            self.assertGreater(change["l2"], 0)
        finally:
            guard.close()

    def test_nonfinite_backward_is_rejected_before_optimizer_mutation(self):
        torch = self.torch
        before = rl_ppo.parameter_snapshot(self.algorithm.actor, torch)
        guard = rl_ppo.OptimizerAudit(self.algorithm, torch, rl_ppo.Deadline(60))
        try:
            self.loss(nonfinite=True).backward()
            with self.assertRaises(rl_ppo.NumericalFailure):
                self.algorithm.optimizer.step()
            self.assertEqual(guard.optimizer_steps, 0)
            self.assertEqual(self.algorithm.optimizer.state, {})
            after = rl_ppo.parameter_snapshot(self.algorithm.actor, torch)
            self.assertEqual(rl_ppo.parameter_change(before, after, torch)["changed_elements"], 0)
        finally:
            guard.close()

    def test_deadline_prevents_an_optimizer_step(self):
        current = [0.0]
        guard = rl_ppo.OptimizerAudit(self.algorithm, self.torch, rl_ppo.Deadline(1, clock=lambda: current[0]))
        try:
            self.loss().backward()
            current[0] = 1.0
            with self.assertRaises(rl_ppo.BudgetExpired):
                self.algorithm.optimizer.step()
            self.assertEqual(guard.optimizer_steps, 0)
            self.assertEqual(self.algorithm.optimizer.state, {})
        finally:
            guard.close()

    def test_normalizer_buffer_mutation_changes_state_identity(self):
        module = self.torch.nn.Linear(2, 1)
        module.register_buffer("normalizer_mean", self.torch.zeros(2))
        before = rl_ppo.module_state_sha256(module, self.torch)
        module.normalizer_mean.add_(1)
        self.assertNotEqual(before, rl_ppo.module_state_sha256(module, self.torch))

    def test_double_cpu_parameter_snapshot_does_not_alias_weights(self):
        module = self.torch.nn.Linear(2, 1).double()
        before = rl_ppo.parameter_snapshot(module, self.torch)
        with self.torch.no_grad():
            module.weight.add_(1)
        after = rl_ppo.parameter_snapshot(module, self.torch)
        self.assertGreater(rl_ppo.parameter_change(before, after, self.torch)["l2"], 0)

    def evaluation_fixture(self, directory, mutate=False):
        torch = self.torch

        class Actor(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                self.register_buffer("normalizer_mean", torch.zeros(2))

            def forward(self, observation):
                if mutate:
                    self.normalizer_mean.add_(1)
                return self.linear(observation["policy"])

        class Observation(dict):
            def to(self, device):
                return Observation({name: value.to(device) for name, value in self.items()})

        class Environment:
            num_envs = 2
            num_actions = 1
            device = "cpu"

            def __init__(self):
                self.unwrapped = self
                self.samples = 0
                self.resets = 0
                self.episodes = 0
                self.rows = []
                self.actions = []
                self.eval = False

            def set_evaluation_mode(self, enabled, seed=None):
                self.eval = enabled

            def reset(self):
                self.resets += 2

            def state_hold_actions(self):
                # Nonzero request proves the runner invokes measured-state-hold
                # rather than silently substituting zero/previous-target hold.
                return torch.full((2, 1), 0.25)

            def get_observations(self):
                return Observation(policy=torch.ones(2, 2))

            def step(self, actions):
                self.actions.append(actions.clone())
                self.samples += 33
                self.resets += 2
                self.episodes += 1
                self.rows = [outcome(env, self.episodes) for env in range(2)]
                return self.get_observations(), torch.ones(2), torch.ones(2, dtype=torch.long), {}

            def step_audit(self):
                return {**measurement_audit(), "physics_samples": self.samples, "sim_time_s": self.samples * 0.002,
                        "reset_count": self.resets, "nonfinite_state": False}

            def pop_episode_outcomes(self):
                result, self.rows = self.rows, []
                return result

        environment = Environment()
        algorithm = types.SimpleNamespace(actor=Actor(), critic=torch.nn.Linear(2, 1))
        algorithm.eval_mode = lambda: (algorithm.actor.eval(), algorithm.critic.eval())
        runner = types.SimpleNamespace(alg=algorithm)
        settings = rl_ppo.normalize_settings(flat_config())
        settings["device"] = "cpu"
        settings["evaluation"].update(episodes=4, num_envs=2, max_control_steps=10)
        streams = rl_ppo._new_streams(Path(directory))
        return environment, runner, settings, streams

    def test_evaluation_uses_measured_state_hold_requests_and_freezes_normalizer(self):
        with tempfile.TemporaryDirectory() as directory:
            env, runner, settings, streams = self.evaluation_fixture(directory)
            try:
                report = rl_ppo._collect_evaluation(env, runner, settings, streams,
                    rl_ppo.Deadline(60), self.torch, update=0, phase="test_zero", state_hold=True)
                self.assertTrue(report["target_completed"])
                self.assertEqual(report["completed_episodes"], 4)
                self.assertEqual(report["optimizer_steps"], 0)
                self.assertEqual(report["actor_state_sha256_before"], report["actor_state_sha256_after"])
                self.assertTrue(all(bool((action == 0.25).all()) for action in env.actions))
                self.assertIn("measured-state-hold", report["action_policy"])
                self.assertFalse(env.eval)
            finally:
                for stream in streams.values():
                    stream.close()

    def test_evaluation_rejects_hidden_normalizer_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            env, runner, settings, streams = self.evaluation_fixture(directory, mutate=True)
            try:
                with self.assertRaises(rl_ppo.NumericalFailure):
                    rl_ppo._collect_evaluation(env, runner, settings, streams,
                        rl_ppo.Deadline(60), self.torch, update=0, phase="test_mutation")
                self.assertFalse(env.eval)
            finally:
                for stream in streams.values():
                    stream.close()

    def training_fixture(self, directory, *, expire_after=None, fail_final_checkpoint=False):
        env, runner, settings, streams = self.evaluation_fixture(directory)
        torch = self.torch
        current = [0.0]
        settings["source_tree_sha256"] = "source"
        settings["training_config_sha256"] = "config"
        settings["ppo"]["max_updates"] = 1
        algorithm = runner.alg
        algorithm.optimizer = torch.optim.Adam(list(algorithm.actor.parameters()) +
                                                 list(algorithm.critic.parameters()), lr=0.001)
        algorithm.act = lambda obs: algorithm.actor(obs)
        algorithm.process_env_step = lambda *args: None
        algorithm.compute_returns = lambda *args: None
        algorithm.train_mode = lambda: (algorithm.actor.train(), algorithm.critic.train())
        algorithm.storage = types.SimpleNamespace(returns=torch.ones(32, 2), advantages=torch.ones(32, 2))
        algorithm.learning_rate = 0.001

        def update():
            for step in range(16):
                algorithm.optimizer.zero_grad()
                loss = algorithm.actor({"policy": torch.ones(2, 2)}).square().mean()
                loss = loss + algorithm.critic(torch.ones(2, 2)).square().mean()
                loss.backward()
                algorithm.optimizer.step()
                if expire_after == step + 1:
                    current[0] = 60.0
            return {"value": 0.1, "surrogate": 0.2, "entropy": 0.3}
        algorithm.update = update

        def save_checkpoint(runner_arg, path, label, info):
            if fail_final_checkpoint and label.endswith("_final"):
                raise OSError("simulated final checkpoint disk failure")
            return {"label": label, **info}

        patches = (
            mock.patch.object(rl_ppo, "_make_runner", return_value=(runner, rl_ppo.build_rsl_config(settings))),
            mock.patch.object(rl_ppo, "_collect_evaluation", return_value={"budget_exhausted": False}),
            mock.patch.object(rl_ppo, "_save_checkpoint", side_effect=save_checkpoint),
        )
        deadline = rl_ppo.Deadline(60, clock=lambda: current[0])
        audit = {"policy_contract": {"toy": True}, "policy_contract_sha256": "contract"}
        return env, settings, streams, deadline, audit, patches

    def test_partial_first_update_timeout_records_actual_optimization_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            env, settings, streams, deadline, audit, patches = self.training_fixture(directory, expire_after=3)
            progress = {}
            try:
                with patches[0], patches[1], patches[2]:
                    with self.assertRaises(rl_ppo.BudgetExpired):
                        rl_ppo._train(env, settings, Path(directory), streams, deadline,
                                      self.torch, audit, progress)
                saved = json.loads((Path(directory) / "training_partial_failure.json").read_text())
                self.assertEqual(saved["completed_updates"], 0)
                self.assertEqual(saved["optimizer_steps"], 3)
                self.assertEqual(saved["partial_update_optimizer_steps"], 3)
                self.assertTrue(saved["training_started"])
                self.assertGreater(saved["actor_total_change"]["l2"], 0)
                self.assertGreater(saved["critic_total_change"]["l2"], 0)
                self.assertEqual(progress["optimizer_steps"], 3)
                self.assertEqual(saved["checkpoints"][-1]["optimizer_steps"], 3)
            finally:
                for stream in streams.values():
                    stream.close()

    def test_final_checkpoint_failure_retains_real_optimizer_count_and_weight_change(self):
        with tempfile.TemporaryDirectory() as directory:
            env, settings, streams, deadline, audit, patches = self.training_fixture(directory, fail_final_checkpoint=True)
            progress = {}
            try:
                with patches[0], patches[1], patches[2]:
                    with self.assertRaises(OSError):
                        rl_ppo._train(env, settings, Path(directory), streams, deadline,
                                      self.torch, audit, progress)
                saved = json.loads((Path(directory) / "training_partial_failure.json").read_text())
                self.assertEqual(saved["completed_updates"], 1)
                self.assertEqual(saved["optimizer_steps"], 16)
                self.assertTrue(saved["training_started"])
                self.assertGreater(saved["actor_total_change"]["l2"], 0)
                self.assertEqual(progress["optimizer_steps"], 16)
            finally:
                for stream in streams.values():
                    stream.close()


if __name__ == "__main__":
    unittest.main()
