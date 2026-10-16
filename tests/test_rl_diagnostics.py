"""No-optimization control interventions and native dispatch regressions."""
import copy
import importlib.util
import unittest

from g1_vla.rl_probe_protocol import RL_PROBE_DEFAULTS

TORCH = importlib.util.find_spec("torch") is not None
RSL = TORCH and importlib.util.find_spec("rsl_rl") is not None and importlib.util.find_spec("tensordict") is not None


class DiagnosticVideoDeliveryTests(unittest.TestCase):
    def test_missing_or_partial_profile_clips_cannot_pass_delivery(self):
        from g1_vla.rl_teacher import _diagnostic_video_delivery
        profiles = [{"profile": profile, "control_steps": 91,
                     "first_video_frame": 91*i, "last_video_frame_exclusive": 91*(i+1)}
                    for i, profile in enumerate(RL_PROBE_DEFAULTS["profiles"])]
        clips = [{"profile": p["profile"]["name"], "first_source_frame": p["first_video_frame"],
                  "last_source_frame_exclusive": p["last_video_frame_exclusive"],
                  "decoded_frames": 91, "encoded_fps": "500/33"} for p in profiles]
        native = {"video_evidence_complete": True, "profiles": profiles}
        videos = [{"encoded": True, "decoded_frames": 364, "encoded_fps": "500/33", "profile_clips": clips}]
        self.assertTrue(_diagnostic_video_delivery(videos, native))
        broken = copy.deepcopy(videos)
        broken[0]["profile_clips"].pop()
        self.assertFalse(_diagnostic_video_delivery(broken, native))
        broken = copy.deepcopy(videos)
        broken[0]["profile_clips"][2]["decoded_frames"] = 90
        self.assertFalse(_diagnostic_video_delivery(broken, native))
        self.assertFalse(_diagnostic_video_delivery(videos, {**native, "video_evidence_complete": False}))

    def test_single_profile_must_match_external_declaration_and_cannot_forge_four_way(self):
        from g1_vla.rl_teacher import _diagnostic_video_delivery
        expected = [copy.deepcopy(RL_PROBE_DEFAULTS["profiles"][3])]
        native = {"video_evidence_complete": True, "profiles": [{"profile": expected[0],
            "first_video_frame": 0, "last_video_frame_exclusive": 91, "control_steps": 91}]}
        videos = [{"encoded": True, "decoded_frames": 91, "encoded_fps": "500/33", "profile_clips": [{
            "profile": expected[0]["name"], "first_source_frame": 0, "last_source_frame_exclusive": 91,
            "decoded_frames": 91, "encoded_fps": "500/33"}]}]
        self.assertTrue(_diagnostic_video_delivery(videos, native, expected))
        self.assertFalse(_diagnostic_video_delivery(videos, native))
        broken = copy.deepcopy(native)
        broken["profiles"][0]["profile"]["control_guard_version"] = "legacy_scalar_v1"
        self.assertFalse(_diagnostic_video_delivery(videos, broken, expected))
        broken_video = copy.deepcopy(videos)
        broken_video[0]["encoded_fps"] = "30/1"
        self.assertFalse(_diagnostic_video_delivery(broken_video, native, expected))


@unittest.skipUnless(TORCH, "real torch is required for control math")
class DiagnosticControlTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        self.cfg = copy.deepcopy(RL_PROBE_DEFAULTS)
        self.cfg["guided_hand_target_rad"] = [.6, -.5, -.4, .6, .6, .6, .6]
        self.seed = [.8, -.2, -.1, .3, .3, .3, .3]
        self.state = {"q_action_rad": torch.zeros(2, 14), "qd_action_rad_s": torch.zeros(2, 14),
            "target_action_rad": torch.zeros(2, 14), "per_step_action_rad": torch.full((14,), .08),
            "joint_limits_action_rad": torch.tensor([[[-2., 2.]] * 14] * 2),
            "palm_position_world_m": torch.zeros(2, 3),
            "palm_quaternion_xyzw": torch.tensor([[0., 0., 0., 1.]] * 2),
            "palm_point_jacobian_w": torch.cat((torch.eye(6), torch.zeros(6, 1)), -1)[None].repeat(2, 1, 1),
            "group_normal_force_n": torch.zeros(2, 3), "episode_control_steps": torch.zeros(2, dtype=torch.long),
            "episode_id": torch.zeros(2, dtype=torch.long), "preload_contact_ready": torch.zeros(2, dtype=torch.bool)}
        self.state["target_action_rad"][:, 7:] = torch.tensor(self.seed)
        self.state["q_action_rad"][:, 7:] = torch.tensor(self.seed)

    def staged_controller(self):
        from g1_vla.rl_diagnostics import GuidedClosure
        cfg = copy.deepcopy(self.cfg)
        cfg.update(profile_plan="geometry_pose_only_v2", guided_target_lead_max_rad=.006)
        goals = [[.8, -.2, -.1, .5, .3, .3, .3], [.8, -.4, -.1, .5, .3, .3, .3],
                 [.8, -.4, -.2, .5, .3, .3, .3], [.7, -.4, -.2, .5, .3, .3, .3],
                 cfg["guided_hand_target_rad"]]
        frames = [{"name": "source", "target_rad": self.seed}] + [
            {"name": "stage"+str(i), "duration_s": .066, "target_rad": q} for i, q in enumerate(goals)]
        candidate = {"geometry_evidence": {"source_detail": {"stage_keyframes": frames}}}
        return GuidedClosure(self.t, cfg, self.seed, self.state, candidate), goals

    def test_staged_closure_requires_measured_arrival_and_safe_force_not_elapsed_time(self):
        controller, goals = self.staged_controller()
        for _ in range(6):
            _, e = controller.actions(self.t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
            self.assertTrue((e["closure_stage_index"] == 0).all())
            self.assertFalse(e["stage_advance_after_this_request"].any())
            self.assertLessEqual(float((e["guided_target_hand_rad"]-self.state["q_action_rad"][:, 7:]).abs().max()), .0060001)
        self.state["q_action_rad"][:, 7:] = self.t.tensor(goals[0])
        self.state["group_normal_force_n"][:] = 2.
        _, e = controller.actions(self.t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertFalse(e["stage_advance_after_this_request"].any())
        self.state["group_normal_force_n"][:] = 0.
        _, e = controller.actions(self.t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertTrue(e["stage_advance_after_this_request"].all())

    def test_preload_cannot_bypass_staged_path_and_episode_reset_restarts_path(self):
        controller, goals = self.staged_controller()
        self.state["preload_contact_ready"][:] = True
        for _ in range(4):
            _, e = controller.actions(self.t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertTrue((e["probe_offset_m"] == 0).all())
        for goal in goals:
            self.state["q_action_rad"][:, 7:] = self.t.tensor(goal)
            _, e = controller.actions(self.t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        for _ in range(3):
            _, e = controller.actions(self.t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertTrue((e["probe_offset_m"] > 0).all())
        self.state["episode_id"][0] += 1
        self.state["q_action_rad"][0, 7:] = self.t.tensor(self.seed)
        _, e = controller.actions(self.t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertEqual(int(e["closure_stage_index"][0]), 0)
        self.assertEqual(float(e["probe_offset_m"][0]), 0.)

    def test_staged_candidate_cannot_change_reset_seed_or_native_limits(self):
        from g1_vla.rl_diagnostics import GuidedClosure
        controller, _ = self.staged_controller()
        # The source check uses actual reset q, independently of file validation.
        cfg = copy.deepcopy(self.cfg)
        cfg.update(profile_plan="geometry_pose_only_v2", guided_target_lead_max_rad=.006)
        frames = [{"name": "source", "target_rad": [0.]*7}] + [
            {"name": "stage", "duration_s": .066, "target_rad": self.cfg["guided_hand_target_rad"]}] * 5
        with self.assertRaisesRegex(ValueError, "actual reset seed"):
            GuidedClosure(self.t, cfg, self.seed, self.state,
                          {"geometry_evidence": {"source_detail": {"stage_keyframes": frames}}})

    def test_geometry_pose_servo_integrates_target_and_pauses_closure_when_palm_droops(self):
        controller, goals = self.staged_controller()
        self.state["target_action_rad"][:, :7] = .04
        self.state["palm_position_world_m"][:, 0] = -.001
        action, evidence = controller.actions(self.t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertGreater(float(action[0, 0]), 0.)
        self.assertEqual(evidence["arm_target_reference"], "previous_applied_bounded_target")
        self.state["q_action_rad"][:, 7:] = self.t.tensor(goals[0])
        self.state["palm_position_world_m"][:, 2] = -.016
        _, evidence = controller.actions(self.t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertTrue(evidence["closure_paused_for_palm_tracking"].all())
        self.assertFalse(evidence["stage_advance_after_this_request"].any())

    def test_hand_intervention_preserves_actor_arm_and_closes_declared_directions(self):
        from g1_vla.rl_diagnostics import GuidedClosure
        t = self.t
        controller = GuidedClosure(t, self.cfg, self.seed, self.state)
        requested = t.linspace(-.7, .7, 28).reshape(2, 14)
        self.state["episode_control_steps"][:] = 30
        result, evidence = controller.actions(requested, self.state, "guided_hand_actor_arm", .066, 1.5)
        self.assertTrue(t.equal(result[:, :7], requested[:, :7]))
        self.assertTrue((result[:, 7:10] < 0).all())
        self.assertTrue((result[:, 10:] > 0).all())
        self.assertTrue((result.abs() <= 1).all())

    def test_force_relief_returns_toward_seed_without_opening_beyond_it(self):
        from g1_vla.rl_diagnostics import GuidedClosure
        t = self.t
        controller = GuidedClosure(t, self.cfg, self.seed, self.state)
        self.state["target_action_rad"][:, 7:] -= .02
        self.state["group_normal_force_n"][:, 0] = 2.
        _, evidence = controller.actions(t.zeros(2, 14), self.state, "guided_hand_actor_arm", .066, 1.5)
        desired = evidence["guided_target_hand_rad"][:, :3]
        self.assertTrue(t.allclose(desired, t.tensor(self.seed[:3])[None] - .015))

    def test_probe_requires_declared_preload_and_pauses_on_loss(self):
        from g1_vla.rl_diagnostics import GuidedClosure
        t = self.t
        controller = GuidedClosure(t, self.cfg, self.seed, self.state)
        for _ in range(3):
            _, e = controller.actions(t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertTrue((e["probe_offset_m"] == 0).all())
        self.state["preload_contact_ready"][:] = True
        for _ in range(2):
            _, e = controller.actions(t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertTrue((e["probe_offset_m"] == 0).all())
        _, e = controller.actions(t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertTrue(t.allclose(e["probe_offset_m"], t.full((2,), .002 * .066)))
        before = e["probe_offset_m"].clone()
        self.state["preload_contact_ready"][:] = False
        _, e = controller.actions(t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertTrue(t.equal(before, e["probe_offset_m"]))
        self.assertTrue(e["upward_probe_request_clamped"].all())
        self.assertTrue((e["requested_linear_velocity_world_m_s"][:, 2] <= 0).all())
        self.state["episode_id"][0] += 1
        self.state["palm_position_world_m"][0] = t.tensor([1., 2., 3.])
        _, e = controller.actions(t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertEqual(float(e["probe_offset_m"][0]), 0.)
        self.assertTrue(t.equal(e["palm_pose_reference_world_m"][0], t.tensor([1., 2., 3.])))

    def test_probe_is_capped_and_pose_servo_has_finite_bounded_requests(self):
        from g1_vla.rl_diagnostics import GuidedClosure
        t = self.t
        controller = GuidedClosure(t, self.cfg, self.seed, self.state)
        self.state["preload_contact_ready"][:] = True
        for _ in range(40):
            a, e = controller.actions(t.zeros(2, 14), self.state, "guided_hand_pose_hold", .066, 1.5)
        self.assertTrue(t.isfinite(a).all())
        self.assertTrue((a.abs() <= 1).all())
        self.assertLessEqual(float(e["probe_offset_m"].max()), .002500001)

    def test_training_is_rejected_and_torch_methods_are_restored(self):
        from g1_vla.rl_diagnostics import inference_only
        t = self.t
        original = t.Tensor.backward
        with inference_only(t):
            with self.assertRaisesRegex(RuntimeError, "prohibits"):
                t.optim.Adam([t.nn.Parameter(t.tensor(1.))])
            with self.assertRaisesRegex(RuntimeError, "prohibits"):
                t.tensor(1.).backward()
        self.assertIs(t.Tensor.backward, original)
        value = t.tensor(2., requires_grad=True)
        value.square().backward()
        self.assertEqual(float(value.grad), 4.)


@unittest.skipUnless(RSL, "installed RSL and torch are required")
class DirectFrozenActorTests(unittest.TestCase):
    def test_actual_rsl_actor_load_and_inference_without_optimizer(self):
        import torch
        from tensordict import TensorDict
        from g1_vla.rl_protocol import RL_TEACHER_DEFAULTS
        from g1_vla.rl_diagnostics import inference_only, make_frozen_actor
        from g1_vla.rl_ppo import module_state_sha256
        settings = copy.deepcopy(RL_TEACHER_DEFAULTS)
        settings["device"] = "cpu"
        obs = TensorDict({"policy": torch.randn(8, 81)}, batch_size=[8])
        with inference_only(torch):
            # Construct the audited model class, then strictly round-trip every
            # parameter and normalization buffer through the production loader.
            from rsl_rl.models import MLPModel
            original = MLPModel(obs, {"actor": ["policy"]}, "actor", 14,
                hidden_dims=[128, 128], activation="elu", obs_normalization=True,
                distribution_cfg={"class_name": "rsl_rl.modules.distribution:GaussianDistribution",
                                  "init_std": .5, "std_type": "log"}).eval()
            saved = {"actor_state_dict": original.state_dict()}
            actor = make_frozen_actor(torch, obs, saved, settings)
            before = module_state_sha256(actor, torch)
            for _ in range(4):
                result = actor(obs)
                self.assertTrue(torch.equal(result, original(obs)))
            self.assertEqual(before, module_state_sha256(actor, torch))
            self.assertTrue(all(not p.requires_grad for p in actor.parameters()))
