"""Phase admission must never erase early unsafe motion or fabricate support."""
import unittest
import torch

from g1_vla.rl_approach_state import ARM_NAMES, NativeApproachAdmission, current_object_height_unsafe


class NativeApproachAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.parameters = dict(maximum_object_xy_drift_m=.002, maximum_object_rotation_rad=.05,
            workspace_min_palm_z_m=.78,workspace_max_palm_z_m=1.1,
            approach_timeout_s=16.,grasp_budget_s=16.,
            joint_arrival_tolerance_rad=.015,joint_velocity_arrival_tolerance_rad_s=.05,
            hand_seed_arrival_tolerance_rad=.015,palm_position_arrival_tolerance_m=.003,
            palm_rotation_arrival_tolerance_rad=.05,handoff_required_consecutive_native_samples=25)
        self.resource = dict(pregrasp_joint_positions_rad=dict.fromkeys(ARM_NAMES,0.),
            source_hand_rad=[.5]*7,pregrasp_palm_position_world_m=[.38,.03,.825],
            pregrasp_palm_quaternion_xyzw=[0.,0.,0.,1.])
        self.state = NativeApproachAdmission(torch,self.resource,self.parameters,2,"cpu")
        self.episodes = torch.tensor([2,2])
        self.cube = torch.tensor([[.35,.055,.775,0.,0.,0.,1.]]).repeat(2,1)
        self.state.reset(torch.tensor([0,1]),self.episodes,0,self.cube)
        self.native = 0

    def sample(self, **kw):
        self.native += 1
        q = torch.zeros(2,14);q[:,7:] = .5
        args = dict(native_step=self.native,episode_ids=self.episodes,q_action=q,
            qd_action=torch.zeros_like(q),palm_position=torch.tensor([[.38,.03,.825]]).repeat(2,1),
            palm_quaternion=self.cube[:,3:].clone(),cube_position=self.cube[:,:3].clone(),
            cube_quaternion=self.cube[:,3:].clone(),env_origins=torch.zeros(2,3),
            measurement_valid=torch.ones(2,dtype=torch.bool),path_collision_clean=torch.ones(2,dtype=torch.bool),
            robot_cube_force_n=torch.zeros(2),mechanical_safe=torch.ones(2,dtype=torch.bool),
            unsafe=torch.zeros(2,dtype=torch.bool),whole_task_clean=torch.ones(2,dtype=torch.bool),
            plan_complete_requested=torch.ones(2,dtype=torch.bool),planner_failed=torch.zeros(2,dtype=torch.bool),
            full_success=torch.zeros(2,dtype=torch.bool))
        args.update(kw)
        return self.state.update(**args)

    def test_admission_requires_continuous_25_real_samples(self):
        for _ in range(24):self.sample()
        self.assertFalse(bool(self.state.admitted.any()))
        row=self.sample()
        self.assertTrue(bool(row["new_admission"].all()))
        self.assertEqual(row["admission_native"].tolist(),[25,25])
        self.assertEqual(row["reset_native"].tolist(),[0,0])
        self.assertEqual(self.state.episode_id.tolist(),[2,2])

    def test_one_missed_arrival_restarts_dwell(self):
        for _ in range(24):self.sample()
        self.sample(plan_complete_requested=torch.zeros(2,dtype=torch.bool))
        self.assertEqual(self.state.count.tolist(),[0,0])
        for _ in range(24):self.sample()
        self.assertFalse(bool(self.state.admitted.any()))
        self.sample();self.assertEqual(self.state.admission_native.tolist(),[50,50])

    def test_early_table_collision_cannot_be_forgiven_at_entry(self):
        self.sample(path_collision_clean=torch.tensor([False,True]))
        for _ in range(30):self.sample()
        self.assertEqual(self.state.admitted.tolist(),[False,True])
        self.assertEqual(self.state.failed.tolist(),[True,False])
        self.assertEqual(self.state.first_failure_code.tolist(),[2,0])

    def test_positive_small_cube_force_forbids_early_touch(self):
        self.sample(robot_cube_force_n=torch.tensor([1e-10,0.]))
        for _ in range(25):self.sample()
        self.assertEqual(self.state.admitted.tolist(),[False,True])
        self.assertEqual(self.state.first_failure_code.tolist(),[4,0])

    def test_force_history_and_original_domain_remain_required(self):
        self.sample(whole_task_clean=torch.tensor([False,True]))
        for _ in range(25):self.sample()
        self.assertEqual(self.state.admitted.tolist(),[False,True])
        self.assertTrue(bool(self.state.failed[0]))

    def test_shifted_cube_recovery_cannot_erase_path_failure(self):
        p=self.cube[:,:3].clone();p[0,0]+=.0021
        self.sample(cube_position=p)
        for _ in range(25):self.sample()
        self.assertEqual(self.state.admitted.tolist(),[False,True])
        self.assertEqual(self.state.first_failure_code.tolist(),[8,0])

    def test_post_entry_cube_load_and_translation_are_expected(self):
        for _ in range(25):self.sample()
        p=self.cube[:,:3].clone();p[:,2]+=.1;p[:,0]+=.02
        self.sample(cube_position=p,robot_cube_force_n=torch.ones(2))
        self.assertFalse(bool(self.state.failed.any()))
        self.assertTrue(bool(self.state.admitted.all()))

    def test_missing_sample_duplicate_and_stale_epoch_are_rejected(self):
        self.sample()
        with self.assertRaises(ValueError):self.sample(native_step=1)
        self.native=1
        with self.assertRaises(ValueError):self.sample(native_step=3)
        self.native=1
        with self.assertRaises(ValueError):self.sample(episode_ids=torch.tensor([3,2]))

    def test_only_real_partial_reset_clears_its_own_history(self):
        self.sample(path_collision_clean=torch.zeros(2,dtype=torch.bool))
        self.state.reset(torch.tensor([0]),torch.tensor([3]),1,self.cube[:1])
        self.episodes[0]=3
        for _ in range(25):self.sample()
        self.assertEqual(self.state.admitted.tolist(),[True,False])
        self.assertEqual(self.state.first_failure_native.tolist(),[-1,1])

    def test_grasp_time_budget_measured_from_actual_admission(self):
        for _ in range(25):self.sample()
        self.state.parameters["grasp_budget_s"]=.004
        self.sample();self.sample();self.assertFalse(bool(self.state.failed.any()))
        self.sample();self.assertEqual(self.state.first_failure_code.tolist(),[32,32])

    def test_workspace_nonfinite_and_wrong_boolean_types_fail_closed(self):
        p=torch.tensor([[.38,.03,1.101],[.38,.03,.825]])
        self.sample(palm_position=p)
        self.assertEqual(self.state.first_failure_code.tolist(),[16,0])
        self.sample(robot_cube_force_n=torch.tensor([0.,float("nan")]))
        self.assertTrue(bool(self.state.failed.all()))
        with self.assertRaises(ValueError):self.sample(measurement_valid=torch.ones(2))

    def test_clone_height_origin_is_explicit_and_not_inferred(self):
        origin=torch.tensor([[0.,0.,2.],[3.,0.,0.]])
        self.cube[:,:3]+=origin
        self.state=NativeApproachAdmission(torch,self.resource,self.parameters,2,"cpu")
        self.state.reset(torch.tensor([0,1]),self.episodes,0,self.cube)
        for _ in range(25):
            self.sample(env_origins=origin,palm_position=torch.tensor([[.38,.03,.825]]).repeat(2,1)+origin)
        self.assertTrue(bool(self.state.admitted.all()))

    def test_same_or_decreasing_reset_cannot_clear_history_or_move_reference(self):
        self.sample(path_collision_clean=torch.zeros(2,dtype=torch.bool))
        for episode,native in ((2,1),(1,1),(3,0),(3,2)):
            with self.assertRaises(ValueError):
                self.state.reset(torch.tensor([0]),torch.tensor([episode]),native,self.cube[:1])
            self.assertTrue(bool(self.state.failed[0]))
            self.assertEqual(int(self.state.episode_id[0]),2)
            self.assertTrue(torch.equal(self.state.cube_reference,self.cube))

    def test_current_sample_7mm_overshoot_cannot_write_an_admission_event(self):
        for _ in range(24):self.sample()
        unsafe=current_object_height_unsafe(torch,torch.tensor([.007,.0]),torch.tensor([.007,.0]),
            torch.zeros(2,dtype=torch.bool),local_cap_m=.006,local_tolerance_m=.0005,full_cap_m=.115)
        row=self.sample(unsafe=unsafe)
        self.assertEqual(row["new_admission"].tolist(),[False,True])
        self.assertEqual(row["first_failure_code"].tolist(),[64,0])
        self.assertEqual(row["admission_native"].tolist(),[-1,25])

    def test_after_main_lift_the_same_7mm_height_uses_original_115mm_ceiling(self):
        unsafe=current_object_height_unsafe(torch,torch.tensor([.007,.116]),torch.tensor([.007,.116]),
            torch.ones(2,dtype=torch.bool),local_cap_m=.006,local_tolerance_m=.0005,full_cap_m=.115)
        self.assertEqual(unsafe.tolist(),[False,True])


if __name__ == "__main__":unittest.main()
