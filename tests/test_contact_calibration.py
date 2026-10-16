"""Calibration math, identity and fresh-FK guards; never run real mj_step."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from g1_vla.config import load_config
from g1_vla.contact_calibration import (CalibrationError, build_fixture_xml, compiled_invariants,
    contact_wrench, convergence_report, fixture_command, phase_boundaries, rotation_distance,
    score_trial, split_solver_trace, residual_evidence_complete, _solver)


class ContactCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "configs" / "experiment.json")
        self.settings = copy.deepcopy(self.config["contact_calibration"])

    def test_waveforms_are_absolute_bounded_and_clear_force_after_single_pulse(self):
        settings = copy.deepcopy(self.settings)
        for dt in settings["physics_timesteps_s"]:
            times = phase_boundaries(settings)
            forces = []
            for index in range(round(times["end"] / dt)):
                command = fixture_command(settings, index * dt, dt)
                force = command["external_force_world_n"]
                self.assertEqual(force[0], 0.)
                self.assertEqual(force[2], 0.)
                self.assertEqual(command["external_torque_world_nm"], [0., 0., 0.])
                self.assertLessEqual(force[1], .1 * .08 * 9.81)
                if command["phase"] != "perturbation":
                    self.assertEqual(force, [0., 0., 0.])
                forces.append(force[1])
            expected_impulse = .1 * .08 * 9.81 * .1 * 2 / math.pi
            self.assertAlmostEqual(sum(forces) * dt, expected_impulse, delta=1e-6)
            opened = fixture_command(settings, times["opening_end"], dt)
            np.testing.assert_allclose(opened["jaw_positions_m"], [[-.0575, 0., .25], [.0575, 0., .25]])
            # An arbitrary call order has no hidden driver integrator.
            self.assertEqual(fixture_command(settings, .8, dt), fixture_command(settings, .8, dt))
        self.assertEqual(settings, self.settings)
        for t, dt in [(-1., .002), (0., 0.), (float("nan"), .002)]:
            with self.assertRaises(CalibrationError):
                fixture_command(settings, t, dt)

    def test_xml_cube_is_free_jaws_are_explicitly_kinematic_and_only_solver_count_differs(self):
        floor = {"friction": [1., .005, .0001], "condim": 3, "solref": [.02, 1.],
                 "solimp": [.9, .95, .001, .5, 2.], "margin": 0., "gap": 0.}
        xmls = []
        for backend in self.settings["solver_backends"]:
            xml, manifest = build_fixture_xml(self.settings, backend, .002, floor_contact=floor)
            root = ET.fromstring(xml)
            cube = root.find(".//body[@name='calibration_cube']")
            self.assertIsNotNone(cube.find("freejoint"))
            self.assertIsNone(cube.get("mocap"))
            self.assertIsNone(root.find("equality"))
            self.assertIsNone(root.find("actuator"))
            self.assertEqual(len(root.findall(".//body[@mocap='true']")), 2)
            self.assertEqual(cube.find("geom").get("friction"), "1.4 0.03 0.003")
            self.assertTrue(manifest["fixture_only"])
            self.assertFalse(manifest["training_eligible"])
            self.assertNotIn("scene_protocol", manifest)
            self.assertIn("fixture_scene_protocol", manifest)
            self.assertAlmostEqual(manifest["initial_soft_precompression_m"], .0015)
            root.find("option").set("noslip_iterations", "0")
            xmls.append(ET.tostring(root))
        self.assertEqual(xmls[0], xmls[1])

    def test_contact_wrench_uses_object_index_frame_and_COM_torque_without_force_cancellation(self):
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        contacts = [SimpleNamespace(geom1=0, geom2=1, frame=rotation.ravel(), pos=np.array([1., 0., 0.]),
            dist=-.001, dim=3, efc_address=0, friction=np.array([2.2, 2.2, .05, .005, .005]))]
        model = SimpleNamespace(geom_bodyid=np.array([1, 2]))
        data = SimpleNamespace(ncon=1, contact=contacts)
        api = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY=1, mjOBJ_GEOM=2),
            mj_id2name=lambda m, obj, idx: "left_jaw" if obj == 1 else "left_jaw_capsule",
            mj_contactForce=lambda m, d, i, output: output.__setitem__(slice(None), [2., 0., 0., 0., 0., 0.]))
        out = contact_wrench(model, data, 2, np.zeros(3), api)
        np.testing.assert_allclose(out["jaw_force_world_n"], [0., 0., 2.])
        np.testing.assert_allclose(out["jaw_torque_about_COM_world_nm"], [0., -2., 0.])
        self.assertEqual(out["maximum_absolute_jaw_contact_force_n"], 2.)
        # Reverse geometry order and contact normal: the force on cube stays identical.
        contacts[0].geom1, contacts[0].geom2 = 1, 0
        contacts[0].frame = (-rotation).ravel()
        reverse = contact_wrench(model, data, 2, np.zeros(3), api)
        np.testing.assert_allclose(reverse["jaw_force_world_n"], out["jaw_force_world_n"])

    def test_trace_split_does_not_treat_zero_noslip_gradient_as_main_convergence(self):
        main = dict(improvement=1e-10, gradient=.1, lineslope=1., neval=2., nupdate=1.)
        noslip = dict(improvement=1e-8, gradient=0., lineslope=0., neval=0., nupdate=0.)
        kwargs = dict(engine_version="3.2.5", newton=True, monolithic=True, main_budget=100,
                      noslip_budget=10, main_tolerance=1e-8, noslip_tolerance=1e-6)
        result = split_solver_trace([main, noslip], 2, **kwargs)
        self.assertTrue(result["split_valid"])
        self.assertEqual((result["main_iterations"], result["noslip_iterations"]), (1, 1))
        self.assertFalse(result["budget_or_tolerance_failure"])
        bad = dict(main, improvement=.1, gradient=.1)
        self.assertTrue(split_solver_trace([bad, noslip], 2, **kwargs)["budget_or_tolerance_failure"])
        for trace in ([noslip, main], [dict(main, neval=1)], [dict(noslip, gradient=1.)]):
            self.assertFalse(split_solver_trace(trace, len(trace), **kwargs)["split_valid"])
        self.assertFalse(split_solver_trace([main], 1000, **kwargs)["split_valid"])
        changed = dict(kwargs, engine_version="3.3.0")
        self.assertFalse(split_solver_trace([main], 1, **changed)["split_valid"])
        exhausted = [main] + [dict(noslip, improvement=.01)] * 10
        self.assertTrue(split_solver_trace(exhausted, 11, **kwargs)["budget_or_tolerance_failure"])

    def ideal_records(self, dt=.002, initial_release_velocity=.02):
        initial = {"COM_world_m": [0., 0., .25], "R_world": np.eye(3).tolist()}
        times = phase_boundaries(self.settings)
        rows = []
        for index in range(round(times["end"] / dt)):
            t = (index + 1) * dt
            free = t > times["opening_end"] + 1e-8
            elapsed = max(0, round((t - times["opening_end"]) / dt))
            z, vz = .25, initial_release_velocity if abs(t - times["opening_end"]) < 1e-8 else 0.
            if free:
                z += elapsed * dt * initial_release_velocity - 9.81 * dt * dt * elapsed * (elapsed + 1) / 2
                vz = initial_release_velocity - 9.81 * elapsed * dt
            force = {"jaw_normal_force_n": {"left_jaw": 0. if free else .4, "right_jaw": 0. if free else .4},
                "jaw_force_world_n": [0., 0., 0. if free else .7848],
                "floor_other_loaded_contacts": 0, "floor_other_geometric_contacts": 0,
                "floor_other_normal_force_n": 0., "all_jaw_geometric_contacts": 0 if free else 4,
                "maximum_absolute_jaw_contact_force_n": 0. if free else .4,
                "maximum_jaw_penetration_m": 0. if free else .0015}
            rows.append({"finite": True, "command": fixture_command(self.settings, index * dt, dt),
                "state": {"time_s": t, "COM_world_m": [0., 0., z], "R_world": np.eye(3).tolist(),
                    "COM_velocity_world_m_s": [0., 0., vz], "omega_world_rad_s": [0., 0., 0.],
                    "xfrc_applied": np.zeros((4, 6)).tolist(), "qfrc_applied": np.zeros(6).tolist()},
                "integration_force": force, "post_refresh_force": copy.deepcopy(force),
                "integration_solver": {"warnings": [0] * 8,
                    "termination": {"split_valid": True, "budget_or_tolerance_failure": False},
                    "nv": 6, "nefc": 0, "qacc_postprocessed": [0.] * 6,
                    "qacc_smooth": [0.] * 6, "qacc_warmstart_MAIN_before_noslip": [0.] * 6,
                    "constraints": {k: [] for k in ("efc_force", "efc_R", "efc_b", "efc_aref", "efc_vel", "efc_pos", "efc_state")},
                    "unregularized_acceleration_residual": [], "post_noslip_projected_pair_diagnostics": [],
                    "iteration_stats": []}})
        return rows, initial

    def test_fixed_windows_and_discrete_release_use_actual_nonzero_initial_velocity(self):
        for dt in (.002, .001):
            rows, initial = self.ideal_records(dt)
            score = score_trial(rows, initial, self.settings, dt, complete=True, video_complete=True)
            self.assertTrue(score["passed"], score["phase_passed"])
            self.assertEqual(score["actual_release_reference"]["COM_velocity_world_m_s"][2], .02)
            # Persist the full recovery/freefall report, not merely all(checks).
            # In particular np.linalg.norm(... )<=limit must serialize as a
            # JSON Boolean without a default=str escape hatch.
            encoded = json.dumps(score, allow_nan=False)
            restored = json.loads(encoded)
            self.assertTrue(restored["passed"])
            self.assertTrue(restored["checks_by_phase"]["recovery"])
            self.assertTrue(restored["checks_by_phase"]["free_fall"])
            for phase_rows in score["checks_by_phase"].values():
                for row in phase_rows:
                    for value in row["checks"].values():
                        self.assertIs(type(value), bool)
            continuous = copy.deepcopy(rows)
            end = phase_boundaries(self.settings)["opening_end"]
            for r in continuous:
                elapsed = r["state"]["time_s"] - end
                if elapsed > 1e-8:
                    r["state"]["COM_world_m"][2] = .25 + .02 * elapsed - .5 * 9.81 * elapsed ** 2
            bad = score_trial(continuous, initial, self.settings, dt, complete=True, video_complete=True)
            self.assertFalse(bad["phase_passed"]["free_fall"])
            self.assertFalse(json.loads(json.dumps(bad, allow_nan=False))["phase_passed"]["free_fall"])

    def test_recovery_reference_cannot_hide_global_drift_or_floor_support(self):
        rows, initial = self.ideal_records()
        for r in rows:
            r["state"]["COM_world_m"][0] = .004
        score = score_trial(rows, initial, self.settings, .002, complete=True, video_complete=True)
        self.assertFalse(score["phase_passed"]["hold"])
        self.assertFalse(score["phase_passed"]["perturb_recovery"])
        rows, initial = self.ideal_records()
        rows[300]["integration_force"]["floor_other_loaded_contacts"] = 1
        self.assertFalse(score_trial(rows, initial, self.settings, .002,
            complete=True, video_complete=True)["phase_passed"]["hold"])

    def test_release_rejects_hidden_actual_force_and_cancelled_opposed_contacts(self):
        changes = [lambda r: r["state"]["xfrc_applied"][0].__setitem__(1, .01),
            lambda r: r["state"]["qfrc_applied"].__setitem__(0, .01),
            lambda r: r["command"]["external_torque_world_nm"].__setitem__(0, .01),
            lambda r: r["integration_force"].__setitem__("maximum_absolute_jaw_contact_force_n", .4),
            lambda r: r["post_refresh_force"].__setitem__("all_jaw_geometric_contacts", 1),
            lambda r: r["post_refresh_force"].__setitem__("floor_other_geometric_contacts", 1)]
        for change in changes:
            rows, initial = self.ideal_records()
            change(rows[-1])
            score = score_trial(rows, initial, self.settings, .002, complete=True, video_complete=True)
            self.assertFalse(score["phase_passed"]["free_fall"])

    def test_incomplete_video_or_unconfirmed_solver_never_admits(self):
        rows, initial = self.ideal_records()
        self.assertFalse(score_trial(rows, initial, self.settings, .002, complete=True, video_complete=False)["passed"])
        self.assertFalse(score_trial(rows[:-1], initial, self.settings, .002, complete=False, video_complete=True)["passed"])
        rows[0]["integration_solver"]["termination"]["split_valid"] = False
        self.assertFalse(score_trial(rows, initial, self.settings, .002, complete=True, video_complete=True)["passed"])

    def test_solver_residual_completeness_requires_shapes_values_and_main_state(self):
        rows, _ = self.ideal_records()
        solver = rows[0]["integration_solver"]
        self.assertTrue(residual_evidence_complete(solver))
        for key, value in [("qacc_postprocessed", [0.] * 5),
                           ("qacc_smooth", [float("nan")] * 6),
                           ("unregularized_acceleration_residual", [0.])]:
            bad = copy.deepcopy(solver)
            bad[key] = value
            self.assertFalse(residual_evidence_complete(bad))
        bad = copy.deepcopy(solver)
        del bad["constraints"]["efc_aref"]
        self.assertFalse(residual_evidence_complete(bad))

    def test_convergence_requires_all_common_times_and_global_pose(self):
        coarse, _ = self.ideal_records(.002)
        fine, _ = self.ideal_records(.001)
        self.assertTrue(convergence_report(coarse, fine, self.settings)["passed"])
        self.assertFalse(convergence_report(coarse, fine[:-2], self.settings)["passed"])
        fine[999]["state"]["COM_world_m"][0] = .004
        self.assertFalse(convergence_report(coarse, fine, self.settings)["passed"])

    def test_convergence_force_windows_cannot_cancel_opposite_errors(self):
        coarse, _ = self.ideal_records(.002)
        fine, _ = self.ideal_records(.001)
        for r in fine:
            t = r["state"]["time_s"]
            if .5 <= t <= 1.:
                r["integration_force"]["jaw_force_world_n"][2] += .03
            if 1.402 <= t <= 1.6:
                r["integration_force"]["jaw_force_world_n"][2] -= .03
        result = convergence_report(coarse, fine, self.settings)
        self.assertFalse(result["checks"]["hold:Fz"])
        self.assertFalse(result["checks"]["recovery:Fz"])

    def test_unified_cli_stage_dispatches_only_the_new_diagnostic_api(self):
        from unittest.mock import patch
        from g1_vla.cli import execute_stage, STAGES
        result = {"passed": False, "fixtureonly": True, "calibrated_candidate": None}
        self.assertIn("contact-calibration", STAGES)
        with patch("g1_vla.cli.readiness", return_value={}), patch(
                "g1_vla.contact_calibration.run_contact_calibration", return_value=result) as run:
            report, success = execute_stage("contact-calibration", self.config, SimpleNamespace(path=Path("test-run")))
        self.assertFalse(success)
        self.assertFalse(report["training_started"])
        run.assert_called_once_with(self.config, Path("test-run"))

    def test_real_compile_and_forward_only_parity_and_solver_residuals(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest("MuJoCo unavailable; this test performs no integration")
        floor = {"friction": [1., .005, .0001], "condim": 3, "solref": [.02, 1.],
                 "solimp": [.9, .95, .001, .5, 2.], "margin": 0., "gap": 0.}
        for dt in self.settings["physics_timesteps_s"]:
            fingerprints = []
            for backend in self.settings["solver_backends"]:
                xml, _ = build_fixture_xml(self.settings, backend, dt, floor_contact=floor)
                model = mujoco.MjModel.from_xml_string(xml.decode())
                fingerprints.append(compiled_invariants(model))
                data = mujoco.MjData(model)
                old_q, old_v = data.qpos.copy(), data.qvel.copy()
                mujoco.mj_forward(model, data)
                evidence = _solver(model, data, mujoco)
                self.assertTrue(evidence["termination"]["split_valid"])
                self.assertTrue(evidence["post_noslip_projected_pair_diagnostics"])
                self.assertTrue(residual_evidence_complete(evidence))
                self.assertEqual(len(evidence["qacc_warmstart_MAIN_before_noslip"]), 6)
                cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calibration_cube")
                force = contact_wrench(model, data, cube, data.xipos[cube], mujoco)
                # Correcting cube material1.4 leaves original max-mixed2.2
                # jaw/cube normal loads unchanged at both declared dt values.
                for value in force["jaw_normal_force_n"].values():
                    self.assertAlmostEqual(value, .8490886448545236, delta=1e-10)
                np.testing.assert_array_equal(data.qpos, old_q)
                np.testing.assert_array_equal(data.qvel, old_v)
                self.assertEqual(data.time, 0.)
            self.assertEqual(fingerprints[0], fingerprints[1])


if __name__ == "__main__":
    unittest.main()
