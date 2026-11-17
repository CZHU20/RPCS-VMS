"""Check that pressure ablation removes each complete residual independently."""

import copy
import sys
from pathlib import Path
import unittest

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from mpi4py import MPI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from solver.configuration import CaseConfig
from solver.monolithic_vms import MonolithicSUPGPSPGSolver
from test_cases.manufactured.parameters import SMALL_TIME_STEP
from test_cases.manufactured.run_pressure_stage_ablation import configuration, initialize
from test_cases.manufactured.run_stabilization_effect import GROUPS, gather_pressure_field, run_case
from test_cases.manufactured.run_boundary_pressure_comparison import compute_metrics, pressure_error_field
from test_cases.manufactured.run_spatial_convergence import compute_error_norms as spatial_errors
from test_cases.manufactured.run_temporal_convergence import compute_error_norms as temporal_errors


def matrix(form):
    result = assemble_matrix(form)
    result.assemble()
    return result


class PressureStageTests(unittest.TestCase):
    def setUp(self):
        self.solvers = {key: initialize(configuration(16, .01, .02, key))
                       for key in ("increment_only", "recovery_only", "full")}
        cfg = configuration(16, .01, .02, "full")
        cfg.cfg["solver"].update(pressure_increment_stabilization=False,
                                 pressure_recovery_stabilization=False)
        self.solvers["neither"] = initialize(cfg)

    def compare(self, a, b):
        a.axpy(-1., b)
        self.assertLess(a.norm(), 1e-12 * max(1., b.norm()))
        a.destroy()
        b.destroy()

    def test_option_validation(self):
        cfg = copy.deepcopy(SMALL_TIME_STEP)
        cfg["solver"]["pressure_increment_stabilization"] = "off"
        cfg["solver"]["pressure_recovery_stabilization"] = "on"
        normalized = CaseConfig(cfg).cfg["solver"]
        self.assertIs(normalized["pressure_increment_stabilization"], False)
        self.assertIs(normalized["pressure_recovery_stabilization"], True)
        cfg["solver"]["pressure_increment_stabilization"] = "invalid"
        with self.assertRaises(ValueError):
            CaseConfig(cfg)

    def test_defaults_match_complete_scheme(self):
        cfg = configuration(16, .01, .02, "full")
        cfg.cfg["solver"].pop("pressure_increment_stabilization")
        cfg.cfg["solver"].pop("pressure_recovery_stabilization")
        default = initialize(cfg)
        for order in (1, 2):
            for name in ("predictor_a", "phi_a", "pressure_a"):
                self.compare(matrix(default.forms[order][name]),
                             matrix(self.solvers["full"].forms[order][name]))
            for name in ("predictor_L", "b_phi_form", "pressure_L"):
                self.compare(assemble_vector(default.forms[order][name]),
                             assemble_vector(self.solvers["full"].forms[order][name]))

    def test_unchanged_predictor_and_independent_stages(self):
        full = self.solvers["full"]
        for order in (1, 2):
            for solver in self.solvers.values():
                self.compare(matrix(solver.forms[order]["predictor_a"]), matrix(full.forms[order]["predictor_a"]))
                self.compare(assemble_vector(solver.forms[order]["predictor_L"]),
                             assemble_vector(full.forms[order]["predictor_L"]))
            for name in ("phi_a", "b_phi_form", "pressure_a", "pressure_L"):
                variant = "increment_only" if name in ("phi_a", "b_phi_form") else "recovery_only"
                assemble = matrix if name.endswith("_a") else assemble_vector
                self.compare(assemble(self.solvers[variant].forms[order][name]), assemble(full.forms[order][name]))

    def test_both_sides_of_stabilization_are_removed(self):
        full, plain = self.solvers["full"], self.solvers["neither"]
        p, q = ufl.TrialFunction(full.Q), ufl.TestFunction(full.Q)
        for order, gamma in ((1, 1.), (2, 1.5)):
            tau = full.element_stabilization.entries[gamma]["tau_m"]
            for name, factor in (("phi_a", 1.), ("pressure_a", full.nu)):
                difference = matrix(full.forms[order][name])
                removed = matrix(plain.forms[order][name])
                difference.axpy(-1., removed)
                removed.destroy()
                self.assertGreater(difference.norm(), 1e-8)
                expected = fem.form(factor * tau * ufl.inner(ufl.grad(p), ufl.grad(q)) * full.mh.dx)
                self.compare(difference, matrix(expected))
            _, history, _, wind = full._history_terms(order)
            acceleration = (gamma * full.vtilde + history) / full.dt
            transport = full._advective_term(wind, full.vtilde)
            rm = acceleration + transport - full._viscous_strong_operator(full.vtilde) + ufl.grad(full.po) - full.f_form
            rp = acceleration + transport + full.nu * (ufl.grad(ufl.div(full.vtilde)) - full._vector_laplacian(full.vtilde)) - full.f_form
            for name, factor, residual in (("b_phi_form", 1., rm), ("pressure_L", full.nu, rp)):
                difference = assemble_vector(full.forms[order][name])
                removed = assemble_vector(plain.forms[order][name])
                difference.axpy(-1., removed)
                removed.destroy()
                self.assertGreater(difference.norm(), 1e-8)
                expected = fem.form(-factor * tau * ufl.inner(residual, ufl.grad(q)) * full.mh.dx)
                self.compare(difference, assemble_vector(expected))

    def test_standard_rotational_recovery_is_retained(self):
        solver = self.solvers["neither"]
        p, q = ufl.TrialFunction(solver.Q), ufl.TestFunction(solver.Q)
        for order, gamma in ((1, 1.), (2, 1.5)):
            self.compare(matrix(solver.forms[order]["phi_a"]), matrix(fem.form(
                solver.dt / gamma * ufl.inner(ufl.grad(p), ufl.grad(q)) * solver.mh.dx)))
            self.compare(matrix(solver.forms[order]["pressure_a"]), matrix(fem.form(p*q*solver.mh.dx)))
            expected = fem.form((-solver.nu * ufl.div(solver.vtilde) + solver.phin + solver.po) * q * solver.mh.dx)
            self.compare(assemble_vector(solver.forms[order]["pressure_L"]), assemble_vector(expected))

    def test_bdf1_and_bdf2_steps_converge(self):
        for solver in self.solvers.values():
            for step in (1, 2):
                solver.solve_step(step * solver.dt, use_bdf1=(step == 1), compute_norms=False)
                self.assertTrue(np.isfinite(solver.pn.x.array).all())
                self.assertTrue(all(item["reason"] > 0 for item in solver.last_step_stats["substeps"].values()))

    def test_nonrotational_recovery_is_the_classical_update(self):
        cfg = configuration(16, .01, .02, "full")
        cfg.cfg["solver"]["pressure_recovery"] = "nonrotational"
        solver = initialize(cfg)
        p, q = ufl.TrialFunction(solver.Q), ufl.TestFunction(solver.Q)
        for order in (1, 2):
            self.compare(matrix(solver.forms[order]["pressure_a"]), matrix(fem.form(p*q*solver.mh.dx)))
            self.compare(assemble_vector(solver.forms[order]["pressure_L"]),
                         assemble_vector(fem.form((solver.po + solver.phin)*q*solver.mh.dx)))
            self.compare(matrix(solver.forms[order]["phi_a"]),
                         matrix(self.solvers["full"].forms[order]["phi_a"]))
        previous = fem.Function(solver.Q)
        for step in (1, 2):
            previous.x.array[:] = solver.po.x.array
            solver.solve_step(step*.01, use_bdf1=(step == 1), compute_norms=False)
            defect = solver.pn - previous - solver.phin
            self.assertLess(fem.assemble_scalar(fem.form(defect**2*solver.mh.dx)), 1.e-14)

    def test_pressure_errors_retain_constant_bias(self):
        solver = self.solvers["full"]
        solver.pn.x.array[:] += 2.
        metrics = compute_metrics(solver.mh, solver.vn, solver.pn)
        self.assertGreater(metrics["global_pressure_error"], 1.9)
        for error in (spatial_errors(solver.mh, solver.vn, solver.pn)[2],
                      temporal_errors(solver.mh, solver.vn, solver.pn, 0.)[2]):
            self.assertAlmostEqual(error, metrics["global_pressure_error"], places=12)
        _, _, plotted = pressure_error_field(solver)
        np.testing.assert_allclose(plotted, 2., rtol=0., atol=1.e-14)


class ReferenceTests(unittest.TestCase):
    def test_parallel_pressure_node_count(self):
        solver = initialize(configuration(64, .001, 1., "full"), MPI.COMM_WORLD)
        tri, pressure = gather_pressure_field(solver)
        if solver.comm.rank == 0:
            self.assertEqual(len(pressure), 65**2)
            self.assertEqual(len(tri.triangles), 2*64**2)
            expected = np.cos(np.pi*tri.x/2)*np.sin(np.pi*tri.y/2)
            np.testing.assert_allclose(pressure, expected, rtol=0., atol=1.e-14)

    def test_p2p1_reference_uses_unstabilized_monolithic_solver(self):
        group = next(group for group in GROUPS if group["velocity_degree"] == 2)
        solver, _, steps = run_case(group, .01, 4, .02)
        self.assertIsInstance(solver, MonolithicSUPGPSPGSolver)
        self.assertFalse(solver.use_stabilization)
        self.assertEqual(solver.V.ufl_element().degree, 2)
        self.assertEqual(solver.Q.ufl_element().degree, 1)
        self.assertEqual(steps, 2)
        self.assertGreater(solver.last_step_stats["linear_reason"], 0)
        for entry in solver.element_stabilization.entries.values():
            self.assertTrue(np.all(entry["tau_m"].x.array == 0.))
            self.assertTrue(np.all(entry["tau_c"].x.array == 0.))


if __name__ == "__main__":
    unittest.main()
