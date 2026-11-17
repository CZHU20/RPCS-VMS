import time

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem import assemble_scalar
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    set_bc,
)
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    Identity,
    TestFunctions,
    TrialFunctions,
    as_vector,
    div,
    dot,
    grad,
    inner,
    min_value,
    nabla_grad,
    sym,
)

from solver.benchmark import PointEvaluator
from solver.expression import ExpressionHandler
from solver.stabilization import ElementStabilization


class MonolithicSUPGPSPGSolver:
    """Monolithic Oseen solver with optional SUPG/PSPG and grad-div control."""

    def __init__(self, comm, mesh_handler, cfg):
        self.comm = comm
        self.mh = mesh_handler
        self.cfg = cfg
        self.solver_options = cfg.get_solver_options()
        self.velocity_degree = int(self.solver_options.get("velocity_degree", 1))
        self.force_monitor_options = cfg.get_force_monitor_options()

        self.nu, self.eq, self.stab = cfg.get_phys(self.mh.mesh)
        self.dt, self.T, self.bdf = cfg.get_time()
        self.use_stabilization = self.stab == "t"
        self.convective_stabilization_enabled = bool(
            self.solver_options.get("convective_stabilization", True)
        )
        self.stabilization_coefficient_mode = str(
            self.solver_options.get("stabilization_coefficient", "dynamic")
        ).lower()
        self.stabilization_alpha = self.solver_options.get(
            "stabilization_alpha", "auto"
        )
        self.stress_form = str(self.solver_options.get("stress_form", "pseudo")).lower()
        self.linear_rtol = float(self.solver_options.get("linear_rtol", 1.0e-8))
        self.linear_atol = float(self.solver_options.get("linear_atol", 1.0e-10))
        self.linear_max_it = int(self.solver_options.get("linear_max_it", 1000))
        self.gmres_restart = int(self.solver_options.get("gmres_restart", 100))
        self.ksp_type = str(
            self.solver_options.get("monolithic_ksp_type", "gmres")
        ).lower()
        self.requested_pc_type = str(
            self.solver_options.get("monolithic_pc", "schur_amg")
        ).lower()
        self.preconditioner_matrix_mode = str(
            self.solver_options.get(
                "monolithic_preconditioner_matrix", "augmented"
            )
        ).lower()
        self.pc_type = self.requested_pc_type
        if self.pc_type in {"pcd", "schur_amg"}:
            self.pc_type = "fieldsplit"
        self.schur_fact_type = str(
            self.solver_options.get("monolithic_schur_fact_type", "lower")
        ).lower()

        self.force_monitor_enabled = bool(
            self.force_monitor_options.get("enabled", False)
        )
        self.force_monitor_frequency = int(
            self.force_monitor_options.get("frequency", 1)
        )
        self.force_output_file = str(
            self.force_monitor_options.get("file", "force_coefficients.csv")
        )

        self.updaters = []
        self.last_step_stats = {}
        self._validate_options()
        self.setup_spaces()
        self.element_stabilization = ElementStabilization(
            self.mh, self.nu, self.dt, self.stabilization_alpha,
            self.use_stabilization,
            self.solver_options.get("stabilization_inverse_estimate", 36.0),
        )
        self.setup_body_force()
        self.setup_monitoring_forms()
        self.setup_force_monitoring()

    def _validate_options(self):
        if self.velocity_degree < 1:
            raise ValueError("velocity_degree must be positive.")
        if self.eq not in {"NS", "Stokes"}:
            raise ValueError(f"Unknown equation: {self.eq}")
        if self.stab not in {"t", "f"}:
            raise ValueError(f"Unknown stabilization flag: {self.stab}")
        if self.bdf not in {1, 2}:
            raise ValueError(f"Unsupported BDF order: {self.bdf}")
        if self.stabilization_coefficient_mode not in {"dynamic", "static"}:
            raise ValueError(
                "Unknown stabilization coefficient mode: "
                f"{self.stabilization_coefficient_mode}"
            )
        if self.stabilization_alpha != "auto" and float(self.stabilization_alpha) < 0.0:
            raise ValueError("stabilization_alpha must be 'auto' or non-negative.")
        if self.stress_form not in {"pseudo", "physical"}:
            raise ValueError(f"Unknown stress form: {self.stress_form}")
        if self.ksp_type not in {"gmres", "fgmres"}:
            raise ValueError(f"Unknown monolithic_ksp_type: {self.ksp_type}")
        if self.pc_type not in {
            "fieldsplit",
            "lu",
            "asm_lu",
            "asm_ilu",
            "hypre",
            "jacobi",
        }:
            raise ValueError(f"Unknown monolithic_pc: {self.pc_type}")
        if self.preconditioner_matrix_mode not in {"incremental", "augmented"}:
            raise ValueError(
                "monolithic_preconditioner_matrix must be 'incremental' or 'augmented'."
            )
        if self.schur_fact_type not in {"diag", "lower", "upper", "full"}:
            raise ValueError(f"Unknown monolithic_schur_fact_type: {self.schur_fact_type}")

    def setup_spaces(self):
        cell = self.mh.mesh.basix_cell()
        velocity_el = basix.ufl.element("Lagrange", cell, self.velocity_degree, shape=(self.mh.gdim,))
        pressure_el = basix.ufl.element("Lagrange", cell, 1)
        self.W = fem.functionspace(
            self.mh.mesh, basix.ufl.mixed_element([velocity_el, pressure_el])
        )
        self.V, self.velocity_parent_dofs = self.W.sub(0).collapse()
        self.Q, self.pressure_parent_dofs = self.W.sub(1).collapse()
        self.mixed_velocity_space = self.W.sub(0)

        self.w = fem.Function(self.W)
        self.w_old = fem.Function(self.W)
        self.w_older = fem.Function(self.W)
        self.u_adv = fem.Function(self.V)

        self.vn = fem.Function(self.V)
        self.vo = fem.Function(self.V)
        self.voo = fem.Function(self.V)
        self.pn = fem.Function(self.Q)
        self.po = fem.Function(self.Q)

        self.zero_velocity = fem.Constant(
            self.mh.mesh, np.zeros(self.mh.gdim, dtype=PETSc.ScalarType)
        )
        self.time_symbol = fem.Constant(self.mh.mesh, PETSc.ScalarType(0.0))

    def setup_body_force(self):
        body_force_cfg = self.cfg.cfg.get("body_force")
        expr_handler = ExpressionHandler()
        if body_force_cfg:
            self.f, updater = expr_handler.create_function(
                self.V,
                body_force_cfg["value"],
                body_force_cfg.get("form", "constant"),
            )
            if updater is not None:
                self.updaters.append(updater)
            if body_force_cfg.get("form", "constant") == "expression":
                self.f_form = expr_handler.create_exact_expression(
                    self.mh.mesh,
                    body_force_cfg["value"],
                    body_force_cfg.get("form", "constant"),
                    time_symbol=self.time_symbol,
                )
            else:
                self.f_form = self.f
        else:
            self.f = fem.Function(self.V)
            self.f.x.array[:] = 0.0
            self.f.x.scatter_forward()
            self.f_form = self.f

    def setup_monitoring_forms(self):
        self.form_diff_u = fem.form(
            inner(self.vn - self.vo, self.vn - self.vo) * self.mh.dx
        )
        self.form_diff_p = fem.form(
            inner(self.pn - self.po, self.pn - self.po) * self.mh.dx
        )

    def setup_force_monitoring(self):
        self.drag_form = None
        self.lift_form = None
        self.pressure_evaluator = None
        self.force_scale = None
        if not self.force_monitor_enabled:
            return

        pressure_points = self.force_monitor_options["pressure_points"]
        boundary_id = int(self.force_monitor_options["boundary_id"])
        characteristic_velocity = float(
            self.force_monitor_options["characteristic_velocity"]
        )
        characteristic_length = float(
            self.force_monitor_options["characteristic_length"]
        )
        viscosity = float(self.nu.value)

        reference_area = self.force_monitor_options.get("reference_area")
        if reference_area is None:
            spanwise_length = self.force_monitor_options.get("spanwise_length")
            reference_area = (
                characteristic_length * float(spanwise_length)
                if spanwise_length is not None
                else characteristic_length
            )
        reference_area = float(reference_area)

        local_force_facets = len(self.mh.ft.find(boundary_id))
        global_force_facets = self.comm.allreduce(local_force_facets, op=MPI.SUM)
        if global_force_facets == 0:
            raise ValueError(
                f"force_monitor.boundary_id {boundary_id} was not found in the mesh tags."
            )

        self.force_scale = (
            2.0 * viscosity / (characteristic_velocity**2 * reference_area)
        )
        sigma = grad(self.vn) - (self.pn / viscosity) * Identity(self.mh.gdim)
        drag_direction = as_vector(
            tuple(PETSc.ScalarType(1.0 if i == 0 else 0.0) for i in range(self.mh.gdim))
        )
        lift_direction = as_vector(
            tuple(PETSc.ScalarType(1.0 if i == 1 else 0.0) for i in range(self.mh.gdim))
        )
        self.drag_form = fem.form(
            -dot(dot(sigma, self.mh.n), drag_direction) * self.mh.ds(boundary_id)
        )
        self.lift_form = fem.form(
            -dot(dot(sigma, self.mh.n), lift_direction) * self.mh.ds(boundary_id)
        )
        self.pressure_evaluator = PointEvaluator(self.mh, pressure_points, self.comm)

    def evaluate_force_observables(self, t, step=None):
        if not self.force_monitor_enabled:
            return None

        drag_force = self.comm.allreduce(assemble_scalar(self.drag_form), op=MPI.SUM)
        lift_force = self.comm.allreduce(assemble_scalar(self.lift_form), op=MPI.SUM)
        pressure_values = self.pressure_evaluator.evaluate(self.pn)
        if self.comm.rank != 0:
            return None

        sample = {
            "time": float(t),
            "drag_force": float(drag_force),
            "lift_force": float(lift_force),
            "drag_coefficient": float(self.force_scale * drag_force),
            "lift_coefficient": float(self.force_scale * lift_force),
            "pressure_difference": float(pressure_values[0, 0] - pressure_values[1, 0]),
        }
        if step is not None:
            sample["step"] = int(step)
        return sample

    def _viscous_bilinear(self, u, v):
        if self.stress_form == "physical":
            return 2.0 * self.nu * inner(sym(grad(u)), sym(grad(v)))
        return self.nu * inner(grad(u), grad(v))

    def _viscous_strong_operator(self, u):
        if self.stress_form == "physical":
            return 2.0 * self.nu * div(sym(grad(u)))
        return self.nu * as_vector(
            [ufl.div(ufl.grad(u[i])) for i in range(self.mh.gdim)]
        )


    def _advective_term(self, advective_velocity, field):
        """Physical transport in the strong momentum residual."""
        return dot(advective_velocity, nabla_grad(field))

    def _convective_term(self, advective_velocity, field):
        return (self._advective_term(advective_velocity, field)
                + 0.5 * div(advective_velocity) * field)

    def set_bcs(self, boundary_data):
        self.velocity_bcs = boundary_data.velocity_bcs
        self.traction_data = boundary_data.traction_data
        self.velocity_boundary_ids = boundary_data.velocity_boundary_ids
        self.updaters.extend(boundary_data.updaters)
        self.setup_forms()
        self.setup_solver()

    def _build_scheme_forms(self, order):
        u_trial, p_trial = TrialFunctions(self.W)
        v_test, q_test = TestFunctions(self.W)
        u_old, _ = ufl.split(self.w_old)
        u_older, _ = ufl.split(self.w_older)

        if order == 1:
            gamma0 = 1.0
            velocity_history = -u_old
        else:
            gamma0 = 1.5
            velocity_history = -2.0 * u_old + 0.5 * u_older
        acceleration = (gamma0 * u_trial + velocity_history) / self.dt

        if self.eq == "NS":
            convective = self._convective_term(self.u_adv, u_trial)
            strong_transport = self._advective_term(self.u_adv, u_trial)
        else:
            convective = self.zero_velocity
            strong_transport = self.zero_velocity

        momentum_residual = (
            acceleration
            + strong_transport
            - self._viscous_strong_operator(u_trial)
            + grad(p_trial)
            - self.f_form
        )
        stabilization_velocity = (
            self.u_adv if self.eq == "NS" and
            self.stabilization_coefficient_mode == "dynamic" else self.zero_velocity
        )
        tau_m, tau_c = self.element_stabilization.coefficients(
            stabilization_velocity, gamma0
        )

        stabilization_test = grad(q_test)
        if self.eq == "NS" and self.convective_stabilization_enabled:
            stabilization_test += self._convective_term(self.u_adv, v_test)

        form = (
            dot(acceleration, v_test) * self.mh.dx
            + dot(convective, v_test) * self.mh.dx
            + self._viscous_bilinear(u_trial, v_test) * self.mh.dx
            - p_trial * div(v_test) * self.mh.dx
            - dot(self.f_form, v_test) * self.mh.dx
            + q_test * div(u_trial) * self.mh.dx
            + tau_m * dot(momentum_residual, stabilization_test) * self.mh.dx
            + tau_c * div(u_trial) * div(v_test) * self.mh.dx
        )

        normal_speed = dot(self.u_adv, self.mh.n)
        for traction_entry in self.traction_data:
            marker_id, traction = traction_entry[:2]
            backflow_enabled = traction_entry[2] if len(traction_entry) > 2 else True
            form -= dot(traction, v_test) * self.mh.ds(marker_id)
            if self.eq == "NS" and backflow_enabled:
                form -= (
                    0.5
                    * min_value(normal_speed, 0.0)
                    * dot(u_trial, v_test)
                    * self.mh.ds(marker_id)
                )

        bilinear_form = ufl.lhs(form)
        schur_grad_scale = PETSc.ScalarType(self.dt / gamma0)
        schur_mass_scale = PETSc.ScalarType(self.dt)
        preconditioner_form = (
            bilinear_form
            + schur_grad_scale * dot(grad(p_trial), grad(q_test)) * self.mh.dx
            + schur_mass_scale * p_trial * q_test * self.mh.dx
        )
        pressure_augmentation = (
            schur_grad_scale * dot(grad(p_trial), grad(q_test))
            + schur_mass_scale * p_trial * q_test
        ) * self.mh.dx

        return {
            "a": fem.form(bilinear_form),
            "P": fem.form(preconditioner_form),
            "P_augmentation": fem.form(pressure_augmentation),
            "L": fem.form(ufl.rhs(form)),
        }

    def setup_forms(self):
        self.forms = {1: self._build_scheme_forms(1), 2: self._build_scheme_forms(2)}

    def setup_solver(self):
        self.A = create_matrix(self.forms[1]["a"])
        preconditioner_form = self.forms[1]["P"]
        self.P = create_matrix(preconditioner_form)
        assemble_matrix(self.P, preconditioner_form, bcs=self.velocity_bcs)
        self.P.assemble()
        self.b = self.w.x.petsc_vec.copy()
        self.ksp = PETSc.KSP().create(self.comm)
        self.ksp.setOptionsPrefix("mono_supg_pspg_")
        self.ksp.setType(self.ksp_type)
        if self.ksp_type in {"gmres", "fgmres"}:
            self.ksp.setGMRESRestart(self.gmres_restart)
            self.ksp.setPCSide(PETSc.PC.Side.RIGHT)
            self.ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        self.ksp.setTolerances(
            rtol=self.linear_rtol, atol=self.linear_atol, max_it=self.linear_max_it
        )
        self.ksp.setErrorIfNotConverged(True)
        pc = self.ksp.getPC()
        if self.pc_type == "fieldsplit":
            owned_size = self.W.dofmap.index_map.size_local * self.W.dofmap.index_map_bs
            velocity_local = np.asarray(self.velocity_parent_dofs, dtype=np.int32)
            pressure_local = np.asarray(self.pressure_parent_dofs, dtype=np.int32)
            velocity_local = velocity_local[velocity_local < owned_size]
            pressure_local = pressure_local[pressure_local < owned_size]
            velocity_global = self.W.dofmap.index_map.local_to_global(velocity_local)
            pressure_global = self.W.dofmap.index_map.local_to_global(pressure_local)
            velocity_global = np.unique(np.sort(velocity_global)).astype(PETSc.IntType)
            pressure_global = np.unique(np.sort(pressure_global)).astype(PETSc.IntType)
            self.velocity_is = PETSc.IS().createGeneral(velocity_global, comm=self.comm)
            self.pressure_is = PETSc.IS().createGeneral(pressure_global, comm=self.comm)
            pc.setType("fieldsplit")
            pc.setFieldSplitIS(("0", self.velocity_is), ("1", self.pressure_is))
            opts = PETSc.Options()
            prefix = self.ksp.getOptionsPrefix()
            opts[f"{prefix}pc_fieldsplit_type"] = "schur"
            opts[f"{prefix}pc_fieldsplit_schur_fact_type"] = self.schur_fact_type
            opts[f"{prefix}pc_fieldsplit_schur_precondition"] = "a11"
            opts[f"{prefix}fieldsplit_0_ksp_type"] = "preonly"
            opts[f"{prefix}fieldsplit_0_pc_type"] = "hypre"
            opts[f"{prefix}fieldsplit_0_pc_hypre_type"] = "boomeramg"
            opts[f"{prefix}fieldsplit_1_ksp_type"] = "preonly"
            opts[f"{prefix}fieldsplit_1_pc_type"] = "hypre"
            opts[f"{prefix}fieldsplit_1_pc_hypre_type"] = "boomeramg"
        elif self.pc_type == "lu":
            pc.setType("lu")
            pc.setFactorSolverType("mumps")
        elif self.pc_type in {"asm_lu", "asm_ilu"}:
            pc.setType("asm")
            opts = PETSc.Options()
            prefix = self.ksp.getOptionsPrefix() or ""
            opts[f"{prefix}sub_ksp_type"] = "preonly"
            opts[f"{prefix}sub_pc_type"] = "lu" if self.pc_type == "asm_lu" else "ilu"
        elif self.pc_type == "hypre":
            pc.setType("hypre")
            pc.setHYPREType("boomeramg")
        else:
            pc.setType("jacobi")
        self.ksp.setFromOptions()

    def effective_solver_configuration(self):
        configuration = {
            "stabilization_form": "SUPG/PSPG+grad-div" if self.use_stabilization else "Galerkin",
            "velocity_degree": self.velocity_degree,
            "pressure_degree": 1,
            "stabilization_coefficient_mode": self.stabilization_coefficient_mode,
            "stabilization_alpha": str(self.stabilization_alpha),
            "preconditioner_matrix_mode": self.preconditioner_matrix_mode,
            "schur_fact_type": self.schur_fact_type,
            "outer_ksp_type": self.ksp.getType(),
            "outer_residual_norm": "unpreconditioned" if self.ksp.getNormType() == PETSc.KSP.NormType.UNPRECONDITIONED else "other",
            "outer_preconditioning_side": "right" if self.ksp.getPCSide() == PETSc.PC.Side.RIGHT else "left",
            "pc_type": self.ksp.getPC().getType(),
            "fieldsplit_0_ksp_type": "",
            "fieldsplit_0_pc_type": "",
            "fieldsplit_1_ksp_type": "",
            "fieldsplit_1_pc_type": "",
        }
        if self.pc_type == "fieldsplit":
            self.ksp.setUp()
            subksps = self.ksp.getPC().getFieldSplitSubKSP()
            if len(subksps) != 2:
                raise RuntimeError(
                    f"Expected two monolithic field splits, got {len(subksps)}."
                )
            for index, subksp in enumerate(subksps):
                configuration[f"fieldsplit_{index}_ksp_type"] = subksp.getType()
                configuration[f"fieldsplit_{index}_pc_type"] = subksp.getPC().getType()

            expected = {
                "outer_ksp_type": "gmres",
                "pc_type": "fieldsplit",
                "fieldsplit_0_ksp_type": "preonly",
                "fieldsplit_0_pc_type": "hypre",
                "fieldsplit_1_ksp_type": "preonly",
                "fieldsplit_1_pc_type": "hypre",
            }
            mismatches = {
                key: (configuration[key], value)
                for key, value in expected.items()
                if configuration[key] != value
            }
            if mismatches:
                raise RuntimeError(
                    "Unexpected effective monolithic PETSc configuration: "
                    f"{mismatches}."
                )
        return configuration

    def _update_advective_velocity(self, order):
        if order == 1:
            self.u_adv.x.array[:] = self.vo.x.array[:]
        else:
            self.u_adv.x.array[:] = 2.0 * self.vo.x.array[:] - self.voo.x.array[:]
        self.u_adv.x.scatter_forward()

    def _update_public_fields(self):
        self.vn.interpolate(self.w.sub(0))
        self.vn.x.scatter_forward()
        self.pn.interpolate(self.w.sub(1))
        self.pn.x.scatter_forward()

    def _assemble_linear_system(self, forms):
        self.A.zeroEntries()
        assemble_matrix(self.A, forms["a"], bcs=self.velocity_bcs)
        self.A.assemble()

        if self.preconditioner_matrix_mode == "incremental":
            # Reuse the common Oseen block and assemble only the Schur augmentation.
            self.P.zeroEntries()
            self.P.axpy(
                PETSc.ScalarType(1.0),
                self.A,
                structure=PETSc.Mat.Structure.SAME_NONZERO_PATTERN,
            )
            assemble_matrix(self.P, forms["P_augmentation"])
            self.P.assemble()
        else:
            self.P.zeroEntries()
            assemble_matrix(self.P, forms["P"], bcs=self.velocity_bcs)
            self.P.assemble()
        self.ksp.setOperators(self.A, self.P)

        with self.b.localForm() as local_b:
            local_b.set(0)
        assemble_vector(self.b, forms["L"])
        apply_lifting(self.b, [forms["a"]], [self.velocity_bcs])
        self.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        set_bc(self.b, self.velocity_bcs)

    def _advance_history(self):
        self.voo.x.array[:] = self.vo.x.array[:]
        self.vo.x.array[:] = self.vn.x.array[:]
        self.vo.x.scatter_forward()
        self.voo.x.scatter_forward()
        self.po.x.array[:] = self.pn.x.array[:]
        self.po.x.scatter_forward()
        self.w_older.x.array[:] = self.w_old.x.array[:]
        self.w_old.x.array[:] = self.w.x.array[:]
        self.w_older.x.scatter_forward()
        self.w_old.x.scatter_forward()

    def solve_step(self, t, use_bdf1=None, compute_norms=True):
        step_start = time.perf_counter()
        self.time_symbol.value = PETSc.ScalarType(t)
        for update in self.updaters:
            update(t)
        order = 1 if use_bdf1 or self.bdf == 1 else 2
        forms = self.forms[order]
        self._update_advective_velocity(order)
        self.element_stabilization.update(1.0 if order == 1 else 1.5)

        assembly_start = time.perf_counter()
        self._assemble_linear_system(forms)
        assembly_time = time.perf_counter() - assembly_start

        solve_start = time.perf_counter()
        self.ksp.solve(self.b, self.w.x.petsc_vec)
        solve_time = time.perf_counter() - solve_start
        reason = self.ksp.getConvergedReason()
        iterations = int(self.ksp.getIterationNumber())
        if reason <= 0:
            raise RuntimeError(
                "Monolithic SUPG/PSPG linear solve failed, "
                f"reason={reason}, iterations={iterations}."
            )
        self.w.x.scatter_forward()

        self._update_public_fields()
        if compute_norms:
            global_diff_u2 = self.comm.allreduce(
                assemble_scalar(self.form_diff_u), op=MPI.SUM
            )
            global_diff_p2 = self.comm.allreduce(
                assemble_scalar(self.form_diff_p), op=MPI.SUM
            )
            norm_diff_u = float(np.sqrt(global_diff_u2))
            norm_diff_p = float(np.sqrt(global_diff_p2))
        else:
            norm_diff_u = np.nan
            norm_diff_p = np.nan

        self._advance_history()

        self.last_step_stats = {
            "order": order,
            "step_time": time.perf_counter() - step_start,
            "assembly_time": assembly_time,
            "solve_time": solve_time,
            "nonlinear_iterations": 0,
            "linear_iterations": iterations,
            "outer_ksp_iterations": iterations,
            "reason": int(reason),
            "linear_reason": int(reason),
            "pc": self.requested_pc_type,
        }
        return norm_diff_u, norm_diff_p
