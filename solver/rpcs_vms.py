import time

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
from ufl import Identity, as_vector, div, dot, grad, inner, min_value, nabla_grad

from solver.benchmark import PointEvaluator
from solver.expression import ExpressionHandler
from solver.stabilization import ElementStabilization, resolve_stabilization_alpha


def _solve_projection_mass(ksp, matrix, rhs, solution, residual, rtol, atol):
    """Apply and verify the consistent mass inverse for velocity correction."""
    ksp.solve(rhs, solution)
    reason = ksp.getConvergedReason()
    if reason <= 0:
        raise RuntimeError(f"Velocity projection mass solve failed, PETSc reason = {reason}.")
    matrix.mult(solution, residual)
    residual.axpy(-1.0, rhs)
    error = residual.norm()
    tolerance = atol + rtol * rhs.norm()
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(
            f"Velocity projection mass residual {error:.3e} exceeds {tolerance:.3e}."
        )
    return error




class FlowSolver:
    def __init__(self, comm, mesh_handler, cfg):
        self.comm = comm
        self.mh = mesh_handler
        self.cfg = cfg
        self.solver_options = cfg.get_solver_options()
        self.force_monitor_options = cfg.get_force_monitor_options()

        self.nu, self.eq, self.stab = cfg.get_phys(self.mh.mesh)
        self.dt, self.T, self.bdf = cfg.get_time()
        self.use_stabilization = self.stab == "t"
        self.stabilization_alpha = self.solver_options.get(
            "stabilization_alpha", "auto"
        )
        self.stabilization_inverse_estimate = float(
            self.solver_options.get("stabilization_inverse_estimate", 36.0)
        )
        self.pressure_recovery_mode = str(
            self.solver_options.get("pressure_recovery", "rotational")
        ).lower()
        self.linear_rtol = float(self.solver_options.get("linear_rtol", 1.0e-8))
        self.linear_atol = float(self.solver_options.get("linear_atol", 1.0e-10))
        self.linear_max_it = int(self.solver_options.get("linear_max_it", 1000))
        self.gmres_restart = int(self.solver_options.get("gmres_restart", 100))
        self.force_monitor_enabled = bool(
            self.force_monitor_options.get("enabled", False)
        )
        self.force_monitor_frequency = int(
            self.force_monitor_options.get("frequency", 1)
        )
        self.force_output_file = str(
            self.force_monitor_options.get("file", "force_coefficients.csv")
        )
        self.last_step_stats = {}

        self.updaters = []
        self._validate_options()
        self.setup_spaces()
        self.element_stabilization = ElementStabilization(
            self.mh, self.nu, self.dt, self.stabilization_alpha,
            self.use_stabilization, self.stabilization_inverse_estimate,
        )
        self.setup_body_force()
        self.setup_monitoring_forms()
        self.setup_force_monitoring()

    def _validate_options(self):
        if self.eq not in {"NS", "Stokes"}:
            raise ValueError(f"Unknown equation: {self.eq}")
        if self.stab not in {"t", "f"}:
            raise ValueError(f"Unknown stabilization flag: {self.stab}")
        if self.bdf not in {1, 2}:
            raise ValueError(f"Unsupported BDF order: {self.bdf}")
        options = self.solver_options
        formulation = str(options.get("formulation", "volume_acceleration")).lower()
        if formulation != "volume_acceleration":
            raise ValueError(f"Unsupported RPCS-VMS formulation: {formulation}")
        resolve_stabilization_alpha(self.stabilization_alpha)
        required = {
            "velocity correction": options.get("velocity_update", True) is True,
            "consistent corrector mass": str(
                options.get("corrector_mass", "auto")
            ).lower() in {"auto", "consistent"},
            "pseudo stress": str(options.get("stress_form", "pseudo")).lower() == "pseudo",
            "omitted second-derivative test feedback": options.get(
                "second_derivative_test_terms", False
            ) is False,
        }
        coefficient_mode = str(options.get("stabilization_coefficient", "dynamic")).lower()
        if self.pressure_recovery_mode not in {"rotational", "nonrotational"}:
            raise ValueError(f"Unknown pressure recovery: {self.pressure_recovery_mode}")
        if coefficient_mode not in {"dynamic", "static"}:
            raise ValueError(f"Unknown stabilization coefficient mode: {coefficient_mode}")
        if self.eq == "NS" and self.use_stabilization:
            required["convective stabilization"] = options.get(
                "convective_stabilization", True
            ) is True
            required["dynamic stabilization coefficient"] = coefficient_mode == "dynamic"
        incompatible = [name for name, valid in required.items() if not valid]
        if incompatible:
            raise ValueError(
                "The derivation's RPCS-VMS method requires "
                + ", ".join(incompatible) + "."
            )

    def _viscous_bilinear(self, u, v):
        return self.nu * inner(grad(u), grad(v))

    def _viscous_strong_operator(self, u):
        return self.nu * self._vector_laplacian(u)

    def setup_spaces(self):
        velocity_degree = self.solver_options.get("velocity_degree", 1)
        self.V = fem.functionspace(
            self.mh.mesh, ("Lagrange", velocity_degree, (self.mh.gdim,))
        )
        pressure_degree = velocity_degree if self.use_stabilization else 1
        self.Q = fem.functionspace(self.mh.mesh, ("Lagrange", pressure_degree))

        self.vn = fem.Function(self.V)
        self.vo = fem.Function(self.V)
        self.voo = fem.Function(self.V)

        self.vtilde = fem.Function(self.V)

        self.pn = fem.Function(self.Q)
        self.po = fem.Function(self.Q)

        self.phin = fem.Function(self.Q)
        self.phio = fem.Function(self.Q)
        self.phioo = fem.Function(self.Q)

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
        diff_u = self.vn - self.vo
        diff_p = self.pn - self.po
        self.form_diff_u = fem.form(inner(diff_u, diff_u) * self.mh.dx)
        self.form_diff_p = fem.form(inner(diff_p, diff_p) * self.mh.dx)

    def setup_force_monitoring(self):
        self.drag_form = None
        self.lift_form = None
        self.pressure_evaluator = None
        self.force_scale = None
        self.force_reference_area = None
        if not self.force_monitor_enabled:
            return

        if self.mh.gdim < 2:
            raise ValueError("Force monitoring requires at least a 2D mesh.")

        pressure_points = self.force_monitor_options["pressure_points"]
        point_dim = len(pressure_points[0])
        if point_dim != self.mh.gdim:
            raise ValueError(
                "force_monitor.pressure_points dimension must match the mesh geometric dimension."
            )

        boundary_id = int(self.force_monitor_options["boundary_id"])
        characteristic_velocity = float(
            self.force_monitor_options["characteristic_velocity"]
        )
        characteristic_length = float(
            self.force_monitor_options["characteristic_length"]
        )
        viscosity = float(self.nu.value)

        reference_area = self.force_monitor_options.get("reference_area")
        if reference_area is not None:
            reference_area = float(reference_area)
        else:
            spanwise_length = self.force_monitor_options.get("spanwise_length")
            if spanwise_length is not None:
                reference_area = characteristic_length * float(spanwise_length)
            elif self.mh.gdim == 3:
                raise ValueError(
                    "force_monitor.reference_area or force_monitor.spanwise_length is required for 3D force monitoring."
                )
            else:
                reference_area = characteristic_length

        self.force_boundary_id = boundary_id
        self.force_reference_area = reference_area
        self.force_scale = 2.0 * viscosity / (characteristic_velocity**2 * reference_area)
        local_force_facets = len(self.mh.ft.find(boundary_id))
        global_force_facets = self.comm.allreduce(local_force_facets, op=MPI.SUM)
        if global_force_facets == 0:
            raise ValueError(
                f"force_monitor.boundary_id {boundary_id} was not found in the mesh tags."
            )

        # Use a unit-viscosity pseudo-stress tensor and normalize the pressure
        # contribution by nu so the reported coefficients remain consistent with
        # the physical drag/lift convention.
        sigma = 1.0 * grad(self.vn) - (self.pn / viscosity) * Identity(self.mh.gdim)
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

        pressure_difference = float(pressure_values[0, 0] - pressure_values[1, 0])
        sample = {
            "time": float(t),
            "drag_force": float(drag_force),
            "lift_force": float(lift_force),
            "drag_coefficient": float(self.force_scale * drag_force),
            "lift_coefficient": float(self.force_scale * lift_force),
            "pressure_difference": pressure_difference,
        }
        if step is not None:
            sample["step"] = int(step)
        return sample

    def set_bcs(self, boundary_data):
        self.velocity_bcs = boundary_data.velocity_bcs
        self.phi_bcs = boundary_data.phi_bcs
        self.traction_data = boundary_data.traction_data
        self.updaters.extend(boundary_data.updaters)
        self.setup_forms()
        self.setup_solvers()

    def _vector_laplacian(self, field):
        return ufl.as_vector(
            [ufl.div(ufl.grad(field[i])) for i in range(self.mh.gdim)]
        )

    def _scalar_laplacian(self, field):
        return ufl.div(ufl.grad(field))

    def _stabilization_coefficient_is_static(self):
        return not self.use_stabilization or self.eq == "Stokes"

    def _matrix_is_static(self, kind):
        if kind == "pressure" and self.pressure_recovery_mode == "nonrotational":
            return True
        if kind in {"phi", "pressure"}:
            return self._stabilization_coefficient_is_static()
        if kind == "predictor":
            return self.eq == "Stokes" and self._stabilization_coefficient_is_static()
        return False

    def _advective_term(self, advective_velocity, field):
        """Physical transport in the strong momentum residuals."""
        return dot(advective_velocity, nabla_grad(field))

    def _convective_term(self, advective_velocity, field):
        return (
            self._advective_term(advective_velocity, field)
            + 0.5 * div(advective_velocity) * field
        )

    def _history_terms(self, order):
        if order == 1:
            gamma0 = 1.0
            velocity_history = -self.vo
            phi_star = self.phio
            advective_velocity = self.vo if self.eq == "NS" else self.zero_velocity
        else:
            gamma0 = 1.5
            velocity_history = -2.0 * self.vo + 0.5 * self.voo
            phi_star = 2.0 * self.phio - self.phioo
            advective_velocity = (
                2.0 * self.vo - self.voo if self.eq == "NS" else self.zero_velocity
            )
        return gamma0, velocity_history, phi_star, advective_velocity

    def _build_scheme_forms(self, order):
        gamma0, velocity_history, phi_star, advective_velocity = (
            self._history_terms(order)
        )

        tau_m, tau_c = self.element_stabilization.coefficients(advective_velocity, gamma0)
        tau_phi = tau_m if self.solver_options.get("pressure_increment_stabilization", True) else 0.0
        tau_p = tau_m if self.solver_options.get("pressure_recovery_stabilization", True) else 0.0

        v_trial = ufl.TrialFunction(self.V)
        w = ufl.TestFunction(self.V)
        phi_trial = ufl.TrialFunction(self.Q)
        q_phi = ufl.TestFunction(self.Q)
        z = ufl.TestFunction(self.V)
        p_trial = ufl.TrialFunction(self.Q)
        q_p = ufl.TestFunction(self.Q)

        predictor_residual = (
            (gamma0 * v_trial + velocity_history) / self.dt
            + self._advective_term(advective_velocity, v_trial)
            - self._viscous_strong_operator(v_trial)
            + grad(self.po)
            - self.f_form
        )
        u_prime = -tau_m * predictor_residual
        phi_prime_predictor = -tau_c * (
            div(v_trial) - (self.dt / gamma0) * self._scalar_laplacian(phi_star)
        )

        predictor_form = (
            dot((gamma0 * v_trial + velocity_history) / self.dt, w) * self.mh.dx
            + dot(self._convective_term(advective_velocity, v_trial), w) * self.mh.dx
            + self._viscous_bilinear(v_trial, w) * self.mh.dx
            - self.po * div(w) * self.mh.dx
            - dot(self.f_form, w) * self.mh.dx
        )

        predictor_form -= (
            dot(self._convective_term(advective_velocity, w), u_prime) * self.mh.dx
        )

        predictor_form -= phi_prime_predictor * div(w) * self.mh.dx

        normal_speed = dot(advective_velocity, self.mh.n)
        for traction_entry in self.traction_data:
            marker_id, traction = traction_entry[:2]
            boundary_backflow_enabled = (
                traction_entry[2] if len(traction_entry) > 2 else True
            )
            predictor_form -= dot(traction, w) * self.mh.ds(marker_id)
            if (
                self.eq == "NS"
                and boundary_backflow_enabled
            ):
                predictor_form -= (
                    0.5
                    * min_value(normal_speed, 0.0)
                    * dot(v_trial, w)
                    * self.mh.ds(marker_id)
                )

        full_residual = (
            (gamma0 * self.vtilde + velocity_history) / self.dt
            + self._advective_term(advective_velocity, self.vtilde)
            - self._viscous_strong_operator(self.vtilde)
            + grad(self.po)
            - self.f_form
        )
        M_form = inner(v_trial, z) * self.mh.dx
        G_form = inner(grad(phi_trial), z) * self.mh.dx
        phi_a = (self.dt / gamma0 + tau_phi) * inner(grad(phi_trial), grad(q_phi)) * self.mh.dx
        b_phi_form = (
            -div(self.vtilde) * q_phi
            - tau_phi * inner(full_residual, grad(q_phi))
        ) * self.mh.dx

        curl_curl_vtilde = grad(div(self.vtilde)) - self._vector_laplacian(self.vtilde)
        acceleration = (gamma0 * self.vtilde + velocity_history) / self.dt
        pressure_residual = (
            acceleration
            + self._advective_term(advective_velocity, self.vtilde)
            + self.nu * curl_curl_vtilde
            - self.f_form
        )
        if self.pressure_recovery_mode == "nonrotational":
            pressure_a = p_trial * q_p * self.mh.dx
            pressure_L = (self.phin + self.po) * q_p * self.mh.dx
        else:
            pressure_a = (
                p_trial * q_p + self.nu * tau_p * dot(grad(p_trial), grad(q_p))
            ) * self.mh.dx
            pressure_L = (
                - self.nu * div(self.vtilde) * q_p * self.mh.dx
                - self.nu * tau_p * dot(pressure_residual, grad(q_p)) * self.mh.dx
                + (self.phin + self.po) * q_p * self.mh.dx
            )

        result = {
            "predictor_a": fem.form(ufl.lhs(predictor_form)),
            "predictor_L": fem.form(ufl.rhs(predictor_form)),
            "pressure_a": fem.form(pressure_a),
            "pressure_L": fem.form(pressure_L),
        }
        result["M_form"] = fem.form(M_form)
        result["G_form"] = fem.form(G_form)
        result["phi_a"] = fem.form(phi_a)
        result["b_phi_form"] = fem.form(b_phi_form)
        return result

    @staticmethod
    def _system_label(name):
        return f"{name} solve"

    def setup_forms(self):
        self.forms = {
            1: self._build_scheme_forms(order=1),
            2: self._build_scheme_forms(order=2),
        }
        for order in self.forms:
            self.element_stabilization.update(1.0 if order == 1 else 1.5)

    def _create_ksp(self, matrix, kind):
        ksp = PETSc.KSP().create(self.comm)
        ksp.setOperators(matrix)

        if kind == "predictor":
            if self.eq == "Stokes":
                ksp.setType("cg")
                ksp.getPC().setType("hypre")
                ksp.getPC().setHYPREType("boomeramg")
            else:
                ksp.setType("gmres")
                ksp.setGMRESRestart(self.gmres_restart)
                ksp.getPC().setType("hypre")
                ksp.getPC().setHYPREType("boomeramg")
        else:
            ksp.setType("cg")
            ksp.getPC().setType("hypre")
            ksp.getPC().setHYPREType("boomeramg")

        ksp.setTolerances(
            rtol=self.linear_rtol,
            atol=self.linear_atol,
            max_it=self.linear_max_it,
        )
        ksp.setInitialGuessNonzero(True)
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        if ksp.getType() == "gmres":
            ksp.setPCSide(PETSc.PC.Side.RIGHT)
        ksp.setFromOptions()
        return ksp

    def _create_system(self, bilinear_form, prototype, solver_kind, system_kind):
        matrix = create_matrix(bilinear_form)
        vector = prototype.x.petsc_vec.copy()
        ksp = self._create_ksp(matrix, solver_kind)
        return {
            "matrix": matrix,
            "vector": vector,
            "ksp": ksp,
            "matrix_static": self._matrix_is_static(system_kind),
            "matrix_assembled": False,
        }

    def setup_solvers(self):
        self.systems = {
            1: {
                "predictor": self._create_system(
                    self.forms[1]["predictor_a"],
                    self.vtilde,
                    "predictor",
                    "predictor",
                ),
                "pressure": self._create_system(
                    self.forms[1]["pressure_a"], self.pn, "elliptic", "pressure"
                ),
            },
            2: {
                "predictor": self._create_system(
                    self.forms[2]["predictor_a"],
                    self.vtilde,
                    "predictor",
                    "predictor",
                ),
                "pressure": self._create_system(
                    self.forms[2]["pressure_a"], self.pn, "elliptic", "pressure"
                ),
            },
        }

        self.phi_nullspace = None
        self.phi_operator_bcs = self.phi_bcs
        if not self.phi_bcs:
            self.phi_nullspace = PETSc.NullSpace().create(constant=True, comm=self.comm)
            self.phi_integral_form = fem.form(self.phin * self.mh.dx)
            one = fem.Constant(self.mh.mesh, PETSc.ScalarType(1.0))
            self.domain_volume = self.comm.allreduce(
                assemble_scalar(fem.form(one * self.mh.dx)), op=MPI.SUM
            )
            index_map = self.Q.dofmap.index_map
            local = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
            global_ids = index_map.local_to_global(local)
            gauge_dofs = local[global_ids == 0]
            # Solve with a nonsingular nodal gauge, then restore the L2 mean.
            self.phi_operator_bcs = [
                fem.dirichletbc(PETSc.ScalarType(0.0), gauge_dofs, self.Q)
            ]

        self._setup_projection()

    def _setup_projection(self):
        self.proj_mass_matrix = self._assemble_matrix_robust(
            self.forms[1]["M_form"], self.velocity_bcs
        )
        self.proj_mass_ksp = PETSc.KSP().create(self.comm)
        self.proj_mass_ksp.setOptionsPrefix("rpcs_projection_mass_")
        self.proj_mass_ksp.setOperators(self.proj_mass_matrix)
        self.proj_mass_ksp.setType("cg")
        mass_pc = self.proj_mass_ksp.getPC()
        mass_pc.setType("jacobi")
        self.proj_mass_rtol = self.linear_rtol
        self.proj_mass_atol = self.linear_atol
        self.proj_mass_ksp.setTolerances(
            rtol=self.proj_mass_rtol, atol=self.proj_mass_atol,
            max_it=self.linear_max_it,
        )
        self.proj_mass_ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        self.proj_mass_ksp.setInitialGuessNonzero(False)
        self.proj_mass_ksp.setFromOptions()
        self.proj_mass_ksp.setUp()
        self.proj_G = self._assemble_matrix_robust(
            self.forms[1]["G_form"], self.velocity_bcs + self.phi_bcs
        )
        self.proj_t = self.proj_G.createVecLeft()
        self.proj_z = self.proj_G.createVecLeft()
        self.proj_mass_residual = self.proj_G.createVecLeft()
        self.proj_phi_system = None

    def _normalize_phi_mean(self):
        if self.phi_nullspace is not None:
            integral = self.comm.allreduce(assemble_scalar(self.phi_integral_form), op=MPI.SUM)
            self.phin.x.array[:] -= integral / self.domain_volume
            self.phin.x.scatter_forward()

    def _assemble_matrix_robust(self, form, bcs):
        matrix = create_matrix(form)
        matrix.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        assemble_matrix(matrix, form, bcs=bcs)
        matrix.assemble()
        return matrix

    def _rebuild_phi_operator(self, order):
        delta = PETSc.ScalarType(self.dt / (1.0 if order == 1 else 1.5))
        if not hasattr(self, "proj_phi_operators"):
            self.proj_phi_operators = {}
        system = self.proj_phi_operators.get(order)
        if system is None:
            matrix = self._assemble_matrix_robust(
                self.forms[order]["phi_a"], self.phi_operator_bcs
            )
            ksp = PETSc.KSP().create(self.comm)
            ksp.setType("cg")
            ksp.setInitialGuessNonzero(True)
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            ksp.setTolerances(
                rtol=self.linear_rtol, atol=self.linear_atol,
                max_it=self.linear_max_it,
            )
            ksp.getPC().setType("hypre")
            ksp.getPC().setHYPREType("boomeramg")
            ksp.setOperators(matrix)
            ksp.setFromOptions()
            system = {"order": order, "delta": delta, "A": matrix, "ksp": ksp}
            self.proj_phi_operators[order] = system
        elif not self._stabilization_coefficient_is_static():
            matrix = system["A"]
            matrix.zeroEntries()
            assemble_matrix(
                matrix, self.forms[order]["phi_a"], bcs=self.phi_operator_bcs
            )
            matrix.assemble()
            system["ksp"].setOperators(matrix)
        self.proj_phi_system = system

    def _solve_phi(self, order):
        step_start = time.perf_counter()
        self._rebuild_phi_operator(order)
        system = self.proj_phi_system

        assembly_start = time.perf_counter()
        b = self.phin.x.petsc_vec.copy()
        with b.localForm() as local_b:
            local_b.set(0)
        assemble_vector(b, self.forms[order]["b_phi_form"])
        b.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )
        if self.phi_nullspace is not None:
            self.phi_nullspace.remove(b)
            start, end = self.Q.dofmap.index_map.local_range
            owns_anchor = start == 0 and end > start
            local_anchor = self.phin.x.array[0] if owns_anchor else 0.0
            anchor = self.comm.allreduce(local_anchor, op=MPI.SUM)
            self.phin.x.array[:] -= anchor
            self.phin.x.scatter_forward()
        set_bc(b, self.phi_operator_bcs)
        assembly_time = time.perf_counter() - assembly_start

        solve_start = time.perf_counter()
        ksp = system["ksp"]
        ksp.solve(b, self.phin.x.petsc_vec)
        solve_time = time.perf_counter() - solve_start
        self._check_convergence(ksp, self._system_label("phi"))
        self.phin.x.scatter_forward()
        self._normalize_phi_mean()
        return {
            "assembly_time": assembly_time,
            "solve_time": solve_time,
            "total_time": time.perf_counter() - step_start,
            "iterations": int(ksp.getIterationNumber()),
            "reason": int(ksp.getConvergedReason()),
        }

    def _solve_corrector(self, order):
        step_start = time.perf_counter()
        delta = self.proj_phi_system["delta"]
        self.proj_G.mult(self.phin.x.petsc_vec, self.proj_t)
        solve_start = time.perf_counter()
        _solve_projection_mass(
            self.proj_mass_ksp, self.proj_mass_matrix, self.proj_t, self.proj_z,
            self.proj_mass_residual, self.proj_mass_rtol, self.proj_mass_atol,
        )
        solve_time = time.perf_counter() - solve_start
        vn = self.vn.x.petsc_vec
        vn.zeroEntries()
        vn.axpy(1.0, self.vtilde.x.petsc_vec)
        vn.axpy(-delta, self.proj_z)
        self.vn.x.scatter_forward()
        return {
            "assembly_time": 0.0,
            "solve_time": solve_time,
            "total_time": time.perf_counter() - step_start,
            "iterations": int(self.proj_mass_ksp.getIterationNumber()),
            "reason": int(self.proj_mass_ksp.getConvergedReason()),
        }

    def _check_convergence(self, ksp, label):
        reason = ksp.getConvergedReason()
        if reason <= 0:
            raise RuntimeError(
                f"{label} failed to converge, PETSc reason = {reason}, iterations = {ksp.getIterationNumber()}"
            )

    def effective_solver_configuration(self, order=2):
        systems = self.systems[order]
        self._rebuild_phi_operator(order)
        systems = {**systems, "phi": self.proj_phi_system}
        configuration = {}
        for name in ("predictor", "phi", "pressure"):
            ksp = systems[name]["ksp"]
            configuration[f"{name}_ksp_type"] = ksp.getType()
            configuration[f"{name}_pc_type"] = ksp.getPC().getType()
        configuration["corrector_ksp_type"] = self.proj_mass_ksp.getType()
        configuration["corrector_pc_type"] = self.proj_mass_ksp.getPC().getType()
        configuration["outer_residual_norm"] = "unpreconditioned" if systems["predictor"]["ksp"].getNormType() == PETSc.KSP.NormType.UNPRECONDITIONED else "other"
        configuration["outer_preconditioning_side"] = "right" if systems["predictor"]["ksp"].getPCSide() == PETSc.PC.Side.RIGHT else "left"

        return configuration

    def _assemble_and_solve(
        self,
        system,
        bilinear_form,
        linear_form,
        solution,
        label,
        bcs=None,
    ):
        step_start = time.perf_counter()
        assembly_start = time.perf_counter()
        matrix = system["matrix"]
        vector = system["vector"]
        ksp = system["ksp"]

        if not system["matrix_static"] or not system["matrix_assembled"]:
            matrix.zeroEntries()
            assemble_matrix(matrix, bilinear_form, bcs=bcs or [])
            matrix.assemble()
            system["matrix_assembled"] = True

        with vector.localForm() as local_vector:
            local_vector.set(0)
        assemble_vector(vector, linear_form)
        if bcs:
            apply_lifting(vector, [bilinear_form], [bcs])
        vector.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )
        if bcs:
            set_bc(vector, bcs)
        assembly_time = time.perf_counter() - assembly_start

        solve_start = time.perf_counter()
        ksp.solve(vector, solution.x.petsc_vec)
        solve_time = time.perf_counter() - solve_start
        self._check_convergence(ksp, label)
        solution.x.scatter_forward()
        return {
            "assembly_time": assembly_time,
            "solve_time": solve_time,
            "total_time": time.perf_counter() - step_start,
            "iterations": int(ksp.getIterationNumber()),
            "reason": int(ksp.getConvergedReason()),
        }

    def solve_step(self, t, use_bdf1, compute_norms=True):
        step_start = time.perf_counter()
        self.time_symbol.value = PETSc.ScalarType(t)
        for update in self.updaters:
            update(t)

        order = 1 if use_bdf1 or self.bdf == 1 else 2
        self.element_stabilization.update(1.0 if order == 1 else 1.5)
        forms = self.forms[order]
        systems = self.systems[order]

        predictor_stats = self._assemble_and_solve(
            systems["predictor"],
            forms["predictor_a"],
            forms["predictor_L"],
            self.vtilde,
            self._system_label("predictor"),
            bcs=self.velocity_bcs,
        )
        phi_stats = self._solve_phi(order)

        corrector_stats = self._solve_corrector(order)

        pressure_stats = self._assemble_and_solve(
            systems["pressure"],
            forms["pressure_a"],
            forms["pressure_L"],
            self.pn,
            self._system_label("pressure"),
        )

        if compute_norms:
            global_diff_u2 = self.comm.allreduce(
                assemble_scalar(self.form_diff_u), op=MPI.SUM
            )
            global_diff_p2 = self.comm.allreduce(
                assemble_scalar(self.form_diff_p), op=MPI.SUM
            )
            norm_diff_u = np.sqrt(global_diff_u2)
            norm_diff_p = np.sqrt(global_diff_p2)
        else:
            norm_diff_u = np.nan
            norm_diff_p = np.nan

        self.voo.x.array[:] = self.vo.x.array[:]
        self.vo.x.array[:] = self.vn.x.array[:]

        self.po.x.array[:] = self.pn.x.array[:]
        self.phioo.x.array[:] = self.phio.x.array[:]
        self.phio.x.array[:] = self.phin.x.array[:]

        substeps = {
            "predictor": predictor_stats,
            "phi": phi_stats,
            "corrector": corrector_stats,
            "pressure": pressure_stats,
        }
        self.last_step_stats = {
            "order": order,
            "step_time": time.perf_counter() - step_start,
            "linear_iterations": int(
                sum(item["iterations"] for item in substeps.values())
            ),
            "substeps": substeps,
        }

        return norm_diff_u, norm_diff_p
