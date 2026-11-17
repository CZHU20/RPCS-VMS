import copy
import math
from dolfinx.fem import Constant
from petsc4py import PETSc


class CaseConfig:
    TOP_LEVEL_KEYS = {
        "case_name",
        "mesh",
        "physics",
        "time",
        "solver",
        "body_force",
        "manufactured_solution",
        "manufactured_omega",
        "boundary_conditions",
        "force_monitor",
        "output",
    }
    BOOL_TRUE = {"true", "t", "yes", "y", "on", "1"}
    BOOL_FALSE = {"false", "f", "no", "n", "off", "0"}

    def __init__(self, parameters):
        self.cfg = copy.deepcopy(parameters)
        if not isinstance(self.cfg, dict):
            self._error("<root>", "parameters must be supplied as a dictionary.")
        self._normalize()

    @staticmethod
    def _join(path, key):
        if path == "<root>":
            return str(key)
        return f"{path}.{key}"

    def _error(self, path, message):
        raise ValueError(f"Config error at {path}: {message}")

    def _check_keys(self, mapping, path, allowed):
        unknown = sorted(set(mapping) - set(allowed))
        if unknown:
            choices = ", ".join(sorted(allowed))
            self._error(
                path,
                f"unknown key(s): {', '.join(unknown)}. Allowed keys are: {choices}.",
            )

    def _mapping(self, key, required=True):
        if key not in self.cfg:
            if required:
                self._error("<root>", f"missing required section '{key}'.")
            self.cfg[key] = {}
        value = self.cfg[key]
        if value is None:
            value = {}
            self.cfg[key] = value
        if not isinstance(value, dict):
            self._error(key, "section must be a dictionary.")
        return value

    def _require(self, mapping, path, key):
        if key not in mapping:
            self._error(path, f"missing required key '{key}'.")
        if mapping[key] is None:
            self._error(self._join(path, key), "value cannot be null.")
        return mapping[key]

    def _string(self, value, path, *, allow_empty=False):
        if not isinstance(value, str):
            self._error(path, "must be a string.")
        if not allow_empty and not value.strip():
            self._error(path, "must be a non-empty string.")
        return value

    def _choice(self, value, path, choices, *, normalize=None):
        normalized = normalize(value) if normalize is not None else str(value).lower()
        if normalized not in choices:
            self._error(path, f"must be one of: {', '.join(sorted(choices))}.")
        return normalized

    def _bool(self, value, path):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in self.BOOL_TRUE:
                return True
            if normalized in self.BOOL_FALSE:
                return False
        self._error(path, "must be a boolean: true/false, yes/no, on/off, or 1/0.")

    def _float(self, value, path, *, positive=False, nonnegative=False):
        if isinstance(value, bool):
            self._error(path, "must be a number, not a boolean.")
        try:
            result = float(value)
        except (TypeError, ValueError):
            self._error(path, "must be a number.")
        if positive and result <= 0.0:
            self._error(path, "must be positive.")
        if nonnegative and result < 0.0:
            self._error(path, "must be non-negative.")
        return result

    def _int(self, value, path, *, positive=False, choices=None):
        if isinstance(value, bool):
            self._error(path, "must be an integer, not a boolean.")
        try:
            result = int(value)
        except (TypeError, ValueError):
            self._error(path, "must be an integer.")
        if result != self._float(value, path):
            self._error(path, "must be an integer.")
        if positive and result < 1:
            self._error(path, "must be >= 1.")
        if choices is not None and result not in choices:
            self._error(path, f"must be one of: {', '.join(map(str, sorted(choices)))}.")
        return result

    def _form(self, value, path):
        return self._choice(value, path, {"constant", "expression"})

    def _vector_value(self, value, path, form):
        if not isinstance(value, list) or len(value) not in {2, 3}:
            self._error(path, "must be a 2- or 3-component list.")

        normalized = []
        for i, component in enumerate(value):
            component_path = f"{path}[{i}]"
            if form == "constant":
                normalized.append(self._float(component, component_path))
                continue
            if isinstance(component, (int, float)) and not isinstance(component, bool):
                normalized.append(str(component))
                continue
            normalized.append(self._string(component, component_path))
        return normalized

    def _points(self, value, path):
        if not isinstance(value, list) or len(value) != 2:
            self._error(path, "must contain exactly two sampling points.")

        normalized_points = []
        expected_dim = None
        for point_index, point in enumerate(value):
            point_path = f"{path}[{point_index}]"
            if not isinstance(point, list) or len(point) not in {2, 3}:
                self._error(point_path, "must be a 2- or 3-coordinate list.")
            if expected_dim is None:
                expected_dim = len(point)
            elif len(point) != expected_dim:
                self._error(path, "all points must use the same dimension.")
            normalized_points.append(
                [
                    self._float(coord, f"{point_path}[{coord_index}]")
                    for coord_index, coord in enumerate(point)
                ]
            )
        return normalized_points

    @staticmethod
    def _equation_name(value):
        normalized = str(value).strip().lower()
        if normalized == "ns":
            return "NS"
        if normalized == "stokes":
            return "Stokes"
        return normalized

    def _normalize(self):
        if "forcing" in self.cfg:
            self._error("<root>.forcing", "section has been renamed to 'body_force'.")
        if "forces" in self.cfg:
            self._error("<root>.forces", "section has been renamed to 'force_monitor'.")
        self._check_keys(self.cfg, "<root>", self.TOP_LEVEL_KEYS)

        case_name = self._require(self.cfg, "<root>", "case_name")
        self.cfg["case_name"] = self._string(case_name, "case_name")

        self._normalize_mesh()
        self._normalize_physics()
        self._normalize_time()
        self._normalize_solver()
        if "manufactured_solution" in self.cfg:
            from solver.manufactured import manufactured_data

            kind = self._choice(self.cfg["manufactured_solution"],
                                "manufactured_solution", {"steady", "unsteady"})
            if self.cfg["physics"]["equation"] != "NS":
                self._error("manufactured_solution", "requires physics.equation: NS.")
            if "body_force" in self.cfg or "boundary_conditions" in self.cfg:
                self._error("manufactured_solution", "generates body force and boundary data; do not supply them separately.")
            omega = self._float(self.cfg.get("manufactured_omega", math.pi),
                                "manufactured_omega", positive=True)
            if not math.isfinite(omega):
                self._error("manufactured_omega", "must be finite.")
            self.cfg["manufactured_omega"] = omega
            self.cfg.update(manufactured_data(self.cfg["physics"]["nu"], kind == "steady", omega))
        elif "manufactured_omega" in self.cfg:
            self._error("manufactured_omega", "requires manufactured_solution.")
        self._normalize_body_force()
        self._normalize_boundary_conditions()
        self._normalize_force_monitor()
        self._normalize_output()

    def _normalize_mesh(self):
        mesh = self._mapping("mesh")
        self._check_keys(mesh, "mesh", {"domain", "boundary"})
        mesh["domain"] = self._string(
            self._require(mesh, "mesh", "domain"), "mesh.domain", allow_empty=True
        )
        mesh["boundary"] = self._string(
            self._require(mesh, "mesh", "boundary"), "mesh.boundary", allow_empty=True
        )

    def _normalize_physics(self):
        physics = self._mapping("physics")
        if "stablization" in physics:
            self._error("physics.stablization", "use 'stabilization' instead.")
        self._check_keys(
            physics, "physics", {"equation", "formula", "stabilization", "nu"}
        )

        physics["equation"] = self._choice(
            self._require(physics, "physics", "equation"),
            "physics.equation",
            {"NS", "Stokes"},
            normalize=self._equation_name,
        )
        physics["formula"] = self._choice(
            physics.get("formula", "rpcs"),
            "physics.formula",
            {"rpcs"},
        )
        physics["stabilization"] = self._choice(
            physics.get("stabilization", "t"), "physics.stabilization", {"t", "f"}
        )
        physics["nu"] = self._float(
            self._require(physics, "physics", "nu"), "physics.nu", positive=True
        )

    def _normalize_time(self):
        time = self._mapping("time")
        self._check_keys(time, "time", {"dt", "t_end", "BDF_order"})
        time["dt"] = self._float(
            self._require(time, "time", "dt"), "time.dt", positive=True
        )
        time["t_end"] = self._float(
            self._require(time, "time", "t_end"), "time.t_end", positive=True
        )
        time["BDF_order"] = self._int(
            self._require(time, "time", "BDF_order"),
            "time.BDF_order",
            choices={1, 2},
        )

    def _normalize_solver(self):
        solver = self._mapping("solver", required=False)
        if "stabilization_velocity" in solver:
            self._error(
                "solver.stabilization_velocity",
                "has been removed; use solver.stabilization_coefficient: dynamic or static.",
            )
        if "matrix_reassembly" in solver:
            self._error(
                "solver.matrix_reassembly",
                "has been removed; matrix reuse is decided internally.",
            )
        self._check_keys(
            solver,
            "solver",
            {
                "corrector_mass",
                "velocity_update",
                "convective_stabilization",
                "stabilization_coefficient",
                "stabilization_alpha",
                "stabilization_inverse_estimate",
                "formulation",
                "second_derivative_test_terms",
                "velocity_degree",
                "stress_form",
                "pressure_recovery",
                "pressure_increment_stabilization",
                "pressure_recovery_stabilization",
                "linear_rtol",
                "linear_atol",
                "linear_max_it",
                "gmres_restart",
                "monolithic_ksp_type",
                "monolithic_pc",
            },
        )

        solver["corrector_mass"] = self._choice(
            solver.get("corrector_mass", "auto"),
            "solver.corrector_mass",
            {"auto", "consistent"},
        )
        solver["formulation"] = self._choice(
            solver.get("formulation", "volume_acceleration"),
            "solver.formulation",
            {"volume_acceleration"},
        )
        solver["velocity_update"] = self._bool(
            solver.get("velocity_update", True), "solver.velocity_update"
        )
        solver["convective_stabilization"] = self._bool(
            solver.get("convective_stabilization", True),
            "solver.convective_stabilization",
        )
        for key in ("pressure_increment_stabilization", "pressure_recovery_stabilization"):
            solver[key] = self._bool(solver.get(key, True), f"solver.{key}")
        solver["stabilization_coefficient"] = self._choice(
            solver.get("stabilization_coefficient", "dynamic"),
            "solver.stabilization_coefficient",
            {"dynamic", "static"},
        )
        stabilization_alpha = solver.get("stabilization_alpha", "auto")
        if isinstance(stabilization_alpha, str):
            if stabilization_alpha.strip().lower() != "auto":
                self._error(
                    "solver.stabilization_alpha",
                    "must be 'auto' or a non-negative number.",
                )
            solver["stabilization_alpha"] = "auto"
        else:
            solver["stabilization_alpha"] = self._float(
                stabilization_alpha,
                "solver.stabilization_alpha",
                nonnegative=True,
            )
        solver["second_derivative_test_terms"] = self._bool(
            solver.get("second_derivative_test_terms", False),
            "solver.second_derivative_test_terms",
        )
        solver["stabilization_inverse_estimate"] = self._float(
            solver.get("stabilization_inverse_estimate", 36.0),
            "solver.stabilization_inverse_estimate",
            positive=True,
        )
        solver["stress_form"] = self._choice(
            solver.get("stress_form", "pseudo"),
            "solver.stress_form",
            {"pseudo", "physical"},
        )
        solver["pressure_recovery"] = self._choice(
            solver.get("pressure_recovery", "rotational"),
            "solver.pressure_recovery",
            {"rotational", "nonrotational"},
        )
        if "velocity_degree" in solver:
            solver["velocity_degree"] = self._int(
                solver["velocity_degree"], "solver.velocity_degree", positive=True
            )
        solver["linear_rtol"] = self._float(
            solver.get("linear_rtol", 1.0e-8), "solver.linear_rtol", positive=True
        )
        solver["linear_atol"] = self._float(
            solver.get("linear_atol", 1.0e-10), "solver.linear_atol", positive=True
        )
        solver["linear_max_it"] = self._int(
            solver.get("linear_max_it", 1000),
            "solver.linear_max_it",
            positive=True,
        )
        solver["gmres_restart"] = self._int(
            solver.get("gmres_restart", 100),
            "solver.gmres_restart",
            positive=True,
        )
        solver["monolithic_ksp_type"] = self._choice(
            solver.get("monolithic_ksp_type", "gmres"),
            "solver.monolithic_ksp_type",
            {"gmres", "fgmres"},
        )
        solver["monolithic_pc"] = self._choice(
            solver.get("monolithic_pc", "schur_amg"),
            "solver.monolithic_pc",
            {
                "pcd",
                "schur_amg",
                "lu",
                "fieldsplit",
                "asm_lu",
                "asm_ilu",
                "hypre",
                "jacobi",
            },
        )

    def _normalize_body_force(self):
        if "body_force" not in self.cfg or self.cfg["body_force"] is None:
            return
        body_force = self.cfg["body_force"]
        if not isinstance(body_force, dict):
            self._error("body_force", "section must be a dictionary.")
        self._check_keys(body_force, "body_force", {"form", "value"})
        form = self._form(body_force.get("form", "constant"), "body_force.form")
        body_force["form"] = form
        body_force["value"] = self._vector_value(
            self._require(body_force, "body_force", "value"),
            "body_force.value",
            form,
        )

    def _normalize_boundary_conditions(self):
        bcs = self._require(self.cfg, "<root>", "boundary_conditions")
        if not isinstance(bcs, list) or not bcs:
            self._error("boundary_conditions", "must be a non-empty list.")

        for i, bc in enumerate(bcs):
            path = f"boundary_conditions[{i}]"
            if not isinstance(bc, dict):
                self._error(path, "entry must be a dictionary.")
            self._check_keys(
                bc,
                path,
                {"id", "type", "form", "value", "backflow_stabilization"},
            )

            bc["id"] = self._int(self._require(bc, path, "id"), f"{path}.id", positive=True)
            bc_type = self._choice(
                self._require(bc, path, "type"),
                f"{path}.type",
                {"velocity", "traction", "pressure"},
            )
            if bc_type == "pressure":
                self._error(
                    f"{path}.type",
                    "pressure boundary conditions have been removed; use traction or velocity.",
                )
            bc["type"] = bc_type

            form = self._form(self._require(bc, path, "form"), f"{path}.form")
            bc["form"] = form
            bc["value"] = self._vector_value(
                self._require(bc, path, "value"), f"{path}.value", form
            )
            if "backflow_stabilization" in bc:
                if bc_type != "traction":
                    self._error(
                        f"{path}.backflow_stabilization",
                        "is only meaningful for traction open boundaries.",
                    )
                bc["backflow_stabilization"] = self._bool(
                    bc["backflow_stabilization"],
                    f"{path}.backflow_stabilization",
                )

    def _normalize_force_monitor(self):
        force_monitor = self._mapping("force_monitor", required=False)
        for old_key, new_key in {
            "filename": "file",
            "cylinder_diameter": "characteristic_length",
            "density": None,
        }.items():
            if old_key in force_monitor:
                replacement = f"use '{new_key}' instead." if new_key else "remove it; density is not used."
                self._error(f"force_monitor.{old_key}", f"deprecated field; {replacement}")

        self._check_keys(
            force_monitor,
            "force_monitor",
            {
                "enabled",
                "boundary_id",
                "pressure_points",
                "characteristic_velocity",
                "characteristic_length",
                "reference_area",
                "spanwise_length",
                "frequency",
                "file",
            },
        )

        force_monitor["enabled"] = self._bool(
            force_monitor.get("enabled", False), "force_monitor.enabled"
        )
        force_monitor["file"] = self._string(
            force_monitor.get("file", "force_coefficients.csv"),
            "force_monitor.file",
        )
        force_monitor["frequency"] = self._int(
            force_monitor.get("frequency", 1),
            "force_monitor.frequency",
            positive=True,
        )
        force_monitor["characteristic_velocity"] = self._float(
            force_monitor.get("characteristic_velocity", 1.0),
            "force_monitor.characteristic_velocity",
            positive=True,
        )
        force_monitor["characteristic_length"] = self._float(
            force_monitor.get("characteristic_length", 1.0),
            "force_monitor.characteristic_length",
            positive=True,
        )

        if "boundary_id" in force_monitor:
            force_monitor["boundary_id"] = self._int(
                force_monitor["boundary_id"],
                "force_monitor.boundary_id",
                positive=True,
            )
        if "reference_area" in force_monitor:
            force_monitor["reference_area"] = self._float(
                force_monitor["reference_area"],
                "force_monitor.reference_area",
                positive=True,
            )
        if "spanwise_length" in force_monitor:
            force_monitor["spanwise_length"] = self._float(
                force_monitor["spanwise_length"],
                "force_monitor.spanwise_length",
                positive=True,
            )

        if force_monitor["enabled"]:
            self._require(force_monitor, "force_monitor", "boundary_id")
            force_monitor["pressure_points"] = self._points(
                self._require(
                    force_monitor, "force_monitor", "pressure_points"
                ),
                "force_monitor.pressure_points",
            )

    def _normalize_output(self):
        output = self._mapping("output")
        if "start" in output:
            self._error("output.start", "use 'start_time' instead.")
        self._check_keys(
            output,
            "output",
            {"start_time", "end_time", "frequency", "progress_frequency", "dir"},
        )

        output["dir"] = self._string(self._require(output, "output", "dir"), "output.dir")
        output["start_time"] = self._float(
            output.get("start_time", 0.0), "output.start_time", nonnegative=True
        )
        if "end_time" in output:
            output["end_time"] = self._float(
                output["end_time"], "output.end_time", nonnegative=True
            )
            if output["end_time"] < output["start_time"]:
                self._error("output.end_time", "must be greater than or equal to output.start_time.")
        else:
            output["end_time"] = None
        output["frequency"] = self._int(
            output.get("frequency", 1), "output.frequency", positive=True
        )
        if "progress_frequency" in output:
            output["progress_frequency"] = self._int(
                output["progress_frequency"],
                "output.progress_frequency",
                positive=True,
            )
        else:
            output["progress_frequency"] = None

    def get_phys(self, mesh):
        nu = self.cfg["physics"]["nu"]
        return (
            Constant(mesh, PETSc.ScalarType(nu)),
            self.cfg["physics"]["equation"],
            self.cfg["physics"]["stabilization"],
        )

    def get_time(self):
        time = self.cfg["time"]
        return time["dt"], time["t_end"], time["BDF_order"]

    def get_solver_options(self):
        return self.cfg.get("solver", {})

    def get_force_monitor_options(self):
        return self.cfg.get("force_monitor", {"enabled": False})
