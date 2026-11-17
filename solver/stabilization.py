"""Element coefficients for the RPCS-VMS metric scale model."""

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc


def resolve_stabilization_alpha(alpha):
    """Keep the automatic threshold choice distinct from a fixed alpha_c."""
    if alpha == "auto":
        return "auto"
    value = float(alpha)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("stabilization_alpha must be 'auto' or finite and non-negative.")
    return value


def momentum_timescale(delta, nu, velocity_metric, metric_squared, inverse_estimate, sqrt):
    return 1.0 / sqrt(
        delta**-2 + velocity_metric + inverse_estimate * nu**2 * metric_squared
    )


class ElementStabilization:
    """Evaluate one scalar value per cell at its DG0 interpolation point.

    Both the advecting field and geometric metric are sampled at that point.
    The resulting DG0 weights are held fixed throughout all four stages, even
    for mapped cells or a velocity that varies within an element.
    """

    def __init__(self, mesh_handler, nu, dt, alpha, enabled, inverse_estimate=36.0):
        self.mh = mesh_handler
        self.nu = nu
        self.dt = dt
        self.alpha = resolve_stabilization_alpha(alpha)
        self.enabled = enabled
        self.inverse_estimate = float(inverse_estimate)
        if not np.isfinite(self.inverse_estimate) or self.inverse_estimate <= 0.0:
            raise ValueError("stabilization_inverse_estimate must be finite and positive.")
        self.space = fem.functionspace(self.mh.mesh, ("DG", 0))
        self.entries = {}

    def coefficients(self, advective_velocity, gamma0):
        key = float(gamma0)
        if key not in self.entries:
            tau_m = fem.Function(self.space)
            tau_c = fem.Function(self.space)
            entry = {"tau_m": tau_m, "tau_c": tau_c}
            if self.enabled:
                jacobian_inverse = ufl.JacobianInverse(self.mh.mesh)
                metric = ufl.dot(ufl.transpose(jacobian_inverse), jacobian_inverse)
                metric_squared = ufl.inner(metric, metric)
                velocity_metric = ufl.dot(advective_velocity, ufl.dot(metric, advective_velocity))
                tau_expression = momentum_timescale(
                    self.dt / key, self.nu, velocity_metric, metric_squared,
                    PETSc.ScalarType(self.inverse_estimate), ufl.sqrt,
                )
                points = self.space.element.interpolation_points
                if callable(points):
                    points = points()
                entry["momentum_expression"] = fem.Expression(tau_expression, points)
                if self.alpha == "auto":
                    entry["scalar_expression"] = None
                    entry["scalar_constant"] = (
                        (1.0 - 2.0 / self.mh.gdim) * float(self.nu.value) / 2.0
                    )
                elif self.alpha == 0.0:
                    entry["scalar_expression"] = None
                    entry["scalar_constant"] = 0.0
                else:
                    entry["scalar_expression"] = fem.Expression(
                        PETSc.ScalarType(self.alpha) / (tau_m * ufl.sqrt(metric_squared)), points
                    )
            self.entries[key] = entry
        entry = self.entries[key]
        return entry["tau_m"], entry["tau_c"]

    def update(self, gamma0):
        entry = self.entries[float(gamma0)]
        if self.enabled:
            entry["tau_m"].interpolate(entry["momentum_expression"])
            entry["tau_m"].x.scatter_forward()
            if entry["scalar_expression"] is None:
                entry["tau_c"].x.array[:] = entry["scalar_constant"]
                entry["tau_c"].x.scatter_forward()
            else:
                entry["tau_c"].interpolate(entry["scalar_expression"])
                entry["tau_c"].x.scatter_forward()
