"""Shared exact fields and data for the manufactured Navier--Stokes tests."""

import ufl
from dolfinx import fem


def manufactured_data(nu, steady=False, omega=ufl.pi):
    """Generate forcing and pseudo-traction from one divergence-free family."""
    base = ("sin(pi*x[1])*cos(pi*x[0]/2)**2",
            "-sin(pi*x[0])*cos(pi*x[1]/2)**2")
    gradient = (
        ("-pi/2*sin(pi*x[0])*sin(pi*x[1])",
         "pi*cos(pi*x[1])*cos(pi*x[0]/2)**2"),
        ("-pi*cos(pi*x[0])*cos(pi*x[1]/2)**2",
         "pi/2*sin(pi*x[1])*sin(pi*x[0])"),
    )
    laplacian = (
        f"-pi**2/2*cos(pi*x[0])*sin(pi*x[1])-pi**2*({base[0]})",
        f"pi**2/2*cos(pi*x[1])*sin(pi*x[0])-pi**2*({base[1]})",
    )
    pressure_base = "cos(pi*x[0]/2)*sin(pi*x[1]/2)"
    pressure_gradient = ("-pi/2*sin(pi*x[0]/2)*sin(pi*x[1]/2)",
                         "pi/2*cos(pi*x[0]/2)*cos(pi*x[1]/2)")
    frequency = "pi" if omega == ufl.pi else repr(float(omega))
    factor = "1" if steady else f"cos(2*{frequency}*t)"
    derivative = "0" if steady else f"-2*{frequency}*sin(2*{frequency}*t)"
    velocity = [f"({factor})*({value})" for value in base]
    pressure = f"({factor})*({pressure_base})"
    force = [
        f"({derivative})*({base[i]})"
        f"+({factor})**2*(({base[0]})*({gradient[i][0]})"
        f"+({base[1]})*({gradient[i][1]}))"
        f"-({float(nu)!r})*({factor})*({laplacian[i]})"
        f"+({factor})*({pressure_gradient[i]})"
        for i in range(2)
    ]
    traction = [
        f"({float(nu)!r})*({factor})*({gradient[i][0]})"
        f"-({pressure if i == 0 else '0'})"
        f"-0.5*min(({velocity[0]}),0)*({velocity[i]})"
        for i in range(2)
    ]
    return {
        "body_force": {"form": "expression", "value": force},
        "boundary_conditions": [
            {"id": marker, "type": "traction" if marker == 2 else "velocity",
             "form": "expression", "value": traction if marker == 2 else velocity}
            for marker in (1, 2, 3, 4)
        ],
    }


def manufactured_fields(mesh, time=0.0, omega=ufl.pi):
    x = ufl.SpatialCoordinate(mesh)
    factor = ufl.cos(2 * omega * time)
    velocity = factor * ufl.as_vector((
        ufl.sin(ufl.pi * x[1]) * ufl.cos(ufl.pi * x[0] / 2)**2,
        -ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1] / 2)**2,
    ))
    pressure = factor * ufl.cos(ufl.pi * x[0] / 2) * ufl.sin(ufl.pi * x[1] / 2)
    return velocity, pressure


def initialize_manufactured_state(solver, velocity, pressure):
    for target, expression in ((solver.vo, velocity), (solver.po, pressure)):
        points = target.function_space.element.interpolation_points
        if callable(points):
            points = points()
        target.interpolate(fem.Expression(expression, points))
        target.x.scatter_forward()
    for name in ("vn", "voo", "vtilde"):
        if hasattr(solver, name):
            target = getattr(solver, name)
            target.x.array[:] = solver.vo.x.array
            target.x.scatter_forward()
    solver.pn.x.array[:] = solver.po.x.array
    solver.pn.x.scatter_forward()
    if hasattr(solver, "w"):
        for target in (solver.w, solver.w_old, solver.w_older):
            target.x.array[solver.velocity_parent_dofs] = solver.vo.x.array
            target.x.array[solver.pressure_parent_dofs] = solver.po.x.array
            target.x.scatter_forward()
    for name in ("phin", "phio", "phioo"):
        if hasattr(solver, name):
            target = getattr(solver, name)
            target.x.array[:] = 0
            target.x.scatter_forward()
