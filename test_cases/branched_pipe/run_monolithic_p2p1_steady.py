import argparse
import sys
from pathlib import Path

import meshio
import numpy as np
from basix.ufl import element, mixed_element
from dolfinx import fem, io
from dolfinx.fem.petsc import NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc
from ufl import TestFunctions, TrialFunction, derivative, div, dot, dx, grad, inner, nabla_grad, split, sym

ROOT_DIR = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solver.benchmark import PointEvaluator
from solver.mesh_loader import MeshHandler
from solver.reporting import write_csv


POUX_REFERENCE = {
    "total_kinetic_energy": 1.9909,
    "top_outflux": 0.19374,
    "bottom_outflux": 0.30623,
    "top_charge_drop": 0.39924,
    "bottom_charge_drop": 0.39925,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Steady monolithic Taylor-Hood P2/P1 check for the bifurcated tube."
    )
    parser.add_argument(
        "--domain",
        type=Path,
        default=CASE_DIR / "mesh" / "mesh_domain.xdmf",
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        default=CASE_DIR / "mesh" / "mesh_boundary.xdmf",
    )
    parser.add_argument(
        "--msh",
        type=Path,
        default=CASE_DIR / "mesh" / "mesh_boundary.xdmf",
    )
    parser.add_argument("--nu", type=float, default=0.00166667)
    parser.add_argument(
        "--stress-form",
        choices=("pseudo", "physical"),
        default="pseudo",
        help="Natural open-boundary stress used by the monolithic weak form.",
    )
    parser.add_argument(
        "--continuation",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Continuation factors multiplying the nonlinear convection term.",
    )
    parser.add_argument("--newton-rtol", type=float, default=1.0e-10)
    parser.add_argument("--newton-atol", type=float, default=1.0e-10)
    parser.add_argument("--newton-max-it", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=CASE_DIR / "output" / "monolithic_p2p1_steady_metrics.csv",
    )
    return parser.parse_args()


def load_boundary_lines(mesh_path):
    mesh = meshio.read(mesh_path)
    mesh_points = mesh.points[:, :2]
    line_blocks = []
    tag_blocks = []
    tag_data = mesh.cell_data.get("gmsh:physical", mesh.cell_data.get("name_to_read"))
    if tag_data is None:
        raise ValueError(f"No boundary tags found in {mesh_path}.")
    for block, tags in zip(mesh.cells, tag_data):
        if block.type == "line":
            line_blocks.append(block.data)
            tag_blocks.append(tags)
    if not line_blocks:
        raise ValueError(f"No line boundary cells found in {mesh_path}.")
    return mesh_points, np.vstack(line_blocks), np.concatenate(tag_blocks)


def boundary_quadrature(mesh_points, lines, tags, selector, normal, order=6):
    xi, wi = np.polynomial.legendre.leggauss(order)
    s_points = 0.5 * (xi + 1.0)
    s_weights = 0.5 * wi
    points = []
    weights = []
    length = 0.0
    normal = np.asarray(normal, dtype=np.float64)
    inward_shift = -normal * 1.0e-9

    for (mesh_i, mesh_j), tag in zip(lines, tags):
        if not selector(mesh_i, mesh_j, tag):
            continue
        p0 = mesh_points[mesh_i]
        p1 = mesh_points[mesh_j]
        segment = p1 - p0
        segment_length = float(np.linalg.norm(segment))
        length += segment_length
        for s, w in zip(s_points, s_weights):
            points.append((1.0 - s) * p0 + s * p1 + inward_shift)
            weights.append(segment_length * float(w))

    if length <= 0.0:
        raise ValueError("Selected boundary has zero length.")
    return length, np.asarray(points, dtype=np.float64), np.asarray(weights), normal


def boundary_integrals(mesh_handler, velocity, pressure, mesh_path):
    mesh_points, lines, tags = load_boundary_lines(mesh_path)
    groups = {
        "inlet": boundary_quadrature(
            mesh_points,
            lines,
            tags,
            selector=lambda _i, _j, tag: tag == 1,
            normal=np.array([-1.0, 0.0]),
        ),
        "top": boundary_quadrature(
            mesh_points,
            lines,
            tags,
            selector=lambda i, j, tag: tag == 2 and mesh_points[[i, j], 1].mean() > 0.0,
            normal=np.array([1.0, 0.0]),
        ),
        "bottom": boundary_quadrature(
            mesh_points,
            lines,
            tags,
            selector=lambda i, j, tag: tag == 2 and mesh_points[[i, j], 1].mean() < 0.0,
            normal=np.array([1.0, 0.0]),
        ),
    }

    all_points = np.vstack([group[1] for group in groups.values()])
    evaluator = PointEvaluator(mesh_handler, all_points, mesh_handler.comm)
    velocity_values = evaluator.evaluate(velocity)
    pressure_values = evaluator.evaluate(pressure)
    if mesh_handler.comm.rank != 0:
        return None

    diagnostics = {}
    offset = 0
    for name, (length, points, weights, normal) in groups.items():
        count = len(points)
        u = velocity_values[offset : offset + count, :2]
        p = pressure_values[offset : offset + count, 0]
        normal_velocity = u @ normal
        diagnostics[name] = {
            "length": length,
            "flux": float(np.sum(weights * normal_velocity)),
            "normal_square": float(np.sum(weights * normal_velocity**2)),
            "pressure_integral": float(np.sum(weights * p)),
        }
        offset += count
    return diagnostics


def make_bcs(mesh, facet_tags, W):
    fdim = mesh.topology.dim - 1
    V, _ = W.sub(0).collapse()

    inlet = fem.Function(V)
    inlet.interpolate(lambda x: np.vstack((12.0 * x[1] * (1.0 - 2.0 * x[1]), 0.0 * x[0])))
    zero = fem.Function(V)

    inlet_facets = facet_tags.find(1)
    wall_facets = facet_tags.find(3)
    inlet_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, inlet_facets)
    wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    return [
        fem.dirichletbc(inlet, inlet_dofs, W.sub(0)),
        fem.dirichletbc(zero, wall_dofs, W.sub(0)),
    ]


def build_problem(mesh_handler, args):
    mesh = mesh_handler.mesh
    gdim = mesh.geometry.dim
    velocity_el = element("Lagrange", mesh.basix_cell(), 2, shape=(gdim,))
    pressure_el = element("Lagrange", mesh.basix_cell(), 1)
    W = fem.functionspace(mesh, mixed_element([velocity_el, pressure_el]))
    w = fem.Function(W)
    u, p = split(w)
    v, q = TestFunctions(W)
    theta = fem.Constant(mesh, PETSc.ScalarType(0.0))
    nu = fem.Constant(mesh, PETSc.ScalarType(args.nu))

    if args.stress_form == "physical":
        viscous = 2.0 * nu * inner(sym(grad(u)), sym(grad(v)))
    else:
        viscous = nu * inner(grad(u), grad(v))

    residual = (
        theta * inner(dot(u, nabla_grad(u)), v) * mesh_handler.dx
        + viscous * mesh_handler.dx
        - p * div(v) * mesh_handler.dx
        + q * div(u) * mesh_handler.dx
    )
    jacobian = derivative(residual, w, TrialFunction(W))
    bcs = make_bcs(mesh, mesh_handler.ft, W)
    return W, w, residual, jacobian, bcs, theta


def configure_nonlinear_problem(residual, jacobian, w, bcs, args):
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": args.newton_rtol,
        "snes_atol": args.newton_atol,
        "snes_max_it": args.newton_max_it,
        "snes_error_if_not_converged": True,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }
    return NonlinearProblem(
        residual,
        w,
        bcs=bcs,
        J=jacobian,
        petsc_options_prefix="bif_monolithic_",
        petsc_options=petsc_options,
    )


def collapse_subfunction(function, index):
    return function.sub(index).collapse()


def rows_from_solution(mesh_handler, w, args):
    u_h = collapse_subfunction(w, 0)
    p_h = collapse_subfunction(w, 1)
    u_expr, _ = split(w)
    energy = mesh_handler.comm.allreduce(
        fem.assemble_scalar(fem.form(0.5 * dot(u_expr, u_expr) * mesh_handler.dx)),
        op=MPI.SUM,
    )
    diagnostics = boundary_integrals(mesh_handler, u_h, p_h, args.msh)
    if mesh_handler.comm.rank != 0:
        return None

    inlet = diagnostics["inlet"]
    top = diagnostics["top"]
    bottom = diagnostics["bottom"]
    top_velocity_part = 0.5 * (inlet["normal_square"] - top["normal_square"])
    bottom_velocity_part = 0.5 * (inlet["normal_square"] - bottom["normal_square"])
    top_pressure_part = inlet["pressure_integral"] - top["pressure_integral"]
    bottom_pressure_part = inlet["pressure_integral"] - bottom["pressure_integral"]
    top_charge_drop = top_velocity_part + top_pressure_part
    bottom_charge_drop = bottom_velocity_part + bottom_pressure_part

    data = {
        "total_kinetic_energy": energy,
        "top_outflux": top["flux"],
        "bottom_outflux": bottom["flux"],
        "top_charge_drop": top_charge_drop,
        "bottom_charge_drop": bottom_charge_drop,
        "top_charge_velocity_part": top_velocity_part,
        "bottom_charge_velocity_part": bottom_velocity_part,
        "top_charge_pressure_part": top_pressure_part,
        "bottom_charge_pressure_part": bottom_pressure_part,
        "inlet_normal_square": inlet["normal_square"],
        "top_normal_square": top["normal_square"],
        "bottom_normal_square": bottom["normal_square"],
        "inlet_pressure_integral": inlet["pressure_integral"],
        "top_pressure_integral": top["pressure_integral"],
        "bottom_pressure_integral": bottom["pressure_integral"],
    }

    units = {
        "total_kinetic_energy": "kg m s^-2",
        "top_outflux": "m^2 s^-1",
        "bottom_outflux": "m^2 s^-1",
        "top_charge_drop": "m^3 s^-2",
        "bottom_charge_drop": "m^3 s^-2",
        "top_charge_velocity_part": "m^3 s^-2",
        "bottom_charge_velocity_part": "m^3 s^-2",
        "top_charge_pressure_part": "m^3 s^-2",
        "bottom_charge_pressure_part": "m^3 s^-2",
        "inlet_normal_square": "m^3 s^-2",
        "top_normal_square": "m^3 s^-2",
        "bottom_normal_square": "m^3 s^-2",
        "inlet_pressure_integral": "m^3 s^-2",
        "top_pressure_integral": "m^3 s^-2",
        "bottom_pressure_integral": "m^3 s^-2",
    }

    rows = []
    for quantity, value in data.items():
        reference = POUX_REFERENCE.get(quantity, "")
        rows.append(
            {
                "quantity": quantity,
                "monolithic_p2p1": value,
                "poux_reference": reference,
                "unit": units[quantity],
            }
        )
    return rows


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    mesh_handler = MeshHandler(comm, str(args.domain), str(args.boundary))
    _, w, residual, jacobian, bcs, theta = build_problem(mesh_handler, args)
    problem = configure_nonlinear_problem(residual, jacobian, w, bcs, args)

    if comm.rank == 0:
        print(
            f"--- monolithic P2/P1 steady NS: stress_form={args.stress_form}, "
            f"nu={args.nu:g} ---"
        )
        sys.stdout.flush()

    for value in args.continuation:
        theta.value = PETSc.ScalarType(value)
        problem.solve()
        reason = problem.solver.getConvergedReason()
        iterations = problem.solver.getIterationNumber()
        converged = reason > 0
        if comm.rank == 0:
            print(
                f"continuation={value:g}, SNES iterations={iterations}, "
                f"reason={reason}, converged={converged}"
            )
            sys.stdout.flush()
        if not converged:
            raise RuntimeError(f"Newton failed at continuation={value:g}.")

    rows = rows_from_solution(mesh_handler, w, args)
    if comm.rank != 0:
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output,
        rows,
        ["quantity", "monolithic_p2p1", "poux_reference", "unit"],
    )
    for row in rows:
        print(
            f"{row['quantity']}: {row['monolithic_p2p1']:.15g} "
            f"{row['unit']} (Poux={row['poux_reference']})"
        )

    values = {row["quantity"]: row["monolithic_p2p1"] for row in rows}
    print(
        "charge_drop_comparison: "
        f"top_full_diff={values['top_charge_drop'] - POUX_REFERENCE['top_charge_drop']:.12g}, "
        f"top_pressure_only_diff={values['top_charge_pressure_part'] - POUX_REFERENCE['top_charge_drop']:.12g}, "
        f"bottom_full_diff={values['bottom_charge_drop'] - POUX_REFERENCE['bottom_charge_drop']:.12g}, "
        f"bottom_pressure_only_diff={values['bottom_charge_pressure_part'] - POUX_REFERENCE['bottom_charge_drop']:.12g}"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
