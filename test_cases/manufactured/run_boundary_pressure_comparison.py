import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import SymLogNorm
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh, plot
from mpi4py import MPI
from petsc4py import PETSc

ROOT_DIR = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solver.boundary import BoundaryFactory
from solver.manufactured import initialize_manufactured_state, manufactured_fields
from solver.mesh_loader import MeshHandler
from solver.monolithic_vms import MonolithicSUPGPSPGSolver
from solver.configuration import CaseConfig
from solver.reporting import configure_matplotlib, save_figure, scientific_tex, scientific_time_step, write_csv
from solver.rpcs_vms import FlowSolver
from test_cases.manufactured.parameters import OUTPUT_DIR, SMALL_TIME_STEP, STEADY


METHODS = (
    {
        "id": "nonrotational",
        "label": "Non-rotational formulation",
        "title": "Non-rotational formulation",
        "solver": "split",
        "pressure_recovery": "nonrotational",
    },
    {
        "id": "rpcs_vms",
        "label": "RPCS-VMS",
        "title": "RPCS-VMS",
        "solver": "split",
        "pressure_recovery": "rotational",
    },
    {
        "id": "monolithic_vms",
        "label": "Monolithic-VMS",
        "title": "Monolithic-VMS",
        "solver": "monolithic",
        "pressure_recovery": None,
    },
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare split and stabilization effects on near-boundary pressure."
    )
    parser.add_argument("--mesh-n", type=int, default=64)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--t-end", type=float)
    solution = parser.add_mutually_exclusive_group()
    solution.add_argument("--unsteady", action="store_true",
                        help="Use the BDF2 small-time-step manufactured problem.")
    solution.add_argument("--steady", dest="unsteady", action="store_false",
                        help="Use the steady manufactured solution with BDF1.")
    parser.set_defaults(unsteady=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate the plot from existing output data.",
    )
    args = parser.parse_args()
    if args.dt is None:
        args.dt = 1.e-3 if args.unsteady else 1.e-2
    if args.t_end is None:
        args.t_end = .5 if args.unsteady else STEADY["time"]["t_end"]
    if (not np.isfinite(args.dt) or not np.isfinite(args.t_end)
            or args.dt <= 0 or args.t_end <= 0
            or not np.isclose(round(args.t_end / args.dt) * args.dt,
                              args.t_end, rtol=0., atol=1.e-12)):
        parser.error("Positive final time must be an integer multiple of positive dt.")
    return args


def resolve(path):
    path = Path(path)
    return str(path if path.is_absolute() else CASE_DIR / path)


def make_config(mesh_n, dt, t_end, pressure_recovery=None, *, unsteady=True):
    cfg = CaseConfig(SMALL_TIME_STEP if unsteady else STEADY)
    cfg.cfg["mesh"]["domain"] = resolve(
        f"square_triangular_mesh/square_n{mesh_n}_domain.xdmf"
    )
    cfg.cfg["mesh"]["boundary"] = resolve(
        f"square_triangular_mesh/square_n{mesh_n}_boundary.xdmf"
    )
    cfg.cfg["physics"]["stabilization"] = "t"
    cfg.cfg["time"]["dt"] = dt
    cfg.cfg["time"]["t_end"] = t_end
    cfg.cfg["time"]["BDF_order"] = 2 if unsteady else 1
    solver_cfg = cfg.cfg["solver"]
    solver_cfg["velocity_degree"] = 1
    solver_cfg["linear_rtol"] = 1.0e-8
    solver_cfg["linear_atol"] = 1.0e-10
    solver_cfg["linear_max_it"] = 1000
    solver_cfg["gmres_restart"] = 100
    solver_cfg["stress_form"] = "pseudo"
    solver_cfg["monolithic_ksp_type"] = "gmres"
    solver_cfg["monolithic_pc"] = "schur_amg"
    if pressure_recovery is not None:
        solver_cfg["pressure_recovery"] = pressure_recovery
    return cfg


def exact_fields(mesh_handler, time=0.0):
    return manufactured_fields(mesh_handler.mesh, time)


def boundary_strip_measure(mesh_handler):
    mesh = mesh_handler.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, tdim)
    exterior_facets = dmesh.exterior_facet_indices(mesh.topology)
    facet_to_cell = mesh.topology.connectivity(fdim, tdim)
    boundary_cells = (np.unique(np.concatenate([
        facet_to_cell.links(facet) for facet in exterior_facets]))
        if len(exterior_facets) else np.empty(0, dtype=np.int32))
    cell_map = mesh.topology.index_map(tdim)
    cell_count = cell_map.size_local + cell_map.num_ghosts
    cell_indices = np.arange(cell_count, dtype=np.int32)
    cell_values = np.full(cell_count, 2, dtype=np.int32)
    cell_values[boundary_cells] = 1
    tags = dmesh.meshtags(mesh, tdim, cell_indices, cell_values)
    dx = ufl.Measure(
        "dx",
        domain=mesh,
        subdomain_data=tags,
        metadata={"quadrature_degree": 5},
    )
    boundary_owned = int(np.count_nonzero(boundary_cells < cell_map.size_local))
    return (dx, mesh.comm.allreduce(boundary_owned, op=MPI.SUM),
            mesh.comm.allreduce(cell_map.size_local - boundary_owned, op=MPI.SUM))


def global_scalar(comm, form):
    return comm.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM)


def compute_metrics(mesh_handler, velocity, pressure, time=0.0):
    comm = mesh_handler.mesh.comm
    velocity_exact, pressure_exact = exact_fields(mesh_handler, time)
    one = fem.Constant(mesh_handler.mesh, PETSc.ScalarType(1.0))
    area = global_scalar(comm, one * mesh_handler.dx)
    exact_mean = global_scalar(comm, pressure_exact * mesh_handler.dx) / area
    numerical_mean = global_scalar(comm, pressure * mesh_handler.dx) / area
    pressure_error = pressure - pressure_exact

    dx_strip, boundary_cells, interior_cells = boundary_strip_measure(mesh_handler)
    boundary_area = global_scalar(comm, one * dx_strip(1))
    interior_area = global_scalar(comm, one * dx_strip(2))
    global_pressure_error = np.sqrt(
        global_scalar(comm, pressure_error**2 * mesh_handler.dx)
    )
    boundary_pressure_error = np.sqrt(
        global_scalar(comm, pressure_error**2 * dx_strip(1))
    )
    interior_pressure_error = np.sqrt(
        global_scalar(comm, pressure_error**2 * dx_strip(2))
    )
    velocity_error = np.sqrt(
        global_scalar(
            comm,
            ufl.inner(velocity - velocity_exact, velocity - velocity_exact)
            * mesh_handler.dx,
        )
    )
    return {
        "global_pressure_error": float(global_pressure_error),
        "boundary_pressure_error": float(boundary_pressure_error),
        "interior_pressure_error": float(interior_pressure_error),
        "velocity_l2_error": float(velocity_error),
        "pressure_mean_error": float(numerical_mean - exact_mean),
        "pressure_numerical_mean": float(numerical_mean),
        "pressure_exact_mean": float(exact_mean),
        "boundary_strip_area": float(boundary_area),
        "interior_area": float(interior_area),
        "boundary_cells": boundary_cells,
        "interior_cells": interior_cells,
    }


def local_triangles(function_space):
    topology, _, geometry = plot.vtk_mesh(function_space)
    triangles = []
    offset = 0
    while offset < len(topology):
        vertex_count = int(topology[offset])
        vertices = topology[offset + 1 : offset + 1 + vertex_count]
        if vertex_count == 3:
            triangles.append(vertices)
        elif vertex_count == 4:
            triangles.append([vertices[0], vertices[1], vertices[2]])
            triangles.append([vertices[0], vertices[2], vertices[3]])
        else:
            raise ValueError(f"Unsupported plotting cell with {vertex_count} vertices.")
        offset += vertex_count + 1
    return np.asarray(triangles, dtype=np.int64), geometry[:, :2]


def pressure_error_field(solver, time=0.0):
    triangles, coordinates = local_triangles(solver.Q)
    values = np.asarray(solver.pn.x.array[: len(coordinates)], dtype=float)
    exact_values = np.cos(2 * np.pi * time) * np.cos(np.pi * coordinates[:, 0] / 2) * np.sin(
        np.pi * coordinates[:, 1] / 2
    )
    errors = values - exact_values
    return triangles, coordinates, errors


def initialize_solver(method, config, mesh_handler):
    if method["solver"] == "monolithic":
        solver = MonolithicSUPGPSPGSolver(MPI.COMM_WORLD, mesh_handler, config)
        boundary_data = BoundaryFactory(
            mesh_handler,
            solver.V,
            solver.Q,
            time_symbol=solver.time_symbol,
            mixed_velocity_space=solver.mixed_velocity_space,
        ).create(config.cfg["boundary_conditions"])
    else:
        solver = FlowSolver(MPI.COMM_WORLD, mesh_handler, config)
        boundary_data = BoundaryFactory(
            mesh_handler,
            solver.V,
            solver.Q,
            time_symbol=solver.time_symbol,
        ).create(config.cfg["boundary_conditions"])
    solver.set_bcs(boundary_data)
    initialize_manufactured_state(solver, *manufactured_fields(mesh_handler.mesh))
    return solver


def run_method(method, mesh_n, dt, final_time, *, unsteady=True):
    config = make_config(mesh_n, dt, final_time, method["pressure_recovery"], unsteady=unsteady)
    mesh_handler = MeshHandler(
        MPI.COMM_WORLD,
        config.cfg["mesh"]["domain"],
        config.cfg["mesh"]["boundary"],
    )
    solver = initialize_solver(method, config, mesh_handler)

    dt = config.cfg["time"]["dt"]
    final_time = config.cfg["time"]["t_end"]
    previous_pressure = fem.Function(solver.Q)
    update_defect = None
    if method["id"] == "nonrotational":
        update_defect = fem.form((solver.pn - previous_pressure - solver.phin)**2 * mesh_handler.dx)
    maximum_update_defect = 0.0
    for step in range(1, round(final_time / dt) + 1):
        previous_pressure.x.array[:] = solver.po.x.array
        solver.solve_step(step * dt, use_bdf1=(not unsteady or step == 1), compute_norms=False)
        if update_defect is not None:
            maximum_update_defect = max(maximum_update_defect, float(np.sqrt(
                MPI.COMM_WORLD.allreduce(fem.assemble_scalar(update_defect), op=MPI.SUM))))
        if MPI.COMM_WORLD.rank == 0 and step % 200 == 0:
            print(f"[{method['label']}] step {step}/{round(final_time / dt)}")

    exact_time = final_time if unsteady else 0.0
    metrics = compute_metrics(mesh_handler, solver.vn, solver.pn, exact_time)
    triangles, coordinates, errors = pressure_error_field(solver, exact_time)
    row = {
        "method_id": method["id"],
        "method": method["title"],
        "mesh_n": mesh_n,
        "cells": metrics["boundary_cells"] + metrics["interior_cells"],
        "dt": dt,
        "final_time": final_time,
        "bdf_order": 2 if unsteady else 1,
        "viscosity": 1.0,
        "velocity_pair": "P1/P1",
        "linear_rtol": 1.0e-8,
        "linear_atol": 1.0e-10,
        "maximum_update_defect_l2": maximum_update_defect if update_defect is not None else np.nan,
        **metrics,
    }
    return row, (triangles, coordinates, errors)


def plot_fields(field_data, output_dir):
    configure_matplotlib()
    all_errors = np.concatenate([field_data[method["id"]][2] for method in METHODS])
    limit = float(np.max(np.abs(all_errors)))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = 1.0
    norm = SymLogNorm(linthresh=0.005, linscale=0.8, vmin=-limit, vmax=limit)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharex=True, sharey=True)
    contour = None
    for axis, method in zip(axes, METHODS):
        triangles, coordinates, errors = field_data[method["id"]]
        triangulation = mtri.Triangulation(
            coordinates[:, 0], coordinates[:, 1], triangles
        )
        contour = axis.tripcolor(
            triangulation,
            errors,
            shading="gouraud",
            cmap="RdBu_r",
            norm=norm,
        )
        axis.set_title(method["label"])
        axis.set_aspect("equal")
        axis.set_xlim(-0.5, 0.5)
        axis.set_ylim(-0.5, 0.5)
        axis.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    fig.subplots_adjust(left=0.075, right=0.88, bottom=0.17, top=0.88, wspace=0.12)
    colorbar_axis = fig.add_axes([0.9, 0.21, 0.018, 0.6])
    colorbar = fig.colorbar(contour, cax=colorbar_axis)
    colorbar.set_ticks([value for value in (-1., -.1, -.01, 0., .01, .1, 1.)
                       if abs(value) <= limit * (1 + 1.e-12)])
    colorbar.set_label("$p_h-p$")
    save_figure(fig, output_dir / "boundary_pressure_error_contours")
    plt.close(fig)


def save_field_data(path, field_data):
    arrays = {}
    for method in METHODS:
        triangles, coordinates, errors = field_data[method["id"]]
        arrays[f"{method['id']}_triangles"] = triangles
        arrays[f"{method['id']}_coordinates"] = coordinates
        arrays[f"{method['id']}_errors"] = errors
    np.savez_compressed(path, **arrays)


def load_field_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Pressure-error field data not found: {path}")
    with np.load(path) as arrays:
        return {
            method["id"]: (
                arrays[f"{method['id']}_triangles"],
                arrays[f"{method['id']}_coordinates"],
                arrays[f"{method['id']}_errors"],
            )
            for method in METHODS
        }


def load_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Boundary-pressure comparison CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    integer_fields = {"mesh_n", "cells", "bdf_order", "boundary_cells", "interior_cells"}
    string_fields = {"method_id", "method", "velocity_pair"}
    for row in rows:
        for key, value in list(row.items()):
            if key in string_fields:
                continue
            row[key] = int(float(value)) if key in integer_fields else float(value)
    return rows


def write_table(rows, output_dir, *, unsteady):
    keys = ("global_pressure_error", "boundary_pressure_error",
            "interior_pressure_error", "velocity_l2_error")
    lines = [r"\noindent\begin{minipage}{\linewidth}", r"\begin{center}",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Method & $\|p-p_h\|_{\Omega}$ & $\|p-p_h\|_{\Omega_h^b}$ & $\|p-p_h\|_{\Omega_h^i}$ & $\|\bm u-\bm u_h\|_{\Omega}$ \\",
             r"\midrule"]
    for row in rows:
        lines.append(row["method"] + " & " + " & ".join(
            scientific_tex(row[key]) for key in keys) + r" \\")
    row = rows[0]
    solution = "Unsteady" if unsteady else "Steady"
    dt_tex = scientific_time_step(row["dt"])
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}",
              r"\refstepcounter{table}", r"\label{tab:boundary_pressure_comparison}",
              r"{\small", r"\noindent\textbf{Table~\thetable}\par",
              f"\\noindent {solution} manufactured-solution $L^2$ errors at $T={row['final_time']:g}$ "
              f"with BDF{row['bdf_order']}, $\\Delta t={dt_tex}$, and $n={row['mesh_n']}$. "
              r"All methods use $P_1/P_1$ elements. Pressure errors are calculated "
              r"over the full domain $\Omega$, boundary strip $\Omega_h^b$, and interior $\Omega_h^i$. "
              r"The velocity error is evaluated over the full domain.\par",
              r"}", r"\end{minipage}\par"]
    (output_dir / "boundary_pressure_comparison.tex").write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise RuntimeError(
            "This contour-producing comparison is intentionally run on one MPI rank."
        )

    output_dir = args.output_dir or OUTPUT_DIR / (
        "boundary_pressure" if args.unsteady else "boundary_pressure_steady")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "boundary_pressure_comparison.csv"
    field_path = output_dir / "boundary_pressure_error_fields.npz"

    if args.replot:
        rows = load_rows(csv_path)
        field_data = load_field_data(field_path)
    else:
        rows = []
        field_data = {}
        for method in METHODS:
            row, fields = run_method(method, args.mesh_n, args.dt, args.t_end, unsteady=args.unsteady)
            rows.append(row)
            field_data[method["id"]] = fields
        write_csv(csv_path, rows)
        save_field_data(field_path, field_data)
        metadata = {
            "configuration": {
                "geometry": "[-0.5, 0.5]^2",
                "mesh": f"{args.mesh_n}x{args.mesh_n} triangular",
                "pair": "P1/P1",
                "nu": 1.0,
                "equation": "Navier-Stokes",
                "manufactured_solution": "unsteady" if args.unsteady else "steady",
                "omega": float(np.pi),
                "dt": args.dt,
                "final_time": args.t_end,
                "bdf_order": 2 if args.unsteady else 1,
                "initial_data": "Nodal interpolants of the exact velocity and pressure at t=0; zero increment",
                "linear_rtol": 1.0e-8,
                "linear_atol": 1.0e-10,
                "linear_max_it": 1000,
                "split_linear_solves": "GMRES-AMG predictor; CG-AMG increment/recovery; CG-Jacobi mass correction",
                "split_convergence_norm": "unpreconditioned",
                "monolithic_linear_solve": "GMRES with lower Schur AMG field split",
                "monolithic_convergence_norm": "unpreconditioned",
            },
            "methods": [method["title"] for method in METHODS],
            "boundary_strip": "Union of all cells touching an exterior facet.",
            "pressure_error_convention": "Physical pressure difference p_h-p without mean removal.",
        }
        (output_dir / "boundary_pressure_comparison_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    plot_fields(field_data, output_dir)
    write_table(rows, output_dir, unsteady=args.unsteady)
    if MPI.COMM_WORLD.rank == 0:
        for row in rows:
            print(
                f"{row['method']}: global={row['global_pressure_error']:.6e}, "
                f"boundary={row['boundary_pressure_error']:.6e}, "
                f"interior={row['interior_pressure_error']:.6e}, "
                f"velocity={row['velocity_l2_error']:.6e}"
            )


if __name__ == "__main__":
    main()
