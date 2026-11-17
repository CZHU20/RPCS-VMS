"""Isolate the two pressure residuals in the paper's small-time-step test."""

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import dolfinx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from solver.boundary import BoundaryFactory
from solver.configuration import CaseConfig
from solver.manufactured import initialize_manufactured_state, manufactured_fields
from solver.mesh_loader import MeshHandler
from solver.reporting import configure_matplotlib, save_figure, scientific_tex as scientific_error, write_csv
from solver.rpcs_vms import FlowSolver
from test_cases.manufactured.parameters import OUTPUT_DIR, SMALL_TIME_STEP
from test_cases.manufactured.run_boundary_pressure_comparison import compute_metrics
from test_cases.manufactured.run_stabilization_effect import (
    CONTOUR_LINE_COLOR, PRESSURE_CMAP, PRESSURE_LEVEL_COUNT,
    gather_pressure_field, scientific_time_step,
    set_compact_colorbar_ticks,
    GROUPS as STABILITY_GROUPS, run_case as run_stability_case,
)

VARIANTS = {
    "increment_only": (True, False),
    "recovery_only": (False, True),
    "full": (True, True),
}
GROUPS = [
    {"id": "increment_only", "row_label": "No recovery stabilization"},
    {"id": "recovery_only", "row_label": "No increment stabilization"},
    {"id": "full", "row_label": "Full RPCS-VMS"},
]
TABLE_METHODS = (
    ("rpcs", r"RPCS ($P_1/P_1$)"),
    ("increment_only", "RPCS-VMS (No recovery stabilization)"),
    ("recovery_only", "RPCS-VMS (No increment stabilization)"),
    ("full", r"RPCS-VMS ($P_1/P_1$)"),
    ("monolithic", r"Monolithic ($P_2/P_1$)"),
)
TABLE_METRICS = ("global_pressure_error", "boundary_pressure_error",
                 "interior_pressure_error", "velocity_l2_error")


def configuration(n, dt, end, variant):
    parameters = dict(SMALL_TIME_STEP)
    parameters["mesh"] = {
        kind: str(Path(__file__).parent / f"square_triangular_mesh/square_n{n}_{kind}.xdmf")
        for kind in ("domain", "boundary")
    }
    parameters["time"] = {"dt": dt, "t_end": end, "BDF_order": 2}
    increment, recovery = VARIANTS[variant]
    parameters["solver"] = {
        **SMALL_TIME_STEP["solver"], "velocity_degree": 1,
        "pressure_recovery": "rotational",
        "pressure_increment_stabilization": increment,
        "pressure_recovery_stabilization": recovery,
    }
    return CaseConfig(parameters)


def initialize(cfg, comm=MPI.COMM_WORLD):
    mh = MeshHandler(comm, cfg.cfg["mesh"]["domain"], cfg.cfg["mesh"]["boundary"])
    solver = FlowSolver(comm, mh, cfg)
    solver.set_bcs(BoundaryFactory(mh, solver.V, solver.Q, time_symbol=solver.time_symbol)
                   .create(cfg.cfg["boundary_conditions"]))
    initialize_manufactured_state(solver, *manufactured_fields(mh.mesh))
    return solver


class Diagnostics:
    def __init__(self, solver):
        self.solver = solver
        u, p = manufactured_fields(solver.mh.mesh, solver.time_symbol)
        ep, eu = solver.pn - p, solver.vn - u
        dx = ufl.Measure("dx", domain=solver.mh.mesh, metadata={"quadrature_degree": 8})
        self.forms = {
            key: fem.form(expr * dx) for key, expr in {
                "pressure_l2": ep**2,
                "pressure_h1_seminorm": ufl.inner(ufl.grad(ep), ufl.grad(ep)),
                "velocity_l2": ufl.inner(eu, eu),
                "divergence_l2": ufl.div(solver.vn)**2,
            }.items()
        }

    def evaluate(self):
        return {
            key: float(np.sqrt(max(0., self.solver.comm.allreduce(
                fem.assemble_scalar(form), op=MPI.SUM))))
            for key, form in self.forms.items()
        }


def run_case(n, dt, end, variant, output):
    comm = MPI.COMM_WORLD
    if not np.isfinite(dt) or not np.isfinite(end) or dt <= 0 or end <= 0:
        raise ValueError("Time step and end time must be finite and positive.")
    count = int(round(end / dt))
    if count < 1 or not np.isclose(count * dt, end, rtol=0, atol=1e-12):
        raise ValueError("Positive end time must be an integer multiple of positive dt.")
    cfg = configuration(n, dt, end, variant)
    solver = initialize(cfg, comm)
    diagnostics = Diagnostics(solver)
    histories = []
    min_reason = 100
    started = time.perf_counter()
    for step in range(1, count + 1):
        solver.solve_step(step * dt, use_bdf1=(step == 1), compute_norms=False)
        min_reason = min(min_reason, *(entry["reason"] for entry in solver.last_step_stats["substeps"].values()))
        metric = diagnostics.evaluate()
        if not all(np.isfinite(value) for value in metric.values()) or min_reason <= 0:
            raise RuntimeError(f"Failed diagnostics or linear solve: {variant}, dt={dt}, step={step}")
        if comm.rank == 0:
            histories.append({"step": step, "time": step * dt, **metric})
            if step % max(1, count // 5) == 0:
                print(f"  {variant}, dt={dt:g}: {step}/{count}", flush=True)
    elapsed = comm.allreduce(time.perf_counter() - started, op=MPI.MAX)
    min_reason = comm.allreduce(min_reason, op=MPI.MIN)
    final_metrics = compute_metrics(solver.mh, solver.vn, solver.pn, count * dt)
    triangulation, pressure = gather_pressure_field(solver)
    result, row = None, None
    if comm.rank == 0:
        row = {
            "variant": variant, "dt": dt, "mesh_n": n, "final_time": count * dt,
            "steps": count, "viscosity": cfg.cfg["physics"]["nu"],
            "increment_stabilization": VARIANTS[variant][0],
            "recovery_stabilization": VARIANTS[variant][1],
            **metric, **final_metrics,
            "max_abs_pressure": float(np.max(np.abs(pressure))),
            "wall_seconds": elapsed, "minimum_ksp_reason": min_reason,
        }
        write_csv(output / f"{variant}_dt{dt:g}_history.csv", histories)
        (output / f"{variant}_dt{dt:g}_configuration.json").write_text(json.dumps(cfg.cfg, indent=2) + "\n")
        result = {"group_id": variant, "dt": dt, "triangulation": triangulation, "p_values": pressure}
        print(f"DONE {variant}, dt={dt:g}: "
              f"pressure L2={metric['pressure_l2']:.8e}, "
              f"velocity L2={metric['velocity_l2']:.8e}, div L2={metric['divergence_l2']:.8e}", flush=True)
    del diagnostics, solver
    gc.collect()
    return result, row


def run_references(n, dt, end):
    rows = []
    for variant, group in (("rpcs", STABILITY_GROUPS[0]), ("monolithic", STABILITY_GROUPS[2])):
        solver, final_time, steps = run_stability_case(group, dt, n, end)
        metrics = compute_metrics(solver.mh, solver.vn, solver.pn, final_time)
        if solver.comm.rank == 0:
            rows.append({"variant": variant, "dt": dt, "mesh_n": n,
                         "final_time": end, "steps": steps, **metrics})
        del solver
        gc.collect()
    return rows


def save_fields(path, results):
    payload = {"group_ids": np.asarray([item["group_id"] for item in results]),
               "dts": np.asarray([item["dt"] for item in results])}
    for index, item in enumerate(results):
        tri = item["triangulation"]
        payload[f"points_{index}"] = np.column_stack((tri.x, tri.y))
        payload[f"triangles_{index}"] = tri.triangles
        payload[f"values_{index}"] = item["p_values"]
    np.savez_compressed(path, **payload)


def load_results(output):
    with (output / "pressure_stage_ablation.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in row.keys() - {"variant", "increment_stabilization", "recovery_stabilization"}:
            row[key] = float(row[key])
    results = []
    with np.load(output / "pressure_stage_fields.npz", allow_pickle=False) as data:
        for index, (group, dt) in enumerate(zip(data["group_ids"], data["dts"], strict=True)):
            points = data[f"points_{index}"]
            values = data[f"values_{index}"]
            tri = mtri.Triangulation(points[:, 0], points[:, 1], data[f"triangles_{index}"])
            matches = [row for row in rows if row["variant"] == group and np.isclose(row["dt"], dt)]
            if len(matches) != 1 or not np.isclose(np.max(np.abs(values)), matches[0]["max_abs_pressure"], rtol=1e-12):
                raise ValueError("Cached pressure fields and metric rows disagree.")
            results.append({"group_id": str(group), "dt": float(dt), "triangulation": tri, "p_values": values})
    if len(results) != len(rows):
        raise ValueError("Cached pressure fields and metric rows have different lengths.")
    return results, rows


def plot_pressure_comparison(results, output, dt):
    configure_matplotlib()
    fields = []
    for group in GROUPS:
        matches = [item for item in results if item["group_id"] == group["id"]
                   and np.isclose(item["dt"], dt, rtol=1e-12, atol=0)]
        if len(matches) != 1:
            raise ValueError("Each variant must have one field at the selected time step.")
        fields.append(matches[0])
    values = np.concatenate([item["p_values"] for item in fields])
    if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 1.0:
        raise ValueError("The pressure fields must fit the common [-1, 1] color scale.")
    fig = plt.figure(figsize=(7.2, 2.25))
    grid = fig.add_gridspec(1, 4, width_ratios=[1., 1., 1., .045], wspace=.08)
    axes = [fig.add_subplot(grid[0, col]) for col in range(3)]
    colorbar_axis = fig.add_subplot(grid[0, 3])
    levels = np.linspace(-1., 1., PRESSURE_LEVEL_COUNT)
    norm = TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1.)
    for ax, group, field in zip(axes, GROUPS, fields, strict=True):
        contour = ax.tricontourf(field["triangulation"], field["p_values"],
                                 levels=levels, cmap=PRESSURE_CMAP, norm=norm)
        ax.tricontour(field["triangulation"], field["p_values"], levels=10,
                      colors=CONTOUR_LINE_COLOR, linewidths=.2, alpha=.28)
        ax.set_aspect("equal")
        ax.set_title(group["row_label"])
        ax.set_xticks([])
        ax.set_yticks([])
    colorbar = fig.colorbar(contour, cax=colorbar_axis)
    colorbar.set_label(r"$p$")
    set_compact_colorbar_ticks(colorbar, contour)
    fig.canvas.draw()
    panel = axes[-1].get_position()
    bar = colorbar_axis.get_position()
    colorbar_axis.set_position([bar.x0, panel.y0, bar.width, panel.height])
    save_figure(fig, output / "pressure_stage_ablation_matrix", png=True)
    plt.close(fig)


def render(output, results, rows, reference_rows):
    dt = min(item["dt"] for item in results)
    plot_pressure_comparison(results, output, dt)
    selected = []
    for variant, label in TABLE_METHODS:
        matches = [row for row in rows + reference_rows if row["variant"] == variant
                   and np.isclose(row["dt"], dt, rtol=1e-12, atol=0)]
        if len(matches) != 1:
            raise ValueError("Each variant must have one metric row per time step.")
        selected.append((label, matches[0]))
    lines = [r"\noindent\begin{minipage}{\linewidth}", r"\begin{center}",
             r"\begin{tabular}{@{}p{0.32\linewidth}cccc@{}}", r"\toprule",
             r"Method & $\|p-p_h\|_{\Omega}$ & $\|p-p_h\|_{\Omega_h^b}$ & $\|p-p_h\|_{\Omega_h^i}$ & $\|\bm u-\bm u_h\|_{\Omega}$ \\",
             r"\midrule"]
    for title, row in selected:
        lines.append(title + " & " + " & ".join(scientific_error(row[key]) for key in TABLE_METRICS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\refstepcounter{table}",
              r"\label{tab:pressure_stage_ablation}", r"{\small",
              r"\noindent\textbf{Table~\thetable}\par",
              f"\\noindent Unsteady manufactured-solution $L^2$ errors at $T={selected[0][1]['final_time']:g}$ "
              f"with BDF2, $\\Delta t={scientific_time_step(dt)}$, and $n={int(selected[0][1]['mesh_n'])}$. "
              r"The error definitions and boundary strip are the same as in Table~\ref{tab:boundary_pressure_comparison}. "
              r"All RPCS formulations use $P_1/P_1$ elements. Monolithic uses unstabilized $P_2/P_1$ elements.\par",
              r"}", r"\end{minipage}\par"]
    (output / "pressure_stage_ablation.tex").write_text("\n".join(lines) + "\n")
    presentation = {
        "dt": dt, "variants": [group["id"] for group in GROUPS],
        "color_range": [-1., 1.],
        "table_metrics": list(TABLE_METRICS),
        "table_methods": [variant for variant, _ in TABLE_METHODS],
        "pressure_error_convention": "Physical pressure difference p_h-p without mean removal.",
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                         for name in ("pressure_stage_fields.npz", "pressure_stage_ablation.csv",
                                      "pressure_stage_references.csv")
                         if (output / name).exists()},
    }
    (output / "presentation.json").write_text(json.dumps(presentation, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dts", type=float, nargs="+", default=[0.1, 0.01, 0.001])
    parser.add_argument("--mesh-n", type=int, default=64)
    parser.add_argument("--t-end", type=float, default=SMALL_TIME_STEP["time"]["t_end"])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR / "pressure_stage_ablation")
    parser.add_argument("--replot", action="store_true")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    output = args.output_dir.resolve()
    if comm.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    comm.barrier()
    if args.replot:
        if comm.rank == 0:
            with (output / "pressure_stage_references.csv").open(newline="") as handle:
                reference_rows = list(csv.DictReader(handle))
            reference_rows = [{key: value if key == "variant" else float(value)
                               for key, value in row.items()} for row in reference_rows]
            render(output, *load_results(output), reference_rows)
        return
    if len(set(args.dts)) != len(args.dts) or any(not np.isfinite(dt) or dt <= 0 for dt in args.dts):
        parser.error("Time steps must be finite, positive and distinct.")
    results, rows = [], []
    for variant in VARIANTS:
        for dt in args.dts:
            result, row = run_case(args.mesh_n, dt, args.t_end, variant, output)
            if comm.rank == 0:
                results.append(result)
                rows.append(row)
    reference_rows = run_references(args.mesh_n, min(args.dts), args.t_end)
    if comm.rank == 0:
        write_csv(output / "pressure_stage_ablation.csv", rows)
        write_csv(output / "pressure_stage_references.csv", reference_rows)
        save_fields(output / "pressure_stage_fields.npz", results)
        render(output, results, rows, reference_rows)
        source_paths = [Path(__file__), ROOT / "solver/rpcs_vms.py", ROOT / "solver/configuration.py",
                        ROOT / "solver/manufactured.py", ROOT / "solver/reporting.py",
                        Path(__file__).with_name("parameters.py"),
                        Path(__file__).with_name("run_boundary_pressure_comparison.py"),
                        Path(__file__).with_name("run_stabilization_effect.py")]
        metadata = {"platform": platform.platform(), "python": sys.version,
                    "dolfinx": dolfinx.__version__, "petsc": PETSc.Sys.getVersion(),
                    "mpi_ranks": comm.size, "mesh_n": args.mesh_n, "dts": args.dts,
                    "final_time": args.t_end, "diagnostic_quadrature_degree": 8,
                    "table_quadrature_degree": 5,
                    "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in source_paths}}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
