import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from dolfinx import plot
from mpi4py import MPI

ROOT_DIR = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solver.boundary import BoundaryFactory
from solver.manufactured import initialize_manufactured_state, manufactured_fields
from solver.mesh_loader import MeshHandler
from solver.monolithic_vms import MonolithicSUPGPSPGSolver
from solver.configuration import CaseConfig
from solver.reporting import configure_matplotlib, save_figure, scientific_time_step, write_csv
from solver.rpcs_vms import FlowSolver
from test_cases.manufactured.parameters import OUTPUT_DIR, SMALL_TIME_STEP
from test_cases.manufactured.run_boundary_pressure_comparison import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run small-time-step pressure stability tests.")
    parser.add_argument("--dts", type=float, nargs="+", default=[0.1, 0.01, 0.001])
    parser.add_argument("--mesh-n", type=int, default=64)
    parser.add_argument("--t-end", type=float, default=SMALL_TIME_STEP["time"]["t_end"])
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate figures from existing output fields without rerunning the calculation.",
    )
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str(CASE_DIR / path)


GROUPS = [
    {
        "id": "P1P1_unstabilized",
        "title": "P1/P1, unstabilized",
        "row_label": r"RPCS ($P_1/P_1$)",
        "velocity_degree": 1,
        "stabilization": "f",
    },
    {
        "id": "P1P1_stabilized",
        "title": "P1/P1, stabilized",
        "row_label": r"RPCS-VMS ($P_1/P_1$)",
        "velocity_degree": 1,
        "stabilization": "t",
    },
    {
        "id": "P2P1_stable",
        "title": "P2/P1 Taylor-Hood",
        "row_label": r"Monolithic $P_2/P_1$",
        "velocity_degree": 2,
        "stabilization": "f",
    },
]

PRESSURE_CMAP = "RdBu_r"
PRESSURE_LEVEL_COUNT = 33
CONTOUR_LINE_COLOR = "#1f2933"


def run_case(group, dt, mesh_n, t_end):
    cfg = CaseConfig(SMALL_TIME_STEP)
    cfg.cfg["mesh"]["domain"] = resolve(
        f"square_triangular_mesh/square_n{mesh_n}_domain.xdmf"
    )
    cfg.cfg["mesh"]["boundary"] = resolve(
        f"square_triangular_mesh/square_n{mesh_n}_boundary.xdmf"
    )
    cfg.cfg["time"]["dt"] = dt
    cfg.cfg["time"]["t_end"] = t_end
    cfg.cfg["time"]["BDF_order"] = 2
    cfg.cfg["physics"]["stabilization"] = group["stabilization"]
    cfg.cfg["solver"]["velocity_degree"] = group["velocity_degree"]

    comm = MPI.COMM_WORLD
    mh = MeshHandler(comm, cfg.cfg["mesh"]["domain"], cfg.cfg["mesh"]["boundary"])
    solver_class = MonolithicSUPGPSPGSolver if group["velocity_degree"] == 2 else FlowSolver
    solver = solver_class(comm, mh, cfg)
    boundary_data = BoundaryFactory(
        mh, solver.V, solver.Q, time_symbol=solver.time_symbol,
        mixed_velocity_space=getattr(solver, "mixed_velocity_space", None),
    ).create(cfg.cfg["boundary_conditions"])
    solver.set_bcs(boundary_data)
    initialize_manufactured_state(solver, *manufactured_fields(mh.mesh))

    t = 0.0
    step = 0
    while t < t_end - 1e-9:
        t = min(t + dt, t_end)
        step += 1
        solver.solve_step(t, use_bdf1=(step == 1))
    return solver, t, step


def local_triangles(Q):
    topology, _, geometry = plot.vtk_mesh(Q)
    triangles = []
    i = 0
    while i < len(topology):
        n = int(topology[i])
        verts = topology[i + 1 : i + 1 + n]
        if n == 3:
            triangles.append(verts)
        elif n == 4:
            triangles.append([verts[0], verts[1], verts[2]])
            triangles.append([verts[0], verts[2], verts[3]])
        i += n + 1
    return np.asarray(triangles, dtype=np.int64), geometry[:, :2]


def gather_pressure_field(solver):
    triangles, geometry = local_triangles(solver.Q)
    solver.pn.x.scatter_forward()
    values = solver.pn.x.array.copy()
    index_map = solver.Q.dofmap.index_map
    local = np.arange(len(geometry), dtype=np.int32)
    global_ids = index_map.local_to_global(local)
    payload = (triangles, geometry, values, global_ids, index_map.size_local)
    gathered = solver.comm.gather(payload, root=0)
    if solver.comm.rank != 0:
        return None, None

    return merge_pressure_partitions(gathered)


def merge_pressure_partitions(gathered):
    all_points = []
    all_values = []
    all_triangles = []
    all_ids = []
    all_owned = []
    offset = 0
    for triangles_i, geometry_i, values_i, ids_i, owned_i in gathered:
        count = len(geometry_i)
        all_points.append(geometry_i)
        all_values.append(values_i[:count])
        all_triangles.append(triangles_i + offset)
        all_ids.append(ids_i)
        all_owned.append(np.arange(count) < owned_i)
        offset += count
    points = np.vstack(all_points)
    values = np.concatenate(all_values)
    triangles = np.vstack(all_triangles)
    ids = np.concatenate(all_ids)
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    owned = np.flatnonzero(np.concatenate(all_owned))
    owners = owned[np.argsort(ids[owned])]
    if not np.array_equal(ids[owners], unique_ids):
        raise ValueError("Each pressure node must have exactly one owning MPI rank.")
    # Identify partition copies by global DOF, keeping the owner's coordinates.
    points, values = points[owners], values[owners]
    triangles = inverse[triangles]
    _, first = np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)
    triangles = triangles[np.sort(first)]
    return mtri.Triangulation(points[:, 0], points[:, 1], triangles), values


def coerce_metric_row(row):
    coerced = dict(row)
    for name in (
        "dt",
        "final_time",
        "max_abs_pressure",
        "global_pressure_error",
        "boundary_pressure_error",
        "interior_pressure_error",
        "velocity_l2_error",
    ):
        if name in coerced:
            coerced[name] = float(coerced[name])
    coerced["steps"] = int(float(coerced["steps"]))
    return coerced


def load_existing_metric_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Small-time-step pressure stability CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [coerce_metric_row(row) for row in csv.DictReader(handle)]


def save_pressure_fields(results, path):
    payload = {"group_ids": np.asarray([item["group_id"] for item in results]),
               "dts": np.asarray([item["dt"] for item in results])}
    for index, item in enumerate(results):
        triangulation = item["triangulation"]
        payload[f"points_{index}"] = np.column_stack((triangulation.x, triangulation.y))
        payload[f"triangles_{index}"] = triangulation.triangles
        payload[f"values_{index}"] = item["p_values"]
    np.savez_compressed(path, **payload)


def load_pressure_fields(path, metric_rows):
    if not path.exists():
        raise FileNotFoundError(f"Pressure-field cache not found: {path}; run the stability tests once.")
    titles = {group["id"]: group["title"] for group in GROUPS}
    results = []
    with np.load(path, allow_pickle=False) as data:
        for index, (group_id, dt) in enumerate(zip(data["group_ids"], data["dts"], strict=True)):
            points = data[f"points_{index}"]
            values = data[f"values_{index}"]
            matches = [row for row in metric_rows if row["group"] == titles[group_id]
                       and np.isclose(row["dt"], dt)]
            if len(matches) != 1 or not np.isclose(np.max(np.abs(values)), matches[0]["max_abs_pressure"]):
                raise ValueError(f"Pressure cache does not match the metrics: {group_id}, dt={dt}")
            results.append({"group_id": str(group_id), "group_title": titles[group_id],
                            "dt": float(dt), "p_values": values,
                            "triangulation": mtri.Triangulation(points[:, 0], points[:, 1],
                                                                 data[f"triangles_{index}"])})
    if len(results) != len(metric_rows):
        raise ValueError("Pressure cache and metric table have different case counts.")
    return results


def set_compact_colorbar_ticks(colorbar, contour):
    lo = float(contour.norm.vmin)
    hi = float(contour.norm.vmax)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return
    if np.isclose(lo, hi):
        colorbar.set_ticks([lo])
    else:
        colorbar.set_ticks(np.linspace(lo, hi, 5))


def pressure_color_scale(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Pressure color limits require finite, nonempty field data.")
    peak = float(np.max(np.abs(values)))
    if peak > 0:
        unit = 10.0 ** np.floor(np.log10(peak))
        pressure_scale = np.ceil(peak / unit) * unit
    else:
        pressure_scale = 1.0
    levels = np.linspace(-pressure_scale, pressure_scale, PRESSURE_LEVEL_COUNT)
    norm = TwoSlopeNorm(vmin=-pressure_scale, vcenter=0.0, vmax=pressure_scale)
    return levels, norm


def plot_rows(results, out_dir):
    configure_matplotlib()
    levels, norm = pressure_color_scale(np.concatenate([item["p_values"] for item in results]))
    for group in GROUPS:
        sub = [item for item in results if item["group_id"] == group["id"]]
        sub.sort(key=lambda item: item["dt"], reverse=True)
        fig = plt.figure(figsize=(7.2, 2.25))
        grid = fig.add_gridspec(
            1, 4, width_ratios=[1.0, 1.0, 1.0, 0.045], wspace=0.04
        )
        axes = [fig.add_subplot(grid[0, col]) for col in range(3)]
        colorbar_axis = fig.add_subplot(grid[0, 3])
        for ax, item in zip(axes, sub):
            contour = ax.tricontourf(
                item["triangulation"],
                item["p_values"],
                levels=levels,
                cmap=PRESSURE_CMAP,
                norm=norm,
            )
            ax.tricontour(
                item["triangulation"],
                item["p_values"],
                levels=10,
                colors=CONTOUR_LINE_COLOR,
                linewidths=0.25,
                alpha=0.32,
            )
            ax.set_aspect("equal")
            ax.set_title(rf"$\Delta t={scientific_time_step(item['dt'])}$")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(group["title"], y=0.98)
        cbar = fig.colorbar(contour, cax=colorbar_axis)
        set_compact_colorbar_ticks(cbar, contour)
        save_figure(
            fig,
            out_dir / f"pressure_stability_{group['id']}",
            png=True,
        )
        plt.close(fig)


def plot_matrix(results, out_dir, *, groups=None, stem="pressure_stability_matrix", color_limit=None):
    configure_matplotlib()
    groups = GROUPS if groups is None else groups
    values = np.concatenate([item["p_values"] for item in results])
    if color_limit is not None:
        if (not np.all(np.isfinite(values)) or not np.isfinite(color_limit)
                or color_limit <= 0 or np.max(np.abs(values)) > color_limit):
            raise ValueError("The shared color limit must contain all finite pressure values.")
        levels = np.linspace(-color_limit, color_limit, PRESSURE_LEVEL_COUNT)
        norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    else:
        levels, norm = pressure_color_scale(values)
    fig = plt.figure(figsize=(7.2, 6.2))
    grid = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.045],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.04,
        hspace=0.08,
    )
    axes = np.asarray(
        [[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(3)]
    )
    colorbar_axis = fig.add_subplot(grid[1, 3])
    for row, group in enumerate(groups):
        sub = [item for item in results if item["group_id"] == group["id"]]
        sub.sort(key=lambda item: item["dt"], reverse=True)
        for col, item in enumerate(sub):
            ax = axes[row, col]
            contour = ax.tricontourf(
                item["triangulation"],
                item["p_values"],
                levels=levels,
                cmap=PRESSURE_CMAP,
                norm=norm,
            )
            ax.tricontour(
                item["triangulation"],
                item["p_values"],
                levels=10,
                colors=CONTOUR_LINE_COLOR,
                linewidths=0.2,
                alpha=0.28,
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(rf"$\Delta t={scientific_time_step(item['dt'])}$")
            if col == 0:
                ax.set_ylabel(group["row_label"], rotation=90, labelpad=18)
    cbar = fig.colorbar(contour, cax=colorbar_axis)
    cbar.set_label(r"$p$")
    set_compact_colorbar_ticks(cbar, contour)
    fig.canvas.draw()
    panel_position = axes[1, -1].get_position()
    bar_position = colorbar_axis.get_position()
    colorbar_axis.set_position([
        bar_position.x0, panel_position.y0, bar_position.width, panel_position.height
    ])
    save_figure(
        fig,
        out_dir / stem,
        png=True,
    )
    plt.close(fig)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    out_dir = OUTPUT_DIR / "small_time_step"
    results = []
    metric_rows = []

    if args.replot:
        if rank == 0:
            metric_rows = load_existing_metric_rows(
                out_dir / "small_time_step_pressure_stability.csv"
            )
            results = load_pressure_fields(out_dir / "small_time_step_pressure_fields.npz", metric_rows)
            plot_rows(results, out_dir)
            plot_matrix(results, out_dir)
        return

    for group in GROUPS:
        for dt in args.dts:
            if rank == 0:
                print(f"Running pressure stability: {group['title']}, dt={dt:g}")
                sys.stdout.flush()
            solver, final_time, steps = run_case(group, dt, args.mesh_n, args.t_end)
            metrics = compute_metrics(solver.mh, solver.vn, solver.pn, final_time)
            triangulation, p_values = gather_pressure_field(solver)
            local_max = float(np.max(np.abs(solver.pn.x.array)))
            max_abs_pressure = comm.allreduce(local_max, op=MPI.MAX)
            if rank == 0:
                results.append(
                    {
                        "group_id": group["id"],
                        "group_title": group["title"],
                        "dt": dt,
                        "triangulation": triangulation,
                        "p_values": p_values,
                    }
                )
                metric_rows.append(
                    {
                        "group": group["title"],
                        "viscosity": float(solver.nu.value),
                        "dt": dt,
                        "steps": steps,
                        "final_time": final_time,
                        "max_abs_pressure": max_abs_pressure,
                        **{key: metrics[key] for key in (
                            "global_pressure_error", "boundary_pressure_error",
                            "interior_pressure_error", "velocity_l2_error")},
                    }
                )
                print(
                    f"{group['title']} dt={dt:g} max|p|={max_abs_pressure:.4e} "
                    f"pressure L2={metrics['global_pressure_error']:.4e}"
                )
                sys.stdout.flush()

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_pressure_fields(results, out_dir / "small_time_step_pressure_fields.npz")
        write_csv(
            out_dir / "small_time_step_pressure_stability.csv",
            metric_rows,
            [
                "group",
                "viscosity",
                "dt",
                "steps",
                "final_time",
                "max_abs_pressure",
                "global_pressure_error",
                "boundary_pressure_error",
                "interior_pressure_error",
                "velocity_l2_error",
            ],
        )
        plot_rows(results, out_dir)
        plot_matrix(results, out_dir)


if __name__ == "__main__":
    main()
