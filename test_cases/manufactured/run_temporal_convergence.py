import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI
from petsc4py import PETSc

ROOT_DIR = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solver.boundary import BoundaryFactory
from solver.manufactured import initialize_manufactured_state, manufactured_fields
from solver.mesh_loader import MeshHandler
from solver.configuration import CaseConfig
from solver.reporting import (
    COLORS,
    configure_matplotlib,
    rate,
    save_figure,
    write_csv,
)
from solver.rpcs_vms import FlowSolver
from test_cases.manufactured.parameters import OUTPUT_DIR, UNSTEADY


def parse_args():
    parser = argparse.ArgumentParser(description="Run temporal MMS convergence cases.")
    parser.add_argument(
        "--mode",
        choices=["all", "main", "pair-q1", "pair-q2", "replot"],
        default="all",
        help="Run a subset of the temporal convergence matrix and merge with existing CSV.",
    )
    parser.add_argument(
        "--dts",
        type=float,
        nargs="+",
        default=[0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125],
        help="Time steps for the main temporal convergence study.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Explicitly reuse existing pair results instead of recomputing selected cases.",
    )
    parser.add_argument("--bdf1-mesh-n", type=int, default=256)
    parser.add_argument("--single-case", action="store_true",
                        help="Run one case in a fresh MPI job and save its result as JSON.")
    parser.add_argument("--mesh-n", type=int, default=256)
    parser.add_argument("--bdf-order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--velocity-degree", type=int, choices=[1, 2], default=1)
    parser.add_argument("--study", choices=["main", "pair"], default="main")
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--bdf2-mesh-n", type=int, default=512)
    parser.add_argument("--pair-q1-mesh-ns", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--pair-q2-mesh-n", type=int, default=256)
    parser.add_argument(
        "--pair-dts",
        type=float,
        nargs="+",
        default=[0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125],
        help="Time steps for Q2/Q1 and refined Q1/Q1 pair comparison.",
    )
    return parser.parse_args()


def coerce_row(row):
    coerced = dict(row)
    coerced["mesh_n"] = int(float(coerced["mesh_n"]))
    coerced["velocity_degree"] = int(float(coerced["velocity_degree"]))
    coerced["bdf_order"] = int(float(coerced["bdf_order"]))
    for name in ("dt", "u_l2", "u_l2_rate", "u_h1", "u_h1_rate", "p_l2", "p_l2_rate"):
        value = coerced.get(name, np.nan)
        coerced[name] = float(value) if value not in ("", "nan", "NaN") else np.nan
    return coerced


def row_key(row):
    return (
        row["study"],
        row["label"],
        row["mesh_n"],
        row["velocity_degree"],
        row["stabilization"],
        row["bdf_order"],
        float(row["dt"]),
    )


def load_existing_rows(path):
    if not path.exists():
        return []
    import csv

    with path.open(newline="", encoding="utf-8") as handle:
        return [coerce_row(row) for row in csv.DictReader(handle)]


def cleanup_runtime(comm):
    gc.collect()
    PETSc.garbage_cleanup(comm)
    gc.collect()


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str(CASE_DIR / path)


def exact_fields(mh, t):
    return manufactured_fields(mh.mesh, t)


def compute_error_norms(mh, vn, pn, t):
    u_exact, p_exact = exact_fields(mh, t)
    error_u_l2_form = fem.form(ufl.inner(vn - u_exact, vn - u_exact) * mh.dx)
    error_u_h1_semi_form = fem.form(
        ufl.inner(ufl.grad(vn) - ufl.grad(u_exact), ufl.grad(vn) - ufl.grad(u_exact))
        * mh.dx
    )
    error_p_l2_form = fem.form(
        (pn - p_exact) ** 2 * mh.dx
    )
    error_u_l2_sq = mh.mesh.comm.allreduce(
        fem.assemble_scalar(error_u_l2_form), op=MPI.SUM
    )
    error_u_h1_semi_sq = mh.mesh.comm.allreduce(
        fem.assemble_scalar(error_u_h1_semi_form), op=MPI.SUM
    )
    error_p_l2_sq = mh.mesh.comm.allreduce(
        fem.assemble_scalar(error_p_l2_form), op=MPI.SUM
    )
    return (
        np.sqrt(error_u_l2_sq),
        np.sqrt(error_u_l2_sq + error_u_h1_semi_sq),
        np.sqrt(error_p_l2_sq),
    )


def mesh_paths(mesh_n):
    return (
        resolve(f"square_quad_mesh/square_n{mesh_n}_domain.xdmf"),
        resolve(f"square_quad_mesh/square_n{mesh_n}_boundary.xdmf"),
    )


def run_case(dt, bdf_order, mesh_n, velocity_degree=1, stabilization="t"):
    cfg = CaseConfig(UNSTEADY)
    cfg.cfg["mesh"]["domain"], cfg.cfg["mesh"]["boundary"] = mesh_paths(mesh_n)
    cfg.cfg["time"]["dt"] = dt
    cfg.cfg["time"]["t_end"] = 1.0
    cfg.cfg["time"]["BDF_order"] = bdf_order
    cfg.cfg["physics"]["stabilization"] = stabilization
    cfg.cfg["solver"]["velocity_degree"] = velocity_degree

    comm = MPI.COMM_WORLD
    mh = MeshHandler(comm, cfg.cfg["mesh"]["domain"], cfg.cfg["mesh"]["boundary"])
    solver = FlowSolver(comm, mh, cfg)
    boundary_data = BoundaryFactory(
        mh, solver.V, solver.Q, time_symbol=solver.time_symbol
    ).create(cfg.cfg["boundary_conditions"])
    solver.set_bcs(boundary_data)
    initialize_manufactured_state(solver, *exact_fields(mh, 0.0))

    t = 0.0
    step = 0
    while t < cfg.cfg["time"]["t_end"] - 1e-9:
        t = min(t + dt, cfg.cfg["time"]["t_end"])
        step += 1
        solver.solve_step(t, use_bdf1=(bdf_order == 1 or step == 1))
    return compute_error_norms(mh, solver.vn, solver.pn, t)


def add_rates(rows, group_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: row["dt"], reverse=True)
        previous = None
        for row in group_rows:
            if previous is None:
                row["u_l2_rate"] = np.nan
                row["u_h1_rate"] = np.nan
                row["p_l2_rate"] = np.nan
            else:
                row["u_l2_rate"] = rate(
                    previous["u_l2"], row["u_l2"], previous["dt"], row["dt"]
                )
                row["u_h1_rate"] = rate(
                    previous["u_h1"], row["u_h1"], previous["dt"], row["dt"]
                )
                row["p_l2_rate"] = rate(
                    previous["p_l2"], row["p_l2"], previous["dt"], row["dt"]
                )
            previous = row


def plot_main(rows, out_dir):
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharex=True)
    metrics = [
        ("u_l2", r"$\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L^2}$"),
        ("u_h1", r"$\|\boldsymbol{u}-\boldsymbol{u}_h\|_{H^1}$"),
        ("p_l2", r"$\|p-p_h\|_{L^2}$"),
    ]
    styles = {
        1: dict(marker="o", color=COLORS["blue"], label="BDF1"),
        2: dict(marker="s", color=COLORS["orange"], label="BDF2"),
    }
    for ax, (metric, ylabel) in zip(axes, metrics):
        for order, style in styles.items():
            sub = [row for row in rows if row["study"] == "main" and row["bdf_order"] == order]
            sub.sort(key=lambda row: row["dt"], reverse=True)
            dts = np.array([row["dt"] for row in sub])
            errors = np.array([row[metric] for row in sub])
            ax.loglog(dts, errors, **style, mfc="white")
            ref_power = order
            ref_base = errors[1] / dts[1] ** ref_power if len(errors) > 1 else errors[0]
            ax.loglog(
                dts,
                ref_base * dts**ref_power,
                "--",
                color=style["color"],
                linewidth=0.9,
                alpha=0.65,
            )
        ax.set_xlabel(r"$\Delta t$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.invert_xaxis()
    legend_handles = [
        Line2D([], [], marker="o", color=COLORS["blue"], mfc="white", label="BDF1"),
        Line2D([], [], marker="s", color=COLORS["orange"], mfc="white", label="BDF2"),
        Line2D([], [], linestyle="--", color=COLORS["blue"], label=r"$O(\Delta t)$"),
        Line2D([], [], linestyle="--", color=COLORS["orange"], label=r"$O(\Delta t^2)$"),
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="lower left",
        ncol=1,
        frameon=False,
        handlelength=1.5,
        borderaxespad=0.25,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save_figure(fig, out_dir / "temporal_convergence_plot")
    plt.close(fig)


def plot_pair(rows, out_dir):
    style_by_label = {
        "Q1/Q1, n=256": dict(marker="o", color=COLORS["blue"]),
        "Q1/Q1, n=512": dict(marker="s", color=COLORS["green"]),
        "Q2/Q1, n=256": dict(marker="^", color=COLORS["orange"]),
    }
    labels = [
        label
        for label in style_by_label
        if any(row["study"] == "pair" and row["label"] == label for row in rows)
    ]
    if not labels:
        return

    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharex=True)
    metrics = [
        ("u_l2", r"$\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L^2}$"),
        ("u_h1", r"$\|\boldsymbol{u}-\boldsymbol{u}_h\|_{H^1}$"),
        ("p_l2", r"$\|p-p_h\|_{L^2}$"),
    ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for label in labels:
            style = style_by_label[label]
            sub = [
                row for row in rows if row["study"] == "pair" and row["label"] == label
            ]
            sub.sort(key=lambda row: row["dt"], reverse=True)
            if not sub:
                continue
            ax.loglog(
                [row["dt"] for row in sub],
                [row[metric] for row in sub],
                label=label,
                mfc="white",
                **style,
            )
        ax.set_xlabel(r"$\Delta t$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.invert_xaxis()
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save_figure(fig, out_dir / "temporal_step_size_comparison")
    plt.close(fig)


def write_outputs(rows, out_dir):
    by_key = {}
    for row in rows:
        by_key[row_key(row)] = row
    rows = list(by_key.values())
    add_rates(rows, ["study", "label", "bdf_order"])
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "temporal_convergence.csv",
        rows,
        [
            "study",
            "label",
            "mesh_n",
            "velocity_degree",
            "stabilization",
            "bdf_order",
            "dt",
            "u_l2",
            "u_l2_rate",
            "u_h1",
            "u_h1_rate",
            "p_l2",
            "p_l2_rate",
        ],
    )
    plot_main(rows, out_dir)
    plot_pair(rows, out_dir)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    out_dir = OUTPUT_DIR / "convergence"
    if args.single_case:
        if len(args.dts) != 1 or args.result_file is None:
            raise ValueError("--single-case requires one --dts value and --result-file.")
        dt = args.dts[0]
        degree = args.velocity_degree
        stabilization = "t" if degree == 1 else "f"
        errors = run_case(dt, args.bdf_order, args.mesh_n, degree, stabilization)
        if not np.all(np.isfinite(errors)):
            raise RuntimeError("The temporal case produced non-finite error norms.")
        if rank == 0:
            row = {
                "study": args.study, "label": f"Q{degree}/Q1, n={args.mesh_n}",
                "mesh_n": args.mesh_n, "velocity_degree": degree,
                "stabilization": stabilization, "bdf_order": args.bdf_order,
                "dt": dt, **dict(zip(("u_l2", "u_h1", "p_l2"), errors)),
            }
            args.result_file.parent.mkdir(parents=True, exist_ok=True)
            args.result_file.write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        return
    csv_path = out_dir / "temporal_convergence.csv"
    rows = load_existing_rows(csv_path) if args.mode in {"pair-q1", "pair-q2", "replot"} else []
    existing_pair_keys = {
        (row["label"], row["dt"])
        for row in rows
        if args.resume and row["study"] == "pair" and row["bdf_order"] == 2
    }

    main_cases = [
        (1, args.bdf1_mesh_n),
        (2, args.bdf2_mesh_n),
    ]
    if args.mode in {"all", "main"}:
        rows = []
        for bdf_order, mesh_n in main_cases:
            for dt in args.dts:
                if rank == 0:
                    print(
                        f"Running temporal convergence: Q1/Q1, n={mesh_n}, "
                        f"BDF{bdf_order}, dt={dt}"
                    )
                    sys.stdout.flush()
                u_l2, u_h1, p_l2 = run_case(
                    dt, bdf_order, mesh_n, velocity_degree=1, stabilization="t"
                )
                cleanup_runtime(comm)
                if rank == 0:
                    rows.append(
                        {
                            "study": "main",
                            "label": f"Q1/Q1, n={mesh_n}",
                            "mesh_n": mesh_n,
                            "velocity_degree": 1,
                            "stabilization": "t",
                            "bdf_order": bdf_order,
                            "dt": dt,
                            "u_l2": u_l2,
                            "u_h1": u_h1,
                            "p_l2": p_l2,
                        }
                    )
                    print(
                        f"BDF{bdf_order} dt={dt:.6g} "
                        f"uL2={u_l2:.4e} uH1={u_h1:.4e} pL2={p_l2:.4e}"
                    )
                    sys.stdout.flush()

    if args.mode == "all" or (args.mode == "pair-q1" and args.resume):
        for row in list(rows):
            if (
                row["study"] == "main"
                and row["bdf_order"] == 2
                and row["dt"] in set(args.pair_dts)
                and row["mesh_n"] in set(args.pair_q1_mesh_ns)
                and (row["label"], row["dt"]) not in existing_pair_keys
            ):
                copied = dict(row)
                copied["study"] = "pair"
                rows.append(copied)
                existing_pair_keys.add((copied["label"], copied["dt"]))

    if args.mode in {"all", "pair-q1"}:
        pair_cases = [
            (f"Q1/Q1, n={mesh_n}", mesh_n, 1, "t")
            for mesh_n in args.pair_q1_mesh_ns
        ]
    elif args.mode == "pair-q2":
        pair_cases = []
    else:
        pair_cases = []

    if args.mode in {"all", "pair-q2"}:
        pair_cases.append(
            (f"Q2/Q1, n={args.pair_q2_mesh_n}", args.pair_q2_mesh_n, 2, "f")
        )

    for label, mesh_n, degree, stabilization in pair_cases:
        for dt in args.pair_dts:
            if (label, dt) in existing_pair_keys:
                continue
            if rank == 0:
                print(f"Running temporal pair comparison: {label}, BDF2, dt={dt}")
                sys.stdout.flush()
            u_l2, u_h1, p_l2 = run_case(dt, 2, mesh_n, degree, stabilization)
            cleanup_runtime(comm)
            if rank == 0:
                rows.append(
                    {
                        "study": "pair",
                        "label": label,
                        "mesh_n": mesh_n,
                        "velocity_degree": degree,
                        "stabilization": stabilization,
                        "bdf_order": 2,
                        "dt": dt,
                        "u_l2": u_l2,
                        "u_h1": u_h1,
                        "p_l2": p_l2,
                    }
                )
                existing_pair_keys.add((label, dt))
                print(
                    f"{label} BDF2 dt={dt:.6g} "
                    f"uL2={u_l2:.4e} uH1={u_h1:.4e} pL2={p_l2:.4e}"
                )
                sys.stdout.flush()

    if rank == 0:
        write_outputs(rows, out_dir)


if __name__ == "__main__":
    main()
