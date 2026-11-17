import argparse
import csv
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
from test_cases.manufactured.parameters import OUTPUT_DIR, STEADY


def parse_args():
    parser = argparse.ArgumentParser(description="Run spatial MMS convergence cases.")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate the plot from an existing output CSV without rerunning the calculation.",
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256],
        help="Mesh resolutions to run.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=["P1P1", "Q1Q1"],
        choices=["P1P1", "Q1Q1"],
        help="Element/mesh families to run.",
    )
    return parser.parse_args()


def coerce_row(row):
    coerced = dict(row)
    coerced["n"] = int(float(coerced["n"]))
    for name in ("h", "u_l2", "u_l2_rate", "u_h1", "u_h1_rate", "p_l2", "p_l2_rate"):
        value = coerced.get(name, np.nan)
        coerced[name] = float(value) if value not in ("", "nan", "NaN") else np.nan
    return coerced


def load_existing_rows(path):
    if not path.exists():
        raise FileNotFoundError(f"Spatial convergence CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [coerce_row(row) for row in csv.DictReader(handle)]


def resolve(path):
    path = Path(path)
    if path.is_absolute():
        return str(path)
    return str(CASE_DIR / path)


def exact_fields(mh):
    return manufactured_fields(mh.mesh)


def compute_error_norms(mh, vn, pn):
    u_exact, p_exact = exact_fields(mh)
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
    error_u_l2 = np.sqrt(error_u_l2_sq)
    error_u_h1 = np.sqrt(error_u_l2_sq + error_u_h1_semi_sq)
    error_p_l2 = np.sqrt(
        mh.mesh.comm.allreduce(fem.assemble_scalar(error_p_l2_form), op=MPI.SUM)
    )
    return error_u_l2, error_u_h1, error_p_l2


def mesh_paths(family, n):
    if family == "Q1Q1":
        mesh_dir = "square_quad_mesh"
    else:
        mesh_dir = "square_triangular_mesh"
    return (
        resolve(f"{mesh_dir}/square_n{n}_domain.xdmf"),
        resolve(f"{mesh_dir}/square_n{n}_boundary.xdmf"),
    )


def run_case(family, n):
    cfg = CaseConfig(STEADY)
    cfg.cfg["mesh"]["domain"], cfg.cfg["mesh"]["boundary"] = mesh_paths(family, n)
    cfg.cfg["physics"]["stabilization"] = "t"
    cfg.cfg["solver"]["velocity_degree"] = 1

    comm = MPI.COMM_WORLD
    mh = MeshHandler(comm, cfg.cfg["mesh"]["domain"], cfg.cfg["mesh"]["boundary"])
    solver = FlowSolver(comm, mh, cfg)
    boundary_data = BoundaryFactory(
        mh, solver.V, solver.Q, time_symbol=solver.time_symbol
    ).create(cfg.cfg["boundary_conditions"])
    solver.set_bcs(boundary_data)
    initialize_manufactured_state(solver, *exact_fields(mh))

    t = 0.0
    step = 0
    while t < cfg.cfg["time"]["t_end"] - 1e-9:
        t = min(t + cfg.cfg["time"]["dt"], cfg.cfg["time"]["t_end"])
        step += 1
        solver.solve_step(t, use_bdf1=True)

    return compute_error_norms(mh, solver.vn, solver.pn)


def add_rates(rows):
    by_family = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    for family_rows in by_family.values():
        family_rows.sort(key=lambda row: row["h"], reverse=True)
        previous = None
        for row in family_rows:
            if previous is None:
                row["u_l2_rate"] = np.nan
                row["u_h1_rate"] = np.nan
                row["p_l2_rate"] = np.nan
            else:
                row["u_l2_rate"] = rate(previous["u_l2"], row["u_l2"], previous["h"], row["h"])
                row["u_h1_rate"] = rate(previous["u_h1"], row["u_h1"], previous["h"], row["h"])
                row["p_l2_rate"] = rate(previous["p_l2"], row["p_l2"], previous["h"], row["h"])
            previous = row


def plot_results(rows, out_dir):
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharex=True)
    metrics = [
        ("u_l2", r"$\|\boldsymbol{u}-\boldsymbol{u}_h\|_{L^2}$"),
        ("u_h1", r"$\|\boldsymbol{u}-\boldsymbol{u}_h\|_{H^1}$"),
        ("p_l2", r"$\|p-p_h\|_{L^2}$"),
    ]
    styles = {
        "P1P1": dict(marker="o", color=COLORS["blue"], label=r"P1/P1"),
        "Q1Q1": dict(marker="s", color=COLORS["orange"], label=r"Q1/Q1"),
    }
    reference_labels = {
        "u_l2": r"$O(h^2)$",
        "u_h1": r"$O(h)$",
        "p_l2": r"$O(h^{1.5})$",
    }
    for ax, (metric, ylabel) in zip(axes, metrics):
        for family, style in styles.items():
            family_rows = [row for row in rows if row["family"] == family]
            family_rows.sort(key=lambda row: row["h"], reverse=True)
            hs = np.array([row["h"] for row in family_rows])
            errors = np.array([row[metric] for row in family_rows])
            ax.loglog(hs, errors, **style, mfc="white")
        ref_h = np.array(sorted({row["h"] for row in rows}, reverse=True))
        ref_power = 1 if metric == "u_h1" else (1.5 if metric == "p_l2" else 2)
        ref_base = np.median([row[metric] / row["h"] ** ref_power for row in rows[-4:]])
        ref_line = ax.loglog(
            ref_h,
            ref_base * ref_h**ref_power,
            "--",
            color=COLORS["gray"],
            linewidth=1.0,
            label=reference_labels[metric],
        )
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.invert_xaxis()
        ax.legend(
            handles=ref_line,
            loc="upper right",
            frameon=False,
            handlelength=1.6,
            borderaxespad=0.25,
        )

    element_handles = [
        Line2D([], [], marker="o", color=COLORS["blue"], mfc="white", label=r"P1/P1"),
        Line2D([], [], marker="s", color=COLORS["orange"], mfc="white", label=r"Q1/Q1"),
    ]
    first_reference_legend = axes[0].legend_
    axes[0].legend(
        handles=element_handles,
        loc="lower left",
        ncol=1,
        frameon=False,
        handlelength=1.6,
        borderaxespad=0.25,
    )
    axes[0].add_artist(first_reference_legend)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save_figure(fig, out_dir / "spatial_convergence_plot")
    plt.close(fig)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    out_dir = OUTPUT_DIR / "convergence"

    if args.replot:
        if rank == 0:
            rows = load_existing_rows(out_dir / "spatial_convergence.csv")
            add_rates(rows)
            plot_results(rows, out_dir)
        return

    rows = []
    for family in args.families:
        for n in args.ns:
            if rank == 0:
                print(f"Running spatial convergence: {family}, n={n}")
                sys.stdout.flush()
            u_l2, u_h1, p_l2 = run_case(family, n)
            if rank == 0:
                rows.append(
                    {
                        "family": family,
                        "n": n,
                        "h": 1.0 / n,
                        "u_l2": u_l2,
                        "u_h1": u_h1,
                        "p_l2": p_l2,
                    }
                )
                print(
                    f"{family:5s} n={n:<4d} h={1.0/n:.5f} "
                    f"uL2={u_l2:.4e} uH1={u_h1:.4e} pL2={p_l2:.4e}"
                )
                sys.stdout.flush()

    if rank == 0:
        add_rates(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(
            out_dir / "spatial_convergence.csv",
            rows,
            [
                "family",
                "n",
                "h",
                "u_l2",
                "u_l2_rate",
                "u_h1",
                "u_h1_rate",
                "p_l2",
                "p_l2_rate",
            ],
        )
        plot_results(rows, out_dir)


if __name__ == "__main__":
    main()
