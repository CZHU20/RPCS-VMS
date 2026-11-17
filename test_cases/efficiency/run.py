import argparse
import copy
import csv
import gc
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import dolfinx
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc


ROOT_DIR = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CASE_DIR / "output"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solver.monolithic_vms import MonolithicSUPGPSPGSolver
from solver.configuration import CaseConfig
from test_cases.cylinder_3d.parameters import CASE as CYLINDER_CASE
from test_cases.efficiency.residual_policy import (
    POLICIES, audit_ksp, make_solver as make_residual_solver, solver_ksps,
)


METHODS = ("monolithic-SUPG/PSPG", "RPCS-VMS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MPI efficiency comparison on the 3D DFG cylinder case."
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="cylinder3d_100_8r")
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--linear-rtol", type=float, default=1.0e-8)
    parser.add_argument("--linear-atol", type=float, default=1.0e-10)
    parser.add_argument("--gmres-restart", type=int, default=100)
    parser.add_argument("--residual-norm", choices=POLICIES, default="preconditioned")
    parser.add_argument(
        "--progress-frequency",
        type=int,
        default=0,
        help="Print progress every N steps; zero avoids timed-region I/O.",
    )
    return parser.parse_args()


def configure_method(base_cfg, method, steps, args):
    cfg = copy.deepcopy(base_cfg)
    cfg.cfg["time"]["t_end"] = cfg.cfg["time"]["dt"] * steps

    # Monitoring forms are prepared during setup, but observables are evaluated
    # only once after the timed integration region.
    cfg.cfg.setdefault("force_monitor", {})["enabled"] = True
    cfg.cfg.setdefault("output", {})["frequency"] = steps + 1

    solver_cfg = cfg.cfg.setdefault("solver", {})
    solver_cfg["linear_rtol"] = args.linear_rtol
    solver_cfg["linear_atol"] = args.linear_atol
    solver_cfg["linear_max_it"] = 1000
    solver_cfg["gmres_restart"] = args.gmres_restart
    solver_cfg["efficiency_residual_norm"] = args.residual_norm
    if method == "RPCS-VMS":
        cfg.cfg["physics"]["formula"] = "rpcs"
        cfg.cfg["physics"]["stabilization"] = "t"
        solver_cfg["velocity_degree"] = 1
        solver_cfg["corrector_mass"] = "consistent"
    elif method == "monolithic-SUPG/PSPG":
        cfg.cfg["physics"]["formula"] = "monolithic_supg_pspg"
        cfg.cfg["physics"]["stabilization"] = "t"
        solver_cfg["convective_stabilization"] = True
        solver_cfg["monolithic_pc"] = "schur_amg"
        solver_cfg["monolithic_ksp_type"] = "gmres"
        solver_cfg["monolithic_preconditioner_matrix"] = "incremental"
        solver_cfg["monolithic_schur_fact_type"] = "lower"
    else:
        raise ValueError(f"Unknown method: {method}")
    return cfg


def make_solver(comm, cfg):
    method = (METHODS[0] if cfg.cfg["physics"]["formula"] == "monolithic_supg_pspg"
              else METHODS[1])
    policy = cfg.cfg["solver"].get("efficiency_residual_norm", "preconditioned")
    return make_residual_solver(comm, cfg, method, policy)


def global_dofs(space):
    index_map = space.dofmap.index_map
    return int(index_map.size_global * space.dofmap.index_map_bs)


def step_row(method, run_id, step, t, stats):
    substeps = stats.get("substeps", {})
    if substeps:
        assembly_time = sum(item["assembly_time"] for item in substeps.values())
        linear_solve_time = sum(item["solve_time"] for item in substeps.values())
    else:
        assembly_time = stats.get("assembly_time", 0.0)
        linear_solve_time = stats.get("solve_time", 0.0)
    row = {
        "method": method,
        "run_id": run_id,
        "step": step,
        "time": t,
        "step_wall_time": np.nan,
        "assembly_time": float(assembly_time),
        "linear_solve_time": float(linear_solve_time),
        "linear_iterations": int(stats.get("linear_iterations", 0)),
        "nonlinear_iterations": int(stats.get("nonlinear_iterations", 0)),
    }
    for name in ("predictor", "phi", "corrector", "pressure"):
        row[f"{name}_iterations"] = int(substeps.get(name, {}).get("iterations", 0))
    return row


def mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def solver_cfg_value(cfg, key, default=None):
    return cfg.cfg.get("solver", {}).get(key, default)


def run_method(comm, base_cfg, method, steps, progress_frequency, args):
    rank = comm.rank
    cfg = configure_method(base_cfg, method, steps, args)
    dt = float(cfg.cfg["time"]["dt"])
    bdf_order = int(cfg.cfg["time"]["BDF_order"])

    comm.Barrier()
    setup_start = MPI.Wtime()
    solver = make_solver(comm, cfg)
    comm.Barrier()
    setup_local = MPI.Wtime() - setup_start
    setup_wall = comm.reduce(setup_local, op=MPI.MAX, root=0)

    velocity_dofs = global_dofs(solver.V)
    pressure_dofs = global_dofs(solver.Q)
    local_step_times = np.empty(steps, dtype=np.float64)
    rows = [] if rank == 0 else None

    comm.Barrier()
    solve_start = MPI.Wtime()
    for step in range(1, steps + 1):
        t = step * dt
        step_start = MPI.Wtime()
        use_bdf1 = bdf_order == 1 or step == 1
        if isinstance(solver, MonolithicSUPGPSPGSolver):
            solver.solve_step(t, use_bdf1=use_bdf1, compute_norms=False)
        else:
            solver.solve_step(t, use_bdf1=use_bdf1, compute_norms=False)
        local_step_times[step - 1] = MPI.Wtime() - step_start

        if rank == 0:
            rows.append(step_row(method, args.run_id, step, t, solver.last_step_stats))
            if progress_frequency > 0 and (
                step == 1 or step == steps or step % progress_frequency == 0
            ):
                print(f"{method}: {step}/{steps}", flush=True)

    comm.Barrier()
    solve_local = MPI.Wtime() - solve_start
    solve_wall = comm.reduce(solve_local, op=MPI.MAX, root=0)

    order = 1 if bdf_order == 1 or steps == 1 else 2
    audits = {name: audit_ksp(ksp) for name, ksp in solver_ksps(solver, order).items()}
    final_observables = solver.evaluate_force_observables(steps * dt, step=steps)

    max_step_times = np.empty_like(local_step_times) if rank == 0 else None
    comm.Reduce(local_step_times, max_step_times, op=MPI.MAX, root=0)

    velocity_norm = solver.vn.x.petsc_vec.norm()
    pressure_norm = solver.pn.x.petsc_vec.norm()
    if not np.isfinite(velocity_norm) or not np.isfinite(pressure_norm):
        raise RuntimeError(f"{method} produced a non-finite final solution.")

    solver_configuration = solver.effective_solver_configuration()

    if rank != 0:
        del solver
        gc.collect()
        return None, None

    for row, elapsed in zip(rows, max_step_times, strict=True):
        row["step_wall_time"] = float(elapsed)

    slug = "rpcs" if method == "RPCS-VMS" else "monolithic"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / f"{args.output_prefix}_{slug}_residual_audit.json"
    audit_path.write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")
    primary_audit = audits.get("predictor", audits.get("monolithic"))
    summary = {
        "method": method,
        "run_id": args.run_id,
        "mpi_ranks": comm.size,
        "steps": steps,
        "dt": dt,
        "setup_wall_time": float(setup_wall),
        "solve_wall_time": float(solve_wall),
        "total_wall_time": float(setup_wall + solve_wall),
        "mean_step_wall_time": float(solve_wall / steps),
        "mean_rank_max_step_time": float(np.mean(max_step_times)),
        "median_step_wall_time": float(np.median(max_step_times)),
        "rank0_assembly_time": float(sum(row["assembly_time"] for row in rows)),
        "rank0_linear_solve_time": float(
            sum(row["linear_solve_time"] for row in rows)
        ),
        "velocity_dofs": velocity_dofs,
        "pressure_dofs": pressure_dofs,
        "total_dofs": velocity_dofs + pressure_dofs,
        "total_linear_iterations": int(sum(row["linear_iterations"] for row in rows)),
        "mean_linear_iterations_per_step": mean(rows, "linear_iterations"),
        "total_nonlinear_iterations": int(
            sum(row["nonlinear_iterations"] for row in rows)
        ),
        "mean_nonlinear_iterations_per_step": mean(rows, "nonlinear_iterations"),
        "mean_predictor_iterations_per_step": mean(rows, "predictor_iterations"),
        "mean_phi_iterations_per_step": mean(rows, "phi_iterations"),
        "mean_corrector_iterations_per_step": mean(rows, "corrector_iterations"),
        "mean_pressure_iterations_per_step": mean(rows, "pressure_iterations"),
        "final_velocity_vector_norm": float(velocity_norm),
        "final_pressure_vector_norm": float(pressure_norm),
        "outer_ksp_type": solver_configuration.get("outer_ksp_type", "split"),
        "pc_type": solver_configuration.get("pc_type", "split"),
        "fieldsplit_0_ksp_type": solver_configuration.get("fieldsplit_0_ksp_type", ""),
        "fieldsplit_0_pc_type": solver_configuration.get("fieldsplit_0_pc_type", ""),
        "fieldsplit_1_ksp_type": solver_configuration.get("fieldsplit_1_ksp_type", ""),
        "fieldsplit_1_pc_type": solver_configuration.get("fieldsplit_1_pc_type", ""),
        "predictor_ksp_type": solver_configuration.get("predictor_ksp_type", ""),
        "predictor_pc_type": solver_configuration.get("predictor_pc_type", ""),
        "phi_ksp_type": solver_configuration.get("phi_ksp_type", ""),
        "phi_pc_type": solver_configuration.get("phi_pc_type", ""),
        "corrector_ksp_type": solver_configuration.get("corrector_ksp_type", ""),
        "corrector_pc_type": solver_configuration.get("corrector_pc_type", ""),
        "pressure_ksp_type": solver_configuration.get("pressure_ksp_type", ""),
        "pressure_pc_type": solver_configuration.get("pressure_pc_type", ""),
        "stabilization_form": solver_configuration.get(
            "stabilization_form", "RPCS-VMS" if method == "RPCS-VMS" else "none"
        ),
        "stabilization_coefficient_mode": solver_cfg_value(
            cfg, "stabilization_coefficient", "dynamic"
        ),
        "stabilization_alpha": str(
            solver_cfg_value(cfg, "stabilization_alpha", "auto")
        ),
        "preconditioner_matrix_mode": solver_configuration.get(
            "preconditioner_matrix_mode", "split"
        ),
        "schur_fact_type": solver_configuration.get("schur_fact_type", "split"),
        "final_drag_coefficient": float(final_observables["drag_coefficient"]),
        "final_lift_coefficient": float(final_observables["lift_coefficient"]),
        "final_pressure_difference": float(final_observables["pressure_difference"]),
        "linear_rtol": args.linear_rtol,
        "linear_atol": args.linear_atol,
        "gmres_restart": args.gmres_restart,
        "linear_residual_norm": primary_audit["norm"],
        "outer_preconditioning_side": primary_audit["side"],
    }
    del solver
    gc.collect()
    return rows, summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, summaries):
    mpi_ranks = int(summaries[0]["mpi_ranks"])
    lines = [
        f"# {mpi_ranks}-rank DFG3D-3 efficiency comparison",
        "",
        "Timed region: solver time integration only; no solution output, force "
        "integration, plotting, per-step barrier, or per-step MPI timing reduction. "
        "Final observables are evaluated after timing.",
        "",
        "| Method | total wall (s) | solve wall (s) | mean step (s) | "
        "mean linear it/step | final Cd | final Cl | final dP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['total_wall_time']:.6f} | "
            f"{row['solve_wall_time']:.6f} | {row['mean_step_wall_time']:.6f} | "
            f"{row['mean_linear_iterations_per_step']:.6f} | "
            f"{row['final_drag_coefficient']:.8f} | "
            f"{row['final_lift_coefficient']:.8f} | "
            f"{row['final_pressure_difference']:.8f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def system_metadata():
    metadata = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        output = subprocess.run(
            ["lscpu", "--json"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        metadata["lscpu"] = json.loads(output)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        metadata["lscpu"] = None
    return metadata


def process_affinity():
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return None


def runtime_metadata(comm, args, rank_affinities):
    config_path = args.config.resolve()
    try:
        config_label = config_path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        config_label = config_path.name
    mpi_description = next(
        (line.strip() for line in MPI.Get_library_version().splitlines() if line.strip()),
        "unknown",
    )
    return {
        "entry_point": "test_cases/efficiency/run.py",
        "platform": platform.platform(),
        "python": sys.version,
        "dolfinx": dolfinx.__version__,
        "petsc": ".".join(map(str, PETSc.Sys.getVersion())),
        "mpi_library": mpi_description,
        "mpi_ranks": comm.size,
        "rank_cpu_affinities": rank_affinities,
        "system": system_metadata(),
        "config": config_label,
        "steps": args.steps,
        "methods": args.methods,
        "run_id": args.run_id,
        "linear_rtol": args.linear_rtol,
        "linear_atol": args.linear_atol,
        "gmres_restart": args.gmres_restart,
        "residual_norm": args.residual_norm,
        "monolithic_outer_ksp": "gmres",
        "monolithic_pressure_split_ksp": "preonly",
        "monolithic_preconditioner_matrix": "incremental",
        "monolithic_schur_fact_type": "lower",
        "monolithic_stabilization": "SUPG/PSPG+grad-div",
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "FI_PROVIDER",
                "RPCS_MPI_BINDING",
            )
        },
    }


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    if args.steps < 1:
        raise ValueError("--steps must be positive.")
    if args.expected_ranks > 0 and comm.size != args.expected_ranks:
        raise ValueError(
            f"Expected {args.expected_ranks} MPI ranks, but got {comm.size}."
        )

    base_cfg = CaseConfig(CYLINDER_CASE)
    all_rows = []
    summaries = []

    for method in args.methods:
        if comm.rank == 0:
            print(f"Running {method} on {comm.size} MPI ranks", flush=True)
        rows, summary = run_method(
            comm, base_cfg, method, args.steps, args.progress_frequency, args
        )
        if comm.rank == 0:
            all_rows.extend(rows)
            summaries.append(summary)
            write_csv(args.output_dir / f"{args.output_prefix}_steps.csv", all_rows)
            write_csv(args.output_dir / f"{args.output_prefix}_summary.csv", summaries)
            write_report(args.output_dir / f"{args.output_prefix}_report.md", summaries)
        comm.Barrier()

    rank_affinities = comm.gather(process_affinity(), root=0)
    if comm.rank == 0:
        metadata_path = args.output_dir / f"{args.output_prefix}_metadata.json"
        metadata_path.write_text(
            json.dumps(runtime_metadata(comm, args, rank_affinities), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Results: {args.output_dir / f'{args.output_prefix}_summary.csv'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        if MPI.COMM_WORLD.rank == 0:
            print(f"ERROR: {exc}", file=sys.stderr)
        MPI.COMM_WORLD.Abort(2)
