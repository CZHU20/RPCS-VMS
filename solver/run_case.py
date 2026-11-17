import csv
import math
import sys
from pathlib import Path

from mpi4py import MPI

from solver.boundary import BoundaryFactory
from solver.mesh_loader import MeshHandler
from solver.manufactured import initialize_manufactured_state, manufactured_fields
from solver.configuration import CaseConfig
from solver.rpcs_vms import FlowSolver
from solver.utils import ForceHistoryWriter, ResultsWriter, Timer


def run_case(parameters, continuity_monitor=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cfg = CaseConfig(parameters)

    if rank == 0:
        print(f"--- RPCS case: {cfg.cfg['case_name']} ---")
        sys.stdout.flush()

    mh = MeshHandler(comm, cfg.cfg["mesh"]["domain"], cfg.cfg["mesh"]["boundary"])
    solver = FlowSolver(comm, mh, cfg)
    boundary_data = BoundaryFactory(
        mh, solver.V, solver.Q, time_symbol=solver.time_symbol
    ).create(cfg.cfg["boundary_conditions"])
    solver.set_bcs(boundary_data)
    if "manufactured_solution" in cfg.cfg:
        initialize_manufactured_state(solver, *manufactured_fields(mh.mesh))

    writer = ResultsWriter(
        comm, f"{cfg.cfg['output']['dir']}/solution", solver.vn, solver.pn
    )
    writer.save(0.0)

    force_writer = None
    if solver.force_monitor_enabled:
        force_output_path = Path(solver.force_output_file)
        if not force_output_path.is_absolute():
            force_output_path = Path(cfg.cfg["output"]["dir"]) / force_output_path
        force_writer = ForceHistoryWriter(comm, str(force_output_path))
        force_writer.write(solver.evaluate_force_observables(0.0, step=0))

    t = 0.0
    step = 0
    dt = solver.dt
    t_end = solver.T
    bdf_order = cfg.cfg["time"]["BDF_order"]
    total_steps = int(math.ceil(t_end / dt))
    output_cfg = cfg.cfg["output"]
    output_end_time = output_cfg.get("end_time")
    progress_frequency = output_cfg.get("progress_frequency")
    if progress_frequency is None:
        progress_frequency = max(1, math.ceil(total_steps / 100))

    continuity_stream = None
    continuity_writer = None
    if continuity_monitor:
        import ufl
        from dolfinx import fem
        divergence_form = fem.form(ufl.div(solver.vn)**2 * mh.dx)
        gradient_form = fem.form(ufl.inner(ufl.grad(solver.vn), ufl.grad(solver.vn)) * mh.dx)
        if rank == 0:
            continuity_stream = (Path(output_cfg["dir"]) / "continuity.csv").open("w", newline="")
            continuity_writer = csv.DictWriter(continuity_stream, fieldnames=(
                "step", "time", "divergence_norm", "gradient_norm", "divergence_to_gradient"))
            continuity_writer.writeheader()

    timer = Timer()
    timer.start(total_steps_expected=total_steps)

    while t < t_end - 1e-9:
        t = min(t + dt, t_end)
        step += 1
        use_bdf1 = bdf_order == 1 or step == 1

        norm_dv, norm_dp = solver.solve_step(t, use_bdf1)

        force_sample = None
        if force_writer is not None and (
            step % solver.force_monitor_frequency == 0 or step == total_steps
        ):
            force_sample = solver.evaluate_force_observables(t, step=step)

        should_log_progress = (
            step == 1
            or step == total_steps
            or step % progress_frequency == 0
        )
        continuity = None
        if continuity_monitor and should_log_progress:
            div2 = comm.allreduce(fem.assemble_scalar(divergence_form), op=MPI.SUM)
            grad2 = comm.allreduce(fem.assemble_scalar(gradient_form), op=MPI.SUM)
            if not math.isfinite(div2) or not math.isfinite(grad2):
                raise RuntimeError("Nonfinite velocity-gradient diagnostics.")
            continuity = math.sqrt(max(div2, 0) / max(grad2, 1e-300))
            if rank == 0:
                continuity_writer.writerow(dict(step=step, time=t,
                    divergence_norm=math.sqrt(max(div2, 0)), gradient_norm=math.sqrt(max(grad2, 0)),
                    divergence_to_gradient=continuity))
                continuity_stream.flush()
        if rank == 0 and should_log_progress:
            elapsed_str, avg_step_time, eta_str = timer.timestamp(step)
            print("-" * 90)
            print(
                f"{'Step':<8} | {'Time':<10} | {'Wall Time':<12} | {'dU':<10} | {'dP':<10} | {'s/step':<8} | {'ETA':<10}"
            )
            print(
                f"{step:<8d} | {t:<10.4f} | {elapsed_str:<12} | {norm_dv:<10.2e} | {norm_dp:<10.2e} | {avg_step_time:<8.4f} | {eta_str:<10}"
            )
            if force_sample is not None:
                print(
                    f"{'Forces':<8} | Cd={force_sample['drag_coefficient']:<10.6f} | Cl={force_sample['lift_coefficient']:<10.6f} | dP={force_sample['pressure_difference']:<10.6f}"
                )
            if continuity is not None:
                print(f"{'Continuity':<10} | ||div u||/||grad u||={continuity:.6e}")
            sys.stdout.flush()

        if force_writer is not None:
            force_writer.write(force_sample)

        in_output_window = t >= output_cfg["start_time"] and (
            output_end_time is None or t <= output_end_time + 1e-12
        )
        if (
            in_output_window
            and step % output_cfg["frequency"] == 0
        ) or step == total_steps:
            writer.save(t)

    if rank == 0:
        print("-" * 80)
        print(f"Simulation completed. Total Wall Time: {timer.timestamp(step)[0]}")
    writer.close()
    if force_writer is not None:
        force_writer.close()
    if continuity_stream is not None:
        continuity_stream.close()
