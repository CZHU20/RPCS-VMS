"""Run the manufactured-solution test group."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--np", type=int, default=2, help="MPI ranks for each calculation.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("spatial", "temporal", "boundary", "small-step", "ablation"),
        default=("spatial", "temporal", "boundary", "small-step", "ablation"),
    )
    return parser.parse_args()


def mpi_command(ranks, script):
    launcher = os.environ.get("MPIEXEC") or shutil.which("mpiexec")
    if not launcher:
        raise RuntimeError("No MPI launcher found; set MPIEXEC.")
    return [launcher, "-n", str(ranks), sys.executable, str(script)]


def main():
    args = parse_args()
    commands = []
    if "spatial" in args.only:
        commands.append(mpi_command(args.np, HERE / "run_spatial_convergence.py"))
    if "temporal" in args.only:
        commands.append(
            [sys.executable, str(HERE / "run_temporal_suite.py"), "--np", str(args.np)]
        )
    if "boundary" in args.only:
        commands.append([sys.executable, str(HERE / "run_boundary_pressure_comparison.py")])
    if "small-step" in args.only:
        commands.append(mpi_command(args.np, HERE / "run_stabilization_effect.py"))
    if "ablation" in args.only:
        commands.append(mpi_command(args.np, HERE / "run_pressure_stage_ablation.py"))
    for command in commands:
        print("$", " ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
