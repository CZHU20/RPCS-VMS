"""Run all RPCS-VMS numerical test groups and generate their plots."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "test_cases"
GROUPS = ("manufactured", "branched-pipe", "cylinder-2d", "cylinder-3d", "efficiency")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--np", type=int, default=2, help="MPI ranks for non-efficiency tests.")
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=GROUPS)
    return parser.parse_args()


def mpi_command(ranks, script, *extra):
    launcher = os.environ.get("MPIEXEC") or shutil.which("mpiexec")
    if not launcher:
        raise RuntimeError("No MPI launcher found; set MPIEXEC.")
    return [launcher, "-n", str(ranks), sys.executable, str(script), *extra]


def execute(command):
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def main():
    args = parse_args()
    if "manufactured" in args.only:
        execute([sys.executable, TESTS / "manufactured/run.py", "--np", args.np])

    if "branched-pipe" in args.only:
        execute(mpi_command(args.np, TESTS / "branched_pipe/run.py"))
        execute(mpi_command(args.np, TESTS / "branched_pipe/run_monolithic_p2p1_steady.py"))
        execute([sys.executable, TESTS / "branched_pipe/postprocess.py"])

    if "cylinder-2d" in args.only:
        execute(mpi_command(args.np, TESTS / "cylinder_2d/run.py"))
        execute([sys.executable, TESTS / "cylinder_2d/postprocess.py"])

    if "cylinder-3d" in args.only:
        execute(
            mpi_command(
                args.np,
                TESTS / "cylinder_3d/run.py",
                "--continuity-monitor",
            )
        )
        execute([sys.executable, TESTS / "cylinder_3d/postprocess.py"])

    if "efficiency" in args.only:
        execute(["bash", TESTS / "efficiency/run.sh"])


if __name__ == "__main__":
    main()
