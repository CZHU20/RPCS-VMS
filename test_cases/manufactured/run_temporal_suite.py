"""Run each temporal-refinement configuration in an independent MPI job."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
from queue import Queue
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).with_name("run_temporal_convergence.py")
REPORT = Path(__file__).resolve().parent / "output/convergence"
DTS = (0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125)
FIELDS = ("study", "label", "mesh_n", "velocity_degree", "stabilization",
          "bdf_order", "dt", "u_l2", "u_l2_rate", "u_h1", "u_h1_rate",
          "p_l2", "p_l2_rate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--np", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cpu-groups", nargs="+",
                        help="Disjoint MPICH CPU lists, one per simultaneous job.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.np < 1 or args.jobs < 1:
        parser.error("--np and --jobs must be positive")
    if args.jobs > 1 and not args.cpu_groups:
        parser.error("concurrent jobs require explicit disjoint --cpu-groups")
    groups = args.cpu_groups or [None]
    if len(groups) != args.jobs:
        parser.error("provide one CPU group per job")
    if args.cpu_groups:
        ids = [int(cpu) for group in groups for cpu in group.split(",")]
        if len(ids) != len(set(ids)) or any(len(g.split(",")) != args.np for g in groups):
            parser.error("CPU groups must be disjoint and each contain --np entries")
    python = Path(os.environ.get("FENICS_PYTHON", sys.executable)).resolve()
    launcher = os.environ.get("MPIEXEC") or str(python.parent / "mpiexec")
    if not Path(launcher).is_file():
        launcher = shutil.which("mpiexec")
    if not launcher:
        raise RuntimeError("No MPI launcher found; set MPIEXEC.")
    version = subprocess.check_output([launcher, "--version"], text=True).lower()
    is_mpich = "hydra" in version or "mpich" in version
    if args.cpu_groups and not is_mpich:
        parser.error("explicit CPU lists currently require MPICH/Hydra")
    env = os.environ.copy()
    env.update({name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    )})
    env["PYTHONUNBUFFERED"] = "1"
    if platform.system() == "Darwin":
        env.setdefault("FI_PROVIDER", "tcp")
    inputs = [ROOT / "solver" / name for name in (
        "rpcs_vms.py", "stabilization.py", "manufactured.py", "configuration.py", "boundary.py",
        "mesh_loader.py", "expression.py",
    )] + [WORKER, WORKER.with_name("parameters.py")]
    signature = hashlib.sha256(b"".join(p.read_bytes() for p in inputs)).hexdigest()
    directory = REPORT / "temporal_cases"
    directory.mkdir(parents=True, exist_ok=True)
    slots = Queue()
    for group in groups:
        slots.put(group)

    def run(case):
        study, order, n, degree, dt = case
        name = f"{study}-bdf{order}-n{n}-q{degree}-dt{dt:g}"
        output = directory / f"{name}.json"
        metadata = directory / f"{name}.metadata.json"
        if args.resume and output.exists() and metadata.exists():
            if json.loads(metadata.read_text())["input_sha256"] == signature:
                return json.loads(output.read_text())
        group = slots.get()
        try:
            binding = []
            if group is not None:
                binding = ["-bind-to", f"user:{group}"]
            elif platform.system() == "Linux":
                binding = ["-bind-to", "core"] if is_mpich else ["--bind-to", "core"]
            command = [launcher, *binding, "-n", str(args.np), str(python),
                       str(WORKER), "--single-case", "--study", study,
                       "--bdf-order", str(order), "--mesh-n", str(n),
                       "--velocity-degree", str(degree), "--dts", str(dt),
                       "--result-file", str(output)]
            print(f"Running {name}", flush=True)
            with (directory / f"{name}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            row = json.loads(output.read_text())
            metadata.write_text(json.dumps({
                "input_sha256": signature, "mpi_ranks": args.np,
                "equation": "NS", "omega": 3.141592653589793,
                "scalar_acceleration": "predictor", "cpu_group": group,
            }, indent=2) + "\n")
            print(f"Completed {name}: {row['u_l2']:.6e}, {row['p_l2']:.6e}", flush=True)
            return row
        finally:
            slots.put(group)

    cases = [(study, order, n, degree, dt)
             for study, order, n, degree in (
                 ("main", 1, 256, 1), ("main", 2, 512, 1),
                 ("pair", 2, 256, 1), ("pair", 2, 256, 2),
             ) for dt in DTS]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(run, cases))
    rows += [{**row, "study": "pair"} for row in rows
             if row["study"] == "main" and row["bdf_order"] == 2]
    with (REPORT / "temporal_convergence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    subprocess.run([str(python), str(WORKER), "--mode", "replot"],
                   cwd=ROOT, env=env, check=True)
    print("Completed 24 independent solves and 30 plotted configurations.", flush=True)


if __name__ == "__main__":
    main()
