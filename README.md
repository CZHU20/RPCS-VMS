# RPCS-VMS flow solver

This repository contains a FEniCSx implementation of the residual-based variational multiscale rotational pressure-correction method (RPCS-VMS), together with manufactured and benchmark flow tests.

Calculated fields, numerical summaries, and figures are not distributed. Every test writes its own data and plots below a case-local `output/` directory.

## Numerical method

The RPCS-VMS time step consists of four stages:

1. Intermediate-velocity prediction with momentum-residual and divergence stabilization.
2. Stabilized pressure-increment solution.
3. Consistent-mass velocity correction.
4. Rotational pressure recovery with residual stabilization.

The implementation of these stages is in `solver/rpcs_vms.py`. The equal-order monolithic SUPG/PSPG comparison is in `solver/monolithic_vms.py`.

## Required packages

The calculations were developed with the following versions:

| Package | Version |
|---|---|
| Python | 3.12 |
| FEniCSx/DOLFINx | 0.10.0 |
| PETSc/petsc4py | 3.24.3 |
| PyVista | 0.46.5 |

The remaining required packages are MPI, mpi4py, NumPy, SciPy, Matplotlib, h5py, meshio, and Gmsh. A conda-forge installation can be created with:

```bash
conda create -n rpcs-vms -c conda-forge \
  python=3.12 fenics-dolfinx=0.10.0 petsc=3.24.3 \
  mpi4py petsc4py numpy scipy matplotlib h5py meshio gmsh pyvista=0.46.5
conda activate rpcs-vms
```

Use an MPI launcher from the same environment as DOLFINx. Set `MPIEXEC` if `mpiexec` is not on the command path.

## Repository layout

```text
solver/                         RPCS-VMS and monolithic solver components
test_cases/manufactured/        Convergence and pressure tests
test_cases/branched_pipe/       Branched-pipe flow test
test_cases/cylinder_2d/         Two-dimensional cylinder test
test_cases/cylinder_3d/         Three-dimensional cylinder test
test_cases/efficiency/          Three-dimensional timing comparison
run_all.py                      Run every test group
```

Each test directory has a `parameters.py` file containing its physical and numerical parameters. Meshes and external benchmark curves are retained as inputs. RPCS-VMS calculation results are always generated locally.

All scripts resolve meshes, benchmark data, and output directories relative to their own location. They can therefore be launched from any working directory.

## Manufactured-solution tests

The manufactured group contains spatial convergence, temporal convergence, near-boundary pressure, small-time-step pressure, and pressure-stage isolation tests:

```bash
python test_cases/manufactured/run.py --np 2
```

Run a subset with, for example:

```bash
python test_cases/manufactured/run.py --np 2 --only spatial boundary
```

After the calculations, regenerate the plots without solving again:

```bash
python test_cases/manufactured/plot.py
```

The non-rotational comparison uses the update `p_new = p_old + phi`,
implemented by its mass equation. The near-boundary test uses T=0.5.
The small-time-step and pressure-stage isolation tests use T=1.
The near-boundary comparison uses P1/P1, n=64, BDF2, and dt=0.001:

```bash
python test_cases/manufactured/run_boundary_pressure_comparison.py
```

This writes `output/boundary_pressure/`. All manufactured-solution pressure
errors are evaluated without mean removal. The spatial and temporal
convergence studies retain T=1. The optional `--steady` argument runs a
separate steady near-boundary diagnostic.

Run or replot the pressure-stage isolation separately:

```bash
python test_cases/manufactured/run.py --np 2 --only ablation
python test_cases/manufactured/run_pressure_stage_ablation.py --replot
```

The three variants retain rotational recovery and independently switch the
increment and recovery residual contributions. The pressure comparison shows
the three variants at the smallest time step, using a shared color scale.
See [the experiment guide](test_cases/manufactured/PRESSURE_STAGE_ABLATION.md)
for settings and output definitions. Focused checks can be run with:

```bash
python -m unittest test_cases.manufactured.test_pressure_stage_ablation test_cases.manufactured.test_pressure_stage_reporting -v
```

## Branched-pipe test

```bash
mpiexec -n 2 python test_cases/branched_pipe/run.py
mpiexec -n 2 python test_cases/branched_pipe/run_monolithic_p2p1_steady.py
python test_cases/branched_pipe/postprocess.py
```

## Two-dimensional cylinder test

```bash
mpiexec -n 2 python test_cases/cylinder_2d/run.py
python test_cases/cylinder_2d/postprocess.py
```

## Three-dimensional cylinder test

```bash
mpiexec -n 8 python test_cases/cylinder_3d/run.py --continuity-monitor
python test_cases/cylinder_3d/postprocess.py
```

This is the most expensive accuracy calculation. It writes full three-dimensional velocity and pressure fields and therefore also requires substantial disk space.

## Efficiency comparison

The timing comparison runs RPCS-VMS and the equal-order monolithic method in separate MPI jobs:

```bash
bash test_cases/efficiency/run.sh
```

The defaults are eight MPI ranks, 100 time steps per method, and three independent runs. The summary reports arithmetic means and the wall-time range. These settings can be changed through `MPI_RANKS`, `STEPS`, and `REPEATS`.

## Run everything

```bash
python run_all.py --np 2
```

This includes the full three-dimensional calculation and the eight-rank timing comparison. To run selected groups:

```bash
python run_all.py --np 2 --only manufactured cylinder-2d
```

## Generated files

Every test creates `test_cases/<name>/output/`. These directories can contain HDF5 fields, XDMF descriptors, CSV summaries, JSON diagnostics, PDF figures, and PNG figures. They are ignored by Git and may be removed before a fresh run.

The FEATFLOW curves under `reference_data/` are external benchmark results from the [DFG 2D-3 benchmark page](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark3_re100.html) and the [3D reference-data page](https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_flow3d/dfg_flow3d_reference.html). 
