# Pressure-stage isolation

This experiment uses the paper's small-time-step manufactured Navier--Stokes
problem without changes to the exact solution, forcing, or boundary data.

## Settings

- Triangular mesh from a 64 x 64 square grid, continuous P1/P1 elements.
- Viscosity 1, temporal frequency pi, final time 1.
- Time steps 0.1, 0.01, and 0.001, BDF2 with a BDF1 first step.
- Exact initial velocity and pressure, zero initial pressure increment.
- Identical intermediate-velocity equation with convection and backflow control.
- Rotational recovery and consistent-mass CG/Jacobi velocity correction.
- Relative tolerance 1e-8 and absolute tolerance 1e-10 in all four solves.
- The two independent switches remove the complete residual contribution on
  both sides of the respective scalar equation. Both default to true.

The figure labels are No recovery stabilization, No increment stabilization,
and Full RPCS-VMS. Rotational
recovery is retained in all variants. Disabling recovery stabilization leaves
its rotational divergence term in place.

## Run

From the repository root, using a Python environment with DOLFINx 0.10.0,
PETSc 3.24.3, NumPy, and Matplotlib:

```bash
python test_cases/manufactured/run_pressure_stage_ablation.py
python test_cases/manufactured/run_pressure_stage_ablation.py --replot
python -m unittest test_cases.manufactured.test_pressure_stage_ablation -v
python -m unittest test_cases.manufactured.test_pressure_stage_reporting -v
```

MPI execution is also supported:

```bash
mpiexec -n 2 python test_cases/manufactured/run_pressure_stage_ablation.py
```

The group driver accepts `--only ablation`. Output is written to
`test_cases/manufactured/output/pressure_stage_ablation/`:

- `pressure_stage_ablation.csv`: final pressure and velocity errors, divergence,
  and convergence checks.
- `*_history.csv`: error and divergence histories at every step.
- `*_configuration.json`: complete normalized configuration for each run.
- `pressure_stage_fields.npz`: final pressure fields for all nine runs.
- `pressure_stage_references.csv`: unstabilized RPCS P1/P1 and monolithic
  P2/P1 reference errors at the smallest time step. These two additional
  calculations are run automatically.
- `pressure_stage_ablation_matrix.pdf` and `.png`: one row of the three variants
  at the smallest time step (0.001 by default), with a shared [-1, 1] color range.
- `pressure_stage_ablation.tex`: the five-method paper table, reporting global,
  boundary-strip, and interior pressure errors and global velocity error.
- `metadata.json`: software versions, MPI size, and source hashes.
- `presentation.json`: selected time step, table methods and error definitions,
  rendering source hash, and hashes of the unchanged calculation inputs.

Error histories use quadrature degree 8. All table errors use the same
quadrature-degree-5 evaluation as the near-boundary pressure comparison.
All pressure errors use p_h-p without mean removal, on the full domain,
boundary strip, and interior. The CSVs also retain the signed mean pressure
bias. The physical pressure fields are shown in the figures. Partition copies
of pressure nodes are merged by global degree-of-freedom numbers before
pressure contours are constructed.

Use one MPI rank to reproduce the manuscript table at its printed precision.
