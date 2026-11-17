# Solver map

`rpcs_vms.py` contains the RPCS-VMS method. The main implementation points are:

- `FlowSolver._build_scheme_forms`: predictor, pressure-increment, velocity-correction, and pressure-recovery forms for BDF1 and BDF2.
- `FlowSolver._solve_phi`: stabilized pressure-increment solve.
- `FlowSolver._solve_corrector`: consistent-mass velocity correction.
- `FlowSolver.solve_step`: predictor, increment, correction, and recovery in order.

The supporting files are:

- `stabilization.py`: elementwise momentum and divergence scales.
- `boundary.py`: velocity, pseudo-traction, and backflow boundary data.
- `manufactured.py`: common exact velocity, pressure, forcing, and traction.
- `monolithic_vms.py`: monolithic Oseen solver for equal-order SUPG/PSPG comparisons and unstabilized P2/P1 references.
- `configuration.py`: validation and normalization of the Python parameter dictionaries.
- `mesh_loader.py`, `expression.py`, `benchmark.py`, and `utils.py`: mesh, expression, monitoring, and output utilities.

For controlled pressure-stage isolation, the solver options
`pressure_increment_stabilization` and `pressure_recovery_stabilization`
independently switch the complete residual contribution in each scalar equation.
Both default to `True`. Rotational recovery is retained when recovery
stabilization is disabled.
