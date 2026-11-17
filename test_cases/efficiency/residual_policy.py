"""Native PETSc stopping policies for isolated efficiency experiments."""

import numpy as np
from petsc4py import PETSc

from solver.boundary import BoundaryFactory
from solver.mesh_loader import MeshHandler
from solver.monolithic_vms import MonolithicSUPGPSPGSolver
from solver.rpcs_vms import FlowSolver

POLICIES = ("unpreconditioned", "preconditioned")


def apply_policy(ksp, policy):
    if policy not in POLICIES:
        raise ValueError(policy)
    norm = (PETSc.KSP.NormType.UNPRECONDITIONED if policy == POLICIES[0]
            else PETSc.KSP.NormType.PRECONDITIONED)
    if ksp.getType() not in {"cg", "gmres"}:
        raise ValueError("This comparison supports CG and GMRES only.")
    # Native GMRES couples these norms to different preconditioning sides.
    ksp.setPCSide(PETSc.PC.Side.RIGHT if ksp.getType() == "gmres"
                  and policy == POLICIES[0] else PETSc.PC.Side.LEFT)
    ksp.setNormType(norm)
    return ksp


class ResidualFlowSolver(FlowSolver):
    def __init__(self, *args, residual_policy, **kwargs):
        self.residual_policy = residual_policy
        super().__init__(*args, **kwargs)

    def _create_ksp(self, *args, **kwargs):
        return apply_policy(super()._create_ksp(*args, **kwargs), self.residual_policy)

    def _setup_projection(self):
        super()._setup_projection()
        apply_policy(self.proj_mass_ksp, self.residual_policy)

    def _rebuild_phi_operator(self, order):
        super()._rebuild_phi_operator(order)
        if not self.proj_phi_system.get("residual_policy_configured"):
            apply_policy(self.proj_phi_system["ksp"], self.residual_policy)
            self.proj_phi_system["residual_policy_configured"] = True


class ResidualMonolithicSolver(MonolithicSUPGPSPGSolver):
    def __init__(self, *args, residual_policy, **kwargs):
        self.residual_policy = residual_policy
        super().__init__(*args, **kwargs)

    def setup_solver(self):
        super().setup_solver()
        apply_policy(self.ksp, self.residual_policy)


def make_solver(comm, cfg, method, policy):
    mh = MeshHandler(comm, cfg.cfg["mesh"]["domain"], cfg.cfg["mesh"]["boundary"])
    if method not in {"RPCS-VMS", "monolithic-SUPG/PSPG"}:
        raise ValueError(method)
    cls = ResidualFlowSolver if method == "RPCS-VMS" else ResidualMonolithicSolver
    solver = cls(comm, mh, cfg, residual_policy=policy)
    boundary = BoundaryFactory(
        mh, solver.V, solver.Q, time_symbol=solver.time_symbol,
        mixed_velocity_space=getattr(solver, "mixed_velocity_space", None),
    ).create(cfg.cfg["boundary_conditions"])
    solver.set_bcs(boundary)
    return solver


def solver_ksps(solver, order):
    if isinstance(solver, FlowSolver):
        return {
            "predictor": solver.systems[order]["predictor"]["ksp"],
            "phi": solver.proj_phi_system["ksp"],
            "corrector": solver.proj_mass_ksp,
            "pressure": solver.systems[order]["pressure"]["ksp"],
        }
    return {"monolithic": solver.ksp}


def audit_ksp(ksp):
    matrix, _ = ksp.getOperators()
    rhs, solution = ksp.getRhs(), ksp.getSolution()
    residual, pc_residual, pc_rhs = rhs.duplicate(), rhs.duplicate(), rhs.duplicate()
    matrix.mult(solution, residual)
    residual.axpy(-1, rhs)
    pc = ksp.getPC()
    pc.apply(residual, pc_residual)
    pc.apply(rhs, pc_rhs)
    true_norm, rhs_norm = residual.norm(), rhs.norm()
    pc_norm, pc_rhs_norm = pc_residual.norm(), pc_rhs.norm()
    rtol, atol, _, _ = ksp.getTolerances()
    result = {
        "ksp": ksp.getType(), "pc": pc.getType(),
        "side": "right" if ksp.getPCSide() == PETSc.PC.Side.RIGHT else "left",
        "norm": "preconditioned" if ksp.getNormType() == PETSc.KSP.NormType.PRECONDITIONED
        else "unpreconditioned",
        "rtol": rtol, "atol": atol, "reason": int(ksp.getConvergedReason()),
        "iterations": int(ksp.getIterationNumber()),
        "reported_residual_norm": float(ksp.getResidualNorm()),
        "true_residual_norm": true_norm, "rhs_norm": rhs_norm,
        "true_relative_residual": true_norm/max(rhs_norm, 1e-300),
        "preconditioned_residual_norm": pc_norm, "preconditioned_rhs_norm": pc_rhs_norm,
        "true_residual_over_true_threshold": true_norm/max(atol, rtol*rhs_norm),
        "preconditioned_residual_over_pc_threshold": pc_norm/max(atol, rtol*pc_rhs_norm),
    }
    for vector in (residual, pc_residual, pc_rhs):
        vector.destroy()
    if result["reason"] <= 0 or not all(
            np.isfinite(v) for v in result.values() if isinstance(v, (float, int))):
        raise RuntimeError("Unconverged or nonfinite residual audit.")
    return result
