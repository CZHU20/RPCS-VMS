"""Read-only divergence and stress diagnostics for saved Q1 cylinder fields."""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc


CASE = Path(__file__).resolve().parent


def point_indices(space, domain, geometry):
    count = space.dofmap.index_map.size_local + space.dofmap.index_map.num_ghosts
    mapping = np.full(count, -1, dtype=np.int64)
    input_ids = domain.geometry.input_global_indices
    for cell in range(domain.topology.index_map(domain.topology.dim).size_local):
        mapping[space.dofmap.cell_dofs(cell)] = input_ids[domain.geometry.dofmap[cell]]
    if (mapping < 0).any():
        raise RuntimeError("Incomplete Q1 geometry-to-function mapping.")
    if not np.allclose(space.tabulate_dof_coordinates(), geometry[mapping], rtol=0, atol=1e-12):
        raise RuntimeError("The Q1 function and geometry node orders differ.")
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=CASE / "output")
    parser.add_argument("--times", type=float, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--output", type=Path,
                        default=CASE / "output/field_diagnostics.json")
    args = parser.parse_args()
    if MPI.COMM_WORLD.size != 1:
        raise ValueError("Run this read-only reconstruction on one rank.")
    comm = MPI.COMM_SELF
    with io.XDMFFile(comm, str(args.results / "solution_v.xdmf"), "r") as handle:
        domain = handle.read_mesh(name="Grid")
    domain.topology.create_connectivity(2, 3)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    v, p = fem.Function(V), fem.Function(Q)
    dx = ufl.Measure("dx", domain=domain, metadata={"quadrature_degree": 5})
    facets = mesh.locate_entities_boundary(domain, 2, lambda x:
        np.isclose((x[0]-0.5)**2+(x[1]-0.2)**2, 0.05**2, rtol=0, atol=1e-10))
    tags = mesh.meshtags(domain, 2, facets, np.ones(len(facets), dtype=np.int32))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=tags,
                     metadata={"quadrature_degree": 5})
    n = ufl.FacetNormal(domain)
    nu, force_scale = 0.001, 2 / 0.041
    pseudo = nu*ufl.grad(v)-p*ufl.Identity(3)
    physical = nu*(ufl.grad(v)+ufl.grad(v).T)-p*ufl.Identity(3)
    forms = {
        "velocity_norm_squared": fem.form(ufl.inner(v, v)*dx),
        "gradient_norm_squared": fem.form(ufl.inner(ufl.grad(v), ufl.grad(v))*dx),
        "divergence_norm_squared": fem.form(ufl.div(v)**2*dx),
        "pressure_gradient_norm_squared": fem.form(ufl.inner(ufl.grad(p), ufl.grad(p))*dx),
    }
    x = ufl.SpatialCoordinate(domain)
    for label, condition in (("upstream", ufl.lt(x[0], 0.4)),
                             ("near_cylinder", ufl.And(ufl.ge(x[0], 0.4), ufl.lt(x[0], 0.65))),
                             ("wake", ufl.ge(x[0], 0.65))):
        forms[label+"_divergence_norm_squared"] = fem.form(
            ufl.conditional(condition, ufl.div(v)**2, 0)*dx)
    for name, stress in (("pseudo", pseudo), ("physical", physical)):
        for i, axis in enumerate(("drag", "lift")):
            forms[f"{name}_{axis}"] = fem.form(-force_scale*ufl.dot(stress, n)[i]*ds(1))
    forms["normal_divergence_lift"] = fem.form(-force_scale*nu*ufl.div(v)*n[1]*ds(1))
    volume = fem.assemble_scalar(fem.form(fem.Constant(domain, PETSc.ScalarType(1))*dx))
    v.interpolate(lambda x: np.vstack((x[1], -x[0], np.zeros_like(x[0]))))
    v.x.scatter_forward()
    analytic_div = float(fem.assemble_scalar(forms["divergence_norm_squared"]))
    analytic_grad = float(fem.assemble_scalar(forms["gradient_norm_squared"]))
    if analytic_div > 1e-18 or abs(analytic_grad-2*volume) > 1e-10:
        raise RuntimeError("Analytic rotation check failed on the output mesh.")
    with (args.results / "force_coefficients.csv").open(newline="") as stream:
        recorded = list(csv.DictReader(stream))
    records = []
    with h5py.File(args.results / "solution_v.h5") as vh, h5py.File(args.results / "solution_p.h5") as ph:
        vg, pg = vh["Mesh/Grid/geometry"][:], ph["Mesh/Grid/geometry"][:]
        assert np.array_equal(vg, pg)
        vi, pi = point_indices(V, domain, vg), point_indices(Q, domain, pg)
        vk, pk = list(vh["Function/f"]), list(ph["Function/f"])
        parse = lambda key: float(key.replace("_", "."))
        for target in args.times:
            vkey, pkey = min(vk, key=lambda k: abs(parse(k)-target)), min(pk, key=lambda k: abs(parse(k)-target))
            if abs(parse(vkey)-target) > 1e-9 or abs(parse(pkey)-target) > 1e-9:
                raise ValueError(f"Missing exact diagnostic snapshot at {target}.")
            v.x.array[:] = vh[f"Function/f/{vkey}"][:][vi].ravel()
            p.x.array[:] = ph[f"Function/f/{pkey}"][:][pi].ravel()
            v.x.scatter_forward()
            p.x.scatter_forward()
            data = {name: float(fem.assemble_scalar(form)) for name, form in forms.items()}
            row = min(recorded, key=lambda r: abs(float(r["time"])-target))
            for quantity, column in (("drag", "drag_coefficient"), ("lift", "lift_coefficient")):
                expected = float(row[column])
                if abs(data[f"pseudo_{quantity}"]-expected) > 1e-7*max(1, abs(expected)):
                    raise RuntimeError("Reconstructed pseudo-stress force disagrees with the MPI run.")
            data.update(time=target, divergence_to_gradient_ratio=np.sqrt(
                data["divergence_norm_squared"] / max(data["gradient_norm_squared"], 1e-300)),
                physical_minus_pseudo_lift=data["physical_lift"]-data["pseudo_lift"])
            nodal = v.x.array.reshape(-1, 3)
            speeds = np.linalg.norm(nodal, axis=1)
            fastest = int(np.argmax(speeds))
            coordinates = V.tabulate_dof_coordinates()
            data["maximum_nodal_speed"] = float(speeds[fastest])
            data["maximum_speed_coordinate"] = coordinates[fastest].tolist()
            no_slip = (np.isclose(coordinates[:, 1], 0) |
                       np.isclose(coordinates[:, 1], 0.41) |
                       np.isclose(coordinates[:, 2], 0) |
                       np.isclose(coordinates[:, 2], 0.41) |
                       np.isclose((coordinates[:, 0]-0.5)**2+
                                  (coordinates[:, 1]-0.2)**2, 0.05**2,
                                  rtol=0, atol=1e-10))
            data["no_slip_nodal_maximum"] = float(np.max(speeds[no_slip], initial=0))
            if abs(data["physical_minus_pseudo_lift"]-data["normal_divergence_lift"]) > 1e-8:
                raise RuntimeError("No-slip stress/divergence identity failed on the cylinder.")
            records.append(data)
            print(json.dumps(data), flush=True)
    result = {"purpose": "Read-only diagnostics; production force definition remains pseudo-stress",
              "volume": float(volume), "cylinder_facets": len(facets),
              "analytic_rotation_divergence_squared": analytic_div,
              "analytic_rotation_gradient_squared": analytic_grad,
              "snapshots": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
