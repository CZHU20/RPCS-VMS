from dolfinx import io
from ufl import Measure, FacetNormal, CellDiameter, avg


class MeshHandler:
    def __init__(self, comm, domain_file, boundary_file):
        self.comm = comm

        # Keep the quadrature degree explicit so all forms use the same integration rule.
        jit_metadata = {"quadrature_degree": 5}

        with io.XDMFFile(comm, domain_file, "r") as xdmf:
            self.mesh = xdmf.read_mesh(name="Grid")
            self.mesh.topology.create_connectivity(self.mesh.topology.dim - 1, self.mesh.topology.dim)

        with io.XDMFFile(comm, boundary_file, "r") as xdmf:
            self.ft = xdmf.read_meshtags(self.mesh, name="Grid")

        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.ft, metadata=jit_metadata)
        self.dS = Measure("dS", domain=self.mesh, metadata=jit_metadata)
        self.dx = Measure("dx", domain=self.mesh, metadata=jit_metadata)
        self.n = FacetNormal(self.mesh)
        self.h = CellDiameter(self.mesh)
        self.h_avg = avg(self.h)
        self.gdim = self.mesh.geometry.dim