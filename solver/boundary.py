from dataclasses import dataclass, field

from dolfinx.fem import dirichletbc, locate_dofs_topological
from mpi4py import MPI

from solver.expression import ExpressionHandler


@dataclass
class BoundaryData:
    velocity_bcs: list = field(default_factory=list)
    phi_bcs: list = field(default_factory=list)
    traction_data: list = field(default_factory=list)
    velocity_boundary_ids: list = field(default_factory=list)
    updaters: list = field(default_factory=list)


class BoundaryFactory:
    def __init__(
        self,
        mesh_handler,
        V,
        Q,
        time_symbol=None,
        mixed_velocity_space=None,
    ):
        self.mh = mesh_handler
        self.V = V
        self.Q = Q
        self.time_symbol = time_symbol
        self.mixed_velocity_space = mixed_velocity_space
        self.expr_handler = ExpressionHandler()

    def _facets_for_marker(self, marker_id):
        facets = self.mh.ft.find(marker_id)
        global_count = self.mh.comm.allreduce(len(facets), op=MPI.SUM)
        if global_count == 0:
            raise ValueError(
                f"Boundary marker id {marker_id} was not found in the mesh tags."
            )
        return facets

    def _validate_vector_value(self, marker_id, value):
        if not isinstance(value, (list, tuple)) or len(value) != self.mh.gdim:
            raise ValueError(
                f"Boundary marker id {marker_id} expects {self.mh.gdim} value components."
            )

    def create(self, bc_configs):
        data = BoundaryData()
        fdim = self.mh.mesh.topology.dim - 1

        for conf in bc_configs:
            marker_id = conf["id"]
            btype = conf["type"]
            form = conf["form"]
            raw_value = conf["value"]
            self._validate_vector_value(marker_id, raw_value)

            if btype == "pressure":
                raise ValueError("Pressure boundary conditions have been removed from the RPCS solver.")

            if btype == "velocity":
                if form == "expression":
                    func, updater = self.expr_handler.create_exact_function(
                        self.V,
                        raw_value,
                        form,
                        time_symbol=self.time_symbol,
                    )
                else:
                    func, updater = self.expr_handler.create_function(self.V, raw_value, form)
                if updater is not None:
                    data.updaters.append(updater)

                facets = self._facets_for_marker(marker_id)
                if self.mixed_velocity_space is None:
                    dofs = locate_dofs_topological(self.V, fdim, facets)
                    data.velocity_bcs.append(dirichletbc(func, dofs))
                else:
                    dofs = locate_dofs_topological(
                        (self.mixed_velocity_space, self.V), fdim, facets
                    )
                    data.velocity_bcs.append(
                        dirichletbc(func, dofs, self.mixed_velocity_space)
                    )
                data.velocity_boundary_ids.append(marker_id)
                continue

            if btype == "traction":
                backflow_stabilization = bool(
                    conf.get("backflow_stabilization", True)
                )
                if form == "expression":
                    traction = self.expr_handler.create_exact_expression(
                        self.mh.mesh,
                        raw_value,
                        form,
                        time_symbol=self.time_symbol,
                    )
                    data.traction_data.append(
                        (marker_id, traction, backflow_stabilization)
                    )
                else:
                    func, updater = self.expr_handler.create_function(self.V, raw_value, form)
                    if updater is not None:
                        data.updaters.append(updater)
                    data.traction_data.append((marker_id, func, backflow_stabilization))

                phi_zero, _ = self.expr_handler.create_function(self.Q, 0.0, "constant")
                facets = self._facets_for_marker(marker_id)
                dofs = locate_dofs_topological(self.Q, fdim, facets)
                data.phi_bcs.append(dirichletbc(phi_zero, dofs))
                continue

            raise ValueError(f"Unknown boundary condition type: {btype}")

        return data
