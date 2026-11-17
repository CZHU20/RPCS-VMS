import csv
from pathlib import Path

import numpy as np
from dolfinx import geometry


class PointEvaluator:
    def __init__(self, mesh_handler, points, comm, fallback=None):
        self.mesh_handler = mesh_handler
        self.comm = comm
        self.points = self._normalize_points(points)
        self.fallback = fallback

        tree = geometry.bb_tree(mesh_handler.mesh, mesh_handler.mesh.topology.dim)
        candidates = geometry.compute_collisions_points(tree, self.points)
        colliding = geometry.compute_colliding_cells(
            mesh_handler.mesh, candidates, self.points
        )
        self.local_cells = [colliding.links(point_id) for point_id in range(len(self.points))]

    @staticmethod
    def _normalize_points(points):
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("Point coordinates must be a 2D array.")
        if array.shape[1] == 2:
            zeros = np.zeros((array.shape[0], 1), dtype=np.float64)
            array = np.hstack((array, zeros))
        elif array.shape[1] != 3:
            raise ValueError("Point coordinates must have 2 or 3 columns.")
        return array

    def evaluate(self, function):
        local_hits = {}
        for point_id, point in enumerate(self.points):
            cells = self.local_cells[point_id]
            if len(cells) == 0:
                continue
            value = function.eval(
                point.reshape(1, 3), np.array([cells[0]], dtype=np.int32)
            )
            local_hits[point_id] = np.asarray(value, dtype=np.float64).reshape(-1)

        gathered_hits = self.comm.gather(local_hits, root=0)
        if self.comm.rank != 0:
            return None

        values = []
        for point_id, point in enumerate(self.points):
            value = None
            for hit_map in gathered_hits:
                if point_id in hit_map:
                    value = hit_map[point_id]
                    break
            if value is None:
                if self.fallback is None:
                    raise RuntimeError(
                        f"Unable to evaluate function at point {point.tolist()}"
                    )
                value = np.asarray(self.fallback(point), dtype=np.float64).reshape(-1)
            values.append(value)

        return np.asarray(values, dtype=np.float64)


def compute_error_metrics(simulated, reference):
    error = simulated - reference
    abs_error = np.abs(error)
    ref_norm = np.linalg.norm(reference)
    return {
        "linf": float(np.max(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "relative_l2": float(np.linalg.norm(error) / ref_norm) if ref_norm > 0 else 0.0,
    }


def build_series_table(abscissa_name, abscissa_values, reference, simulated):
    table = []
    for abscissa, ref_value, sim_value in zip(abscissa_values, reference, simulated):
        table.append(
            {
                abscissa_name: float(abscissa),
                "reference": float(ref_value),
                "simulated": float(sim_value),
                "abs_error": float(abs(sim_value - ref_value)),
            }
        )
    return table


def write_rows_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty CSV table.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)