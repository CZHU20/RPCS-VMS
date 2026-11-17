import os
import time
import datetime
import csv

from dolfinx import fem, io


class ResultsWriter:
    def __init__(self, comm, filename, v, p):
        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.xdmf_v = io.XDMFFile(comm, filename + "_v.xdmf", "w")
        self.xdmf_v.write_mesh(v.function_space.mesh)
        self.xdmf_p = io.XDMFFile(comm, filename + "_p.xdmf", "w")
        self.xdmf_p.write_mesh(v.function_space.mesh)
        self.v = v
        self.p = p

        mesh = v.function_space.mesh
        gdim = mesh.geometry.dim

        # Always export a P1 velocity field so the output format stays the same
        # for both P1 and P2 velocity spaces.
        V_vis = fem.functionspace(mesh, ("Lagrange", 1, (gdim, )))
        self.v_vis = fem.Function(V_vis)

    def save(self, t):
        self.v_vis.interpolate(self.v)
        self.xdmf_v.write_function(self.v_vis, t)
        self.xdmf_p.write_function(self.p, t)

    def close(self):
        self.xdmf_v.close()
        self.xdmf_p.close()


class ForceHistoryWriter:
    def __init__(self, comm, filename):
        self.comm = comm
        self.handle = None
        self.writer = None

        if self.comm.rank != 0:
            return

        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self.handle = open(filename, "w", newline="", encoding="utf-8")
        fieldnames = [
            "step",
            "time",
            "drag_force",
            "lift_force",
            "drag_coefficient",
            "lift_coefficient",
            "pressure_difference",
        ]
        self.writer = csv.DictWriter(self.handle, fieldnames=fieldnames)
        self.writer.writeheader()
        self.handle.flush()

    def write(self, sample):
        if self.comm.rank != 0 or sample is None:
            return
        row = {
            "step": sample.get("step", ""),
            "time": sample["time"],
            "drag_force": sample["drag_force"],
            "lift_force": sample["lift_force"],
            "drag_coefficient": sample["drag_coefficient"],
            "lift_coefficient": sample["lift_coefficient"],
            "pressure_difference": sample["pressure_difference"],
        }
        self.writer.writerow(row)
        self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()


class Timer:
    """Track wall time and estimate the remaining runtime."""

    def __init__(self):
        self.start_time = None
        self.last_check_time = None
        self.total_steps = 0
        self.current_step = 0

    def start(self, total_steps_expected=None):
        """Start the timer."""
        self.start_time = time.time()
        self.last_check_time = self.start_time
        self.total_steps = total_steps_expected

    def timestamp(self, current_step):
        """Return elapsed wall time, average step time, and ETA."""
        now = time.time()

        total_elapsed = now - self.start_time

        interval_elapsed = now - self.last_check_time
        steps_done_in_interval = current_step - self.current_step

        if steps_done_in_interval > 0:
            avg_step_time = interval_elapsed / steps_done_in_interval
        else:
            avg_step_time = 0.0

        eta_str = "N/A"
        if self.total_steps and avg_step_time > 0:
            remaining_steps = self.total_steps - current_step
            remaining_seconds = remaining_steps * avg_step_time
            eta_str = str(datetime.timedelta(seconds=int(remaining_seconds)))

        self.last_check_time = now
        self.current_step = current_step

        total_str = str(datetime.timedelta(seconds=int(total_elapsed)))

        return total_str, avg_step_time, eta_str
