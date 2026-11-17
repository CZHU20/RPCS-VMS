import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Circle

ROOT_DIR = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solver.reporting import configure_matplotlib, save_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Plot DFG2D-3 mesh and streamlines.")
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=CASE_DIR / "output" / "solution_v.h5",
    )
    parser.add_argument(
        "--msh-path",
        type=Path,
        default=CASE_DIR / "mesh" / "mesh_domain.xdmf",
    )
    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 6.0, 8.0],
    )
    return parser.parse_args()


def parse_time_key(key):
    return float(key.replace("_", "."))


def match_time_key(keys, target):
    return min(keys, key=lambda key: abs(parse_time_key(key) - target))


def quad_to_tri(topology):
    if topology.shape[1] == 3:
        return topology
    triangles = np.empty((topology.shape[0] * 2, 3), dtype=np.int64)
    triangles[0::2, 0] = topology[:, 0]
    triangles[0::2, 1] = topology[:, 1]
    triangles[0::2, 2] = topology[:, 2]
    triangles[1::2, 0] = topology[:, 0]
    triangles[1::2, 1] = topology[:, 2]
    triangles[1::2, 2] = topology[:, 3]
    return triangles


def draw_macro_mesh(ax, msh_path, fallback_triangulation):
    try:
        mesh = meshio.read(msh_path)
        points = mesh.points[:, :2]
        segments = []
        for block in mesh.cells:
            if block.type not in {"quad", "triangle"}:
                continue
            for cell in block.data:
                pts = points[cell]
                for i in range(len(cell)):
                    segments.append([pts[i], pts[(i + 1) % len(cell)]])
        ax.add_collection(
            LineCollection(segments, colors="#111827", linewidths=0.12, alpha=0.5)
        )
    except Exception:
        ax.triplot(fallback_triangulation, color="#111827", lw=0.15, alpha=0.5)


def add_time_label(ax, time_value):
    text = ax.text(
        0.035,
        0.91,
        rf"$t={time_value:g}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "boxstyle": "round,pad=0.18",
            "alpha": 1.0,
            "linewidth": 0.75,
        },
        zorder=7,
    )
    text.get_bbox_patch().set_linestyle("-")


def main():
    args = parse_args()
    configure_matplotlib()
    if not args.h5_path.exists():
        raise FileNotFoundError(f"Velocity file not found: {args.h5_path}")

    with h5py.File(args.h5_path, "r") as handle:
        geometry = handle["Mesh/Grid/geometry"][:, :2]
        topology = handle["Mesh/Grid/topology"][:]
        triangulation = mtri.Triangulation(
            geometry[:, 0], geometry[:, 1], quad_to_tri(topology)
        )
        keys = list(handle["Function/f"].keys())
        matched = [(target, match_time_key(keys, target)) for target in args.times]
        snapshots = [handle[f"Function/f/{key}"][:, :2] for _, key in matched]

    x_grid, y_grid = np.meshgrid(
        np.linspace(0.0, 2.2, 450),
        np.linspace(0.0, 0.41, 115),
    )
    cylinder_center = (0.2, 0.2)
    cylinder_radius = 0.05
    cylinder_mask = (
        (x_grid - cylinder_center[0]) ** 2 + (y_grid - cylinder_center[1]) ** 2
    ) < cylinder_radius**2

    n_snapshots = len(args.times)
    n_rows = 1 + n_snapshots
    fig = plt.figure(figsize=(7.2, 8.7))
    grid = fig.add_gridspec(
        n_rows,
        2,
        width_ratios=[1.0, 0.055],
        height_ratios=[1.0] * n_rows,
        wspace=0.08,
        hspace=0.11,
    )
    axes = [fig.add_subplot(grid[row, 0]) for row in range(n_rows)]
    colorbar_grid = GridSpecFromSubplotSpec(
        n_snapshots, 1, subplot_spec=grid[1:, 1], height_ratios=[1.0] * n_snapshots
    )
    colorbar_lo = max(0, (n_snapshots - 2) // 2)
    colorbar_hi = min(n_snapshots, colorbar_lo + 2)
    colorbar_axis = fig.add_subplot(colorbar_grid[colorbar_lo:colorbar_hi, 0])
    fig.add_subplot(grid[0, 1]).axis("off")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 2.2)
        ax.set_ylim(0.0, 0.41)
        ax.set_yticks([0.0, 0.2, 0.4])
        ax.tick_params(direction="in")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    draw_macro_mesh(axes[0], args.msh_path, triangulation)
    axes[0].add_patch(Circle(cylinder_center, cylinder_radius, color="#1f3b73", zorder=5))

    u_grids = []
    v_grids = []
    speed_grids = []
    for velocity in snapshots:
        interp_u = mtri.LinearTriInterpolator(triangulation, velocity[:, 0])
        interp_v = mtri.LinearTriInterpolator(triangulation, velocity[:, 1])
        u_grid = interp_u(x_grid, y_grid)
        v_grid = interp_v(x_grid, y_grid)
        if np.ma.isMaskedArray(u_grid):
            u_grid = u_grid.filled(np.nan)
        if np.ma.isMaskedArray(v_grid):
            v_grid = v_grid.filled(np.nan)
        u_grid[cylinder_mask] = np.nan
        v_grid[cylinder_mask] = np.nan
        speed = np.sqrt(u_grid**2 + v_grid**2)
        u_grids.append(u_grid)
        v_grids.append(v_grid)
        speed_grids.append(speed)

    norm = plt.Normalize(
        vmin=0.0,
        vmax=float(np.nanmax(np.concatenate([s.ravel() for s in speed_grids]))),
    )

    for ax, (target, key), u_grid, v_grid, speed in zip(
        axes[1:], matched, u_grids, v_grids, speed_grids
    ):
        stream = ax.streamplot(
            x_grid,
            y_grid,
            u_grid,
            v_grid,
            color=speed,
            cmap="viridis",
            norm=norm,
            linewidth=0.7,
            density=[1.48, 0.88],
            arrowstyle="-",
            minlength=0.06,
            integration_direction="both",
        )
        ax.add_patch(
            Circle(
                cylinder_center,
                cylinder_radius,
                facecolor="#1f3b73",
                edgecolor="black",
                linewidth=0.4,
                zorder=5,
            )
        )
        add_time_label(ax, parse_time_key(key))
    axes[-1].set_xlabel(r"$x$")
    for ax in axes:
        ax.set_ylabel(r"$y$")
    colorbar = fig.colorbar(stream.lines, cax=colorbar_axis, label=r"$|\mathbf{u}|$")
    colorbar.set_ticks([0.0, 1.0, 2.0])
    save_figure(
        fig,
        CASE_DIR / "output" / "cylinder_2d_streamlines",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
