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

ROOT_DIR = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solver.reporting import configure_matplotlib, save_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Plot bifurcated tube mesh and streamlines.")
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
    parser.add_argument("--time", type=float, default=50.0)
    return parser.parse_args()


def parse_time_key(key):
    return float(key.replace("_", "."))


def match_time_key(keys, target):
    return min(keys, key=lambda key: abs(parse_time_key(key) - target))


def draw_boundaries(ax):
    lines = [
        [(0, 0.5), (8, 0.5)],
        [(0, -0.5), (8, -0.5)],
        [(0, 0), (0.5, 0), (0.5, -0.5)],
        [(8, -0.1), (1.5, -0.1), (1.5, 0.2), (8, 0.2)],
        [(8, -0.5), (8, -0.1)],
        [(8, 0.2), (8, 0.5)],
    ]
    for line in lines:
        ax.plot(
            [point[0] for point in line],
            [point[1] for point in line],
            color="black",
            linewidth=1.2,
            zorder=10,
        )


def draw_mesh(ax, msh_path, fallback_triangulation):
    try:
        mesh = meshio.read(msh_path)
        points = mesh.points[:, :2]
        segments = []
        for block in mesh.cells:
            if block.type != "triangle":
                continue
            for tri in block.data:
                tri_points = points[tri]
                for i in range(3):
                    segments.append([tri_points[i], tri_points[(i + 1) % 3]])
        ax.add_collection(
            LineCollection(segments, colors="#111827", linewidths=0.12, alpha=0.42)
        )
    except Exception:
        ax.triplot(fallback_triangulation, color="#111827", lw=0.15, alpha=0.42)


def is_solid(points):
    x = points[:, 0]
    y = points[:, 1]
    return ((x < 0.5) & (y < 0)) | ((x > 1.5) & (y > -0.1) & (y < 0.2))


def streamline_seeds():
    seed_blocks = []
    seed_blocks.append(np.column_stack([np.full(13, 0.04), np.linspace(-0.44, 0.44, 13)]))
    seed_blocks.append(np.column_stack([np.linspace(0.08, 3.95, 18), np.full(18, 0.43)]))
    seed_blocks.append(np.column_stack([np.linspace(0.78, 1.55, 18), np.full(18, 0.485)]))
    seed_blocks.append(np.column_stack([np.linspace(1.48, 2.30, 18), np.full(18, 0.455)]))
    seed_blocks.append(np.column_stack([np.linspace(1.52, 2.42, 16), np.full(16, 0.275)]))
    seed_blocks.append(np.column_stack([np.full(14, 1.515), np.linspace(0.215, 0.49, 14)]))
    seed_blocks.append(np.column_stack([np.full(12, 1.565), np.linspace(0.23, 0.48, 12)]))
    seed_blocks.append(np.column_stack([np.linspace(0.55, 1.45, 12), np.full(12, -0.43)]))
    seed_blocks.append(
        np.array(
            [
                (0.58, -0.42),
                (0.72, -0.36),
                (0.90, -0.30),
                (1.10, -0.25),
                (1.30, -0.18),
                (1.45, -0.12),
                (1.50, 0.24),
                (1.54, 0.22),
                (1.58, 0.24),
                (1.62, 0.23),
                (1.62, 0.28),
                (1.68, 0.31),
                (1.80, 0.34),
                (1.95, 0.42),
                (2.15, 0.38),
                (1.52, -0.14),
                (1.70, -0.22),
                (2.05, -0.32),
                (2.60, -0.40),
            ]
        )
    )
    seeds = np.vstack(seed_blocks)
    in_box = (
        (seeds[:, 0] >= 0.0)
        & (seeds[:, 0] <= 4.0)
        & (seeds[:, 1] >= -0.5)
        & (seeds[:, 1] <= 0.5)
    )
    return seeds[in_box & ~is_solid(seeds)]


def main():
    args = parse_args()
    configure_matplotlib()
    out_dir = CASE_DIR / "output"
    if not args.h5_path.exists():
        raise FileNotFoundError(f"Velocity file not found: {args.h5_path}")

    with h5py.File(args.h5_path, "r") as handle:
        geometry = handle["Mesh/Grid/geometry"][:, :2]
        topology = handle["Mesh/Grid/topology"][:]
        triangulation = mtri.Triangulation(geometry[:, 0], geometry[:, 1], topology)
        key = match_time_key(list(handle["Function/f"].keys()), args.time)
        velocity = handle[f"Function/f/{key}"][:, :2]
        matched_time = parse_time_key(key)

    x_grid, y_grid = np.meshgrid(
        np.linspace(0, 4.0, 620),
        np.linspace(-0.5, 0.5, 240),
    )
    solid_mask = ((x_grid < 0.5) & (y_grid < 0)) | (
        (x_grid > 1.5) & (y_grid > -0.1) & (y_grid < 0.2)
    )
    interp_u = mtri.LinearTriInterpolator(triangulation, velocity[:, 0])
    interp_v = mtri.LinearTriInterpolator(triangulation, velocity[:, 1])
    u_grid = interp_u(x_grid, y_grid)
    v_grid = interp_v(x_grid, y_grid)
    if np.ma.isMaskedArray(u_grid):
        u_grid = u_grid.filled(np.nan)
    if np.ma.isMaskedArray(v_grid):
        v_grid = v_grid.filled(np.nan)
    u_grid[solid_mask] = np.nan
    v_grid[solid_mask] = np.nan
    speed = np.sqrt(u_grid**2 + v_grid**2)

    fig = plt.figure(figsize=(7.2, 4.4))
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 0.05], height_ratios=[1.0, 1.0], wspace=0.06, hspace=0.13
    )
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])]
    colorbar_grid = GridSpecFromSubplotSpec(
        5, 1, subplot_spec=grid[1, 1], height_ratios=[0.45, 1.0, 1.0, 1.0, 0.45]
    )
    colorbar_axis = fig.add_subplot(colorbar_grid[1:4, 0])
    fig.add_subplot(grid[0, 1]).axis("off")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([-0.5, 0.0, 0.5])
        ax.tick_params(direction="in")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_xlim(0.0, 8.0)
    axes[1].set_xlim(0.0, 4.0)

    draw_mesh(axes[0], args.msh_path, triangulation)
    draw_boundaries(axes[0])
    axes[0].set_ylabel(r"$y$")

    stream = axes[1].streamplot(
        x_grid,
        y_grid,
        u_grid,
        v_grid,
        color=speed,
        cmap="viridis",
        linewidth=0.62,
        density=[2.25, 1.28],
        arrowstyle="-",
        minlength=0.035,
        integration_direction="both",
    )
    axes[1].streamplot(
        x_grid,
        y_grid,
        u_grid,
        v_grid,
        color=speed,
        cmap="viridis",
        linewidth=0.82,
        start_points=streamline_seeds(),
        arrowstyle="-",
        minlength=0.03,
        integration_direction="both",
    )
    draw_boundaries(axes[1])
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    fig.colorbar(stream.lines, cax=colorbar_axis, label=r"$|\mathbf{u}|$")
    save_figure(fig, out_dir / "bifurcated_tube_streamlines")
    plt.close(fig)


if __name__ == "__main__":
    main()
