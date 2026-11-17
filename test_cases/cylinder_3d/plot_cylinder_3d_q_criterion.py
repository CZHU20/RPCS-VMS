"""Render Q-criterion snapshots for the three-dimensional cylinder case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solver.reporting import configure_matplotlib


ROOT = Path(__file__).resolve().parent
DEFAULT_H5_PATH = ROOT / "output" / "solution_v.h5"
DEFAULT_OUTPUT_STEM = ROOT / "output" / "cylinder_3d_q_criterion"
CYLINDER_CENTER = (0.5, 0.2, 0.205)
CYLINDER_RADIUS = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 3D Q-criterion snapshots for flow past a cylinder."
    )
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--times", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0])
    parser.add_argument(
        "--q-level",
        type=float,
        default=None,
        help="Explicit Q-criterion isosurface level. If omitted, a positive-Q percentile is used.",
    )
    parser.add_argument("--q-percentile", type=float, default=65.0)
    parser.add_argument(
        "--global-q-percentile",
        type=float,
        default=65.0,
        help=(
            "Use one fixed Q level for all snapshots, computed as this percentile "
            "of the pooled positive-Q values. It is "
            "ignored when --q-level is set."
        ),
    )
    parser.add_argument(
        "--per-snapshot-q-percentile",
        action="store_true",
        help="Use --q-percentile independently for each snapshot instead of a fixed global level.",
    )
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    return parser.parse_args()


def parse_time_key(key: str) -> float:
    return float(key.replace("_", "."))


def match_time_keys(keys: list[str], target_times: list[float]) -> list[tuple[float, str]]:
    matched: list[tuple[float, str]] = []
    for time_value in target_times:
        best_key = min(keys, key=lambda key: abs(parse_time_key(key) - time_value))
        matched.append((parse_time_key(best_key), best_key))
    return matched


def build_unstructured_grid(topology: np.ndarray, geometry: np.ndarray) -> pv.UnstructuredGrid:
    cells = np.empty((topology.shape[0], 9), dtype=topology.dtype)
    cells[:, 0] = 8
    cells[:, 1:] = topology
    celltypes = np.full(topology.shape[0], pv.CellType.HEXAHEDRON, dtype=np.uint8)
    return pv.UnstructuredGrid(cells.ravel(), celltypes, geometry)


def crop_white_margins(image: np.ndarray, tolerance: int = 248, padding: int = 18) -> np.ndarray:
    rgb = image[..., :3]
    mask = np.any(rgb < tolerance, axis=2)
    if not np.any(mask):
        return image
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    row_min = max(int(rows[0]) - padding, 0)
    row_max = min(int(rows[-1]) + padding + 1, image.shape[0])
    col_min = max(int(cols[0]) - padding, 0)
    col_max = min(int(cols[-1]) + padding + 1, image.shape[1])
    return image[row_min:row_max, col_min:col_max]


def camera_position():
    focus = (1.25, 0.205, 0.205)
    return [(1.12, -1.95, 0.74), focus, (0.0, 0.0, 1.0)]


def q_level_from_field(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.percentile(positive, percentile))


def global_q_level_from_snapshots(
    base_grid: pv.UnstructuredGrid,
    velocity_snapshots: list[np.ndarray],
    percentile: float,
) -> float:
    positive_values = []
    for velocity_data in velocity_snapshots:
        grid = base_grid.copy(deep=True)
        grid.point_data["Velocity"] = velocity_data
        grid.set_active_vectors("Velocity")
        q_grid = grid.compute_derivative(scalars="Velocity", qcriterion=True)
        q_values = np.asarray(q_grid.point_data["qcriterion"], dtype=float)
        q_values = q_values[np.isfinite(q_values)]
        q_values = q_values[q_values > 0.0]
        if q_values.size:
            positive_values.append(q_values)
    if not positive_values:
        return 0.0
    return float(np.percentile(np.concatenate(positive_values), percentile))


def extract_qcriterion_surface(
    base_grid: pv.UnstructuredGrid,
    velocity_data: np.ndarray,
    explicit_q_level: float | None,
    q_percentile: float,
) -> tuple[pv.PolyData, float]:
    grid = base_grid.copy(deep=True)
    grid.point_data["Velocity"] = velocity_data
    grid.point_data["Speed"] = np.linalg.norm(velocity_data, axis=1)
    grid.set_active_vectors("Velocity")
    grid = grid.compute_derivative(scalars="Velocity", qcriterion=True)

    q_values = grid.point_data["qcriterion"]
    q_level = explicit_q_level if explicit_q_level is not None else q_level_from_field(
        np.asarray(q_values), q_percentile
    )
    surface = grid.contour(isosurfaces=[q_level], scalars="qcriterion")
    if surface.n_points > 0:
        surface = surface.clean()
        surface = filter_q_surface_components(surface, base_grid.bounds)
        if surface.n_points > 200:
            surface = surface.smooth(
                n_iter=16,
                relaxation_factor=0.055,
                boundary_smoothing=False,
                feature_smoothing=False,
            )
    return surface, q_level


def filter_q_surface_components(
    surface: pv.PolyData,
    domain_bounds: tuple[float, float, float, float, float, float],
) -> pv.PolyData:
    if surface.n_cells == 0:
        return surface

    connected = surface.connectivity("all")
    region_ids = np.asarray(connected.cell_data.get("RegionId", []), dtype=int)
    if region_ids.size == 0:
        return surface

    x_min, x_max = float(domain_bounds[0]), float(domain_bounds[1])
    inlet_limit = x_min + 0.08 * (x_max - x_min)
    outlet_limit = x_max - 0.08 * (x_max - x_min)
    min_cells = max(220, int(0.003 * connected.n_cells))
    kept_cell_ids: list[np.ndarray] = []

    for region_id in np.unique(region_ids):
        cell_ids = np.flatnonzero(region_ids == region_id)
        component = connected.extract_cells(cell_ids)
        bounds = component.bounds
        touches_inlet = bounds[0] <= inlet_limit
        touches_outlet = bounds[1] >= outlet_limit
        if len(cell_ids) < min_cells:
            continue
        if (touches_inlet or touches_outlet) and len(cell_ids) < 5000:
            continue
        kept_cell_ids.append(cell_ids)

    if not kept_cell_ids:
        return surface

    filtered = connected.extract_cells(np.concatenate(kept_cell_ids))
    return filtered.extract_surface().clean()


def speed_limits_from_surfaces(
    surfaces: list[pv.PolyData], velocity_snapshots: list[np.ndarray]
) -> tuple[float, float]:
    surface_speeds = []
    for surface in surfaces:
        if surface.n_points > 0 and "Speed" in surface.point_data:
            surface_speeds.append(np.asarray(surface.point_data["Speed"], dtype=float))
    if surface_speeds:
        values = np.concatenate(surface_speeds)
        values = values[np.isfinite(values)]
        if values.size:
            lower = max(0.0, float(np.nanpercentile(values, 1.0)))
            upper = float(np.nanpercentile(values, 99.0))
            if upper > lower:
                return lower, upper

    all_speeds = np.concatenate(
        [np.linalg.norm(snapshot, axis=1) for snapshot in velocity_snapshots]
    )
    return (0.0, float(np.nanpercentile(all_speeds, 99.5)))


def render_qcriterion_panel(
    base_grid: pv.UnstructuredGrid,
    surface: pv.PolyData,
    speed_limits: tuple[float, float],
) -> np.ndarray:
    plotter = pv.Plotter(off_screen=True, window_size=(2200, 980))
    plotter.set_background("white")
    plotter.enable_anti_aliasing("ssaa")
    plotter.enable_parallel_projection()
    try:
        plotter.enable_eye_dome_lighting()
    except Exception:
        pass

    if surface.n_points > 0:
        plotter.add_mesh(
            surface,
            scalars="Speed",
            cmap="viridis",
            clim=speed_limits,
            show_scalar_bar=False,
            smooth_shading=True,
            specular=0.24,
            specular_power=12,
            ambient=0.36,
            diffuse=0.82,
        )

    channel_outline = pv.Box(bounds=base_grid.bounds)
    plotter.add_mesh(
        channel_outline,
        color="#475569",
        style="wireframe",
        line_width=1.4,
        opacity=0.62,
    )

    cylinder = pv.Cylinder(
        center=CYLINDER_CENTER,
        direction=(0.0, 0.0, 1.0),
        radius=CYLINDER_RADIUS,
        height=0.41,
        resolution=128,
    )
    plotter.add_mesh(
        cylinder,
        color="#b6bcc7",
        opacity=0.62,
        smooth_shading=True,
        specular=0.25,
        specular_power=12,
    )
    plotter.camera_position = camera_position()
    plotter.camera.SetParallelScale(0.60)
    plotter.camera.SetClippingRange(0.01, 50.0)
    plotter.add_light(pv.Light(position=(0.15, -1.55, 1.65), intensity=0.45))
    plotter.add_light(pv.Light(position=(1.9, 0.55, 1.15), intensity=0.20))

    image = plotter.screenshot(transparent_background=False, return_img=True)
    plotter.close()
    return crop_white_margins(image, padding=8)


def add_time_label(ax, time_value: float) -> None:
    ax.text(
        0.035,
        0.91,
        rf"$t={time_value:g}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "boxstyle": "round,pad=0.18",
            "alpha": 0.82,
        },
    )


def main() -> None:
    args = parse_args()
    if args.q_level is not None and args.per_snapshot_q_percentile:
        raise ValueError("--q-level and --per-snapshot-q-percentile are mutually exclusive.")
    active_global_q_percentile = None
    if args.q_level is None and not args.per_snapshot_q_percentile:
        active_global_q_percentile = args.global_q_percentile
    configure_matplotlib()

    with h5py.File(args.h5_path, "r") as handle:
        topology = handle["Mesh/Grid/topology"][:]
        geometry = handle["Mesh/Grid/geometry"][:]
        keys = list(handle["Function/f"].keys())
        matched = match_time_keys(keys, args.times)
        base_grid = build_unstructured_grid(topology, geometry)
        velocity_snapshots = [handle[f"Function/f/{key}"][:] for _, key in matched]

    fixed_q_level = args.q_level
    if fixed_q_level is None and active_global_q_percentile is not None:
        fixed_q_level = global_q_level_from_snapshots(
            base_grid, velocity_snapshots, active_global_q_percentile
        )

    surfaces = []
    q_levels = []
    for velocity_snapshot in velocity_snapshots:
        surface, level = extract_qcriterion_surface(
            base_grid=base_grid,
            velocity_data=velocity_snapshot,
            explicit_q_level=fixed_q_level,
            q_percentile=args.q_percentile,
        )
        surfaces.append(surface)
        q_levels.append(level)
    speed_limits = speed_limits_from_surfaces(surfaces, velocity_snapshots)

    q_images = []
    for surface in surfaces:
        image = render_qcriterion_panel(
            base_grid=base_grid,
            surface=surface,
            speed_limits=speed_limits,
        )
        q_images.append(image)

    nrows = len(q_images)
    figure = plt.figure(figsize=(7.2, 7.3))
    grid = figure.add_gridspec(
        nrows,
        2,
        width_ratios=[1.0, 0.035],
        hspace=0.025,
        wspace=0.03,
    )

    for index, (image, (matched_time, _)) in enumerate(zip(q_images, matched)):
        ax = figure.add_subplot(grid[index, 0])
        ax.imshow(image)
        ax.axis("off")
        add_time_label(ax, matched_time)

    cbar_start = max((nrows - 2) // 2, 0)
    cbar_stop = min(cbar_start + 2, nrows)
    colorbar_axis = figure.add_subplot(grid[cbar_start:cbar_stop, 1])
    colorbar = figure.colorbar(
        ScalarMappable(norm=Normalize(*speed_limits), cmap="viridis"),
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.set_label(r"$|\mathbf{u}|$")
    colorbar_axis.yaxis.set_ticks_position("right")

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output_stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    figure.savefig(args.output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    matched_pairs = ", ".join(
        f"{target:.2f}->{actual:.5f}" for target, (actual, _) in zip(args.times, matched)
    )
    level_text = ", ".join(f"{level:.4g}" for level in q_levels)
    print(f"Saved {args.output_stem.with_suffix('.pdf')}")
    print(f"Saved {args.output_stem.with_suffix('.png')}")
    print(f"Matched times: {matched_pairs}")
    if args.q_level is not None:
        print(f"Q-level mode: explicit {args.q_level:.6g}")
    elif active_global_q_percentile is not None:
        print(
            f"Q-level mode: fixed global {active_global_q_percentile:g}th percentile "
            f"({fixed_q_level:.6g})"
        )
    else:
        print(f"Q-level mode: per-snapshot {args.q_percentile:g}th percentile")
    print(f"Q levels: {level_text}")


if __name__ == "__main__":
    main()
