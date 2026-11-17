import argparse
import csv
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import meshio
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solver.reporting import (
    COLORS,
    configure_matplotlib,
    save_figure,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract bifurcated tube observables.")
    parser.add_argument(
        "--velocity-h5",
        type=Path,
        default=CASE_DIR / "output" / "solution_v.h5",
    )
    parser.add_argument(
        "--pressure-h5",
        type=Path,
        default=CASE_DIR / "output" / "solution_p.h5",
    )
    parser.add_argument("--time", type=float, default=50.0)
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=CASE_DIR / "mesh" / "mesh_boundary.xdmf",
        help="Tagged boundary mesh used for the boundary-integral diagnostics.",
    )
    parser.add_argument(
        "--p2p1-reference-csv",
        type=Path,
        default=CASE_DIR / "output" / "monolithic_p2p1_steady_metrics.csv",
        help="Monolithic Taylor-Hood P2/P1 steady reference metrics.",
    )
    return parser.parse_args()


def parse_time_key(key):
    return float(key.replace("_", "."))


def match_time_key(keys, target):
    return min(keys, key=lambda key: abs(parse_time_key(key) - target))


def sorted_time_keys(keys):
    return sorted(keys, key=parse_time_key)


def load_reference_values(path):
    required = {
        "total_kinetic_energy", "top_outflux", "bottom_outflux",
        "top_charge_drop", "bottom_charge_drop",
    }
    values = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            quantity = row.get("quantity")
            value = row.get("monolithic_p2p1")
            if quantity in required and value not in (None, ""):
                values[quantity] = float(value)
    if required - values.keys() or not all(np.isfinite(value) for value in values.values()):
        raise ValueError(f"Incomplete or non-finite monolithic reference: {path}")
    return values


def liu_prescribed_influx(time):
    time = np.asarray(time, dtype=float)
    ramp_time = np.minimum(time, 1.0)
    return 0.25 * (1.0 - np.cos(np.pi * ramp_time))


def load_field(path, target_time, components):
    with h5py.File(path, "r") as handle:
        geometry = handle["Mesh/Grid/geometry"][:, :2]
        topology = handle["Mesh/Grid/topology"][:]
        key = match_time_key(list(handle["Function/f"].keys()), target_time)
        data = handle[f"Function/f/{key}"][:]
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return geometry, topology, data[:, :components], parse_time_key(key)


def mesh_to_solution_node_map(mesh_points, solution_points, decimals=12):
    coordinate_to_index = {
        tuple(np.round(point, decimals)): index
        for index, point in enumerate(solution_points[:, :2])
    }
    mapping = []
    for point in mesh_points[:, :2]:
        key = tuple(np.round(point, decimals))
        if key not in coordinate_to_index:
            raise ValueError(f"Could not map boundary point {point} to the solution mesh.")
        mapping.append(coordinate_to_index[key])
    return np.asarray(mapping, dtype=int)


def load_boundary_lines(mesh_path, solution_points):
    mesh = meshio.read(mesh_path)
    mesh_points = mesh.points[:, :2]
    mesh_to_solution = mesh_to_solution_node_map(mesh_points, solution_points)
    line_blocks = []
    tag_blocks = []
    tag_data = mesh.cell_data.get("gmsh:physical", mesh.cell_data.get("name_to_read"))
    if tag_data is None:
        raise ValueError(f"No boundary tags found in {mesh_path}.")
    for block, tags in zip(mesh.cells, tag_data):
        if block.type == "line":
            line_blocks.append(block.data)
            tag_blocks.append(tags)
    if not line_blocks:
        raise ValueError(f"No line boundary cells found in {mesh_path}.")
    return (
        mesh_points,
        mesh_to_solution,
        np.vstack(line_blocks),
        np.concatenate(tag_blocks),
    )


def boundary_integrals(mesh_points, mesh_to_solution, lines, tags, velocity, pressure, selector, normal):
    length = 0.0
    flux = 0.0
    normal_square = 0.0
    pressure_integral = 0.0
    for (mesh_i, mesh_j), tag in zip(lines, tags):
        if not selector(mesh_i, mesh_j, tag):
            continue
        sol_i = mesh_to_solution[mesh_i]
        sol_j = mesh_to_solution[mesh_j]
        point_i = mesh_points[mesh_i]
        point_j = mesh_points[mesh_j]
        segment_length = float(np.linalg.norm(point_j - point_i))
        normal_i = float(np.dot(velocity[sol_i, :2], normal))
        normal_j = float(np.dot(velocity[sol_j, :2], normal))
        length += segment_length
        flux += 0.5 * segment_length * (normal_i + normal_j)
        normal_square += (
            segment_length
            / 3.0
            * (normal_i * normal_i + normal_i * normal_j + normal_j * normal_j)
        )
        pressure_integral += 0.5 * segment_length * (
            pressure[sol_i, 0] + pressure[sol_j, 0]
        )
    if length <= 0.0:
        raise ValueError("Selected boundary has zero length.")
    return {
        "length": length,
        "flux": flux,
        "normal_square": normal_square,
        "pressure_integral": pressure_integral,
    }


def boundary_diagnostics(mesh_path, geometry, velocity, pressure):
    mesh_points, mesh_to_solution, lines, tags = load_boundary_lines(mesh_path, geometry)
    inlet = boundary_integrals(
        mesh_points,
        mesh_to_solution,
        lines,
        tags,
        velocity,
        pressure,
        selector=lambda _i, _j, tag: tag == 1,
        normal=np.array([-1.0, 0.0]),
    )
    top = boundary_integrals(
        mesh_points,
        mesh_to_solution,
        lines,
        tags,
        velocity,
        pressure,
        selector=lambda i, j, tag: tag == 2 and mesh_points[[i, j], 1].mean() > 0.0,
        normal=np.array([1.0, 0.0]),
    )
    bottom = boundary_integrals(
        mesh_points,
        mesh_to_solution,
        lines,
        tags,
        velocity,
        pressure,
        selector=lambda i, j, tag: tag == 2 and mesh_points[[i, j], 1].mean() < 0.0,
        normal=np.array([1.0, 0.0]),
    )
    return inlet, top, bottom, float(mesh_points[:, 0].max())


def outflux_history(velocity_h5, mesh_path):
    with h5py.File(velocity_h5, "r") as handle:
        geometry = handle["Mesh/Grid/geometry"][:, :2]
        mesh_points, mesh_to_solution, lines, tags = load_boundary_lines(
            mesh_path, geometry
        )
        rows = []
        for key in sorted_time_keys(handle["Function/f"].keys()):
            time = parse_time_key(key)
            data = handle[f"Function/f/{key}"][:]
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            velocity = data[:, :2]
            inlet = boundary_integrals(
                mesh_points,
                mesh_to_solution,
                lines,
                tags,
                velocity,
                pressure=np.zeros((velocity.shape[0], 1)),
                selector=lambda _i, _j, tag: tag == 1,
                normal=np.array([-1.0, 0.0]),
            )
            top = boundary_integrals(
                mesh_points,
                mesh_to_solution,
                lines,
                tags,
                velocity,
                pressure=np.zeros((velocity.shape[0], 1)),
                selector=lambda i, j, tag: tag == 2
                and mesh_points[[i, j], 1].mean() > 0.0,
                normal=np.array([1.0, 0.0]),
            )
            bottom = boundary_integrals(
                mesh_points,
                mesh_to_solution,
                lines,
                tags,
                velocity,
                pressure=np.zeros((velocity.shape[0], 1)),
                selector=lambda i, j, tag: tag == 2
                and mesh_points[[i, j], 1].mean() < 0.0,
                normal=np.array([1.0, 0.0]),
            )
            top_flux = top["flux"]
            bottom_flux = bottom["flux"]
            outlet_sum = top_flux + bottom_flux
            computed_influx = -inlet["flux"]
            prescribed_influx = float(liu_prescribed_influx(time))
            if abs(outlet_sum) > 1.0e-14:
                top_share = top_flux / outlet_sum
                bottom_share = bottom_flux / outlet_sum
            else:
                top_share = np.nan
                bottom_share = np.nan
            rows.append(
                {
                    "time": time,
                    "top_outflux": top_flux,
                    "bottom_outflux": bottom_flux,
                    "outlet_sum": outlet_sum,
                    "computed_influx": computed_influx,
                    "prescribed_influx": prescribed_influx,
                    "top_share": top_share,
                    "bottom_share": bottom_share,
                    "mass_defect_vs_prescribed": outlet_sum - prescribed_influx,
                    "mass_defect_vs_boundary": outlet_sum - computed_influx,
                }
            )
    return rows


def plot_outflux_history(rows, out_dir):
    configure_matplotlib()
    plot_rows = [row for row in rows if row["time"] <= 10.0 + 1.0e-12]
    time = np.array([row["time"] for row in plot_rows], dtype=float)
    top = np.array([row["top_outflux"] for row in plot_rows], dtype=float)
    bottom = np.array([row["bottom_outflux"] for row in plot_rows], dtype=float)
    outlet_sum = np.array([row["outlet_sum"] for row in plot_rows], dtype=float)
    prescribed = np.array([row["prescribed_influx"] for row in plot_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 2.75))
    ax.plot(time, top, color=COLORS["orange"], label=r"$Q_{\rm top}$")
    ax.plot(time, bottom, color=COLORS["blue"], label=r"$Q_{\rm bottom}$")
    ax.plot(time, outlet_sum, color="black", label=r"$Q_{\rm top}+Q_{\rm bottom}$")
    ax.plot(
        time,
        prescribed,
        color=COLORS["gray"],
        linestyle="none",
        marker="o",
        markersize=4.0,
        markevery=20,
        markerfacecolor="none",
        markeredgewidth=0.9,
        label=r"$Q_{\rm in}$",
    )
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"Flux")
    ax.set_xlim(float(time.min()), float(time.max()))
    ax.set_ylim(-0.02, 0.56)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        borderaxespad=0.0,
        ncol=1,
    )

    fig.tight_layout()
    return save_figure(
        fig,
        out_dir / "bifurcated_tube_outflux_distribution",
    )


def total_kinetic_energy(geometry, topology, velocity, density=1.0):
    points = geometry[:, :2]
    energy = 0.0
    for cell in topology:
        p0, p1, p2 = points[cell[:3]]
        area = 0.5 * abs(
            (p1[0] - p0[0]) * (p2[1] - p0[1])
            - (p1[1] - p0[1]) * (p2[0] - p0[0])
        )
        values = velocity[cell[:3], :2]
        nodal = np.sum(values * values, axis=1)
        cross_terms = (
            float(np.dot(values[0], values[1]))
            + float(np.dot(values[1], values[2]))
            + float(np.dot(values[2], values[0]))
        )
        energy += 0.5 * density * area / 6.0 * (float(np.sum(nodal)) + cross_terms)
    return float(energy)


def main():
    args = parse_args()
    p2p1_reference = load_reference_values(args.p2p1_reference_csv)
    geometry, topology, velocity, matched_time = load_field(args.velocity_h5, args.time, 2)
    p_geometry, _, pressure, _ = load_field(args.pressure_h5, args.time, 1)
    if geometry.shape != p_geometry.shape or not np.allclose(geometry, p_geometry):
        raise ValueError("Velocity and pressure output meshes do not share the same nodes.")

    density = 1.0
    inlet, top, bottom, outlet_x = boundary_diagnostics(
        args.mesh_path, geometry, velocity, pressure
    )

    kinetic_energy = total_kinetic_energy(geometry, topology, velocity, density=density)
    top_outflux = top["flux"]
    bottom_outflux = bottom["flux"]
    top_velocity_part = 0.5 * (inlet["normal_square"] - top["normal_square"])
    bottom_velocity_part = 0.5 * (inlet["normal_square"] - bottom["normal_square"])
    top_pressure_part = (
        inlet["pressure_integral"] - top["pressure_integral"]
    ) / density
    bottom_pressure_part = (
        inlet["pressure_integral"] - bottom["pressure_integral"]
    ) / density
    top_charge_drop = top_velocity_part + top_pressure_part
    bottom_charge_drop = bottom_velocity_part + bottom_pressure_part

    rows = [
        {
            "quantity": "matched_time",
            "present": matched_time,
            "p2p1_reference": "",
            "unit": "",
        },
        {
            "quantity": "outlet_x",
            "present": outlet_x,
            "p2p1_reference": "",
            "unit": "",
        },
        {
            "quantity": "inlet_length",
            "present": inlet["length"],
            "p2p1_reference": "",
            "unit": "m",
        },
        {
            "quantity": "top_outlet_length",
            "present": top["length"],
            "p2p1_reference": "",
            "unit": "m",
        },
        {
            "quantity": "bottom_outlet_length",
            "present": bottom["length"],
            "p2p1_reference": "",
            "unit": "m",
        },
        {
            "quantity": "total_kinetic_energy",
            "present": kinetic_energy,
            "p2p1_reference": p2p1_reference["total_kinetic_energy"],
            "unit": "kg m s^-2",
        },
        {
            "quantity": "top_outflux",
            "present": top_outflux,
            "p2p1_reference": p2p1_reference["top_outflux"],
            "unit": "m^2 s^-1",
        },
        {
            "quantity": "bottom_outflux",
            "present": bottom_outflux,
            "p2p1_reference": p2p1_reference["bottom_outflux"],
            "unit": "m^2 s^-1",
        },
        {
            "quantity": "top_charge_drop",
            "present": top_charge_drop,
            "p2p1_reference": p2p1_reference["top_charge_drop"],
            "unit": "m^3 s^-2",
        },
        {
            "quantity": "bottom_charge_drop",
            "present": bottom_charge_drop,
            "p2p1_reference": p2p1_reference["bottom_charge_drop"],
            "unit": "m^3 s^-2",
        },
        {
            "quantity": "top_charge_velocity_part",
            "present": top_velocity_part,
            "p2p1_reference": "",
            "unit": "m^3 s^-2",
        },
        {
            "quantity": "top_charge_pressure_part",
            "present": top_pressure_part,
            "p2p1_reference": "",
            "unit": "m^3 s^-2",
        },
        {
            "quantity": "bottom_charge_velocity_part",
            "present": bottom_velocity_part,
            "p2p1_reference": "",
            "unit": "m^3 s^-2",
        },
        {
            "quantity": "bottom_charge_pressure_part",
            "present": bottom_pressure_part,
            "p2p1_reference": "",
            "unit": "m^3 s^-2",
        },
    ]

    out_dir = CASE_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        out_dir / "bifurcated_tube_metrics.csv",
        rows,
        ["quantity", "present", "p2p1_reference", "unit"],
    )
    history_rows = outflux_history(args.velocity_h5, args.mesh_path)
    write_csv(
        out_dir / "bifurcated_tube_outflux_history.csv",
        history_rows,
        [
            "time",
            "top_outflux",
            "bottom_outflux",
            "outlet_sum",
            "computed_influx",
            "prescribed_influx",
            "top_share",
            "bottom_share",
            "mass_defect_vs_prescribed",
            "mass_defect_vs_boundary",
        ],
    )
    plot_outflux_history(history_rows, out_dir)
    plt.close("all")

    for row in rows:
        print(f"{row['quantity']}: {row['present']} {row['unit']}")
    print(
        "charge_drop_components: "
        f"top_velocity={top_velocity_part:.12g}, "
        f"top_pressure={top_pressure_part:.12g}, "
        f"bottom_velocity={bottom_velocity_part:.12g}, "
        f"bottom_pressure={bottom_pressure_part:.12g}"
    )


if __name__ == "__main__":
    main()
