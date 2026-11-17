"""Generate the FEATFLOW 3D cylinder-flow coarsest mesh with boundary and volume tags.

The 2D cross-section is encoded explicitly as 24 quadrilateral cells following
the published coarse-mesh picture. The cylinder-adjacent ring is kept circular
in the coarse mesh, then the surface is extruded in the z direction with 4
layers to obtain 96 hexahedra. The generated XDMF files are written as
mesh_domain.xdmf and mesh_boundary.xdmf so they can be consumed directly by
the three-dimensional cylinder test.

Physical tags:
1 = Inlet
2 = Outlet
3 = Walls
4 = Cylinder
10 = Fluid volume
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import gmsh
import meshio
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MESH_NAME = "flow_past_cylinder_3d_coarsest"
MSH_FILE = SCRIPT_DIR / f"{MESH_NAME}.msh"
DOMAIN_XDMF = SCRIPT_DIR / "mesh_domain.xdmf"
BOUNDARY_XDMF = SCRIPT_DIR / "mesh_boundary.xdmf"
SECTION_SVG = SCRIPT_DIR / f"{MESH_NAME}_section_labeled.svg"
SECTION_PNG = SCRIPT_DIR / f"{MESH_NAME}_section_labeled.png"
POINTS_CSV = SCRIPT_DIR / f"{MESH_NAME}_section_points.csv"
CELLS_CSV = SCRIPT_DIR / f"{MESH_NAME}_section_cells.csv"


# Geometry from the 3D cylinder benchmark picture.
L = 2.5
H = 0.41
W = 0.41
D = 0.1
R = D / 2.0
X_C = 0.5
Y_C = 0.2
LC = 0.1

# Coarsest FEATFLOW level from the published sketch: 24 quads in 2D,
# extruded with 4 layers to obtain 96 hexahedra.
N_SPAN = 4

# Circular first layer around the cylinder.
FIRST_LAYER_RADIUS = 0.10
FIRST_LAYER_DX = 0.5 * FIRST_LAYER_RADIUS
FIRST_LAYER_DY = 0.8660254037844386 * FIRST_LAYER_RADIUS


X_LEFT_SPLIT = 0.3
X_TOP_MID = 0.55
X_RIGHT_SPLIT = 0.85
X_COL_1 = 1.15
X_COL_2 = 1.65

Y_MID = 0.2
Y_RING_TOP = 0.28
Y_RING_BOT = 0.12

X_RING_LEFT = 0.4
X_RING_UL = 0.44
X_RING_RIGHT_MID = 0.6
X_RING_RIGHT = 0.7

HEX_DX = 0.5 * R
HEX_DY = 0.8660254037844386 * R

INNER_RING_POINTS = ("in_l", "in_ul", "in_ur", "in_r", "in_lr", "in_ll")
OUTER_RING_POINTS = ("olm", "oul", "ot", "orm", "ob", "oll")


def build_cross_section_definition():
    point_coords = {
        "bl": (0.0, 0.0),
        "lm": (0.0, Y_MID),
        "tl": (0.0, H),
        "x1b": (X_LEFT_SPLIT, 0.0),
        "x1m": (X_LEFT_SPLIT, Y_MID),
        "x1t": (X_LEFT_SPLIT, H),
        "x2b": (X_TOP_MID, 0.0),
        "x2t": (X_TOP_MID, H),
        "x3b": (X_RIGHT_SPLIT, 0.0),
        "x3m": (X_RIGHT_SPLIT, Y_MID),
        "x3t": (X_RIGHT_SPLIT, H),
        "x4b": (X_COL_1, 0.0),
        "x4m": (X_COL_1, Y_MID),
        "x4t": (X_COL_1, H),
        "x5b": (X_COL_2, 0.0),
        "x5m": (X_COL_2, Y_MID),
        "x5t": (X_COL_2, H),
        "x6b": (L, 0.0),
        "x6m": (L, Y_MID),
        "x6t": (L, H),
        "olm": (X_C - FIRST_LAYER_RADIUS, Y_C),
        "oul": (X_C - FIRST_LAYER_DX, Y_C + FIRST_LAYER_DY),
        "ot": (X_C + FIRST_LAYER_DX, Y_C + FIRST_LAYER_DY),
        "orm": (X_C + FIRST_LAYER_RADIUS, Y_C),
        "ob": (X_C + FIRST_LAYER_DX, Y_C - FIRST_LAYER_DY),
        "oll": (X_C - FIRST_LAYER_DX, Y_C - FIRST_LAYER_DY),
        "ur": (X_RING_RIGHT, Y_RING_TOP),
        "rb": (X_RING_RIGHT, Y_MID),
        "lr": (X_RING_RIGHT, Y_RING_BOT),
        "in_l": (X_C - R, Y_C),
        "in_ul": (X_C - HEX_DX, Y_C + HEX_DY),
        "in_ur": (X_C + HEX_DX, Y_C + HEX_DY),
        "in_r": (X_C + R, Y_C),
        "in_lr": (X_C + HEX_DX, Y_C - HEX_DY),
        "in_ll": (X_C - HEX_DX, Y_C - HEX_DY),
    }

    surface_specs = [
        ["tl", "x1t", "x1m", "lm"],
        ["lm", "x1m", "x1b", "bl"],
        ["x3t", "x4t", "x4m", "x3m"],
        ["x3m", "x4m", "x4b", "x3b"],
        ["x4t", "x5t", "x5m", "x4m"],
        ["x4m", "x5m", "x5b", "x4b"],
        ["x5t", "x6t", "x6m", "x5m"],
        ["x5m", "x6m", "x6b", "x5b"],
        ["x1t", "x2t", "ot", "oul"],
        ["x2t", "x3t", "ur", "ot"],
        ["x1m", "x1t", "oul", "olm"],
        ["x1b", "x1m", "olm", "oll"],
        ["x1b", "x2b", "ob", "oll"],
        ["x2b", "x3b", "lr", "ob"],
        ["ot", "ur", "rb", "orm"],
        ["ur", "x3t", "x3m", "rb"],
        ["lr", "rb", "x3m", "x3b"],
        ["ob", "orm", "rb", "lr"],
        ["olm", "oul", "in_ul", "in_l"],
        ["oul", "ot", "in_ur", "in_ul"],
        ["ot", "orm", "in_r", "in_ur"],
        ["ob", "orm", "in_r", "in_lr"],
        ["oll", "ob", "in_lr", "in_ll"],
        ["olm", "oll", "in_ll", "in_l"],
    ]

    return point_coords, surface_specs


def write_cross_section_exports(point_coords, surface_specs):
    ordered_points = list(point_coords.items())

    with POINTS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["point_id", "point_name", "x", "y"])
        for point_id, (name, (x_coord, y_coord)) in enumerate(ordered_points, start=1):
            writer.writerow([point_id, name, f"{x_coord:.12g}", f"{y_coord:.12g}"])

    with CELLS_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell_id", "v1", "v2", "v3", "v4"])
        for cell_id, vertices in enumerate(surface_specs, start=1):
            writer.writerow([cell_id, *vertices])

    margin = 36.0
    canvas_width = 1800.0
    scale = (canvas_width - 2.0 * margin) / L
    canvas_height = 2.0 * margin + H * scale

    def map_point(coords):
        x_coord, y_coord = coords
        return margin + x_coord * scale, margin + (H - y_coord) * scale

    def is_ring_edge(start: str, end: str):
        return (
            start in INNER_RING_POINTS and end in INNER_RING_POINTS
        ) or (
            start in OUTER_RING_POINTS and end in OUTER_RING_POINTS
        )

    def sample_arc(start_coord, end_coord, sample_count: int = 20):
        start_vector = np.array(start_coord, dtype=np.float64) - np.array((X_C, Y_C), dtype=np.float64)
        end_vector = np.array(end_coord, dtype=np.float64) - np.array((X_C, Y_C), dtype=np.float64)
        radius = float(np.hypot(start_vector[0], start_vector[1]))
        start_angle = float(np.arctan2(start_vector[1], start_vector[0]))
        end_angle = float(np.arctan2(end_vector[1], end_vector[0]))
        delta = end_angle - start_angle
        if delta > np.pi:
            delta -= 2.0 * np.pi
        elif delta < -np.pi:
            delta += 2.0 * np.pi

        angles = np.linspace(start_angle, start_angle + delta, sample_count)
        return [
            (
                X_C + radius * np.cos(angle),
                Y_C + radius * np.sin(angle),
            )
            for angle in angles
        ]

    point_ids = {name: index for index, (name, _) in enumerate(ordered_points, start=1)}

    segments = []
    seen_edges = set()
    for vertices in surface_specs:
        for index, start in enumerate(vertices):
            end = vertices[(index + 1) % len(vertices)]
            edge_key = tuple(sorted((start, end)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            if is_ring_edge(start, end):
                arc_points = sample_arc(point_coords[start], point_coords[end])
                for arc_start, arc_end in zip(arc_points[:-1], arc_points[1:]):
                    segments.append((map_point(arc_start), map_point(arc_end)))
            else:
                segments.append((map_point(point_coords[start]), map_point(point_coords[end])))

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width:.0f}" height="{canvas_height:.0f}" viewBox="0 0 {canvas_width:.0f} {canvas_height:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for (x1, y1), (x2, y2) in segments:
        svg_parts.append(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="black" stroke-width="2"/>'
        )

    for name, coords in ordered_points:
        x_coord, y_coord = map_point(coords)
        point_id = point_ids[name]
        svg_parts.append(
            f'<circle cx="{x_coord:.3f}" cy="{y_coord:.3f}" r="4" fill="#d7263d"/>'
        )
        svg_parts.append(
            f'<text x="{x_coord + 8:.3f}" y="{y_coord - 8:.3f}" font-size="14" fill="#1d4ed8">{point_id}</text>'
        )

    svg_parts.append("</svg>")
    SECTION_SVG.write_text("\n".join(svg_parts), encoding="utf-8")

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return

    image = Image.new("RGB", (int(round(canvas_width)), int(round(canvas_height))), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for start_point, end_point in segments:
        draw.line([start_point, end_point], fill="black", width=2)

    for name, coords in ordered_points:
        x_coord, y_coord = map_point(coords)
        point_id = str(point_ids[name])
        draw.ellipse((x_coord - 4, y_coord - 4, x_coord + 4, y_coord + 4), fill="#d7263d")
        draw.text((x_coord + 8, y_coord - 10), point_id, fill="#1d4ed8", font=font)

    image.save(SECTION_PNG)


def create_mesh(mesh, cell_type, data_name="Grid"):
    """Extract one cell type together with its physical markers."""
    cells = []
    data = []

    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type != cell_type:
            continue

        cells.append(cell_block.data)

        try:
            phys_tags = mesh.cell_data["gmsh:physical"][i]
        except KeyError:
            phys_tags = np.zeros(len(cell_block.data), dtype=np.int32)

        data.append(phys_tags)

    if not cells:
        return None

    all_cells = np.vstack(cells)
    all_data = np.concatenate(data)

    return meshio.Mesh(
        points=mesh.points,
        cells=[(cell_type, all_cells)],
        cell_data={data_name: [all_data]},
    )


def write_xdmf_outputs(msh_path: Path, domain_out: Path, boundary_out: Path):
    mesh = meshio.read(msh_path)

    domain_mesh = create_mesh(mesh, "hexahedron", data_name="Grid")
    if domain_mesh is None:
        raise RuntimeError("No hexahedron cells were found in the generated mesh.")
    meshio.write(domain_out, domain_mesh)

    boundary_mesh = create_mesh(mesh, "quad", data_name="Grid")
    if boundary_mesh is None:
        raise RuntimeError("No quadrilateral boundary cells were found in the generated mesh.")
    meshio.write(boundary_out, boundary_mesh)


def select_circular_surfaces(radius: float, tolerance: float = 1.0e-8):
    selected = []

    for _, tag in gmsh.model.getEntities(2):
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
        if len(node_tags) == 0:
            continue

        coords = node_coords.reshape(-1, 3)
        radii = np.hypot(coords[:, 0] - X_C, coords[:, 1] - Y_C)
        if radii.size and np.max(np.abs(radii - radius)) < tolerance:
            selected.append(tag)

    return selected


def project_circular_surface_nodes(surface_tags, radius: float):
    if not surface_tags:
        return

    for surface_tag in surface_tags:
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes(2, surface_tag, includeBoundary=True)
        if len(node_tags) == 0:
            continue

        for index, node_tag in enumerate(node_tags.tolist()):
            x_coord = float(node_coords[3 * index])
            y_coord = float(node_coords[3 * index + 1])
            z_coord = float(node_coords[3 * index + 2])
            radial_x = x_coord - X_C
            radial_y = y_coord - Y_C
            radial_norm = float(np.hypot(radial_x, radial_y))
            if radial_norm == 0.0:
                continue

            projected_x = X_C + radius * radial_x / radial_norm
            projected_y = Y_C + radius * radial_y / radial_norm
            gmsh.model.mesh.setNode(int(node_tag), [projected_x, projected_y, z_coord], [])

    gmsh.model.mesh.rebuildNodeCache()


def refine_mesh(refine_levels: int):
    if refine_levels < 0:
        raise ValueError("refine_levels must be >= 0.")

    for refine_index in range(refine_levels):
        print(f"Refining mesh: pass {refine_index + 1}/{refine_levels}")
        gmsh.model.mesh.refine()
        gmsh.model.mesh.reclassifyNodes()
        project_circular_surface_nodes(select_cylinder_surfaces(), R)
        project_circular_surface_nodes(select_circular_surfaces(FIRST_LAYER_RADIUS), FIRST_LAYER_RADIUS)


def select_plane_surfaces(axis: str, value: float, tolerance: float = 1.0e-8):
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    selected = []

    for _, tag in gmsh.model.getEntities(2):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
        bbox_min = (xmin, ymin, zmin)[axis_index]
        bbox_max = (xmax, ymax, zmax)[axis_index]
        if abs(bbox_min - value) < tolerance and abs(bbox_max - value) < tolerance:
            selected.append(tag)

    return selected


def select_cylinder_surfaces(tolerance: float = 1.0e-8):
    return select_circular_surfaces(R, tolerance)


def generate_mesh(refine_levels: int = 0):
    point_coords, surface_specs = build_cross_section_definition()
    write_cross_section_exports(point_coords, surface_specs)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.add("flow_past_cylinder_3d_coarsest")

    geo = gmsh.model.geo

    def add_point(x_coord: float, y_coord: float):
        return geo.addPoint(x_coord, y_coord, 0.0, LC)

    p_center = add_point(X_C, Y_C)
    point_tags = {name: add_point(*coords) for name, coords in point_coords.items()}

    line_tags: dict[tuple[str, str], int] = {}

    def segment(start: str, end: str):
        key = (start, end)
        reverse_key = (end, start)
        if key in line_tags:
            return line_tags[key]
        if reverse_key in line_tags:
            return -line_tags[reverse_key]
        if (
            start in INNER_RING_POINTS and end in INNER_RING_POINTS
        ) or (
            start in OUTER_RING_POINTS and end in OUTER_RING_POINTS
        ):
            tag = geo.addCircleArc(point_tags[start], p_center, point_tags[end])
        else:
            tag = geo.addLine(point_tags[start], point_tags[end])
        line_tags[key] = tag
        return tag

    def add_surface(vertices: list[str]):
        curves = [segment(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
        loop = geo.addCurveLoop(curves)
        surface = geo.addPlaneSurface([loop])
        geo.mesh.setTransfiniteSurface(surface, cornerTags=[point_tags[name] for name in vertices])
        geo.mesh.setRecombine(2, surface)
        return surface

    surface_list = [add_surface(vertices) for vertices in surface_specs]

    for tag in {abs(tag) for tag in line_tags.values()}:
        geo.mesh.setTransfiniteCurve(tag, 2)

    geo.extrude([(2, surface) for surface in surface_list], 0.0, 0.0, W, numElements=[N_SPAN], recombine=True)
    geo.synchronize()

    inlet_surfaces = select_plane_surfaces("x", 0.0)
    outlet_surfaces = select_plane_surfaces("x", L)
    wall_surfaces = (
        select_plane_surfaces("y", 0.0)
        + select_plane_surfaces("y", H)
        + select_plane_surfaces("z", 0.0)
        + select_plane_surfaces("z", W)
    )
    volume_tags = [tag for _, tag in gmsh.model.getEntities(3)]

    gmsh.model.addPhysicalGroup(2, inlet_surfaces, 1)
    gmsh.model.setPhysicalName(2, 1, "Inlet")

    gmsh.model.addPhysicalGroup(2, outlet_surfaces, 2)
    gmsh.model.setPhysicalName(2, 2, "Outlet")

    gmsh.model.addPhysicalGroup(2, wall_surfaces, 3)
    gmsh.model.setPhysicalName(2, 3, "Walls")

    gmsh.model.addPhysicalGroup(3, volume_tags, 10)
    gmsh.model.setPhysicalName(3, 10, "Fluid")

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.reclassifyNodes()
    cylinder_surfaces = select_cylinder_surfaces()
    gmsh.model.addPhysicalGroup(2, cylinder_surfaces, 4)
    gmsh.model.setPhysicalName(2, 4, "Cylinder")
    project_circular_surface_nodes(cylinder_surfaces, R)
    project_circular_surface_nodes(select_circular_surfaces(FIRST_LAYER_RADIUS), FIRST_LAYER_RADIUS)
    refine_mesh(refine_levels)
    gmsh.write(str(MSH_FILE))
    gmsh.finalize()

    write_xdmf_outputs(MSH_FILE, DOMAIN_XDMF, BOUNDARY_XDMF)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the FEATFLOW 3D cylinder mesh.")
    parser.add_argument(
        "--refine-levels",
        type=int,
        default=0,
        help="Apply this many uniform gmsh refinement passes after the coarse mesh is generated.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_mesh(refine_levels=args.refine_levels)
