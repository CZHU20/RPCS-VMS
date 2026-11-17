import meshio
import numpy as np
from pathlib import Path


def create_mesh(mesh, cell_type, data_name="name_to_read"):
    """
    Extract cells of a given type and merge them into one block.

    Writing a single block avoids generating a Mixed topology in XDMF.
    """
    cells = []
    data = []

    # meshio stores each cell type in separate blocks, so collect all blocks
    # of the requested type and merge them before exporting.
    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type == cell_type:
            cells.append(cell_block.data)

            try:
                phys_tags = mesh.cell_data["gmsh:physical"][i]
            except KeyError:
                phys_tags = np.zeros(len(cell_block.data), dtype=np.int32)
            data.append(phys_tags)

    if not cells:
        return None

    # Flatten to a single array so meshio writes a pure Quadrilateral or
    # Triangle topology instead of a Mixed topology.
    all_cells = np.vstack(cells)
    all_data = np.concatenate(data)

    points_2d = mesh.points[:, :2]

    new_mesh = meshio.Mesh(
        points=points_2d,
        cells=[(cell_type, all_cells)],
        cell_data={data_name: [all_data]}
    )
    return new_mesh

def convert_msh_to_xdmf(msh_file):
    msh_file = Path(msh_file)
    output_dir = msh_file.resolve().parent
    print(f"Reading {msh_file}...")
    mesh = meshio.read(msh_file)

    # --- 1. Export the computational domain ---
    print("Processing Domain (Quad)...")
    domain_mesh = create_mesh(mesh, "quad")

    if domain_mesh:
        meshio.write(output_dir / "mesh_domain.xdmf", domain_mesh)
        print("Saved mesh_domain.xdmf (2D geometry, quadrilateral topology)")
    else:
        print("No quadrilateral cells found. Checking for triangles...")
        domain_mesh_tri = create_mesh(mesh, "triangle")
        if domain_mesh_tri:
            meshio.write(output_dir / "mesh_domain.xdmf", domain_mesh_tri)
            print("Saved mesh_domain.xdmf (2D geometry, triangle topology)")
        else:
            print("Error: no domain cells found.")

    # --- 2. Export the boundary markers ---
    print("Processing Boundary (Line)...")
    boundary_mesh = create_mesh(mesh, "line")

    if boundary_mesh:
        meshio.write(output_dir / "mesh_boundary.xdmf", boundary_mesh)
        print("Saved mesh_boundary.xdmf")
    else:
        print("Warning: no boundary lines found.")

if __name__ == "__main__":
    INPUT_FILE = Path(__file__).resolve().with_name("cylinder_2d_quad_mesh.msh")

    if INPUT_FILE.exists():
        convert_msh_to_xdmf(INPUT_FILE)
    else:
        print(f"File not found: {INPUT_FILE}")
