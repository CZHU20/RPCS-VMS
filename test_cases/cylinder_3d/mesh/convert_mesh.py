import meshio
import numpy as np
from pathlib import Path


def create_mesh(mesh, cell_type, data_name="name_to_read"):
    """
    Extract cells of a given type together with their physical markers.
    """
    cells = []
    data = []

    # meshio stores each cell type in separate blocks. Merge the requested
    # blocks before exporting to keep the XDMF topology simple.
    for i, cell_block in enumerate(mesh.cells):
        if cell_block.type == cell_type:
            cells.append(cell_block.data)

            try:
                phys_tags = mesh.cell_data["gmsh:physical"][i]
            except KeyError:
                print(f"Warning: no physical tags found for {cell_type} block {i}.")
                phys_tags = np.zeros(len(cell_block.data), dtype=np.int32)

            data.append(phys_tags)

    if not cells:
        return None

    all_cells = np.vstack(cells)
    all_data = np.concatenate(data)

    new_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[(cell_type, all_cells)],
        cell_data={data_name: [all_data]}
    )
    return new_mesh

def convert_hex_msh(msh_file, domain_out="mesh_domain.xdmf", boundary_out="mesh_boundary.xdmf"):
    msh_file = Path(msh_file)
    output_dir = msh_file.resolve().parent
    print(f"Reading {msh_file}...")

    try:
        mesh = meshio.read(msh_file)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("Processing Volume (Hexahedron)...")
    domain_mesh = create_mesh(mesh, "hexahedron", data_name="Grid")

    if domain_mesh:
        meshio.write(output_dir / domain_out, domain_mesh)
        print(f"Volume mesh saved to: {domain_out}")
    else:
        print("Error: no hexahedral cells found. Is this a 3D hex mesh?")

    print("Processing Boundary (Quad)...")
    boundary_mesh = create_mesh(mesh, "quad", data_name="Grid")

    if boundary_mesh:
        meshio.write(output_dir / boundary_out, boundary_mesh)
        print(f"Boundary mesh saved to: {boundary_out}")
    else:
        print("Warning: no quadrilateral boundary cells found. Did you define physical surfaces?")

if __name__ == "__main__":
    INPUT_FILE = Path(__file__).resolve().with_name("cylinder_3d_benchmark.msh")

    if INPUT_FILE.exists():
        convert_hex_msh(INPUT_FILE)
    else:
        print(f"Input file not found: {INPUT_FILE}")
