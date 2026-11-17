import gmsh
import meshio
from pathlib import Path


HERE = Path(__file__).resolve().parent

def generate_mesh():
    gmsh.initialize()
    gmsh.model.add("splitter")

    lc = 0.05  # Global mesh size
    lc_fine = 0.01  # Corner refinement mesh size
    lc_boundary = 0.0125  # Inlet/outlet cross-section refinement mesh size

    # Points based on the geometry
    p1 = gmsh.model.geo.addPoint(0, 0.5, 0, lc)
    p2 = gmsh.model.geo.addPoint(0, 0, 0, lc)
    p3 = gmsh.model.geo.addPoint(0.5, 0, 0, lc)
    p4 = gmsh.model.geo.addPoint(0.5, -0.5, 0, lc)
    p5 = gmsh.model.geo.addPoint(8.0, -0.5, 0, lc)
    p6 = gmsh.model.geo.addPoint(8.0, -0.1, 0, lc)
    p7 = gmsh.model.geo.addPoint(1.5, -0.1, 0, lc)
    p8 = gmsh.model.geo.addPoint(1.5, 0.2, 0, lc)
    p9 = gmsh.model.geo.addPoint(8.0, 0.2, 0, lc)
    p10 = gmsh.model.geo.addPoint(8.0, 0.5, 0, lc)

    # Lines
    l1 = gmsh.model.geo.addLine(p1, p2)    # Inlet
    l2 = gmsh.model.geo.addLine(p2, p3)    # Wall (Step top)
    l3 = gmsh.model.geo.addLine(p3, p4)    # Wall (Step right)
    l4 = gmsh.model.geo.addLine(p4, p5)    # Wall (Bottom)
    l5 = gmsh.model.geo.addLine(p5, p6)    # Outlet Bottom
    l6 = gmsh.model.geo.addLine(p6, p7)    # Wall (Splitter bottom)
    l7 = gmsh.model.geo.addLine(p7, p8)    # Wall (Splitter left)
    l8 = gmsh.model.geo.addLine(p8, p9)    # Wall (Splitter top)
    l9 = gmsh.model.geo.addLine(p9, p10)   # Outlet Top
    l10 = gmsh.model.geo.addLine(p10, p1)  # Wall (Top)

    # Curve loop / Surface
    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4, l5, l6, l7, l8, l9, l10])
    surf = gmsh.model.geo.addPlaneSurface([cl])

    gmsh.model.geo.synchronize()

    # Physical Groups
    # 1: Inlet
    gmsh.model.addPhysicalGroup(1, [l1], 1)
    gmsh.model.setPhysicalName(1, 1, "Inlet")

    # 2: Outlets
    gmsh.model.addPhysicalGroup(1, [l5, l9], 2)
    gmsh.model.setPhysicalName(1, 2, "Outlet")

    # 3: Walls
    gmsh.model.addPhysicalGroup(1, [l2, l3, l4, l6, l7, l8, l10], 3)
    gmsh.model.setPhysicalName(1, 3, "Wall")

    # 10: Domain
    gmsh.model.addPhysicalGroup(2, [surf], 10)
    gmsh.model.setPhysicalName(2, 10, "Domain")

    # Corner Refinement
    gmsh.model.mesh.field.add("Distance", 1)
    # Refine at corners: step (p3) and splitter tips (p7, p8)
    gmsh.model.mesh.field.setNumbers(1, "NodesList", [p3, p7, p8])

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_fine)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", lc)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.075)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.75)

    # Cross-section refinement at inlet and the two right outlets. These
    # facets are used directly in the outflux and charge-drop diagnostics.
    gmsh.model.mesh.field.add("Distance", 3)
    gmsh.model.mesh.field.setNumbers(3, "CurvesList", [l1, l5, l9])
    gmsh.model.mesh.field.setNumber(3, "Sampling", 120)

    gmsh.model.mesh.field.add("Threshold", 4)
    gmsh.model.mesh.field.setNumber(4, "InField", 3)
    gmsh.model.mesh.field.setNumber(4, "SizeMin", lc_boundary)
    gmsh.model.mesh.field.setNumber(4, "SizeMax", lc)
    gmsh.model.mesh.field.setNumber(4, "DistMin", 0.04)
    gmsh.model.mesh.field.setNumber(4, "DistMax", 0.45)

    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)

    # Generate Mesh
    gmsh.model.mesh.generate(2)
    gmsh.write(str(HERE / "bifurcated_tube.msh"))
    gmsh.finalize()


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:, :2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]})
    return out_mesh

if __name__ == "__main__":
    generate_mesh()
    
    # Convert mesh
    msh = meshio.read(HERE / "bifurcated_tube.msh")
    triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
    meshio.write(HERE / "mesh_domain.xdmf", triangle_mesh)
    
    line_mesh = create_mesh(msh, "line", prune_z=True)
    meshio.write(HERE / "mesh_boundary.xdmf", line_mesh)
    print("Mesh generation and conversion complete.")
