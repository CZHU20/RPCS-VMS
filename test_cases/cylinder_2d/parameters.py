"""Physical and numerical parameters for the two-dimensional cylinder test."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

CASE = {
    "case_name": "cylinder_2d",
    "mesh": {
        "domain": str(HERE / "mesh/mesh_domain.xdmf"),
        "boundary": str(HERE / "mesh/mesh_boundary.xdmf"),
    },
    "physics": {
        "equation": "NS",
        "formula": "rpcs",
        "stabilization": "t",
        "nu": 0.001,
    },
    "time": {"dt": 0.000625, "t_end": 8.0, "BDF_order": 2},
    "solver": {
        "corrector_mass": "consistent",
        "velocity_update": True,
        "convective_stabilization": True,
        "stabilization_alpha": "auto",
    },
    "boundary_conditions": [
        {
            "id": 1,
            "type": "velocity",
            "form": "expression",
            "value": [
                "4.0*1.5*sin(pi*t/8)*x[1]*(0.41-x[1])/(0.41**2)",
                "0.0",
            ],
        },
        {"id": 2, "type": "traction", "form": "constant", "value": [0.0, 0.0]},
        {"id": 3, "type": "velocity", "form": "constant", "value": [0.0, 0.0]},
        {"id": 4, "type": "velocity", "form": "constant", "value": [0.0, 0.0]},
    ],
    "force_monitor": {
        "enabled": True,
        "boundary_id": 4,
        "pressure_points": [[0.15, 0.2], [0.25, 0.2]],
        "characteristic_velocity": 1.0,
        "characteristic_length": 0.1,
        "frequency": 1,
        "file": "force_coefficients.csv",
    },
    "output": {"start_time": 0.0, "frequency": 20, "dir": str(OUTPUT_DIR)},
}
