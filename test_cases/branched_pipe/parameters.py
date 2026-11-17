"""Physical and numerical parameters for the branched-pipe test."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

CASE = {
    "case_name": "branched_pipe",
    "mesh": {
        "domain": str(HERE / "mesh/mesh_domain.xdmf"),
        "boundary": str(HERE / "mesh/mesh_boundary.xdmf"),
    },
    "physics": {
        "equation": "NS",
        "formula": "rpcs",
        "stabilization": "t",
        "nu": 0.00166667,
    },
    "time": {"dt": 0.0005, "t_end": 50.0, "BDF_order": 2},
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
                "6.0*(1.0-cos(pi*min(t,1.0)))*x[1]*(1.0-2.0*x[1])",
                "0.0",
            ],
        },
        {"id": 2, "type": "traction", "form": "constant", "value": [0.0, 0.0]},
        {"id": 3, "type": "velocity", "form": "constant", "value": [0.0, 0.0]},
    ],
    "output": {
        "start_time": 0.0,
        "end_time": 10.0,
        "frequency": 20,
        "dir": str(OUTPUT_DIR),
    },
}
