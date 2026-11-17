"""Parameters shared by the manufactured-solution tests."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"

COMMON_SOLVER = {
    "corrector_mass": "consistent",
    "velocity_update": True,
    "convective_stabilization": True,
    "stabilization_alpha": "auto",
}

STEADY = {
    "case_name": "manufactured_steady",
    "mesh": {"domain": "", "boundary": ""},
    "physics": {
        "equation": "NS",
        "formula": "rpcs",
        "stabilization": "t",
        "nu": 1.0,
    },
    "time": {"dt": 0.01, "t_end": 1.0, "BDF_order": 1},
    "solver": COMMON_SOLVER,
    "manufactured_solution": "steady",
    "output": {"start_time": 0.0, "frequency": 1, "dir": str(OUTPUT_DIR)},
}

UNSTEADY = {
    "case_name": "manufactured_unsteady",
    "mesh": {
        "domain": str(HERE / "square_quad_mesh/square_n128_domain.xdmf"),
        "boundary": str(HERE / "square_quad_mesh/square_n128_boundary.xdmf"),
    },
    "physics": {
        "equation": "NS",
        "formula": "rpcs",
        "stabilization": "t",
        "nu": 1.0,
    },
    "time": {"dt": 0.1, "t_end": 1.0, "BDF_order": 1},
    "solver": COMMON_SOLVER,
    "manufactured_solution": "unsteady",
    "output": {"start_time": 0.0, "frequency": 1, "dir": str(OUTPUT_DIR)},
}

SMALL_TIME_STEP = {
    "case_name": "manufactured_small_time_step",
    "mesh": {
        "domain": str(HERE / "square_triangular_mesh/square_n64_domain.xdmf"),
        "boundary": str(HERE / "square_triangular_mesh/square_n64_boundary.xdmf"),
    },
    "physics": {
        "equation": "NS",
        "formula": "rpcs",
        "stabilization": "t",
        "nu": 1.0,
    },
    "time": {"dt": 0.1, "t_end": 1.0, "BDF_order": 2},
    "solver": COMMON_SOLVER,
    "manufactured_solution": "unsteady",
    "output": {"start_time": 0.0, "frequency": 1, "dir": str(OUTPUT_DIR)},
}
