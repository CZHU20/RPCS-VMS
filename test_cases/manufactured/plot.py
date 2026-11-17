"""Regenerate all manufactured-test plots from calculated output."""

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main():
    commands = (
        [sys.executable, str(HERE / "run_spatial_convergence.py"), "--replot"],
        [sys.executable, str(HERE / "run_temporal_convergence.py"), "--mode", "replot"],
        [sys.executable, str(HERE / "run_boundary_pressure_comparison.py"), "--replot"],
        [sys.executable, str(HERE / "run_stabilization_effect.py"), "--replot"],
        [sys.executable, str(HERE / "run_pressure_stage_ablation.py"), "--replot"],
    )
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
