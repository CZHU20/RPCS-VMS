"""Calculate three-dimensional cylinder metrics and generate its plots."""

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    subprocess.run([sys.executable, str(HERE / "extract_cylinder_3d_metrics.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "check_cylinder_fields.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "plot_cylinder_3d_q_criterion.py")], check=True)
