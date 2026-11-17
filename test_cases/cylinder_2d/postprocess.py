"""Calculate two-dimensional cylinder metrics and generate its plots."""

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    subprocess.run([sys.executable, str(HERE / "extract_metrics.py")], check=True)
    subprocess.run([sys.executable, str(HERE / "plot_streamlines_with_mesh.py")], check=True)
