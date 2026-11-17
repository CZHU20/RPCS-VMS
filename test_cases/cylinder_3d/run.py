"""Run the three-dimensional cylinder RPCS-VMS calculation."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.run_case import run_case
from test_cases.cylinder_3d.parameters import CASE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continuity-monitor",
        action="store_true",
        help="Write velocity-divergence diagnostics during the calculation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run_case(CASE, continuity_monitor=arguments.continuity_monitor)
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
