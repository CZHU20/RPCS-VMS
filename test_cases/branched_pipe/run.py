"""Run the branched-pipe RPCS-VMS calculation."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solver.run_case import run_case
from test_cases.branched_pipe.parameters import CASE


if __name__ == "__main__":
    try:
        run_case(CASE)
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
