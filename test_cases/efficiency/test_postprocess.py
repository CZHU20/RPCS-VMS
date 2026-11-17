"""Check efficiency summaries without running a flow calculation."""

import unittest

from test_cases.efficiency.postprocess import aggregate_summary


class SummaryTests(unittest.TestCase):
    def test_three_runs_use_arithmetic_means(self):
        rows = [{"method": "RPCS-VMS", "run_id": f"run{i}",
                 "solve_wall_time": str(value), "mpi_ranks": "8",
                 "mean_linear_iterations_per_step": str(10 + value)}
                for i, value in enumerate((1., 2., 9.), start=1)]
        summary = aggregate_summary(rows)
        self.assertEqual(summary["solve_wall_time"], 4.)
        self.assertEqual(summary["mean_linear_iterations_per_step"], 14.)
        self.assertEqual(summary["solve_wall_time_min"], 1.)
        self.assertEqual(summary["solve_wall_time_max"], 9.)
        self.assertEqual(summary["mpi_ranks"], 8)
        self.assertEqual(summary["repeats"], 3)
        self.assertEqual(summary["run_id"], "mean")

    def test_single_run_is_preserved(self):
        summary = aggregate_summary([{"method": "RPCS-VMS", "run_id": "run1",
                                      "solve_wall_time": "3.5"}])
        self.assertEqual(summary["solve_wall_time"], 3.5)
        self.assertEqual(summary["solve_wall_time_min"], 3.5)
        self.assertEqual(summary["solve_wall_time_max"], 3.5)
        self.assertEqual(summary["repeats"], 1)
        self.assertEqual(summary["run_id"], "single")


if __name__ == "__main__":
    unittest.main()
