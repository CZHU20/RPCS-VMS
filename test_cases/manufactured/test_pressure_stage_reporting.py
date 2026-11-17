"""Check the pressure ablation figure, five-method table, and MPI graph."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.tri as mtri
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from test_cases.manufactured import run_pressure_stage_ablation as ablation
from test_cases.manufactured import run_stabilization_effect as stability
from test_cases.manufactured import run_boundary_pressure_comparison as boundary


class ReportingTests(unittest.TestCase):
    def setUp(self):
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])
        self.results, self.rows = [], []
        for group in ablation.GROUPS:
            for dt in (.1, .01, .001):
                values = np.array([-.7, -.1, .3, .8]) * (1. - dt)
                tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
                self.results.append({"group_id": group["id"], "dt": dt,
                                     "triangulation": tri, "p_values": values})
                self.rows.append({"variant": group["id"], "dt": dt, "mesh_n": 64,
                                  "final_time": 1., "max_abs_pressure": float(np.max(np.abs(values))),
                                  **{key: (i + 1) * .001 for i, key in enumerate(ablation.TABLE_METRICS)}})
        self.references = [{"variant": variant, "dt": .001, "mesh_n": 64, "final_time": 1.,
                            **{key: (i + 1) * .002 for i, key in enumerate(ablation.TABLE_METRICS)}}
                           for variant in ("rpcs", "monolithic")]

    def test_three_panels_at_smallest_step_and_current_table(self):
        original = copy.deepcopy(self.rows)
        with tempfile.TemporaryDirectory() as folder, patch.object(ablation, "save_figure") as save:
            out = Path(folder)
            ablation.render(out, self.results, self.rows, self.references)
            figure = save.call_args.args[0]
            self.assertEqual(len(figure.axes), 4)
            self.assertEqual([ax.get_title() for ax in figure.axes[:3]],
                             ["No recovery stabilization", "No increment stabilization", "Full RPCS-VMS"])
            for ax in figure.axes[:3]:
                self.assertEqual(ax.collections[0].norm.vmin, -1.)
                self.assertEqual(ax.collections[0].norm.vmax, 1.)
            table = (out / "pressure_stage_ablation.tex").read_text()
            self.assertIn(r"\begin{tabular}{@{}p{0.32\linewidth}cccc@{}}", table)
            self.assertIn(r"$\|p-p_h\|_{\Omega_h^b}$", table)
            self.assertIn(r"$\|\bm u-\bm u_h\|_{\Omega}$", table)
            self.assertNotIn(r"\max", table)
            self.assertIn(r"$\Delta t=10^{-3}$", table)
            self.assertIn("$T=1$", table)
            for variant, label in ablation.TABLE_METHODS:
                self.assertIn(label + " & ", table)
                row = next(r for r in self.rows + self.references if r["variant"] == variant and r["dt"] == .001)
                for key in ablation.TABLE_METRICS:
                    self.assertIn(ablation.scientific_error(row[key]), table)
            presentation = json.loads((out / "presentation.json").read_text())
            self.assertEqual(presentation["dt"], .001)
            self.assertEqual(presentation["color_range"], [-1., 1.])
            self.assertEqual(presentation["table_metrics"], list(ablation.TABLE_METRICS))
        self.assertEqual(original, self.rows)

    def test_missing_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(ablation, "save_figure"):
            with self.assertRaises(ValueError):
                ablation.render(Path(folder), self.results, self.rows, self.references[:1])

    def test_duplicate_or_missing_variant_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                ablation.plot_pressure_comparison(self.results[:-1], Path(folder), .001)
            with self.assertRaises(ValueError):
                ablation.plot_pressure_comparison(self.results + [self.results[-1]], Path(folder), .001)

    def test_pressure_outside_shared_scale_is_rejected(self):
        results = copy.deepcopy(self.results)
        results[-1]["p_values"][0] = 1.01
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            ablation.plot_pressure_comparison(results, Path(folder), .001)


class StabilityFigureTests(unittest.TestCase):
    def test_boundary_table_definitions_and_caption(self):
        keys = ("global_pressure_error", "boundary_pressure_error",
                "interior_pressure_error", "velocity_l2_error")
        rows = [{"method": method["title"], "dt": .001, "mesh_n": 64,
                 "final_time": .5, "bdf_order": 2,
                 **{key: .001 * (i + 1) for i, key in enumerate(keys)}}
                for method in boundary.METHODS]
        original = copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as folder:
            boundary.write_table(rows, Path(folder), unsteady=True)
            table = (Path(folder) / "boundary_pressure_comparison.tex").read_text()
        self.assertIn(r"$\|p-p_h\|_{\Omega}$", table)
        self.assertIn(r"$\|\bm u-\bm u_h\|_{\Omega}$", table)
        self.assertIn("$T=0.5$", table)
        self.assertIn(r"$\Delta t=10^{-3}$", table)
        self.assertNotIn("mean removal", table)
        self.assertEqual(original, rows)

    def test_pressure_tests_use_case_specific_end_times(self):
        with patch.object(sys, "argv", ["boundary"]):
            args = boundary.parse_args()
        self.assertTrue(args.unsteady)
        self.assertEqual(args.dt, .001)
        self.assertEqual(args.t_end, .5)
        with patch.object(sys, "argv", ["stability"]):
            self.assertEqual(stability.parse_args().t_end, 1.)
        self.assertEqual(ablation.SMALL_TIME_STEP["time"]["t_end"], 1.)
        with patch.object(sys, "argv", ["boundary", "--steady"]):
            self.assertEqual(boundary.parse_args().t_end, 1.)

    def test_partition_nodes_use_global_ids_and_owner_values(self):
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        triangles = np.array([[0, 1, 2], [1, 3, 2]])
        values = np.array([-.7, -.1, .3, .8])
        order = np.array([2, 3, 0, 1])
        inverse = np.argsort(order)
        copies = points[order].copy()
        copies[2:] += 1.e-13
        ghost_values = values[order].copy()
        ghost_values[2:] += .01
        merged, pressure = stability.merge_pressure_partitions([
            (triangles, points, values, np.arange(4), 2),
            (inverse[triangles], copies, ghost_values, order, 2),
        ])
        self.assertEqual(len(pressure), 4)
        self.assertEqual(len(merged.triangles), 2)
        np.testing.assert_array_equal(pressure, values)
        np.testing.assert_array_equal(np.column_stack((merged.x, merged.y)), points)
        np.testing.assert_array_equal(np.sort(merged.triangles, axis=1), np.sort(triangles, axis=1))

    def test_scientific_time_step_format(self):
        for value, label in ((.1, "10^{-1}"), (.01, "10^{-2}"), (.001, "10^{-3}"),
                             (.025, r"2.5\times10^{-2}")):
            self.assertEqual(stability.scientific_time_step(value), label)
        for value in (0., -1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                stability.scientific_time_step(value)

    def test_pressure_matrix_titles_and_colorbar_variable(self):
        fixture = ReportingTests()
        fixture.setUp()
        results = fixture.results
        for field, row in zip(results, fixture.rows, strict=True):
            field["group_id"] = stability.GROUPS[
                next(i for i, g in enumerate(ablation.GROUPS) if g["id"] == row["variant"])
            ]["id"]
        with tempfile.TemporaryDirectory() as folder, patch.object(stability, "save_figure") as save:
            stability.plot_matrix(results, Path(folder))
            figure = save.call_args.args[0]
            self.assertEqual([ax.get_title() for ax in figure.axes[:3]],
                             [r"$\Delta t=10^{-1}$", r"$\Delta t=10^{-2}$", r"$\Delta t=10^{-3}$"])
            self.assertEqual(figure.axes[-1].get_ylabel(), r"$p$")


if __name__ == "__main__":
    unittest.main()
