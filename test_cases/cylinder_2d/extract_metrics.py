import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
CASE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from solver.reporting import (
    COLORS,
    configure_matplotlib,
    save_figure,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare DFG2D-3 force histories.")
    parser.add_argument(
        "--force-csv",
        type=Path,
        default=CASE_DIR / "output" / "force_coefficients.csv",
    )
    return parser.parse_args()


def load_force_csv(path):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    return rows


def load_reference_quantity(quantity):
    reference_dir = CASE_DIR / "reference_data"
    reference_dir.mkdir(exist_ok=True)
    local_path = reference_dir / f"dfg2d3_{quantity}.csv"
    if not local_path.exists():
        raise FileNotFoundError(f"Missing FEATFLOW reference data: {local_path}")

    data = []
    with local_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            data.append((float(row["time"]), float(row["reference"])))
    array = np.asarray(data, dtype=float)
    return array[:, 0], array[:, 1]


def comparison_series(force_rows, quantity):
    time = np.asarray([row["time"] for row in force_rows], dtype=float)
    if quantity == "drag":
        simulated = np.asarray([row["drag_coefficient"] for row in force_rows], dtype=float)
    elif quantity == "lift":
        simulated = np.asarray([row["lift_coefficient"] for row in force_rows], dtype=float)
    elif quantity == "pressure_difference":
        simulated = np.asarray([row["pressure_difference"] for row in force_rows], dtype=float)
    else:
        raise ValueError(quantity)
    ref_time, ref_value = load_reference_quantity(quantity)
    reference = np.interp(time, ref_time, ref_value)
    return time, reference, simulated


def write_comparison_csvs(force_rows, out_dir):
    outputs = {}
    for quantity in ("drag", "lift", "pressure_difference"):
        time, reference, simulated = comparison_series(force_rows, quantity)
        rows = [
            {
                "time": t,
                "reference": ref,
                "simulated": sim,
                "abs_error": abs(sim - ref),
            }
            for t, ref, sim in zip(time, reference, simulated)
        ]
        path = out_dir / f"{quantity}_comparison.csv"
        write_csv(path, rows, ["time", "reference", "simulated", "abs_error"])
        outputs[quantity] = (time, reference, simulated)
    return outputs


def plot_comparisons(series, out_dir):
    configure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), sharex=True)
    labels = {
        "drag": (r"$C_D$", "Drag"),
        "lift": (r"$C_L$", "Lift"),
        "pressure_difference": (r"$\Delta p$", "Pressure difference"),
    }
    x_min = None
    x_max = None
    for ax, quantity in zip(axes, ("drag", "lift", "pressure_difference")):
        time, reference, simulated = series[quantity]
        mask = time > 0.0
        positive_time = time[mask]
        ax.plot(positive_time, simulated[mask], color=COLORS["blue"], label="RPCS-VMS")
        ax.plot(positive_time, reference[mask], "--", color=COLORS["red"], label="FEATFLOW")
        x_min = float(positive_time[0]) if x_min is None else min(x_min, float(positive_time[0]))
        x_max = float(positive_time[-1]) if x_max is None else max(x_max, float(positive_time[-1]))
        ax.set_title(labels[quantity][1])
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(labels[quantity][0])
        ax.grid(True, linestyle=":", linewidth=0.5)
    for ax in axes:
        ax.set_xlim(x_min, x_max)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir / "cylinder_2d_benchmark_comparison")
    plt.close(fig)


def extract_metrics(series):
    time_d, ref_d, sim_d = series["drag"]
    time_l, ref_l, sim_l = series["lift"]
    time_p, ref_p, sim_p = series["pressure_difference"]
    metrics = []
    for name, time, reference, simulated, reducer in [
        ("t(C_D,max)", time_d, ref_d, sim_d, lambda t, y: t[np.argmax(y)]),
        ("C_D,max", time_d, ref_d, sim_d, lambda t, y: np.max(y)),
        ("t(C_L,max)", time_l, ref_l, sim_l, lambda t, y: t[np.argmax(y)]),
        ("C_L,max", time_l, ref_l, sim_l, lambda t, y: np.max(y)),
        ("Delta p(t=8)", time_p, ref_p, sim_p, lambda t, y: y[np.argmin(np.abs(t - 8.0))]),
    ]:
        ref_value = float(reducer(time, reference))
        sim_value = float(reducer(time, simulated))
        metrics.append(
            {
                "metric": name,
                "reference": ref_value,
                "present": sim_value,
                "abs_error": abs(sim_value - ref_value),
                "rel_error_percent": abs(sim_value - ref_value) / abs(ref_value) * 100
                if ref_value
                else 0.0,
            }
        )
    return metrics


def main():
    args = parse_args()
    if not args.force_csv.exists():
        raise FileNotFoundError(f"Force history not found: {args.force_csv}")
    force_rows = load_force_csv(args.force_csv)
    out_dir = CASE_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    series = write_comparison_csvs(force_rows, out_dir)
    plot_comparisons(series, out_dir)
    metrics = extract_metrics(series)
    write_csv(
        out_dir / "cylinder_2d_metrics.csv",
        metrics,
        ["metric", "reference", "present", "abs_error", "rel_error_percent"],
    )
    for row in metrics:
        print(
            f"{row['metric']}: ref={row['reference']:.6g}, "
            f"present={row['present']:.6g}, rel={row['rel_error_percent']:.3g}%"
        )


if __name__ == "__main__":
    main()
