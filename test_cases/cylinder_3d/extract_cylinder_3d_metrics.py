import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
    parser = argparse.ArgumentParser(description="Compare DFG3D-3 force histories.")
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


def reference_path():
    local = CASE_DIR / "reference_data" / "featflow_3d_force_coefficients.csv"
    if not local.is_file():
        raise FileNotFoundError(f"Missing 3D FEATFLOW reference data: {local}")
    return local


def load_reference():
    rows = []
    with reference_path().open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    time = np.asarray([row["time"] for row in rows], dtype=float)
    drag = np.asarray([row["drag_coefficient"] for row in rows], dtype=float)
    lift = np.asarray([row["lift_coefficient"] for row in rows], dtype=float)
    return time, drag, lift


def build_series(force_rows):
    time = np.asarray([row["time"] for row in force_rows], dtype=float)
    drag = np.asarray([row["drag_coefficient"] for row in force_rows], dtype=float)
    lift = np.asarray([row["lift_coefficient"] for row in force_rows], dtype=float)
    ref_time, ref_drag, ref_lift = load_reference()
    return {
        "drag": (time, np.interp(time, ref_time, ref_drag), drag),
        "lift": (time, np.interp(time, ref_time, ref_lift), lift),
    }


def plot_comparison(series, out_dir):
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharex=True)
    x_min = None
    x_max = None
    for ax, quantity, ylabel, title in [
        (axes[0], "drag", r"$C_D$", "Drag"),
        (axes[1], "lift", r"$C_L$", "Lift"),
    ]:
        time, reference, simulated = series[quantity]
        mask = time > 0.0
        positive_time = time[mask]
        ax.plot(positive_time, simulated[mask], color=COLORS["blue"], label="RPCS-VMS")
        ax.plot(positive_time, reference[mask], "--", color=COLORS["red"], label="FEATFLOW")
        x_min = float(positive_time[0]) if x_min is None else min(x_min, float(positive_time[0]))
        x_max = float(positive_time[-1]) if x_max is None else max(x_max, float(positive_time[-1]))
        ax.set_title(title)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        if quantity == "lift":
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, -3))
            ax.yaxis.set_major_formatter(formatter)
        ax.grid(True, linestyle=":", linewidth=0.5)
    for ax in axes:
        ax.set_xlim(x_min, x_max)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir / "cylinder_3d_benchmark_comparison")
    plt.close(fig)


def extract_metrics(series):
    time_d, ref_d, sim_d = series["drag"]
    time_l, ref_l, sim_l = series["lift"]
    metrics = []
    for name, time, reference, simulated, reducer in [
        ("C_D,max", time_d, ref_d, sim_d, lambda t, y: np.max(y)),
        ("C_L,max", time_l, ref_l, sim_l, lambda t, y: np.max(y)),
        ("C_L,min", time_l, ref_l, sim_l, lambda t, y: np.min(y)),
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
    out_dir = CASE_DIR / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    force_rows = load_force_csv(args.force_csv)
    series = build_series(force_rows)
    plot_comparison(series, out_dir)
    metrics = extract_metrics(series)
    write_csv(
        out_dir / "cylinder_3d_metrics.csv",
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
