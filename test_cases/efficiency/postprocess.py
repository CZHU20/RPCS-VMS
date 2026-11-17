import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METHODS = ("monolithic-SUPG/PSPG", "RPCS-VMS")
DISPLAY_LABELS = {
    "monolithic-SUPG/PSPG": "Monolithic SUPG/PSPG+GD",
    "RPCS-VMS": "RPCS-VMS",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate independent efficiency runs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--input-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def convert(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def aggregate_summary(rows):
    result = {}
    for key in rows[0]:
        values = [convert(row[key]) for row in rows]
        if all(isinstance(value, (int, float)) for value in values):
            result[key] = float(np.mean(values))
        else:
            result[key] = values[0] if len(set(values)) == 1 else "varies"
    result["run_id"] = "single" if len(rows) == 1 else "mean"
    result["repeats"] = len(rows)
    result["solve_wall_time_min"] = min(float(row["solve_wall_time"]) for row in rows)
    result["solve_wall_time_max"] = max(float(row["solve_wall_time"]) for row in rows)
    return result


def write_report(path, summaries):
    mono_wall = next(
        row["solve_wall_time"] for row in summaries if row["method"] == METHODS[0]
    )
    mpi_ranks = int(summaries[0]["mpi_ranks"])
    repeats = int(summaries[0]["repeats"])
    run_description = (
        "a single independent run"
        if repeats == 1
        else f"{repeats} independent runs"
    )
    lines = [
        f"# {mpi_ranks}-rank DFG3D-3 efficiency comparison",
        "",
        f"Each method was run in a fresh MPI job. Values are arithmetic means over {run_description}. "
        "Final observables were evaluated outside timing.",
        "",
        "| Method | DOFs | setup wall (s) | solve wall (s) | total wall (s) | "
        "solve range (s) | mean step (s) | Krylov it/step | speedup | "
        "final Cd | final Cl | final dP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        speedup = mono_wall / row["solve_wall_time"]
        lines.append(
            f"| {DISPLAY_LABELS[row['method']]} | {int(row['total_dofs'])} | "
            f"{row['setup_wall_time']:.3f} | "
            f"{row['solve_wall_time']:.3f} | "
            f"{row['total_wall_time']:.3f} | "
            f"{row['solve_wall_time_min']:.3f}--{row['solve_wall_time_max']:.3f} | "
            f"{row['mean_step_wall_time']:.6f} | "
            f"{row['mean_linear_iterations_per_step']:.3f} | {speedup:.3f} | "
            f"{row['final_drag_coefficient']:.8f} | "
            f"{row['final_lift_coefficient']:.8f} | "
            f"{row['final_pressure_difference']:.8f} |"
        )
    monolithic = summaries[0]
    vms = summaries[1]
    lines.extend(["", "Final RPCS-VMS differences from monolithic SUPG/PSPG+GD:"])
    for label, key in (
        ("Cd", "final_drag_coefficient"),
        ("Cl", "final_lift_coefficient"),
        ("dP", "final_pressure_difference"),
    ):
        scale = max(abs(monolithic[key]), abs(vms[key]), 1.0e-12)
        relative_difference = abs(vms[key] - monolithic[key]) / scale
        lines.append(f"- {label}: {relative_difference:.6%}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main():
    args = parse_args()
    summary_paths = sorted(args.input_dir.glob(f"{args.input_prefix}_*_summary.csv"))
    step_paths = sorted(args.input_dir.glob(f"{args.input_prefix}_*_steps.csv"))
    metadata_paths = sorted(args.input_dir.glob(f"{args.input_prefix}_*_metadata.json"))
    if not summary_paths:
        raise FileNotFoundError("No efficiency run summaries found.")

    run_summaries = []
    for path in summary_paths:
        run_summaries.extend(read_csv(path))
    grouped = defaultdict(list)
    for row in run_summaries:
        grouped[row["method"]].append(row)
    missing = [method for method in METHODS if method not in grouped]
    if missing:
        raise RuntimeError(f"Missing methods: {', '.join(missing)}")

    summaries = [aggregate_summary(grouped[method]) for method in METHODS]
    mono_wall = summaries[0]["solve_wall_time"]
    for row in summaries:
        row["speedup_vs_monolithic"] = mono_wall / row["solve_wall_time"]

    all_steps = []
    for path in step_paths:
        all_steps.extend(read_csv(path))
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]

    output_base = args.input_dir / args.output_prefix
    write_csv(output_base.with_name(f"{args.output_prefix}_runs.csv"), run_summaries)
    write_csv(output_base.with_name(f"{args.output_prefix}_summary.csv"), summaries)
    write_csv(output_base.with_name(f"{args.output_prefix}_steps.csv"), all_steps)
    write_report(output_base.with_name(f"{args.output_prefix}_report.md"), summaries)
    output_base.with_name(f"{args.output_prefix}_metadata.json").write_text(
        json.dumps({"independent_runs": metadata}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
