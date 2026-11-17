import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#1f77b4",
    "orange": "#d95f02",
    "green": "#1b9e77",
    "purple": "#7570b3",
    "red": "#d62728",
    "gray": "#4b5563",
}


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Charis SIL", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "figure.titlesize": 10,
            "figure.constrained_layout.use": False,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, local_stem, png=True):
    local_stem = Path(local_stem)
    local_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = local_stem.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    outputs = [pdf_path]
    if png:
        png_path = local_stem.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight")
        outputs.append(png_path)
    return outputs


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scientific_tex(value):
    coefficient, exponent = f"{float(value):.4e}".split("e")
    return rf"${coefficient}\times10^{{{int(exponent)}}}$"


def scientific_time_step(dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("The time step must be finite and positive.")
    coefficient, exponent = np.format_float_scientific(dt, unique=True, trim="-").split("e")
    power = rf"10^{{{int(exponent)}}}"
    return power if coefficient == "1" else rf"{coefficient}\times{power}"


def rate(previous_error, current_error, previous_h, current_h):
    if previous_error <= 0 or current_error <= 0:
        return math.nan
    return math.log(previous_error / current_error) / math.log(previous_h / current_h)


def finite_range(values, padding=0.05):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0, 1.0
    lo = float(array.min())
    hi = float(array.max())
    if lo == hi:
        delta = abs(lo) * padding if lo else padding
        return lo - delta, hi + delta
    delta = (hi - lo) * padding
    return lo - delta, hi + delta
