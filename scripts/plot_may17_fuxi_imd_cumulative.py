#!/usr/bin/env python3
"""Plot the saved May-17 FuXi cumulative rainfall against observations and IMD climatology."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CSV = Path(
    "outputs/may17_fuxi_ecmwf_arco/"
    "20260517_lead1_30_cumulative_timeseries.csv"
)
DEFAULT_OUTPUT = Path(
    "outputs/cumulative_rainfall_fixed/"
    "20260517_lead1_30_fuxi_vs_era5gt_imd_climatology_cumulative_rainfall.png"
)

REQUIRED_COLUMNS = {
    "lead_day",
    "valid_date",
    "arco_cumulative_mm",
    "fuxi_cumulative_p10_mm",
    "fuxi_cumulative_ens_mean_mm",
    "fuxi_cumulative_p90_mm",
    "imd_1991_2020_climatology_cumulative_mm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["valid_date"])
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path} contains no rows")
    expected_leads = np.arange(1, len(frame) + 1)
    if not np.array_equal(frame["lead_day"].to_numpy(), expected_leads):
        raise ValueError("lead days must be consecutive and start at one")
    cumulative_columns = [
        "arco_cumulative_mm",
        "fuxi_cumulative_p10_mm",
        "fuxi_cumulative_ens_mean_mm",
        "fuxi_cumulative_p90_mm",
        "imd_1991_2020_climatology_cumulative_mm",
    ]
    values = frame[cumulative_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("cumulative rainfall contains missing or non-finite values")
    if np.any(np.diff(values, axis=0) < -1e-5):
        raise ValueError("cumulative rainfall must not decrease with lead time")
    if np.any(frame["fuxi_cumulative_p10_mm"] > frame["fuxi_cumulative_ens_mean_mm"]):
        raise ValueError("FuXi p10 exceeds its ensemble mean")
    if np.any(frame["fuxi_cumulative_ens_mean_mm"] > frame["fuxi_cumulative_p90_mm"]):
        raise ValueError("FuXi ensemble mean exceeds p90")
    return frame


def end_label(ax, x: float, y: float, text: str, color: str, offset: float = 0.0) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(18, offset),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=color,
        bbox={"boxstyle": "round,pad=0.22", "fc": "white", "ec": color, "lw": 1.1},
        clip_on=False,
    )


def make_plot(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    lead = frame["lead_day"].to_numpy(dtype=int)
    dates = frame["valid_date"]
    fuxi = frame["fuxi_cumulative_ens_mean_mm"].to_numpy(dtype=float)
    fuxi_p10 = frame["fuxi_cumulative_p10_mm"].to_numpy(dtype=float)
    fuxi_p90 = frame["fuxi_cumulative_p90_mm"].to_numpy(dtype=float)
    era5 = frame["arco_cumulative_mm"].to_numpy(dtype=float)
    imd = frame["imd_1991_2020_climatology_cumulative_mm"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12.6, 7.0), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.83, top=0.82, bottom=0.18)

    ax.fill_between(
        lead,
        fuxi_p10,
        fuxi_p90,
        color="#2ca25f",
        alpha=0.16,
        linewidth=0,
        label="FuXi-S2S member p10-p90",
    )
    ax.plot(lead, fuxi, color="#20a464", lw=3.0, label="FuXi-S2S 50-member mean")
    ax.plot(lead, era5, color="#172033", lw=2.8, label="ERA5 ground truth")
    ax.plot(
        lead,
        imd,
        color="#1764aa",
        lw=2.8,
        linestyle=(0, (7, 4)),
        label="IMD 1991-2020 climatology",
    )

    tick_leads = np.asarray([1, 7, 14, 21, 28, len(frame)], dtype=int)
    tick_leads = np.unique(tick_leads[tick_leads <= len(frame)])
    tick_labels = [
        f"L{value}\n{dates.iloc[value - 1]:%d %b}" for value in tick_leads
    ]
    ax.set_xticks(tick_leads, tick_labels)
    ax.set_xlim(0.2, len(frame) + 2.4)
    ymax = max(float(fuxi_p90[-1]), float(era5[-1]), float(imd[-1]))
    ax.set_ylim(0, max(110.0, ymax * 1.15))
    ax.set_xlabel("Lead day and valid date", fontsize=11)
    ax.set_ylabel("Cumulative rainfall (mm)", fontsize=11)
    ax.grid(True, color="#d8e0e8", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9.5)

    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="#ccd4dd",
        fontsize=9.5,
    )

    end_label(ax, lead[-1], imd[-1], f"IMD climatology  {imd[-1]:.0f} mm", "#1764aa", 10)
    end_label(ax, lead[-1], era5[-1], f"ERA5 GT  {era5[-1]:.0f} mm", "#172033", 0)
    end_label(ax, lead[-1], fuxi[-1], f"FuXi  {fuxi[-1]:.0f} mm", "#20a464", 0)

    fig.text(
        0.055,
        0.94,
        "FuXi-S2S 30-Day Cumulative Rainfall over India",
        fontsize=21,
        fontweight="bold",
        color="#1d2733",
        ha="left",
    )
    fig.text(
        0.055,
        0.885,
        "IC: 17 May 2026 00 UTC  |  Valid: 18 May-16 June 2026  |  Area-weighted India mean",
        fontsize=11.5,
        color="#5b6674",
        ha="left",
    )
    fig.text(
        0.055,
        0.055,
        "FuXi daily rainfall is the saved 24-hour model-step mean rate converted to mm/day (TP x 24). "
        "Shading is the cumulative member p10-p90 range.\n"
        "ERA5 provides ground truth; IMD is the lead-date-matched 1991-2020 daily climatology.",
        fontsize=8.4,
        color="#5b6674",
        ha="left",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    frame = load_data(args.csv)
    make_plot(frame, args.output, args.dpi)
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
