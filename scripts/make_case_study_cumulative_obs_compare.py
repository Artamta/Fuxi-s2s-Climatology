#!/usr/bin/env python3
"""Build cumulative rainfall comparison plots with available observed data.

The script reuses the existing final-paper case-study CSVs. For historical
cases it can plot the available IMD observed cumulative rainfall. For real-time
2026 cases it records which observed sources are available and still writes a
forecast/climatology comparison figure.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CASE_STUDY = Path("/home/raj.ayush/s2s/s2s_anlysis/final_paper/case-study")
DEFAULT_OUTPUT_DIR = Path("outputs/case_study_cumulative_obs_compare")
ARCO_LATEST_SEEN = "2026-06-17"

COLORS = {
    "imd_clim": "#1559a6",
    "imd_obs": "#111827",
    "fuxi": "#2ca25f",
    "ecmwf": "#ff8c1a",
    "era5": "#7c3aed",
    "text": "#1f2933",
    "muted": "#5b6472",
    "grid": "#dce3ea",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-study-dir", type=Path, default=DEFAULT_CASE_STUDY)
    parser.add_argument("--init-date", default="20260623", help="Initialization date as YYYYMMDD.")
    parser.add_argument("--current-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imerg-file", type=Path, help="Optional pre-downloaded IMERG daily NetCDF to document/use later.")
    return parser.parse_args()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#2c2f33",
            "axes.linewidth": 0.85,
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "savefig.bbox": "tight",
        }
    )


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_case_dataframe(case_dir: Path, init_date: str) -> tuple[pd.DataFrame, Path, str]:
    data_dir = case_dir / "data"
    if init_date == "20260623":
        path = first_existing(
            [
                data_dir / f"{init_date}_all_india_cumulative_timeseries_with_era5.csv",
                data_dir / f"{init_date}_all_india_cumulative_timeseries.csv",
            ]
        )
        if path is None:
            raise SystemExit(f"No 20260623 cumulative CSV found in {data_dir}")
        df = pd.read_csv(path, parse_dates=["valid_date"])
        out = pd.DataFrame(
            {
                "lead_day": df["lead_day"],
                "valid_date": df["valid_date"],
                "imd_climatology": df["imd_mean"],
                "fuxi": df["fuxi_mean"],
                "ecmwf": df["ecmwf_mean"],
            }
        )
        if "era5_mean" in df.columns:
            out["era5_climatology"] = df["era5_mean"]
        return out, path, "forecast_climatology"

    historical = first_existing(
        [
            data_dir / f"{init_date}_imd_clim_era5_gt_fuxi_ecmwf_42day_cumulative.csv",
            data_dir / f"{init_date}_imd_clim_gt_fuxi_ecmwf_42day_cumulative.csv",
        ]
    )
    if historical is None:
        raise SystemExit(f"No historical observed cumulative CSV found for {init_date} in {data_dir}")
    df = pd.read_csv(historical, parse_dates=["valid_date"])
    out = pd.DataFrame(
        {
            "lead_day": df["lead_day"],
            "valid_date": df["valid_date"],
            "imd_climatology": df["imd_clim_mean"],
            "imd_observed": df["imd_gt"],
            "fuxi": df["fuxi"],
            "ecmwf": df["ecmwf"],
        }
    )
    if "era5_clim" in df.columns:
        out["era5_climatology"] = df["era5_clim"]
    return out, historical, "historical_observed"


def add_availability(df: pd.DataFrame, init_date: str, current_date: str, imerg_file: Path | None) -> dict:
    current = pd.Timestamp(current_date)
    valid_start = pd.Timestamp(df["valid_date"].min())
    valid_end = pd.Timestamp(df["valid_date"].max())
    check_end = min(current, valid_end)
    has_imd_observed = bool("imd_observed" in df.columns and df["imd_observed"].notna().any())

    notes = []
    if has_imd_observed:
        obs_days = int(df["imd_observed"].notna().sum())
        notes.append(f"Historical IMD observed cumulative rainfall present for {obs_days} lead days.")
    else:
        notes.append("No observed cumulative rainfall column is present in the case-study CSV.")

    imerg_status = "not_provided"
    if imerg_file is not None:
        imerg_status = "exists" if imerg_file.exists() else "missing"
        notes.append(f"IMERG file argument: {imerg_file} ({imerg_status}).")
    else:
        notes.append("No local IMERG file was provided or found by this script.")

    if has_imd_observed:
        arco_status = "not_checked_historical_imd_observed_present"
        notes.append("ARCO ERA5 was not needed because historical IMD observed rainfall is already present.")
    elif pd.Timestamp(ARCO_LATEST_SEEN) >= valid_start:
        arco_status = f"potentially_available_through_{ARCO_LATEST_SEEN}"
        notes.append(f"ARCO ERA5 latest observed date seen is {ARCO_LATEST_SEEN}; some valid dates may be available.")
    else:
        arco_status = f"not_available_for_valid_window_latest_seen_{ARCO_LATEST_SEEN}"
        notes.append(
            f"ARCO ERA5 rejected 2026-06-24; latest available date seen is {ARCO_LATEST_SEEN}, "
            f"before valid start {valid_start.date()}."
        )

    return {
        "init_date": init_date,
        "valid_start": str(valid_start.date()),
        "valid_end": str(valid_end.date()),
        "current_date_requested": str(current.date()),
        "verification_window_end": str(check_end.date()),
        "historical_imd_observed_available": has_imd_observed,
        "imerg_status": imerg_status,
        "arco_era5_status": arco_status,
        "notes": notes,
    }


def plot_comparison(df: pd.DataFrame, init_date: str, availability: dict, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.2, 7.0))
    x = df["lead_day"].to_numpy()

    if "imd_climatology" in df.columns:
        ax.plot(x, df["imd_climatology"], color=COLORS["imd_clim"], lw=2.8, label="IMD climatology")
    if "era5_climatology" in df.columns:
        ax.plot(x, df["era5_climatology"], color=COLORS["era5"], lw=2.4, label="ERA5 climatology")
    ax.plot(x, df["fuxi"], color=COLORS["fuxi"], lw=3.0, label="FuXi-S2S ensemble mean")
    ax.plot(x, df["ecmwf"], color=COLORS["ecmwf"], lw=3.0, label="ECMWF-S2S ensemble mean")
    if "imd_observed" in df.columns:
        ax.plot(x, df["imd_observed"], color=COLORS["imd_obs"], lw=3.2, label=f"IMD observed rainfall ({init_date[:4]})")

    ticks = [item for item in [1, 7, 14, 21, 28, 35, 42] if item <= int(df["lead_day"].max())]
    date_lookup = {int(row.lead_day): pd.Timestamp(row.valid_date) for row in df.itertuples()}
    labels = [f"L{lead}\n{date_lookup[lead].strftime('%b %-d')}" for lead in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_xlim(0.2, max(42.8, float(df["lead_day"].max()) + 0.8))
    ymax = max(float(df[col].max()) for col in df.columns if col not in {"lead_day", "valid_date"}) * 1.12
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Lead day and valid date")
    ax.set_ylabel("Cumulative rainfall (mm)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor="#d9dee5", framealpha=0.95)

    valid_start = availability["valid_start"]
    valid_end = availability["valid_end"]
    if availability["historical_imd_observed_available"]:
        title = "Historical Cumulative Rainfall Verification"
    else:
        title = "Cumulative Rainfall Forecast and Climatology Comparison"
    fig.text(0.055, 0.965, title, fontsize=19, fontweight="bold", color=COLORS["text"])
    fig.text(0.055, 0.925, f"IC {init_date} | valid {valid_start} to {valid_end} | All-India area mean", fontsize=11.5, color=COLORS["muted"])
    if not availability["historical_imd_observed_available"]:
        fig.text(
            0.055,
            0.055,
            "Observed line not plotted: no local IMERG file was provided and ARCO ERA5 is not available for this valid window yet.",
            fontsize=9.2,
            color="#9a3412",
        )
        bottom = 0.16
    else:
        bottom = 0.13
    fig.subplots_adjust(left=0.075, right=0.97, bottom=bottom, top=0.86)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{init_date}_cumulative_rainfall_fuxi_ecmwf_available_obs.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    configure_style()
    df, source_csv, mode = load_case_dataframe(args.case_study_dir, args.init_date)
    availability = add_availability(df, args.init_date, args.current_date, args.imerg_file)
    availability["source_csv"] = str(source_csv)
    availability["mode"] = mode
    availability["created_utc"] = datetime.now(timezone.utc).isoformat()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.output_dir / f"{args.init_date}_cumulative_rainfall_available_obs.csv"
    df.to_csv(csv_out, index=False)
    json_out = args.output_dir / f"{args.init_date}_observation_availability.json"
    json_out.write_text(json.dumps(availability, indent=2) + "\n")
    fig_out = plot_comparison(df, args.init_date, availability, args.output_dir)

    print(f"source csv : {source_csv}")
    print(f"wrote csv  : {csv_out}")
    print(f"wrote json : {json_out}")
    print(f"wrote fig  : {fig_out}")
    for note in availability["notes"]:
        print(f"note       : {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
