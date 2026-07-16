#!/usr/bin/env python3
"""Build the fixed 42-day May-17 cumulative rainfall comparison."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from build_june17_climatology import DEFAULT_BBOX, parse_ints
from plot_may17_fuxi_ecmwf_arco import (
    date_range_for,
    mask_and_weights,
    read_ecmwf,
    read_fuxi,
    read_imd_climatology,
    select_latlon,
    weighted_mean,
)
from plot_one_member_india_forecast import prepare_india_geometries


DEFAULT_FUXI = Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/raw/20260517")
DEFAULT_ECMWF = Path(
    "/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/processed/"
    "ecmwf_20260517_tp_ens50_lead42_india_1p5deg_daily_mm.nc"
)
DEFAULT_ERA5_GT = Path(
    "/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/truth/"
    "arco_era5_tp_daily_20260517.nc"
)
DEFAULT_ERA5_CLIM = Path("/storage/raj.ayush/benchmark(jfm)/era5_climatology.nc")
DEFAULT_IMD_CLIM = Path(
    "/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/"
    "imd_rain_1991_2020_daily_climatology.nc"
)
DEFAULT_OUTPUT_DIR = Path("outputs/cumulative_rainfall_fixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic-date", default="20260517")
    parser.add_argument("--lead-days", type=int, default=42)
    parser.add_argument("--members", default="0:49")
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    parser.add_argument("--fuxi-raw-dir", type=Path, default=DEFAULT_FUXI)
    parser.add_argument("--ecmwf-file", type=Path, default=DEFAULT_ECMWF)
    parser.add_argument("--era5-ground-truth", type=Path, default=DEFAULT_ERA5_GT)
    parser.add_argument("--era5-climatology", type=Path, default=DEFAULT_ERA5_CLIM)
    parser.add_argument("--imd-climatology", type=Path, default=DEFAULT_IMD_CLIM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def require_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("missing required input(s): " + ", ".join(missing))


def read_partial_era5_ground_truth(
    path: Path,
    lead_days: int,
    bbox: tuple[float, float, float, float],
    india_geometries: list,
) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as ds:
        available = min(int(ds.sizes["lead_day"]), lead_days)
        da = select_latlon(ds["tp_daily"].isel(lead_day=slice(0, available)), bbox)
        lat = da.lat.values.astype("float32")
        lon = da.lon.values.astype("float32")
        _, weights = mask_and_weights(lat, lon, india_geometries)
        values = da.values.astype("float32")
        daily = np.asarray([weighted_mean(field, weights) for field in values], dtype=np.float32)
        valid_time = pd.to_datetime(da.valid_time.values)
    return daily, valid_time.to_numpy()


def read_era5_climatology(
    path: Path,
    valid_dates: pd.DatetimeIndex,
    bbox: tuple[float, float, float, float],
    india_geometries: list,
) -> np.ndarray:
    lon_min, lon_max, lat_min, lat_max = bbox
    with h5py.File(path, "r") as ds:
        if "tp" not in ds or "dayofyear" not in ds:
            raise ValueError(f"{path}: expected tp(dayofyear, latitude, longitude)")
        units = ds["tp"].attrs.get("units", b"")
        if isinstance(units, bytes):
            units = units.decode("ascii", errors="replace")
        if str(units).strip() != "m":
            raise ValueError(f"{path}: expected tp units 'm', found {units!r}")

        lat_all = np.asarray(ds["latitude"][:], dtype=np.float32)
        lon_all = np.asarray(ds["longitude"][:], dtype=np.float32)
        lat_idx = np.flatnonzero((lat_all >= lat_min) & (lat_all <= lat_max))
        lon_idx = np.flatnonzero((lon_all >= lon_min) & (lon_all <= lon_max))
        if lat_idx.size == 0 or lon_idx.size == 0:
            raise ValueError("ERA5 climatology does not intersect the requested domain")
        lat_slice = slice(int(lat_idx.min()), int(lat_idx.max()) + 1)
        lon_slice = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)
        lat = lat_all[lat_slice]
        lon = lon_all[lon_slice]
        _, weights = mask_and_weights(lat, lon, india_geometries)

        available_doys = np.asarray(ds["dayofyear"][:], dtype=int)
        doy_lookup = {int(day): idx for idx, day in enumerate(available_doys)}
        daily = []
        for date in valid_dates:
            doy = int(pd.Timestamp(date).dayofyear)
            if doy not in doy_lookup:
                raise ValueError(f"{path}: missing day of year {doy}")
            field_mm = (
                np.asarray(ds["tp"][doy_lookup[doy], lat_slice, lon_slice], dtype=np.float32)
                * np.float32(1000.0)
            )
            daily.append(weighted_mean(field_mm, weights))
    result = np.asarray(daily, dtype=np.float32)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("ERA5 climatology produced invalid India-mean daily rainfall")
    return result


def member_statistics(daily_members: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    cumulative_members = np.cumsum(daily_members, axis=1)
    return {
        f"{prefix}_daily_ens_mean_mm": daily_members.mean(axis=0),
        f"{prefix}_cumulative_p10_mm": np.percentile(cumulative_members, 10, axis=0),
        f"{prefix}_cumulative_ens_mean_mm": cumulative_members.mean(axis=0),
        f"{prefix}_cumulative_p90_mm": np.percentile(cumulative_members, 90, axis=0),
    }


def build_table(args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    bbox = tuple(float(value) for value in args.bbox)
    members = parse_ints(args.members)
    valid_dates = date_range_for(args.ic_date, args.lead_days)
    india_outline, _ = prepare_india_geometries("50m", None, None, False)

    print("reading FuXi-S2S 50-member forecast...", flush=True)
    fuxi = read_fuxi(args.fuxi_raw_dir, members, args.lead_days, bbox, india_outline)
    print("reading ECMWF-S2S 50-member forecast...", flush=True)
    ecmwf = read_ecmwf(args.ecmwf_file, args.lead_days, bbox, india_outline)
    print("reading available ERA5 ground truth...", flush=True)
    era5_gt_daily, era5_gt_dates = read_partial_era5_ground_truth(
        args.era5_ground_truth, args.lead_days, bbox, india_outline
    )
    expected_gt_dates = valid_dates[: len(era5_gt_dates)].to_numpy()
    if not np.array_equal(era5_gt_dates.astype("datetime64[D]"), expected_gt_dates.astype("datetime64[D]")):
        raise ValueError("ERA5 ground-truth valid dates do not match the forecast leads")
    print("reading ERA5 daily climatology...", flush=True)
    era5_clim_daily = read_era5_climatology(
        args.era5_climatology, valid_dates, bbox, india_outline
    )
    print("reading IMD 1991-2020 daily climatology...", flush=True)
    imd_clim_daily = read_imd_climatology(args.imd_climatology, valid_dates, india_outline)
    if imd_clim_daily is None:
        raise ValueError("IMD climatology could not be read")

    data: dict[str, object] = {
        "lead_day": np.arange(1, args.lead_days + 1),
        "valid_date": valid_dates.strftime("%Y-%m-%d"),
    }
    data.update(member_statistics(fuxi.daily_member_mean, "fuxi"))
    data.update(member_statistics(ecmwf.daily_member_mean, "ecmwf"))

    gt_daily_full = np.full(args.lead_days, np.nan, dtype=np.float32)
    gt_cumulative_full = np.full(args.lead_days, np.nan, dtype=np.float32)
    gt_daily_full[: len(era5_gt_daily)] = era5_gt_daily
    gt_cumulative_full[: len(era5_gt_daily)] = np.cumsum(era5_gt_daily)
    data["era5_gt_daily_mm"] = gt_daily_full
    data["era5_gt_cumulative_mm"] = gt_cumulative_full
    data["era5_climatology_daily_mm"] = era5_clim_daily
    data["era5_climatology_cumulative_mm"] = np.cumsum(era5_clim_daily)
    data["imd_1991_2020_climatology_daily_mm"] = imd_clim_daily
    data["imd_1991_2020_climatology_cumulative_mm"] = np.cumsum(imd_clim_daily)
    frame = pd.DataFrame(data)

    cumulative_columns = [name for name in frame if "cumulative" in name]
    for name in cumulative_columns:
        finite = frame[name].dropna().to_numpy(dtype=float)
        if finite.size and np.any(np.diff(finite) < -1e-5):
            raise ValueError(f"{name} decreases with lead time")
    if np.any(frame["fuxi_cumulative_p10_mm"] > frame["fuxi_cumulative_ens_mean_mm"]):
        raise ValueError("FuXi p10 exceeds ensemble mean")
    if np.any(frame["ecmwf_cumulative_p10_mm"] > frame["ecmwf_cumulative_ens_mean_mm"]):
        raise ValueError("ECMWF p10 exceeds ensemble mean")

    metadata = {
        "ic_date": args.ic_date,
        "valid_start": str(valid_dates[0].date()),
        "valid_end": str(valid_dates[-1].date()),
        "lead_days": args.lead_days,
        "members_per_model": len(members),
        "area_method": "cosine-latitude weighted mean over the India outline shapefile mask",
        "era5_ground_truth_available_leads": len(era5_gt_daily),
        "era5_climatology_baseline_note": (
            "The local benchmark code identifies this file as ERA5 1990-2019 day-of-year "
            "climatology, but the NetCDF itself does not record baseline years."
        ),
        "sources": {
            "fuxi": str(args.fuxi_raw_dir),
            "ecmwf": str(args.ecmwf_file),
            "era5_ground_truth": str(args.era5_ground_truth),
            "era5_climatology": str(args.era5_climatology),
            "imd_climatology": str(args.imd_climatology),
        },
    }
    return frame, metadata


def endpoint(ax: plt.Axes, x: float, y: float, text: str, color: str, offset: float = 0) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(x + 0.8, y + offset),
        color=color,
        fontsize=9.2,
        fontweight="bold",
        va="center",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color, "linewidth": 1.1},
    )


def make_plot(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    lead = frame["lead_day"].to_numpy(dtype=float)
    dates = pd.to_datetime(frame["valid_date"])
    fig, ax = plt.subplots(figsize=(15.2, 8.7))
    fig.subplots_adjust(left=0.08, right=0.84, top=0.80, bottom=0.19)

    ax.fill_between(
        lead,
        frame["fuxi_cumulative_p10_mm"],
        frame["fuxi_cumulative_p90_mm"],
        color="#2ca25f",
        alpha=0.14,
        linewidth=0,
        label="FuXi-S2S member p10-p90",
    )
    ax.fill_between(
        lead,
        frame["ecmwf_cumulative_p10_mm"],
        frame["ecmwf_cumulative_p90_mm"],
        color="#ef8a24",
        alpha=0.10,
        linewidth=0,
        label="ECMWF-S2S member p10-p90",
    )
    ax.plot(lead, frame["fuxi_cumulative_ens_mean_mm"], color="#20a464", lw=3.2, label="FuXi-S2S 50-member mean")
    ax.plot(lead, frame["ecmwf_cumulative_ens_mean_mm"], color="#e47d12", lw=3.0, label="ECMWF-S2S 50-member mean")
    ax.plot(lead, frame["era5_gt_cumulative_mm"], color="#172033", lw=3.1, label="ERA5 ground truth (available)")
    ax.plot(lead, frame["era5_climatology_cumulative_mm"], color="#7a3db8", lw=2.7, ls=(0, (7, 4)), label="ERA5 climatology")
    ax.plot(lead, frame["imd_1991_2020_climatology_cumulative_mm"], color="#1764aa", lw=2.7, ls=(0, (3, 3)), label="IMD 1991-2020 climatology")

    tick_leads = np.asarray([1, 7, 14, 21, 28, 35, 42])
    tick_dates = dates.iloc[tick_leads - 1]
    ax.set_xticks(tick_leads)
    ax.set_xticklabels([f"L{day}\n{date:%d %b}" for day, date in zip(tick_leads, tick_dates)])
    ax.set_xlim(0.2, 49)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Lead day and valid date", fontsize=11.5)
    ax.set_ylabel("Cumulative rainfall (mm)", fontsize=12)
    ax.grid(True, color="#d8e0e8", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5df", fontsize=9.5)

    last = frame.iloc[-1]
    endpoint(ax, 42, float(last["ecmwf_cumulative_ens_mean_mm"]), f"ECMWF  {last['ecmwf_cumulative_ens_mean_mm']:.0f} mm", "#e47d12", 5)
    endpoint(ax, 42, float(last["imd_1991_2020_climatology_cumulative_mm"]), f"IMD clim  {last['imd_1991_2020_climatology_cumulative_mm']:.0f} mm", "#1764aa", -4)
    endpoint(ax, 42, float(last["era5_climatology_cumulative_mm"]), f"ERA5 clim  {last['era5_climatology_cumulative_mm']:.0f} mm", "#7a3db8", 0)
    endpoint(ax, 42, float(last["fuxi_cumulative_ens_mean_mm"]), f"FuXi  {last['fuxi_cumulative_ens_mean_mm']:.0f} mm", "#20a464", 0)
    gt = frame["era5_gt_cumulative_mm"].dropna()
    endpoint(ax, float(gt.index[-1] + 1), float(gt.iloc[-1]), f"ERA5 GT L{len(gt)}  {gt.iloc[-1]:.0f} mm", "#172033", 0)

    fig.text(0.055, 0.925, "42-Day Cumulative Rainfall over India", fontsize=21, fontweight="bold", color="#1d2733")
    fig.text(
        0.055,
        0.865,
        "FuXi-S2S and ECMWF-S2S  |  IC: 17 May 2026 00 UTC  |  Valid: 18 May-28 June 2026",
        fontsize=11.5,
        color="#5b6674",
    )
    fig.text(
        0.055,
        0.055,
        "All series use a cosine-latitude weighted India-shapefile mean. Forecast rainfall is accumulated from daily mm/day values.\n"
        "ERA5 ground truth currently ends at L30. ERA5 climatology is day-of-year matched; its file omits baseline-year metadata (local benchmark code labels it 1990-2019).",
        fontsize=8.4,
        color="#5b6674",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    require_inputs(
        [args.ecmwf_file, args.era5_ground_truth, args.era5_climatology, args.imd_climatology]
    )
    frame, metadata = build_table(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.ic_date}_lead1_{args.lead_days}_fuxi_ecmwf_era5gt_climatologies_cumulative_rainfall"
    csv_path = args.output_dir / f"{stem}.csv"
    png_path = args.output_dir / f"{stem}.png"
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    frame.to_csv(csv_path, index=False)
    make_plot(frame, png_path, args.dpi)

    final = frame.iloc[-1]
    gt = frame["era5_gt_cumulative_mm"].dropna()
    metadata["final_cumulative_mm"] = {
        "fuxi_ensemble_mean_l42": float(final["fuxi_cumulative_ens_mean_mm"]),
        "ecmwf_ensemble_mean_l42": float(final["ecmwf_cumulative_ens_mean_mm"]),
        "era5_climatology_l42": float(final["era5_climatology_cumulative_mm"]),
        "imd_1991_2020_climatology_l42": float(final["imd_1991_2020_climatology_cumulative_mm"]),
        "era5_ground_truth_available_l30": float(gt.iloc[-1]),
    }
    metadata["outputs"] = {"csv": str(csv_path), "png": str(png_path), "pdf": str(png_path.with_suffix('.pdf'))}
    metadata["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")
    print(f"wrote {png_path.with_suffix('.pdf')}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
