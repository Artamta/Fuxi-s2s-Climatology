#!/usr/bin/env python3
"""Make a June-17 FuXi-S2S cumulative rainfall plot over India.

The plot is intentionally simple: FuXi ensemble cumulative rainfall, FuXi
June-17 model climatology, IMD daily rainfall climatology, and any real
observed rainfall that is locally available for the valid dates.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from shapely import contains_xy
from shapely.ops import unary_union

from build_june17_climatology import (
    DATA_VAR,
    DEFAULT_BBOX,
    DEFAULT_OUTPUT as DEFAULT_FUXI_CLIMATOLOGY,
    convert_units,
    coord_slice,
    parse_ints,
)
from make_fuxi_weekly_analysis import find_raw_dir, member_path
from plot_one_member_india_forecast import prepare_india_geometries


DEFAULT_OUTPUT_DIR = Path("outputs/june17_cumulative_rainfall")
DEFAULT_IMD_CLIMATOLOGY = Path(
    "/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/"
    "imd_rain_1991_2020_daily_climatology.nc"
)
DEFAULT_IMD_OBS_ROOT = Path("/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/netcdf")

COLORS = {
    "fuxi": "#2ca25f",
    "fuxi_member": "#006d3c",
    "fuxi_clim": "#6fbf73",
    "imd_clim": "#1559a6",
    "imd_obs": "#111827",
    "text": "#1f2933",
    "muted": "#5b6472",
    "grid": "#dce3ea",
    "warning": "#9a3412",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic-date", default="20260617", help="FuXi IC date as YYYYMMDD.")
    parser.add_argument("--raw-dir", type=Path, help="Path like .../raw/YYYYMMDD containing member/MM/SS.nc")
    parser.add_argument("--members", default="0:49")
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    parser.add_argument("--fuxi-climatology", type=Path, default=DEFAULT_FUXI_CLIMATOLOGY)
    parser.add_argument("--imd-climatology", type=Path, default=DEFAULT_IMD_CLIMATOLOGY)
    parser.add_argument("--imd-obs-root", type=Path, default=DEFAULT_IMD_OBS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--map-scale", default="50m", choices=("10m", "50m", "110m"))
    parser.add_argument("--india-shapefile", type=Path)
    parser.add_argument("--district-shapefile", type=Path)
    parser.add_argument("--current-date", default="2026-07-09")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def configure_style(dpi: int) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#2c2f33",
            "axes.linewidth": 0.85,
            "axes.labelsize": 11.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "savefig.bbox": "tight",
        }
    )


def mask_and_weights(lat: np.ndarray, lon: np.ndarray, geometries: list) -> tuple[np.ndarray, np.ndarray]:
    union = unary_union(geometries)
    lon2d, lat2d = np.meshgrid(lon, lat)
    mask = contains_xy(union, lon2d, lat2d)
    weights = np.cos(np.deg2rad(lat2d)).astype("float64")
    weights = np.where(mask, weights, 0.0)
    if not np.any(weights > 0):
        raise RuntimeError("India mask has no grid cells on this grid")
    return mask, weights


def weighted_mean(field: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(field) & (weights > 0)
    if not np.any(valid):
        return float("nan")
    return float(np.nansum(field[valid] * weights[valid]) / np.nansum(weights[valid]))


def read_raw_metadata(path: Path, bbox: tuple[float, float, float, float]):
    lon_min, lon_max, lat_min, lat_max = bbox
    with Dataset(path) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float32)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float32)
        lat_sel = coord_slice(lat, lat_min, lat_max)
        lon_sel = coord_slice(lon, lon_min, lon_max)
        channels = np.asarray(ds.variables["channel"][:]).astype(str)
        matches = np.flatnonzero(channels == "tp")
        if matches.size == 0:
            raise ValueError(f"tp not found in {path}")
        return int(matches[0]), lat[lat_sel], lon[lon_sel], lat_sel, lon_sel


def fuxi_member_daily_means(raw_dir: Path, members: list[int], bbox: tuple[float, float, float, float], weights: np.ndarray) -> np.ndarray:
    first = member_path(raw_dir, members[0], 1)
    tp_idx, _, _, lat_sel, lon_sel = read_raw_metadata(first, bbox)
    daily = np.zeros((len(members), 42), dtype=np.float32)

    for member_idx, member in enumerate(members):
        for lead_day in range(1, 43):
            path = member_path(raw_dir, member, lead_day)
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            with Dataset(path) as ds:
                arr = np.asarray(ds.variables[DATA_VAR][0, 0, tp_idx, lat_sel, lon_sel], dtype=np.float32)
            arr = convert_units("tp", arr)
            daily[member_idx, lead_day - 1] = weighted_mean(arr, weights)
        if member_idx == 0 or (member_idx + 1) % 10 == 0 or member_idx + 1 == len(members):
            print(f"processed FuXi member {member_idx + 1}/{len(members)}", flush=True)
    return daily


def fuxi_climatology_daily(path: Path, weights: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    try:
        if "tp" not in [str(item) for item in ds.variable.values]:
            return None
        arr = ds["daily_mean"].sel(variable="tp").values.astype("float32")
        return np.asarray([weighted_mean(arr[idx], weights) for idx in range(arr.shape[0])], dtype=np.float32)
    finally:
        ds.close()


def month_day_key(date: pd.Timestamp) -> str:
    return f"{date.month:02d}-{date.day:02d}"


def imd_climatology_daily(path: Path, valid_dates: pd.DatetimeIndex, india_geoms: list) -> np.ndarray | None:
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    try:
        lat = ds.lat.values.astype("float32")
        lon = ds.lon.values.astype("float32")
        _, weights = mask_and_weights(lat, lon, india_geoms)
        month_days = [str(item) for item in ds["month_day"].values.astype(str)]
        lookup = {item: idx for idx, item in enumerate(month_days)}
        out = []
        for date in valid_dates:
            key = month_day_key(pd.Timestamp(date))
            if key == "02-29":
                out.append(np.nan)
                continue
            idx = lookup[key]
            out.append(weighted_mean(ds["rain_mean"].isel(day=idx).values.astype("float32"), weights))
        return np.asarray(out, dtype=np.float32)
    finally:
        ds.close()


def imd_observed_daily(obs_root: Path, valid_dates: pd.DatetimeIndex, current_date: pd.Timestamp, india_geoms: list) -> tuple[np.ndarray, str]:
    years = sorted(set(pd.Timestamp(date).year for date in valid_dates if pd.Timestamp(date) <= current_date))
    if not years:
        return np.full((len(valid_dates),), np.nan, dtype=np.float32), "no_valid_dates_reached"

    out = np.full((len(valid_dates),), np.nan, dtype=np.float32)
    status_items = []
    for year in years:
        path = obs_root / f"imd_rain_{year}.nc"
        if not path.exists():
            status_items.append(f"{year}:missing")
            continue
        ds = xr.open_dataset(path)
        try:
            lat = ds.lat.values.astype("float32")
            lon = ds.lon.values.astype("float32")
            _, weights = mask_and_weights(lat, lon, india_geoms)
            times = pd.to_datetime(ds.time.values)
            lookup = {pd.Timestamp(item).normalize(): idx for idx, item in enumerate(times)}
            used = 0
            for out_idx, date in enumerate(valid_dates):
                date = pd.Timestamp(date).normalize()
                if date.year != year or date > current_date:
                    continue
                if date in lookup:
                    out[out_idx] = weighted_mean(ds["rain"].isel(time=lookup[date]).values.astype("float32"), weights)
                    used += 1
            status_items.append(f"{year}:used_{used}")
        finally:
            ds.close()
    return out, ",".join(status_items)


def make_dataframe(
    ic_date: str,
    members: list[int],
    raw_dir: Path,
    bbox: tuple[float, float, float, float],
    india_geoms: list,
    fuxi_clim_path: Path,
    imd_clim_path: Path,
    imd_obs_root: Path,
    current_date: str,
) -> tuple[pd.DataFrame, dict]:
    init = pd.Timestamp(datetime.strptime(ic_date, "%Y%m%d"))
    valid_dates = pd.date_range(init + timedelta(days=1), periods=42, freq="D")

    tp_idx, lat, lon, _, _ = read_raw_metadata(member_path(raw_dir, members[0], 1), bbox)
    del tp_idx
    _, weights = mask_and_weights(lat, lon, india_geoms)
    member_daily = fuxi_member_daily_means(raw_dir, members, bbox, weights)
    member_cum = np.cumsum(member_daily, axis=1)
    fuxi_clim_daily = fuxi_climatology_daily(fuxi_clim_path, weights)
    imd_clim_daily = imd_climatology_daily(imd_clim_path, valid_dates, india_geoms)
    imd_obs_daily, imd_obs_status = imd_observed_daily(imd_obs_root, valid_dates, pd.Timestamp(current_date), india_geoms)

    df = pd.DataFrame(
        {
            "lead_day": np.arange(1, 43, dtype=np.int32),
            "valid_date": valid_dates,
            "fuxi_mean_daily_mm": member_daily.mean(axis=0),
            "fuxi_p10_cumulative_mm": np.percentile(member_cum, 10, axis=0),
            "fuxi_mean_cumulative_mm": member_cum.mean(axis=0),
            "fuxi_p90_cumulative_mm": np.percentile(member_cum, 90, axis=0),
            "fuxi_member00_cumulative_mm": member_cum[0],
        }
    )
    if fuxi_clim_daily is not None:
        df["fuxi_model_climatology_cumulative_mm"] = np.cumsum(fuxi_clim_daily)
    if imd_clim_daily is not None:
        df["imd_1991_2020_climatology_cumulative_mm"] = np.cumsum(imd_clim_daily)
    if np.isfinite(imd_obs_daily).any():
        df["imd_observed_cumulative_mm"] = np.where(np.isfinite(imd_obs_daily), np.cumsum(np.nan_to_num(imd_obs_daily, nan=0.0)), np.nan)
        first_nan = np.flatnonzero(~np.isfinite(imd_obs_daily))
        if first_nan.size:
            df.loc[first_nan[0] :, "imd_observed_cumulative_mm"] = np.nan

    availability = {
        "ic_date": ic_date,
        "valid_start": str(valid_dates[0].date()),
        "valid_end": str(valid_dates[-1].date()),
        "current_date": str(pd.Timestamp(current_date).date()),
        "forecast_raw_dir": str(raw_dir),
        "forecast_members": ",".join(f"{item:02d}" for item in members),
        "fuxi_samples": f"{len(members)} members x 42 lead days",
        "fuxi_tp_units": "mm/day (raw FuXi tp clipped at zero and multiplied by 24)",
        "imd_observed_status": imd_obs_status,
        "imd_observed_days_plotted": int(np.isfinite(imd_obs_daily).sum()),
        "imd_climatology_file": str(imd_clim_path) if imd_clim_path.exists() else "missing",
        "fuxi_climatology_file": str(fuxi_clim_path) if fuxi_clim_path.exists() else "missing",
        "imerg_status": "not_found_locally_or_not_provided",
        "era5_status": "not_available_for_valid_window_in_local_files; ARCO was previously seen available only through 2026-06-17",
        "gfs_status": "not_found_locally",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    return df, availability


def plot_dataframe(df: pd.DataFrame, availability: dict, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.4, 7.2))
    x = df["lead_day"].to_numpy()

    ax.fill_between(x, df["fuxi_p10_cumulative_mm"], df["fuxi_p90_cumulative_mm"], color=COLORS["fuxi"], alpha=0.14, linewidth=0)
    ax.plot(x, df["fuxi_mean_cumulative_mm"], color=COLORS["fuxi"], lw=3.1, label="FuXi-S2S ensemble mean")
    ax.plot(x, df["fuxi_member00_cumulative_mm"], color=COLORS["fuxi_member"], lw=2.1, ls=(0, (6, 4)), label="FuXi member 00")
    if "fuxi_model_climatology_cumulative_mm" in df:
        ax.plot(x, df["fuxi_model_climatology_cumulative_mm"], color=COLORS["fuxi_clim"], lw=2.7, ls="--", label="FuXi June-17 model climatology")
    if "imd_1991_2020_climatology_cumulative_mm" in df:
        ax.plot(x, df["imd_1991_2020_climatology_cumulative_mm"], color=COLORS["imd_clim"], lw=2.8, label="IMD 1991-2020 climatology")
    if "imd_observed_cumulative_mm" in df:
        ax.plot(x, df["imd_observed_cumulative_mm"], color=COLORS["imd_obs"], lw=3.2, label="IMD observed rainfall")

    ticks = [1, 7, 14, 21, 28, 35, 42]
    date_lookup = {int(row.lead_day): pd.Timestamp(row.valid_date) for row in df.itertuples()}
    ax.set_xticks(ticks, [f"L{lead}\n{date_lookup[lead].strftime('%b %-d')}" for lead in ticks])
    ymax_cols = [col for col in df.columns if col.endswith("_cumulative_mm")]
    ax.set_ylim(0, max(float(df[col].max(skipna=True)) for col in ymax_cols) * 1.12)
    ax.set_xlim(0.2, 42.8)
    ax.set_xlabel("Lead day and valid date")
    ax.set_ylabel("Cumulative rainfall (mm)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor="#d9dee5", framealpha=0.95)

    fig.text(0.055, 0.965, "June-17 IC Cumulative Rainfall over India", fontsize=20, fontweight="bold", color=COLORS["text"])
    fig.text(
        0.055,
        0.925,
        f"IC {availability['ic_date']} | valid {availability['valid_start']} to {availability['valid_end']} | India area mean",
        fontsize=11.5,
        color=COLORS["muted"],
    )
    if availability["imd_observed_days_plotted"] == 0:
        fig.text(
            0.055,
            0.055,
            "No real observed line plotted: local IMD observed rainfall is unavailable for 2026, no local IMERG was found, and local/ARCO ERA5 does not cover this valid window yet.",
            fontsize=9.1,
            color=COLORS["warning"],
        )
        bottom = 0.16
    else:
        fig.text(
            0.055,
            0.055,
            f"Observed line is partial: {availability['imd_observed_days_plotted']} valid day(s) available through {availability['current_date']}.",
            fontsize=9.1,
            color=COLORS["muted"],
        )
        bottom = 0.16
    fig.subplots_adjust(left=0.075, right=0.97, bottom=bottom, top=0.86)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def main() -> int:
    args = parse_args()
    configure_style(args.dpi)
    raw_dir = find_raw_dir(args.ic_date, args.raw_dir)
    members = parse_ints(args.members)
    india_outline_geoms, _ = prepare_india_geometries(args.map_scale, args.india_shapefile, args.district_shapefile, False)

    df, availability = make_dataframe(
        ic_date=args.ic_date,
        members=members,
        raw_dir=raw_dir,
        bbox=tuple(args.bbox),
        india_geoms=india_outline_geoms,
        fuxi_clim_path=args.fuxi_climatology,
        imd_clim_path=args.imd_climatology,
        imd_obs_root=args.imd_obs_root,
        current_date=args.current_date,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.output_dir / f"{args.ic_date}_india_cumulative_rainfall.csv"
    json_out = args.output_dir / f"{args.ic_date}_truth_availability.json"
    fig_out = args.output_dir / f"{args.ic_date}_india_cumulative_rainfall_fuxi_available_truth.png"
    df.to_csv(csv_out, index=False)
    json_out.write_text(json.dumps(availability, indent=2) + "\n")
    plot_dataframe(df, availability, fig_out)

    print(f"raw dir    : {raw_dir}")
    print(f"wrote csv : {csv_out}")
    print(f"wrote json: {json_out}")
    print(f"wrote fig : {fig_out}")
    print(f"obs days  : {availability['imd_observed_days_plotted']}")
    print(f"obs status: {availability['imd_observed_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
