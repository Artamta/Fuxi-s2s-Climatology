#!/usr/bin/env python3
"""Plot observed weekly wind and rainfall anomalies for the June-17 case."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from numcodecs import get_codec

from plot_one_member_india_forecast import prepare_india_geometries
from plot_weekly_wind_rainfall import draw_rain, draw_wind, save_both


DEFAULT_WIND = Path(
    "/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/era5_gt/"
    "era5t_wind850_daily_20260618_20260708.nc"
)
DEFAULT_WIND_CLIM = Path(
    "/storage/bedartha/public/datasets/as_downloaded/weatherbench2/"
    "era5-daily-climatology/"
    "1990-2017-daily_clim_daily_mean_61_dw_240x121_"
    "equiangular_with_poles_conservative.zarr"
)
DEFAULT_RAIN = Path(
    "/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/imerg_late/processed/"
    "june17_imerg_anomaly_imerg_final_1p5deg_partial_week4.nc"
)
DEFAULT_OUTPUT = Path("outputs/wind_rainfall_ground_truth_20260617")
WEEKS = (
    (pd.Timestamp("2026-06-18"), pd.Timestamp("2026-06-24")),
    (pd.Timestamp("2026-06-25"), pd.Timestamp("2026-07-01")),
    (pd.Timestamp("2026-07-02"), pd.Timestamp("2026-07-08")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind", type=Path, default=DEFAULT_WIND)
    parser.add_argument("--wind-climatology", type=Path, default=DEFAULT_WIND_CLIM)
    parser.add_argument("--rainfall", type=Path, default=DEFAULT_RAIN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def zarr_metadata(root: Path, name: str) -> tuple[dict, dict]:
    metadata = json.loads((root / ".zmetadata").read_text(encoding="utf-8"))["metadata"]
    return metadata[f"{name}/.zarray"], metadata.get(f"{name}/.zattrs", {})


def decode_chunk(root: Path, name: str, key: str, metadata: dict) -> np.ndarray:
    payload = (root / name / key).read_bytes()
    compressor = metadata.get("compressor")
    if compressor:
        payload = get_codec(compressor).decode(payload)
    values = np.frombuffer(payload, dtype=np.dtype(metadata["dtype"]))
    return values.reshape(tuple(metadata["chunks"]), order=metadata.get("order", "C"))


def coordinate(root: Path, name: str) -> np.ndarray:
    metadata, _ = zarr_metadata(root, name)
    return decode_chunk(root, name, "0", metadata).reshape(-1)[: metadata["shape"][0]]


def wind_climatology(
    root: Path, dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dayofyear = coordinate(root, "dayofyear").astype(int)
    levels = coordinate(root, "level").astype(int)
    lat = coordinate(root, "latitude").astype(np.float32)
    lon = coordinate(root, "longitude").astype(np.float32)
    level_matches = np.flatnonzero(levels == 850)
    if level_matches.size != 1:
        raise ValueError("ERA5 climatology does not contain exactly one 850-hPa level")
    level_index = int(level_matches[0])
    doy_lookup = {int(day): index for index, day in enumerate(dayofyear)}

    components = []
    for name in ("u_component_of_wind", "v_component_of_wind"):
        metadata, attrs = zarr_metadata(root, name)
        if attrs.get("units") != "m s**-1":
            raise ValueError(f"{name}: unexpected units {attrs.get('units')!r}")
        chunk_days = int(metadata["chunks"][0])
        cache = {}
        fields = []
        for date in dates:
            doy = int(date.dayofyear)
            array_index = doy_lookup[doy]
            day_chunk, within = divmod(array_index, chunk_days)
            if day_chunk not in cache:
                cache[day_chunk] = decode_chunk(
                    root, name, f"{day_chunk}.{level_index}.0.0", metadata
                )
            # Stored dimensions are day, level, longitude, latitude.
            fields.append(cache[day_chunk][within, 0].T)
        components.append(np.asarray(fields, dtype=np.float32))
    return lat, lon, components[0], components[1]


def weekly_mean(data: xr.DataArray) -> np.ndarray:
    values = []
    for start, end in WEEKS:
        selected = data.sel(time=slice(start, end))
        if selected.sizes.get("time") != 7:
            raise ValueError(f"{data.name}: {start:%Y-%m-%d} week is incomplete")
        values.append(selected.mean("time", skipna=True).values)
    return np.asarray(values, dtype=np.float32)


def labels() -> list[str]:
    return [f"(Week{index}: {start:%d%b}-{end:%d%b})" for index, (start, end) in enumerate(WEEKS, 1)]


def plot_wind(
    lat: np.ndarray,
    lon: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    outline: list,
    states: list,
    output: Path,
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.9), subplot_kw={"projection": ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.045, right=0.985, top=0.78, bottom=0.13, wspace=0.13)
    fig.text(0.04, 0.945, "Observed Weekly 850-hPa Wind Anomaly", color="#d92525", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.885, "ERA5/ERA5T daily mean minus ERA5 1990-2017 daily climatology", color="#0737c8", fontsize=10, fontweight="bold", ha="center")
    for week, ax in enumerate(axes):
        draw_wind(ax, lon, lat, u[week], v[week], outline, states, compact=True)
        ax.set_title(labels()[week], color="#0a41ff", fontsize=9, fontweight="bold")
    fig.text(0.5, 0.035, "Arrows are component-wise anomalies: (u850 observed - climatology, v850 observed - climatology), m/s.", ha="center", fontsize=8)
    return save_both(fig, output / "era5_wind850_anomaly_20260617_3week.png", dpi)


def plot_rain(
    lat: np.ndarray,
    lon: np.ndarray,
    rain: np.ndarray,
    outline: list,
    states: list,
    output: Path,
    dpi: int,
) -> list[Path]:
    fig = plt.figure(figsize=(13.4, 5.35), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.07], left=0.045, right=0.985, top=0.78, bottom=0.13, wspace=0.13, hspace=0.22)
    fig.text(0.04, 0.945, "Observed Weekly Rainfall Anomaly (mm/day)", color="#d92525", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.885, "IMERG Late observed rainfall minus IMERG Final 2001-2025 daily climatology", color="#0737c8", fontsize=10, fontweight="bold", ha="center")
    mappable = None
    for week in range(3):
        ax = fig.add_subplot(grid[0, week], projection=ccrs.PlateCarree())
        mappable = draw_rain(ax, lon, lat, rain[week], outline, states, compact=True)
        ax.set_title(labels()[week], color="#0a41ff", fontsize=9, fontweight="bold")
    cax = fig.add_subplot(grid[1, :])
    colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=[-20, -15, -10, -5, -2, 2, 5, 10, 15, 20], extend="both")
    colorbar.ax.tick_params(labelsize=8)
    return save_both(fig, output / "imerg_rainfall_anomaly_20260617_3week.png", dpi)


def plot_combined(
    wind_lat: np.ndarray,
    wind_lon: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    rain_lat: np.ndarray,
    rain_lon: np.ndarray,
    rain: np.ndarray,
    outline: list,
    states: list,
    output: Path,
    dpi: int,
) -> list[Path]:
    fig = plt.figure(figsize=(15.8, 8.0), facecolor="white")
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.07], left=0.045, right=0.985, top=0.80, bottom=0.11, hspace=0.25, wspace=0.13)
    fig.text(0.035, 0.955, "Observed Wind-Rainfall Anomaly Verification  IC=20260617", color="#d92525", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.905, "Three complete verification weeks: ERA5/ERA5T wind and IMERG rainfall", color="#0737c8", fontsize=10, fontweight="bold", ha="center")
    fig.text(0.025, 0.65, "850-hPa wind vector anomaly", rotation=90, va="center", fontsize=10.5, fontweight="bold")
    fig.text(0.025, 0.33, "Rainfall anomaly (mm/day)", rotation=90, va="center", fontsize=10.5, fontweight="bold")
    mappable = None
    for week in range(3):
        wind_ax = fig.add_subplot(grid[0, week], projection=ccrs.PlateCarree())
        draw_wind(wind_ax, wind_lon, wind_lat, u[week], v[week], outline, states, compact=True)
        wind_ax.set_title(labels()[week], color="#0a41ff", fontsize=9, fontweight="bold")
        rain_ax = fig.add_subplot(grid[1, week], projection=ccrs.PlateCarree())
        mappable = draw_rain(rain_ax, rain_lon, rain_lat, rain[week], outline, states, compact=True)
    cax = fig.add_subplot(grid[2, :])
    colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=[-20, -15, -10, -5, -2, 2, 5, 10, 15, 20], extend="both")
    colorbar.ax.tick_params(labelsize=8)
    fig.text(0.5, 0.035, "Wind anomaly: ERA5/ERA5T minus ERA5 1990-2017 climatology. Rainfall anomaly: IMERG Late minus IMERG Final 2001-2025 climatology.", ha="center", fontsize=7.8)
    return save_both(fig, output / "observed_wind850_rainfall_anomaly_20260617_3week_combined.png", dpi)


def main() -> int:
    args = parse_args()
    for path in (args.wind, args.wind_climatology, args.rainfall):
        if not path.exists():
            raise FileNotFoundError(path)
    all_dates = pd.date_range(WEEKS[0][0], WEEKS[-1][1], freq="D")
    with xr.open_dataset(args.wind) as wind_ds:
        wind = wind_ds.sel(time=all_dates)
        observed_u = weekly_mean(wind.u850)
        observed_v = weekly_mean(wind.v850)
        wind_lat = wind.lat.values.astype(np.float32)
        wind_lon = wind.lon.values.astype(np.float32)

    clim_lat, clim_lon, clim_u_daily, clim_v_daily = wind_climatology(args.wind_climatology, all_dates)
    clim_u = np.stack([clim_u_daily[index * 7 : (index + 1) * 7].mean(axis=0) for index in range(3)])
    clim_v = np.stack([clim_v_daily[index * 7 : (index + 1) * 7].mean(axis=0) for index in range(3)])
    clim_u_da = xr.DataArray(clim_u, dims=("week", "lat", "lon"), coords={"week": [1, 2, 3], "lat": clim_lat, "lon": clim_lon}).sortby("lat").sortby("lon")
    clim_v_da = xr.DataArray(clim_v, dims=("week", "lat", "lon"), coords={"week": [1, 2, 3], "lat": clim_lat, "lon": clim_lon}).sortby("lat").sortby("lon")
    target = {"lat": wind_lat, "lon": wind_lon}
    clim_u_on_gt = clim_u_da.interp(target).values.astype(np.float32)
    clim_v_on_gt = clim_v_da.interp(target).values.astype(np.float32)
    u_anomaly = observed_u - clim_u_on_gt
    v_anomaly = observed_v - clim_v_on_gt

    with xr.open_dataset(args.rainfall) as rain_ds:
        rain = rain_ds["imerg_late_minus_imerg_final_climatology_weekly"].isel(week=slice(0, 3)).values.astype(np.float32)
        rain_lat = rain_ds.lat.values.astype(np.float32)
        rain_lon = rain_ds.lon.values.astype(np.float32)

    if not all(np.isfinite(array).any() for array in (u_anomaly, v_anomaly, rain)):
        raise ValueError("anomaly fields contain no finite values")
    outline, states = prepare_india_geometries("50m", None, None, False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    products = []
    products += plot_wind(wind_lat, wind_lon, u_anomaly, v_anomaly, outline, states, args.output_dir, args.dpi)
    products += plot_rain(rain_lat, rain_lon, rain, outline, states, args.output_dir, args.dpi)
    products += plot_combined(wind_lat, wind_lon, u_anomaly, v_anomaly, rain_lat, rain_lon, rain, outline, states, args.output_dir, args.dpi)

    analysis = xr.Dataset(
        {
            "era5_u850_observed_weekly": (("week", "wind_lat", "wind_lon"), observed_u),
            "era5_v850_observed_weekly": (("week", "wind_lat", "wind_lon"), observed_v),
            "era5_u850_climatology_weekly": (("week", "wind_lat", "wind_lon"), clim_u_on_gt),
            "era5_v850_climatology_weekly": (("week", "wind_lat", "wind_lon"), clim_v_on_gt),
            "era5_u850_anomaly_weekly": (("week", "wind_lat", "wind_lon"), u_anomaly),
            "era5_v850_anomaly_weekly": (("week", "wind_lat", "wind_lon"), v_anomaly),
            "imerg_rainfall_anomaly_weekly": (("week", "rain_lat", "rain_lon"), rain),
        },
        coords={
            "week": [1, 2, 3],
            "wind_lat": wind_lat,
            "wind_lon": wind_lon,
            "rain_lat": rain_lat,
            "rain_lon": rain_lon,
            "week_start": ("week", [start.to_datetime64() for start, _ in WEEKS]),
            "week_end": ("week", [end.to_datetime64() for _, end in WEEKS]),
        },
        attrs={
            "title": "Observed weekly wind and rainfall anomalies for IC 20260617",
            "wind_observation": str(args.wind),
            "wind_climatology": str(args.wind_climatology),
            "wind_climatology_years": "1990-2017",
            "rainfall_analysis": str(args.rainfall),
            "rainfall_climatology_years": "2001-2025",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    analysis_path = args.output_dir / "observed_weekly_wind_rainfall_anomaly_20260617.nc"
    analysis.to_netcdf(analysis_path)
    products.append(analysis_path)
    manifest = args.output_dir / "manifest.json"
    manifest.write_text(json.dumps({"products": [str(path) for path in products], "weeks": [{"week": i, "start": str(start.date()), "end": str(end.date())} for i, (start, end) in enumerate(WEEKS, 1)], "wind_observation": str(args.wind), "wind_climatology": str(args.wind_climatology), "rainfall_analysis": str(args.rainfall)}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {analysis_path}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
