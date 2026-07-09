#!/usr/bin/env python3
"""Plot May-17 FuXi/ECMWF rainfall forecasts against ARCO ERA5 truth."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from shapely import contains_xy
from shapely.ops import unary_union

from build_june17_climatology import DATA_VAR, DEFAULT_BBOX, convert_units, coord_slice, parse_ints
from make_fuxi_weekly_analysis import member_path
from plot_one_member_india_forecast import Grid, add_map, mask_to_geometries, prepare_india_geometries


DEFAULT_IC_DATE = "20260517"
DEFAULT_LEAD_DAYS = 30
DEFAULT_FUXI_RAW_DIR = Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/raw/20260517")
DEFAULT_ECMWF_FILE = Path(
    "/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/processed/"
    "ecmwf_20260517_tp_ens50_lead42_india_1p5deg_daily_mm.nc"
)
DEFAULT_ECMWF_CF_FILE = Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/raw/tp/20260517_cf.nc")
DEFAULT_ARCO_FILE = Path(
    "/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/truth/"
    "arco_era5_tp_daily_20260517.nc"
)
DEFAULT_OUTPUT_DIR = Path("outputs/may17_fuxi_ecmwf_arco")

COLORS = {
    "fuxi": "#2ca25f",
    "fuxi_dark": "#006d3c",
    "ecmwf": "#ff8c1a",
    "ecmwf_dark": "#b85200",
    "arco": "#1559a6",
    "text": "#1f2933",
    "muted": "#5b6472",
    "grid": "#dce3ea",
}


@dataclass
class ModelProduct:
    name: str
    lat: np.ndarray
    lon: np.ndarray
    daily_member_mean: np.ndarray
    accum_member: np.ndarray

    @property
    def daily_ens_mean(self) -> np.ndarray:
        return self.daily_member_mean.mean(axis=0)

    @property
    def cumulative_member(self) -> np.ndarray:
        return np.cumsum(self.daily_member_mean, axis=1)

    @property
    def cumulative_ens_mean(self) -> np.ndarray:
        return self.cumulative_member.mean(axis=0)

    @property
    def accum_mean(self) -> np.ndarray:
        return self.accum_member.mean(axis=0)


@dataclass
class TruthProduct:
    lat: np.ndarray
    lon: np.ndarray
    daily_mean: np.ndarray
    accum: np.ndarray

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.daily_mean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic-date", default=DEFAULT_IC_DATE)
    parser.add_argument("--lead-days", type=int, default=DEFAULT_LEAD_DAYS)
    parser.add_argument("--members", default="0:49")
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    parser.add_argument("--fuxi-raw-dir", type=Path, default=DEFAULT_FUXI_RAW_DIR)
    parser.add_argument("--ecmwf-file", type=Path, default=DEFAULT_ECMWF_FILE)
    parser.add_argument("--ecmwf-cf-file", type=Path, default=DEFAULT_ECMWF_CF_FILE)
    parser.add_argument("--arco-file", type=Path, default=DEFAULT_ARCO_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--map-scale", default="50m", choices=("10m", "50m", "110m"))
    parser.add_argument("--india-shapefile", type=Path)
    parser.add_argument("--district-shapefile", type=Path)
    parser.add_argument("--draw-districts", action="store_true")
    parser.add_argument("--no-state-lines", action="store_true")
    parser.add_argument("--mask-maps-to-india", action="store_true")
    return parser.parse_args()


def configure_style(dpi: int) -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": dpi,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#2c2f33",
            "axes.linewidth": 0.85,
            "axes.labelsize": 11.0,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
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


def fuxi_metadata(path: Path, bbox: tuple[float, float, float, float]) -> tuple[int, np.ndarray, np.ndarray, slice, slice]:
    lon_min, lon_max, lat_min, lat_max = bbox
    with Dataset(path) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float32)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float32)
        lat_sel = coord_slice(lat, lat_min, lat_max)
        lon_sel = coord_slice(lon, lon_min, lon_max)
        channels = np.asarray(ds.variables["channel"][:]).astype(str)
        matches = np.flatnonzero(channels == "tp")
        if matches.size == 0:
            raise ValueError(f"tp channel not found in {path}")
        return int(matches[0]), lat[lat_sel], lon[lon_sel], lat_sel, lon_sel


def read_fuxi(
    raw_dir: Path,
    members: list[int],
    lead_days: int,
    bbox: tuple[float, float, float, float],
    india_geoms: list,
) -> ModelProduct:
    first = member_path(raw_dir, members[0], 1)
    tp_idx, lat, lon, lat_sel, lon_sel = fuxi_metadata(first, bbox)
    _, weights = mask_and_weights(lat, lon, india_geoms)
    daily_mean = np.zeros((len(members), lead_days), dtype=np.float32)
    accum = np.zeros((len(members), len(lat), len(lon)), dtype=np.float32)

    for member_idx, member in enumerate(members):
        member_accum = np.zeros((len(lat), len(lon)), dtype=np.float32)
        for lead_day in range(1, lead_days + 1):
            path = member_path(raw_dir, member, lead_day)
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            with Dataset(path) as ds:
                field = np.asarray(ds.variables[DATA_VAR][0, 0, tp_idx, lat_sel, lon_sel], dtype=np.float32)
            field = convert_units("tp", field)
            daily_mean[member_idx, lead_day - 1] = weighted_mean(field, weights)
            member_accum += field
        accum[member_idx] = member_accum
        if member_idx == 0 or (member_idx + 1) % 10 == 0 or member_idx + 1 == len(members):
            print(f"read FuXi member {member_idx + 1}/{len(members)}", flush=True)

    return ModelProduct("FuXi-S2S", lat, lon, daily_mean, accum)


def select_latlon(da: xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    lon_min, lon_max, lat_min, lat_max = bbox
    lat_name = "lat" if "lat" in da.coords else "latitude"
    lon_name = "lon" if "lon" in da.coords else "longitude"
    lat = da[lat_name]
    lon = da[lon_name]
    return da.sel(
        {
            lat_name: lat[(lat >= lat_min) & (lat <= lat_max)],
            lon_name: lon[(lon >= lon_min) & (lon <= lon_max)],
        }
    )


def read_ecmwf(path: Path, lead_days: int, bbox: tuple[float, float, float, float], india_geoms: list) -> ModelProduct:
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path)
    try:
        da = select_latlon(ds["tp"].isel(lead_time=slice(0, lead_days)), bbox)
        lat = da.lat.values.astype("float32")
        lon = da.lon.values.astype("float32")
        _, weights = mask_and_weights(lat, lon, india_geoms)
        values = da.transpose("member", "lead_time", "lat", "lon").values.astype("float32")
        daily_mean = np.zeros((values.shape[0], lead_days), dtype=np.float32)
        for member_idx in range(values.shape[0]):
            for lead_idx in range(lead_days):
                daily_mean[member_idx, lead_idx] = weighted_mean(values[member_idx, lead_idx], weights)
        accum = values.sum(axis=1).astype("float32")
        return ModelProduct("ECMWF-S2S", lat, lon, daily_mean, accum)
    finally:
        ds.close()


def open_ecmwf_raw(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path)
    except ValueError as exc:
        if "Failed to decode variable 'step'" not in str(exc):
            raise
        return xr.open_dataset(path, decode_timedelta=False)


def read_ecmwf_control_cumulative(
    path: Path,
    lead_days: int,
    bbox: tuple[float, float, float, float],
    india_geoms: list,
) -> np.ndarray | None:
    if not path.exists():
        return None
    ds = open_ecmwf_raw(path)
    try:
        da = select_latlon(ds["tp"].isel(step=slice(0, lead_days)), bbox)
        lat_name = "lat" if "lat" in da.coords else "latitude"
        lon_name = "lon" if "lon" in da.coords else "longitude"
        lat = da[lat_name].values.astype("float32")
        lon = da[lon_name].values.astype("float32")
        _, weights = mask_and_weights(lat, lon, india_geoms)
        values = da.values.astype("float32")
        return np.asarray([weighted_mean(values[idx], weights) for idx in range(lead_days)], dtype=np.float32)
    finally:
        ds.close()


def read_arco(path: Path, lead_days: int, bbox: tuple[float, float, float, float], india_geoms: list) -> TruthProduct:
    if not path.exists():
        raise FileNotFoundError(path)
    ds = xr.open_dataset(path)
    try:
        available = int(ds.sizes["lead_day"])
        if available < lead_days:
            raise ValueError(f"ARCO truth has only {available} complete lead day(s); requested {lead_days}")
        da = select_latlon(ds["tp_daily"].isel(lead_day=slice(0, lead_days)), bbox)
        lat = da.lat.values.astype("float32")
        lon = da.lon.values.astype("float32")
        _, weights = mask_and_weights(lat, lon, india_geoms)
        values = da.values.astype("float32")
        daily_mean = np.asarray([weighted_mean(values[idx], weights) for idx in range(lead_days)], dtype=np.float32)
        accum = values.sum(axis=0).astype("float32")
        return TruthProduct(lat, lon, daily_mean, accum)
    finally:
        ds.close()


def date_range_for(ic_date: str, lead_days: int) -> pd.DatetimeIndex:
    init = datetime.strptime(ic_date, "%Y%m%d")
    return pd.date_range(init + timedelta(days=1), periods=lead_days, freq="D")


def build_timeseries(
    ic_date: str,
    lead_days: int,
    fuxi: ModelProduct,
    ecmwf: ModelProduct,
    truth: TruthProduct,
    ecmwf_control_cumulative: np.ndarray | None,
) -> pd.DataFrame:
    valid_dates = date_range_for(ic_date, lead_days)
    fuxi_cum = fuxi.cumulative_member
    ecmwf_cum = ecmwf.cumulative_member
    data = {
        "lead_day": np.arange(1, lead_days + 1, dtype=np.int32),
        "valid_date": valid_dates.strftime("%Y-%m-%d"),
        "arco_daily_mm": truth.daily_mean,
        "arco_cumulative_mm": truth.cumulative,
        "fuxi_daily_ens_mean_mm": fuxi.daily_ens_mean,
        "fuxi_cumulative_p10_mm": np.percentile(fuxi_cum, 10, axis=0),
        "fuxi_cumulative_ens_mean_mm": fuxi_cum.mean(axis=0),
        "fuxi_cumulative_p90_mm": np.percentile(fuxi_cum, 90, axis=0),
        "fuxi_member00_cumulative_mm": fuxi_cum[0],
        "ecmwf_daily_ens_mean_mm": ecmwf.daily_ens_mean,
        "ecmwf_cumulative_p10_mm": np.percentile(ecmwf_cum, 10, axis=0),
        "ecmwf_cumulative_ens_mean_mm": ecmwf_cum.mean(axis=0),
        "ecmwf_cumulative_p90_mm": np.percentile(ecmwf_cum, 90, axis=0),
    }
    if ecmwf_control_cumulative is not None:
        data["ecmwf_control_cumulative_mm"] = ecmwf_control_cumulative
    return pd.DataFrame(data)


def plot_cumulative(df: pd.DataFrame, ic_date: str, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.6, 7.1), facecolor="white")
    x = df["lead_day"].to_numpy()

    ax.fill_between(x, df["fuxi_cumulative_p10_mm"], df["fuxi_cumulative_p90_mm"], color=COLORS["fuxi"], alpha=0.14, linewidth=0)
    ax.plot(x, df["fuxi_cumulative_ens_mean_mm"], color=COLORS["fuxi"], lw=3.0, label="FuXi-S2S ensemble mean")
    ax.fill_between(x, df["ecmwf_cumulative_p10_mm"], df["ecmwf_cumulative_p90_mm"], color=COLORS["ecmwf"], alpha=0.16, linewidth=0)
    ax.plot(x, df["ecmwf_cumulative_ens_mean_mm"], color=COLORS["ecmwf"], lw=3.0, label="ECMWF-S2S ensemble mean")
    ax.plot(x, df["arco_cumulative_mm"], color=COLORS["arco"], lw=3.2, label="ARCO ERA5 truth")

    tick_days = [1, 7, 14, 21, 28, int(x[-1])]
    tick_days = list(dict.fromkeys(tick_days))
    valid_lookup = {int(row.lead_day): pd.Timestamp(row.valid_date) for row in df.itertuples()}
    ax.set_xticks(tick_days, [f"L{lead}\n{valid_lookup[lead].strftime('%b %-d')}" for lead in tick_days])
    ymax = max(
        float(df["fuxi_cumulative_p90_mm"].max()),
        float(df["ecmwf_cumulative_p90_mm"].max()),
        float(df["arco_cumulative_mm"].max()),
    )
    ax.set_ylim(0, ymax * 1.13)
    ax.set_xlim(0.2, float(x[-1]) + 0.8)
    ax.set_xlabel("Lead day and valid date")
    ax.set_ylabel("All-India cumulative rainfall (mm)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#d9dee5", framealpha=0.96)

    fig.text(0.055, 0.965, "May-17 IC Cumulative Rainfall Verification", fontsize=20, fontweight="bold", color=COLORS["text"])
    fig.text(
        0.055,
        0.925,
        f"IC {ic_date} | lead days 1-{int(x[-1])} | ARCO ERA5 complete truth window | area-weighted India mean",
        fontsize=11.4,
        color=COLORS["muted"],
    )
    fig.text(
        0.055,
        0.045,
        "Shaded bands show member p10-p90. FuXi and ECMWF both use 50 perturbed members; ARCO ERA5 uses daily totals from 24 hourly fields.",
        fontsize=9.2,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.16, top=0.86)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def annotate_endpoint(ax: plt.Axes, x: float, y: float, label: str, color: str, offset: float = 0.0) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + 0.7, y + offset),
        ha="left",
        va="center",
        color=color,
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=color, linewidth=1.0, alpha=0.96),
    )


def plot_cumulative_paper_style(df: pd.DataFrame, ic_date: str, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15.2, 7.6), facecolor="white")
    x = df["lead_day"].to_numpy()

    ax.fill_between(x, df["fuxi_cumulative_p10_mm"], df["fuxi_cumulative_p90_mm"], color=COLORS["fuxi"], alpha=0.12, linewidth=0)
    ax.fill_between(x, df["ecmwf_cumulative_p10_mm"], df["ecmwf_cumulative_p90_mm"], color=COLORS["ecmwf"], alpha=0.14, linewidth=0)
    ax.plot(x, df["arco_cumulative_mm"], color=COLORS["arco"], lw=3.1, label="ARCO ERA5 truth")
    ax.plot(x, df["fuxi_cumulative_ens_mean_mm"], color=COLORS["fuxi"], lw=3.0, label="FuXi-S2S ensemble mean")
    ax.plot(x, df["fuxi_member00_cumulative_mm"], color=COLORS["fuxi_dark"], lw=2.4, ls=(0, (7, 5)), label="FuXi member 00")
    ax.plot(x, df["ecmwf_cumulative_ens_mean_mm"], color=COLORS["ecmwf"], lw=3.0, label="ECMWF-S2S ensemble mean")
    if "ecmwf_control_cumulative_mm" in df.columns:
        ax.plot(x, df["ecmwf_control_cumulative_mm"], color=COLORS["ecmwf_dark"], lw=2.4, ls=(0, (7, 5)), label="ECMWF control")

    tick_days = [1, 7, 14, 21, 28, int(x[-1])]
    tick_days = list(dict.fromkeys(tick_days))
    valid_lookup = {int(row.lead_day): pd.Timestamp(row.valid_date) for row in df.itertuples()}
    ax.set_xticks(tick_days, [f"L{lead}\n{valid_lookup[lead].strftime('%b %-d')}" for lead in tick_days])
    ymax = max(
        float(df["fuxi_cumulative_p90_mm"].max()),
        float(df["ecmwf_cumulative_p90_mm"].max()),
        float(df["arco_cumulative_mm"].max()),
        float(df["ecmwf_control_cumulative_mm"].max()) if "ecmwf_control_cumulative_mm" in df.columns else 0.0,
    )
    ax.set_ylim(0, ymax * 1.22)
    ax.set_xlim(0.2, float(x[-1]) + 2.4)
    ax.set_xlabel("Lead day and valid date")
    ax.set_ylabel("Cumulative rainfall (mm)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#d9dee5",
        framealpha=0.96,
        borderpad=0.6,
        columnspacing=1.4,
    )

    final_x = float(x[-1])
    final_values = {
        "ECMWF": float(df["ecmwf_cumulative_ens_mean_mm"].iloc[-1]),
        "ARCO": float(df["arco_cumulative_mm"].iloc[-1]),
        "FuXi": float(df["fuxi_cumulative_ens_mean_mm"].iloc[-1]),
    }
    annotate_endpoint(ax, final_x, final_values["ECMWF"], f"ECMWF {final_values['ECMWF']:.0f} mm", COLORS["ecmwf"], offset=6.0)
    annotate_endpoint(ax, final_x, final_values["ARCO"], f"ARCO {final_values['ARCO']:.0f} mm", COLORS["arco"], offset=0.0)
    annotate_endpoint(ax, final_x, final_values["FuXi"], f"FuXi {final_values['FuXi']:.0f} mm", COLORS["fuxi"], offset=-5.0)

    valid_dates = pd.to_datetime(df["valid_date"])
    fig.text(0.055, 0.965, f"{int(x[-1])}-Day Cumulative Rainfall Forecast over India", fontsize=22, fontweight="bold", color=COLORS["text"])
    fig.text(
        0.055,
        0.925,
        f"Initialized {pd.Timestamp(datetime.strptime(ic_date, '%Y%m%d')).strftime('%-d %b %Y')} | "
        f"valid {valid_dates.iloc[0].strftime('%-d %b')}-{valid_dates.iloc[-1].strftime('%-d %b')} | "
        "FuXi-S2S and ECMWF-S2S versus ARCO ERA5 truth",
        fontsize=12,
        color=COLORS["muted"],
    )
    fig.text(
        0.055,
        0.035,
        "ARCO ERA5 truth is available as complete daily totals through lead day 30; shaded bands show member p10-p90.",
        fontsize=8.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.15, top=0.86)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def rainfall_cmap(levels: np.ndarray) -> mcolors.Colormap:
    cmap = mpl.colormaps["YlGnBu"].resampled(len(levels) - 1)
    cmap.set_under("#ffffff")
    cmap.set_over("#08306b")
    return cmap


def difference_cmap(levels: np.ndarray) -> mcolors.Colormap:
    cmap = mpl.colormaps["RdBu"].resampled(len(levels) - 1)
    cmap.set_under("#7f0000")
    cmap.set_over("#053061")
    return cmap


def plot_spatial(
    ic_date: str,
    lead_days: int,
    fuxi: ModelProduct,
    ecmwf: ModelProduct,
    truth: TruthProduct,
    bbox: tuple[float, float, float, float],
    output: Path,
    map_scale: str,
    draw_states: bool,
    india_outline_geoms: list,
    india_state_geoms: list,
    mask_maps_to_india: bool,
) -> Path:
    fuxi_grid = Grid(fuxi.lat, fuxi.lon)
    ecmwf_grid = Grid(ecmwf.lat, ecmwf.lon)
    truth_grid = Grid(truth.lat, truth.lon)
    fuxi_accum = fuxi.accum_mean
    ecmwf_accum = ecmwf.accum_mean
    truth_accum = truth.accum
    difference = fuxi_accum - ecmwf_accum

    if mask_maps_to_india:
        fuxi_accum = mask_to_geometries(fuxi_accum, fuxi_grid, india_outline_geoms)
        ecmwf_accum = mask_to_geometries(ecmwf_accum, ecmwf_grid, india_outline_geoms)
        truth_accum = mask_to_geometries(truth_accum, truth_grid, india_outline_geoms)
        difference = mask_to_geometries(difference, fuxi_grid, india_outline_geoms)

    rain_levels = np.asarray([0, 25, 50, 100, 150, 200, 300, 500, 750, 1000], dtype=float)
    diff_levels = np.asarray([-300, -200, -100, -50, -25, -10, 10, 25, 50, 100, 200, 300], dtype=float)
    rain_cmap = rainfall_cmap(rain_levels)
    diff_cmap = difference_cmap(diff_levels)
    rain_norm = mcolors.BoundaryNorm(rain_levels, rain_cmap.N)
    diff_norm = mcolors.BoundaryNorm(diff_levels, diff_cmap.N)

    fig = plt.figure(figsize=(12.6, 10.3), facecolor="white")
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1, 1, 0.08], hspace=0.28, wspace=0.12)
    panels = [
        ("ARCO ERA5 truth", truth.lon, truth.lat, truth_accum, rain_cmap, rain_norm, rain_levels, "rain"),
        ("FuXi-S2S ensemble mean", fuxi.lon, fuxi.lat, fuxi_accum, rain_cmap, rain_norm, rain_levels, "rain"),
        ("ECMWF-S2S ensemble mean", ecmwf.lon, ecmwf.lat, ecmwf_accum, rain_cmap, rain_norm, rain_levels, "rain"),
        ("FuXi minus ECMWF", fuxi.lon, fuxi.lat, difference, diff_cmap, diff_norm, diff_levels, "diff"),
    ]

    rain_mappable = None
    diff_mappable = None
    for idx, (title, lon, lat, field, cmap, norm, _, kind) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection=ccrs.PlateCarree())
        mappable = ax.contourf(
            lon,
            lat,
            field,
            levels=rain_levels if kind == "rain" else diff_levels,
            cmap=cmap,
            norm=norm,
            extend="both",
            transform=ccrs.PlateCarree(),
        )
        add_map(ax, bbox, map_scale, draw_states, india_outline_geoms, india_state_geoms)
        ax.set_title(title, color=COLORS["text"], fontsize=12, pad=6)
        if kind == "rain":
            rain_mappable = mappable
        else:
            diff_mappable = mappable

    cax1 = fig.add_subplot(gs[2, 0])
    cb1 = fig.colorbar(rain_mappable, cax=cax1, orientation="horizontal", ticks=rain_levels)
    cb1.set_label(f"{lead_days}-day cumulative rainfall (mm)")
    cb1.ax.tick_params(labelsize=8.8)

    cax2 = fig.add_subplot(gs[2, 1])
    cb2 = fig.colorbar(diff_mappable, cax=cax2, orientation="horizontal", ticks=diff_levels)
    cb2.set_label("FuXi minus ECMWF (mm)")
    cb2.ax.tick_params(labelsize=8.8)

    valid_dates = date_range_for(ic_date, lead_days)
    fig.text(0.055, 0.965, f"Spatial {lead_days}-Day Cumulative Rainfall", fontsize=20, fontweight="bold", color=COLORS["text"])
    fig.text(
        0.055,
        0.93,
        f"IC {ic_date} | valid {valid_dates[0]:%d %b %Y}-{valid_dates[-1]:%d %b %Y} | shaded {'India mask' if mask_maps_to_india else 'full domain'} with India/state boundaries",
        fontsize=11.2,
        color=COLORS["muted"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def interp_to_grid(field: np.ndarray, lat: np.ndarray, lon: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    da = xr.DataArray(field, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    target_lat_sorted = np.sort(target_lat)
    out = da.sortby("lat").sortby("lon").interp(lat=target_lat_sorted, lon=target_lon)
    out = out.sel(lat=target_lat)
    return out.values.astype("float32")


def plot_bias_spatial(
    ic_date: str,
    lead_days: int,
    fuxi: ModelProduct,
    ecmwf: ModelProduct,
    truth: TruthProduct,
    bbox: tuple[float, float, float, float],
    output: Path,
    map_scale: str,
    draw_states: bool,
    india_outline_geoms: list,
    india_state_geoms: list,
    mask_maps_to_india: bool,
) -> Path:
    fuxi_grid = Grid(fuxi.lat, fuxi.lon)
    truth_grid = Grid(truth.lat, truth.lon)
    fuxi_accum = fuxi.accum_mean
    ecmwf_on_fuxi = interp_to_grid(ecmwf.accum_mean, ecmwf.lat, ecmwf.lon, fuxi.lat, fuxi.lon)
    truth_on_fuxi = interp_to_grid(truth.accum, truth.lat, truth.lon, fuxi.lat, fuxi.lon)
    truth_accum = truth.accum

    fuxi_bias = fuxi_accum - truth_on_fuxi
    ecmwf_bias = ecmwf_on_fuxi - truth_on_fuxi
    model_diff = fuxi_accum - ecmwf_on_fuxi

    if mask_maps_to_india:
        truth_accum = mask_to_geometries(truth_accum, truth_grid, india_outline_geoms)
        fuxi_bias = mask_to_geometries(fuxi_bias, fuxi_grid, india_outline_geoms)
        ecmwf_bias = mask_to_geometries(ecmwf_bias, fuxi_grid, india_outline_geoms)
        model_diff = mask_to_geometries(model_diff, fuxi_grid, india_outline_geoms)

    rain_levels = np.asarray([0, 25, 50, 100, 150, 200, 300, 500, 750, 1000], dtype=float)
    diff_levels = np.asarray([-300, -200, -100, -50, -25, -10, 10, 25, 50, 100, 200, 300], dtype=float)
    rain_cmap = rainfall_cmap(rain_levels)
    diff_cmap = difference_cmap(diff_levels)
    rain_norm = mcolors.BoundaryNorm(rain_levels, rain_cmap.N)
    diff_norm = mcolors.BoundaryNorm(diff_levels, diff_cmap.N)

    fig = plt.figure(figsize=(12.6, 10.3), facecolor="white")
    gs = fig.add_gridspec(nrows=3, ncols=2, height_ratios=[1, 1, 0.08], hspace=0.28, wspace=0.12)
    panels = [
        ("ARCO ERA5 truth", truth.lon, truth.lat, truth_accum, rain_cmap, rain_norm, rain_levels, "rain"),
        ("FuXi minus ARCO", fuxi.lon, fuxi.lat, fuxi_bias, diff_cmap, diff_norm, diff_levels, "diff"),
        ("ECMWF minus ARCO", fuxi.lon, fuxi.lat, ecmwf_bias, diff_cmap, diff_norm, diff_levels, "diff"),
        ("FuXi minus ECMWF", fuxi.lon, fuxi.lat, model_diff, diff_cmap, diff_norm, diff_levels, "diff"),
    ]

    rain_mappable = None
    diff_mappable = None
    for idx, (title, lon, lat, field, cmap, norm, levels, kind) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection=ccrs.PlateCarree())
        mappable = ax.contourf(
            lon,
            lat,
            field,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
            transform=ccrs.PlateCarree(),
        )
        add_map(ax, bbox, map_scale, draw_states, india_outline_geoms, india_state_geoms)
        ax.set_title(title, color=COLORS["text"], fontsize=12, pad=6)
        if kind == "rain":
            rain_mappable = mappable
        else:
            diff_mappable = mappable

    cax1 = fig.add_subplot(gs[2, 0])
    cb1 = fig.colorbar(rain_mappable, cax=cax1, orientation="horizontal", ticks=rain_levels)
    cb1.set_label(f"{lead_days}-day cumulative rainfall (mm)")
    cb1.ax.tick_params(labelsize=8.8)

    cax2 = fig.add_subplot(gs[2, 1])
    cb2 = fig.colorbar(diff_mappable, cax=cax2, orientation="horizontal", ticks=diff_levels)
    cb2.set_label("Difference (mm)")
    cb2.ax.tick_params(labelsize=8.8)

    valid_dates = date_range_for(ic_date, lead_days)
    fig.text(0.055, 0.965, f"Spatial {lead_days}-Day Rainfall Bias", fontsize=20, fontweight="bold", color=COLORS["text"])
    fig.text(
        0.055,
        0.93,
        f"IC {ic_date} | valid {valid_dates[0]:%d %b %Y}-{valid_dates[-1]:%d %b %Y} | ARCO truth interpolated to the model grid for bias panels",
        fontsize=11.2,
        color=COLORS["muted"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def write_spatial_cache(output: Path, ic_date: str, lead_days: int, fuxi: ModelProduct, ecmwf: ModelProduct, truth: TruthProduct) -> Path:
    ecmwf_on_fuxi = interp_to_grid(ecmwf.accum_mean, ecmwf.lat, ecmwf.lon, fuxi.lat, fuxi.lon)
    arco_on_model = interp_to_grid(truth.accum, truth.lat, truth.lon, fuxi.lat, fuxi.lon)
    ds = xr.Dataset(
        data_vars={
            "fuxi_accum_mean": (("model_lat", "model_lon"), fuxi.accum_mean.astype("float32")),
            "ecmwf_accum_mean": (("model_lat", "model_lon"), ecmwf_on_fuxi.astype("float32")),
            "arco_on_model_grid": (("model_lat", "model_lon"), arco_on_model.astype("float32")),
            "fuxi_minus_arco": (("model_lat", "model_lon"), (fuxi.accum_mean - arco_on_model).astype("float32")),
            "ecmwf_minus_arco": (("model_lat", "model_lon"), (ecmwf_on_fuxi - arco_on_model).astype("float32")),
            "fuxi_minus_ecmwf": (("model_lat", "model_lon"), (fuxi.accum_mean - ecmwf_on_fuxi).astype("float32")),
            "arco_accum": (("arco_lat", "arco_lon"), truth.accum.astype("float32")),
        },
        coords={
            "model_lat": fuxi.lat.astype("float32"),
            "model_lon": fuxi.lon.astype("float32"),
            "arco_lat": truth.lat.astype("float32"),
            "arco_lon": truth.lon.astype("float32"),
        },
        attrs={
            "title": "May-17 FuXi/ECMWF/ARCO cumulative rainfall comparison fields",
            "ic_date": ic_date,
            "lead_days": int(lead_days),
            "units": "mm",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    for name in ds.data_vars:
        ds[name].attrs["units"] = "mm"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    encoding = {name: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)} for name in ds.data_vars}
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.replace(output)
    return output


def write_manifest(output: Path, payload: dict) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> int:
    args = parse_args()
    configure_style(args.dpi)
    members = parse_ints(args.members)
    bbox = tuple(args.bbox)
    india_outline_geoms, india_state_geoms = prepare_india_geometries(
        args.map_scale,
        args.india_shapefile,
        args.district_shapefile,
        args.draw_districts,
    )

    print("loading FuXi...")
    fuxi = read_fuxi(args.fuxi_raw_dir, members, args.lead_days, bbox, india_outline_geoms)
    print("loading ECMWF...")
    ecmwf = read_ecmwf(args.ecmwf_file, args.lead_days, bbox, india_outline_geoms)
    ecmwf_control_cumulative = read_ecmwf_control_cumulative(args.ecmwf_cf_file, args.lead_days, bbox, india_outline_geoms)
    print("loading ARCO...")
    truth = read_arco(args.arco_file, args.lead_days, bbox, india_outline_geoms)

    df = build_timeseries(args.ic_date, args.lead_days, fuxi, ecmwf, truth, ecmwf_control_cumulative)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_cumulative_timeseries.csv"
    fields_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_spatial_fields.nc"
    line_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_cumulative_rainfall_fuxi_ecmwf_arco.png"
    paper_line_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_cumulative_rainfall_paperstyle_fuxi_ecmwf_arco.png"
    spatial_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_spatial_4panel_fuxi_ecmwf_arco.png"
    bias_spatial_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_spatial_bias_4panel_fuxi_ecmwf_arco.png"
    manifest_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_verification_manifest.json"

    df.to_csv(csv_out, index=False)
    write_spatial_cache(fields_out, args.ic_date, args.lead_days, fuxi, ecmwf, truth)
    plot_cumulative(df, args.ic_date, line_out)
    plot_cumulative_paper_style(df, args.ic_date, paper_line_out)
    plot_spatial(
        ic_date=args.ic_date,
        lead_days=args.lead_days,
        fuxi=fuxi,
        ecmwf=ecmwf,
        truth=truth,
        bbox=bbox,
        output=spatial_out,
        map_scale=args.map_scale,
        draw_states=not args.no_state_lines,
        india_outline_geoms=india_outline_geoms,
        india_state_geoms=india_state_geoms,
        mask_maps_to_india=args.mask_maps_to_india,
    )
    plot_bias_spatial(
        ic_date=args.ic_date,
        lead_days=args.lead_days,
        fuxi=fuxi,
        ecmwf=ecmwf,
        truth=truth,
        bbox=bbox,
        output=bias_spatial_out,
        map_scale=args.map_scale,
        draw_states=not args.no_state_lines,
        india_outline_geoms=india_outline_geoms,
        india_state_geoms=india_state_geoms,
        mask_maps_to_india=args.mask_maps_to_india,
    )

    summary = {
        "ic_date": args.ic_date,
        "lead_days": args.lead_days,
        "valid_start": df["valid_date"].iloc[0],
        "valid_end": df["valid_date"].iloc[-1],
        "fuxi_raw_dir": str(args.fuxi_raw_dir),
        "ecmwf_file": str(args.ecmwf_file),
        "ecmwf_cf_file": str(args.ecmwf_cf_file),
        "arco_file": str(args.arco_file),
        "members": len(members),
        "timeseries_csv": str(csv_out),
        "spatial_fields": str(fields_out),
        "cumulative_plot": str(line_out),
        "cumulative_paperstyle_plot": str(paper_line_out),
        "spatial_plot": str(spatial_out),
        "spatial_bias_plot": str(bias_spatial_out),
        "final_cumulative_mm": {
            "arco_era5": float(df["arco_cumulative_mm"].iloc[-1]),
            "fuxi_ens_mean": float(df["fuxi_cumulative_ens_mean_mm"].iloc[-1]),
            "ecmwf_ens_mean": float(df["ecmwf_cumulative_ens_mean_mm"].iloc[-1]),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(manifest_out, summary)

    print(f"wrote csv     : {csv_out}")
    print(f"wrote fields  : {fields_out}")
    print(f"wrote line    : {line_out}")
    print(f"wrote paper   : {paper_line_out}")
    print(f"wrote spatial : {spatial_out}")
    print(f"wrote bias map: {bias_spatial_out}")
    print(f"wrote manifest: {manifest_out}")
    print("final cumulative mm:", summary["final_cumulative_mm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
