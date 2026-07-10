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
DEFAULT_IMD_CLIMATOLOGY = Path(
    "/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/"
    "imd_rain_1991_2020_daily_climatology.nc"
)
DEFAULT_IMD_REGION_MASKS = Path("/storage/raj.ayush/s2s-forecast-data-prev/era5/daily/imd_region_masks.nc")
DEFAULT_OUTPUT_DIR = Path("outputs/may17_fuxi_ecmwf_arco")

REGION_VARS = {
    "Northwest India": "northwest_india",
    "Central India": "central_india",
    "South Peninsula": "south_peninsula",
    "East & Northeast India": "east_northeast_india",
}

COLORS = {
    "fuxi": "#2ca25f",
    "fuxi_dark": "#006d3c",
    "ecmwf": "#ff8c1a",
    "ecmwf_dark": "#b85200",
    "arco": "#111827",
    "imd_clim": "#1559a6",
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
    daily_member_field: np.ndarray | None = None

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
    daily_field: np.ndarray | None = None

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
    parser.add_argument("--imd-climatology", type=Path, default=DEFAULT_IMD_CLIMATOLOGY)
    parser.add_argument("--imd-region-masks", type=Path, default=DEFAULT_IMD_REGION_MASKS)
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
    daily_field = np.zeros((len(members), lead_days, len(lat), len(lon)), dtype=np.float32)
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
            daily_field[member_idx, lead_day - 1] = field
            daily_mean[member_idx, lead_day - 1] = weighted_mean(field, weights)
            member_accum += field
        accum[member_idx] = member_accum
        if member_idx == 0 or (member_idx + 1) % 10 == 0 or member_idx + 1 == len(members):
            print(f"read FuXi member {member_idx + 1}/{len(members)}", flush=True)

    return ModelProduct("FuXi-S2S", lat, lon, daily_mean, accum, daily_field)


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
        return ModelProduct("ECMWF-S2S", lat, lon, daily_mean, accum, values)
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
        return TruthProduct(lat, lon, daily_mean, accum, values)
    finally:
        ds.close()


def date_range_for(ic_date: str, lead_days: int) -> pd.DatetimeIndex:
    init = datetime.strptime(ic_date, "%Y%m%d")
    return pd.date_range(init + timedelta(days=1), periods=lead_days, freq="D")


def month_day_key(date: pd.Timestamp) -> str:
    return f"{date.month:02d}-{date.day:02d}"


def read_imd_climatology(path: Path, valid_dates: pd.DatetimeIndex, india_geoms: list) -> np.ndarray | None:
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    try:
        if "rain_mean" not in ds:
            raise ValueError(f"{path}: missing rain_mean")
        if "month_day" not in ds:
            raise ValueError(f"{path}: missing month_day coordinate")
        lat = ds.lat.values.astype("float32")
        lon = ds.lon.values.astype("float32")
        _, weights = mask_and_weights(lat, lon, india_geoms)
        month_days = [str(item) for item in ds["month_day"].values.astype(str)]
        lookup = {item: idx for idx, item in enumerate(month_days)}
        daily = []
        for date in valid_dates:
            key = month_day_key(pd.Timestamp(date))
            if key == "02-29":
                daily.append(np.nan)
                continue
            if key not in lookup:
                raise ValueError(f"{path}: missing climatology day {key}")
            field = ds["rain_mean"].isel(day=lookup[key]).values.astype("float32")
            daily.append(weighted_mean(field, weights))
        return np.asarray(daily, dtype=np.float32)
    finally:
        ds.close()


def read_imd_climatology_grid(path: Path, valid_dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    try:
        if "rain_mean" not in ds:
            raise ValueError(f"{path}: missing rain_mean")
        month_days = [str(item) for item in ds["month_day"].values.astype(str)]
        lookup = {item: idx for idx, item in enumerate(month_days)}
        fields = []
        for date in valid_dates:
            key = month_day_key(pd.Timestamp(date))
            if key == "02-29":
                fields.append(np.full((ds.sizes["lat"], ds.sizes["lon"]), np.nan, dtype=np.float32))
                continue
            if key not in lookup:
                raise ValueError(f"{path}: missing climatology day {key}")
            fields.append(ds["rain_mean"].isel(day=lookup[key]).values.astype("float32"))
        return ds.lat.values.astype("float32"), ds.lon.values.astype("float32"), np.stack(fields, axis=0)
    finally:
        ds.close()


def load_region_masks(path: Path) -> xr.Dataset | None:
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    rename = {old: new for old, new in {"latitude": "lat", "longitude": "lon"}.items() if old in ds.dims}
    if rename:
        ds = ds.rename(rename)
    return ds.sortby("lat").sortby("lon")


def mask_weights(mask_da: xr.DataArray, lat: np.ndarray, lon: np.ndarray, label: str) -> np.ndarray:
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)
    target_lat_sorted = np.sort(lat)
    mask = (
        mask_da.astype("float32")
        .interp(lat=target_lat_sorted, lon=lon, method="nearest", kwargs={"fill_value": "extrapolate"})
        .sel(lat=lat)
        .values
        >= 0.5
    )
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, np.newaxis], mask.shape).astype("float64").copy()
    weights[~mask] = 0.0
    if not np.any(weights > 0):
        raise RuntimeError(f"{label} has no grid cells on target grid")
    return weights


def region_weights(mask_ds: xr.Dataset, region_var: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if region_var not in mask_ds:
        raise ValueError(f"missing IMD region mask variable: {region_var}")
    return mask_weights(mask_ds[region_var], lat, lon, f"IMD region mask {region_var}")


def region_union_weights(mask_ds: xr.Dataset, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    union = None
    for region_var in REGION_VARS.values():
        if region_var not in mask_ds:
            raise ValueError(f"missing IMD region mask variable: {region_var}")
        region = mask_ds[region_var] >= 0.5
        union = region if union is None else (union | region)
    if union is None:
        raise RuntimeError("no IMD homogeneous region masks found")
    return mask_weights(union.astype("float32"), lat, lon, "IMD homogeneous region union")


def weighted_member_daily(fields: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = np.nansum(weights)
    return (np.nansum(fields * weights[np.newaxis, np.newaxis, :, :], axis=(-2, -1)) / denominator).astype("float32")


def weighted_field_daily(fields: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = np.nansum(weights)
    return (np.nansum(fields * weights[np.newaxis, :, :], axis=(-2, -1)) / denominator).astype("float32")


def build_regional_timeseries(
    ic_date: str,
    lead_days: int,
    fuxi: ModelProduct,
    ecmwf: ModelProduct,
    truth: TruthProduct,
    imd_climatology_path: Path,
    region_masks_path: Path,
) -> pd.DataFrame | None:
    if fuxi.daily_member_field is None or ecmwf.daily_member_field is None or truth.daily_field is None:
        raise RuntimeError("regional plotting requires daily gridded fields")
    mask_ds = load_region_masks(region_masks_path)
    if mask_ds is None:
        return None
    valid_dates = date_range_for(ic_date, lead_days)
    imd_grid = read_imd_climatology_grid(imd_climatology_path, valid_dates)
    rows = []
    try:
        for region_name, region_var in REGION_VARS.items():
            fuxi_daily = weighted_member_daily(fuxi.daily_member_field, region_weights(mask_ds, region_var, fuxi.lat, fuxi.lon))
            ecmwf_daily = weighted_member_daily(ecmwf.daily_member_field, region_weights(mask_ds, region_var, ecmwf.lat, ecmwf.lon))
            truth_daily = weighted_field_daily(truth.daily_field, region_weights(mask_ds, region_var, truth.lat, truth.lon))
            imd_daily = None
            if imd_grid is not None:
                imd_lat, imd_lon, imd_fields = imd_grid
                imd_daily = weighted_field_daily(imd_fields, region_weights(mask_ds, region_var, imd_lat, imd_lon))

            fuxi_cum = np.cumsum(fuxi_daily, axis=1)
            ecmwf_cum = np.cumsum(ecmwf_daily, axis=1)
            truth_cum = np.cumsum(truth_daily)
            imd_cum = np.cumsum(imd_daily) if imd_daily is not None else None

            for lead_idx, valid_date in enumerate(valid_dates):
                row = {
                    "region": region_name,
                    "lead_day": lead_idx + 1,
                    "valid_date": valid_date.strftime("%Y-%m-%d"),
                    "era5_gt_daily_mm": truth_daily[lead_idx],
                    "era5_gt_cumulative_mm": truth_cum[lead_idx],
                    "fuxi_daily_ens_mean_mm": fuxi_daily[:, lead_idx].mean(),
                    "fuxi_cumulative_p10_mm": np.percentile(fuxi_cum[:, lead_idx], 10),
                    "fuxi_cumulative_ens_mean_mm": fuxi_cum[:, lead_idx].mean(),
                    "fuxi_cumulative_p90_mm": np.percentile(fuxi_cum[:, lead_idx], 90),
                    "ecmwf_daily_ens_mean_mm": ecmwf_daily[:, lead_idx].mean(),
                    "ecmwf_cumulative_p10_mm": np.percentile(ecmwf_cum[:, lead_idx], 10),
                    "ecmwf_cumulative_ens_mean_mm": ecmwf_cum[:, lead_idx].mean(),
                    "ecmwf_cumulative_p90_mm": np.percentile(ecmwf_cum[:, lead_idx], 90),
                }
                if imd_daily is not None and imd_cum is not None:
                    row["imd_1991_2020_climatology_daily_mm"] = imd_daily[lead_idx]
                    row["imd_1991_2020_climatology_cumulative_mm"] = imd_cum[lead_idx]
                rows.append(row)
        return pd.DataFrame(rows)
    finally:
        mask_ds.close()


def build_region_union_timeseries(
    ic_date: str,
    lead_days: int,
    fuxi: ModelProduct,
    ecmwf: ModelProduct,
    truth: TruthProduct,
    imd_climatology_path: Path,
    region_masks_path: Path,
) -> pd.DataFrame | None:
    if fuxi.daily_member_field is None or ecmwf.daily_member_field is None or truth.daily_field is None:
        raise RuntimeError("region-union plotting requires daily gridded fields")
    mask_ds = load_region_masks(region_masks_path)
    if mask_ds is None:
        return None
    valid_dates = date_range_for(ic_date, lead_days)
    imd_grid = read_imd_climatology_grid(imd_climatology_path, valid_dates)
    try:
        fuxi_daily = weighted_member_daily(fuxi.daily_member_field, region_union_weights(mask_ds, fuxi.lat, fuxi.lon))
        ecmwf_daily = weighted_member_daily(ecmwf.daily_member_field, region_union_weights(mask_ds, ecmwf.lat, ecmwf.lon))
        truth_daily = weighted_field_daily(truth.daily_field, region_union_weights(mask_ds, truth.lat, truth.lon))
        imd_daily = None
        if imd_grid is not None:
            imd_lat, imd_lon, imd_fields = imd_grid
            imd_daily = weighted_field_daily(imd_fields, region_union_weights(mask_ds, imd_lat, imd_lon))

        fuxi_cum = np.cumsum(fuxi_daily, axis=1)
        ecmwf_cum = np.cumsum(ecmwf_daily, axis=1)
        truth_cum = np.cumsum(truth_daily)
        data = {
            "lead_day": np.arange(1, lead_days + 1, dtype=np.int32),
            "valid_date": valid_dates.strftime("%Y-%m-%d"),
            "arco_daily_mm": truth_daily,
            "arco_cumulative_mm": truth_cum,
            "fuxi_daily_ens_mean_mm": fuxi_daily.mean(axis=0),
            "fuxi_cumulative_p10_mm": np.percentile(fuxi_cum, 10, axis=0),
            "fuxi_cumulative_ens_mean_mm": fuxi_cum.mean(axis=0),
            "fuxi_cumulative_p90_mm": np.percentile(fuxi_cum, 90, axis=0),
            "fuxi_member00_cumulative_mm": fuxi_cum[0],
            "ecmwf_daily_ens_mean_mm": ecmwf_daily.mean(axis=0),
            "ecmwf_cumulative_p10_mm": np.percentile(ecmwf_cum, 10, axis=0),
            "ecmwf_cumulative_ens_mean_mm": ecmwf_cum.mean(axis=0),
            "ecmwf_cumulative_p90_mm": np.percentile(ecmwf_cum, 90, axis=0),
        }
        if imd_daily is not None:
            data["imd_1991_2020_climatology_daily_mm"] = imd_daily
            data["imd_1991_2020_climatology_cumulative_mm"] = np.cumsum(imd_daily)
        return pd.DataFrame(data)
    finally:
        mask_ds.close()


def build_timeseries(
    ic_date: str,
    lead_days: int,
    fuxi: ModelProduct,
    ecmwf: ModelProduct,
    truth: TruthProduct,
    ecmwf_control_cumulative: np.ndarray | None,
    imd_climatology_daily: np.ndarray | None,
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
    if imd_climatology_daily is not None:
        data["imd_1991_2020_climatology_daily_mm"] = imd_climatology_daily
        data["imd_1991_2020_climatology_cumulative_mm"] = np.cumsum(imd_climatology_daily)
    return pd.DataFrame(data)


def cumulative_ymax(df: pd.DataFrame, columns: list[str]) -> float:
    values = [float(df[column].max(skipna=True)) for column in columns if column in df.columns]
    if not values:
        return 1.0
    return max(values)


def plot_cumulative(df: pd.DataFrame, ic_date: str, output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13.6, 7.1), facecolor="white")
    x = df["lead_day"].to_numpy()

    ax.fill_between(x, df["fuxi_cumulative_p10_mm"], df["fuxi_cumulative_p90_mm"], color=COLORS["fuxi"], alpha=0.14, linewidth=0)
    ax.plot(x, df["fuxi_cumulative_ens_mean_mm"], color=COLORS["fuxi"], lw=3.0, label="FuXi-S2S ensemble mean")
    ax.fill_between(x, df["ecmwf_cumulative_p10_mm"], df["ecmwf_cumulative_p90_mm"], color=COLORS["ecmwf"], alpha=0.16, linewidth=0)
    ax.plot(x, df["ecmwf_cumulative_ens_mean_mm"], color=COLORS["ecmwf"], lw=3.0, label="ECMWF-S2S ensemble mean")
    ax.plot(x, df["arco_cumulative_mm"], color=COLORS["arco"], lw=3.2, label="ERA5 GT")
    if "imd_1991_2020_climatology_cumulative_mm" in df.columns:
        ax.plot(
            x,
            df["imd_1991_2020_climatology_cumulative_mm"],
            color=COLORS["imd_clim"],
            lw=2.8,
            ls=(0, (5, 3)),
            label="IMD 1991-2020 climatology",
        )

    tick_days = [1, 7, 14, 21, 28, int(x[-1])]
    tick_days = list(dict.fromkeys(tick_days))
    valid_lookup = {int(row.lead_day): pd.Timestamp(row.valid_date) for row in df.itertuples()}
    ax.set_xticks(tick_days, [f"L{lead}\n{valid_lookup[lead].strftime('%b %-d')}" for lead in tick_days])
    ymax = cumulative_ymax(
        df,
        [
            "fuxi_cumulative_p90_mm",
            "ecmwf_cumulative_p90_mm",
            "arco_cumulative_mm",
            "imd_1991_2020_climatology_cumulative_mm",
        ],
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
        f"IC {ic_date} | lead days 1-{int(x[-1])} | ERA5 GT complete truth window | IMD climatology reference | area-weighted India mean",
        fontsize=11.4,
        color=COLORS["muted"],
    )
    fig.text(
        0.055,
        0.045,
        "Shaded bands show member p10-p90. FuXi and ECMWF both use 50 perturbed members; ERA5 GT uses daily totals from 24 hourly fields.",
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


def endpoint_offsets(values: dict[str, float], min_separation: float = 7.0) -> dict[str, float]:
    adjusted = {}
    previous = None
    for name, value in sorted(values.items(), key=lambda item: item[1]):
        text_y = value if previous is None else max(value, previous + min_separation)
        adjusted[name] = text_y - value
        previous = text_y
    return adjusted


def plot_cumulative_paper_style(
    df: pd.DataFrame,
    ic_date: str,
    output: Path,
    area_label: str = "India",
    footnote_extra: str = "",
) -> Path:
    fig, ax = plt.subplots(figsize=(15.2, 7.6), facecolor="white")
    x = df["lead_day"].to_numpy()

    ax.fill_between(x, df["fuxi_cumulative_p10_mm"], df["fuxi_cumulative_p90_mm"], color=COLORS["fuxi"], alpha=0.12, linewidth=0)
    ax.fill_between(x, df["ecmwf_cumulative_p10_mm"], df["ecmwf_cumulative_p90_mm"], color=COLORS["ecmwf"], alpha=0.14, linewidth=0)
    ax.plot(x, df["arco_cumulative_mm"], color=COLORS["arco"], lw=3.1, label="ERA5 GT")
    ax.plot(x, df["fuxi_cumulative_ens_mean_mm"], color=COLORS["fuxi"], lw=3.0, label="FuXi-S2S ensemble mean")
    ax.plot(x, df["fuxi_member00_cumulative_mm"], color=COLORS["fuxi_dark"], lw=2.4, ls=(0, (7, 5)), label="FuXi member 00")
    ax.plot(x, df["ecmwf_cumulative_ens_mean_mm"], color=COLORS["ecmwf"], lw=3.0, label="ECMWF-S2S ensemble mean")
    if "ecmwf_control_cumulative_mm" in df.columns:
        ax.plot(x, df["ecmwf_control_cumulative_mm"], color=COLORS["ecmwf_dark"], lw=2.4, ls=(0, (7, 5)), label="ECMWF control")
    if "imd_1991_2020_climatology_cumulative_mm" in df.columns:
        ax.plot(
            x,
            df["imd_1991_2020_climatology_cumulative_mm"],
            color=COLORS["imd_clim"],
            lw=2.8,
            ls=(0, (5, 3)),
            label="IMD 1991-2020 climatology",
        )

    tick_days = [1, 7, 14, 21, 28, int(x[-1])]
    tick_days = list(dict.fromkeys(tick_days))
    valid_lookup = {int(row.lead_day): pd.Timestamp(row.valid_date) for row in df.itertuples()}
    ax.set_xticks(tick_days, [f"L{lead}\n{valid_lookup[lead].strftime('%b %-d')}" for lead in tick_days])
    ymax = cumulative_ymax(
        df,
        [
            "fuxi_cumulative_p90_mm",
            "ecmwf_cumulative_p90_mm",
            "arco_cumulative_mm",
            "ecmwf_control_cumulative_mm",
            "imd_1991_2020_climatology_cumulative_mm",
        ],
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
        "ERA5 GT": float(df["arco_cumulative_mm"].iloc[-1]),
        "FuXi": float(df["fuxi_cumulative_ens_mean_mm"].iloc[-1]),
    }
    if "imd_1991_2020_climatology_cumulative_mm" in df.columns:
        final_values["IMD"] = float(df["imd_1991_2020_climatology_cumulative_mm"].iloc[-1])
    offsets = endpoint_offsets(final_values)
    annotate_endpoint(ax, final_x, final_values["ECMWF"], f"ECMWF {final_values['ECMWF']:.0f} mm", COLORS["ecmwf"], offset=offsets["ECMWF"])
    annotate_endpoint(ax, final_x, final_values["ERA5 GT"], f"ERA5 GT {final_values['ERA5 GT']:.0f} mm", COLORS["arco"], offset=offsets["ERA5 GT"])
    annotate_endpoint(ax, final_x, final_values["FuXi"], f"FuXi {final_values['FuXi']:.0f} mm", COLORS["fuxi"], offset=offsets["FuXi"])
    if "IMD" in final_values:
        annotate_endpoint(ax, final_x, final_values["IMD"], f"IMD clim {final_values['IMD']:.0f} mm", COLORS["imd_clim"], offset=offsets["IMD"])

    valid_dates = pd.to_datetime(df["valid_date"])
    fig.text(0.055, 0.965, f"{int(x[-1])}-Day Cumulative Rainfall Forecast over {area_label}", fontsize=22, fontweight="bold", color=COLORS["text"])
    fig.text(
        0.055,
        0.925,
        f"Initialized {pd.Timestamp(datetime.strptime(ic_date, '%Y%m%d')).strftime('%-d %b %Y')} | "
        f"valid {valid_dates.iloc[0].strftime('%-d %b')}-{valid_dates.iloc[-1].strftime('%-d %b')} | "
        "FuXi-S2S and ECMWF-S2S versus ERA5 GT and IMD climatology",
        fontsize=12,
        color=COLORS["muted"],
    )
    fig.text(
        0.055,
        0.035,
        "ERA5 GT is available as complete daily totals through lead day 30; IMD line is 1991-2020 daily rainfall climatology; shaded bands show member p10-p90."
        + (f" {footnote_extra}" if footnote_extra else ""),
        fontsize=8.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.15, top=0.86)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_regional_cumulative(df: pd.DataFrame, ic_date: str, lead_days: int, output: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.4), facecolor="white")
    axes = axes.ravel()

    for idx, (ax, region) in enumerate(zip(axes, REGION_VARS)):
        sub = df[df["region"] == region].copy()
        x = sub["lead_day"].to_numpy()

        ax.fill_between(x, sub["fuxi_cumulative_p10_mm"], sub["fuxi_cumulative_p90_mm"], color=COLORS["fuxi"], alpha=0.12, linewidth=0, label="FuXi p10-p90")
        ax.fill_between(x, sub["ecmwf_cumulative_p10_mm"], sub["ecmwf_cumulative_p90_mm"], color=COLORS["ecmwf"], alpha=0.13, linewidth=0, label="ECMWF p10-p90")
        ax.plot(x, sub["era5_gt_cumulative_mm"], color=COLORS["arco"], lw=2.7, label="ERA5 GT")
        if "imd_1991_2020_climatology_cumulative_mm" in sub.columns:
            ax.plot(
                x,
                sub["imd_1991_2020_climatology_cumulative_mm"],
                color=COLORS["imd_clim"],
                lw=2.5,
                ls=(0, (5, 3)),
                label="IMD 1991-2020 climatology",
            )
        ax.plot(x, sub["fuxi_cumulative_ens_mean_mm"], color=COLORS["fuxi"], lw=2.7, label="FuXi-S2S ensemble mean")
        ax.plot(x, sub["ecmwf_cumulative_ens_mean_mm"], color=COLORS["ecmwf"], lw=2.7, label="ECMWF-S2S ensemble mean")

        tick_days = [1, 7, 14, 21, 28, int(x[-1])]
        tick_days = list(dict.fromkeys(tick_days))
        valid_lookup = {int(row.lead_day): pd.Timestamp(row.valid_date) for row in sub.itertuples()}
        ax.set_xticks(tick_days, [f"L{lead}\n{valid_lookup[lead].strftime('%b %-d')}" for lead in tick_days])
        ax.set_xlim(0.2, float(x[-1]) + 0.8)
        ymax = cumulative_ymax(
            sub,
            [
                "fuxi_cumulative_p90_mm",
                "ecmwf_cumulative_p90_mm",
                "era5_gt_cumulative_mm",
                "imd_1991_2020_climatology_cumulative_mm",
            ],
        )
        ax.set_ylim(0, ymax * 1.16)
        ax.set_title(region, loc="left", color=COLORS["text"], fontsize=13, pad=8)
        ax.set_xlabel("Lead day and valid date")
        ax.set_ylabel("Cumulative rainfall (mm)" if idx % 2 == 0 else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        final = sub.iloc[-1]
        totals = (
            f"ERA5 GT {final['era5_gt_cumulative_mm']:.0f} | "
            f"IMD clim {final.get('imd_1991_2020_climatology_cumulative_mm', np.nan):.0f} | "
            f"FuXi {final['fuxi_cumulative_ens_mean_mm']:.0f} | "
            f"ECMWF {final['ecmwf_cumulative_ens_mean_mm']:.0f} mm"
        )
        ax.text(
            0.012,
            0.972,
            totals,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            color=COLORS["text"],
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#d9dee5", alpha=0.95),
        )
    valid_dates = pd.to_datetime(df["valid_date"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.905),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#d9dee5",
        framealpha=0.95,
        borderpad=0.55,
        columnspacing=1.2,
        fontsize=9.0,
    )
    fig.text(0.045, 0.975, f"IMD Homogeneous Regions: {lead_days}-Day Cumulative Rainfall", fontsize=20, fontweight="bold", color=COLORS["text"])
    fig.text(
        0.045,
        0.94,
        f"Initialized {pd.Timestamp(datetime.strptime(ic_date, '%Y%m%d')).strftime('%-d %b %Y')} | "
        f"valid {valid_dates.iloc[0].strftime('%-d %b')}-{valid_dates.iloc[-1].strftime('%-d %b')} | "
        "FuXi-S2S and ECMWF-S2S versus ERA5 GT and IMD 1991-2020 climatology",
        fontsize=11.4,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.80, wspace=0.14, hspace=0.30)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def regional_final_totals(df: pd.DataFrame | None) -> dict:
    if df is None:
        return {}
    totals = {}
    for region, sub in df.groupby("region", sort=False):
        final = sub.sort_values("lead_day").iloc[-1]
        totals[region] = {
            "era5_gt": float(final["era5_gt_cumulative_mm"]),
            "fuxi_ens_mean": float(final["fuxi_cumulative_ens_mean_mm"]),
            "ecmwf_ens_mean": float(final["ecmwf_cumulative_ens_mean_mm"]),
        }
        if "imd_1991_2020_climatology_cumulative_mm" in final:
            totals[region]["imd_1991_2020_climatology"] = float(final["imd_1991_2020_climatology_cumulative_mm"])
    return totals


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
        ("ERA5 GT", truth.lon, truth.lat, truth_accum, rain_cmap, rain_norm, rain_levels, "rain"),
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
        ("ERA5 GT", truth.lon, truth.lat, truth_accum, rain_cmap, rain_norm, rain_levels, "rain"),
        ("FuXi minus ERA5 GT", fuxi.lon, fuxi.lat, fuxi_bias, diff_cmap, diff_norm, diff_levels, "diff"),
        ("ECMWF minus ERA5 GT", fuxi.lon, fuxi.lat, ecmwf_bias, diff_cmap, diff_norm, diff_levels, "diff"),
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
        f"IC {ic_date} | valid {valid_dates[0]:%d %b %Y}-{valid_dates[-1]:%d %b %Y} | ERA5 GT interpolated to the model grid for bias panels",
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
    print("loading IMD climatology...")
    imd_climatology_daily = read_imd_climatology(
        args.imd_climatology,
        date_range_for(args.ic_date, args.lead_days),
        india_outline_geoms,
    )

    df = build_timeseries(args.ic_date, args.lead_days, fuxi, ecmwf, truth, ecmwf_control_cumulative, imd_climatology_daily)
    regional_df = build_regional_timeseries(
        args.ic_date,
        args.lead_days,
        fuxi,
        ecmwf,
        truth,
        args.imd_climatology,
        args.imd_region_masks,
    )
    region_union_df = build_region_union_timeseries(
        args.ic_date,
        args.lead_days,
        fuxi,
        ecmwf,
        truth,
        args.imd_climatology,
        args.imd_region_masks,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_cumulative_timeseries.csv"
    regional_csv_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_imd_homogeneous_regions_cumulative_timeseries.csv"
    region_union_csv_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_india_imd_region_union_cumulative_timeseries.csv"
    fields_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_spatial_fields.nc"
    line_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_cumulative_rainfall_fuxi_ecmwf_arco.png"
    paper_line_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_cumulative_rainfall_paperstyle_fuxi_ecmwf_arco.png"
    region_union_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_india_imd_region_union_cumulative_rainfall.png"
    regional_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_imd_homogeneous_regions_cumulative_rainfall.png"
    spatial_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_spatial_4panel_fuxi_ecmwf_arco.png"
    bias_spatial_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_spatial_bias_4panel_fuxi_ecmwf_arco.png"
    manifest_out = args.output_dir / f"{args.ic_date}_lead1_{args.lead_days}_verification_manifest.json"

    df.to_csv(csv_out, index=False)
    if regional_df is not None:
        regional_df.to_csv(regional_csv_out, index=False)
    if region_union_df is not None:
        region_union_df.to_csv(region_union_csv_out, index=False)
    write_spatial_cache(fields_out, args.ic_date, args.lead_days, fuxi, ecmwf, truth)
    plot_cumulative(df, args.ic_date, line_out)
    plot_cumulative_paper_style(df, args.ic_date, paper_line_out)
    if region_union_df is not None:
        plot_cumulative_paper_style(
            region_union_df,
            args.ic_date,
            region_union_out,
            area_label="India (IMD Region-Mask Union)",
            footnote_extra="Area mean uses the union of the four IMD homogeneous rainfall-region masks.",
        )
    if regional_df is not None:
        plot_regional_cumulative(regional_df, args.ic_date, args.lead_days, regional_out)
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
        "imd_climatology_file": str(args.imd_climatology) if args.imd_climatology.exists() else "missing",
        "imd_region_masks": str(args.imd_region_masks) if args.imd_region_masks.exists() else "missing",
        "members": len(members),
        "timeseries_csv": str(csv_out),
        "regional_timeseries_csv": str(regional_csv_out) if regional_df is not None else None,
        "india_imd_region_union_timeseries_csv": str(region_union_csv_out) if region_union_df is not None else None,
        "spatial_fields": str(fields_out),
        "cumulative_plot": str(line_out),
        "cumulative_paperstyle_plot": str(paper_line_out),
        "india_imd_region_union_plot": str(region_union_out) if region_union_df is not None else None,
        "regional_cumulative_plot": str(regional_out) if regional_df is not None else None,
        "spatial_plot": str(spatial_out),
        "spatial_bias_plot": str(bias_spatial_out),
        "final_cumulative_mm": {
            "arco_era5": float(df["arco_cumulative_mm"].iloc[-1]),
            "fuxi_ens_mean": float(df["fuxi_cumulative_ens_mean_mm"].iloc[-1]),
            "ecmwf_ens_mean": float(df["ecmwf_cumulative_ens_mean_mm"].iloc[-1]),
            "imd_1991_2020_climatology": (
                float(df["imd_1991_2020_climatology_cumulative_mm"].iloc[-1])
                if "imd_1991_2020_climatology_cumulative_mm" in df.columns
                else None
            ),
        },
        "regional_final_cumulative_mm": regional_final_totals(regional_df),
        "india_imd_region_union_final_cumulative_mm": (
            {
                "era5_gt": float(region_union_df["arco_cumulative_mm"].iloc[-1]),
                "fuxi_ens_mean": float(region_union_df["fuxi_cumulative_ens_mean_mm"].iloc[-1]),
                "ecmwf_ens_mean": float(region_union_df["ecmwf_cumulative_ens_mean_mm"].iloc[-1]),
                "imd_1991_2020_climatology": (
                    float(region_union_df["imd_1991_2020_climatology_cumulative_mm"].iloc[-1])
                    if "imd_1991_2020_climatology_cumulative_mm" in region_union_df.columns
                    else None
                ),
            }
            if region_union_df is not None
            else {}
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(manifest_out, summary)

    print(f"wrote csv     : {csv_out}")
    if regional_df is not None:
        print(f"wrote regions : {regional_csv_out}")
    if region_union_df is not None:
        print(f"wrote union   : {region_union_csv_out}")
    print(f"wrote fields  : {fields_out}")
    print(f"wrote line    : {line_out}")
    print(f"wrote paper   : {paper_line_out}")
    if region_union_df is not None:
        print(f"wrote IMDIndia: {region_union_out}")
    if regional_df is not None:
        print(f"wrote regional: {regional_out}")
    print(f"wrote spatial : {spatial_out}")
    print(f"wrote bias map: {bias_spatial_out}")
    print(f"wrote manifest: {manifest_out}")
    print("final cumulative mm:", summary["final_cumulative_mm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
