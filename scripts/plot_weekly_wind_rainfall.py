#!/usr/bin/env python3
"""Plot FuXi four-week 850-hPa wind and rainfall anomalies for 17 June."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyproj

# The compute-node environment can expose an incompatible system PROJ database.
# Pin pyproj to the database shipped with the validated FuXi environment.
pyproj.datadir.set_data_dir("/home/raj.ayush/.conda/envs/s2s-hind/share/proj")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.patches import Rectangle


HELPERS = Path("/home/raj.ayush/s2s/fuxi_s2s_Hindcast/scripts")
sys.path.insert(0, str(HELPERS))

from plot_one_member_india_forecast import prepare_india_geometries  # noqa: E402


DEFAULT_ANALYSIS = Path(
    "/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/"
    "daily_mean_strict00z_june17_wind/analysis/"
    "fuxi_weekly_wind_rainfall_anomaly_20260617.nc"
)
DEFAULT_OUTPUT = Path(
    "/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/"
    "daily_mean_strict00z_june17_wind/figures"
)
BBOX = (60.0, 100.0, 0.0, 40.0)
WEST_COAST_BOX = (68.0, 78.0, 7.0, 17.0)
RAIN_LEVELS = np.asarray([-20, -15, -10, -5, -2, 2, 5, 10, 15, 20], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--india-shapefile", type=Path)
    return parser.parse_args()


def rainfall_cmap() -> mcolors.ListedColormap:
    cmap = mcolors.ListedColormap(
        [
            "#ff5200",
            "#ff8e1d",
            "#ffca59",
            "#fff4a5",
            "#ffffff",
            "#c8c8e9",
            "#8c8cbf",
            "#6464a3",
            "#3c3c87",
        ],
        name="imd_rainfall_anomaly",
    )
    cmap.set_under("#d70e00")
    cmap.set_over("#00001e")
    return cmap


def add_map(ax, outline_geoms: list, state_geoms: list, compact: bool = False) -> None:
    ax.set_extent(BBOX, crs=ccrs.PlateCarree())
    ax.coastlines("50m", linewidth=0.75, zorder=5)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.5,
        edgecolor="0.35",
        zorder=5,
    )
    if outline_geoms:
        ax.add_geometries(
            outline_geoms,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=1.05,
            zorder=7,
        )
    if state_geoms:
        ax.add_geometries(
            state_geoms,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="0.12",
            linewidth=0.25,
            alpha=0.85,
            zorder=8,
        )
    ax.set_xticks([60, 70, 80, 90, 100], crs=ccrs.PlateCarree())
    ax.set_yticks([0, 10, 20, 30, 40], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f"))
    ax.tick_params(labelsize=7 if compact else 9, length=2.5, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_edgecolor("black")


def add_west_coast_box(ax) -> None:
    lon_min, lon_max, lat_min, lat_max = WEST_COAST_BOX
    ax.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            facecolor="none",
            edgecolor="black",
            linewidth=1.45,
            linestyle="--",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
    )


def week_label(dataset: xr.Dataset, week_index: int) -> str:
    start = pd.Timestamp(dataset.week_period_start.values[week_index])
    end = pd.Timestamp(dataset.week_period_end.values[week_index])
    return f"(Week{week_index + 1}: 00Z{start:%d%b}-00Z{end:%d%b})"


def draw_wind(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    outline_geoms: list,
    state_geoms: list,
    compact: bool = False,
):
    add_map(ax, outline_geoms, state_geoms, compact=compact)
    stride = 2
    quiver = ax.quiver(
        lon[::stride],
        lat[::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
        transform=ccrs.PlateCarree(),
        color="#151515",
        pivot="mid",
        angles="xy",
        scale_units="inches",
        scale=9.0,
        width=0.0040 if compact else 0.0035,
        headwidth=3.5,
        headlength=4.5,
        headaxislength=4.0,
        zorder=9,
    )
    add_west_coast_box(ax)
    ax.quiverkey(
        quiver,
        X=0.83,
        Y=0.06,
        U=2.0,
        label="2 m/s",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": 6.5 if compact else 8.0},
    )
    return quiver


def draw_rain(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    rainfall: np.ndarray,
    outline_geoms: list,
    state_geoms: list,
    compact: bool = False,
):
    cmap = rainfall_cmap()
    norm = mcolors.BoundaryNorm(RAIN_LEVELS, cmap.N)
    mappable = ax.contourf(
        lon,
        lat,
        rainfall,
        levels=RAIN_LEVELS,
        cmap=cmap,
        norm=norm,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    add_map(ax, outline_geoms, state_geoms, compact=compact)
    add_west_coast_box(ax)
    return mappable


def common_header(fig, title: str) -> None:
    fig.text(
        0.04,
        0.966,
        f"{title}  IC=20260617",
        color="#d92525",
        fontsize=14,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.5,
        0.932,
        "FuXi S2S 50-member ensemble mean minus 20-year FuXi S2S climatology (2002-2021; 11 members/year)",
        color="#0737c8",
        fontsize=9.2,
        fontweight="bold",
        ha="center",
    )


def save_both(fig, path: Path, dpi: int) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = path.with_suffix(".pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")
    print(f"wrote {pdf}")
    return [path, pdf]


def plot_wind_only(
    dataset: xr.Dataset,
    u: np.ndarray,
    v: np.ndarray,
    outline_geoms: list,
    state_geoms: list,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9.2, 8.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.08, hspace=0.30, wspace=0.13)
    common_header(fig, "FuXi S2S Weekly 850-hPa Wind Anomaly")
    for week, ax in enumerate(axes.flat):
        draw_wind(
            ax,
            dataset.lon.values,
            dataset.lat.values,
            u[week],
            v[week],
            outline_geoms,
            state_geoms,
        )
        ax.set_title(week_label(dataset, week), color="#0a41ff", fontsize=10, fontweight="bold", pad=4)
    fig.text(
        0.5,
        0.025,
        "Arrows: weekly mean vector anomaly (u850 forecast - climatology, v850 forecast - climatology), m/s. Dashed box: 68-78E, 7-17N.",
        ha="center",
        fontsize=7.6,
        color="#303030",
    )
    return save_both(
        fig,
        output_dir / "fuxi_wind850_anomaly_20260617_4week.png",
        dpi,
    )


def plot_rain_only(
    dataset: xr.Dataset,
    rainfall: np.ndarray,
    outline_geoms: list,
    state_geoms: list,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    fig = plt.figure(figsize=(9.2, 9.0), facecolor="white")
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.065], hspace=0.28, wspace=0.13)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.09)
    common_header(fig, "FuXi S2S Weekly Rainfall Anomaly (mm/day)")
    mappable = None
    for week in range(4):
        ax = fig.add_subplot(grid[week // 2, week % 2], projection=ccrs.PlateCarree())
        mappable = draw_rain(
            ax,
            dataset.lon.values,
            dataset.lat.values,
            rainfall[week],
            outline_geoms,
            state_geoms,
        )
        ax.set_title(week_label(dataset, week), color="#0a41ff", fontsize=10, fontweight="bold", pad=4)
    color_axis = fig.add_subplot(grid[2, :])
    colorbar = fig.colorbar(
        mappable,
        cax=color_axis,
        orientation="horizontal",
        ticks=RAIN_LEVELS,
        extend="both",
    )
    colorbar.ax.tick_params(labelsize=9, length=3, pad=2)
    fig.text(
        0.5,
        0.025,
        "Weekly ensemble-mean rainfall rate minus the lead-matched FuXi S2S model climatology; TP converted from mm/hour to mm/day before averaging.",
        ha="center",
        fontsize=7.6,
        color="#303030",
    )
    return save_both(
        fig,
        output_dir / "fuxi_tp_anomaly_20260617_4week.png",
        dpi,
    )


def plot_combined(
    dataset: xr.Dataset,
    u: np.ndarray,
    v: np.ndarray,
    rainfall: np.ndarray,
    outline_geoms: list,
    state_geoms: list,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    fig = plt.figure(figsize=(16.0, 8.4), facecolor="white")
    grid = fig.add_gridspec(
        3,
        4,
        height_ratios=[1, 1, 0.065],
        hspace=0.30,
        wspace=0.16,
        left=0.045,
        right=0.985,
        top=0.83,
        bottom=0.11,
    )
    fig.text(
        0.035,
        0.965,
        "FuXi S2S Coupled Wind-Rainfall Anomaly Diagnostic  IC=20260617",
        color="#d92525",
        fontsize=15,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.5,
        0.928,
        "50-member ensemble mean minus 20-year FuXi S2S climatology (2002-2021; 11 members/year)",
        color="#0737c8",
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    fig.text(0.26, 0.875, "850-hPa Wind Vector Anomaly (m/s)", ha="center", fontsize=11.5, fontweight="bold")
    fig.text(0.755, 0.875, "Rainfall Anomaly (mm/day)", ha="center", fontsize=11.5, fontweight="bold")

    mappable = None
    for week in range(4):
        row = week // 2
        pair_column = week % 2
        wind_ax = fig.add_subplot(grid[row, pair_column], projection=ccrs.PlateCarree())
        draw_wind(
            wind_ax,
            dataset.lon.values,
            dataset.lat.values,
            u[week],
            v[week],
            outline_geoms,
            state_geoms,
            compact=True,
        )
        wind_ax.set_title(week_label(dataset, week), color="#0a41ff", fontsize=8.5, fontweight="bold", pad=3)

        rain_ax = fig.add_subplot(grid[row, pair_column + 2], projection=ccrs.PlateCarree())
        mappable = draw_rain(
            rain_ax,
            dataset.lon.values,
            dataset.lat.values,
            rainfall[week],
            outline_geoms,
            state_geoms,
            compact=True,
        )
        rain_ax.set_title(week_label(dataset, week), color="#0a41ff", fontsize=8.5, fontweight="bold", pad=3)

    color_axis = fig.add_subplot(grid[2, 2:])
    colorbar = fig.colorbar(
        mappable,
        cax=color_axis,
        orientation="horizontal",
        ticks=RAIN_LEVELS,
        extend="both",
    )
    colorbar.ax.tick_params(labelsize=8, length=3, pad=2)
    fig.text(
        0.5,
        0.035,
        "Weekly means of daily FuXi fields. Wind anomaly is computed component-wise; rainfall anomaly uses mm/day. Dashed box: west-coast diagnostic region (68-78E, 7-17N).",
        ha="center",
        fontsize=7.7,
        color="#303030",
    )
    return save_both(
        fig,
        output_dir / "fuxi_wind850_rainfall_anomaly_20260617_4week_combined.png",
        dpi,
    )


def main() -> int:
    args = parse_args()
    outline_geoms, state_geoms = prepare_india_geometries(
        "50m", args.india_shapefile, None, False
    )
    with xr.open_dataset(args.analysis) as dataset:
        anomaly = dataset.anomaly_weekly
        u = anomaly.sel(variable="u850").isel(week=slice(0, 4)).values
        v = anomaly.sel(variable="v850").isel(week=slice(0, 4)).values
        rainfall = anomaly.sel(variable="tp").isel(week=slice(0, 4)).values
        products = []
        products.extend(
            plot_wind_only(
                dataset, u, v, outline_geoms, state_geoms, args.output_dir, args.dpi
            )
        )
        products.extend(
            plot_rain_only(
                dataset,
                rainfall,
                outline_geoms,
                state_geoms,
                args.output_dir,
                args.dpi,
            )
        )
        products.extend(
            plot_combined(
                dataset,
                u,
                v,
                rainfall,
                outline_geoms,
                state_geoms,
                args.output_dir,
                args.dpi,
            )
        )

    manifest = args.output_dir / "plot_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "analysis": str(args.analysis),
                "products": [str(path) for path in products],
                "rainfall_levels_mm_day": RAIN_LEVELS.tolist(),
                "wind_quiver_key_m_s": 2.0,
                "west_coast_box": list(WEST_COAST_BOX),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
