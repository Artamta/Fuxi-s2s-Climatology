#!/usr/bin/env python3
"""Create PPT-ready FuXi weekly slides matching the ERPAS 4-week layout."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plot_fuxi_weekly_analysis import imd_temperature_actual_cmap, imd_temperature_anomaly_cmap, listed_cmap
from plot_one_member_india_forecast import DEFAULT_BBOX, Grid, add_map, mask_to_geometries, prepare_india_geometries


DEFAULT_ANALYSIS = Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc")
PPT_WEEKS = (1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-file", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ppt_ready_20260617"))
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--map-scale", default="50m", choices=("10m", "50m", "110m"))
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    parser.add_argument("--india-shapefile", type=Path)
    parser.add_argument("--district-shapefile", type=Path)
    parser.add_argument("--draw-districts", action="store_true")
    parser.add_argument("--no-state-lines", action="store_true")
    parser.add_argument("--no-mask-to-india", action="store_true", help="Shade full domain for temperature slides instead of masking to India.")
    parser.add_argument("--rainfall-scale", choices=("ppt", "fuxi"), default="ppt")
    return parser.parse_args()


def rainfall_actual_cmap(ncolors: int) -> mcolors.Colormap:
    if ncolors == 5:
        return listed_cmap(
            "erpas_ppt_rain_actual",
            ["#ffffff", "#8affa1", "#4ef45b", "#12cc15", "#009600"],
            under="#ffffff",
            over="#007800",
        )
    if ncolors == 4:
        return listed_cmap(
            "erpas_ppt_rain_actual",
            ["#8affa1", "#4ef45b", "#12cc15", "#009600"],
            under="#ffffff",
            over="#007800",
        )
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "erpas_ppt_rain_actual_scaled",
        ["#8affa1", "#4ef45b", "#12cc15", "#009600"],
        N=ncolors,
    )
    cmap.set_under("#ffffff")
    cmap.set_over("#007800")
    return cmap


def rainfall_anomaly_cmap() -> mcolors.Colormap:
    return listed_cmap(
        "erpas_ppt_rain_anomaly",
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
        under="#d70e00",
        over="#00001e",
    )


def week_label(init: datetime, week_index: int, start_day: int, end_day: int) -> str:
    start = init + timedelta(days=int(start_day))
    end = init + timedelta(days=int(end_day))
    return f"(Week{week_index}: {start:%d%b}-{end:%d%b})"


def slide_title(fig: plt.Figure, text: str) -> None:
    fig.text(0.5, 0.965, text, ha="center", va="top", color="#0000cc", fontsize=18, fontweight="bold")


def plot_2x2_block(
    fig: plt.Figure,
    outer,
    ds: xr.Dataset,
    field: np.ndarray,
    variable_title: str,
    levels: np.ndarray,
    cmap: mcolors.Colormap,
    extend: str,
    bbox: tuple[float, float, float, float],
    map_scale: str,
    draw_states: bool,
    india_outline_geoms: list,
    india_state_geoms: list,
    mask_india: bool,
) -> None:
    ic_date = str(ds.attrs.get("ic_date", "unknown"))
    init = datetime.strptime(ic_date, "%Y%m%d")
    lat = ds.lat.values.astype("float32")
    lon = ds.lon.values.astype("float32")
    grid = Grid(lat=lat, lon=lon)
    if mask_india:
        field = np.stack([mask_to_geometries(item, grid, india_outline_geoms) for item in field], axis=0)

    sub = outer.subgridspec(nrows=3, ncols=2, height_ratios=[0.12, 1, 0.11], hspace=0.24, wspace=0.13)
    title_ax = fig.add_subplot(sub[0, :])
    title_ax.axis("off")
    title_ax.text(0.02, 0.65, variable_title, color="#e33b3b", fontsize=9.5, fontweight="bold", ha="left")
    title_ax.text(0.98, 0.65, f"IC={ic_date}", color="black", fontsize=9.5, fontweight="bold", ha="right")

    panel_sub = sub[1, :].subgridspec(nrows=2, ncols=2, hspace=0.2, wspace=0.14)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    mappable = None
    for idx, week_number in enumerate(PPT_WEEKS):
        ax = fig.add_subplot(panel_sub[idx // 2, idx % 2], projection=ccrs.PlateCarree())
        mappable = ax.contourf(
            lon,
            lat,
            field[week_number - 1],
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend=extend,
            transform=ccrs.PlateCarree(),
        )
        add_map(ax, bbox, map_scale, draw_states, india_outline_geoms, india_state_geoms)
        ax.set_title(
            week_label(
                init,
                week_number,
                int(ds.week_start_day.values[week_number - 1]),
                int(ds.week_end_day.values[week_number - 1]),
            ),
            fontsize=8.2,
            color="#0a41ff",
            pad=3,
            fontweight="bold",
        )
        ax.tick_params(labelsize=6.5, length=2, pad=1)

    cax = fig.add_subplot(sub[2, :])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=levels)
    cb.ax.tick_params(labelsize=9, length=3, pad=2)


def make_rainfall_slide(
    ds: xr.Dataset,
    output_dir: Path,
    bbox: tuple[float, float, float, float],
    dpi: int,
    map_scale: str,
    draw_states: bool,
    india_outline_geoms: list,
    india_state_geoms: list,
    rainfall_scale: str,
) -> Path:
    actual_levels = np.asarray([0, 2, 5, 10, 20, 40], dtype=float)
    anomaly_levels = np.asarray([-20, -15, -10, -5, -2, 2, 5, 10, 15, 20], dtype=float)
    suffix = "rainfall"
    if rainfall_scale == "fuxi":
        actual_levels = np.asarray([0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20], dtype=float)
        anomaly_levels = np.asarray([-2, -1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1, 2], dtype=float)
        suffix = "rainfall_fuxiscale"

    fig = plt.figure(figsize=(12, 9), facecolor="white")
    slide_title(fig, "Predicted week wise rainfall (by FuXi-S2S)")
    outer = fig.add_gridspec(nrows=1, ncols=2, left=0.045, right=0.955, top=0.875, bottom=0.18, wspace=0.07)
    plot_2x2_block(
        fig,
        outer[0, 0],
        ds,
        ds.forecast_weekly.sel(variable="tp").values.astype("float32"),
        "FuXi-S2S Actual Rainfall (mm/day)",
        actual_levels,
        rainfall_actual_cmap(len(actual_levels) - 1),
        "both",
        bbox,
        map_scale,
        draw_states,
        india_outline_geoms,
        india_state_geoms,
        mask_india=False,
    )
    plot_2x2_block(
        fig,
        outer[0, 1],
        ds,
        ds.anomaly_weekly.sel(variable="tp").values.astype("float32"),
        "FuXi-S2S Rainfall Anomaly (mm/day)",
        anomaly_levels,
        rainfall_anomaly_cmap(),
        "both",
        bbox,
        map_scale,
        draw_states,
        india_outline_geoms,
        india_state_geoms,
        mask_india=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"fuxi_ppt_weekwise_{suffix}_{ds.attrs.get('ic_date', 'unknown')}.png"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"wrote {output}")
    return output


def make_t2m_slide(
    ds: xr.Dataset,
    product: str,
    output_dir: Path,
    bbox: tuple[float, float, float, float],
    dpi: int,
    map_scale: str,
    draw_states: bool,
    india_outline_geoms: list,
    india_state_geoms: list,
    mask_india: bool,
) -> Path:
    if product == "actual":
        title = "Predicted week wise 2m temperature actual (by FuXi-S2S)"
        variable_title = "FuXi-S2S 2m Temperature Actual (degC)"
        field = ds.forecast_weekly.sel(variable="t2m").values.astype("float32")
        levels = np.asarray([0, 10, 20, 25, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46], dtype=float)
        cmap = imd_temperature_actual_cmap("tmax")
        suffix = "t2m_actual"
    elif product == "anomaly":
        title = "Predicted week wise 2m temperature anomaly (by FuXi-S2S)"
        variable_title = "FuXi-S2S 2m Temperature Anomaly (degC)"
        field = ds.anomaly_weekly.sel(variable="t2m").values.astype("float32")
        levels = np.asarray([-10, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 10], dtype=float)
        cmap = imd_temperature_anomaly_cmap()
        suffix = "t2m_anomaly"
    else:
        raise ValueError(product)

    fig = plt.figure(figsize=(12, 9), facecolor="white")
    slide_title(fig, title)
    outer = fig.add_gridspec(nrows=1, ncols=1, left=0.18, right=0.82, top=0.875, bottom=0.18)
    plot_2x2_block(
        fig,
        outer[0, 0],
        ds,
        field,
        variable_title,
        levels,
        cmap,
        "both",
        bbox,
        map_scale,
        draw_states,
        india_outline_geoms,
        india_state_geoms,
        mask_india=mask_india,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"fuxi_ppt_weekwise_{suffix}_{ds.attrs.get('ic_date', 'unknown')}.png"
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"wrote {output}")
    return output


def main() -> int:
    args = parse_args()
    india_outline_geoms, india_state_geoms = prepare_india_geometries(
        args.map_scale,
        args.india_shapefile,
        args.district_shapefile,
        args.draw_districts,
    )
    with xr.open_dataset(args.analysis_file) as ds:
        make_rainfall_slide(
            ds,
            args.output_dir,
            tuple(args.bbox),
            args.dpi,
            args.map_scale,
            not args.no_state_lines,
            india_outline_geoms,
            india_state_geoms,
            args.rainfall_scale,
        )
        make_t2m_slide(
            ds,
            "actual",
            args.output_dir,
            tuple(args.bbox),
            args.dpi,
            args.map_scale,
            not args.no_state_lines,
            india_outline_geoms,
            india_state_geoms,
            not args.no_mask_to_india,
        )
        make_t2m_slide(
            ds,
            "anomaly",
            args.output_dir,
            tuple(args.bbox),
            args.dpi,
            args.map_scale,
            not args.no_state_lines,
            india_outline_geoms,
            india_state_geoms,
            not args.no_mask_to_india,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
