#!/usr/bin/env python3
"""Plot FuXi weekly forecast and anomaly maps from an analysis NetCDF."""

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

from plot_one_member_india_forecast import (
    DEFAULT_BBOX,
    Grid,
    add_map,
    mask_to_geometries,
    prepare_india_geometries,
)


PRODUCTS = ("tp_forecast", "tp_anomaly", "t2m_forecast", "t2m_anomaly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/weekly_analysis_plots"))
    parser.add_argument("--products", default=",".join(PRODUCTS))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--map-scale", default="50m", choices=("10m", "50m", "110m"))
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    parser.add_argument("--india-shapefile", type=Path)
    parser.add_argument("--district-shapefile", type=Path)
    parser.add_argument("--draw-districts", action="store_true")
    parser.add_argument("--no-state-lines", action="store_true")
    parser.add_argument("--no-mask-to-india", action="store_true")
    parser.add_argument("--rainfall-scale", choices=("fuxi", "imd"), default="fuxi")
    return parser.parse_args()


def week_label(init: datetime, week_index: int, start_day: int, end_day: int) -> str:
    start = init + timedelta(days=int(start_day))
    end = init + timedelta(days=int(end_day))
    return f"(Week{week_index}: 00Z{start:%d%b}-00Z{end:%d%b})"


def imd_green_cmap(ncolors: int) -> mcolors.Colormap:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "imd_rainfall",
        ["#ffffff", "#72f88b", "#20d83c", "#04aa18", "#00790a", "#004500"],
        N=ncolors,
    )
    cmap.set_under("#ffffff")
    cmap.set_over("#003000")
    return cmap


def imd_anomaly_cmap() -> mcolors.Colormap:
    colors = [
        "#d7301f",
        "#fc5b2b",
        "#fdae42",
        "#fee08b",
        "#ffffcc",
        "#ffffff",
        "#d8daeb",
        "#9e9ac8",
        "#5e5aa6",
        "#1d1a62",
    ]
    cmap = mcolors.ListedColormap(colors, name="imd_rainfall_anomaly")
    cmap.set_under("#b30000")
    cmap.set_over("#08083f")
    return cmap


def temp_anomaly_cmap() -> mcolors.Colormap:
    colors = [
        "#313695",
        "#5e63b6",
        "#9e9ac8",
        "#d8daeb",
        "#eeeeee",
        "#f7f7f7",
        "#fff7bc",
        "#fed976",
        "#fd8d3c",
        "#f03b20",
        "#bd0026",
        "#800026",
    ]
    cmap = mcolors.ListedColormap(colors, name="temperature_anomaly")
    cmap.set_under("#1f1f78")
    cmap.set_over("#67000d")
    return cmap


def product_style(product: str, rainfall_scale: str):
    if product == "tp_forecast":
        levels = (
            np.asarray([1, 5, 10, 20, 40], dtype=float)
            if rainfall_scale == "imd"
            else np.asarray([0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20], dtype=float)
        )
        return {
            "variable": "tp",
            "data": "forecast_weekly",
            "title": "Forecast Rainfall (mm/day)",
            "suffix": "tp_forecast",
            "levels": levels,
            "cmap": imd_green_cmap(len(levels) - 1),
            "extend": "both",
        }
    if product == "tp_anomaly":
        levels = (
            np.asarray([-15, -10, -7, -3, -1, 1, 3, 7, 10, 15], dtype=float)
            if rainfall_scale == "imd"
            else np.asarray([-2, -1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1, 2], dtype=float)
        )
        return {
            "variable": "tp",
            "data": "anomaly_weekly",
            "title": "Forecast Rainfall Anomaly (mm/day)",
            "suffix": "tp_anomaly",
            "levels": levels,
            "cmap": imd_anomaly_cmap(),
            "extend": "both",
        }
    if product == "t2m_forecast":
        return {
            "variable": "t2m",
            "data": "forecast_weekly",
            "title": "2m Temperature Actual (degC)",
            "suffix": "t2m_forecast",
            "levels": np.asarray([0, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46], dtype=float),
            "cmap": plt.get_cmap("YlOrRd", 17),
            "extend": "both",
        }
    if product == "t2m_anomaly":
        return {
            "variable": "t2m",
            "data": "anomaly_weekly",
            "title": "2m Temperature Anomaly (degC)",
            "suffix": "t2m_anomaly",
            "levels": np.asarray([-10, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 10], dtype=float),
            "cmap": temp_anomaly_cmap(),
            "extend": "both",
        }
    raise ValueError(f"unsupported product: {product}")


def plot_product(
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
    rainfall_scale: str,
) -> Path:
    style = product_style(product, rainfall_scale)
    ic_date = str(ds.attrs.get("ic_date", "unknown"))
    init = datetime.strptime(ic_date, "%Y%m%d")
    data = ds[style["data"]].sel(variable=style["variable"]).values.astype("float32")
    lat = ds.lat.values.astype("float32")
    lon = ds.lon.values.astype("float32")
    grid = Grid(lat=lat, lon=lon)

    if mask_india:
        data = np.stack([mask_to_geometries(field, grid, india_outline_geoms) for field in data], axis=0)

    levels = style["levels"]
    cmap = style["cmap"]
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    fig = plt.figure(figsize=(8.0, 11.2), facecolor="white")
    gs = fig.add_gridspec(nrows=4, ncols=2, height_ratios=[1, 1, 1, 0.075], hspace=0.32, wspace=0.15)
    fig.text(0.04, 0.965, f"{style['title']}  IC={ic_date}", color="#e33b3b", fontsize=14, fontweight="bold", ha="left")
    fig.text(0.5, 0.935, "FuXi-S2S ensemble mean | June-17 model climatology", color="#0026cc", fontsize=10.5, ha="center", fontweight="bold")

    mappable = None
    for week_idx in range(data.shape[0]):
        ax = fig.add_subplot(gs[week_idx // 2, week_idx % 2], projection=ccrs.PlateCarree())
        mappable = ax.contourf(
            lon,
            lat,
            data[week_idx],
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend=style["extend"],
            transform=ccrs.PlateCarree(),
        )
        add_map(ax, bbox, map_scale, draw_states, india_outline_geoms, india_state_geoms)
        ax.set_title(
            week_label(init, week_idx + 1, int(ds.week_start_day.values[week_idx]), int(ds.week_end_day.values[week_idx])),
            fontsize=10,
            color="#0a41ff",
            pad=4,
            fontweight="bold",
        )

    cax = fig.add_subplot(gs[3, :])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=levels)
    cb.ax.tick_params(labelsize=8, length=3)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"fuxi_{style['suffix']}_{ic_date}_6week.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")
    return output


def main() -> int:
    args = parse_args()
    products = [item.strip() for item in args.products.split(",") if item.strip()]
    unknown = sorted(set(products) - set(PRODUCTS))
    if unknown:
        raise SystemExit(f"unknown product(s): {','.join(unknown)}")
    india_outline_geoms, india_state_geoms = prepare_india_geometries(
        args.map_scale,
        args.india_shapefile,
        args.district_shapefile,
        args.draw_districts,
    )
    with xr.open_dataset(args.analysis_file) as ds:
        for product in products:
            plot_product(
                ds=ds,
                product=product,
                output_dir=args.output_dir,
                bbox=tuple(args.bbox),
                dpi=args.dpi,
                map_scale=args.map_scale,
                draw_states=not args.no_state_lines,
                india_outline_geoms=india_outline_geoms,
                india_state_geoms=india_state_geoms,
                mask_india=not args.no_mask_to_india,
                rainfall_scale=args.rainfall_scale,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
