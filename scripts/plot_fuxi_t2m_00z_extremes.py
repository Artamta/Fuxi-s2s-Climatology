#!/usr/bin/env python3
"""Plot weekly min/max diagnostics from FuXi daily 00Z t2m snapshots."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from plot_fuxi_weekly_analysis import imd_temperature_actual_cmap, legacy_temp_anomaly_cmap, week_label
from plot_one_member_india_forecast import DEFAULT_BBOX, Grid, add_map, mask_to_geometries, prepare_india_geometries


DEFAULT_EXTREMES = Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_t2m_00z_extremes_20260617.nc")
PRODUCTS = ("t2m_00z_min_actual", "t2m_00z_min_anomaly", "t2m_00z_max_actual", "t2m_00z_max_anomaly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extremes_file", type=Path, nargs="?", default=DEFAULT_EXTREMES)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/temp_6week_00z_extremes"))
    parser.add_argument("--products", default=",".join(PRODUCTS))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--map-scale", default="50m", choices=("10m", "50m", "110m"))
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    parser.add_argument("--india-shapefile", type=Path)
    parser.add_argument("--district-shapefile", type=Path)
    parser.add_argument("--draw-districts", action="store_true")
    parser.add_argument("--no-state-lines", action="store_true")
    parser.add_argument("--no-mask-to-india", action="store_true")
    return parser.parse_args()


def product_style(product: str):
    if product == "t2m_00z_min_actual":
        return {
            "data": "forecast_00z_min",
            "title": "2m Temperature 00Z Weekly Min (degC)",
            "suffix": "t2m_00z_min_actual",
            "levels": np.asarray([0, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32], dtype=float),
            "cmap": imd_temperature_actual_cmap("tmin"),
            "extend": "both",
        }
    if product == "t2m_00z_max_actual":
        return {
            "data": "forecast_00z_max",
            "title": "2m Temperature 00Z Weekly Max (degC)",
            "suffix": "t2m_00z_max_actual",
            "levels": np.asarray([26, 28, 30, 32, 34, 36, 38, 39, 40, 41, 42, 43, 44, 45], dtype=float),
            "cmap": imd_temperature_actual_cmap("tmax"),
            "extend": "both",
        }
    if product == "t2m_00z_min_anomaly":
        return {
            "data": "anomaly_00z_min",
            "title": "2m Temperature 00Z Weekly Min Anomaly (degC)",
            "suffix": "t2m_00z_min_anomaly",
            "levels": np.asarray([-10, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 10], dtype=float),
            "cmap": legacy_temp_anomaly_cmap(),
            "extend": "both",
        }
    if product == "t2m_00z_max_anomaly":
        return {
            "data": "anomaly_00z_max",
            "title": "2m Temperature 00Z Weekly Max Anomaly (degC)",
            "suffix": "t2m_00z_max_anomaly",
            "levels": np.asarray([-10, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 10], dtype=float),
            "cmap": legacy_temp_anomaly_cmap(),
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
) -> Path:
    style = product_style(product)
    ic_date = str(ds.attrs.get("ic_date", "unknown"))
    init = datetime.strptime(ic_date, "%Y%m%d")
    data = ds[style["data"]].values.astype("float32")
    lat = ds.lat.values.astype("float32")
    lon = ds.lon.values.astype("float32")
    grid = Grid(lat=lat, lon=lon)

    if mask_india:
        data = np.stack([mask_to_geometries(field, grid, india_outline_geoms) for field in data], axis=0)

    levels = style["levels"]
    cmap = style["cmap"]
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    fig = plt.figure(figsize=(8.4, 11.4), facecolor="white")
    gs = fig.add_gridspec(nrows=4, ncols=2, height_ratios=[1, 1, 1, 0.09], hspace=0.34, wspace=0.15)
    fig.text(0.04, 0.965, f"{style['title']}  IC={ic_date}", color="#e33b3b", fontsize=14, fontweight="bold", ha="left")
    fig.text(0.5, 0.935, "FuXi-S2S ensemble mean | 00Z t2m proxy, not true Tmin/Tmax", color="#0026cc", fontsize=10.5, ha="center", fontweight="bold")

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
    cb.ax.tick_params(labelsize=10, length=4, pad=3)
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
    with xr.open_dataset(args.extremes_file) as ds:
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
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
