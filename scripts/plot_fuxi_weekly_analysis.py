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


PRODUCTS = ("tp_forecast", "tp_actual", "tp_anomaly", "t2m_forecast", "t2m_anomaly")


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
    parser.add_argument("--temperature-actual-scale", choices=("tmax", "tmin", "legacy"), default="tmax")
    parser.add_argument("--weeks", help="Week numbers to plot, for example 1-4 or 1,2,3,4. Default plots all weeks.")
    parser.add_argument(
        "--product-title",
        action="append",
        default=[],
        metavar="PRODUCT=TITLE",
        help="Override the top title for a product, for example tp_actual='FuXi S2S Forecast'.",
    )
    return parser.parse_args()


def week_label(init: datetime, week_index: int, start_day: int, end_day: int) -> str:
    start = init + timedelta(days=int(start_day))
    end = init + timedelta(days=int(end_day))
    return f"(Week{week_index}: 00Z{start:%d%b}-00Z{end:%d%b})"


def parse_week_indices(raw_weeks: str | None, available_weeks: np.ndarray) -> np.ndarray:
    if not raw_weeks:
        return np.arange(len(available_weeks), dtype=np.int32)

    requested: list[int] = []
    for chunk in raw_weeks.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"week range must increase: {item}")
            requested.extend(range(start, end + 1))
        else:
            requested.append(int(item))

    if not requested:
        raise ValueError("--weeks did not contain any week numbers")

    week_to_index = {int(week): idx for idx, week in enumerate(available_weeks)}
    missing = [week for week in requested if week not in week_to_index]
    if missing:
        available_text = ",".join(str(int(item)) for item in available_weeks)
        missing_text = ",".join(str(item) for item in missing)
        raise ValueError(f"requested week(s) {missing_text} not available; available weeks: {available_text}")

    # Preserve order while removing duplicates.
    deduped = list(dict.fromkeys(requested))
    return np.asarray([week_to_index[week] for week in deduped], dtype=np.int32)


def parse_title_overrides(entries: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--product-title must use PRODUCT=TITLE format: {entry}")
        product, title = entry.split("=", 1)
        product = product.strip()
        title = title.strip()
        if product not in PRODUCTS:
            raise ValueError(f"unknown product title override: {product}")
        if not title:
            raise ValueError(f"empty title override for {product}")
        overrides[product] = title
    return overrides


def week_output_tag(week_numbers: np.ndarray) -> str:
    weeks = [int(item) for item in week_numbers]
    if weeks == list(range(1, len(weeks) + 1)):
        return f"{len(weeks)}week"
    return "weeks" + "-".join(str(item) for item in weeks)


def forecast_member_label(ds: xr.Dataset) -> str:
    members_raw = str(ds.attrs.get("forecast_members", "")).strip()
    members = [item.strip() for item in members_raw.split(",") if item.strip()]
    return f"{len(members)}-member" if members else "ensemble"


def climatology_label(ds: xr.Dataset) -> str:
    years_raw = str(ds.attrs.get("climatology_years", "")).strip()
    climatology_file = ds.attrs.get("climatology_file")
    if not years_raw and climatology_file:
        path = Path(str(climatology_file))
        if path.exists():
            with xr.open_dataset(path) as clim:
                years_raw = str(clim.attrs.get("years", "")).strip()

    years = [int(item.strip()) for item in years_raw.split(",") if item.strip().isdigit()]
    if years:
        count = len(years)
        year_range = f"{min(years)}-{max(years)}"
        return f"{count}-year FuXi S2S climatology ({year_range})"
    return "FuXi S2S climatology"


def product_subtitle_and_note(product: str, ds: xr.Dataset) -> tuple[str, str]:
    member_label = forecast_member_label(ds)
    clim_label = climatology_label(ds)

    if product in {"tp_forecast", "tp_actual"}:
        return (
            "FuXi S2S ensemble mean",
            f"Weekly mean rainfall rate (mm/day): {member_label} ensemble mean averaged over each 7-day lead week.",
        )
    if product == "tp_anomaly":
        return (
            f"FuXi S2S ensemble mean anomaly vs {clim_label}",
            f"Weekly mean anomaly (mm/day): {member_label} forecast weekly mean minus {clim_label}.",
        )
    if product == "t2m_forecast":
        return (
            "FuXi S2S ensemble mean",
            f"Weekly mean 2m temperature (degC): {member_label} ensemble mean averaged over each 7-day lead week.",
        )
    if product == "t2m_anomaly":
        return (
            f"FuXi S2S ensemble mean anomaly vs {clim_label}",
            f"Weekly mean 2m temperature anomaly (degC): {member_label} forecast weekly mean minus {clim_label}.",
        )
    raise ValueError(f"unsupported product: {product}")


def listed_cmap(name: str, colors: list[str], under: str | None = None, over: str | None = None) -> mcolors.Colormap:
    cmap = mcolors.ListedColormap(colors, name=name)
    if under is not None:
        cmap.set_under(under)
    if over is not None:
        cmap.set_over(over)
    return cmap


def imd_green_cmap(ncolors: int) -> mcolors.Colormap:
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "imd_rainfall",
        ["#8affa1", "#4ef45b", "#12cc15", "#009600", "#007800"],
        N=ncolors,
    )
    cmap.set_under("#ffffff")
    cmap.set_over("#007800")
    return cmap


def imd_rainfall_actual_cmap() -> mcolors.Colormap:
    return listed_cmap(
        "imd_rainfall_actual",
        ["#ffffff", "#8affa1", "#4ef45b", "#12cc15", "#009600"],
        under="#ffffff",
        over="#007800",
    )


def imd_anomaly_cmap() -> mcolors.Colormap:
    return listed_cmap(
        "imd_rainfall_anomaly",
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


def imd_temperature_actual_cmap(scale: str) -> mcolors.Colormap:
    if scale == "tmin":
        return listed_cmap(
            "imd_tmin_actual",
            [
                "#bebee2",
                "#fffacd",
                "#fff191",
                "#ffe271",
                "#ffca59",
                "#ffa635",
                "#ff8e1d",
                "#ff6a00",
                "#ff3a00",
                "#eb1800",
                "#c30400",
                "#9b0000",
                "#730000",
            ],
            under="#9696c6",
            over="#5f0000",
        )
    if scale == "tmax":
        return listed_cmap(
            "imd_tmax_actual",
            [
                "#fffacd",
                "#fff4a5",
                "#ffee7d",
                "#ffd665",
                "#ffbe4d",
                "#ffa635",
                "#ff8e1d",
                "#ff7605",
                "#ff5200",
                "#eb1800",
                "#c30400",
                "#9b0000",
                "#730000",
            ],
            under="#aaaad4",
            over="#5f0000",
        )
    cmap = plt.get_cmap("YlOrRd", 17)
    cmap.set_under("#fff7bc")
    cmap.set_over("#67000d")
    return cmap


def imd_temperature_anomaly_cmap() -> mcolors.Colormap:
    return listed_cmap(
        "imd_temperature_anomaly",
        [
            "#383880",
            "#5a5a9c",
            "#8282b8",
            "#aaaad4",
            "#c8c8e9",
            "#dcdcf7",
            "#fffacd",
            "#ffa635",
            "#ff7605",
            "#ff5e00",
            "#ff2e00",
            "#c30400",
        ],
        under="#10103a",
        over="#730000",
    )


def legacy_temp_anomaly_cmap() -> mcolors.Colormap:
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
    cmap = mcolors.ListedColormap(colors, name="legacy_temperature_anomaly")
    cmap.set_under("#1f1f78")
    cmap.set_over("#67000d")
    return cmap


def product_style(product: str, rainfall_scale: str, temperature_actual_scale: str):
    if product in {"tp_forecast", "tp_actual"}:
        levels = (
            np.asarray([0, 2, 5, 10, 20, 40], dtype=float)
            if rainfall_scale == "imd"
            else np.asarray([0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20], dtype=float)
        )
        return {
            "variable": "tp",
            "data": "forecast_weekly",
            "title": "FuXi S2S Forecast",
            "suffix": "tp_actual" if product == "tp_actual" else "tp_forecast",
            "levels": levels,
            "cmap": imd_rainfall_actual_cmap() if rainfall_scale == "imd" else imd_green_cmap(len(levels) - 1),
            "extend": "both",
        }
    if product == "tp_anomaly":
        levels = (
            np.asarray([-20, -15, -10, -5, -2, 2, 5, 10, 15, 20], dtype=float)
            if rainfall_scale == "imd"
            else np.asarray([-2, -1, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1, 2], dtype=float)
        )
        return {
            "variable": "tp",
            "data": "anomaly_weekly",
            "title": "FuXi S2S Anomaly",
            "suffix": "tp_anomaly",
            "levels": levels,
            "cmap": imd_anomaly_cmap(),
            "extend": "both",
        }
    if product == "t2m_forecast":
        if temperature_actual_scale == "tmin":
            levels = np.asarray([0, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32], dtype=float)
        elif temperature_actual_scale == "tmax":
            levels = np.asarray([26, 28, 30, 32, 34, 36, 38, 39, 40, 41, 42, 43, 44, 45], dtype=float)
        else:
            levels = np.asarray([0, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46], dtype=float)
        return {
            "variable": "t2m",
            "data": "forecast_weekly",
            "title": "2m Temperature Actual (degC)",
            "suffix": "t2m_forecast",
            "levels": levels,
            "cmap": imd_temperature_actual_cmap(temperature_actual_scale),
            "extend": "both",
        }
    if product == "t2m_anomaly":
        if temperature_actual_scale == "legacy":
            levels = np.asarray([-10, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 10], dtype=float)
            cmap = legacy_temp_anomaly_cmap()
        else:
            levels = np.asarray([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6], dtype=float)
            cmap = imd_temperature_anomaly_cmap()
        return {
            "variable": "t2m",
            "data": "anomaly_weekly",
            "title": "2m Temperature Anomaly (degC)",
            "suffix": "t2m_anomaly",
            "levels": levels,
            "cmap": cmap,
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
    temperature_actual_scale: str,
    week_indices: np.ndarray,
    title_overrides: dict[str, str],
) -> Path:
    style = product_style(product, rainfall_scale, temperature_actual_scale)
    ic_date = str(ds.attrs.get("ic_date", "unknown"))
    init = datetime.strptime(ic_date, "%Y%m%d")
    data = ds[style["data"]].sel(variable=style["variable"]).values.astype("float32")[week_indices]
    week_numbers = ds.week.values[week_indices]
    week_start_day = ds.week_start_day.values[week_indices]
    week_end_day = ds.week_end_day.values[week_indices]
    lat = ds.lat.values.astype("float32")
    lon = ds.lon.values.astype("float32")
    grid = Grid(lat=lat, lon=lon)

    if mask_india:
        data = np.stack([mask_to_geometries(field, grid, india_outline_geoms) for field in data], axis=0)

    levels = style["levels"]
    cmap = style["cmap"]
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    panel_count = data.shape[0]
    panel_rows = int(np.ceil(panel_count / 2))
    fig_height = 3.45 * panel_rows + 1.05
    fig = plt.figure(figsize=(8.4, fig_height), facecolor="white")
    gs = fig.add_gridspec(nrows=panel_rows + 1, ncols=2, height_ratios=[1] * panel_rows + [0.09], hspace=0.34, wspace=0.15)
    title = title_overrides.get(product, style["title"])
    subtitle, note = product_subtitle_and_note(product, ds)
    fig.text(0.04, 0.965, f"{title}  IC={ic_date}", color="#e33b3b", fontsize=14, fontweight="bold", ha="left")
    fig.text(0.5, 0.935, subtitle, color="#0026cc", fontsize=10.0, ha="center", fontweight="bold")
    fig.text(0.5, 0.018, note, color="#343434", fontsize=7.4, ha="center")

    mappable = None
    for plot_idx in range(panel_count):
        ax = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2], projection=ccrs.PlateCarree())
        mappable = ax.contourf(
            lon,
            lat,
            data[plot_idx],
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend=style["extend"],
            transform=ccrs.PlateCarree(),
        )
        add_map(ax, bbox, map_scale, draw_states, india_outline_geoms, india_state_geoms)
        ax.set_title(
            week_label(init, int(week_numbers[plot_idx]), int(week_start_day[plot_idx]), int(week_end_day[plot_idx])),
            fontsize=10,
            color="#0a41ff",
            pad=4,
            fontweight="bold",
        )

    cax = fig.add_subplot(gs[panel_rows, :])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=levels)
    cb.ax.tick_params(labelsize=10, length=4, pad=3)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"fuxi_{style['suffix']}_{ic_date}_{week_output_tag(week_numbers)}.png"
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
    try:
        title_overrides = parse_title_overrides(args.product_title)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    india_outline_geoms, india_state_geoms = prepare_india_geometries(
        args.map_scale,
        args.india_shapefile,
        args.district_shapefile,
        args.draw_districts,
    )
    with xr.open_dataset(args.analysis_file) as ds:
        try:
            week_indices = parse_week_indices(args.weeks, ds.week.values)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
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
                temperature_actual_scale=args.temperature_actual_scale,
                week_indices=week_indices,
                title_overrides=title_overrides,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
