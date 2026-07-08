#!/usr/bin/env python3
"""Quick India weekly maps for one FuXi-S2S member."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from netCDF4 import Dataset
from pyproj import CRS, Transformer
from shapely import contains_xy
from shapely.ops import transform, unary_union


WEEKS = ((1, 7), (8, 14), (15, 21), (22, 28), (29, 35), (36, 42))
DEFAULT_BBOX = (66.5, 98.5, 5.0, 38.5)  # lon_min, lon_max, lat_min, lat_max
DEFAULT_STATE_SHAPEFILE = Path("/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp")
DEFAULT_DISTRICT_SHAPEFILE = Path("/storage/raj.ayush/archive/s2s-forecast-/DISTRICT_BOUNDARY.shp")
KNOWN_RAW_ROOTS = (
    Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/raw"),
    Path("/storage/raj.ayush/All_Model_Data/fuxi/test/raw"),
    Path("/storage/raj.ayush/All_Model_Data/fuxi/op2026_ens50/raw"),
    Path("/storage/raj.ayush/All_Model_Data/fuxi/jjas2026_ens50/raw"),
    Path("/storage/raj.ayush/All_Model_Data/fuxi/jjas2019/raw"),
)


@dataclass(frozen=True)
class Grid:
    lat: np.ndarray
    lon: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic-date", help="IC date as YYYYMMDD. Used for auto-search if --raw-dir is omitted.")
    parser.add_argument("--raw-dir", type=Path, help="Path like .../raw/YYYYMMDD containing member/MM/SS.nc")
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument("--variables", default="tp,t2m", help="Comma-separated variables: tp,t2m")
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quick_plots"))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--map-scale", default="50m", choices=("10m", "50m", "110m"))
    parser.add_argument("--no-state-lines", action="store_true")
    parser.add_argument("--india-shapefile", type=Path, help="Optional India/states shapefile to draw and mask against.")
    parser.add_argument("--district-shapefile", type=Path, help="Optional district shapefile for internal boundaries.")
    parser.add_argument("--draw-districts", action="store_true")
    parser.add_argument("--no-mask-to-india", action="store_true", help="Shade full domain instead of only India.")
    parser.add_argument("--tp-scale", type=float, default=1.0, help="Optional multiplier for tp. Default leaves FuXi units unchanged.")
    return parser.parse_args()


def find_raw_dir(ic_date: str | None, raw_dir: Path | None) -> Path:
    if raw_dir is not None:
        if not raw_dir.is_dir():
            raise SystemExit(f"raw dir not found: {raw_dir}")
        return raw_dir
    if not ic_date:
        raise SystemExit("Provide --ic-date or --raw-dir")
    candidates = [root / ic_date for root in KNOWN_RAW_ROOTS]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise SystemExit("Could not auto-find raw dir. Tried:\n" + "\n".join(str(item) for item in candidates))


def coord_slice(values: np.ndarray, lower: float, upper: float) -> slice:
    mask = (values >= lower) & (values <= upper)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"No coordinate values inside [{lower}, {upper}]")
    return slice(int(idx[0]), int(idx[-1]) + 1)


def read_shapefile_geometries(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"shapefile not found: {path}")
    reader = shpreader.Reader(str(path))
    geoms = list(reader.geometries())
    prj = path.with_suffix(".prj")
    if not prj.exists():
        return geoms

    source_crs = CRS.from_wkt(prj.read_text())
    if source_crs.is_geographic:
        return geoms

    transformer = Transformer.from_crs(source_crs, CRS.from_epsg(4326), always_xy=True)
    return [transform(transformer.transform, geom) for geom in geoms]


def natural_earth_india_geometries(map_scale: str) -> tuple[list, list]:
    countries_path = shpreader.natural_earth(
        resolution=map_scale,
        category="cultural",
        name="admin_0_countries",
    )
    country_geoms = []
    for record in shpreader.Reader(countries_path).records():
        attrs = record.attributes
        if attrs.get("ADMIN") == "India" or attrs.get("ADM0_A3") == "IND":
            country_geoms.append(record.geometry)

    states_path = shpreader.natural_earth(
        resolution=map_scale,
        category="cultural",
        name="admin_1_states_provinces_lines",
    )
    state_geoms = []
    for record in shpreader.Reader(states_path).records():
        attrs = record.attributes
        if (
            attrs.get("admin") == "India"
            or attrs.get("adm0_a3") == "IND"
            or attrs.get("ADM0_NAME") == "India"
            or attrs.get("ADM0_A3") == "IND"
        ):
            state_geoms.append(record.geometry)
    return country_geoms, state_geoms


def prepare_india_geometries(
    map_scale: str,
    india_shapefile: Path | None,
    district_shapefile: Path | None,
    draw_districts: bool,
) -> tuple[list, list]:
    boundary_path = india_shapefile
    if boundary_path is None and DEFAULT_STATE_SHAPEFILE.exists():
        boundary_path = DEFAULT_STATE_SHAPEFILE
    if boundary_path is not None:
        state_geoms = read_shapefile_geometries(boundary_path)
        outline_geoms = [unary_union(state_geoms)]
        if draw_districts:
            district_path = district_shapefile
            if district_path is None and DEFAULT_DISTRICT_SHAPEFILE.exists():
                district_path = DEFAULT_DISTRICT_SHAPEFILE
            if district_path is not None:
                return outline_geoms, read_shapefile_geometries(district_path)
        return outline_geoms, state_geoms
    return natural_earth_india_geometries(map_scale)


def mask_to_geometries(field: np.ndarray, grid: Grid, geometries: list) -> np.ndarray:
    if not geometries:
        return field
    union = unary_union(geometries)
    lon2d, lat2d = np.meshgrid(grid.lon, grid.lat)
    inside = contains_xy(union, lon2d, lat2d)
    return np.where(inside, field, np.nan).astype(np.float32)


def read_channel(path: Path, variable: str, bbox: tuple[float, float, float, float]) -> tuple[np.ndarray, Grid]:
    lon_min, lon_max, lat_min, lat_max = bbox
    with Dataset(path) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float32)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float32)
        lat_sel = coord_slice(lat, lat_min, lat_max)
        lon_sel = coord_slice(lon, lon_min, lon_max)
        channels = np.asarray(ds.variables["channel"][:]).astype(str)
        matches = np.flatnonzero(channels == variable)
        if matches.size == 0:
            raise ValueError(f"{variable} not found in {path}")
        var = ds.variables["__xarray_dataarray_variable__"]
        arr = np.asarray(var[0, 0, int(matches[0]), lat_sel, lon_sel], dtype=np.float32)
        grid = Grid(lat=lat[lat_sel], lon=lon[lon_sel])
    return arr, grid


def weekly_mean(
    raw_dir: Path,
    member: int,
    variable: str,
    week: tuple[int, int],
    bbox: tuple[float, float, float, float],
    tp_scale: float,
) -> tuple[np.ndarray, Grid]:
    acc = None
    grid = None
    for step in range(week[0], week[1] + 1):
        path = raw_dir / "member" / f"{member:02d}" / f"{step:02d}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        arr, grid = read_channel(path, variable, bbox)
        if variable == "t2m" and np.nanmean(arr) > 100:
            arr = arr - np.float32(273.15)
        if variable == "tp":
            arr = arr * np.float32(tp_scale)
        if acc is None:
            acc = np.zeros_like(arr, dtype=np.float32)
        acc += arr
    return acc / np.float32(week[1] - week[0] + 1), grid


def week_label(init: datetime, week_index: int, week: tuple[int, int]) -> str:
    start = init + timedelta(days=week[0])
    end = init + timedelta(days=week[1])
    return f"(Week{week_index}: {start:%d%b}-{end:%d%b})"


def add_map(
    ax: plt.Axes,
    bbox: tuple[float, float, float, float],
    map_scale: str,
    draw_states: bool,
    india_outline_geoms: list,
    india_state_geoms: list,
) -> None:
    lon_min, lon_max, lat_min, lat_max = bbox
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.coastlines(map_scale, linewidth=0.8, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale(map_scale), linewidth=0.55, edgecolor="0.35")
    if india_outline_geoms:
        ax.add_geometries(
            india_outline_geoms,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=1.15,
            zorder=6,
        )
    if draw_states:
        if india_state_geoms:
            ax.add_geometries(
                india_state_geoms,
                crs=ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="0.10",
                linewidth=0.28,
                alpha=0.8,
                zorder=7,
            )
    ax.set_xticks([70, 77, 84, 91], crs=ccrs.PlateCarree())
    ax.set_yticks([10, 15, 20, 25, 30, 35], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f"))
    ax.tick_params(labelsize=9, length=3, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_edgecolor("black")


def variable_style(variable: str) -> tuple[str, str, np.ndarray, mcolors.Colormap, str]:
    if variable == "tp":
        levels = np.array([0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20], dtype=float)
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "tp_green",
            ["#ffffff", "#b8ffbe", "#63ed7a", "#18bf35", "#078212", "#004400"],
            N=len(levels) - 1,
        )
        cmap.set_under("#ffffff")
        cmap.set_over("#002d00")
        return "Forecast Rainfall (mm/day)", "tp", levels, cmap, "Forecast Rainfall"
    if variable == "t2m":
        levels = np.array([0, 4, 8, 12, 16, 20, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46], dtype=float)
        cmap = plt.get_cmap("YlOrRd", len(levels) - 1)
        cmap.set_under("#fff7bc")
        cmap.set_over("#67000d")
        return "2m Temperature Actual (degC)", "t2m", levels, cmap, "2m Temperature Actual"
    raise ValueError(f"unsupported variable: {variable}")


def plot_variable(
    raw_dir: Path,
    ic_date: str,
    member: int,
    variable: str,
    output_dir: Path,
    bbox: tuple[float, float, float, float],
    dpi: int,
    map_scale: str,
    draw_states: bool,
    india_outline_geoms: list,
    india_state_geoms: list,
    mask_india: bool,
    tp_scale: float,
) -> Path:
    title, suffix, levels, cmap, super_title = variable_style(variable)
    norm = mcolors.BoundaryNorm(levels, cmap.N)
    init = datetime.strptime(ic_date, "%Y%m%d")

    fields = []
    grid = None
    for week in WEEKS:
        field, grid = weekly_mean(raw_dir, member, variable, week, bbox, tp_scale)
        if mask_india:
            field = mask_to_geometries(field, grid, india_outline_geoms)
        fields.append(field)

    fig = plt.figure(figsize=(8.0, 11.2), facecolor="white")
    gs = fig.add_gridspec(nrows=4, ncols=2, height_ratios=[1, 1, 1, 0.075], hspace=0.32, wspace=0.15)
    fig.text(0.04, 0.965, f"{title}  IC={ic_date}", color="#e33b3b", fontsize=14, fontweight="bold", ha="left")
    fig.text(0.5, 0.935, f"FuXi-S2S member {member:02d} | weekly mean", color="#0026cc", fontsize=11, ha="center", fontweight="bold")

    mappable = None
    for idx, field in enumerate(fields, start=1):
        row = (idx - 1) // 2
        col = (idx - 1) % 2
        ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
        mappable = ax.contourf(
            grid.lon,
            grid.lat,
            field,
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both" if variable == "t2m" else "max",
            transform=ccrs.PlateCarree(),
        )
        add_map(ax, bbox, map_scale, draw_states, india_outline_geoms, india_state_geoms)
        ax.set_title(week_label(init, idx, WEEKS[idx - 1]), fontsize=10, color="#0a41ff", pad=4, fontweight="bold")

    cax = fig.add_subplot(gs[3, :])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", ticks=levels)
    cb.ax.tick_params(labelsize=8, length=3)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"fuxi_{suffix}_member{member:02d}_{ic_date}_india_weekly_6panel.png"
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {output}")
    return output


def main() -> int:
    args = parse_args()
    raw_dir = find_raw_dir(args.ic_date, args.raw_dir)
    ic_date = args.ic_date or raw_dir.name
    variables = [item.strip() for item in args.variables.split(",") if item.strip()]
    india_outline_geoms, india_state_geoms = prepare_india_geometries(
        args.map_scale,
        args.india_shapefile,
        args.district_shapefile,
        args.draw_districts,
    )
    print(f"raw dir : {raw_dir}")
    print(f"ic date : {ic_date}")
    print(f"member  : {args.member:02d}")
    print(f"india mask: {not args.no_mask_to_india}")
    for variable in variables:
        plot_variable(
            raw_dir=raw_dir,
            ic_date=ic_date,
            member=args.member,
            variable=variable,
            output_dir=args.output_dir,
            bbox=tuple(args.bbox),
            dpi=args.dpi,
            map_scale=args.map_scale,
            draw_states=not args.no_state_lines,
            india_outline_geoms=india_outline_geoms,
            india_state_geoms=india_state_geoms,
            mask_india=not args.no_mask_to_india,
            tp_scale=args.tp_scale,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
