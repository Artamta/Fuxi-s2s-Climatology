#!/usr/bin/env python3
"""Build a reusable FuXi-S2S June-17 model climatology.

The output stores daily lead means and six weekly means averaged across all
requested hindcast years and ensemble members.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset


WEEKS = ((1, 7), (8, 14), (15, 21), (22, 28), (29, 35), (36, 42))
DEFAULT_RAW_ROOT = Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/raw")
DEFAULT_OUTPUT = Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/climatology_fuxi17june.nc")
DEFAULT_BBOX = (66.5, 98.5, 5.0, 38.5)  # lon_min, lon_max, lat_min, lat_max
DATA_VAR = "__xarray_dataarray_variable__"


def parse_ints(value: str) -> list[int]:
    if ":" in value:
        start, end = value.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_variables(value: str) -> list[str]:
    variables = [item.strip() for item in value.split(",") if item.strip()]
    if not variables:
        raise argparse.ArgumentTypeError("at least one variable is required")
    return variables


def coord_slice(values: np.ndarray, lower: float, upper: float) -> slice:
    mask = (values >= lower) & (values <= upper)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"No coordinate values inside [{lower}, {upper}]")
    return slice(int(idx[0]), int(idx[-1]) + 1)


def convert_units(name: str, values: np.ndarray) -> np.ndarray:
    out = values.astype("float32", copy=False)
    if name == "t2m" and float(np.nanmean(out)) > 100.0:
        out = out - np.float32(273.15)
    if name == "tp":
        out = np.clip(out, 0.0, None)
    return out.astype("float32", copy=False)


def read_metadata(path: Path, variables: list[str], bbox: tuple[float, float, float, float]):
    lon_min, lon_max, lat_min, lat_max = bbox
    with Dataset(path) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float32)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float32)
        lat_sel = coord_slice(lat, lat_min, lat_max)
        lon_sel = coord_slice(lon, lon_min, lon_max)
        channels = np.asarray(ds.variables["channel"][:]).astype(str)
        channel_indices = []
        missing = []
        for variable in variables:
            matches = np.flatnonzero(channels == variable)
            if matches.size == 0:
                missing.append(variable)
            else:
                channel_indices.append(int(matches[0]))
        if missing:
            raise ValueError(f"{path}: missing channel(s): {','.join(missing)}")
        return channel_indices, lat[lat_sel], lon[lon_sel], lat_sel, lon_sel


def sample_path(raw_root: Path, year: int, member: int, lead_day: int) -> Path:
    return raw_root / f"{year}0617" / "member" / f"{member:02d}" / f"{lead_day:02d}.nc"


def accumulate_year_member(task):
    raw_root, year, member, variables, bbox, lead_days = task
    first = sample_path(raw_root, year, member, lead_days[0])
    channel_indices, lat, lon, lat_sel, lon_sel = read_metadata(first, variables, bbox)
    sums = np.zeros((len(variables), len(lead_days), len(lat), len(lon)), dtype=np.float64)
    counts = np.zeros((len(lead_days),), dtype=np.int32)

    for lead_idx, lead_day in enumerate(lead_days):
        path = sample_path(raw_root, year, member, lead_day)
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        with Dataset(path) as ds:
            values = np.asarray(ds.variables[DATA_VAR][0, 0, channel_indices, lat_sel, lon_sel], dtype=np.float32)
        if values.ndim == 2:
            values = values[np.newaxis, :, :]
        for var_idx, variable in enumerate(variables):
            sums[var_idx, lead_idx] += convert_units(variable, values[var_idx])
        counts[lead_idx] += 1

    return year, member, sums, counts, lat, lon


def write_climatology(
    output: Path,
    variables: list[str],
    years: list[int],
    members: list[int],
    bbox: tuple[float, float, float, float],
    lat: np.ndarray,
    lon: np.ndarray,
    daily_mean: np.ndarray,
    daily_counts: np.ndarray,
    raw_root: Path,
) -> Path:
    lead_days = np.arange(1, daily_mean.shape[1] + 1, dtype=np.int32)
    week_numbers = np.arange(1, len(WEEKS) + 1, dtype=np.int32)
    week_start = np.asarray([item[0] for item in WEEKS], dtype=np.int32)
    week_end = np.asarray([item[1] for item in WEEKS], dtype=np.int32)
    weekly_mean = np.stack(
        [daily_mean[:, start - 1 : end].mean(axis=1) for start, end in WEEKS],
        axis=1,
    ).astype("float32")
    weekly_counts = np.asarray([daily_counts[start - 1 : end].sum() for start, end in WEEKS], dtype=np.int32)

    ds = xr.Dataset(
        data_vars={
            "daily_mean": (("variable", "lead_day", "lat", "lon"), daily_mean.astype("float32")),
            "weekly_mean": (("variable", "week", "lat", "lon"), weekly_mean),
            "daily_sample_count": (("lead_day",), daily_counts.astype("int32")),
            "weekly_sample_count": (("week",), weekly_counts),
            "week_start_day": (("week",), week_start),
            "week_end_day": (("week",), week_end),
        },
        coords={
            "variable": variables,
            "lead_day": lead_days,
            "week": week_numbers,
            "lat": lat.astype("float32"),
            "lon": lon.astype("float32"),
        },
        attrs={
            "title": "FuXi-S2S June-17 model climatology",
            "ic_mmdd": "0617",
            "years": ",".join(str(item) for item in years),
            "members": ",".join(f"{item:02d}" for item in members),
            "raw_root": str(raw_root),
            "bbox_lon_min_lon_max_lat_min_lat_max": ",".join(str(item) for item in bbox),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "units_tp": "mm/day, FuXi output convention",
            "units_t2m": "degC",
        },
    )
    ds["daily_mean"].attrs["description"] = "Mean by lead day across hindcast years and members."
    ds["weekly_mean"].attrs["description"] = "Mean by 7-day week across hindcast years, members, and lead days."

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    encoding = {
        "daily_mean": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
        "weekly_mean": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
    }
    ds.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
    tmp.replace(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--years", default="2002:2021")
    parser.add_argument("--members", default="0:10")
    parser.add_argument("--variables", type=parse_variables, default=parse_variables("tp,t2m"))
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    years = parse_ints(args.years)
    members = parse_ints(args.members)
    lead_days = list(range(1, 43))
    tasks = [(args.raw_root, year, member, args.variables, tuple(args.bbox), lead_days) for year in years for member in members]
    print(f"raw root  : {args.raw_root}")
    print(f"years     : {years[0]}-{years[-1]} ({len(years)})")
    print(f"members   : {members[0]:02d}-{members[-1]:02d} ({len(members)})")
    print(f"variables : {','.join(args.variables)}")
    print(f"workers   : {args.workers}")
    print(f"samples   : {len(tasks)} member-years x {len(lead_days)} lead days")

    daily_sum = None
    daily_counts = np.zeros((len(lead_days),), dtype=np.int32)
    ref_lat = ref_lon = None
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(accumulate_year_member, task) for task in tasks]
        for future in as_completed(futures):
            year, member, sums, counts, lat, lon = future.result()
            if daily_sum is None:
                daily_sum = np.zeros_like(sums, dtype=np.float64)
                ref_lat, ref_lon = lat, lon
            if not np.allclose(ref_lat, lat) or not np.allclose(ref_lon, lon):
                raise RuntimeError(f"grid mismatch at {year} member {member:02d}")
            daily_sum += sums
            daily_counts += counts
            done += 1
            if done == 1 or done % 20 == 0 or done == len(tasks):
                print(f"processed {done}/{len(tasks)} member-years", flush=True)

    if daily_sum is None:
        raise RuntimeError("no samples processed")
    if np.any(daily_counts != len(tasks)):
        raise RuntimeError(f"incomplete daily sample counts: {daily_counts.tolist()}")

    daily_mean = daily_sum / daily_counts[np.newaxis, :, np.newaxis, np.newaxis]
    output = write_climatology(args.output, args.variables, years, members, tuple(args.bbox), ref_lat, ref_lon, daily_mean, daily_counts, args.raw_root)
    print(f"wrote {output} ({output.stat().st_size / 1024**2:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
