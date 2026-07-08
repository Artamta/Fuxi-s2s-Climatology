#!/usr/bin/env python3
"""Create exact June-17 FuXi-S2S input.nc files from ARCO ERA5.

The script fetches all requested years in grouped ARCO calls, then writes one
FuXi-compatible input file per initialization date:

  data/ic_inputs/YYYY0617/input.nc

This is faster and cleaner than opening ARCO once per year.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from earth2studio.data import ARCO


LEVELS_DESC = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
PL_PREFIXES = ["z", "t", "u", "v", "q"]
SURFACE_MAP = [
    ("t2m", "t2m"),
    ("d2m", "d2m"),
    ("sst", "sst"),
    ("ttr", "ttr"),
    ("u10m", "10u"),
    ("v10m", "10v"),
    ("u100m", "100u"),
    ("v100m", "100v"),
    ("msl", "msl"),
    ("tcwv", "tcwv"),
    ("tp", "tp"),
]


def parse_years(value: str) -> list[int]:
    if ":" in value:
        start, end = value.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def validate_mmdd(value: str) -> str:
    if len(value) != 4 or not value.isdigit():
        raise argparse.ArgumentTypeError("MMDD must look like 0617")
    pd.Timestamp(f"2001-{value[:2]}-{value[2:]}")
    return value


def ymd(year: int, mmdd: str) -> str:
    return f"{year}{mmdd}"


def input_path(output_dir: Path, year: int, mmdd: str) -> Path:
    date = ymd(year, mmdd)
    return output_dir / date / "input.nc"


def pending_years(output_dir: Path, years: list[int], mmdd: str, overwrite: bool) -> list[int]:
    if overwrite:
        return years
    return [year for year in years if not input_path(output_dir, year, mmdd).exists()]


def build_records(years: list[int], mmdd: str) -> tuple[list[pd.Timestamp], list[tuple[int, pd.Timestamp, pd.Timestamp]]]:
    records = []
    times = []
    for year in years:
        init_time = pd.Timestamp(f"{year}-{mmdd[:2]}-{mmdd[2:]}")
        prev_time = init_time - pd.Timedelta(days=1)
        records.append((year, prev_time, init_time))
        times.extend([prev_time, init_time])
    return pd.to_datetime(times), records


def fetch_group(source: ARCO, times: pd.DatetimeIndex, source_vars: list[str], out_channels: list[str], verbose_name: str):
    print(f"fetch {verbose_name}: {source_vars[0]}..{source_vars[-1]}", flush=True)
    t0 = time.time()
    da = source([time.to_pydatetime() for time in times], source_vars)
    da = da.sel(time=times)
    da = da.isel(lat=slice(None, None, 6), lon=slice(None, None, 6)).astype("float32")
    print(f"  done {verbose_name} in {time.time() - t0:.1f}s", flush=True)
    return da.values, out_channels, da.lat.values.astype("float32"), da.lon.values.astype("float32")


def build_input_cube(years: list[int], mmdd: str, timeout: int, verbose: bool):
    times, records = build_records(years, mmdd)
    source = ARCO(cache=True, verbose=verbose, async_timeout=timeout)

    chunks = []
    channels = []
    lat = lon = None

    for prefix in PL_PREFIXES:
        source_vars = [f"{prefix}{level}" for level in LEVELS_DESC]
        values, out_channels, lat, lon = fetch_group(source, times, source_vars, source_vars, prefix)
        chunks.append(values)
        channels.extend(out_channels)

    surface_source = [source_name for source_name, _ in SURFACE_MAP]
    surface_channels = [out_name for _, out_name in SURFACE_MAP]
    values, out_channels, lat, lon = fetch_group(source, times, surface_source, surface_channels, "surface")
    chunks.append(values)
    channels.extend(out_channels)

    data = np.concatenate(chunks, axis=1).astype("float32")
    channel_index = {channel: idx for idx, channel in enumerate(channels)}

    data[:, channel_index["tp"], :, :] = np.clip(data[:, channel_index["tp"], :, :] * 1000.0, 0.0, 1000.0)
    data[:, channel_index["ttr"], :, :] = data[:, channel_index["ttr"], :, :] / 3600.0

    data = data.reshape(len(records), 2, len(channels), len(lat), len(lon))
    return data, records, channels, lat, lon


def write_input(
    output_dir: Path,
    mmdd: str,
    year: int,
    prev_time: pd.Timestamp,
    init_time: pd.Timestamp,
    values: np.ndarray,
    channels: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    overwrite: bool,
) -> Path:
    out_file = input_path(output_dir, year, mmdd)
    if out_file.exists() and out_file.stat().st_size > 0 and not overwrite:
        print(f"skip {ymd(year, mmdd)} existing {out_file}", flush=True)
        return out_file

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(".nc.tmp")
    if tmp_file.exists():
        tmp_file.unlink()

    da = xr.DataArray(
        values.astype("float32"),
        dims=("time", "channel", "lat", "lon"),
        coords={
            "time": [prev_time.to_datetime64(), init_time.to_datetime64()],
            "channel": channels,
            "lat": lat,
            "lon": lon,
        },
        name="data",
        attrs={
            "source": "ARCO ERA5 via earth2studio.data.ARCO",
            "init_date": ymd(year, mmdd),
            "history_time": f"{prev_time:%Y-%m-%dT00:00:00}",
            "init_time": f"{init_time:%Y-%m-%dT00:00:00}",
            "tp_note": "Converted from m to mm and clipped to [0, 1000], matching FuXi data_util.make_input",
            "ttr_note": "Divided by 3600, matching FuXi data_util.make_input",
        },
    )
    da.to_netcdf(tmp_file)
    tmp_file.replace(out_file)
    print(f"wrote {out_file} ({out_file.stat().st_size / 1024**2:.1f} MiB)", flush=True)
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="2002:2021", help="Year range like 2002:2021 or comma list")
    parser.add_argument("--mmdd", type=validate_mmdd, default="0617")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ic_inputs"))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    years = parse_years(args.years)
    todo = pending_years(args.output_dir, years, args.mmdd, args.overwrite)
    print("=" * 72)
    print("FuXi-S2S exact IC input builder")
    print(f"years      : {years[0]}-{years[-1]} ({len(years)} total)")
    print(f"mmdd       : {args.mmdd}")
    print(f"output_dir : {args.output_dir.resolve()}")
    print(f"pending    : {len(todo)}")
    print("=" * 72)

    if not todo:
        print("all requested inputs already exist")
        return 0

    data, records, channels, lat, lon = build_input_cube(todo, args.mmdd, args.timeout, args.verbose)
    for idx, (year, prev_time, init_time) in enumerate(records):
        write_input(args.output_dir, args.mmdd, year, prev_time, init_time, data[idx], channels, lat, lon, args.overwrite)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

