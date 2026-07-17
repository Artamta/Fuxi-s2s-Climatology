#!/usr/bin/env python3
"""Download regional hourly ERA5 u850/v850 and write UTC daily means."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_OUTPUT = Path(
    "/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/era5_gt/"
    "era5t_wind850_daily_20260618_20260708.nc"
)
CDS_URL = "https://cds.climate.copernicus.eu/api"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-06-18")
    parser.add_argument("--end", default="2026-07-08")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cds-config", type=Path, default=Path.home() / ".cdsapirc")
    parser.add_argument(
        "--resume-june-job-id",
        help="Resume an already accepted June CDS job instead of submitting it again.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_key(path: Path) -> str:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    if not values.get("key"):
        raise ValueError(f"{path}: missing CDS key")
    return values["key"]


def request_month(client: cdsapi.Client, dates: pd.DatetimeIndex, target: Path) -> None:
    request = {
        "product_type": ["reanalysis"],
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "year": [f"{dates[0].year:04d}"],
        "month": [f"{dates[0].month:02d}"],
        "day": [f"{day:02d}" for day in sorted(set(dates.day))],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "pressure_level": ["850"],
        "grid": ["1.5", "1.5"],
        "area": [40, 60, 0, 100],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    client.retrieve("reanalysis-era5-pressure-levels", request, str(target))


def resume_job(client: cdsapi.Client, job_id: str, target: Path) -> None:
    print(f"resuming accepted CDS job {job_id}", flush=True)
    job = client.client._retrieve_api.get_job(job_id)
    job.get_remote().download(str(target))


def find_variable(dataset: xr.Dataset, aliases: tuple[str, ...]) -> xr.DataArray:
    for name in aliases:
        if name in dataset:
            return dataset[name]
    raise KeyError(f"none of {aliases} found; variables are {list(dataset.data_vars)}")


def normalize_component(data: xr.DataArray, name: str) -> xr.DataArray:
    rename = {}
    for old, new in {
        "valid_time": "time",
        "latitude": "lat",
        "longitude": "lon",
        "pressure_level": "level",
    }.items():
        if old in data.dims or old in data.coords:
            rename[old] = new
    if rename:
        data = data.rename(rename)
    for dimension in tuple(data.dims):
        if dimension not in {"time", "level", "lat", "lon"} and data.sizes[dimension] == 1:
            data = data.isel({dimension: 0}, drop=True)
    if "level" in data.dims:
        data = data.sel(level=850, method="nearest", drop=True)
    data = data.sortby("time").sortby("lat", ascending=False).sortby("lon")
    data = data.resample(time="1D").mean(skipna=True)
    return data.astype("float32").rename(name)


def main() -> int:
    args = parse_args()
    dates = pd.date_range(args.start, args.end, freq="D")
    if dates.empty:
        raise ValueError("empty date range")
    if args.output.exists() and not args.overwrite:
        print(f"existing output: {args.output}")
        return 0

    client = cdsapi.Client(
        url=CDS_URL,
        key=read_key(args.cds_config),
        quiet=False,
        progress=True,
    )
    parts = []
    for period in dates.to_period("M").unique():
        month_dates = dates[dates.to_period("M") == period]
        part = args.output.parent / f".{args.output.stem}_{month_dates[0]:%Y%m}.download.nc"
        if not part.exists() or part.stat().st_size == 0:
            if month_dates[0].month == 6 and args.resume_june_job_id:
                resume_job(client, args.resume_june_job_id, part)
            else:
                request_month(client, month_dates, part)
        parts.append(part)

    opened = [xr.open_dataset(path) for path in parts]
    try:
        merged = xr.combine_by_coords(opened, combine_attrs="drop_conflicts")
        u = normalize_component(
            find_variable(merged, ("u", "u_component_of_wind")), "u850"
        )
        v = normalize_component(
            find_variable(merged, ("v", "v_component_of_wind")), "v850"
        )
        expected = pd.DatetimeIndex(dates).normalize()
        actual = pd.DatetimeIndex(u.time.values).normalize()
        missing = expected.difference(actual)
        if len(missing):
            raise ValueError(f"download is missing dates: {missing.tolist()}")
        output = xr.Dataset({"u850": u.sel(time=expected), "v850": v.sel(time=expected)})
        output = output.assign_attrs(
            title="ERA5/ERA5T UTC daily-mean 850-hPa wind ground truth",
            source="Copernicus CDS reanalysis-era5-pressure-levels",
            valid_start=str(expected[0].date()),
            valid_end=str(expected[-1].date()),
            daily_method="arithmetic mean of 24 hourly values, 00-23 UTC",
            units="m s-1",
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        for name in output.data_vars:
            output[name].attrs.update(units="m s-1")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".nc.tmp")
        temporary.unlink(missing_ok=True)
        encoding = {
            name: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for name in output.data_vars
        }
        output.to_netcdf(temporary, encoding=encoding)
        temporary.replace(args.output)
    finally:
        for dataset in opened:
            dataset.close()

    manifest = args.output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "output": str(args.output),
                "valid_start": args.start,
                "valid_end": args.end,
                "source": "Copernicus CDS reanalysis-era5-pressure-levels",
                "variables": ["u850", "v850"],
                "daily_method": "mean of 00-23 UTC hourly values",
                "request_domain": "40N-0N, 60E-100E at 1.5 degrees",
                "credential_note": "CDS credential read locally and never stored in outputs",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for path in parts:
        path.unlink(missing_ok=True)
    print(f"wrote {args.output}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
