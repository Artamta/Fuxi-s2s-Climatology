#!/usr/bin/env python3
"""Create weekly min/max diagnostics from FuXi daily 00Z t2m snapshots.

This is a proxy product for quick inspection only. FuXi-S2S raw files contain
one daily t2m snapshot at 00Z, not true daily Tmin/Tmax channels.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset

from build_june17_climatology import (
    DATA_VAR,
    DEFAULT_BBOX,
    DEFAULT_OUTPUT as DEFAULT_CLIMATOLOGY,
    WEEKS,
    convert_units,
    parse_ints,
    read_metadata,
)
from make_fuxi_weekly_analysis import DEFAULT_OUTPUT_DIR, find_raw_dir, member_path


def accumulate_member(task):
    raw_dir, member, bbox = task
    first = member_path(raw_dir, member, 1)
    channel_indices, lat, lon, lat_sel, lon_sel = read_metadata(first, ["t2m"], bbox)
    channel_idx = int(channel_indices[0])
    sums = np.zeros((42, len(lat), len(lon)), dtype=np.float64)
    counts = np.zeros((42,), dtype=np.int32)

    for lead_idx, lead_day in enumerate(range(1, 43)):
        path = member_path(raw_dir, member, lead_day)
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        with Dataset(path) as ds:
            values = np.asarray(ds.variables[DATA_VAR][0, 0, channel_idx, lat_sel, lon_sel], dtype=np.float32)
        sums[lead_idx] += convert_units("t2m", values)
        counts[lead_idx] += 1

    return member, sums, counts, lat, lon


def weekly_extremes(daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weekly_min = np.stack([daily[start - 1 : end].min(axis=0) for start, end in WEEKS], axis=0).astype("float32")
    weekly_max = np.stack([daily[start - 1 : end].max(axis=0) for start, end in WEEKS], axis=0).astype("float32")
    return weekly_min, weekly_max


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic-date", help="Forecast IC date as YYYYMMDD.")
    parser.add_argument("--raw-dir", type=Path, help="Path like .../raw/YYYYMMDD containing member/MM/SS.nc")
    parser.add_argument("--members", default="0:49")
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY)
    parser.add_argument("--bbox", type=float, nargs=4, default=None, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    raw_dir = find_raw_dir(args.ic_date, args.raw_dir)
    ic_date = args.ic_date or raw_dir.name
    members = parse_ints(args.members)
    if not args.climatology.exists():
        raise SystemExit(f"climatology not found: {args.climatology}")

    clim = xr.open_dataset(args.climatology)
    try:
        if "t2m" not in [str(item) for item in clim.variable.values]:
            raise SystemExit("climatology missing t2m")
        clim_daily = clim["daily_mean"].sel(variable="t2m").values.astype("float32")
        clim_lat = clim.lat.values.astype("float32")
        clim_lon = clim.lon.values.astype("float32")
        bbox = tuple(args.bbox) if args.bbox is not None else (
            float(clim_lon.min()),
            float(clim_lon.max()),
            float(clim_lat.min()),
            float(clim_lat.max()),
        )
    finally:
        clim.close()

    output = args.output or DEFAULT_OUTPUT_DIR / f"fuxi_t2m_00z_extremes_{ic_date}.nc"
    tasks = [(raw_dir, member, bbox) for member in members]
    print(f"raw dir     : {raw_dir}")
    print(f"ic date     : {ic_date}")
    print(f"members     : {members[0]:02d}-{members[-1]:02d} ({len(members)})")
    print(f"climatology : {args.climatology}")
    print(f"workers     : {args.workers}")

    forecast_sum = None
    sample_counts = np.zeros((42,), dtype=np.int32)
    ref_lat = ref_lon = None
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(accumulate_member, task) for task in tasks]
        for future in as_completed(futures):
            member, sums, counts, lat, lon = future.result()
            if forecast_sum is None:
                forecast_sum = np.zeros_like(sums, dtype=np.float64)
                ref_lat, ref_lon = lat, lon
            if not np.allclose(ref_lat, lat) or not np.allclose(ref_lon, lon):
                raise RuntimeError(f"grid mismatch at member {member:02d}")
            forecast_sum += sums
            sample_counts += counts
            done += 1
            if done == 1 or done % 10 == 0 or done == len(tasks):
                print(f"processed {done}/{len(tasks)} members", flush=True)

    if forecast_sum is None:
        raise RuntimeError("no forecast members processed")
    if np.any(sample_counts != len(members)):
        raise RuntimeError(f"incomplete daily sample counts: {sample_counts.tolist()}")
    if not np.allclose(ref_lat, clim_lat) or not np.allclose(ref_lon, clim_lon):
        raise RuntimeError("forecast grid does not match climatology grid")

    forecast_daily = (forecast_sum / sample_counts[:, np.newaxis, np.newaxis]).astype("float32")
    forecast_min, forecast_max = weekly_extremes(forecast_daily)
    clim_min, clim_max = weekly_extremes(clim_daily)
    anomaly_min = (forecast_min - clim_min).astype("float32")
    anomaly_max = (forecast_max - clim_max).astype("float32")

    week_numbers = np.arange(1, len(WEEKS) + 1, dtype=np.int32)
    week_start = np.asarray([item[0] for item in WEEKS], dtype=np.int32)
    week_end = np.asarray([item[1] for item in WEEKS], dtype=np.int32)
    ds = xr.Dataset(
        data_vars={
            "forecast_00z_min": (("week", "lat", "lon"), forecast_min),
            "forecast_00z_max": (("week", "lat", "lon"), forecast_max),
            "climatology_00z_min": (("week", "lat", "lon"), clim_min),
            "climatology_00z_max": (("week", "lat", "lon"), clim_max),
            "anomaly_00z_min": (("week", "lat", "lon"), anomaly_min),
            "anomaly_00z_max": (("week", "lat", "lon"), anomaly_max),
            "forecast_sample_count": (("lead_day",), sample_counts.astype("int32")),
            "week_start_day": (("week",), week_start),
            "week_end_day": (("week",), week_end),
        },
        coords={
            "week": week_numbers,
            "lead_day": np.arange(1, 43, dtype=np.int32),
            "lat": ref_lat.astype("float32"),
            "lon": ref_lon.astype("float32"),
        },
        attrs={
            "title": "FuXi-S2S weekly min/max of daily 00Z t2m snapshots",
            "ic_date": ic_date,
            "forecast_raw_dir": str(raw_dir),
            "forecast_members": ",".join(f"{item:02d}" for item in members),
            "climatology_file": str(args.climatology),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "units_t2m": "degC",
            "warning": "Proxy product only: min/max are taken across one daily 00Z t2m snapshot per lead day, not true daily Tmin/Tmax.",
        },
    )
    for name in ("forecast_00z_min", "forecast_00z_max", "climatology_00z_min", "climatology_00z_max"):
        ds[name].attrs["description"] = "Weekly extreme over daily 00Z t2m snapshots."
    ds["anomaly_00z_min"].attrs["description"] = "forecast_00z_min - climatology_00z_min."
    ds["anomaly_00z_max"].attrs["description"] = "forecast_00z_max - climatology_00z_max."

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    encoding = {
        name: {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}
        for name in (
            "forecast_00z_min",
            "forecast_00z_max",
            "climatology_00z_min",
            "climatology_00z_max",
            "anomaly_00z_min",
            "anomaly_00z_max",
        )
    }
    ds.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
    tmp.replace(output)
    print(f"wrote {output} ({output.stat().st_size / 1024**2:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
