#!/usr/bin/env python3
"""Create a weekly FuXi forecast/anomaly product from a June-17 climatology."""

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
    TP_UNITS_NOTE,
    WEEKS,
    convert_units,
    coord_slice,
    parse_ints,
    parse_variables,
    read_metadata,
)


KNOWN_RAW_ROOTS = (
    Path("/storage/raj.ayush/All_Model_Data/fuxi/test/raw"),
    Path("/storage/raj.ayush/All_Model_Data/fuxi/op2026_ens50/raw"),
    Path("/storage/raj.ayush/All_Model_Data/fuxi/jjas2026_ens50/raw"),
    Path("/storage/raj.ayush/All_Model_Data/fuxi/jjas2019/raw"),
    Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/raw"),
)
DEFAULT_OUTPUT_DIR = Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis")


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


def member_path(raw_dir: Path, member: int, lead_day: int) -> Path:
    return raw_dir / "member" / f"{member:02d}" / f"{lead_day:02d}.nc"


def accumulate_member(task):
    raw_dir, member, variables, bbox = task
    first = member_path(raw_dir, member, 1)
    channel_indices, lat, lon, lat_sel, lon_sel = read_metadata(first, variables, bbox)
    sums = np.zeros((len(variables), len(WEEKS), len(lat), len(lon)), dtype=np.float64)
    counts = np.zeros((len(WEEKS),), dtype=np.int32)

    for week_idx, (start, end) in enumerate(WEEKS):
        for lead_day in range(start, end + 1):
            path = member_path(raw_dir, member, lead_day)
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
            with Dataset(path) as ds:
                values = np.asarray(ds.variables[DATA_VAR][0, 0, channel_indices, lat_sel, lon_sel], dtype=np.float32)
            if values.ndim == 2:
                values = values[np.newaxis, :, :]
            for var_idx, variable in enumerate(variables):
                sums[var_idx, week_idx] += convert_units(variable, values[var_idx])
            counts[week_idx] += 1
    return member, sums, counts, lat, lon


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic-date", help="Forecast IC date as YYYYMMDD.")
    parser.add_argument("--raw-dir", type=Path, help="Path like .../raw/YYYYMMDD containing member/MM/SS.nc")
    parser.add_argument("--members", default="0:49")
    parser.add_argument("--variables", type=parse_variables, default=parse_variables("tp,t2m"))
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
        clim_variables = [str(item) for item in clim.variable.values]
        missing = [variable for variable in args.variables if variable not in clim_variables]
        if missing:
            raise SystemExit(f"climatology missing variable(s): {','.join(missing)}")
        clim_weekly = clim["weekly_mean"].sel(variable=args.variables).values.astype("float32")
        clim_lat = clim.lat.values.astype("float32")
        clim_lon = clim.lon.values.astype("float32")
        bbox = tuple(args.bbox) if args.bbox is not None else (
            float(clim_lon.min()),
            float(clim_lon.max()),
            float(clim_lat.min()),
            float(clim_lat.max()),
        )
        week_start = clim["week_start_day"].values.astype("int32")
        week_end = clim["week_end_day"].values.astype("int32")
    finally:
        clim.close()

    output = args.output or DEFAULT_OUTPUT_DIR / f"fuxi_weekly_analysis_{ic_date}.nc"
    tasks = [(raw_dir, member, args.variables, bbox) for member in members]
    print(f"raw dir     : {raw_dir}")
    print(f"ic date     : {ic_date}")
    print(f"members     : {members[0]:02d}-{members[-1]:02d} ({len(members)})")
    print(f"variables   : {','.join(args.variables)}")
    print(f"climatology : {args.climatology}")
    print(f"workers     : {args.workers}")

    forecast_sum = None
    sample_counts = np.zeros((len(WEEKS),), dtype=np.int32)
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
    expected = len(members) * 7
    if np.any(sample_counts != expected):
        raise RuntimeError(f"incomplete weekly sample counts: {sample_counts.tolist()}, expected {expected}")
    if not np.allclose(ref_lat, clim_lat) or not np.allclose(ref_lon, clim_lon):
        raise RuntimeError("forecast grid does not match climatology grid")

    forecast_weekly = (forecast_sum / sample_counts[np.newaxis, :, np.newaxis, np.newaxis]).astype("float32")
    anomaly_weekly = (forecast_weekly - clim_weekly).astype("float32")
    ds = xr.Dataset(
        data_vars={
            "forecast_weekly": (("variable", "week", "lat", "lon"), forecast_weekly),
            "climatology_weekly": (("variable", "week", "lat", "lon"), clim_weekly),
            "anomaly_weekly": (("variable", "week", "lat", "lon"), anomaly_weekly),
            "forecast_sample_count": (("week",), sample_counts.astype("int32")),
            "week_start_day": (("week",), week_start),
            "week_end_day": (("week",), week_end),
        },
        coords={
            "variable": args.variables,
            "week": np.arange(1, len(WEEKS) + 1, dtype=np.int32),
            "lat": ref_lat.astype("float32"),
            "lon": ref_lon.astype("float32"),
        },
        attrs={
            "title": "FuXi-S2S weekly forecast and model-climatology anomaly",
            "ic_date": ic_date,
            "forecast_raw_dir": str(raw_dir),
            "forecast_members": ",".join(f"{item:02d}" for item in members),
            "climatology_file": str(args.climatology),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "units_tp": TP_UNITS_NOTE,
            "units_t2m": "degC",
            "note": "Anomaly is forecast_weekly minus FuXi June-17 model climatology.",
        },
    )
    ds["forecast_weekly"].attrs["description"] = "Ensemble weekly mean for the target IC."
    ds["climatology_weekly"].attrs["description"] = "June-17 FuXi model climatology used for anomaly."
    ds["anomaly_weekly"].attrs["description"] = "forecast_weekly - climatology_weekly."

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    encoding = {
        "forecast_weekly": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
        "climatology_weekly": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
        "anomaly_weekly": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)},
    }
    ds.to_netcdf(tmp, engine="netcdf4", encoding=encoding)
    tmp.replace(output)
    print(f"wrote {output} ({output.stat().st_size / 1024**2:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
