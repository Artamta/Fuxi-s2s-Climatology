#!/usr/bin/env python3
"""Download ECMWF-S2S total precipitation and prepare daily increments."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_RAW_DIR = Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/raw")
DEFAULT_PROCESSED_DIR = Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/processed")
DEFAULT_AREA = (40.0, 60.0, 0.0, 100.0)  # north, west, south, east
DEFAULT_GRID = (1.5, 1.5)
DATASET = "s2s-forecasts"
FORECAST_TYPES = {
    "cf": "control_forecast",
    "pf": "perturbed_forecast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic-date", default="20260517", help="Initialization date as YYYYMMDD.")
    parser.add_argument("--lead-days", type=int, default=42)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--members", type=int, default=50, help="Number of perturbed members in the comparable product.")
    parser.add_argument("--area", type=float, nargs=4, default=DEFAULT_AREA, metavar=("NORTH", "WEST", "SOUTH", "EAST"))
    parser.add_argument("--grid", type=float, nargs=2, default=DEFAULT_GRID, metavar=("DLAT", "DLON"))
    parser.add_argument("--forecast-types", default="cf,pf", help="Comma-separated cf,pf raw files to download.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-between", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_ic_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%d")


def parse_forecast_types(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in FORECAST_TYPES]
    if unknown:
        raise ValueError(f"unknown forecast type(s): {unknown}; expected cf,pf")
    return items


def raw_path(raw_dir: Path, ic_date: str, ftype: str) -> Path:
    return raw_dir / "tp" / f"{ic_date}_{ftype}.nc"


def processed_path(processed_dir: Path, ic_date: str, members: int, lead_days: int) -> Path:
    return processed_dir / f"ecmwf_{ic_date}_tp_ens{members}_lead{lead_days}_india_1p5deg_daily_mm.nc"


def request_for(ic_date: str, ftype: str, lead_days: int, area: tuple[float, ...], grid: tuple[float, ...]) -> dict:
    init = parse_ic_date(ic_date)
    return {
        "origin": "ecmwf",
        "forecast_type": FORECAST_TYPES[ftype],
        "level_type": "single_level",
        "variable": "tp",
        "year": f"{init.year:04d}",
        "month": f"{init.month:02d}",
        "day": f"{init.day:02d}",
        "time": "00:00:00",
        "step": [str(hour) for hour in range(24, (lead_days * 24) + 1, 24)],
        "area": [float(item) for item in area],
        "grid": [float(item) for item in grid],
        "data_format": "netcdf",
    }


def is_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def download_one(
    client,
    target: Path,
    request: dict,
    retries: int,
    sleep_between: float,
    overwrite: bool,
) -> str:
    if is_nonempty(target) and not overwrite:
        logging.info("SKIP existing %s size=%d", target, target.stat().st_size)
        return "skipped"

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    for attempt in range(1, retries + 1):
        try:
            logging.info("START %s attempt=%d", target, attempt)
            client.retrieve(DATASET, request, str(tmp))
            if not is_nonempty(tmp):
                raise RuntimeError(f"download produced empty file: {tmp}")
            tmp.replace(target)
            logging.info("DONE  %s size=%d", target, target.stat().st_size)
            if sleep_between:
                time.sleep(sleep_between)
            return "downloaded"
        except Exception as exc:  # noqa: BLE001 - keep this resumable for batch use
            logging.exception("FAIL  %s attempt=%d: %s", target, attempt, exc)
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                time.sleep(min(300.0, 30.0 * attempt))
    return "failed"


def daily_increments_from_cumulative(da: xr.DataArray, lead_days: int, members: int) -> xr.DataArray:
    if "step" not in da.dims:
        raise ValueError("ECMWF tp field has no step dimension")
    da = da.isel(step=slice(0, lead_days))
    if "number" in da.dims:
        if da.sizes["number"] < members:
            raise ValueError(f"requested {members} members but only {da.sizes['number']} are present")
        da = da.isel(number=slice(0, members)).rename(number="member")
        da = da.assign_coords(member=np.arange(members, dtype=np.int32))
    else:
        da = da.expand_dims(member=np.asarray([0], dtype=np.int32))

    da = da.transpose("member", "step", "latitude", "longitude")
    cumulative = da.to_numpy().astype("float32", copy=False)
    padded = np.concatenate([np.zeros_like(cumulative[:, :1]), cumulative], axis=1)
    daily = np.diff(padded, axis=1)
    daily = np.clip(daily, 0.0, None).astype("float32")
    return xr.DataArray(
        daily,
        dims=("member", "lead_time", "lat", "lon"),
        coords={
            "member": da.member.values.astype(np.int32),
            "lead_time": np.arange(1, daily.shape[1] + 1, dtype=np.int32),
            "lat": da.latitude.values.astype("float32"),
            "lon": da.longitude.values.astype("float32"),
        },
        name="tp",
        attrs={
            **da.attrs,
            "long_name": "ECMWF S2S daily total precipitation increments",
            "units": "mm/day",
            "conversion": "daily increments from accumulated tp; kg m**-2 treated as mm water equivalent; negative packing artifacts clipped to zero",
        },
    )


def process_pf(raw_file: Path, output: Path, ic_date: str, lead_days: int, members: int, overwrite: bool) -> Path:
    if is_nonempty(output) and not overwrite:
        logging.info("SKIP existing processed %s size=%d", output, output.stat().st_size)
        return output
    if not is_nonempty(raw_file):
        raise FileNotFoundError(f"missing ECMWF perturbed forecast file: {raw_file}")

    init = parse_ic_date(ic_date)
    try:
        ds_raw = xr.open_dataset(raw_file)
    except ValueError as exc:
        if "Failed to decode variable 'step'" not in str(exc):
            raise
        ds_raw = xr.open_dataset(raw_file, decode_timedelta=False)
    try:
        daily = daily_increments_from_cumulative(ds_raw["tp"], lead_days=lead_days, members=members)
        valid_time = np.asarray([init + timedelta(days=lead) for lead in range(1, lead_days + 1)], dtype="datetime64[ns]")
        ds = xr.Dataset(
            data_vars={"tp": daily},
            coords={
                "member": daily.member.values,
                "lead_time": daily.lead_time.values,
                "lat": daily.lat.values,
                "lon": daily.lon.values,
                "valid_time": ("lead_time", valid_time),
                "init_time": np.datetime64(init, "ns"),
            },
            attrs={
                "model": "ECMWF S2S IFS",
                "init_date": ic_date,
                "source_file": str(raw_file),
                "source_variable": "tp / total_precipitation",
                "source_units": "kg m**-2 accumulated since initialization",
                "units": "mm/day",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "comparison_note": f"First {members} perturbed members only; lead days 1-{lead_days}; daily increments from cumulative ECMWF tp.",
            },
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        encoding = {"tp": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}}
        ds.to_netcdf(tmp, encoding=encoding)
        tmp.replace(output)
    finally:
        ds_raw.close()
    logging.info("WROTE processed %s size=%d", output, output.stat().st_size)
    return output


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    forecast_types = parse_forecast_types(args.forecast_types)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    raw_targets = {ftype: raw_path(args.raw_dir, args.ic_date, ftype) for ftype in forecast_types}
    processed = processed_path(args.processed_dir, args.ic_date, args.members, args.lead_days)
    logging.info("ECMWF-S2S tp downloader")
    logging.info("IC date       : %s", args.ic_date)
    logging.info("Lead days     : %d", args.lead_days)
    logging.info("Raw targets   : %s", {key: str(value) for key, value in raw_targets.items()})
    logging.info("Processed     : %s", processed)

    if args.dry_run:
        for ftype, target in raw_targets.items():
            logging.info("DRY %s exists=%s target=%s", ftype, is_nonempty(target), target)
        logging.info("DRY processed exists=%s target=%s", is_nonempty(processed), processed)
        return 0

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    if not args.process_only:
        import cdsapi

        client = cdsapi.Client(quiet=True)
        for ftype, target in raw_targets.items():
            req = request_for(args.ic_date, ftype, args.lead_days, tuple(args.area), tuple(args.grid))
            status = download_one(client, target, req, args.retries, args.sleep_between, args.overwrite)
            counts[status] += 1
        if counts["failed"]:
            logging.error("Download failures: %s", counts)
            return 1

    processed_out = None
    if not args.download_only:
        processed_out = process_pf(raw_path(args.raw_dir, args.ic_date, "pf"), processed, args.ic_date, args.lead_days, args.members, args.overwrite)

    manifest = {
        "ic_date": args.ic_date,
        "lead_days": args.lead_days,
        "members": args.members,
        "dataset": DATASET,
        "raw": {ftype: str(path) for ftype, path in raw_targets.items()},
        "processed": str(processed_out) if processed_out else None,
        "area_north_west_south_east": [float(item) for item in args.area],
        "grid": [float(item) for item in args.grid],
        "counts": counts,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = args.processed_dir / f"ecmwf_{args.ic_date}_tp_download_manifest.json"
    write_manifest(manifest_path, manifest)
    logging.info("WROTE manifest %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
