#!/usr/bin/env python3
"""Download ARCO ERA5 daily total precipitation truth for a FuXi IC."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from earth2studio.data import ARCO


DEFAULT_BBOX = (66.5, 98.5, 5.0, 38.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ic-date", default="20260517", help="Initialization date as YYYYMMDD.")
    parser.add_argument("--lead-days", type=int, default=42)
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    parser.add_argument("--output-dir", type=Path, default=Path("/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/truth"))
    parser.add_argument("--chunk-days", type=int, default=3, help="Number of daily totals to fetch per ARCO request.")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def valid_dates(ic_date: str, lead_days: int) -> list[pd.Timestamp]:
    init = pd.Timestamp(datetime.strptime(ic_date, "%Y%m%d"))
    return [init + pd.Timedelta(days=lead_day) for lead_day in range(1, lead_days + 1)]


def day_hours(date: pd.Timestamp) -> list[datetime]:
    start = date.to_pydatetime()
    return [start + timedelta(hours=hour) for hour in range(24)]


def date_chunks(items: list[pd.Timestamp], chunk_days: int) -> list[list[pd.Timestamp]]:
    if chunk_days < 1:
        raise ValueError("--chunk-days must be >= 1")
    return [items[start : start + chunk_days] for start in range(0, len(items), chunk_days)]


def availability_message(exc: Exception) -> str:
    text = str(exc).splitlines()[0]
    text = re.sub(r"\s+", " ", text).strip()
    return text[:400]


def fetch_daily_chunk(
    source: ARCO,
    dates: list[pd.Timestamp],
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon_min, lon_max, lat_min, lat_max = bbox
    times = [time for date in dates for time in day_hours(date)]
    da = source(times, ["tp"]).sel(time=pd.to_datetime(times))
    da = da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    lat = da.lat.values.astype("float32")
    lon = da.lon.values.astype("float32")
    values = da.values[:, 0].astype("float32")
    values = values.reshape(len(dates), 24, len(lat), len(lon))
    daily_mm = np.clip(values, 0.0, None).sum(axis=1) * np.float32(1000.0)
    return daily_mm.astype("float32"), lat, lon


def download_daily(
    ic_date: str,
    requested_dates: list[pd.Timestamp],
    bbox: tuple[float, float, float, float],
    timeout: int,
    chunk_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    source = ARCO(cache=True, verbose=False, async_timeout=timeout)
    fields: list[np.ndarray] = []
    status: list[dict] = []
    lat = lon = None

    for chunk in date_chunks(requested_dates, chunk_days):
        try:
            daily, chunk_lat, chunk_lon = fetch_daily_chunk(source, chunk, bbox)
        except Exception as chunk_exc:
            for date in chunk:
                try:
                    daily, chunk_lat, chunk_lon = fetch_daily_chunk(source, [date], bbox)
                except Exception as day_exc:
                    status.append(
                        {
                            "lead_day": int((date - pd.Timestamp(datetime.strptime(ic_date, "%Y%m%d"))).days),
                            "valid_date": str(date.date()),
                            "status": "unavailable_or_incomplete",
                            "message": availability_message(day_exc),
                        }
                    )
                    print(f"missing {date.date()}: {availability_message(day_exc)}", flush=True)
                    continue
                lat, lon = chunk_lat, chunk_lon
                fields.append(daily[0])
                status.append(
                    {
                        "lead_day": int((date - pd.Timestamp(datetime.strptime(ic_date, "%Y%m%d"))).days),
                        "valid_date": str(date.date()),
                        "status": "complete",
                        "hours": 24,
                    }
                )
                print(f"downloaded {date.date()} after chunk fallback", flush=True)
            continue

        lat, lon = chunk_lat, chunk_lon
        for idx, date in enumerate(chunk):
            fields.append(daily[idx])
            status.append(
                {
                    "lead_day": int((date - pd.Timestamp(datetime.strptime(ic_date, "%Y%m%d"))).days),
                    "valid_date": str(date.date()),
                    "status": "complete",
                    "hours": 24,
                }
            )
        print(f"downloaded {chunk[0].date()}..{chunk[-1].date()}", flush=True)

    if lat is None or lon is None:
        raise RuntimeError("no complete ARCO ERA5 daily precipitation days were downloaded")
    return np.stack(fields, axis=0).astype("float32"), lat, lon, status


def write_outputs(
    output_dir: Path,
    ic_date: str,
    requested_dates: list[pd.Timestamp],
    daily_mm: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    status: list[dict],
    bbox: tuple[float, float, float, float],
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    nc_out = output_dir / f"arco_era5_tp_daily_{ic_date}.nc"
    csv_out = output_dir / f"arco_era5_tp_daily_{ic_date}_summary.csv"
    json_out = output_dir / f"arco_era5_tp_daily_{ic_date}_availability.json"
    if nc_out.exists() and not overwrite:
        raise FileExistsError(f"{nc_out} exists; pass --overwrite to replace it")

    complete = [item for item in status if item["status"] == "complete"]
    complete_leads = np.asarray([item["lead_day"] for item in complete], dtype=np.int32)
    complete_dates = pd.to_datetime([item["valid_date"] for item in complete])

    ds = xr.Dataset(
        data_vars={
            "tp_daily": (("lead_day", "lat", "lon"), daily_mm),
        },
        coords={
            "lead_day": complete_leads,
            "valid_time": (("lead_day",), complete_dates.values),
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "title": "ARCO ERA5 daily total precipitation truth for FuXi-S2S verification",
            "ic_date": ic_date,
            "requested_valid_start": str(requested_dates[0].date()),
            "requested_valid_end": str(requested_dates[-1].date()),
            "available_valid_start": str(complete_dates[0].date()) if len(complete_dates) else "",
            "available_valid_end": str(complete_dates[-1].date()) if len(complete_dates) else "",
            "requested_lead_days": len(requested_dates),
            "available_lead_days": len(complete),
            "source": "ARCO ERA5 via earth2studio.data.ARCO",
            "source_variable": "tp / total_precipitation",
            "units": "mm/day",
            "daily_method": "sum of 24 hourly ARCO ERA5 tp fields from valid-date 00Z through 23Z UTC; tiny negative values clipped to zero; metres converted to millimetres",
            "bbox_lon_min_lon_max_lat_min_lat_max": ",".join(str(item) for item in bbox),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    ds["tp_daily"].attrs["long_name"] = "daily total precipitation"
    ds["tp_daily"].attrs["units"] = "mm/day"
    ds["valid_time"].attrs["description"] = "UTC date represented by the 00Z..23Z hourly total precipitation sum."
    encoding = {"tp_daily": {"zlib": True, "complevel": 4, "_FillValue": np.float32(np.nan)}}

    tmp = nc_out.with_suffix(".nc.tmp")
    if tmp.exists():
        tmp.unlink()
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.replace(nc_out)

    summary = []
    field_idx = 0
    for item in status:
        row = dict(item)
        if item["status"] == "complete":
            field = daily_mm[field_idx]
            row.update(
                {
                    "tp_min_mm_day": float(np.nanmin(field)),
                    "tp_mean_mm_day": float(np.nanmean(field)),
                    "tp_max_mm_day": float(np.nanmax(field)),
                }
            )
            field_idx += 1
        summary.append(row)
    pd.DataFrame(summary).to_csv(csv_out, index=False)

    availability = {
        "ic_date": ic_date,
        "requested_lead_days": len(requested_dates),
        "complete_lead_days": len(complete),
        "missing_or_incomplete_lead_days": len(requested_dates) - len(complete),
        "netcdf": str(nc_out),
        "summary_csv": str(csv_out),
        "status": status,
    }
    json_out.write_text(json.dumps(availability, indent=2), encoding="utf-8")
    return nc_out, csv_out, json_out


def main() -> int:
    args = parse_args()
    requested = valid_dates(args.ic_date, args.lead_days)
    bbox = tuple(args.bbox)
    daily_mm, lat, lon, status = download_daily(
        ic_date=args.ic_date,
        requested_dates=requested,
        bbox=bbox,
        timeout=args.timeout,
        chunk_days=args.chunk_days,
    )
    nc_out, csv_out, json_out = write_outputs(
        output_dir=args.output_dir,
        ic_date=args.ic_date,
        requested_dates=requested,
        daily_mm=daily_mm,
        lat=lat,
        lon=lon,
        status=status,
        bbox=bbox,
        overwrite=args.overwrite,
    )
    print(f"wrote {nc_out}")
    print(f"wrote {csv_out}")
    print(f"wrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
