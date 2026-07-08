#!/usr/bin/env python3
"""Validate generated FuXi input.nc files without loading everything."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


EXPECTED_CHANNELS = 76
EXPECTED_LAT = 121
EXPECTED_LON = 240


def parse_years(value: str) -> list[int]:
    if ":" in value:
        start, end = value.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def validate_one(path: Path, init_date: str) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size == 0:
        return False, "missing or empty"

    ds = xr.open_dataarray(path)
    try:
        sizes = dict(ds.sizes)
        if sizes.get("time") != 2:
            return False, f"time={sizes.get('time')}, expected 2"
        if sizes.get("channel") != EXPECTED_CHANNELS:
            return False, f"channel={sizes.get('channel')}, expected {EXPECTED_CHANNELS}"
        if sizes.get("lat") != EXPECTED_LAT or sizes.get("lon") != EXPECTED_LON:
            return False, f"grid={sizes.get('lat')}x{sizes.get('lon')}, expected {EXPECTED_LAT}x{EXPECTED_LON}"

        times = pd.to_datetime(ds.time.values)
        expected_init = pd.Timestamp(f"{init_date[:4]}-{init_date[4:6]}-{init_date[6:]}")
        if times[-1] != expected_init or times[-2] != expected_init - pd.Timedelta(days=1):
            return False, f"times={times.tolist()}"

        sample = ds.isel(time=[0, 1], channel=[0, 5, 20, 75], lat=[0, 60, 120], lon=[0, 120, 239]).values
        if not np.isfinite(sample).any():
            return False, "sparse sample has no finite values"
        return True, f"ok size={path.stat().st_size / 1024**2:.1f} MiB"
    finally:
        ds.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/ic_inputs"))
    parser.add_argument("--years", default="2002:2021")
    parser.add_argument("--mmdd", default="0617")
    args = parser.parse_args()

    ok_count = 0
    rows = []
    for year in parse_years(args.years):
        init_date = f"{year}{args.mmdd}"
        path = args.input_dir / init_date / "input.nc"
        ok, detail = validate_one(path, init_date)
        rows.append((init_date, ok, detail, path))
        ok_count += int(ok)
        print(f"{init_date}: {'OK' if ok else 'FAIL'} - {detail}")

    print(f"summary: {ok_count}/{len(rows)} ok")
    return 0 if ok_count == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

