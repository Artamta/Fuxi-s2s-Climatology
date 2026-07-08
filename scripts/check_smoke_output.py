#!/usr/bin/env python3
"""Check a small FuXi smoke-test output tree."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True, help="Directory containing member/MM/SS.nc")
    parser.add_argument("--members", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    failures = []
    for member in range(args.members):
        for step in range(1, args.steps + 1):
            path = args.raw_dir / "member" / f"{member:02d}" / f"{step:02d}.nc"
            if not path.exists() or path.stat().st_size == 0:
                failures.append(f"missing {path}")
                continue
            ds = xr.open_dataarray(path)
            try:
                sizes = dict(ds.sizes)
                if sizes.get("time") != 1 or sizes.get("lead_time") != 1:
                    failures.append(f"{path}: unexpected time/lead sizes {sizes}")
                if sizes.get("channel") != 76 or sizes.get("lat") != 121 or sizes.get("lon") != 240:
                    failures.append(f"{path}: unexpected shape {sizes}")
                sample = ds.isel(time=0, lead_time=0, channel=[0, 5, 75], lat=[0, 60, 120], lon=[0, 120, 239]).values
                if not np.isfinite(sample).any():
                    failures.append(f"{path}: sparse sample has no finite values")
            finally:
                ds.close()

    if failures:
        print("SMOKE CHECK FAILED")
        for item in failures:
            print(f"  {item}")
        return 1

    print(f"SMOKE CHECK OK: {args.members} member(s) x {args.steps} step(s) under {args.raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

