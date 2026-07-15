#!/usr/bin/env python3
"""Run the audited FuXi pipeline while retaining u850, v850, and TP."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


PIPELINE = Path("/home/raj.ayush/s2s/s2s_anlysis/clean/model-runs/fuxi/scripts")
sys.path.insert(0, str(PIPELINE))

import run_fuxi_forecast as base  # noqa: E402


ALLOWED_CHANNELS = ("u850", "v850", "tp")


def retained_channels(config: dict[str, Any]) -> tuple[str, ...]:
    channels = tuple(str(value) for value in config.get("retained_variables", ()))
    if channels != ALLOWED_CHANNELS:
        raise ValueError(
            f"this wind workflow requires {ALLOWED_CHANNELS}, found {channels}"
        )
    return channels


def read_raw_fields(
    path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    channels: tuple[str, ...],
) -> np.ndarray:
    with xr.open_dataarray(path) as source:
        available = {str(value) for value in source.channel.values.tolist()}
        missing = set(channels) - available
        if missing:
            raise ValueError(f"{path} missing channels {sorted(missing)}")
        selected = source.sel(channel=list(channels), lat=lat, lon=lon).squeeze(
            drop=True
        )
        selected = selected.transpose("channel", "lat", "lon")
        values = selected.values.astype(np.float32, copy=False)
    expected = (len(channels), len(lat), len(lon))
    if values.shape != expected:
        raise ValueError(f"{path} produced shape {values.shape}, expected {expected}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains missing retained fields over the domain")
    return values


def combine_output(
    raw_dir: Path,
    output: Path,
    date: pd.Timestamp,
    config: dict[str, Any],
) -> None:
    channels = retained_channels(config)
    members = int(config["members"])
    steps = int(config["lead_days"])
    lat, lon = base.expected_grid(config)
    values = np.empty(
        (members, steps, len(channels), len(lat), len(lon)), dtype=np.float32
    )
    for member in range(members):
        for step in range(1, steps + 1):
            path = raw_dir / "member" / f"{member:02d}" / f"{step:02d}.nc"
            values[member, step - 1] = read_raw_fields(path, lat, lon, channels)

    period_start, period_end, valid_time = base.temporal_contract.forecast_periods(
        date, steps, config
    )
    timing = base.temporal_contract.provenance(date, config)
    period_bounds = np.stack([period_start, period_end], axis=1)
    data_vars = {
        channel: (
            ("member", "lead_day", "latitude", "longitude"),
            values[:, :, channel_index],
        )
        for channel_index, channel in enumerate(channels)
    }
    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "member": np.arange(members, dtype=np.int16),
            "lead_day": np.arange(1, steps + 1, dtype=np.int16),
            "latitude": lat,
            "longitude": lon,
            "valid_time": ("lead_day", valid_time),
            "forecast_period_start": ("lead_day", period_start),
            "forecast_period_end": ("lead_day", period_end),
            "forecast_period_bounds": (("lead_day", "bounds"), period_bounds),
            "bounds": np.asarray([0, 1], dtype=np.int8),
            "init_time": date.to_datetime64(),
            "forecast_reference_time": date.to_datetime64(),
            "model_state_time": np.datetime64(timing["model_state_time"]),
            "information_cutoff_time": np.datetime64(
                timing["information_cutoff_time"]
            ),
        },
        attrs={
            "model": "FuXi-S2S",
            "run_label": config["run_label"],
            "init_date": date.strftime("%Y-%m-%d"),
            "benchmark_mode": timing["benchmark_mode"],
            "strict_operational": str(timing["strict_operational"]).lower(),
            "information_cutoff_matches_issue_time": str(
                timing["information_cutoff_matches_issue_time"]
            ).lower(),
            "valid_time_role": timing["valid_time_role"],
            "input_source": config["input"]["source"],
            "ensemble": (
                f"{members} repeated calls to the official stochastic ONNX model; "
                "no separate control member"
            ),
            "member_generation": config["member_generation"],
            "input_daily_statistic": config["input"]["daily_statistic"],
            "input_hourly_sampling": config["input"]["hourly_sampling"],
            "input_time_zone": config["input"]["time_zone"],
            "domain": "exact physics grid: 39-0 N, 60-99 E, 1.5 degrees",
            "forecast_time_statistic": "global daily mean at daily resolution",
            "retained_channels": ",".join(channels),
            "excluded_entry_point": "inference_ensemble.py is not used",
            "model_onnx_sha256": config["model"]["onnx_sha256"],
            "model_external_data_sha256": config["model"][
                "external_data_sha256"
            ],
        },
    )
    dataset["valid_time"].attrs.update(
        {
            "long_name": "forecast valid time",
            "bounds": "forecast_period_bounds",
            "representation": timing["valid_time_role"],
        }
    )
    for channel in ("u850", "v850"):
        component = "zonal" if channel == "u850" else "meridional"
        dataset[channel].attrs.update(
            {
                "long_name": f"850 hPa {component} wind component",
                "units": "m s-1",
                "cell_methods": "time: mean (24-hour period)",
            }
        )
    dataset["tp"].attrs.update(
        {
            "long_name": "FuXi-S2S total precipitation mean rate",
            "units": "mm h-1",
            "cell_methods": "time: mean (24-hour period)",
            "comparison_conversion": "multiply by 24 for mm day-1",
        }
    )

    encoding: dict[str, dict[str, Any]] = {
        name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "chunksizes": (1, min(7, steps), len(lat), len(lon)),
        }
        for name in channels
    }
    time_encoding = {
        "units": "hours since 1970-01-01 00:00:00",
        "calendar": "proleptic_gregorian",
    }
    for name in (
        "valid_time",
        "forecast_period_start",
        "forecast_period_end",
        "forecast_period_bounds",
        "init_time",
        "forecast_reference_time",
        "model_state_time",
        "information_cutoff_time",
    ):
        encoding[name] = time_encoding.copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.unlink(missing_ok=True)
    dataset.to_netcdf(temporary, encoding=encoding)
    validate_output(temporary, date, config)
    temporary.replace(output)


def validate_output(
    path: Path, date: pd.Timestamp, config: dict[str, Any]
) -> dict[str, Any]:
    channels = retained_channels(config)
    with xr.open_dataset(path) as dataset:
        expected_sizes = {
            "member": int(config["members"]),
            "lead_day": int(config["lead_days"]),
            "latitude": int(config["domain"]["latitude_count"]),
            "longitude": int(config["domain"]["longitude_count"]),
        }
        for dimension, size in expected_sizes.items():
            if dataset.sizes.get(dimension) != size:
                raise ValueError(
                    f"{dimension}: expected {size}, found {dataset.sizes.get(dimension)}"
                )
        if set(dataset.data_vars) != set(channels):
            raise ValueError(
                f"expected {channels}, found {sorted(dataset.data_vars)}"
            )
        lat, lon = base.expected_grid(config)
        if not np.allclose(dataset.latitude, lat) or not np.allclose(
            dataset.longitude, lon
        ):
            raise ValueError("final grid does not match the physics-model grid")
        if pd.Timestamp(dataset.init_time.values) != date:
            raise ValueError(f"wrong init time: {dataset.init_time.values}")
        steps = int(config["lead_days"])
        expected_leads = np.arange(1, steps + 1, dtype=np.int16)
        if not np.array_equal(dataset.lead_day.values, expected_leads):
            raise ValueError("wrong lead-day coordinate")
        period_start, period_end, valid_time = (
            base.temporal_contract.forecast_periods(date, steps, config)
        )
        for name, expected in {
            "forecast_period_start": period_start,
            "forecast_period_end": period_end,
            "valid_time": valid_time,
        }.items():
            if not np.array_equal(
                dataset[name].values.astype("datetime64[ns]"), expected
            ):
                raise ValueError(f"{name} does not match the timing contract")
        timing = base.temporal_contract.provenance(date, config)
        if bool(timing["strict_operational"]):
            if pd.Timestamp(dataset.forecast_reference_time.values) != date:
                raise ValueError("wrong strict forecast-reference time")
            if pd.Timestamp(dataset.information_cutoff_time.values) != date:
                raise ValueError("strict information cutoff is not the issue time")
        if (
            dataset.attrs.get("forecast_time_statistic")
            != "global daily mean at daily resolution"
        ):
            raise ValueError("forecast daily-mean statistic is not declared")

        stats: dict[str, Any] = {}
        for name in channels:
            values = dataset[name].values
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains missing or infinite values")
            stats[name] = {
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        if stats["tp"]["minimum"] < -1e-6 or stats["tp"]["maximum"] > 100.0:
            raise ValueError(f"implausible FuXi TP range: {stats['tp']}")
        for name in ("u850", "v850"):
            if stats[name]["minimum"] < -200.0 or stats[name]["maximum"] > 200.0:
                raise ValueError(f"implausible FuXi wind range: {name} {stats[name]}")
        speed_max = float(np.hypot(dataset.u850.values, dataset.v850.values).max())
        if speed_max > 200.0:
            raise ValueError(f"implausible 850-hPa wind speed: {speed_max}")
        if int(config["members"]) > 1:
            spread = max(
                float(abs(dataset[name].isel(member=0) - dataset[name].isel(member=1)).max())
                for name in channels
            )
            if spread <= 1e-6:
                raise ValueError("FuXi members 0 and 1 are identical")
        else:
            spread = None
        return {
            "size_bytes": path.stat().st_size,
            "member_0_1_max_difference": spread,
            "wind_speed_maximum": speed_max,
            "fields": stats,
        }


def main() -> int:
    args = base.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    channels = retained_channels(config)
    base.EXPECTED_CHANNELS = channels
    base.combine_output = combine_output
    base.validate_output = validate_output
    return base.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
