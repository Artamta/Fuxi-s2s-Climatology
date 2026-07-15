#!/usr/bin/env python3
"""Build lead-matched June-17 FuXi wind and rainfall anomaly fields."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(
    "/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/"
    "daily_mean_strict00z_june17_wind"
)
HISTORICAL_YEARS = tuple(range(2002, 2022))
TARGET_YEAR = 2026
CHANNELS = ("u850", "v850", "tp")
EARTH_RADIUS_M = 6_371_000.0
WEST_COAST_BOX = (68.0, 78.0, 7.0, 17.0)


def forecast_path(year: int) -> Path:
    return ROOT / "forecasts" / f"annual{year}" / f"{year}0617.nc"


def validate_forecast(dataset: xr.Dataset, year: int) -> None:
    expected_members = 50 if year == TARGET_YEAR else 11
    expected_sizes = {
        "member": expected_members,
        "lead_day": 42,
        "latitude": 27,
        "longitude": 27,
    }
    for name, size in expected_sizes.items():
        if dataset.sizes.get(name) != size:
            raise ValueError(
                f"{year}: {name}={dataset.sizes.get(name)}, expected {size}"
            )
    if set(dataset.data_vars) != set(CHANNELS):
        raise ValueError(f"{year}: unexpected variables {sorted(dataset.data_vars)}")
    if dataset.attrs.get("benchmark_mode") != "strict_information_matched_00utc":
        raise ValueError(f"{year}: forecast is not strict information-matched 00 UTC")
    if dataset.attrs.get("input_daily_statistic") != "daily_mean":
        raise ValueError(f"{year}: forecast input is not a daily mean")
    if "official_onnx_native_stochastic" not in dataset.attrs.get(
        "member_generation", ""
    ):
        raise ValueError(f"{year}: forecast is not the official stochastic ensemble")
    expected_date = pd.Timestamp(f"{year}-06-17")
    if pd.Timestamp(dataset.forecast_reference_time.values) != expected_date:
        raise ValueError(f"{year}: wrong forecast-reference time")
    if pd.Timestamp(dataset.information_cutoff_time.values) != expected_date:
        raise ValueError(f"{year}: wrong information-cutoff time")


def converted(dataset: xr.Dataset, channel: str) -> xr.DataArray:
    values = dataset[channel].astype(np.float64)
    return values * 24.0 if channel == "tp" else values


def spherical_diagnostics(
    u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return horizontal divergence and vertical relative vorticity in s-1."""

    lat_rad = np.deg2rad(lat.astype(np.float64))
    lon_rad = np.deg2rad(lon.astype(np.float64))
    cos_lat = np.cos(lat_rad)[:, None]
    du_dlambda = np.gradient(u, lon_rad, axis=-1, edge_order=2)
    dv_dlambda = np.gradient(v, lon_rad, axis=-1, edge_order=2)
    d_vcos_dphi = np.gradient(v * cos_lat, lat_rad, axis=-2, edge_order=2)
    d_ucos_dphi = np.gradient(u * cos_lat, lat_rad, axis=-2, edge_order=2)
    divergence = (du_dlambda + d_vcos_dphi) / (EARTH_RADIUS_M * cos_lat)
    vorticity = (dv_dlambda - d_ucos_dphi) / (EARTH_RADIUS_M * cos_lat)
    return divergence, vorticity


def area_mean(field: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> float:
    lon_min, lon_max, lat_min, lat_max = WEST_COAST_BOX
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    subset = field[np.ix_(lat_mask, lon_mask)]
    weights = np.cos(np.deg2rad(lat[lat_mask]))[:, None]
    finite = np.isfinite(subset)
    return float(np.sum(np.where(finite, subset * weights, 0.0)) / np.sum(finite * weights))


def main() -> int:
    paths = [forecast_path(year) for year in (*HISTORICAL_YEARS, TARGET_YEAR)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing forecasts:\n" + "\n".join(missing))

    historical_sum = None
    target_daily = None
    latitude = longitude = None
    source_records: list[dict[str, object]] = []
    for year in (*HISTORICAL_YEARS, TARGET_YEAR):
        path = forecast_path(year)
        with xr.open_dataset(path) as dataset:
            validate_forecast(dataset, year)
            fields = np.stack(
                [
                    converted(dataset, channel).mean("member").values
                    for channel in CHANNELS
                ],
                axis=0,
            )
            if latitude is None:
                latitude = dataset.latitude.values.astype(np.float32)
                longitude = dataset.longitude.values.astype(np.float32)
            elif not np.allclose(latitude, dataset.latitude.values) or not np.allclose(
                longitude, dataset.longitude.values
            ):
                raise ValueError(f"{year}: physics-grid mismatch")
            source_records.append(
                {
                    "year": year,
                    "path": str(path),
                    "members": int(dataset.sizes["member"]),
                    "lead_days": int(dataset.sizes["lead_day"]),
                }
            )
            if year == TARGET_YEAR:
                target_daily = fields
            else:
                if historical_sum is None:
                    historical_sum = np.zeros_like(fields, dtype=np.float64)
                historical_sum += fields

    if target_daily is None or historical_sum is None:
        raise RuntimeError("target or historical forecast collection is empty")
    climatology_daily = historical_sum / len(HISTORICAL_YEARS)
    anomaly_daily = target_daily - climatology_daily

    weekly_shape = (len(CHANNELS), 6, len(latitude), len(longitude))
    forecast_weekly = np.empty(weekly_shape, dtype=np.float32)
    climatology_weekly = np.empty(weekly_shape, dtype=np.float32)
    anomaly_weekly = np.empty(weekly_shape, dtype=np.float32)
    for week in range(6):
        window = slice(week * 7, (week + 1) * 7)
        forecast_weekly[:, week] = target_daily[:, window].mean(axis=1)
        climatology_weekly[:, week] = climatology_daily[:, window].mean(axis=1)
        anomaly_weekly[:, week] = anomaly_daily[:, window].mean(axis=1)

    u_index = CHANNELS.index("u850")
    v_index = CHANNELS.index("v850")
    tp_index = CHANNELS.index("tp")
    vector_magnitude = np.hypot(
        anomaly_weekly[u_index], anomaly_weekly[v_index]
    ).astype(np.float32)
    divergence, vorticity = spherical_diagnostics(
        anomaly_weekly[u_index].astype(np.float64),
        anomaly_weekly[v_index].astype(np.float64),
        latitude,
        longitude,
    )

    issue_date = pd.Timestamp("2026-06-17")
    week_period_start = np.asarray(
        [issue_date + pd.Timedelta(days=7 * week) for week in range(6)],
        dtype="datetime64[ns]",
    )
    week_period_end = week_period_start + np.timedelta64(7, "D")
    created_utc = datetime.now(timezone.utc).isoformat()
    common_attrs = {
        "title": "FuXi-S2S June-17 weekly 850-hPa wind and rainfall anomalies",
        "ic_date": "20260617",
        "forecast_reference_time": "2026-06-17T00:00:00Z",
        "information_cutoff_time": "2026-06-17T00:00:00Z",
        "input_days": "2026-06-15,2026-06-16",
        "benchmark_mode": "strict_information_matched_00utc",
        "input_daily_statistic": "daily_mean",
        "forecast_members": "50",
        "climatology_years": ",".join(str(year) for year in HISTORICAL_YEARS),
        "climatology_members_per_year": "11",
        "climatology_method": (
            "member mean within each year followed by equal-weight mean across years"
        ),
        "member_generation": "official ONNX native stochastic latent-prior ensemble",
        "wind_units": "m s-1",
        "tp_units": "mm day-1 (FuXi mm h-1 daily-mean rate multiplied by 24)",
        "west_coast_box": "68-78E,7-17N",
        "created_utc": created_utc,
        "note": (
            "Weeks are means of seven consecutive 24-hour forecast periods. "
            "Wind anomalies subtract u and v climatologies component-wise."
        ),
    }
    analysis = xr.Dataset(
        {
            "forecast_weekly": (
                ("variable", "week", "lat", "lon"),
                forecast_weekly,
            ),
            "climatology_weekly": (
                ("variable", "week", "lat", "lon"),
                climatology_weekly,
            ),
            "anomaly_weekly": (
                ("variable", "week", "lat", "lon"),
                anomaly_weekly,
            ),
            "wind_vector_anomaly_magnitude": (
                ("week", "lat", "lon"),
                vector_magnitude,
            ),
            "wind_anomaly_divergence": (
                ("week", "lat", "lon"),
                divergence.astype(np.float32),
            ),
            "wind_anomaly_relative_vorticity": (
                ("week", "lat", "lon"),
                vorticity.astype(np.float32),
            ),
            "week_start_lead": (("week",), np.arange(1, 43, 7, dtype=np.int16)),
            "week_end_lead": (("week",), np.arange(7, 43, 7, dtype=np.int16)),
        },
        coords={
            "variable": list(CHANNELS),
            "week": np.arange(1, 7, dtype=np.int16),
            "lat": latitude,
            "lon": longitude,
            "week_period_start": ("week", week_period_start),
            "week_period_end": ("week", week_period_end),
        },
        attrs=common_attrs,
    )
    for name in ("wind_anomaly_divergence", "wind_anomaly_relative_vorticity"):
        analysis[name].attrs["units"] = "s-1"
    analysis["wind_vector_anomaly_magnitude"].attrs.update(
        units="m s-1",
        clarification="magnitude of the vector-component anomaly, not scalar wind-speed anomaly",
    )

    output = ROOT / "analysis" / "fuxi_weekly_wind_rainfall_anomaly_20260617.nc"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".nc.part")
    temporary.unlink(missing_ok=True)
    float_vars = [
        "forecast_weekly",
        "climatology_weekly",
        "anomaly_weekly",
        "wind_vector_anomaly_magnitude",
        "wind_anomaly_divergence",
        "wind_anomaly_relative_vorticity",
    ]
    analysis.to_netcdf(
        temporary,
        encoding={
            name: {"zlib": True, "complevel": 4, "dtype": "float32"}
            for name in float_vars
        },
    )
    temporary.replace(output)

    diagnostic_csv = ROOT / "analysis" / "west_coast_weekly_diagnostics.csv"
    with diagnostic_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "week",
                "period_start",
                "period_end_exclusive",
                "u850_anomaly_m_s",
                "v850_anomaly_m_s",
                "vector_anomaly_magnitude_m_s",
                "divergence_1e5_s-1",
                "relative_vorticity_1e5_s-1",
                "rainfall_anomaly_mm_day",
            ]
        )
        for week in range(6):
            u_mean = area_mean(anomaly_weekly[u_index, week], latitude, longitude)
            v_mean = area_mean(anomaly_weekly[v_index, week], latitude, longitude)
            writer.writerow(
                [
                    week + 1,
                    str(pd.Timestamp(week_period_start[week]).date()),
                    str(pd.Timestamp(week_period_end[week]).date()),
                    u_mean,
                    v_mean,
                    float(np.hypot(u_mean, v_mean)),
                    area_mean(divergence[week], latitude, longitude) * 1e5,
                    area_mean(vorticity[week], latitude, longitude) * 1e5,
                    area_mean(anomaly_weekly[tp_index, week], latitude, longitude),
                ]
            )

    manifest = ROOT / "analysis" / "build_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "created_utc": created_utc,
                "science_contract": common_attrs,
                "sources": source_records,
                "products": [str(output), str(diagnostic_csv)],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output}")
    print(f"wrote {diagnostic_csv}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
