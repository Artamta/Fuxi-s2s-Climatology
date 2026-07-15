# June 17 Wind-Rainfall Anomaly Workflow

This workflow produces lead-matched FuXi-S2S 850-hPa wind-vector and rainfall
anomalies for the 17 June 2026 initialization.

## Science contract

- Issue time: `2026-06-17 00:00 UTC`.
- Inputs: complete UTC daily-mean ERA5 fields for 15 and 16 June. No data after
  the issue time are used.
- Forecast: 50 official stochastic FuXi-S2S members, 42 daily leads.
- Model climatology: exact 17 June forecasts for 2002-2021, with 11 members per
  year. Members are averaged within each year, then the 20 years are weighted
  equally.
- Grid: the native regional FuXi physics grid, 1.5 degrees (`39-0N`, `60-99E`).
- Retained variables: `u850`, `v850`, and `tp`.
- Weekly fields: arithmetic means over seven consecutive 24-hour lead periods.
- Wind anomaly: forecast-minus-climatology computed separately for `u850` and
  `v850`; arrows show that component-wise vector anomaly.
- Rainfall anomaly: forecast-minus-climatology after converting FuXi `tp` from
  `mm/hour` to `mm/day` by multiplying by 24.

## Run

The storage working root is:

```text
/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/daily_mean_strict00z_june17_wind
```

Submit the historical array, the target forecast, and then post-processing:

```bash
sbatch slurm/forecast_hist.sbatch
sbatch slurm/forecast_target.sbatch
sbatch slurm/postprocess.sbatch
```

The launchers use `config/run_config_hist11.json` and
`config/run_config_target50.json`. The forecast driver wraps the audited base
runner in `s2s_anlysis/clean/model-runs/fuxi` and retains only the three fields
needed by this diagnostic.

## Products

Post-processing writes the reusable analysis NetCDF, a west-coast diagnostic
CSV, and separate/combined four-week PNG and PDF figures. Presentation copies
are placed under:

```text
outputs/wind_rainfall_anomaly_20260617
```

Generated forecasts, NetCDF files, figures, and logs remain outside Git. The
run and plot manifests stored with those products preserve provenance.
