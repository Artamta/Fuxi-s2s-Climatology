# Run Manifest

## 2026-07-09: June 17 IC Inputs

Generated exact June 17 FuXi-S2S `input.nc` files for:

`2002-2021`

Command:

```bash
/home/raj.ayush/.conda/envs/earth2/bin/python scripts/make_june17_inputs.py \
  --years 2002:2021 \
  --mmdd 0617 \
  --output-dir data/ic_inputs \
  --timeout 1800
```

Output pattern:

`data/ic_inputs/YYYY0617/input.nc`

Validation command:

```bash
python scripts/validate_fuxi_inputs.py \
  --input-dir data/ic_inputs \
  --years 2002:2021 \
  --mmdd 0617
```

Validation result:

`20/20 ok`

Each file is about `16.9 MiB` and has expected FuXi input dimensions:

`time=2, channel=76, lat=121, lon=240`

Generated data are ignored by git through `.gitignore`.

## Smoke Test

Reusable one-member smoke test:

```bash
sbatch slurm/smoke_fuxi_one_member.sbatch
```

Default test:

- date: `20190617`
- members: `1`
- lead steps: `2`
- partition: `GPU-AI_prio`
- output: `outputs/smoke/20190617/raw/20190617/member/00/{01,02}.nc`

Completed run:

- date: `2026-07-09`
- job id: `65442`
- node: `gpu2`
- result: `SMOKE CHECK OK`
- files:
  - `outputs/smoke/20190617/raw/20190617/member/00/01.nc`
  - `outputs/smoke/20190617/raw/20190617/member/00/02.nc`

The smoke output files are generated artifacts and remain ignored by git.

## Full June 17 Forecast Plan

SLURM array script:

```bash
mkdir -p /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/logs
sbatch slurm/run_june17_forecasts.sbatch
```

Recommended climatology run:

- dates: all `20` June 17 ICs from `2002-2021`
- members: `11`
- steps: `42`
- partition: `GPU-AI_prio`
- output root: `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/raw/YYYY0617`
- expected files: `9,240`
- expected storage: about `76 GiB`
- expected runtime after scheduling: roughly `1-2 h` for the serial array, depending on GPU and filesystem speed

Control-only diagnostic option:

```bash
FUXI_MEMBERS=1 sbatch slurm/run_june17_forecasts.sbatch
```

Control-only estimate:

- expected files: `840`
- expected storage: about `7 GiB`
- expected runtime after scheduling: roughly `5-15 min`

Full ensemble option:

```bash
FUXI_MEMBERS=50 sbatch slurm/run_june17_forecasts.sbatch
```

Full ensemble estimate:

- expected files: `42,000`
- expected storage: about `346 GiB`
- expected runtime after scheduling: roughly `1-4 h`, depending on GPU concurrency and filesystem speed

The repository is on `/home`, which is almost full, so full forecast output must stay under `/storage`.

Note: array job `65459` was a control-only launch and was canceled after confirming the workflow, because the climatology target was changed to 11 members.

## 2026-07-09: June 17 Model Climatology And 20260617 Analysis

Verified completed June-17 hindcast output:

- dates: `20/20` present for `2002-2021`
- members: `00-10`
- lead files: `42/42` for every member-year
- complete member-years: `220/220`

Reusable climatology command:

```bash
python scripts/build_june17_climatology.py \
  --years 2002:2021 \
  --members 0:10 \
  --variables tp,t2m \
  --output /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/climatology_fuxi17june.nc \
  --workers 8
```

Output:

`/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/climatology_fuxi17june.nc`

Dimensions:

`variable=2, lead_day=42, week=6, lat=22, lon=21`

20260617 ensemble analysis command:

```bash
python scripts/make_fuxi_weekly_analysis.py \
  --ic-date 20260617 \
  --members 0:49 \
  --variables tp,t2m \
  --climatology /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/climatology_fuxi17june.nc \
  --output /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --workers 8
```

Output:

`/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc`

Quick range check after FuXi `tp * 24` conversion to `mm/day`:

- climatology `tp`: min `0.1154`, mean `2.5179`, max `10.9880`
- climatology `t2m`: min `-8.82`, mean `22.72`, max `34.41`
- forecast `tp`: min `0.0068`, mean `6.2258`, max `45.4889`
- anomaly `tp`: min `-3.0883`, mean `3.7079`, max `40.4990`
- forecast `t2m`: min `-8.00`, mean `23.05`, max `38.60`
- anomaly `t2m`: min `-7.79`, mean `0.34`, max `7.10`

Plot command:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --output-dir outputs/weekly_analysis_20260617
```

Strict IMD-scale plot command:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --rainfall-scale imd \
  --temperature-actual-scale tmax \
  --output-dir outputs/weekly_analysis_20260617_imdscale
```

Plots:

- `outputs/weekly_analysis_20260617/fuxi_tp_forecast_20260617_6week.png`
- `outputs/weekly_analysis_20260617/fuxi_tp_anomaly_20260617_6week.png`
- `outputs/weekly_analysis_20260617/fuxi_t2m_forecast_20260617_6week.png`
- `outputs/weekly_analysis_20260617/fuxi_t2m_anomaly_20260617_6week.png`

Strict IMD-scale plots:

- `outputs/weekly_analysis_20260617_imdscale/fuxi_tp_forecast_20260617_6week.png`
- `outputs/weekly_analysis_20260617_imdscale/fuxi_tp_anomaly_20260617_6week.png`
- `outputs/weekly_analysis_20260617_imdscale/fuxi_t2m_forecast_20260617_6week.png`
- `outputs/weekly_analysis_20260617_imdscale/fuxi_t2m_anomaly_20260617_6week.png`

IMD colorbar sampling source:

- page: `https://mausam.imd.gov.in/responsive/extendedRangeForecast.php`
- sampled assets: `rfactual_MME2026070100.png`, `rfanom_MME2026070100.png`, `tmaxactual_MME2026070100.png`, `tminactual_MME2026070100.png`, `tmaxanom_MME2026070100.png`, `tminanom_MME2026070100.png`

Rainfall strict scale:

- actual: `0,2,5,10,20,40`
- anomaly: `-20,-15,-10,-5,-2,2,5,10,15,20`

Note: FuXi output contains `t2m` and `tp`; no true Tmin/Tmax channel exists in these files.

## 2026-07-09: PPT-Ready ERPAS Subset

Reference PDF:

`imd-ppt/ERPAS_Real-Time_Forecast_20260617 - Google Slides.pdf`

Relevant PDF slides:

- slide 6: `Predicted week wise rainfall (by MPME)`
- slide 14: `Predicted week wise temperature actual (by MPME)`
- slide 15: `Predicted week wise temperature anomaly (by MPME)`

PPT-ready command:

```bash
python scripts/plot_fuxi_ppt_weekly.py \
  --analysis-file /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --output-dir outputs/ppt_ready_20260617
```

Readable-rainfall command:

```bash
python scripts/plot_fuxi_ppt_weekly.py \
  --analysis-file /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --rainfall-scale fuxi \
  --output-dir outputs/ppt_ready_20260617_readable
```

PPT-ready outputs:

- `outputs/ppt_ready_20260617/fuxi_ppt_weekwise_rainfall_20260617.png`
- `outputs/ppt_ready_20260617/fuxi_ppt_weekwise_t2m_actual_20260617.png`
- `outputs/ppt_ready_20260617/fuxi_ppt_weekwise_t2m_anomaly_20260617.png`
- `outputs/ppt_ready_20260617_readable/fuxi_ppt_weekwise_rainfall_fuxiscale_20260617.png`

Whole-region PPT-ready command:

```bash
python scripts/plot_fuxi_ppt_weekly.py \
  --analysis-file /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --no-mask-to-india \
  --output-dir outputs/ppt_ready_20260617_whole_region
```

Whole-region PPT-ready outputs:

- `outputs/ppt_ready_20260617_whole_region/fuxi_ppt_weekwise_rainfall_20260617.png`
- `outputs/ppt_ready_20260617_whole_region/fuxi_ppt_weekwise_t2m_actual_20260617.png`
- `outputs/ppt_ready_20260617_whole_region/fuxi_ppt_weekwise_t2m_anomaly_20260617.png`

Quick summary CSV:

`/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_summary_20260617.csv`

The CSV is only for quick spatial min/mean/max checking. Gridded forecast, climatology, and anomaly fields are stored in NetCDF.

## 2026-07-09: Separate Six-Week Rainfall Maps

Readable FuXi-scale command:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --products tp_actual,tp_anomaly \
  --output-dir outputs/rainfall_6week_20260617_readable
```

Strict IMD-scale command:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --products tp_actual,tp_anomaly \
  --rainfall-scale imd \
  --output-dir outputs/rainfall_6week_20260617_imdscale
```

Whole-region strict IMD-scale command:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --products tp_actual,tp_anomaly \
  --rainfall-scale imd \
  --no-mask-to-india \
  --output-dir outputs/rainfall_6week_20260617_whole_region_imdscale
```

Outputs:

- `outputs/rainfall_6week_20260617_readable/fuxi_tp_actual_20260617_6week.png`
- `outputs/rainfall_6week_20260617_readable/fuxi_tp_anomaly_20260617_6week.png`
- `outputs/rainfall_6week_20260617_imdscale/fuxi_tp_actual_20260617_6week.png`
- `outputs/rainfall_6week_20260617_imdscale/fuxi_tp_anomaly_20260617_6week.png`
- `outputs/rainfall_6week_20260617_whole_region_imdscale/fuxi_tp_actual_20260617_6week.png`
- `outputs/rainfall_6week_20260617_whole_region_imdscale/fuxi_tp_anomaly_20260617_6week.png`

Strict ERPAS/IMD rainfall scale:

- actual: `0,2,5,10,20,40`
- anomaly: `-20,-15,-10,-5,-2,2,5,10,15,20`

Note: FuXi `tp` is clipped at zero and multiplied by `24` in the shared unit conversion before climatology, forecast, anomaly, CSV, and plots are written.

The whole-region outputs use `--no-mask-to-india`; shaded values remain visible over the full plotted domain while India/state boundaries are still overlaid.
