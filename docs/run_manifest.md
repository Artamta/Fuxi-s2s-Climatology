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

Figure text convention: forecast rainfall maps are labeled as the 50-member FuXi S2S ensemble weekly mean, while anomaly maps are labeled as the 50-member forecast weekly mean minus the 20-year FuXi S2S climatology for 2002-2021.

## 2026-07-09: Four-Week Whole-Region Rainfall Maps

Command:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --products tp_actual,tp_anomaly \
  --rainfall-scale imd \
  --no-mask-to-india \
  --weeks 1-4 \
  --product-title 'tp_actual=FuXi S2S Forecast' \
  --product-title 'tp_anomaly=FuXi S2S Anomaly' \
  --output-dir outputs/rainfall_6week_20260617_whole_region_imdscale
```

Outputs:

- `outputs/rainfall_6week_20260617_whole_region_imdscale/fuxi_tp_actual_20260617_4week.png`
- `outputs/rainfall_6week_20260617_whole_region_imdscale/fuxi_tp_anomaly_20260617_4week.png`

## 2026-07-09: Whole-Region Six-Week Temperature Maps

2m temperature actual/anomaly command:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --products t2m_forecast,t2m_anomaly \
  --temperature-actual-scale legacy \
  --no-mask-to-india \
  --output-dir outputs/temp_6week_20260617_whole_region
```

Outputs:

- `outputs/temp_6week_20260617_whole_region/fuxi_t2m_forecast_20260617_6week.png`
- `outputs/temp_6week_20260617_whole_region/fuxi_t2m_anomaly_20260617_6week.png`

FuXi-S2S raw channels include `t2m` but no true Tmin/Tmax channel. The following proxy product uses weekly min/max across the available daily 00Z `t2m` snapshots.

Proxy analysis command:

```bash
python scripts/make_fuxi_t2m_00z_extremes.py \
  --ic-date 20260617 \
  --members 0:49 \
  --climatology /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/climatology_fuxi17june.nc \
  --output /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_t2m_00z_extremes_20260617.nc \
  --workers 8
```

Proxy plot command:

```bash
python scripts/plot_fuxi_t2m_00z_extremes.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_t2m_00z_extremes_20260617.nc \
  --no-mask-to-india \
  --output-dir outputs/temp_6week_20260617_whole_region_00z_extremes
```

Proxy outputs:

- `outputs/temp_6week_20260617_whole_region_00z_extremes/fuxi_t2m_00z_min_actual_20260617_6week.png`
- `outputs/temp_6week_20260617_whole_region_00z_extremes/fuxi_t2m_00z_min_anomaly_20260617_6week.png`
- `outputs/temp_6week_20260617_whole_region_00z_extremes/fuxi_t2m_00z_max_actual_20260617_6week.png`
- `outputs/temp_6week_20260617_whole_region_00z_extremes/fuxi_t2m_00z_max_anomaly_20260617_6week.png`

Proxy range check:

- forecast `00Z t2m` weekly min: min `-8.47`, mean `22.60`, max `38.44`
- forecast `00Z t2m` weekly max: min `-7.49`, mean `23.46`, max `38.87`
- anomaly `00Z t2m` weekly min: min `-7.44`, mean `0.13`, max `7.09`
- anomaly `00Z t2m` weekly max: min `-8.16`, mean `0.51`, max `7.34`

Warning: the proxy product is not true daily Tmin/Tmax; it is min/max of one daily 00Z `t2m` value per lead day.

## 2026-07-09: Case-Study Cumulative Rainfall Observed Comparison

Source case-study folder:

`/home/raj.ayush/s2s/s2s_anlysis/final_paper/case-study`

Current 2026 ECMWF/FuXi package command:

```bash
python scripts/make_case_study_cumulative_obs_compare.py \
  --init-date 20260623 \
  --current-date 2026-07-09 \
  --output-dir outputs/case_study_cumulative_obs_compare
```

Historical observed comparison command:

```bash
python scripts/make_case_study_cumulative_obs_compare.py \
  --init-date 20190627 \
  --current-date 2019-08-08 \
  --output-dir outputs/case_study_cumulative_obs_compare
```

Outputs:

- `outputs/case_study_cumulative_obs_compare/20260623_cumulative_rainfall_fuxi_ecmwf_available_obs.png`
- `outputs/case_study_cumulative_obs_compare/20260623_cumulative_rainfall_available_obs.csv`
- `outputs/case_study_cumulative_obs_compare/20260623_observation_availability.json`
- `outputs/case_study_cumulative_obs_compare/20190627_cumulative_rainfall_fuxi_ecmwf_available_obs.png`
- `outputs/case_study_cumulative_obs_compare/20190627_cumulative_rainfall_available_obs.csv`
- `outputs/case_study_cumulative_obs_compare/20190627_observation_availability.json`

Availability result for IC `20260623`:

- local IMD observed rainfall files are present only through `2025`
- no local IMERG file was found or provided
- ARCO ERA5 rejected `2026-06-24`; latest available date seen was `2026-06-17`, before the valid window starts

Availability result for IC `20190627`:

- historical IMD observed cumulative rainfall is present for all `42` lead days

Note: the 2026 product is therefore forecast/climatology-only for now, with an availability JSON. Rerun the same command when IMERG or newer ERA5 data is available.

## 2026-07-09: June-17 IC Cumulative Rainfall

Command:

```bash
python scripts/make_june17_cumulative_rainfall.py \
  --ic-date 20260617 \
  --members 0:49 \
  --current-date 2026-07-09 \
  --output-dir outputs/june17_cumulative_rainfall
```

Outputs:

- `outputs/june17_cumulative_rainfall/20260617_india_cumulative_rainfall_fuxi_available_truth.png`
- `outputs/june17_cumulative_rainfall/20260617_india_cumulative_rainfall.csv`
- `outputs/june17_cumulative_rainfall/20260617_truth_availability.json`

Inputs:

- FuXi raw forecast: `/storage/raj.ayush/All_Model_Data/fuxi/test/raw/20260617`
- FuXi members: `00-49`
- FuXi samples: `50 members x 42 lead days`
- FuXi rainfall units: `mm/day = clip(raw tp, 0) * 24`
- India mask: local India/state shapefile through `prepare_india_geometries`

Final 42-day all-India cumulative totals:

- FuXi ensemble mean: `267.95 mm`
- FuXi p10-p90 member range: `236.78-297.72 mm`
- FuXi member 00: `273.71 mm`
- FuXi June-17 model climatology: `103.49 mm`
- IMD 1991-2020 climatology: `340.09 mm`

Truth availability on `2026-07-09`:

- local IMD observed status: `2026:missing`
- observed days plotted: `0`
- local IMERG: not found/provided
- local ERA5: does not cover the June-July 2026 valid window
- ARCO ERA5: previously seen available only through `2026-06-17`, before the valid window starts
- GFS: not found locally

Note: the figure is ready for partial verification. When IMERG, ERA5, GFS analysis, or IMD observed rainfall is available, add the source and rerun the same workflow to draw the real cumulative line through the available dates.

## 2026-07-09: May-17 Verification IC And Forecast

Purpose: create a previous-month verification case where ground truth should be available for more of the 42-day forecast window than the June-17 real-time case.

Input command:

```bash
/home/raj.ayush/.conda/envs/earth2/bin/python scripts/make_june17_inputs.py \
  --years 2026 \
  --mmdd 0517 \
  --output-dir /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/inputs \
  --timeout 1800
```

Input written:

- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/inputs/20260517/input.nc`
- shape: `2 x 76 x 121 x 240`
- times: `2026-05-16 00Z`, `2026-05-17 00Z`
- grid: native FuXi `1.5 deg`

Validation command:

```bash
python scripts/validate_fuxi_inputs.py \
  --input-dir /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/inputs \
  --years 2026 \
  --mmdd 0517
```

Validation result:

- `20260517: OK - ok size=16.9 MiB`

Forecast submission:

```bash
sbatch \
  --job-name=fuxi_20260517_ens50 \
  --array=1-1%1 \
  --output=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/logs/fuxi_20260517_%A_%a.out \
  --error=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/logs/fuxi_20260517_%A_%a.err \
  --export=ALL,FUXI_DATES_FILE=/home/raj.ayush/s2s/fuxi_s2s_Hindcast/config/may17_dates_2026.txt,FUXI_INPUT_ROOT=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/inputs,FUXI_OUTPUT_ROOT=/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17,FUXI_MEMBERS=50,FUXI_STEPS=42,FUXI_SEED=42,FUXI_OVERWRITE=0,FUXI_CHECK_PY=/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python \
  slurm/run_june17_forecasts.sbatch
```

Submitted job:

- job id: `65914`
- state at launch check: `RUNNING` on `gpu1`
- model runtime: `8.9 min`
- model output: `50 members x 42 lead days = 2100 NetCDF files`
- output size: `18G`
- output root: `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/raw/20260517`
- manual validation: `SMOKE CHECK OK: 50 member(s) x 42 step(s)`

Note: job `65914` was launched before the runner default was patched to use the FuXi Python for post-run checking, so SLURM marked the batch step failed after the model completed. The raw forecast tree is complete and passed the manual checker.

## 2026-07-09: May-17 ARCO ERA5 Total Precipitation Truth

Purpose: download real daily total precipitation for verification of the May-17 FuXi-S2S forecast, using the same India-region domain as the plotting/verification workflow.

Command:

```bash
/home/raj.ayush/.conda/envs/earth2/bin/python scripts/download_arco_tp_truth.py \
  --ic-date 20260517 \
  --lead-days 42 \
  --output-dir /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/truth \
  --chunk-days 3 \
  --timeout 900 \
  --overwrite
```

Outputs:

- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/truth/arco_era5_tp_daily_20260517.nc`
- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/truth/arco_era5_tp_daily_20260517_summary.csv`
- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/truth/arco_era5_tp_daily_20260517_availability.json`

NetCDF contents:

- variable: `tp_daily`
- units: `mm/day`
- dimensions: `lead_day=30, lat=135, lon=129`
- valid dates: `2026-05-18` through `2026-06-16`
- method: sum of 24 hourly ARCO ERA5 `tp` fields from `00Z` through `23Z` UTC, clipped at zero and converted from metres to millimetres
- bbox: `66.5E-98.5E, 5.0N-38.5N`

Availability result:

- complete lead days: `30`
- missing or incomplete lead days: `12`
- lead day `31` (`2026-06-17`) is incomplete because ARCO currently exposes only `2026-06-17 00Z`, not all 24 hours
- lead days `32-42` (`2026-06-18` through `2026-06-28`) are unavailable from ARCO at the time of this run

Range check over the downloaded domain:

- min: `0.00 mm/day`
- mean: `4.78 mm/day`
- max: `283.12 mm/day`

## 2026-07-09: May-17 ECMWF-S2S Total Precipitation Forecast

Purpose: download ECMWF-S2S `tp` for the same `20260517` IC as the FuXi verification run and prepare a FuXi-comparable daily rainfall product.

Command:

```bash
/home/raj.ayush/.conda/envs/fuxi/bin/python scripts/download_ecmwf_s2s_tp.py \
  --ic-date 20260517 \
  --lead-days 42 \
  --raw-dir /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/raw \
  --processed-dir /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/processed \
  --members 50 \
  --retries 2 \
  --sleep-between 1 \
  --overwrite
```

Raw outputs:

- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/raw/tp/20260517_cf.nc`
- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/raw/tp/20260517_pf.nc`

Processed outputs:

- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/processed/ecmwf_20260517_tp_ens50_lead42_india_1p5deg_daily_mm.nc`
- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/processed/ecmwf_20260517_tp_download_manifest.json`
- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/processed/ecmwf_20260517_tp_ens50_lead42_summary.csv`

NetCDF contents:

- raw `cf`: `step=42, latitude=27, longitude=27`
- raw `pf`: `number=100, step=42, latitude=27, longitude=27`
- processed: `member=50, lead_time=42, lat=27, lon=27`
- valid dates: `2026-05-18` through `2026-06-28`
- grid/domain: `1.5 deg`, `40N-0N, 60E-100E`

Method:

- raw ECMWF `tp` is accumulated precipitation in `kg m**-2`, numerically equal to millimetres of water
- processed file uses the first 50 perturbed ECMWF members to match the FuXi 50-member case
- daily `mm/day` is produced by differencing accumulated `tp` along lead time
- tiny negative packing artifacts are clipped to zero

Range check for processed daily rainfall:

- min: `0.00 mm/day`
- mean: `4.91 mm/day`
- max: `533.52 mm/day`

Consistency check:

- cumulative sum of the processed daily increments reproduces the raw accumulated ECMWF field for the first 50 perturbed members
- maximum absolute float32 difference: `0.086 mm`

Comparison window:

- ECMWF forecast is present for all `42` lead days
- ARCO ERA5 truth is complete for lead days `1-30`, so clean ECMWF/FuXi/truth verification is currently lead days `1-30`

## 2026-07-09: May-17 FuXi/ECMWF/ERA5 GT Verification Plots

Purpose: make a compact verification package for the clean May-17 observed window where FuXi, ECMWF, and ERA5 ground truth are all available.

Command:

```bash
python scripts/plot_may17_fuxi_ecmwf_arco.py \
  --ic-date 20260517 \
  --lead-days 30 \
  --imd-climatology /storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/imd_rain_1991_2020_daily_climatology.nc \
  --output-dir outputs/may17_fuxi_ecmwf_arco
```

Inputs:

- FuXi raw forecast: `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/raw/20260517`
- FuXi members: `00-49`
- ECMWF processed forecast: `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/processed/ecmwf_20260517_tp_ens50_lead42_india_1p5deg_daily_mm.nc`
- ERA5 GT source: `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/truth/arco_era5_tp_daily_20260517.nc`
- IMD rainfall climatology: `/storage/raj.ayush/All_Model_Data/ground_truth/imd_rainfall/climatology/imd_rain_1991_2020_daily_climatology.nc`

Outputs:

- `outputs/may17_fuxi_ecmwf_arco/20260517_lead1_30_cumulative_rainfall_fuxi_ecmwf_arco.png`
- `outputs/may17_fuxi_ecmwf_arco/20260517_lead1_30_cumulative_rainfall_paperstyle_fuxi_ecmwf_arco.png`
- `outputs/may17_fuxi_ecmwf_arco/20260517_lead1_30_spatial_4panel_fuxi_ecmwf_arco.png`
- `outputs/may17_fuxi_ecmwf_arco/20260517_lead1_30_spatial_bias_4panel_fuxi_ecmwf_arco.png`
- `outputs/may17_fuxi_ecmwf_arco/20260517_lead1_30_cumulative_timeseries.csv`
- `outputs/may17_fuxi_ecmwf_arco/20260517_lead1_30_spatial_fields.nc`
- `outputs/may17_fuxi_ecmwf_arco/20260517_lead1_30_verification_manifest.json`

Figure set:

- cumulative all-India rainfall line plot for lead days `1-30`, now including the IMD 1991-2020 climatology reference
- paper-style cumulative all-India rainfall plot matching the final-paper case-study layout, including endpoint labels, FuXi member 00, ECMWF control, and IMD climatology
- four-panel spatial cumulative rainfall map: ERA5 GT, FuXi mean, ECMWF mean, FuXi minus ECMWF
- four-panel spatial bias map: ERA5 GT, FuXi minus ERA5 GT, ECMWF minus ERA5 GT, FuXi minus ECMWF

Final all-India cumulative rainfall through lead day `30`:

- ERA5 GT: `70.74 mm`
- FuXi-S2S ensemble mean: `33.91 mm`
- FuXi member 00: `64.30 mm`
- ECMWF-S2S ensemble mean: `95.13 mm`
- ECMWF control: `87.06 mm`
- IMD 1991-2020 climatology: `96.33 mm`

Notes:

- line-plot means are area-weighted over India using the local India/state shapefile geometry
- IMD climatology is a 1991-2020 daily rainfall normal for the same valid month-days (`18 May-16 Jun`), not an observed truth line
- spatial maps shade the full plotted domain by default and overlay India/state boundaries
- ERA5 GT is interpolated to the model grid only for bias panels and cached fields
