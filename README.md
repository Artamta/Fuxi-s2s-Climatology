# FuXi-S2S June 17 Hindcast Climatology

This repo tracks the scripts used to generate exact June 17 FuXi-S2S inputs and forecast products for model-relative anomaly plots.

Generated NetCDF/Zarr data are intentionally ignored by git. The working data root is:

`/home/raj.ayush/s2s/fuxi_s2s_Hindcast/data`

## Goal

For IMD-facing plots, generate FuXi forecasts initialized on June 17 for a 20-year hindcast set:

`2002-2021`

Then compute weekly rainfall forecasts and anomalies relative to FuXi's own June-17 model climatology.

## Step 1: Make June 17 Inputs

Create FuXi `input.nc` files for all years:

```bash
/home/raj.ayush/.conda/envs/earth2/bin/python \
  scripts/make_june17_inputs.py \
  --years 2002:2021 \
  --mmdd 0617 \
  --output-dir data/ic_inputs
```

Each output is:

`data/ic_inputs/YYYY0617/input.nc`

The input contains two time steps, previous day 00 UTC and init day 00 UTC, on the FuXi 1.5 degree grid with 76 channels.

Validate after generation:

```bash
python scripts/validate_fuxi_inputs.py --input-dir data/ic_inputs --years 2002:2021 --mmdd 0617
```

## FuXi Model Location

Existing model runner:

`/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/FuXi-S2S`

Model weights:

`/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/FuXi-S2S/model/fuxi_s2s.onnx`

Inference environment:

`/home/raj.ayush/.conda/envs/fuxi_s2s/bin/python`

## Step 2: Run June 17 Forecasts

The full forecast runner is a SLURM array over the 20 June 17 initial dates:

```bash
mkdir -p /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/logs
sbatch slurm/run_june17_forecasts.sbatch
```

Default recommended climatology run:

- years: `2002-2021`
- lead steps: `42`
- members: `11`
- partition: `GPU-AI_prio`
- output root: `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/raw/YYYY0617`

For a control-only diagnostic:

```bash
FUXI_MEMBERS=1 sbatch slurm/run_june17_forecasts.sbatch
```

For the full 50-member ensemble:

```bash
FUXI_MEMBERS=50 sbatch slurm/run_june17_forecasts.sbatch
```

Approximate output volume for all 20 years:

- `1` member: about `7 GiB`
- `11` members: about `76 GiB`
- `50` members: about `346 GiB`

For IMD-style model-climatology anomaly maps, use the same ensemble treatment for the forecast and climatology. The practical first target here is the 11-member ensemble mean; 50 members can be added later if ensemble spread/probability products or smoother ensemble statistics are needed.

## Quick One-Member India Plots

For a fast IMD-style check from one existing member:

```bash
python scripts/plot_one_member_india_forecast.py \
  --ic-date 20260617 \
  --member 0 \
  --variables tp,t2m \
  --output-dir outputs/quick_plots_imd6
```

The plotter creates six weekly panels, masks to India by default, and uses the local state boundary shapefile when available:

`/storage/raj.ayush/archive/s2s-forecast-/STATE_BOUNDARY.shp`

Use `--draw-districts` only when a dense district-boundary plot is needed.

## Step 3: Build Climatology And Anomalies

Build the reusable June-17 FuXi model climatology:

```bash
python scripts/build_june17_climatology.py \
  --years 2002:2021 \
  --members 0:10 \
  --variables tp,t2m \
  --output /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/climatology_fuxi17june.nc \
  --workers 8
```

Create the 20260617 ensemble forecast/anomaly product:

```bash
python scripts/make_fuxi_weekly_analysis.py \
  --ic-date 20260617 \
  --members 0:49 \
  --variables tp,t2m \
  --climatology /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/climatology_fuxi17june.nc \
  --output /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --workers 8
```

Plot the four six-week India maps:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --output-dir outputs/weekly_analysis_20260617
```

For strict IMD colorbar comparison, use the sampled IMD color blocks and IMD rainfall thresholds:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --rainfall-scale imd \
  --temperature-actual-scale tmax \
  --output-dir outputs/weekly_analysis_20260617_imdscale
```

Outputs:

- `fuxi_tp_forecast_20260617_6week.png`
- `fuxi_tp_anomaly_20260617_6week.png`
- `fuxi_t2m_forecast_20260617_6week.png`
- `fuxi_t2m_anomaly_20260617_6week.png`

The rainfall plot defaults to a FuXi-readable rainfall scale for quick field inspection. The `imd` rainfall option uses sampled ERPAS/IMD colors with actual rainfall levels `0,2,5,10,20,40` and anomaly levels `-20,-15,-10,-5,-2,2,5,10,15,20`.

The IMD colorbar colors were sampled from the current ERPAS image assets linked by the IMD extended-range page, including `rfactual`, `rfanom`, `tmaxactual`, `tminactual`, `tmaxanom`, and `tminanom`.

Rainfall figure subtitles/footers distinguish the products: forecast maps are labeled as the 50-member FuXi S2S ensemble weekly mean, while anomaly maps are labeled as that forecast weekly mean minus the 20-year FuXi S2S climatology for 2002-2021.

Note: these FuXi forecast files contain `t2m` and `tp`; they do not contain true daily Tmin/Tmax channels. The current temperature maps are therefore `t2m` actual and `t2m` anomaly, not Tmin/Tmax.

## PPT-Ready Four-Week Slides

To match the uploaded ERPAS Google Slides/PDF layout for the selected week-wise products:

```bash
python scripts/plot_fuxi_ppt_weekly.py \
  --analysis-file /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --output-dir outputs/ppt_ready_20260617
```

This writes:

- `fuxi_ppt_weekwise_rainfall_20260617.png`
- `fuxi_ppt_weekwise_t2m_actual_20260617.png`
- `fuxi_ppt_weekwise_t2m_anomaly_20260617.png`

The rainfall slide above uses the ERPAS/PPT rainfall thresholds: actual rainfall `0,2,5,10,20,40` and anomaly `-20,-15,-10,-5,-2,2,5,10,15,20`. A FuXi-scale rainfall version can also be made for checking internal spatial structure:

```bash
python scripts/plot_fuxi_ppt_weekly.py \
  --analysis-file /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --rainfall-scale fuxi \
  --output-dir outputs/ppt_ready_20260617_readable
```

For quick numeric checking, a compact weekly summary CSV was written to:

`/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_summary_20260617.csv`

For whole-region PPT-ready slides, keep the same rainfall slide layout and do not mask temperature fields to India:

```bash
python scripts/plot_fuxi_ppt_weekly.py \
  --analysis-file /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --no-mask-to-india \
  --output-dir outputs/ppt_ready_20260617_whole_region
```

## Separate Six-Week Rainfall Maps

For separate actual-rainfall and rainfall-anomaly figures through week 6:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --products tp_actual,tp_anomaly \
  --output-dir outputs/rainfall_6week_20260617_readable
```

Strict IMD-scale version:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --products tp_actual,tp_anomaly \
  --rainfall-scale imd \
  --output-dir outputs/rainfall_6week_20260617_imdscale
```

Whole-region strict IMD-scale version, without clipping the shaded field to India:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --products tp_actual,tp_anomaly \
  --rainfall-scale imd \
  --no-mask-to-india \
  --output-dir outputs/rainfall_6week_20260617_whole_region_imdscale
```

Four-week versions of the same whole-region rainfall maps can be written into the same folder:

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

## Whole-Region Six-Week Temperature Maps

For 2m temperature actual/anomaly maps over the full plotted region:

```bash
python scripts/plot_fuxi_weekly_analysis.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --products t2m_forecast,t2m_anomaly \
  --temperature-actual-scale legacy \
  --no-mask-to-india \
  --output-dir outputs/temp_6week_20260617_whole_region
```

FuXi-S2S raw output has `t2m`, but no true daily `tmin`/`tmax` channels. A proxy diagnostic can be made from the current files by taking the weekly min/max across the available daily 00Z `t2m` snapshots:

```bash
python scripts/make_fuxi_t2m_00z_extremes.py \
  --ic-date 20260617 \
  --members 0:49 \
  --climatology /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/june17/climatology_fuxi17june.nc \
  --output /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_t2m_00z_extremes_20260617.nc \
  --workers 8

python scripts/plot_fuxi_t2m_00z_extremes.py \
  /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_t2m_00z_extremes_20260617.nc \
  --no-mask-to-india \
  --output-dir outputs/temp_6week_20260617_whole_region_00z_extremes
```

Do not present the proxy product as true Tmin/Tmax; it is min/max of one daily 00Z `t2m` value per lead day.

## Case-Study Cumulative Rainfall Comparison

The existing final-paper case-study package has FuXi-S2S and ECMWF-S2S cumulative rainfall CSVs for IC `20260623`, plus a historical `20190627` case with IMD observed rainfall. To rebuild lightweight observed-availability comparison plots in this repo:

```bash
python scripts/make_case_study_cumulative_obs_compare.py \
  --init-date 20260623 \
  --current-date 2026-07-09 \
  --output-dir outputs/case_study_cumulative_obs_compare

python scripts/make_case_study_cumulative_obs_compare.py \
  --init-date 20190627 \
  --current-date 2019-08-08 \
  --output-dir outputs/case_study_cumulative_obs_compare
```

Current status for `20260623`: no local IMERG file was found/provided, local IMD observed rainfall only goes through `2025`, and ARCO ERA5 rejected the first valid date `2026-06-24` because its latest available date was `2026-06-17`. The script still writes a forecast/climatology figure plus an availability JSON so it can be rerun when observations arrive.

## June-17 IC Cumulative Rainfall

For the June-17 FuXi-S2S IC, make an all-India cumulative rainfall line plot directly from the 50-member raw forecast tree:

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

Current status: FuXi forecast, FuXi June-17 model climatology, and IMD 1991-2020 rainfall climatology are plotted. No real observed line is plotted yet because local IMD observed rainfall has no `2026` file, no local IMERG file was found/provided, local ERA5 files do not cover June-July 2026, and ARCO ERA5 was previously only available through `2026-06-17`.

## Notes

- Exact June 17 FuXi model forecast archives do not exist in the downloaded FuXi archive cadence, so we generate exact June 17 ICs ourselves.
- Input generation uses ARCO ERA5 through Earth2Studio and mirrors FuXi `data_util.make_input` conventions.
- FuXi `tp` is clipped at zero and multiplied by `24` to make the plotted/products units `mm/day`.
- `ttr` is divided by `3600`.

## May-17 Verification Case

For a previous-month case with more observation availability, generate a single FuXi IC for `20260517` and run a 50-member forecast:

```bash
/home/raj.ayush/.conda/envs/earth2/bin/python scripts/make_june17_inputs.py \
  --years 2026 \
  --mmdd 0517 \
  --output-dir /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/inputs \
  --timeout 1800

python scripts/validate_fuxi_inputs.py \
  --input-dir /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/inputs \
  --years 2026 \
  --mmdd 0517
```

Forecast output root:

`/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/raw/20260517`

Current status: `20260517` has a complete 50-member x 42-lead-day FuXi forecast under the May-17 storage folder, and the raw tree passed `scripts/check_smoke_output.py`.

Download ARCO ERA5 daily total precipitation truth for the same valid window:

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

Current ARCO status for this IC: complete daily `tp` truth is available for lead days `1-30` (`2026-05-18` through `2026-06-16`). Lead day `31` is incomplete because ARCO currently has only `2026-06-17 00Z`, not the full UTC day, and lead days `32-42` are unavailable. The NetCDF therefore stores only complete daily totals, with the unavailable days recorded in the CSV/JSON sidecars.

Download matching ECMWF-S2S `tp` forecast data for the same IC:

```bash
/home/raj.ayush/.conda/envs/fuxi/bin/python scripts/download_ecmwf_s2s_tp.py \
  --ic-date 20260517 \
  --lead-days 42 \
  --raw-dir /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/raw \
  --processed-dir /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/processed \
  --members 50
```

Raw outputs:

- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/raw/tp/20260517_cf.nc`
- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/raw/tp/20260517_pf.nc`

Processed comparable output:

- `/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/may17/ecmwf/processed/ecmwf_20260517_tp_ens50_lead42_india_1p5deg_daily_mm.nc`

The raw ECMWF `tp` files are accumulated precipitation in `kg m**-2`, which is numerically equivalent to millimetres of water. The processed file uses the first 50 perturbed members only, differences the cumulative field into daily increments, clips tiny negative packing artifacts to zero, and stores `tp(member, lead_time, lat, lon)` in `mm/day`.
