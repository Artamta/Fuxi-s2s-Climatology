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

The rainfall plot defaults to a FuXi-readable rainfall scale because the 20260617 FuXi ensemble rain values are much smaller than the strict ERPAS/IMD scale. The `imd` rainfall option uses sampled ERPAS/IMD colors with actual rainfall levels `0,2,5,10,20,40` and anomaly levels `-20,-15,-10,-5,-2,2,5,10,15,20`.

The IMD colorbar colors were sampled from the current ERPAS image assets linked by the IMD extended-range page, including `rfactual`, `rfanom`, `tmaxactual`, `tminactual`, `tmaxanom`, and `tminanom`.

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

The rainfall slide above uses the ERPAS/PPT rainfall thresholds: actual rainfall `0,2,5,10,20,40` and anomaly `-20,-15,-10,-5,-2,2,5,10,15,20`. Because FuXi rainfall is much smaller, a presentation-readable rainfall version can also be made:

```bash
python scripts/plot_fuxi_ppt_weekly.py \
  --analysis-file /storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_analysis_20260617.nc \
  --rainfall-scale fuxi \
  --output-dir outputs/ppt_ready_20260617_readable
```

For quick numeric checking, a compact weekly summary CSV was written to:

`/storage/raj.ayush/fuxi_s2s_Hindcast_outputs/analysis/fuxi_weekly_summary_20260617.csv`

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

## Notes

- Exact June 17 FuXi model forecast archives do not exist in the downloaded FuXi archive cadence, so we generate exact June 17 ICs ourselves.
- Input generation uses ARCO ERA5 through Earth2Studio and mirrors FuXi `data_util.make_input` conventions.
- `tp` is converted from m to mm and clipped to `[0, 1000]`.
- `ttr` is divided by `3600`.
