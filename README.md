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

## Notes

- Exact June 17 FuXi model forecast archives do not exist in the downloaded FuXi archive cadence, so we generate exact June 17 ICs ourselves.
- Input generation uses ARCO ERA5 through Earth2Studio and mirrors FuXi `data_util.make_input` conventions.
- `tp` is converted from m to mm and clipped to `[0, 1000]`.
- `ttr` is divided by `3600`.

