# FuXi Input Conventions

The June 17 input builder mirrors the existing local FuXi utility:

`/home/raj.ayush/s2s/s2s_anlysis/analysis-code/data-download/fuxi_s2s/FuXi-S2S/data_util.py`

## Dimensions

`input.nc` is an xarray `DataArray` with:

`time, channel, lat, lon`

Expected shape:

`2, 76, 121, 240`

The two times are:

- previous day at 00 UTC
- initialization day at 00 UTC

## Channel Order

Pressure-level groups are ordered by variable, then level descending from 1000 hPa to 50 hPa:

`z1000 ... z50, t1000 ... t50, u1000 ... u50, v1000 ... v50, q1000 ... q50`

Surface channels:

`t2m, d2m, sst, ttr, 10u, 10v, 100u, 100v, msl, tcwv, tp`

## Unit Transforms

- `tp`: ARCO ERA5 metres to millimetres, clipped to `[0, 1000]`
- `ttr`: divided by `3600`

