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

