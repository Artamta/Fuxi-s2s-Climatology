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
