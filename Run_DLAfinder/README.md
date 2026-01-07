# Unified DESI DLA Finder Runner

This folder provides a unified runner that supports both mock and observational data.
It handles:
1) sightline generation (from spectra FITS),
2) prediction (GPU by default), and
3) DLA catalog generation.

Main entrypoint:
- `desi_DLAfinder_run.py`

Submit template:
- `submit_desi_DLAfinder_run.sh`

## Quick Start (Mock)

```bash
python3 desi_DLAfinder_run.py \
  --data-type mock \
  --spectra-root /path/to/mock/spectra-16 \
  --sightline-root /global/cfs/cdirs/desi/users/tingtan/DLA_finder/mocks/y3_saclay/sightlines \
  --list-cache-root /global/u1/t/tanting/DESI_analysis/DESI_CNN_DLA/data \
  --release y3_saclay \
  --value 0 --length 10 \
  --batch-size 256 --max-windows 8192 \
  --scratch-out /pscratch/sd/t/tanting/DLAfinder_out/y3_saclay \
  --generate-sightlines
```

## Quick Start (Data)

```bash
python3 desi_DLAfinder_run.py \
  --data-type data \
  --spectra-root /global/cfs/cdirs/desi/spectro/redux/kibo/healpix/main/dark \
  --sightline-root /global/cfs/cdirs/desi/users/tingtan/DLA_finder/data/loa/sightlines_main_dark_v0 \
  --list-cache-root /global/u1/t/tanting/DESI_analysis/DESI_CNN_DLA/data \
  --release loa --survey main --program dark --version v0 \
  --value 0 --length 10 \
  --batch-size 256 --max-windows 8192 \
  --scratch-out /pscratch/sd/t/tanting/DLAfinder_out/loa \
  --generate-sightlines
```

## What The Runner Does

1) Build or load a cached file list
   - Uses `--spectra-root` to discover input FITS files.
   - Stores a cached list in `--list-cache-root` as `filelist_<tag>.npz`.
   - This cache avoids rescanning tens of thousands of directories on every run.
2) Generate sightlines (optional)
   - Enabled by `--generate-sightlines`.
   - Skips existing sightlines unless `--force-sightlines` is set.
3) Predict + catalog
   - GPU by default; use `--cpu-only` for CPU.
   - Writes prediction `.npy` and DLA catalog `.fits`.

## File Layout Defaults

The runner uses default filename patterns per data type; override with flags if needed.

### Mock defaults
- Spectra: `spectra-16-{id}.fits`
- ZBEST: `zbest-16-{id}.fits`
- Truth: `truth-16-{id}.fits`
- Sightlines: `sightlines-{id}.npy`
- Pred: `sightlines-pred_gpu-{id}.npy`
- DLA catalog: `dlacat_gpu-{id}.fits`
- Output layout: `k/j` (two-level directory under `--sightline-root`)

### Data defaults
- Spectra: `spectra-main-dark-{id}.fits.gz`
- ZBEST: `zbest-16-{id}.fits`
- Truth: (empty)
- Sightlines: `{id}-pre-sightlines.npy`
- Pred: `{id}-pre-sightlines-pred.npy`
- DLA catalog: `{id}-dlacat.fits`
- Output layout: `k` (one-level directory under `--sightline-root`)

You can override any pattern:
- `--spectra-pattern`
- `--zbest-pattern`
- `--truth-pattern`
- `--sightline-pattern`
- `--pred-pattern`
- `--dlacat-pattern`
- `--output-layout` (`k` or `k/j`)

## Parameters (Common)

- `--data-type` `mock|data`
- `--spectra-root` root directory of input spectra
- `--sightline-root` root directory for generated sightlines
- `--list-cache-root` directory for cached lists (required; keeps run startup fast)
- `--release`, `--survey`, `--program`, `--version` (for list cache tag)
- `--value` start index (default 0)
- `--length` number of files (default: run to end)
- `--batch-size`, `--max-windows` (GPU batch tuning)
- `--scratch-out` alternate output root for predictions and catalogs
- `--generate-sightlines` generate sightlines if missing
- `--force-sightlines` always regenerate sightlines
- `--cpu-only` disable GPU
- `--stack-dlacat` stack per-file catalogs into one FITS
- `--stack-output` output path for stacked catalog (default: `dlacat.fits`)
- `--stack-scope` `range|all` (stack current range or all cached files)

## Cached File List

The runner stores a cached list in `--list-cache-root`:
- `filelist_<tag>.npz`

The tag is derived from `data-type`, `release`, `survey`, `program`, and `version`,
or can be overridden by `--list-cache-tag`.
Use `--rebuild-list` to force regeneration.

## Submit Script (GPU, 4 tasks)

See `submit_desi_DLAfinder_run.sh` for a template that:
- runs 4 tasks per node (1 GPU per task),
- splits the range across tasks, and
- logs each task to a separate file.

## Tips

- For A100 40GB, start with `--batch-size 256` and `--max-windows 8192`.
- If you see out-of-memory, reduce `--max-windows` first.
- For performance tests, you can skip catalog generation by commenting out
  `save_pred_all` in the runner.
