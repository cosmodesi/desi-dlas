# Unified DESI DLA Finder Runner

Single entrypoint for both mock and observational data. It handles:
- sightline generation (from spectra FITS)
- prediction (GPU by default)
- DLA catalog generation
- optional stacking of per-file catalogs

Main entrypoint:
- `Run_DLAfinder/desi_DLAfinder_run.py`

Submit template:
- `Run_DLAfinder/submit_desi_DLAfinder_run.sh` (set `REPO_ROOT` to your local `desi-dlas` path)

## Environment Setup (Perlmutter)

Create a GPU environment (choose your own path):

```bash
module load python
conda create -y -p /path/to/conda_envs/CNN_GPU python=3.10
source activate /path/to/conda_envs/CNN_GPU
pip install 'tensorflow[and-cuda]==2.15.*'
```

## Decide Your Run Settings

You need these choices for any run:
- data type: `mock` or `data`
- release label (used in cache tag and output naming)
- spectra root (where input FITS live)
- sightline root (where sightlines are written)
- list cache root (where the file list cache is stored)
- scratch output (optional; where predictions and catalogs are written)

Create (or confirm) these directories exist:
- `--sightline-root`
- `--list-cache-root`
- `--scratch-out` (if used)

## Path Conventions

Mock (layout `k/j`):
- spectra: `<spectra-root>/<k>/<j>/spectra-16-<j>.fits`
- zbest: `<spectra-root>/<k>/<j>/zbest-16-<j>.fits`
- sightlines: `<sightline-root>/<k>/<j>/sightlines-<j>.npy`
- pred: `<scratch-out or sightline-root>/<k>/<j>/sightlines-pred_gpu-<j>.npy`
- dlacat: `<scratch-out or sightline-root>/<k>/<j>/dlacat_gpu-<j>.fits`

Data (layout `k`):
- spectra: `<spectra-root>/<k>/<j>/spectra-main-dark-<j>.fits.gz`
- zbest: `<spectra-root>/<k>/<j>/zbest-16-<j>.fits`
- sightlines: `<sightline-root>/<k>/<j>-pre-sightlines.npy`
- pred: `<scratch-out or sightline-root>/<k>/<j>-pre-sightlines-pred.npy`
- dlacat: `<scratch-out or sightline-root>/<k>/<j>-dlacat.fits`

If your filenames differ, override with the `--*-pattern` flags.

## Quick Start (Minimal Commands)

### Mock (only three required paths)

Required paths:
- `--spectra-root`: mock spectra root
- `--sightline-root`: output sightlines root
- `--list-cache-root`: cache directory (keeps startup fast)

```bash
python3 desi_DLAfinder_run.py \
  --data-type mock \
  --release <release_name> \
  --spectra-root <mock_spectra_root> \
  --sightline-root <sightline_output_root> \
  --list-cache-root <list_cache_root> \
  --generate-sightlines
```

Optional:
- `--scratch-out` to write predictions/catalogs to a separate location
- `--batch-size` and `--max-windows` for GPU tuning (defaults: 512 / 16384)

### CPU Sightline Generation (Fast, GPU-Free)

Use the dedicated CPU-only helper to generate sightlines without touching GPUs:

```bash
module load python
conda activate /global/cfs/cdirs/desi/users/tingtan/conda_envs/CNN_GPU

python3 desi_DLAfinder_make_sightlines_cpu.py \
  --data-type mock \
  --spectra-root <mock_spectra_root> \
  --sightline-root <sightline_output_root> \
  --list-cache-root <list_cache_root> \
  --release <release_name> \
  --workers 64
```

Notes:
- Uses the same file-list cache naming as the unified runner, so you can reuse `--list-cache-root`.
- Set `--force-sightlines` to overwrite existing sightlines.

## Use Retrained Models (Optional)

By default, prediction uses the legacy checkpoints. To switch to retrained models,
set environment variables before running:

```bash
export DESIDLAS_CKPT_LOW=/pscratch/sd/t/<user>/retraining/models/low/current_135000
export DESIDLAS_CKPT_MID=/pscratch/sd/t/<user>/retraining/models/mid/current_99999
```

Unset them to return to the default models.

### Data (minimal + survey/program/version)

```bash
python3 desi_DLAfinder_run.py \
  --data-type data \
  --release <release_name> --survey <survey> --program <program> --version <version> \
  --spectra-root <data_spectra_root> \
  --sightline-root <sightline_output_root> \
  --list-cache-root <list_cache_root> \
  --generate-sightlines
```

Optional:
- `--scratch-out` to write predictions/catalogs to a separate location
- `--batch-size` and `--max-windows` for GPU tuning (defaults: 512 / 16384)

## What The Runner Does

1) Builds or loads a cached file list (`filelist_<tag>.npz`).
2) Generates sightlines if `--generate-sightlines` is set.
3) Runs prediction and writes per-file catalogs.
4) Optionally stacks per-file catalogs into one FITS.

If you change data type or filename patterns, run once with `--rebuild-list`.

## Stacking Catalogs

Enable stacking at the end of a run:

```bash
--stack-dlacat --stack-scope all
```

Defaults:
- output path: `<scratch-out or sightline-root>/dlacat.fits`
- scope: `all` (use `--stack-scope range` to stack only the current chunk)

## Parameters (Common)

- `--data-type` `mock|data`
- `--spectra-root` root directory of input spectra
- `--sightline-root` root directory for generated sightlines
- `--list-cache-root` directory for cached lists (required)
- `--release`, `--survey`, `--program`, `--version` (cache tag components)
- `--value` start index (default 0)
- `--length` number of files (default: run to end)
- `--batch-size`, `--max-windows` (GPU batch tuning; defaults: 512 / 16384)
- `--scratch-out` alternate output root for predictions and catalogs
- `--generate-sightlines` generate sightlines if missing
- `--force-sightlines` always regenerate sightlines
- `--cpu-only` disable GPU
- `--stack-dlacat` stack per-file catalogs into one FITS
- `--stack-output` output path for stacked catalog
- `--stack-scope` `range|all`

## Submit Script (GPU, 4 tasks)

Use `submit_desi_DLAfinder_run.sh` to run 4 tasks per node (1 GPU each).
It splits the file list across tasks and writes per-task logs.

## Tips

- For A100 40GB, start with `--batch-size 512` and `--max-windows 16384`.
- If you see out-of-memory, reduce `--max-windows` first.
