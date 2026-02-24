# Unified DESI DLA Finder Runner

Single entrypoint for both mock and observational data. It handles:
- sightline generation (from spectra FITS)
- prediction (GPU by default)
- DLA catalog generation
- optional stacking of per-file catalogs

Main entrypoint:
- `Run_DLAfinder/desi_DLAfinder_run.py`

References:
- training overview: `TRAINING_OVERVIEW.md`
- retraining guide: `Run_DLAfinder/README-training.md`

## What The Runner Does

1. Builds or loads a cached file list (`filelist_<tag>.npz`).
2. Generates sightlines if `--generate-sightlines` is set.
3. Runs prediction and writes per-file catalogs.
4. Optionally stacks per-file catalogs into one FITS.

If you change data type or filename patterns, run once with `--rebuild-list`.

## Preparations Before Running

1. Clone repo and checkout the correct branch:

```bash
git clone https://github.com/cosmodesi/desi-dlas/
cd desi-dlas
git checkout main-unified
```

2. Create a GPU environment (choose your own path):

```bash
module load python
conda create -y -p /path/to/conda_envs/CNN_GPU python=3.10
source activate /path/to/conda_envs/CNN_GPU
pip install 'tensorflow[and-cuda]==2.15.*'
```

3. Install desi-dlas dependencies:

```bash
pip install -e /path/to/desi-dlas
```

If you cannot install editable, set:

```bash
export PYTHONPATH=/path/to/desi-dlas:$PYTHONPATH
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

## Running

### Quick Start (Minimal Commands)

This is for running only one sightline or a small subset.

### Mock (only three required paths)

Required paths:
- `--spectra-root`: mock spectra root
- `--sightline-root`: output sightlines root
- `--list-cache-root`: cache directory (keeps startup fast)

```bash
python3 Run_DLAfinder/desi_DLAfinder_run.py \
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

### Data (minimal + survey/program/version)

```bash
python3 Run_DLAfinder/desi_DLAfinder_run.py \
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

### Interactive Node Example (Mock, 4 GPUs)

You can run in an interactive node or save this block as a `.sh` script.

```bash
salloc -N 1 -C gpu -t 04:00:00 --gpus 4 --qos interactive --account desi_g

module load python
conda activate /global/cfs/cdirs/desi/users/tingtan/conda_envs/CNN_GPU

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=2
ulimit -n 65535

export RUNNER=/global/u1/t/tanting/DESI_analysis/desi-dlas/Run_DLAfinder/desi_DLAfinder_run.py
export SCR_OUT=/global/cfs/cdirs/desi/users/tingtan/DLA_finder/mocks/y3_saclay

# ---- Mock dataset ----
DATA_TYPE=mock
SPECTRA_ROOT=/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/spectra-16
SIGHTLINE_ROOT=$SCR_OUT/sightlines
LIST_CACHE_ROOT=$SCR_OUT/data
DLACAT_ROOT=$SCR_OUT/dlacat
export LIST_CACHE_ROOT=$SCR_OUT/data
export DESIDLAS_CKPT_LOW1=/global/cfs/cdirs/desi/users/tingtan/DLA_finder/retraining/models/low1/current_199999
export DESIDLAS_CKPT_LOW2=//global/cfs/cdirs/desi/users/tingtan/DLA_finder/retraining/models/low2/current_499999
export DESIDLAS_CKPT_MID=/global/cfs/cdirs/desi/users/tingtan/DLA_finder/retraining/models/mid/current_460000
RELEASE=y3_saclay
SURVEY=
PROGRAM=
VERSION=

# ---- Prediction settings ----
BATCH_SIZE=512
MAX_WINDOWS=16384

# ---- Work range ----
BASE_START=0
GPU_PER_NODE=4

# Option A (manual): set TOTAL explicitly
TOTAL=1127

# Option B (auto): infer TOTAL from latest filelist cache
# Uncomment this block if you do not know TOTAL.
# TOTAL=$(python3 - <<'PY'
# import glob, os, numpy as np
# root = os.environ.get("LIST_CACHE_ROOT", "")
# files = sorted(glob.glob(os.path.join(root, "filelist_*.npz")), key=os.path.getmtime)
# if not files:
#     raise SystemExit("No list cache found in LIST_CACHE_ROOT. Run once with --rebuild-list.")
# d = np.load(files[-1], allow_pickle=True)
# print(len(d["spectra"]))
# PY
# )

CHUNK=$(( (TOTAL + GPU_PER_NODE - 1) / GPU_PER_NODE ))
export TOTAL CHUNK BASE_START GPU_PER_NODE

echo "TOTAL=$TOTAL CHUNK=$CHUNK BASE_START=$BASE_START"

srun --ntasks=${GPU_PER_NODE} --gpus-per-task=1 --cpus-per-task=4 \
--gpu-bind=single:1 --cpu-bind=cores \
--output="$SCR_OUT/log/dlaf_unified_%j_task%t.out" \
--error="$SCR_OUT/log/dlaf_unified_%j_task%t.err" bash -lc '
echo "[task $SLURM_LOCALID] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

upper=$(( BASE_START + TOTAL ))

start=$(( BASE_START + SLURM_LOCALID * CHUNK ))
end=$(( start + CHUNK ))

if [ $start -ge $upper ]; then
  echo "[GPU $SLURM_LOCALID] start=$start exceeds upper=$upper, skipping."
  exit 0
fi
if [ $end -gt $upper ]; then
  end=$upper
fi
len=$(( end - start ))

echo "[GPU $SLURM_LOCALID] running range: [$start, $end) total $len"

python3 '"$RUNNER"' \
  --data-type '"$DATA_TYPE"' \
  --spectra-root '"$SPECTRA_ROOT"' \
  --sightline-root '"$SIGHTLINE_ROOT"' \
  --list-cache-root '"$LIST_CACHE_ROOT"' \
  --release '"$RELEASE"' \
  --value $start --length $len \
  --batch-size '"$BATCH_SIZE"' --max-windows '"$MAX_WINDOWS"' \
  --scratch-out '"$DLACAT_ROOT"'
'
```

### Submit Jobs (Batch)

Use `Run_DLAfinder/submit_desi_DLAfinder_run.sh` to run 4 tasks per node (1 GPU each).
It splits the file list across tasks and writes per-task logs.

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

## Stacking Catalogs After Running

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
- `--skip-existing-pred` skip already-predicted files
- `--fill-missing-dlacat` repair missing catalogs from existing predictions

## Use Retrained Models (Optional)

By default, prediction uses the legacy checkpoints. To switch to retrained models,
set environment variables before running:

```bash
export DESIDLAS_CKPT_LOW1=/global/cfs/cdirs/desi/users/tingtan/DLA_finder/retraining/models/low1/current_199999
export DESIDLAS_CKPT_LOW2=//global/cfs/cdirs/desi/users/tingtan/DLA_finder/retraining/models/low2/current_499999
export DESIDLAS_CKPT_MID=/global/cfs/cdirs/desi/users/tingtan/DLA_finder/retraining/models/mid/current_460000
```

Unset them to return to the default models. If you are not on NERSC, you will
need to set these explicitly.

## Environment Variables (Common)
- `DESIDLAS_CKPT_LOW1`, `DESIDLAS_CKPT_LOW2`, `DESIDLAS_CKPT_MID` override model checkpoints
- `DESIDLAS_PEAK_THRESH` peak threshold (default 0.2)
- `DESIDLAS_LEVEL` confidence level (default 0.5)
- `DESIDLAS_FORCE_CPU=1` force CPU prediction
- `DESIDLAS_CPU_WORKERS=<N>` CPU worker count for CPU prediction

## Submit Script (GPU, 4 tasks)

Use `Run_DLAfinder/submit_desi_DLAfinder_run.sh` to run 4 tasks per node (1 GPU each).
It splits the file list across tasks and writes per-task logs.

## Tips

- For A100 40GB, start with `--batch-size 512` and `--max-windows 16384`.
- If you see out-of-memory, reduce `--max-windows` first.
- If you have an existing `filelist_*.npz` in `--list-cache-root`, you can omit
  `--spectra-root` unless you need to rebuild the list.
