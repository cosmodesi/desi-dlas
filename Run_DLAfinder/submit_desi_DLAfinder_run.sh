#!/bin/bash -l
# Example submit script for unified runner (adjust paths/params).

#SBATCH -C gpu
#SBATCH -A desi
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH -J dlaf_unified_4x
#SBATCH -o /global/cfs/cdirs/desi/users/tingtan/DLA_finder/mocks/y3_saclay/log/dlaf_unified_%j_task%t.out
#SBATCH -e /global/cfs/cdirs/desi/users/tingtan/DLA_finder/mocks/y3_saclay/log/dlaf_unified_%j_task%t.err

module load python
conda activate /global/cfs/cdirs/desi/users/tingtan/conda_envs/CNN_GPU

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=2
ulimit -n 65535

export RUNNER=/global/u1/t/tanting/DESI_analysis/desi-dlas/Run_DLAfinder/desi_DLAfinder_run.py
export SCR_OUT=/global/cfs/cdirs/desi/users/tingtan/DLA_finder/mocks/y3_saclay

# Optional: use retrained checkpoints (unset to use defaults)
# export DESIDLAS_CKPT_LOW1=/global/cfs/cdirs/desi/users/tingtan/DLA_finder/retraining/models/low1/current_199999
# export DESIDLAS_CKPT_LOW2=//global/cfs/cdirs/desi/users/tingtan/DLA_finder/retraining/models/low2/current_499999
# export DESIDLAS_CKPT_MID=/global/cfs/cdirs/desi/users/tingtan/DLA_finder/retraining/models/mid/current_460000

# ---- Configure dataset ----
DATA_TYPE=mock          # mock | data
SPECTRA_ROOT=/global/cfs/cdirs/desicollab/mocks/lya_forest/develop/saclay/qq_desi_y3/v4.7.5/mock-0/juraLy8-124/spectra-16
SIGHTLINE_ROOT=$SCR_OUT/sightlines
LIST_CACHE_ROOT=$SCR_OUT/data
DLACAT_ROOT=$SCR_OUT/dlacat

RELEASE=y3_saclay       # optional
SURVEY=                 # optional (data)
PROGRAM=                # optional (data)
VERSION=                # optional (data)

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
    --release '"$RELEASE"' --survey '"$SURVEY"' --program '"$PROGRAM"' --version '"$VERSION"' \
    --value $start --length $len \
    --batch-size '"$BATCH_SIZE"' --max-windows '"$MAX_WINDOWS"' \
    --scratch-out "'"$DLACAT_ROOT"'"
'
