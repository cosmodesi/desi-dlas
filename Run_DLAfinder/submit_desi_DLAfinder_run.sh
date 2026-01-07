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
#SBATCH -o /global/u1/t/tanting/DESI_analysis/DESI_CNN_DLA/log/dlaf_unified_%A_%a.out
#SBATCH -e /global/u1/t/tanting/DESI_analysis/DESI_CNN_DLA/log/dlaf_unified_%A_%a.err

module load python
conda activate /global/cfs/cdirs/desi/users/tingtan/conda_envs/CNN_GPU

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=2
ulimit -n 65535

export RUNNER=/global/u1/t/tanting/DESI_analysis/DESI_CNN_DLA/GPU_HS/desi_DLAfinder_run.py
export SCR_OUT=$SCRATCH/DLAfinder_out/y3_saclay

# ---- Configure dataset ----
DATA_TYPE=mock          # mock | data
SPECTRA_ROOT=/path/to/spectra/root
SIGHTLINE_ROOT=/path/to/sightlines/root
LIST_CACHE_ROOT=/global/u1/t/tanting/DESI_analysis/DESI_CNN_DLA/data

RELEASE=y3_saclay       # optional
SURVEY=main             # optional (data)
PROGRAM=dark            # optional (data)
VERSION=v0              # optional (data)

# ---- Prediction settings ----
BATCH_SIZE=256
MAX_WINDOWS=8192

# ---- Work range ----
TOTAL=200
GPU_PER_NODE=4
CHUNK=$(( (TOTAL + GPU_PER_NODE - 1) / GPU_PER_NODE ))
BASE_START=$(( SLURM_ARRAY_TASK_ID * GPU_PER_NODE * CHUNK ))

srun --ntasks=${GPU_PER_NODE} --gpus-per-task=1 --cpus-per-task=4 \
     --gpu-bind=single:1 --cpu-bind=cores bash -lc '
  echo "[task $SLURM_LOCALID] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

  start=$(( '"$BASE_START"' + $SLURM_LOCALID * '"$CHUNK"' ))
  end=$(( start + '"$CHUNK"' ))
  if [ $start -ge '"$TOTAL"' ]; then
      echo "[GPU $SLURM_LOCALID] start=$start 超出TOTAL='"$TOTAL"', 跳过。"
      exit 0
  fi
  if [ $end -gt '"$TOTAL"' ]; then
      end='"$TOTAL"'
  fi
  len=$(( end - start ))

  echo "[GPU $SLURM_LOCALID] 跑索引区间: [$start, $end) 共 $len"

  python3 '"$RUNNER"' \
    --data-type '"$DATA_TYPE"' \
    --spectra-root '"$SPECTRA_ROOT"' \
    --sightline-root '"$SIGHTLINE_ROOT"' \
    --list-cache-root '"$LIST_CACHE_ROOT"' \
    --release '"$RELEASE"' --survey '"$SURVEY"' --program '"$PROGRAM"' --version '"$VERSION"' \
    --value $start --length $len \
    --batch-size '"$BATCH_SIZE"' --max-windows '"$MAX_WINDOWS"' \
    --scratch-out "'"$SCR_OUT"'" \
    --generate-sightlines
'
