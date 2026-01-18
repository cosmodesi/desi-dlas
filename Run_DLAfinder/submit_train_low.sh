#!/bin/bash -l
# Train low1-SNR model on Perlmutter (single GPU).

#SBATCH -C gpu
#SBATCH -A desi
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH -J train_low1
#SBATCH -o /pscratch/sd/t/tanting/retraining/logs/train_low1_%j.out
#SBATCH -e /pscratch/sd/t/tanting/retraining/logs/train_low1_%j.err

module load python
conda activate /global/cfs/cdirs/desi/users/tingtan/conda_envs/CNN_GPU

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=2
ulimit -n 65535

mkdir -p /pscratch/sd/t/tanting/retraining/logs
mkdir -p /pscratch/sd/t/tanting/retraining/models/low1

python3 /global/u1/t/tanting/DESI_analysis/desi-dlas/desidlas/training/training.py \
  -r "/pscratch/sd/t/tanting/retraining/shards/low1/*_*.npy" \
  -e "/pscratch/sd/t/tanting/retraining/shards/low1/*_*.npy" \
  -c /pscratch/sd/t/tanting/retraining/models/low1/current \
  -t 400 -m 1
