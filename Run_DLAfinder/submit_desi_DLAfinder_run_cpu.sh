#!/bin/bash -l
# CPU submit script for unified runner (multi-node).

#SBATCH -C cpu
#SBATCH -A desi
#SBATCH -q regular
#SBATCH --nodes=12
#SBATCH --time=02:30:00
#SBATCH --job-name=DLAfinder_mock_Y3_cpu
#SBATCH --output=/pscratch/sd/t/tanting/DLAfinder_out/y3_saclay/log/dlaf_cpu_%j.out
#SBATCH --error=/pscratch/sd/t/tanting/DLAfinder_out/y3_saclay/log/dlaf_cpu_%j.err

module load python
conda activate /global/cfs/cdirs/desi/users/tingtan/conda_envs/CNN_GPU

export OMP_NUM_THREADS=256
export MKL_NUM_THREADS=256
export TF_CPP_MIN_LOG_LEVEL=2
ulimit -n 65535

export RUNNER=/global/u1/t/tanting/DESI_analysis/desi-dlas/Run_DLAfinder/desi_DLAfinder_run.py
export SCR_OUT=/pscratch/sd/t/tanting/DLAfinder_out/y3_saclay

DATA_TYPE=mock
SPECTRA_ROOT=/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/spectra-16
SIGHTLINE_ROOT=$SCR_OUT/sightlines
LIST_CACHE_ROOT=$SCR_OUT/data
RELEASE=y3_saclay

# Optional: retrained checkpoints
# export DESIDLAS_CKPT_LOW1=/pscratch/sd/<user>/retraining/models/low1/current_XXXXXX
# export DESIDLAS_CKPT_LOW2=/pscratch/sd/<user>/retraining/models/low2/current_XXXXXX
# export DESIDLAS_CKPT_MID=/pscratch/sd/<user>/retraining/models/mid/current_XXXXXX

TOTAL=$(python -c "import glob,os,numpy as np; root=os.environ['LIST_CACHE_ROOT']; files=sorted(glob.glob(os.path.join(root,'filelist_*.npz')), key=os.path.getmtime); d=np.load(files[-1], allow_pickle=True); print(len(d['spectra']))")
NODES=12
CHUNK=$(( (TOTAL + NODES - 1) / NODES ))
BASE_START=0

for node in $(seq 0 $((NODES - 1))); do
  srun -N 1 -n 1 -c 256 bash -lc "
  start=\$(( $BASE_START + $node * $CHUNK ))
  end=\$(( start + $CHUNK ))
  if [ \$start -ge $TOTAL ]; then
      echo \"[CPU $node] start=\$start >= TOTAL=$TOTAL, skip\"
      exit 0
  fi
  if [ \$end -gt $TOTAL ]; then end=$TOTAL; fi
  len=\$(( end - start ))

  echo \"[CPU $node] range: [\$start, \$end) count=\$len\"

  python3 $RUNNER \
    --data-type $DATA_TYPE \
    --spectra-root $SPECTRA_ROOT \
    --sightline-root $SIGHTLINE_ROOT \
    --list-cache-root $LIST_CACHE_ROOT \
    --release $RELEASE \
    --value \$start --length \$len \
    --scratch-out $SCR_OUT \
    --cpu-only
  " &
done
wait
echo "All tasks completed!"
