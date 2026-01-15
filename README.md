# desi-dlas
## DLA finder(s) for DESI data

# Brought to you by:
## Jiaqi Zou
## Ben Wang

## Z. Cai
## J. Xavier Prochaska

# Current charged by:
## Ting Tan

## CPU Sightline Generation

Use the dedicated CPU-only helper to generate sightlines without touching GPUs:

```
module load python
conda activate /global/cfs/cdirs/desi/users/tingtan/conda_envs/CNN_GPU

python3 Run_DLAfinder/desi_DLAfinder_make_sightlines_cpu.py \
  --data-type mock \
  --spectra-root /path/to/spectra/root \
  --sightline-root /path/to/sightlines/root \
  --list-cache-root /path/to/list-cache/root \
  --release y3_saclay \
  --workers 64
```

Notes:
- Uses the same file-list cache naming as the unified runner, so you can reuse `--list-cache-root`.
- Set `--force-sightlines` to overwrite existing sightlines.
