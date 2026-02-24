# desi-dlas
DLA finder(s) for DESI data. This version adds:
- GPU-accelerated prediction
- retrained models on DESI Y3 mocks
- an NHI afterburner stage (see overview and Tan et al., in prep.)

This repository provides:
- A unified runner for mock and observational data
- CPU/GPU prediction
- Optional sightline generation and DLA catalog stacking
- Training utilities and documentation

## Quick Start (Prediction)
Minimal example for mock data (run from repo root):

```bash
module load python
conda activate /path/to/conda_env

pip install -e .

python3 Run_DLAfinder/desi_DLAfinder_run.py \
  --data-type mock \
  --release <release_name> \
  --spectra-root <mock_spectra_root> \
  --sightline-root <sightline_output_root> \
  --list-cache-root <list_cache_root> \
  --generate-sightlines
```

## Install / Environment
Recommended:
- Python 3.10
- TensorFlow 2.15 (GPU optional)
- Numpy, Scipy, Astropy, Tqdm, Matplotlib

Example GPU env on Perlmutter:

```bash
module load python
conda create -y -p /path/to/conda_envs/CNN_GPU python=3.10
conda activate /path/to/conda_envs/CNN_GPU
pip install 'tensorflow[and-cuda]==2.15.*'
pip install -e .
```

If you cannot `pip install -e .`, set:

```bash
export PYTHONPATH=/path/to/desi-dlas:$PYTHONPATH
```

## Model Checkpoints
Prediction requires model checkpoints. Default paths are on NERSC and may not be
accessible elsewhere. Override with:

```bash
export DESIDLAS_CKPT_LOW1=/path/to/models/low1/current_XXXXXX
export DESIDLAS_CKPT_LOW2=/path/to/models/low2/current_XXXXXX
export DESIDLAS_CKPT_MID=/path/to/models/mid/current_XXXXXX
```

## Documentation
- Unified runner: `Run_DLAfinder/README.md`
- Training/model overview: `TRAINING_OVERVIEW.md`

## Authors
Contributors:
- Jiaqi Zou
- Ben Wang
- Z. Cai
- J. Xavier Prochaska

Currently maintained by:
- Ting Tan
