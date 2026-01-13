# Training and Model Details (DESI DLA Finder)

This document summarizes the current training workflow and model details in this
repository. Source references are included so the logic can be audited.

## Data and Outputs (current retraining setup)

Inputs (DESI Y3 mock example):
- spectra root: `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/spectra-16`

Outputs:
- sightlines: `/pscratch/sd/t/tanting/retraining/sightlines`
- shards: `/pscratch/sd/t/tanting/retraining/shards`
- models: `/pscratch/sd/t/tanting/retraining/models`

Scripts:
- `desidlas/training/make_sightlines_mock.py`
- `desidlas/training/make_training_shards.py`
- `desidlas/training/training.py`

Reference: `Run_DLAfinder/README-training.md`

## Training Workflow (Step by Step)

1) Generate sightlines (truth-aware)
   - Uses `truth-16-<id>.fits` if present.
   - Writes `sightlines-<id>.npy` under the output root.

   Example:
   ```bash
   python3 desidlas/training/make_sightlines_mock.py \
     --spectra-root <spectra_root> \
     --out-root /pscratch/sd/t/<user>/retraining/sightlines \
     --workers 32
   ```

2) Build training shards (mid + low)
   - Splits by S/N: `s2n < 3` → low, otherwise mid.
   - Low uses smoothed fluxes (4-channel), mid uses raw flux.
   - Output shards saved under `/shards/mid` and `/shards/low`.

   Example:
   ```bash
   python3 desidlas/training/make_training_shards.py \
     --sightline-root /pscratch/sd/t/<user>/retraining/sightlines \
     --out-root /pscratch/sd/t/<user>/retraining/shards \
     --chunk-size 200 --workers 32
   ```

3) Train mid or low
   - Mid: `INPUT_SIZE=400, matrix_size=1`
   - Low: `INPUT_SIZE=600, matrix_size=4`
   - Supports `--split` to create a validation split from the training glob.
   - Supports overrides for `--learning-rate` and `--pos-weight`.

   Example (mid):
   ```bash
   python3 desidlas/training/training.py \
     -r "/pscratch/sd/t/<user>/retraining/shards/mid/*_*.npy" \
     -e "/pscratch/sd/t/<user>/retraining/shards/mid/*_*.npy" \
     -c /pscratch/sd/t/<user>/retraining/models/mid/current \
     -t 400 -m 1 \
     --split 0.1
   ```

   Example (low):
   ```bash
   python3 desidlas/training/training.py \
     -r "/pscratch/sd/t/<user>/retraining/shards/low/*_*.npy" \
     -e "/pscratch/sd/t/<user>/retraining/shards/low/*_*.npy" \
     -c /pscratch/sd/t/<user>/retraining/models/low/current \
     -t 600 -m 4 \
     --split 0.1 \
     --learning-rate 2e-5 \
     --pos-weight 3.0
   ```

4) Resume training from checkpoint
   - Use `-l /path/to/current_<step>` (no `.ckpt` suffix).
   - Training resumes from checkpoint step or parses step from filename.

   Example:
   ```bash
   python3 desidlas/training/training.py \
     -r "/pscratch/sd/t/<user>/retraining/shards/low/*_*.npy" \
     -e "/pscratch/sd/t/<user>/retraining/shards/low/*_*.npy" \
     -c /pscratch/sd/t/<user>/retraining/models/low/current \
     -l /pscratch/sd/t/<user>/retraining/models/low/current_40000 \
     -t 600 -m 4
   ```

## Model Architecture (CNN)

Source: `desidlas/training/model.py`

Input:
- Mid: shape `[batch, 400]` → reshaped to `[batch, 400, 1, 1]`
- Low: shape `[batch, 4, 600]` → reshaped to `[batch, 600, 1, 4]`

Network:
- 3 convolution layers (1D implemented as 2D conv with width=1)
- 3 pooling layers (max pooling)
- FC1 shared layer
- FC2 split into 3 heads (classifier / offset / coldensity)
- Readout heads: 1 neuron each

Outputs:
- `prediction` (binary classifier)
- `y_nn_offset` (offset regression)
- `y_nn_coldensity` (column density regression)

Losses:
- Classifier: weighted sigmoid cross-entropy (`pos_weight` supported)
- Offset regression: masked MSE (computed on positive samples only)
- Coldensity regression: masked weighted MSE (positive samples only)
- L2 regularization on conv + FC layers

Optimizer:
- Adam (`tf.compat.v1.train.AdamOptimizer`)

## Hyperparameters

Source: `desidlas/training/parameterset.py`

Default values are taken from `parameters[k][0]`. Overrides supported in
`training.py`:
- `--learning-rate` (float)
- `--pos-weight` (float)

Low-SNR defaults (added in training runner):
- learning rate capped at `<= 2e-5` if `matrix_size==4`
- `pos_weight=3.0` if not specified

## Dataset Loader Behavior

Source: `desidlas/data_model/Dataset.py`

Key behaviors:
- Accepts a glob or explicit list of shard files.
- Skips unreadable or empty shard files.
- Supports 2D or 3D flux tensors.
- Optional `shard_sample` to load only a random subset of shards per buffer
  (speeds up I/O while still sampling the full dataset over time).

In `training.py`, you can pass:
```
--shard-sample 20
```
to sample 20 shard files per buffer load.

## Prediction Model Selection (Runtime)

By default, prediction uses the legacy checkpoints. You can override at runtime:

```bash
export DESIDLAS_CKPT_LOW=/pscratch/sd/t/<user>/retraining/models/low/current_135000
export DESIDLAS_CKPT_MID=/pscratch/sd/t/<user>/retraining/models/mid/current_99999
```

Sources:
- CPU prediction: `desidlas/prediction/multiprocess_partprediction.py`
- GPU prediction: `desidlas/prediction/multiprocess_partprediction_gpu.py`

## Notes and Guardrails

- Truth files are required for labeled training data. Without truth, shards may
  be empty.
- Low SNR uses smoothed flux (4 channels), mid uses raw flux.
- For large shard counts, use `--shard-sample` or larger `--chunk-size` to reduce I/O.
- If prediction is interrupted, use `--skip-existing-pred --fill-missing-dlacat`
  to avoid recomputing and to repair missing catalogs.
