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

2) Build training shards (mid + two low buckets)
   - Splits by S/N:
     - `s2n < 1.5` → low1
     - `1.5 <= s2n < 3` → low2
     - `s2n >= 3` → mid
   - Drop very low S/N by default (`--low-min-s2n 1.0`).
   - Low buckets use smoothed fluxes (4-channel), mid uses raw flux.
   - Output shards saved under `/shards/mid`, `/shards/low1`, `/shards/low2`.

   Example:
   ```bash
   python3 desidlas/training/make_training_shards.py \
     --sightline-root /pscratch/sd/t/<user>/retraining/sightlines \
     --out-root /pscratch/sd/t/<user>/retraining/shards \
     --chunk-size 200 --workers 32 \
     --low-min-s2n 1.0 --low-mid-s2n 1.5 \
     --low-pos-sample-percent 0.2 --low-pos-frac 0.25 --mid-pos-frac 0.5
   ```

3) Train mid or low
   - Mid: `INPUT_SIZE=400, matrix_size=1`
   - Low: `INPUT_SIZE=600, matrix_size=4`
   - Supports `--split` to create a validation split from the training glob.
   - Supports overrides for `--learning-rate`, `--pos-weight`, and `--training-iters`.

   Example (mid):
   ```bash
   python3 desidlas/training/training.py \
     -r "/pscratch/sd/t/<user>/retraining/shards/mid/*_*.npy" \
     -e "/pscratch/sd/t/<user>/retraining/shards/mid/*_*.npy" \
     -c /pscratch/sd/t/<user>/retraining/models/mid/current \
     -t 400 -m 1 \
     --split 0.1
   ```

   Example (low1):
   ```bash
   python3 desidlas/training/training.py \
     -r "/pscratch/sd/t/<user>/retraining/shards/low1/*_*.npy" \
     -e "/pscratch/sd/t/<user>/retraining/shards/low1/*_*.npy" \
     -c /pscratch/sd/t/<user>/retraining/models/low1/current \
     -t 600 -m 4 \
     --split 0.1 \
     --learning-rate 5e-5 \
     --pos-weight 1.0 \
     --training-iters 800000
   ```

   Example (low2):
   ```bash
   python3 desidlas/training/training.py \
     -r "/pscratch/sd/t/<user>/retraining/shards/low2/*_*.npy" \
     -e "/pscratch/sd/t/<user>/retraining/shards/low2/*_*.npy" \
     -c /pscratch/sd/t/<user>/retraining/models/low2/current \
     -t 600 -m 4 \
     --split 0.1 \
     --learning-rate 5e-5 \
     --pos-weight 1.0 \
     --training-iters 800000
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

Low input handling detail:
- Low shards are stored channel-first (`[batch, 4, 600]`).
- Training transposes to `[batch, 600, 4]` before reshaping to NHWC.

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

Architecture details (defaults from `parameterset.py`):
- Conv1: kernel=40, filters=100, stride=5
- Pool1: kernel=7, stride=1 (max pool)
- Conv2: kernel=32, filters=256, stride=2
- Pool2: kernel=4, stride=5 (max pool)
- Conv3: kernel=20, filters=128, stride=1
- Pool3: kernel=6, stride=6 (max pool)
- FC1: 500 neurons (shared)
- FC2 heads: 700 / 500 / 150 neurons (classifier / offset / coldensity)
- Dropout keep_prob: 0.9
- L2 regularization: 0.005

Losses:
- Classifier: weighted sigmoid cross-entropy (`pos_weight` supported)
- Offset regression: masked MSE (computed on positive samples only)
- Coldensity regression: masked weighted MSE (positive samples only)
- L2 regularization on conv + FC layers

Optimizer:
- Adam (`tf.compat.v1.train.AdamOptimizer`)

Training loop details:
- `training_iters`: 100000 by default.
- Every 200 iterations: evaluate on a random 10k sample from the training buffer.
- Every 5000 iterations (and at end): evaluate on the full test dataset.
- Checkpoint saved every 5000 iterations and at end.
- Resume uses `global_step` from checkpoint; if missing, parses step from filename.

Notes:
- Training runs under TF1-style graph/session (`tf.compat.v1`).
- Default device string is `'/gpu:1'`, but `allow_soft_placement=True` will fall back
  to available GPU or CPU if needed.

## Hyperparameters

Source: `desidlas/training/parameterset.py`

Default values are taken from `parameters[k][0]`. Overrides supported in
`training.py`:
- `--learning-rate` (float)
- `--pos-weight` (float)

Low-SNR defaults (added in training runner):
- learning rate capped at `<= 5e-5` if `matrix_size==4`
- `pos_weight=1.0` if not specified

Key default values (from `parameters[k][0]`):
- learning_rate: `5e-4`
- training_iters: `100000`
- batch_size: `400`
- dropout_keep_prob: `0.9`
- l2_regularization_penalty: `0.005`
- fc1_n_neurons: `500`
- fc2_1_n_neurons: `700`
- fc2_2_n_neurons: `500`
- fc2_3_n_neurons: `150`
- conv1_kernel/filters/stride: `40 / 100 / 5`
- conv2_kernel/filters/stride: `32 / 256 / 2`
- conv3_kernel/filters/stride: `20 / 128 / 1`
- pool1_kernel/stride: `7 / 1`
- pool2_kernel/stride: `4 / 5`
- pool3_kernel/stride: `6 / 6`

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

## Labeling, Windowing, and Sampling

Source files:
- `desidlas/datasets/preprocess.py`
- `desidlas/datasets/datasetting.py`
- `desidlas/datasets/get_dataset.py`
- `desidlas/dla_cnn/defs.py`

Constants:
- `REST_RANGE = [900, 1346, 1748]`
- `kernel = 400` (mid/high)
- `smooth_kernel = 600` (low)
- `best_v['all'] = 44735` m/s (rebin velocity)

Labeling (`label_sightline`):
- Builds `classification`, `offsets`, and `column_density` arrays over the
  DLA search region.
- Positive regions are centered on each DLA with a half-width of
  `kernel * pos_sample_kernel_percent / 2` (default 0.3).
- Regions around DLA and LyB are masked as `-1` in `classification`.
- `offsets` encode pixel distance from DLA center with opposite sign on each side.
- `column_density` is filled with each DLA’s NHI where applicable.

Windowing (`split_sightline_into_samples`):
- Builds a sliding window of length `kernel` centered on every pixel in the
  DLA search region.
- Uses padding to avoid dropping edge windows.
- Returns `fluxes_matrix`, `lam_matrix`, and per-window labels.

Sampling (`select_samples_50p_pos_neg`):
- For each sightline, randomly selects positive/negative windows based on a
  target positive fraction (default 0.5 for mid, 0.25 for low).
- This is a per-sightline balance step, not a global class rebalance.

Shard format (training):
- Each shard is a dict keyed by `sightline.id`.
- Each entry contains:
  - `FLUX`: array of windows
    - mid: `[n_samples, 400]`
    - low: `[n_samples, 4, 600]` (raw + 3 medians)
  - `labels_classifier`, `labels_offset`, `col_density`

Low-SNR smoothing (`smooth_flux`):
- For each window, compute median filters with widths 3, 7, 15.
- Stack as `[raw, smooth3, smooth7, smooth15]`.

## Improvement Levers for Algorithm Review

Common levers an algorithm reviewer may want to test:
- **Sampling balance**: per-sightline 50/50 can overweight rare DLAs in noisy
  spectra; try global balance or hard-negative mining.
- **S/N thresholds**: `s2n < 3` for low, otherwise mid. Adjust to 2/4/5 or
  add a high-SNR bucket for improved specialization.
- **Label mask width**: `pos_sample_kernel_percent` (default 0.3) controls
  DLA positive span; tune for stability/recall.
- **Loss weighting**: `pos_weight` or separate weights for offset/coldensity.
- **Learning rate schedule**: fixed LR only; try cosine/step decay or warmup.
- **Regularization**: L2 and dropout are static; explore larger dropout for low.
- **Kernel/window size**: kernel=400/600; evaluate longer windows for low S/N.
- **Architecture**: 3-layer conv trunk with 3 heads; consider deeper trunk
  or shared attention over windows.
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
