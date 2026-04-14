# Retraining (DESI Y3 mock, mid + low1/low2)

This guide uses the DESI Y3 mock at:
- `/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/spectra-16`

Outputs:
- sightlines: `/pscratch/sd/t/tanting/retraining/sightlines`
- shards: `/pscratch/sd/t/tanting/retraining/shards`
- models: `/pscratch/sd/t/tanting/retraining/models`

Scripts:
- `desidlas/training/make_sightlines_mock.py`
- `desidlas/training/make_training_shards.py`

Note:
- Training requires truth labels. The sightline script will use `truth-16-<id>.fits`
  if present; otherwise the sightline will have no DLA labels and shards may be empty.

## 1) Generate sightlines

```bash
python3 desidlas/training/make_sightlines_mock.py
```

## 2) Build training shards (mid + low1/low2)

Raw low-SNR shards, compatible with `-t 400 -m 1`:

```bash
python3 desidlas/training/make_training_shards.py
```

Smoothed low-SNR shards, compatible with `-t 600 -m 4` for low1/low2:

```bash
python3 desidlas/training/make_training_shards.py \
  --low-min-s2n 0 \
  --low-mid-s2n 1.5 \
  --low-smooth \
  --low-pos-sample-percent 0.2 \
  --low-pos-frac 0.5 \
  --mid-pos-frac 0.5
```

## 3) Train mid / low1 / low2

Mid:

```bash
python3 desidlas/training/training.py \
  -r "/pscratch/sd/t/tanting/retraining/shards/mid/*_*.npy" \
  -e "/pscratch/sd/t/tanting/retraining/shards/mid/*_*.npy" \
  -c /pscratch/sd/t/tanting/retraining/models/mid/current \
  -t 400 -m 1
```

Low1:

```bash
python3 desidlas/training/training.py \
  -r "/pscratch/sd/t/tanting/retraining/shards/low1/*_*.npy" \
  -e "/pscratch/sd/t/tanting/retraining/shards/low1/*_*.npy" \
  -c /pscratch/sd/t/tanting/retraining/models/low1/current \
  -t 600 -m 4
```

Low2:

```bash
python3 desidlas/training/training.py \
  -r "/pscratch/sd/t/tanting/retraining/shards/low2/*_*.npy" \
  -e "/pscratch/sd/t/tanting/retraining/shards/low2/*_*.npy" \
  -c /pscratch/sd/t/tanting/retraining/models/low2/current \
  -t 600 -m 4
```

Notes:
- The scripts are configured for mid + low1/low2 only.
- Update paths inside the scripts if you change the output locations.
- If low1/low2 are trained with smoothed shards, set `DESIDLAS_LOW_SMOOTH=1`
  during prediction so the runtime input shape matches the checkpoint.
- To test only one low bucket with smoothed inputs, use `DESIDLAS_LOW1_SMOOTH=1`
  or `DESIDLAS_LOW2_SMOOTH=1`.
