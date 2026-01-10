# Retraining (DESI Y3 mock, mid + low)

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

## 2) Build training shards (mid + low)

```bash
python3 desidlas/training/make_training_shards.py
```

## 3) Train mid / low

Mid:

```bash
python3 desidlas/training/training.py \
  -r "/pscratch/sd/t/tanting/retraining/shards/mid/*_*.npy" \
  -e "/pscratch/sd/t/tanting/retraining/shards/mid/*_*.npy" \
  -c /pscratch/sd/t/tanting/retraining/models/mid/current \
  -t 400 -m 1
```

Low:

```bash
python3 desidlas/training/training.py \
  -r "/pscratch/sd/t/tanting/retraining/shards/low/*_*.npy" \
  -e "/pscratch/sd/t/tanting/retraining/shards/low/*_*.npy" \
  -c /pscratch/sd/t/tanting/retraining/models/low/current \
  -t 600 -m 4
```

Notes:
- The scripts are configured for mid + low only.
- Update paths inside the scripts if you change the output locations.
