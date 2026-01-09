import glob
import os
import numpy as np

from desidlas.datasets import preprocess
from desidlas.datasets.get_dataset import make_datasets, smooth_flux
from desidlas.datasets.datasetting import split_sightline_into_samples, select_samples_50p_pos_neg
from desidlas.dla_cnn import defs


SIGHTLINE_GLOB = "/pscratch/sd/t/tanting/retraining/sightlines/**/*.npy"
OUT_ROOT = "/pscratch/sd/t/tanting/retraining/shards"
CHUNK_SIZE = 50


def make_smoothdatasets_chunked(sightlines, output, chunk_size=CHUNK_SIZE):
    dataset = {}
    count = 0
    file_idx = 0
    for sightline in sightlines:
        if sightline == []:
            continue
        preprocess.label_sightline(sightline, kernel=defs.smooth_kernel, REST_RANGE=defs.REST_RANGE)
        data_split = split_sightline_into_samples(
            sightline, REST_RANGE=defs.REST_RANGE, kernel=defs.smooth_kernel, v=defs.best_v['all']
        )
        sample_masks = select_samples_50p_pos_neg(sightline, kernel=defs.smooth_kernel)
        if len(sample_masks) > 0:
            flux = np.vstack([data_split[0][m] for m in sample_masks])
            labels_classifier = np.hstack([data_split[1][m] for m in sample_masks])
            labels_offset = np.hstack([data_split[2][m] for m in sample_masks])
            col_density = np.hstack([data_split[3][m] for m in sample_masks])
            flux_matrix = smooth_flux(flux)
            dataset[sightline.id] = {
                'FLUX': flux_matrix,
                'labels_classifier': labels_classifier,
                'labels_offset': labels_offset,
                'col_density': col_density
            }

        count += 1
        if count >= chunk_size:
            outpath = "{}_{}.npy".format(output, file_idx)
            np.save(outpath, dataset)
            dataset = {}
            count = 0
            file_idx += 1

    if dataset:
        outpath = "{}_{}.npy".format(output, file_idx)
        np.save(outpath, dataset)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "mid"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "low"), exist_ok=True)

    files = sorted(glob.glob(SIGHTLINE_GLOB, recursive=True))
    for f in files:
        sightlines = np.load(f, allow_pickle=True)

        mid = []
        low = []
        for s in sightlines:
            if s == []:
                continue
            if not hasattr(s, "s2n") or s.s2n is None:
                s.s2n = preprocess.estimate_s2n(s)
            if s.s2n < 3:
                low.append(s)
            else:
                mid.append(s)

        base = os.path.basename(f).replace(".npy", "")
        if mid:
            make_datasets(
                mid,
                output=os.path.join(OUT_ROOT, "mid", base),
                validate=False,
                chunk_size=CHUNK_SIZE
            )
        if low:
            make_smoothdatasets_chunked(
                low,
                output=os.path.join(OUT_ROOT, "low", base),
                chunk_size=CHUNK_SIZE
            )


if __name__ == "__main__":
    main()
