import glob
import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from desidlas.datasets import preprocess
from desidlas.datasets.input_set import get_lam_data, split_sightline_into_samples
from desidlas.datasets.get_flux import smooth_flux
from desidlas.parameters import REST_RANGE, kernel, best_v, pos_sample_kernel_percent


SIGHTLINE_GLOB = "/pscratch/sd/t/tanting/retraining/sightlines/**/*.npy"
OUT_ROOT = "/pscratch/sd/t/tanting/retraining/shards"
CHUNK_SIZE = 50


def label_sightline(sightline, kernel_size, rest_range):
    lam, _, ix_dla_range = get_lam_data(sightline.loglam, sightline.z_qso, rest_range)
    samplerange_px = int(kernel_size * pos_sample_kernel_percent / 2)
    ix_dlas = []
    coldensity_dlas = []
    for dla in sightline.dlas:
        rest_wave = dla.central_wavelength / (1 + sightline.z_qso)
        if rest_range[0] < rest_wave < rest_range[1]:
            ix_dlas.append(np.abs(lam[ix_dla_range] - dla.central_wavelength).argmin())
            coldensity_dlas.append(dla.col_density)

    classification = np.zeros((np.sum(ix_dla_range)), dtype=np.float32)
    for ix_dla in ix_dlas:
        classification[ix_dla - samplerange_px * 2:ix_dla + samplerange_px * 2 + 1] = -1
        lyb_ix = sightline.get_lyb_index(ix_dla)
        classification[lyb_ix - samplerange_px:lyb_ix + samplerange_px + 1] = -1
    for ix_dla in ix_dlas:
        classification[ix_dla - samplerange_px:ix_dla + samplerange_px + 1] = 1

    offsets_array = np.full([np.sum(ix_dla_range)], np.nan, dtype=np.float32)
    column_density = np.full([np.sum(ix_dla_range)], np.nan, dtype=np.float32)
    for i in range(int(samplerange_px + 1)):
        for ix_dla, j in zip(ix_dlas, range(len(ix_dlas))):
            offsets_array[ix_dla + i] = -i if np.isnan(offsets_array[ix_dla + i]) else offsets_array[ix_dla + i]
            offsets_array[ix_dla - i] = i if np.isnan(offsets_array[ix_dla - i]) else offsets_array[ix_dla - i]
            column_density[ix_dla + i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla + i]) else column_density[ix_dla + i]
            column_density[ix_dla - i] = coldensity_dlas[j] if np.isnan(column_density[ix_dla - i]) else column_density[ix_dla - i]

    sightline.classification = np.nan_to_num(classification)
    sightline.offsets = np.nan_to_num(offsets_array)
    sightline.column_density = np.nan_to_num(column_density)


def select_samples_50p_pos_neg(sightline):
    num_pos = np.sum(sightline.classification == 1, dtype=np.float64)
    num_neg = np.sum(sightline.classification == 0, dtype=np.float64)
    n_samples = int(min(num_pos, num_neg))
    if n_samples == 0:
        return []
    r = np.random.permutation(len(sightline.classification))
    pos_ixs = r[sightline.classification[r] == 1][0:n_samples]
    neg_ixs = r[sightline.classification[r] == 0][0:n_samples]
    return np.hstack((pos_ixs, neg_ixs))


def make_smoothdatasets_chunked(sightlines, output, chunk_size=CHUNK_SIZE):
    dataset = {}
    count = 0
    file_idx = 0
    for sightline in sightlines:
        if sightline == []:
            continue
        label_sightline(sightline, kernel['lowsnr'], REST_RANGE)
        data_split = split_sightline_into_samples(
            sightline, REST_RANGE=REST_RANGE, kernel=kernel['lowsnr'], v=best_v['all'], continuum=False
        )
        sample_masks = select_samples_50p_pos_neg(sightline)
        if len(sample_masks) > 0:
            flux = np.vstack([data_split[0][m] for m in sample_masks])
            labels_classifier = np.hstack([data_split[1][m] for m in sample_masks])
            labels_offset = np.hstack([data_split[2][m] for m in sample_masks])
            col_density = np.hstack([data_split[3][m] for m in sample_masks])
            flux_matrix = np.asarray(smooth_flux(flux))
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
            dataset = {}
            count = 0
            file_idx = 0
            for sightline in mid:
                label_sightline(sightline, kernel['highsnr'], REST_RANGE)
                data_split = split_sightline_into_samples(
                    sightline, REST_RANGE=REST_RANGE, kernel=kernel['highsnr'], v=best_v['all'], continuum=False
                )
                sample_masks = select_samples_50p_pos_neg(sightline)
                if len(sample_masks) > 0:
                    flux = np.vstack([data_split[0][m] for m in sample_masks])
                    labels_classifier = np.hstack([data_split[1][m] for m in sample_masks])
                    labels_offset = np.hstack([data_split[2][m] for m in sample_masks])
                    col_density = np.hstack([data_split[3][m] for m in sample_masks])
                    dataset[sightline.id] = {
                        'FLUX': flux,
                        'labels_classifier': labels_classifier,
                        'labels_offset': labels_offset,
                        'col_density': col_density
                    }

                count += 1
                if count >= CHUNK_SIZE:
                    outpath = "{}_{}.npy".format(os.path.join(OUT_ROOT, "mid", base), file_idx)
                    np.save(outpath, dataset)
                    dataset = {}
                    count = 0
                    file_idx += 1

            if dataset:
                outpath = "{}_{}.npy".format(os.path.join(OUT_ROOT, "mid", base), file_idx)
                np.save(outpath, dataset)
        if low:
            make_smoothdatasets_chunked(
                low,
                output=os.path.join(OUT_ROOT, "low", base),
                chunk_size=CHUNK_SIZE
            )


if __name__ == "__main__":
    main()
