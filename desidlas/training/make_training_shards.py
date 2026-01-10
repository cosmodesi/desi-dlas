import argparse
import glob
import os
import sys
from multiprocessing import Pool, cpu_count
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Build mid/low training shards from sightlines.")
    parser.add_argument("--sightline-root", default=os.path.dirname(SIGHTLINE_GLOB.rstrip("*/")))
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--k-start", type=int, default=None, help="Start k (inclusive).")
    parser.add_argument("--k-end", type=int, default=None, help="End k (exclusive).")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count()))
    return parser.parse_args()


def _select_k_dirs(root, k_start, k_end):
    k_dirs = [k for k in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, k))]
    if k_start is None and k_end is None:
        return k_dirs
    if all(k.isdigit() for k in k_dirs):
        k_ints = [int(k) for k in k_dirs]
        selected = [k for k, ki in zip(k_dirs, k_ints) if (k_start is None or ki >= k_start) and (k_end is None or ki < k_end)]
        return selected
    start = k_start or 0
    end = k_end or len(k_dirs)
    return k_dirs[start:end]


def _prune_empty_shards(prefix):
    for path in glob.glob(prefix + "_*.npy"):
        try:
            data = np.load(path, allow_pickle=True).item()
        except Exception:
            continue
        if not data:
            os.remove(path)


def _process_file(args):
    f, out_root, chunk_size = args
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
        out_prefix = os.path.join(out_root, "mid", base)
        make_datasets(mid, output=out_prefix, validate=False, chunk_size=chunk_size)
        _prune_empty_shards(out_prefix)
    if low:
        out_prefix = os.path.join(out_root, "low", base)
        make_smoothdatasets_chunked(low, output=out_prefix, chunk_size=chunk_size)
        _prune_empty_shards(out_prefix)


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    os.makedirs(os.path.join(args.out_root, "mid"), exist_ok=True)
    os.makedirs(os.path.join(args.out_root, "low"), exist_ok=True)

    sightline_root = args.sightline_root
    k_dirs = _select_k_dirs(sightline_root, args.k_start, args.k_end)
    files = []
    for k in k_dirs:
        k_dir = os.path.join(sightline_root, k)
        files.extend(glob.glob(os.path.join(k_dir, "*", "sightlines-*.npy")))

    tasks = [(f, args.out_root, args.chunk_size) for f in sorted(files)]
    if args.workers <= 1:
        for task in tasks:
            _process_file(task)
    else:
        with Pool(args.workers) as pool:
            pool.map(_process_file, tasks)


if __name__ == "__main__":
    main()
