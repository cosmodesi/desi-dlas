import argparse
import os
import sys
from multiprocessing import Pool, cpu_count

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from desidlas.datasets.get_sightlines import get_sightlines


SPECTRA_ROOT = "/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/spectra-16"
OUT_ROOT = "/pscratch/sd/t/tanting/retraining/sightlines"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate sightlines from mock spectra.")
    parser.add_argument("--spectra-root", default=SPECTRA_ROOT)
    parser.add_argument("--out-root", default=OUT_ROOT)
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


def _process_one(args):
    spectra_root, out_root, k, j = args
    data_dir = os.path.join(spectra_root, k, j)
    if not os.path.isdir(data_dir):
        return
    if not os.listdir(data_dir):
        return

    out_dir = os.path.join(out_root, k, j)
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, "sightlines-{}.npy".format(j))
    if os.path.exists(outpath):
        return

    spectra = os.path.join(data_dir, "spectra-16-{}.fits".format(j))
    zbest = os.path.join(data_dir, "zbest-16-{}.fits".format(j))
    truth_path = os.path.join(data_dir, "truth-16-{}.fits".format(j))
    truth = truth_path if os.path.exists(truth_path) else []

    get_sightlines(spectra, truth, zbest, outpath)


def main():
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    k_dirs = _select_k_dirs(args.spectra_root, args.k_start, args.k_end)
    tasks = []
    for k in k_dirs:
        k_dir = os.path.join(args.spectra_root, k)
        for j in sorted(os.listdir(k_dir)):
            tasks.append((args.spectra_root, args.out_root, k, j))

    if args.workers <= 1:
        for task in tasks:
            _process_one(task)
    else:
        with Pool(args.workers) as pool:
            pool.map(_process_one, tasks)


if __name__ == "__main__":
    main()
