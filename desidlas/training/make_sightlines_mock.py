import os
from desidlas.datasets.get_sightlines import get_sightlines


SPECTRA_ROOT = "/global/cfs/projectdirs/desi/mocks/lya_forest/london/qq_desi_y3/v5.9.5/mock-0/jura-124/spectra-16"
OUT_ROOT = "/pscratch/sd/t/tanting/retraining/sightlines"


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    for k in sorted(os.listdir(SPECTRA_ROOT)):
        k_dir = os.path.join(SPECTRA_ROOT, k)
        if not os.path.isdir(k_dir):
            continue
        for j in sorted(os.listdir(k_dir)):
            data_dir = os.path.join(k_dir, j)
            if not os.path.isdir(data_dir):
                continue
            if not os.listdir(data_dir):
                continue

            out_dir = os.path.join(OUT_ROOT, k, j)
            os.makedirs(out_dir, exist_ok=True)
            outpath = os.path.join(out_dir, "sightlines-{}.npy".format(j))
            if os.path.exists(outpath):
                continue

            spectra = os.path.join(data_dir, "spectra-16-{}.fits".format(j))
            zbest = os.path.join(data_dir, "zbest-16-{}.fits".format(j))
            truth = []

            get_sightlines(spectra, truth, zbest, outpath)
        print("{} done".format(k))


if __name__ == "__main__":
    main()
