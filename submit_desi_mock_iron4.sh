#!/bin/bash -l
#SBATCH -C cpu
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --job-name=desi_mock_iron4
#SBATCH --output=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron4-%j.out
#SBATCH --error=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron4-%j.err

load_picca_plots
export dla_finder=/global/u1/t/tanting/DESI_analysis/desi-dlas
for index in 270
#460 470 480 490 500 510 520 530 540 550 560 570 580 590 600
do
    sed -ie ''68,70s/value/$index/g'' $dla_finder/desi_mock_iron4.py
    srun -n 1 -c 64 python $dla_finder/desi_mock_iron4.py
    sed -ie ''68,70s/$index/value/g'' $dla_finder/desi_mock_iron4.py
done
