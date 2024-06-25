#!/bin/bash -l
#SBATCH -C cpu
#SBATCH --account=desi
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=desi_mock_iron6
#SBATCH --output=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron6-%j.out
#SBATCH --error=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron6-%j.err

load_picca_plots
export dla_finder=/global/u1/t/tanting/DESI_analysis/desi-dlas

for ((index = 16400; index <= 16500; index += 10))
#460 470 480 490 500 510 520 530 540 550 560 570 580 590 600
do
    echo $index
    sed -ie ''69,70s/value/$index/g'' $dla_finder/desi_mock_iron6.py
    srun -n 1 -c 256 python $dla_finder/desi_mock_iron6.py
    sed -ie ''69,70s/$index/value/g'' $dla_finder/desi_mock_iron6.py
done