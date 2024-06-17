#!/bin/bash -l
#SBATCH -C cpu
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --job-name=desi_mock_iron2
#SBATCH --output=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron2-%j.out
#SBATCH --error=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron2-%j.err

load_picca_plots
export dla_finder=/global/u1/t/tanting/DESI_analysis/desi-dlas
for index in 70 80 90
#160 170 180 190 200 210 220 230 240 250 260 270 280 290 300
do
    sed -ie ''68,70s/value/$index/g'' $dla_finder/desi_mock_iron2.py
    srun -n 1 -c 64 python $dla_finder/desi_mock_iron2.py
    sed -ie ''68,70s/$index/value/g'' $dla_finder/desi_mock_iron2.py
done
