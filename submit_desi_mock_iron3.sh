#!/bin/bash -l
#SBATCH -C cpu
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --job-name=desi_mock_iron3
#SBATCH --output=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron3-%j.out
#SBATCH --error=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron3-%j.err
load_picca_plots
export dla_finder=/global/u1/t/tanting/DESI_analysis/desi-dlas
for index in 100 110 120
#310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 610
do
    sed -ie ''68,70s/value/$index/g'' $dla_finder/desi_mock_iron3.py
    srun -n 1 -c 64 python $dla_finder/desi_mock_iron3.py
    sed -ie ''68,70s/$index/value/g'' $dla_finder/desi_mock_iron3.py
done