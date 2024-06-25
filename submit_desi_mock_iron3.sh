#!/bin/bash -l
#SBATCH -C cpu
#SBATCH --account=desi
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=desi_mock_iron3
#SBATCH --output=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron3-%j.out
#SBATCH --error=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron3-%j.err
load_picca_plots
export dla_finder=/global/u1/t/tanting/DESI_analysis/desi-dlas

for ((index = 12950; index <= 13000; index += 10))
do
    echo $index
    sed -ie ''69,70s/value/$index/g'' $dla_finder/desi_mock_iron3.py
    srun -n 1 -c 256 python $dla_finder/desi_mock_iron3.py
    sed -ie ''69,70s/$index/value/g'' $dla_finder/desi_mock_iron3.py
done