#!/bin/bash -l
#SBATCH -C cpu
#SBATCH --account=desi
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=desi_mock_iron1
#SBATCH --output=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron1-%j.out
#SBATCH --error=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron1-%j.err
load_picca_plots
export dla_finder=/global/u1/t/tanting/DESI_analysis/desi-dlas
#export dla_finder=/global/cfs/cdirs/desi/users/jqzou/dla_finder
export desidlas=/global/cfs/cdirs/desi/users/jqzou/dla_finder
#srun -n 1 -c 64 python $dla_finder/desi_mock_iron1.py
for index in 0 10
do
    sed -ie ''68,70s/value/$index/g'' $dla_finder/desi_mock_iron1.py
    srun -n 1 -c 64 python $dla_finder/desi_mock_iron1.py
    sed -ie ''68,70s/$index/value/g'' $dla_finder/desi_mock_iron1.py
done