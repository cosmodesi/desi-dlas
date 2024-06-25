#!/bin/bash -l
#SBATCH -C cpu
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --time=06:30:00
#SBATCH --job-name=desi_mock_iron5
#SBATCH --output=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron5-%j.out
#SBATCH --error=/global/u1/t/tanting/DESI_analysis/desi-dlas/log/log-desi_mock_iron5-%j.err

load_picca_plots
export dla_finder=/global/u1/t/tanting/DESI_analysis/desi-dlas

for ((index = 9850; index <= 10000; index += 10))
#460 470 480 490 500 510 520 530 540 550 560 570 580 590 600
do
    echo $index
    sed -ie ''69,70s/value/$index/g'' $dla_finder/desi_mock_iron5.py
    srun -n 1 -c 256 python $dla_finder/desi_mock_iron5.py
    sed -ie ''69,70s/$index/value/g'' $dla_finder/desi_mock_iron5.py
done

for ((index = 14000; index <= 15000; index += 10))
#460 470 480 490 500 510 520 530 540 550 560 570 580 590 600
do
    echo $index
    sed -ie ''69,70s/value/$index/g'' $dla_finder/desi_mock_iron5.py
    srun -n 1 -c 256 python $dla_finder/desi_mock_iron5.py
    sed -ie ''69,70s/$index/value/g'' $dla_finder/desi_mock_iron5.py
done