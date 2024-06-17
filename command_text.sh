export dla_finder=/global/u1/t/tanting/DESI_analysis/desi-dlas
for index in 10 20 30
do
    sed -ie '65,67s/value/$index/g' $dla_finder/desi_mock_iron2.py
    echo $index
    echo ''$index''
    sed -ie '65,67s/$index/value/g' $dla_finder/desi_mock_iron2.py
done
