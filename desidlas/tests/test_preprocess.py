#from dla_cnn.data_model.DataMarker import Marker
#no Marker in the mock spectra

from desidlas.dla_cnn import defs
#load basic parameter value 
REST_RANGE = defs.REST_RANGE
kernel = defs.kernel
best_v=defs.best_v

#load modules in preprocess.py
from desidlas.datasets.preprocess import label_sightline
from desidlas.datasets.preprocess import rebin
from desidlas.datasets.preprocess import normalize
from desidlas.datasets.preprocess import estimate_s2n
from desidlas.datasets.preprocess import generate_summary_table
from desidlas.datasets.preprocess import write_summary_table

#load DesiMock
from desidlas.datasets.DesiMock import DesiMock


#load spectra file for testing
spectra='desidlas/tests/datafile/spectra/spectra-16-1375.fits'
truth='desidlas/tests/datafile/spectra/truth-16-1375.fits'
zbest='desidlas/tests/datafile/spectra/zbest-16-1375.fits'

#
specs = DesiMock()
specs.read_fits_file(spectra,truth,zbest)

def test_label_sightline():
    # Generate sightline
    
    # Call label_sightline
    pass

def test_rebin():
    pass
