#from dla_cnn.data_model.DataMarker import Marker
#no Marker in the mock spectra

from pkg_resources import resource_filename
import os

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

datafile_path = os.path.join(resource_filename('desidlas', 'tests'), 'datafile')

#load spectra file for testing
spectra= os.path.join(datafile_path, 'spectra', 'spectra-16-1375.fits')
truth=os.path.join(datafile_path, 'spectra', 'truth-16-1375.fits')
zbest=os.path.join(datafile_path, 'spectra', 'zbest-16-1375.fits')

#read spectra data using DesiMock
specs = DesiMock()
specs.read_fits_file(spectra,truth,zbest)

#get sightline id in the spectra file
keys = list(specs.data.keys())
jj=keys[0] #choose 1 sightline to do the test

#m
sightline = specs.get_sightline(jj,camera = 'all', rebin=False, normalize=True)

def test_estimate_s2n(sightline):
    # estimate the signal-to-noise ratio
    sightline.s2n=estimate_s2n(sightline)
    

def test_label_sightline():
    # Generate sightline
    
    # Call label_sightline
    pass

def test_rebin():
    pass
