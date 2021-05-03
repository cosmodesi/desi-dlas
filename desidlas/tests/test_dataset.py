#load basic module
import numpy as np
import scipy.signal as signal
from desidlas.datasets.datasetting import split_sightline_into_samples,select_samples_50p_pos_neg,pad_sightline
from desidlas.datasets.preprocess import label_sightline
from desidlas.dla_cnn.spectra_utils import get_lam_data

#load the data for the test
#this file is the output of get_sightlines.py, modules are tested in test_preprocess
sightline_path='desidlas/tests/datafile/sightlines-16-1375.npy'

sightlines=np.load(sightline_path,allow_pickle = True,encoding='latin1')

##load modules in get_datasets.py
from desidlas.datasets.get_dataset import make_datasets
from desidlas.datasets.get_dataset import make_smoothdatasets


def test_datasets:
  dataset=make_datasets(sightlines,validate=True)

  
#for the spectra with S/N<3, we need to do the smoothing for the flux
from desidlas.datasets.get_dataset import make_smoothdatasets

def test_smoothdatasets:
  
  smoothdataset=make_smoothdatasets(sightlines,validate=True)
