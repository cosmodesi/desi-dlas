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

#test the flux,lam and the labels in the dataset
#labels:FLUX,lam,labels_classifier,labels_offset,col_density,wavelength_dlas,coldensity_dlas
def test_labels(spec):
  #check the flux and lam
  assert len(spec['FLUX'])>0 ,'no flux data'
  assert len(spec['lam'])>0 ,'no lam data'
  
  if len(spec['wavelength_dlas'])>0:
    #if there is a dla in this spectra, some value of these labels should be above 0
    assert np.max(spec['labels_classifier'])>0, 'classifier labels are not correct'
    assert np.max(spec['labels_offset'])>0, 'offset labels are not correct'
    assert np.max(spec['coldensity'])>0, 'col_density labels are not correct'
    

def test_datasets:
  dataset=make_datasets(sightlines,validate=True)
  for key in dataset.keys():
    spec=dataset[key]
    test_labels(spec)

  
#for the spectra with S/N<3, we need to do the smoothing for the flux
from desidlas.datasets.get_dataset import make_smoothdatasets

def test_smoothdatasets:
  
  smoothdataset=make_smoothdatasets(sightlines,validate=True)
  
  for key in dataset.keys():
    spec=dataset[key]
    test_labels(spec)
